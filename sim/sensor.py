"""Synthetic lidar + inverse sensor model -> log-odds grid.

Conventions:
- occ: 1=occupied, 0=free (ground truth)
- logodds L: positive => occupied. Update: L += l_occ on hit, L += l_free on miss.
- Values: l_occ=+0.60, l_free=-0.73, clamp [-4,+4].
  Derived from empirical calibration on our synthetic lidar (see SENSOR_FIX.md):
  - OctoMap's +0.85/-0.4 was tuned for real laser range finders where misses
    are ambiguous (max-range, absorption, specular). Not our case.
  - Our synthetic lidar uses TRUE range for hit/miss decisions; only hit
    LOCATION is noisy (sigma=0.15). Misses are highly reliable (P=99.86%).
  - Measured likelihood ratio gives asymmetry 0.82:1 (free update stronger),
    vs OctoMap's 2.12:1 (hit stronger). We use the measured ratio.
  - Absolute magnitudes scaled down from theoretical (+5.34/-6.55) because
    rays from the same scan are correlated, not independent observations.
  - Result: 2 clearing steps per false hit (was 5), 40% fewer phantoms,
    zero increase in false negatives [measured, SENSOR_FIX.md].
"""
import numpy as np

# Optional TSU activity instrumentation (dashboard "in use" light).
# Zero-cost when disabled. Safe fallback if instrumentation.py is missing.
try:
    from instrumentation import tsu_event, is_enabled as _instr_enabled
except ImportError:
    try:
        from dashboard.tsu_activity.instrumentation import tsu_event, is_enabled as _instr_enabled
    except ImportError:
        def tsu_event(*a, **k):
            return None

        def _instr_enabled():
            return False


def cast_ray(occ, origin, direction, max_range=60.0, step=0.5):
    """March along ray; return distance to first occupied voxel (or max_range)."""
    nx, ny, nz = occ.shape
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    t = step
    while t <= max_range:
        p = o + d * t
        ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            return max_range  # left the map: no hit
        if occ[ix, iy, iz]:
            return t
        t += step
    return max_range


def scan_pose(occ, pose, n_az=120, elevations=(-0.26, 0.0, 0.26),
              max_range=40.0, range_noise=0.15, seed=0):
    """Full 3D scan from one pose. Returns list of (origin, direction, measured_range)."""
    rng = np.random.default_rng(seed)
    out = []
    for k, el in enumerate(elevations):
        for i in range(n_az):
            az = 2 * np.pi * i / n_az
            d = np.array([np.cos(el) * np.cos(az),
                          np.cos(el) * np.sin(az),
                          np.sin(el)])
            true_r = cast_ray(occ, pose, d, max_range=max_range)
            meas_r = true_r + rng.normal(0, range_noise)
            meas_r = max(0.5, meas_r)
            out.append((pose, d, meas_r, true_r))
    return out


def integrate_scan(logodds, occ_shape, scans, l_occ=0.60, l_free=-0.73,
                   l_min=-4.0, l_max=4.0, step=0.5, hit_radius=0.75,
                   max_range=40.0, obs_count=None):
    """Inverse sensor model: accumulate log-odds. Returns (logodds, hit_count).

    Args:
        obs_count: optional int array (same shape as logodds). If provided,
            incremented once per voxel per ray update (free or hit). Used for
            the weak-evidence fix: voxels with few observations are treated
            as unknown rather than blocked. When None (default), no counting
            is done and behavior is identical to the original.
    """
    nx, ny, nz = occ_shape
    n_rays = len(scans)
    # Only pay for voxel-touch tracking when instrumentation is on.
    _track = _instr_enabled()
    tsu_event("sensor_start", phase="host", n_rays=n_rays)
    import time as _time
    _t0 = _time.time()
    n_hits = 0
    voxels_touched = set() if _track else None
    for (origin, direction, meas_r, true_r) in scans:
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        t = step
        # free space along the ray
        while t < meas_r - hit_radius:
            p = o + d * t
            ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                logodds[ix, iy, iz] = np.clip(logodds[ix, iy, iz] + l_free, l_min, l_max)
                if obs_count is not None:
                    obs_count[ix, iy, iz] += 1
                if _track:
                    voxels_touched.add((ix, iy, iz))
            else:
                break
            t += step
        # hit voxel (only if the ray actually hit something, not max_range)
        if true_r < max_range - 1e-6:
            n_hits += 1
            p = o + d * meas_r
            ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                logodds[ix, iy, iz] = np.clip(logodds[ix, iy, iz] + l_occ, l_min, l_max)
                if obs_count is not None:
                    obs_count[ix, iy, iz] += 1
                if _track:
                    voxels_touched.add((ix, iy, iz))
    tsu_event("sensor_end", phase="host", n_rays=n_rays, n_hits=n_hits,
              voxels_updated=len(voxels_touched) if _track else -1,
              dt_ms=round((_time.time() - _t0) * 1000.0, 1))
    return logodds


def quantize_8bit(logodds, lo=-4.0, hi=4.0):
    """Quantize log-odds to 8-bit in [lo, hi]. Returns float array of quantized values."""
    q = np.round((np.clip(logodds, lo, hi) - lo) / (hi - lo) * 255.0) / 255.0 * (hi - lo) + lo
    return q.astype(np.float64)
