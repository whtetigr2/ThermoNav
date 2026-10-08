"""Particle-obscurant sensor model for Phase 4 (Paul's "low vis mode").

A particle-laden atmosphere (rain, fog, snow, smoke, ash) does TWO things
to lidar, modeled here at the per-ray return level:

1. ATTENUATION: the true surface return may not make it back.
   P(return survives to range r) = exp(-sigma * r), Beer-Lambert.
   sigma [1/m] is the extinction coefficient. A ray whose return is
   attenuated produces NO DETECTION — and a no-detection ray carries NO
   INFORMATION (it is skipped in integration, not treated as free space;
   see note below).

2. BACKSCATTER: particles near the sensor return the pulse themselves ->
   phantom obstacles, concentrated near-field. Modeled as an independent
   per-ray Bernoulli event, p_bs = 1 - exp(-sigma * L_bs); on trigger the
   false return range r_f ~ Exponential(mean=lambda_bs), clipped to
   [0.5, 10] m. If backscatter triggers it wins (the near particle is seen
   first) — documented simplification.

WHY no-return rays are skipped (not cleared as free space):
  The production sensor model (sensor.py, SENSOR_FIX.md) treats misses as
  highly reliable free-space evidence (P=99.86%) — valid in clear air,
  where a miss means "ray traveled 40 m and hit nothing." Under obscurant
  a miss is AMBIGUOUS (absorption vs. out-of-range), and clearing free
  space on attenuated rays would ERASE real walls behind fog — the exact
  wrong behavior. So: no return -> no update. This is the honest model;
  it makes fog *remove evidence* (weakening |h|) rather than *assert
  emptiness*.

INTEGRITY: the obscurant touches ONLY the sensor (scan returns). GT
occupancy is used only inside the raycaster (as in flight) and never
reaches the planner/sampler. The 5-tuple's 4th/5th elements are the
SENSOR's hit/no-return decision, not GT.

Parameters (all ASSUMED test values unless noted):
  sigma: extinction coefficient [1/m]. Density ladder mapped from
    meteorological visibility V via Koschmieder V = 3.912/sigma:
      clear    V=10km  sigma=0.0004
      light    V=1km   sigma=0.004
      moderate V=200m  sigma=0.02
      heavy    V=50m   sigma=0.08
      whiteout V=10m   sigma=0.4
    At 40 m max range these give P_survive = 0.98 / 0.85 / 0.45 / 0.045 / ~0.
  L_bs = 3.0 m [assumed]: effective near-field path for backscatter.
    p_bs at the ladder: 0.001 / 0.012 / 0.058 / 0.21 / 0.70.
  lambda_bs = 2.0 m [assumed]: mean of the false-return range distribution.

ARCHETYPES (documented refinement, not five separate models):
  fog  : fine uniform particles — the ladder above (attenuation-dominated).
  rain : sparse large drops — sigma_rain = sigma_fog/4, L_bs=1.0 (sparser
         events), lambda_bs=4.0 (wider range when a drop IS hit).
  Snow/smoke/ash differ in particle size, absorption-vs-scattering ratio,
  and wavelength dependence — NOT separately modeled; fog-like (fine,
  uniform) and rain-like (sparse, large) bracket the behavior space for
  this LEGO version.
"""
import numpy as np

import sensor as _sensor  # cast_ray only; integrate/scan NOT reused

# ---------------------------------------------------------------- config
# Module-level obscurant config; set via configure(). Keeps scan_pose a
# drop-in replacement for sensor.scan_pose.
_CFG = {
    "sigma": 0.0,          # extinction coefficient [1/m]; 0 = clear air
    "archetype": "fog",    # 'fog' | 'rain'
    "L_bs": 3.0,           # near-field path for backscatter probability [m]
    "lambda_bs": 2.0,      # mean false-return range [m]
}

# (sigma, L_bs, lambda_bs) per archetype x density label
ARCHETYPES = {
    "fog": {
        "clear":    (0.0004, 3.0, 2.0),
        "light":    (0.004,  3.0, 2.0),
        "moderate": (0.02,   3.0, 2.0),
        "heavy":    (0.08,   3.0, 2.0),
        "whiteout": (0.4,    3.0, 2.0),
    },
    "rain": {
        # sigma/4 (large drops, less extinction); sparser but wider backscatter
        "clear":    (0.0001, 1.0, 4.0),
        "light":    (0.001,  1.0, 4.0),
        "moderate": (0.005,  1.0, 4.0),
        "heavy":    (0.02,   1.0, 4.0),
        "whiteout": (0.1,    1.0, 4.0),
    },
}
DENSITIES = ["clear", "light", "moderate", "heavy", "whiteout"]


def configure(archetype="fog", density="clear", sigma=None):
    """Set the obscurant. Either (archetype, density-label) or explicit sigma."""
    if sigma is None:
        sigma, L_bs, lambda_bs = ARCHETYPES[archetype][density]
    else:
        L_bs, lambda_bs = ARCHETYPES[archetype]["moderate"][1:3]
        density = f"custom-sigma-{sigma}"
    _CFG.update({"sigma": float(sigma), "archetype": archetype,
                 "L_bs": float(L_bs), "lambda_bs": float(lambda_bs),
                 "density": density})
    return dict(_CFG)


def get_config():
    return dict(_CFG)


def scan_pose(occ, pose, n_az=120, elevations=(-0.26, 0.0, 0.26),
              max_range=40.0, range_noise=0.15, seed=0):
    """Obscured scan. Returns list of
    (origin, direction, meas_r, sensor_hit, has_return).

    sensor_hit: the SENSOR got a return classified as a surface hit.
    has_return: the SENSOR got any usable return (hit or confident miss).
      has_return=False -> attenuated to nothing; integrate SKIPS the ray.
    """
    rng = np.random.default_rng(seed)
    sigma = _CFG["sigma"]
    L_bs, lambda_bs = _CFG["L_bs"], _CFG["lambda_bs"]
    p_bs = 1.0 - np.exp(-sigma * L_bs) if sigma > 0 else 0.0
    out = []
    for k, el in enumerate(elevations):
        for i in range(n_az):
            az = 2 * np.pi * i / n_az
            d = np.array([np.cos(el) * np.cos(az),
                          np.cos(el) * np.sin(az),
                          np.sin(el)])
            r_true = _sensor.cast_ray(occ, pose, d, max_range=max_range)
            # --- backscatter first: near particle wins if it triggers ---
            if rng.random() < p_bs:
                r_f = float(rng.exponential(lambda_bs))
                r_f = min(max(r_f, 0.5), 10.0)
                meas_r = max(0.5, r_f + rng.normal(0, range_noise))
                out.append((pose, d, meas_r, True, True))
                continue
            # --- attenuation of the true return ---
            p_survive = np.exp(-sigma * r_true) if sigma > 0 else 1.0
            if rng.random() > p_survive:
                # no detection: NO INFORMATION (skipped in integration)
                out.append((pose, d, max_range, False, False))
                continue
            # --- clean return, as production ---
            meas_r = r_true + rng.normal(0, range_noise)
            meas_r = max(0.5, meas_r)
            hit = bool(r_true < max_range - 1e-6)
            out.append((pose, d, meas_r, hit, True))
    return out


def integrate_scan(logodds, occ_shape, scans, l_occ=0.60, l_free=-0.73,
                   l_min=-4.0, l_max=4.0, step=0.5, hit_radius=0.75,
                   max_range=40.0, obs_count=None):
    """Inverse sensor model for obscured scans. 5-tuples; no-return rays
    (has_return=False) are SKIPPED — they carry no information. Returns
    (logodds, n_hits, n_skipped)."""
    nx, ny, nz = occ_shape
    n_hits, n_skipped = 0, 0
    for tup in scans:
        if len(tup) == 5:
            origin, direction, meas_r, sensor_hit, has_return = tup
        else:  # 4-tuple (production format): every ray has a return
            origin, direction, meas_r, true_r = tup
            sensor_hit, has_return = bool(true_r < max_range - 1e-6), True
        if not has_return:
            n_skipped += 1
            continue
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        t = step
        while t < meas_r - hit_radius:
            p = o + d * t
            ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                logodds[ix, iy, iz] = np.clip(logodds[ix, iy, iz] + l_free,
                                             l_min, l_max)
                if obs_count is not None:
                    obs_count[ix, iy, iz] += 1
            else:
                break
            t += step
        if sensor_hit:
            n_hits += 1
            p = o + d * meas_r
            ix, iy, iz = int(p[0]), int(p[1]), int(p[2])
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                logodds[ix, iy, iz] = np.clip(logodds[ix, iy, iz] + l_occ,
                                             l_min, l_max)
                if obs_count is not None:
                    obs_count[ix, iy, iz] += 1
    return logodds, n_hits, n_skipped
