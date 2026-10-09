"""Chromatic Gibbs sampler for the occupancy MRF.

Model: x_v in {0,1} (1=occupied).
Energy: E = -sum_v h_v x_v - J * sum_{<u,v>} [x_u == x_v]
Agreement as pairwise: [x_u==x_v] = 2 x_u x_v - x_u - x_v + 1.
Conditional: P(x_v=1 | rest) = sigmoid(h_v + J * sum_{u in N(v)} (2 x_u - 1)).

3D grid is bipartite -> 2 colors, vectorized updates.
"""
import numpy as np

# Optional TSU activity instrumentation (dashboard "in use" light).
# Zero-cost when disabled: tsu_event() is a single boolean check.
# The try/except keeps gibbs.py working even if instrumentation.py
# is missing (e.g. stripped-down deployments).
try:
    from instrumentation import tsu_event, should_emit_progress
except ImportError:
    try:
        from dashboard.tsu_activity.instrumentation import tsu_event, should_emit_progress
    except ImportError:
        def tsu_event(*a, **k):
            return None

        def should_emit_progress(s):
            return False


def neighbor_agreement_sum(x, color_mask):
    """For voxels of given color, compute S_v = sum_{valid u in N(v)} (2 x_u - 1).

    Neighbors of a color-c voxel are all color-(1-c): read x directly (no update conflict).
    Out-of-bounds neighbors contribute 0 (not counted), not -1.
    """
    nx, ny, nz = x.shape
    S = np.zeros_like(x, dtype=np.float64)
    xf = x.astype(np.float64)
    shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for (dx, dy, dz) in shifts:
        xs = np.roll(xf, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        valid = np.ones_like(xf)
        # roll with shift=-dx puts x[i+dx] at position i; invalid index is
        # i=n-1 for dx=+1, i=0 for dx=-1.
        if dx == 1:
            xs[-1, :, :] = 0; valid[-1, :, :] = 0
        elif dx == -1:
            xs[0, :, :] = 0; valid[0, :, :] = 0
        if dy == 1:
            xs[:, -1, :] = 0; valid[:, -1, :] = 0
        elif dy == -1:
            xs[:, 0, :] = 0; valid[:, 0, :] = 0
        if dz == 1:
            xs[:, :, -1] = 0; valid[:, :, -1] = 0
        elif dz == -1:
            xs[:, :, 0] = 0; valid[:, :, 0] = 0
        S += np.where(color_mask, valid * (2 * xs - 1), 0.0)
    return S


def make_color_masks(shape):
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    black = ((ii + jj + kk) % 2 == 0)
    return black, ~black


def gibbs_sweep(x, h, J, masks, rng):
    """One full chromatic sweep (both colors). x, h: arrays; J: scalar."""
    black, white = masks
    for mask in (black, white):
        S = neighbor_agreement_sum(x, mask)
        logit = np.where(mask, h + J * S, 0.0)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        draw = rng.random(x.shape)
        x = np.where(mask, (draw < prob).astype(np.float64), x)
    return x


def sample_posterior(h, J, n_sweeps=300, burn_in=100, seed=0, x_init=None):
    """Run chromatic Gibbs; return (marginals, n_sweeps timing info).

    marginals[v] = mean of x_v over post-burn-in sweeps.

    Instrumentation: emits tsu_event("sampling_start" / "sweep_progress" /
    "sampling_end") — no-ops when instrumentation is disabled.
    """
    import time
    rng = np.random.default_rng(seed)
    masks = make_color_masks(h.shape)
    x = np.zeros_like(h) if x_init is None else x_init.copy()
    acc = np.zeros_like(h)
    n_acc = 0
    n_voxels = int(np.prod(h.shape))
    tsu_event("sampling_start", n_sweeps=n_sweeps, burn_in=burn_in,
              n_voxels=n_voxels, J=float(J))
    t0 = time.time()
    for s in range(n_sweeps):
        x = gibbs_sweep(x, h, J, masks, rng)
        if should_emit_progress(s):
            tsu_event("sweep_progress", sweep=s, total=n_sweeps,
                      elapsed_ms=round((time.time() - t0) * 1000.0, 1))
        if s >= burn_in:
            acc += x
            n_acc += 1
    dt = time.time() - t0
    marg = acc / max(n_acc, 1)
    tsu_event("sampling_end", sweeps=n_sweeps, dt_ms=round(dt * 1000.0, 1),
              mean_p=round(float(marg.mean()), 4))
    return marg, dt, x
