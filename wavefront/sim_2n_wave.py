#!/usr/bin/env python3
"""LEGO 2n-wave — geodesic wavefront planner (Paul's spec 2026-10-17 10:31).

Thermodynamic wavefront:
  1. Program h per mission: obstacles +6 (pinned +1), free -6, goal clamped -1.
  2. Init all free spins +1. Run Gibbs sampling.
  3. The -1 region GROWS from the goal via ferromagnetic J. Flip order = wavefront.
  4. Per-cell flip time -> geodesic field. Drone follows the gradient.

The Gibbs DYNAMICS are the computation (not equilibrium magnetization, not BFS).
Ablation: J=0 -> uniform flip field -> no routing (dynamics load-bearing).

Real THRML IsingEBM + sample_states. Paul's PC, CPU-only, nice 19.
"""
import os, sys, json, hashlib, time
import numpy as np
import jax
import jax.numpy as jnp
from collections import deque

# CURATION PATCH (2026-10-23): the research original imported the maze
# generator from a lab-internal path. Here it resolves to the vendored
# copy in this directory (see maze_gen.py provenance header). Nothing
# else in the sampling/field/routing code is changed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze_gen import build_ceiling_maze  # noqa: E402

from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

OUTDIR = os.environ.get(
    "LEGO2N_OUTDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
os.makedirs(OUTDIR, exist_ok=True)

# --- R3 SAVE PATCH (2026-10-22, RUN_ORDER R3 / PLAN_RERUN.md) ---------
# The 2026-10-17 original saved NO field artifacts (capstone U1-F6).
# Everything below this marker is additive: a save helper + one call
# per bar case + capture of the per-chain flip cube in run_wavefront.
# Sampling, field construction, smoothing, routing and outputs are
# bit-identical to originals/sim_2n_wave.py (see SAVE_PATCH.diff).
FIELD_DIR = os.environ.get(
    "LEGO2N_FIELD_DIR", os.path.join(OUTDIR, "rerun_fields"))
os.makedirs(FIELD_DIR, exist_ok=True)


def save_flip_field(name, built, tmed, tmean, obs_set, start=None):
    """Save the wavefront field artifacts for one case (R3)."""
    import platform
    try:
        import importlib.metadata as _md
        thrml_v = _md.version("thrml")
    except Exception:
        thrml_v = "unknown"
    ws = built["world_shape"]
    dim = built["dim"]
    obs_mask = np.zeros(ws[::-1], dtype=bool)
    if dim == 2:
        for (x, y) in obs_set:
            obs_mask[y, x] = True
    else:
        for (x, y, z) in obs_set:
            obs_mask[z, y, x] = True
    code_sha = hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()
    np.savez_compressed(
        os.path.join(FIELD_DIR, name + ".npz"),
        flip_cube=built["_flipcube"],
        tmean_world=to_world_field(tmean, built, obs_set),
        tmed_world=to_world_field(tmed, built, obs_set),
        obs_mask=obs_mask,
        goal=np.array(built["gh"]) - 1,  # halo coords -> world coords
        start=np.array(start) if start is not None else np.array([]),
        world_shape=np.array(ws),
        shape_h=np.array(built["shape_h"]),
        J=np.float32(built["params"]["J"]),
        H_FREE=np.float32(built["params"]["H_FREE"]),
        H_OBS=np.float32(built["params"]["H_OBS"]),
        beta=BETA, n_sweeps=built["n_sweeps"],
        n_chains=built["n_chains"],
        prog_hash=built["prog_hash"], code_sha256=code_sha,
        thrml_version=thrml_v, jax_version=jax.__version__,
        numpy_version=np.__version__, python_version=platform.python_version(),
    )
    print("  [save] %s.npz" % name, flush=True)
# --- END R3 SAVE PATCH (helper) --------------------------------------

# 2D / 3D wavefront params (beta=3 sharpens the sigmoid for 3D SNR)
# 2D: J=2.5,h0=6 -> trans p=0.997, spont p~0
# 3D: J=1.3,h0=5.8 -> trans p=0.973, spont p=6e-6 (derived: 4J<h0<6J, Z1-legal)
P2 = dict(J=np.float32(2.5), H_FREE=np.float32(-6.0), H_OBS=np.float32(6.0))
P3 = dict(J=np.float32(1.3), H_FREE=np.float32(-5.8), H_OBS=np.float32(6.0))
BETA = np.float32(3.0)


def build_wave(world_shape, obs_set, goal, params, n_sweeps, n_chains):
    """Flash the wavefront model for one mission. Halo = 1-cell obstacle shell.
    world_shape: (W,H) or (W,H,D). obs_set: set of (x,y[ ,z]). goal: (x,y[,z]).
    Returns built dict."""
    dim = len(world_shape)
    W = world_shape[0]
    shape_h = tuple(s + 2 for s in world_shape)  # halo
    J, H_FREE, H_OBS = params["J"], params["H_FREE"], params["H_OBS"]

    # index helpers
    if dim == 2:
        cells = [(x, y) for y in range(shape_h[1]) for x in range(shape_h[0])]
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        cells = [(x, y, z) for z in range(shape_h[2])
                 for y in range(shape_h[1]) for x in range(shape_h[0])]
        nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def is_halo(c):
        return any(v == 0 or v == s - 1 for v, s in zip(c, shape_h))

    def to_world(c):
        return tuple(v - 1 for v in c)

    gh = tuple(v + 1 for v in goal)  # goal in halo coords
    lat = {c: SpinNode() for c in cells}
    nodes = [lat[c] for c in cells]
    idx = {n: i for i, n in enumerate(nodes)}
    biases = np.zeros(len(nodes), dtype=np.float32)
    for c in cells:
        if is_halo(c):
            biases[idx[lat[c]]] = H_OBS
        else:
            w = to_world(c)
            biases[idx[lat[c]]] = H_OBS if w in obs_set else H_FREE

    edges, weights = [], []
    seen = set()
    for c in cells:
        for dn in nbrs:
            nb = tuple(a + b for a, b in zip(c, dn))
            if nb in lat and (nb, c) not in seen:
                edges.append((lat[c], lat[nb]))
                weights.append(J)
                seen.add((c, nb))
    weights = np.array(weights, dtype=np.float32)

    gnode = lat[gh]
    parity = lambda c: sum(c) % 2  # noqa: E731
    blkA = Block([lat[c] for c in cells if parity(c) == 0 and c != gh])
    blkB = Block([lat[c] for c in cells if parity(c) == 1 and c != gh])
    free_blocks = [blkA, blkB]
    clamp_block = Block([gnode])

    node_block = {}
    for bi, b in enumerate(free_blocks):
        for n in b.nodes:
            node_block[n] = bi
    intra = sum(1 for (a, b), w in zip(edges, weights)
                if w != 0 and a in node_block and b in node_block
                and node_block[a] == node_block[b])
    deg = {}
    for (a, b), w in zip(edges, weights):
        if w != 0:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1

    prog_hash = hashlib.sha256(biases.tobytes() + weights.tobytes()).hexdigest()
    # reverse index: node -> halo coords
    xy_of = {lat[c]: c for c in cells}
    return dict(nodes=nodes, edges=edges, weights=weights, biases=biases,
                free_blocks=free_blocks, clamp_block=clamp_block,
                xy_of=xy_of, lat=lat, gh=gh, shape_h=shape_h, dim=dim,
                world_shape=world_shape, intra=intra,
                max_deg=max(deg.values()) if deg else 0,
                max_abs_h=float(np.abs(biases).max()),
                max_abs_J=float(np.abs(weights).max()),
                n_free=sum(len(b.nodes) for b in free_blocks),
                n_sweeps=n_sweeps, n_chains=n_chains,
                prog_hash=prog_hash, params={k: float(v)
                                             for k, v in params.items()})


def run_wavefront(built):
    """Sample; return per-cell first-flip time (halo coords), median over chains."""
    n_sweeps, n_chains = built["n_sweeps"], built["n_chains"]
    model = IsingEBM(built["nodes"], built["edges"],
                     jnp.array(built["biases"]), jnp.array(built["weights"]),
                     jnp.array(BETA))
    program = IsingSamplingProgram(model, built["free_blocks"],
                                   [built["clamp_block"]])
    schedule = SamplingSchedule(n_warmup=0, n_samples=n_sweeps,
                                steps_per_sample=1)

    def infer_single(k):
        init = [jnp.ones((len(b.nodes),), dtype=jnp.bool_)
                for b in built["free_blocks"]]
        states = sample_states(k, program, schedule, init,
                               [jnp.array([False])], built["free_blocks"])
        return states

    run = jax.jit(jax.vmap(infer_single))
    t0 = time.time()
    states = run(jax.random.split(jax.random.key(0), n_chains))
    wall_sample = time.time() - t0

    dim = built["dim"]
    sh = built["shape_h"]
    # canonical order: 2D -> (chains, Hy, Wx); 3D -> (chains, Dz, Hy, Wx)
    fshape = (n_chains, sh[1], sh[0]) if dim == 2 else \
        (n_chains, sh[2], sh[1], sh[0])
    flip = np.full(fshape, n_sweeps + 1)
    for bi, blk in enumerate(built["free_blocks"]):
        arr = np.asarray(states[bi])  # (chains, sweeps, cells)
        for ci, node in enumerate(blk.nodes):
            c = built["xy_of"][node]
            seq = arr[:, :, ci]
            fl = (seq == 0)
            has = fl.any(axis=1)
            val = np.where(has, fl.argmax(axis=1), n_sweeps + 1)
            if dim == 2:
                x, y = c
                flip[:, y, x] = val
            else:
                x, y, z = c
                flip[:, z, y, x] = val
    gh = built["gh"]
    if dim == 2:
        flip[:, gh[1], gh[0]] = 0
    else:
        flip[:, gh[2], gh[1], gh[0]] = 0
    tmed = np.median(flip, axis=0)
    tmean = flip.mean(axis=0)
    built["_flipcube"] = flip  # R3 SAVE PATCH: capture per-chain cube
    return tmed, tmean, wall_sample


def to_world_field(tfield, built, obs_set):
    """Strip halo -> world coords; obstacles = inf."""
    dim, ws = built["dim"], built["world_shape"]
    sl = tuple(slice(1, -1) for _ in range(dim))
    t = tfield[sl].astype(float)
    if dim == 2:
        W, H = ws
        for y in range(H):
            for x in range(W):
                if (x, y) in obs_set:
                    t[y, x] = np.inf
    else:
        W, H, D = ws
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    if (x, y, z) in obs_set:
                        t[z, y, x] = np.inf
    return t


def smooth_field(t, obs_set, ws):
    """3x3(x3) mean ignoring inf."""
    dim = len(ws)
    ts = t.copy()
    if dim == 2:
        W, H = ws
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if (x, y) in obs_set:
                    continue
                w = t[y - 1:y + 2, x - 1:x + 2]
                f = w[np.isfinite(w)]
                if len(f):
                    ts[y, x] = f.mean()
    else:
        W, H, D = ws
        for z in range(1, D - 1):
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if (x, y, z) in obs_set:
                        continue
                    w = t[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
                    f = w[np.isfinite(w)]
                    if len(f):
                        ts[z, y, x] = f.mean()
    return ts


def follow_gradient(t, start, goal, obs_set, ws, max_steps=2000):
    """Robust gradient following on the flip-time field."""
    dim = len(ws)
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)] if dim == 2 else [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def tget(c):
        return t[c[::-1]] if dim == 2 else t[c[2], c[1], c[0]]

    def blocked(c):
        if any(v < 0 or v >= s for v, s in zip(c, ws)):
            return True
        return c in obs_set

    pos, traj, visited = start, [start], {start}
    for _ in range(max_steps):
        if pos == goal:
            break
        cur = tget(pos)
        cands = []
        for dn in nbrs:
            nb = tuple(a + b for a, b in zip(pos, dn))
            if not blocked(nb):
                cands.append((tget(nb), nb in visited, nb))
        if not cands:
            break
        cands.sort(key=lambda r: (r[0], r[1]))
        moved = False
        for tv, vis, nb in cands:
            if tv < cur - 1e-9 and not vis:
                pos = nb
                moved = True
                break
        if not moved:
            for tv, vis, nb in cands:
                if not vis:
                    pos = nb
                    moved = True
                    break
        if not moved:
            break
        visited.add(pos)
        traj.append(pos)
    return dict(reached=pos == goal, steps=len(traj) - 1, final=pos,
                traj=traj)


def bfs_dist_3d(obs_set, goal, ws):
    W, H, D = ws
    dist = np.full((D, H, W), -1)
    dq = deque([goal])
    dist[goal[2], goal[1], goal[0]] = 0
    while dq:
        x, y, z = dq.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D \
                    and (nx, ny, nz) not in obs_set \
                    and dist[nz, ny, nx] < 0:
                dist[nz, ny, nx] = dist[z, y, x] + 1
                dq.append((nx, ny, nz))
    return dist


# ------------------------------------------------------------------ bars
def bar1_2d_corr():
    """Bar 1: flip-time field from Gibbs dynamics ~ geodesic (2D)."""
    print("--- Bar 1: 2D wavefront vs BFS ---", flush=True)
    N = 24
    rng = np.random.default_rng(5)
    obs = set()
    for wx in (8, 16):
        gy = int(rng.integers(4, 18))
        for y in range(2, 22):
            if not (gy <= y < gy + 4):
                obs.add((wx, y))
    goal, start = (21, 12), (2, 12)
    built = build_wave((N, N), obs, goal, P2, n_sweeps=120, n_chains=16)
    print("  model: %d spins, hash %s" % (built["n_free"] + 1,
                                         built["prog_hash"][:12]), flush=True)
    tmed, tmean, ws = run_wavefront(built)
    save_flip_field("bar1_2d", built, tmed, tmean, obs, start=start)  # R3
    t = to_world_field(tmean, built, obs)
    ts = smooth_field(t, obs, (N, N))
    # BFS reference
    dist = np.full((N, N), -1)
    dq = deque([goal])
    dist[goal[1], goal[0]] = 0
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in obs \
                    and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                dq.append((nx, ny))
    mask = np.array([[(x, y) not in obs for x in range(N)]
                     for y in range(N)]) & (dist >= 0)
    tm, dm = ts[mask], dist[mask].astype(float)
    fin = np.isfinite(tm)
    corr = float(np.corrcoef(tm[fin], dm[fin])[0, 1])
    print("  corr = %.3f (n=%d), wavefront wall %.1fs" % (
        corr, fin.sum(), ws), flush=True)
    r = follow_gradient(ts, start, goal, obs, (N, N))
    print("  follow: reached=%s steps=%d (bfs %d)" % (
        r["reached"], r["steps"], dist[start[1], start[0]]), flush=True)
    return {"corr": corr, "reached": r["reached"], "steps": r["steps"],
            "bfs": int(dist[start[1], start[0]]), "wall_s": ws}


def bar4_ablation_j0():
    """Bar 4: J=0 -> no coupling -> uniform field -> routing fails."""
    print("--- Bar 4: ablation J=0 ---", flush=True)
    N = 24
    rng = np.random.default_rng(5)
    obs = set()
    for wx in (8, 16):
        gy = int(rng.integers(4, 18))
        for y in range(2, 22):
            if not (gy <= y < gy + 4):
                obs.add((wx, y))
    goal, start = (21, 12), (2, 12)
    p0 = dict(P2)
    p0["J"] = np.float32(0.0)
    built = build_wave((N, N), obs, goal, p0, n_sweeps=40, n_chains=8)
    tmed, tmean, ws = run_wavefront(built)
    save_flip_field("bar4_j0", built, tmed, tmean, obs, start=start)  # R3
    t = to_world_field(tmean, built, obs)
    fin = t[np.isfinite(t)]
    print("  J=0 field: mean=%.2f std=%.2f (uniform => std~0)" % (
        fin.mean(), fin.std()), flush=True)
    r = follow_gradient(t, start, goal, obs, (N, N), max_steps=500)
    print("  J=0 follow: reached=%s steps=%d" % (r["reached"], r["steps"]),
          flush=True)
    return {"field_std": float(fin.std()), "reached": bool(r["reached"])}


def bar25_maze_3d(n_mazes=3, seed0=5000):
    """Bars 2+5: 3D ceilinged mazes (2m generator, roof at z=15)."""
    print("--- Bars 2+5: 3D ceilinged mazes ---", flush=True)
    out = []
    for m in range(n_mazes):
        rng = np.random.default_rng(seed0 + m)
        gen = build_ceiling_maze(rng)
        # regenerate on sampler-collapse (2m methodology)
        while gen is None:
            rng = np.random.default_rng(int(rng.integers(0, 1e9)))
            gen = build_ceiling_maze(rng)
        rocks, start, goal, spec, wall = gen
        ws = (40, 40, 16)
        t0 = time.time()
        built = build_wave(ws, rocks, goal, P3, n_sweeps=260, n_chains=16)
        tmed, tmean, wsw = run_wavefront(built)
        t = to_world_field(tmean, built, rocks)
        ts = smooth_field(t, rocks, ws)
        # R3 SAVE PATCH: save the 3D field (the original computed no corr)
        save_flip_field("maze%d_3d" % m, built, tmed, tmean, rocks,
                        start=start)
        # corr vs BFS on a coarse check (sample cells)
        r = follow_gradient(ts, start, goal, rocks, ws, max_steps=3000)
        dt = time.time() - t0
        print("  maze %d: reached=%s steps=%d wavefront=%.0fs total=%.0fs" % (
            m, r["reached"], r["steps"], wsw, dt), flush=True)
        out.append({"maze": m, "reached": bool(r["reached"]),
                    "steps": r["steps"], "final": list(r["final"]),
                    "wavefront_s": round(wsw, 1), "total_s": round(dt, 1),
                    "rails": {"max_abs_h": built["max_abs_h"],
                              "max_abs_J": built["max_abs_J"],
                              "max_deg": built["max_deg"],
                              "intra": built["intra"],
                              "n_spins": built["n_free"] + 1}})
    return out


def main():
    t0 = time.time()
    print("=== LEGO 2n-wave: geodesic wavefront planner ===", flush=True)
    results = {"bars": {}}
    b1 = bar1_2d_corr()
    results["bar1"] = b1
    b4 = bar4_ablation_j0()
    results["bar4"] = b4
    mz = bar25_maze_3d()
    results["mazes"] = mz

    z1 = all(m["rails"]["max_abs_h"] <= 6 and m["rails"]["max_abs_J"] <= 6
             and m["rails"]["max_deg"] <= 16 and m["rails"]["intra"] == 0
             for m in mz)
    results["bars"] = {
        "bar1_wavefront_geodesic": bool(b1["corr"] >= 0.95),
        "bar2_maze_reached": [m["reached"] for m in mz],
        "bar3_z1_rails": bool(z1),
        "bar4_ablation_j0_kills_routing": bool(
            not b4["reached"] and b4["field_std"] < 2.0),
        "bar5_ceilinged": bool(all(m["reached"] for m in mz)),
    }
    results["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUTDIR, "results_2n_wave.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("=== BARS ===", flush=True)
    print(json.dumps(results["bars"], indent=1), flush=True)
    print("wall %.0fs" % results["wall_s"], flush=True)


if __name__ == "__main__":
    main()
