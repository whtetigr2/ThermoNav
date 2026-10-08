#!/usr/bin/env python3
"""S16 perception sampler TRACKING stress test.

Question: does the warm-started THRML Gibbs sampler track rapid posterior
shifts (fast sensor-pose jumps), or get trapped in the pre-jump minimum?

Mirrors production exactly:
  - h built through the real sensor pipeline (scan_pose -> integrate_scan ->
    quantize_8bit), boundary prior, logodds ACCUMULATED across the jump
    (as in fly.py / head_to_head.py mission loops).
  - THRML perception program with bias-swapping (head_to_head_thrml), 2
    checkerboard blocks, sequential block Gibbs.
  - Step 0: cold run (300 sweeps / 100 burn-in, all-False init).
    Later: warm run from x_last (previous run's last sample).
  - GT occupancy used ONLY in the raycaster (as in flight) and for
    scenario construction (pose validity). Never fed to the sampler.
    Reference runs are analysis-only, never fed back.

Protocol per (room, jump_k):
  1. Converge at pose A: cold runs (2 seeds) -> marg_A, x_last_A per seed.
  2. Jump sensor pose by k voxels; integrate new scan into SAME logodds -> h_B.
  3. Warm-start from x_last_A at budgets {50, 100, 200} sweeps
     (burn_in = budget//4, mirroring the 200/50 production ratio), 2 seeds.
  4. Reference at B: cold 1200/200 runs, 3 seeds -> ref_B, noise floor.
  5. Metrics: marginal L1 to ref, marginal L1 to pre-jump state,
     best-energy gap, TRAPPED criterion (defined up front, see below).
  6. Control: gradual 1-voxel x4 chain with production warm schedule.

TRAPPED criterion (fixed before running):
  L1_wr = mean|marg_warm - ref_B| ; L1_wp = mean|marg_warm - marg_A|
  trapped = (L1_wp < L1_wr) AND (L1_wr > 2 * noise_floor)
  i.e. the warm marginals are closer to the PRE-jump state than to the
  converged post-jump posterior, and the gap to the reference is real
  (not Monte Carlo noise).

Every number [measured in simulation, THRML backend]. JAX preallocation
disabled via XLA_PYTHON_CLIENT_PREALLOCATE=false (set by the caller).
"""
import os, sys, time, json

sys.path.insert(0, "/home/hatch/workspace/flynav-redteam/slow-drone")
sys.path.insert(0, "/home/hatch/workspace/flynav-redteam/slow-drone/drift_decomp")

import numpy as np

import jax
import jax.numpy as jnp
from thrml import SamplingSchedule

import sensor
import head_to_head_thrml as hth

# ---------------------------------------------------------------- config
SLOWDRONE = "/home/hatch/workspace/flynav-redteam/slow-drone"
OUTDIR = os.path.join(SLOWDRONE, "dynamics_sim")

ROOMS = ["indoorcd_128", "indoorcd_004", "indoorcd_053"]
PREFERRED_START = {
    "indoorcd_128": (16.5, 6.5, 8.5),
    "indoorcd_004": (26.5, 8.5, 8.5),
    "indoorcd_053": (16.5, 16.5, 8.5),
}
JUMPS = [1, 2, 4, 8]
BUDGETS = [50, 100, 200]          # warm sweeps; burn_in = budget // 4
COLD_SWEEPS, COLD_BURN = 300, 100
REF_SWEEPS, REF_BURN = 1200, 200
N_COLD_SEEDS, N_REF_SEEDS, N_WARM_SEEDS = 2, 3, 2
J01 = 0.25
NOISE, N_AZ = 0.15, 60
ELEVATIONS = (-0.26, 0.0, 0.26)

rng_global = np.random.default_rng(7)


# ---------------------------------------------------------------- helpers
def load_occ(room):
    return np.load(os.path.join(SLOWDRONE, "real_lidar", f"{room}_occ.npy"))


def voxel_free(occ, pose):
    ix, iy, iz = int(pose[0]), int(pose[1]), int(pose[2])
    nx, ny, nz = occ.shape
    return (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz
            and not occ[ix, iy, iz])


def pick_start(occ, preferred):
    if voxel_free(occ, preferred):
        return tuple(float(v) for v in preferred)
    # fallback: nearest free voxel to center
    nx, ny, nz = occ.shape
    best, bd = None, 1e9
    for x in range(2, nx - 2, 2):
        for y in range(2, ny - 2, 2):
            if not occ[x, y, 8]:
                d = (x - nx / 2) ** 2 + (y - ny / 2) ** 2
                if d < bd:
                    bd, best = d, (x + 0.5, y + 0.5, 8.5)
    assert best is not None
    return best


def pick_jump_dir(occ, pose, jumps):
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
        ok = True
        for k in jumps:
            p = (pose[0] + k * d[0], pose[1] + k * d[1], pose[2] + k * d[2])
            if not voxel_free(occ, p):
                ok = False
                break
        if ok:
            return d
    raise RuntimeError("no clean jump direction")


def fresh_logodds(shape):
    lo = np.zeros(shape)
    lo[0, :, :] = 2.0; lo[-1, :, :] = 2.0
    lo[:, 0, :] = 2.0; lo[:, -1, :] = 2.0
    lo[:, :, 0] = 2.0; lo[:, :, -1] = 2.0
    return lo


def scan_into(logodds, occ, pose, scan_seed):
    scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEVATIONS,
                             max_range=40.0, range_noise=NOISE, seed=scan_seed)
    sensor.integrate_scan(logodds, occ.shape, scans, max_range=40.0)
    return sensor.quantize_8bit(logodds)


_PAIRS_CACHE = {}


def neighbor_pairs(shape):
    if shape in _PAIRS_CACHE:
        return _PAIRS_CACHE[shape]
    nx, ny, nz = shape
    pairs = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = (x * ny + y) * nz + z
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    x2, y2, z2 = x + dx, y + dy, z + dz
                    if x2 < nx and y2 < ny and z2 < nz:
                        pairs.append((i, (x2 * ny + y2) * nz + z2))
    arr = np.asarray(pairs, dtype=np.int64)
    _PAIRS_CACHE[shape] = arr
    return arr


def run_thrml(entry, h, n_sweeps, burn_in, seed, x_init):
    """One THRML perception run. Returns (marginals, samples_bool, wall_s)."""
    prog = hth._swap_perc_biases(entry, h)
    n_samples = int(n_sweeps - burn_in)
    sched = SamplingSchedule(int(burn_in), n_samples, 1)
    if x_init is None:
        x0 = np.zeros(entry["n"], dtype=bool)
    else:
        x0 = np.asarray(x_init).ravel().astype(bool)
    init = [jnp.asarray(x0[entry["black_idx"]]),
            jnp.asarray(x0[entry["white_idx"]])]
    key = jax.random.key(int(seed))
    t0 = time.time()
    sm = entry["sampler"](key, prog, sched, init)
    jax.block_until_ready(sm)
    dt = time.time() - t0
    arr = np.asarray(sm[0], dtype=bool)  # (n_samples, n)
    marg = arr.mean(axis=0)
    return marg, arr, dt


def best_energy(h, samples):
    """Min NumPy-{0,1} energy over samples. h: (n,) fields, J01 agreement."""
    pairs = neighbor_pairs((32, 32, 16))
    x = samples.astype(np.float64)
    E = -(x * h.ravel()).sum(axis=1)
    E -= J01 * (x[:, pairs[:, 0]] == x[:, pairs[:, 1]]).sum(axis=1)
    return float(E.min())


def l1(a, b, mask=None):
    d = np.abs(a - b)
    return float(d[mask].mean()) if mask is not None else float(d.mean())

# ---------------------------------------------------------------- experiment
def main():
    entry = hth._perc_program((32, 32, 16))
    results = {"config": {
        "rooms": ROOMS, "jumps": JUMPS, "budgets": BUDGETS,
        "cold": [COLD_SWEEPS, COLD_BURN], "ref": [REF_SWEEPS, REF_BURN],
        "backend": "THRML", "claim": "Z1-legal formulation, validated in simulation",
        "trapped_criterion": ("L1_wp < L1_wr AND L1_wr > 2*noise_floor, "
                              "where L1_wp=|warm-pre|, L1_wr=|warm-ref|"),
    }, "rooms": {}}

    for ri, room in enumerate(ROOMS):
        occ = load_occ(room)
        pose_a = pick_start(occ, PREFERRED_START[room])
        jdir = pick_jump_dir(occ, pose_a, JUMPS)
        print(f"\n=== {room}: pose_a={tuple(round(v,1) for v in pose_a)} "
              f"jump_dir={jdir}", flush=True)
        rres = {"pose_a": pose_a, "jump_dir": jdir, "jumps": {}}

        # ---- converge at pose A (cold, 2 seeds)
        lo_a = fresh_logodds(occ.shape)
        scan_seed_a = 1000 + ri * 100
        h_a = scan_into(lo_a, occ, pose_a, scan_seed_a)
        marg_a_list, xlast_a = [], []
        for s in range(N_COLD_SEEDS):
            marg, samples, dt = run_thrml(entry, h_a, COLD_SWEEPS, COLD_BURN,
                                          seed=10 + s, x_init=None)
            marg_a_list.append(marg)
            xlast_a.append(samples[-1])
            print(f"  cold@A seed {s}: {dt:.1f}s", flush=True)
        marg_a = np.mean(marg_a_list, axis=0)
        rres["coldA_interseed_L1"] = l1(marg_a_list[0], marg_a_list[1])

        # ---- jumps
        for ji, k in enumerate(JUMPS):
            pose_b = (pose_a[0] + k * jdir[0], pose_a[1] + k * jdir[1],
                      pose_a[2] + k * jdir[2])
            # production behavior: integrate new scan into the SAME logodds
            lo_b = lo_a.copy()
            scan_seed_b = 1000 + ri * 100 + 10 + ji
            h_b = scan_into(lo_b, occ, pose_b, scan_seed_b)
            obs_b = np.abs(lo_b) > 0.05

            # reference: long cold runs at B (analysis only, never fed back)
            ref_margs = []
            ref_best = np.inf
            for s in range(N_REF_SEEDS):
                marg, samples, dt = run_thrml(entry, h_b, REF_SWEEPS, REF_BURN,
                                              seed=5000 + ri * 100 + ji * 10 + s,
                                              x_init=None)
                ref_margs.append(marg)
                ref_best = min(ref_best, best_energy(h_b, samples[::4]))
            ref = np.mean(ref_margs, axis=0)
            noise_floor = float(np.mean(
                [l1(ref_margs[i], ref_margs[j])
                 for i in range(N_REF_SEEDS) for j in range(i + 1, N_REF_SEEDS)]))

            jres = {"pose_b": pose_b, "noise_floor_L1": noise_floor,
                    "ref_best_energy": ref_best, "budgets": {}}
            # warm-started runs at B, paired: warm seed s <- xlast_A seed s
            for budget in BUDGETS:
                burn = budget // 4
                bres = {"trapped_seeds": 0, "seeds": []}
                for s in range(N_WARM_SEEDS):
                    marg_w, samples_w, dt = run_thrml(
                        entry, h_b, budget, burn,
                        seed=9000 + ri * 100 + ji * 10 + s,
                        x_init=xlast_a[s % N_COLD_SEEDS])
                    L1_wr = l1(marg_w, ref)
                    L1_wp = l1(marg_w, marg_a)
                    L1_wr_obs = l1(marg_w, ref, obs_b.ravel())
                    trapped = bool((L1_wp < L1_wr) and (L1_wr > 2 * noise_floor))
                    bres["trapped_seeds"] += int(trapped)
                    bres["seeds"].append({
                        "L1_to_ref": round(L1_wr, 4),
                        "L1_to_ref_obs": round(L1_wr_obs, 4),
                        "L1_to_prejump": round(L1_wp, 4),
                        "energy_gap": round(best_energy(h_b, samples_w) - ref_best, 3),
                        "trapped": trapped, "wall_s": round(dt, 1),
                    })
                bres["trapped"] = bres["trapped_seeds"] == N_WARM_SEEDS
                jres["budgets"][str(budget)] = bres
                print(f"  jump {k} budget {budget}: "
                      f"L1_ref={np.mean([x['L1_to_ref'] for x in bres['seeds']]):.4f} "
                      f"trapped={bres['trapped']}", flush=True)
            rres["jumps"][str(k)] = jres

        # ---- control: gradual 1-voxel x4 chain, production warm schedule
        pose, lo_c = pose_a, fresh_logodds(occ.shape)
        h_c = scan_into(lo_c, occ, pose, 1000 + ri * 100)
        marg, samples, _ = run_thrml(entry, h_c, COLD_SWEEPS, COLD_BURN,
                                     seed=10, x_init=None)
        x_last = samples[-1]
        for step in range(1, 5):
            pose = (pose[0] + jdir[0], pose[1] + jdir[1], pose[2] + jdir[2])
            h_c = scan_into(lo_c, occ, pose, 1000 + ri * 100 + 40 + step)
            marg, samples, _ = run_thrml(entry, h_c, 200, 50,
                                         seed=9000 + step, x_init=x_last)
            x_last = samples[-1]
        # reference at final pose of the chain
        ref_margs = [run_thrml(entry, h_c, REF_SWEEPS, REF_BURN,
                               seed=6000 + s, x_init=None)[0]
                     for s in range(N_REF_SEEDS)]
        ref_c = np.mean(ref_margs, axis=0)
        nf_c = float(np.mean([l1(ref_margs[i], ref_margs[j])
                              for i in range(N_REF_SEEDS)
                              for j in range(i + 1, N_REF_SEEDS)]))
        rres["control"] = {
            "final_pose": pose,
            "L1_chain_to_ref": round(l1(marg, ref_c), 4),
            "noise_floor_L1": round(nf_c, 4),
        }
        print(f"  control chain L1 to ref: {rres['control']['L1_chain_to_ref']:.4f} "
              f"(floor {nf_c:.4f})", flush=True)
        results["rooms"][room] = rres

    out = os.path.join(OUTDIR, "TRACKING_TEST.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
