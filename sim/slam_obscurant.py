#!/usr/bin/env python3
"""Phase 4 Stage A: perception sampler robustness under particle obscurant.

Mirrors sampler_tracking_test.py (TRACKING_TEST.md) exactly, except the
sensor: scan_pose/integrate_scan are replaced by obscurant.scan_pose /
obscurant.integrate_scan at densities {clear, light, moderate, heavy,
whiteout}. Everything else identical: h through quantize_8bit, boundary
prior, logodds accumulated across slams, THRML program with bias-swapping,
2 checkerboard blocks, sequential block Gibbs.

PRE-REGISTERED HYPOTHESIS (from the task brief):
  The tracking test's 0/132 no-trapping result came from a field-dominated
  landscape (|h|<=4 vs J=0.25, 16:1). Obscurants weaken TRUE field (fewer
  returns = weaker evidence) while backscatter ADDS SPURIOUS field. Two
  candidate failure modes:
  (H1) field-weakening -> coupling-dominated -> sampler TRAPS in stale minima;
  (H2) field-corruption -> sampler confidently tracks PHANTOMS (no trapping,
       but wrong map -> D5 freezes).
  The experiment distinguishes them: H1 fires the TRAPPED criterion;
  H2 shows low L1_to_ref (tracks *something* confidently) with HIGH phantom
  rate (marg>0.5 on GT-free voxels).

Scenarios per (room, density):
  jump4     : converge at pose A (cold), jump 4 voxels, warm 200. (motion
              tracking under obscurant)
  dens_slam : converge at CLEAR (cold), slam density clear->D, warm 200.
              (fog-rolls-in; D in {light, moderate, heavy, whiteout})
  appear    : 3x3x3 cube APPEARS near drone under density D, warm 200.
  disappear : 3x3x3 cube DISAPPEARS under density D, warm 200.

TRAPPED criterion (same as TRACKING_TEST, fixed before running):
  trapped = (L1_wp < L1_wr) AND (L1_wr > 2*noise_floor).

New metrics vs density: phantom voxel rate (marg>0.5 & GT-free), miss rate
(GT-occ & marg<0.5), mean|h| on observed voxels (field strength).

GT discipline: GT occupancy only in the raycaster (as in flight) and for
scenario construction / metric computation. Never fed to the sampler.
References analysis-only, never fed back.

Every number [measured in simulation, THRML backend].
XLA_PYTHON_CLIENT_PREALLOCATE=false set by the caller.
"""
import os, sys, time, json

sys.path.insert(0, "/home/hatch/workspace/flynav-redteam/slow-drone")
sys.path.insert(0, "/home/hatch/workspace/flynav-redteam/slow-drone/dynamics_sim")

import numpy as np

import jax
import jax.numpy as jnp
from thrml import SamplingSchedule

import obscurant
import head_to_head_thrml as hth

SLOWDRONE = "/home/hatch/workspace/flynav-redteam/slow-drone"
OUTDIR = os.path.join(SLOWDRONE, "dynamics_sim")

ROOMS_JUMP = ["indoorcd_128", "indoorcd_004", "indoorcd_053"]
ROOMS_SLAM = ["indoorcd_128", "indoorcd_004"]
ROOMS_CUBE = ["indoorcd_004"]
PREFERRED_START = {
    "indoorcd_128": (16.5, 6.5, 8.5),
    "indoorcd_004": (26.5, 8.5, 8.5),
    "indoorcd_053": (16.5, 16.5, 8.5),
}
DENSITIES = ["clear", "light", "moderate", "heavy", "whiteout"]
ARCHETYPE = "fog"
BUDGET, BURN = 200, 50
COLD_SWEEPS, COLD_BURN = 300, 100
REF_SWEEPS, REF_BURN = 800, 200
N_SEEDS = 2
J01 = 0.25
NOISE, N_AZ = 0.15, 60
ELEVATIONS = (-0.26, 0.0, 0.26)
JUMP_K = 4


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


def fresh_logodds(shape):
    lo = np.zeros(shape)
    lo[0, :, :] = 2.0; lo[-1, :, :] = 2.0
    lo[:, 0, :] = 2.0; lo[:, -1, :] = 2.0
    lo[:, :, 0] = 2.0; lo[:, :, -1, ] = 2.0
    return lo


def scan_into(logodds, occ, pose, scan_seed, density):
    obscurant.configure(ARCHETYPE, density)
    scans = obscurant.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEVATIONS,
                                max_range=40.0, range_noise=NOISE,
                                seed=scan_seed)
    _, n_hits, n_skipped = obscurant.integrate_scan(logodds, occ.shape, scans,
                                                   max_range=40.0)
    return sensor_quant(logodds), n_hits, n_skipped


def sensor_quant(logodds):
    lo = np.clip(logodds, -4.0, 4.0)
    q = np.round((lo + 4.0) / 8.0 * 255.0) / 255.0 * 8.0 - 4.0
    return q.astype(np.float64)


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
    prog = hth._swap_perc_biases(entry, h)
    n_samples = int(n_sweeps - burn_in)
    sched = SamplingSchedule(int(burn_in), n_samples, 1)
    x0 = (np.zeros(entry["n"], dtype=bool) if x_init is None
          else np.asarray(x_init).ravel().astype(bool))
    init = [jnp.asarray(x0[entry["black_idx"]]),
            jnp.asarray(x0[entry["white_idx"]])]
    key = jax.random.key(int(seed))
    t0 = time.time()
    sm = entry["sampler"](key, prog, sched, init)
    jax.block_until_ready(sm)
    dt = time.time() - t0
    arr = np.asarray(sm[0], dtype=bool)
    return arr.mean(axis=0), arr, dt


def l1(a, b, mask=None):
    d = np.abs(a - b)
    return float(d[mask].mean()) if mask is not None else float(d.mean())


def phantom_miss_rates(marg, occ, h):
    """Operationally-meaningful rates, matching what D5/planner act on
    (P_OBSTACLE=0.3, observed voxels only):
      phantom: GT-free & observed & marg>0.3 (would make D5 refuse)
      miss:    GT-occ & observed & marg<0.3 (would let D5 fly into a wall)
    Boundary prior voxels excluded (pinned at 2.0 by design)."""
    marg = np.asarray(marg).reshape(occ.shape)
    h = np.asarray(h).reshape(occ.shape)
    interior = np.ones(occ.shape, dtype=bool)
    interior[0, :, :] = False; interior[-1, :, :] = False
    interior[:, 0, :] = False; interior[:, -1, :] = False
    interior[:, :, 0] = False; interior[:, :, -1] = False
    observed = interior & (np.abs(h) > 0.05)
    free = observed & (occ == 0)
    occupied = observed & (occ == 1)
    ph = float(((marg > 0.3) & free).sum() / max(1, free.sum()))
    mi = float(((marg < 0.3) & occupied).sum() / max(1, occupied.sum()))
    return ph, mi


def main():
    entry = hth._perc_program((32, 32, 16))
    results = {"config": {
        "archetype": ARCHETYPE, "densities": DENSITIES,
        "budget": [BUDGET, BURN], "cold": [COLD_SWEEPS, COLD_BURN],
        "ref": [REF_SWEEPS, REF_BURN], "backend": "THRML",
        "claim": "Z1-legal formulation, validated in simulation",
        "hypothesis": "H1 field-weakening->trapping vs H2 field-corruption->phantoms",
        "trapped_criterion": "L1_wp < L1_wr AND L1_wr > 2*noise_floor",
    }, "scenarios": {}}

    def refs_for(h, base_seed):
        ms = []
        for s in range(N_SEEDS):
            m, _, _ = run_thrml(entry, h, REF_SWEEPS, REF_BURN,
                                seed=base_seed + s, x_init=None)
            ms.append(m)
        ref = np.mean(ms, axis=0)
        nf = float(np.mean([l1(ms[i], ms[j]) for i in range(N_SEEDS)
                            for j in range(i + 1, N_SEEDS)])) if N_SEEDS > 1 else 0.016
        return ref, nf

    # ---- Scenario 1: translational jump under fixed density ----
    for ri, room in enumerate(ROOMS_JUMP):
        occ = load_occ(room)
        pose_a = pick_start(occ, PREFERRED_START[room])
        pose_b = (pose_a[0] + JUMP_K, pose_a[1], pose_a[2])
        assert voxel_free(occ, pose_b), f"{room} jump target blocked"
        for di, dens in enumerate(DENSITIES):
            key = f"jump4/{room}/{dens}"
            # converge at A under density
            lo_a = fresh_logodds(occ.shape)
            h_a, _, _ = scan_into(lo_a, occ, pose_a, 1000 + ri * 100, dens)
            xlast, margs_a = [], []
            for s in range(N_SEEDS):
                m, samples, _ = run_thrml(entry, h_a, COLD_SWEEPS, COLD_BURN,
                                          seed=10 + s, x_init=None)
                xlast.append(samples[-1]); margs_a.append(m)
            marg_a = np.mean(margs_a, axis=0)
            # jump: integrate new scan into SAME logodds (production behavior)
            lo_b = lo_a.copy()
            h_b, _, _ = scan_into(lo_b, occ, pose_b, 2000 + ri * 100 + di, dens)
            ref, nf = refs_for(h_b, 5000 + ri * 100 + di * 10)
            ph_ref, mi_ref = phantom_miss_rates(ref, occ, h_b)
            seeds = []
            for s in range(N_SEEDS):
                marg_w, _, dt = run_thrml(entry, h_b, BUDGET, BURN,
                                          seed=9000 + ri * 100 + di * 10 + s,
                                          x_init=xlast[s])
                L1_wr, L1_wp = l1(marg_w, ref), l1(marg_w, marg_a)
                trapped = bool((L1_wp < L1_wr) and (L1_wr > 2 * nf))
                ph, mi = phantom_miss_rates(marg_w, occ, h_b)
                seeds.append({"L1_to_ref": round(L1_wr, 4),
                              "L1_to_prejump": round(L1_wp, 4),
                              "trapped": trapped,
                              "phantom_rate": round(ph, 4),
                              "miss_rate": round(mi, 4),
                              "wall_s": round(dt, 1)})
            results["scenarios"][key] = {
                "noise_floor_L1": round(nf, 4),
                "ref_phantom_rate": round(ph_ref, 4),
                "ref_miss_rate": round(mi_ref, 4),
                "mean_abs_h": round(float(np.abs(h_b).mean()), 3),
                "trapped": all(x["trapped"] for x in seeds),
                "seeds": seeds}
            print(f"{key}: L1_ref={np.mean([x['L1_to_ref'] for x in seeds]):.4f} "
                  f"phantom={np.mean([x['phantom_rate'] for x in seeds]):.4f} "
                  f"trapped={results['scenarios'][key]['trapped']}", flush=True)

    # ---- Scenario 2: density slam (fog rolls in) ----
    for ri, room in enumerate(ROOMS_SLAM):
        occ = load_occ(room)
        pose = pick_start(occ, PREFERRED_START[room])
        lo_clear = fresh_logodds(occ.shape)
        h_clear, _, _ = scan_into(lo_clear, occ, pose, 3000 + ri * 100, "clear")
        xlast, margs_c = [], []
        for s in range(N_SEEDS):
            m, samples, _ = run_thrml(entry, h_clear, COLD_SWEEPS, COLD_BURN,
                                      seed=20 + s, x_init=None)
            xlast.append(samples[-1]); margs_c.append(m)
        marg_clear = np.mean(margs_c, axis=0)
        for di, dens in enumerate([d for d in DENSITIES if d != "clear"]):
            key = f"dens_slam/{room}/clear->{dens}"
            lo_d = lo_clear.copy()
            h_d, _, _ = scan_into(lo_d, occ, pose, 4000 + ri * 100 + di, dens)
            ref, nf = refs_for(h_d, 6000 + ri * 100 + di * 10)
            ph_ref, mi_ref = phantom_miss_rates(ref, occ, h_d)
            seeds = []
            for s in range(N_SEEDS):
                marg_w, _, dt = run_thrml(entry, h_d, BUDGET, BURN,
                                          seed=9500 + ri * 100 + di * 10 + s,
                                          x_init=xlast[s])
                L1_wr, L1_wp = l1(marg_w, ref), l1(marg_w, marg_clear)
                trapped = bool((L1_wp < L1_wr) and (L1_wr > 2 * nf))
                ph, mi = phantom_miss_rates(marg_w, occ, h_d)
                seeds.append({"L1_to_ref": round(L1_wr, 4),
                              "L1_to_preclear": round(L1_wp, 4),
                              "trapped": trapped,
                              "phantom_rate": round(ph, 4),
                              "miss_rate": round(mi, 4),
                              "wall_s": round(dt, 1)})
            results["scenarios"][key] = {
                "noise_floor_L1": round(nf, 4),
                "ref_phantom_rate": round(ph_ref, 4),
                "ref_miss_rate": round(mi_ref, 4),
                "mean_abs_h": round(float(np.abs(h_d).mean()), 3),
                "trapped": all(x["trapped"] for x in seeds),
                "seeds": seeds}
            print(f"{key}: L1_ref={np.mean([x['L1_to_ref'] for x in seeds]):.4f} "
                  f"phantom={np.mean([x['phantom_rate'] for x in seeds]):.4f} "
                  f"trapped={results['scenarios'][key]['trapped']}", flush=True)

    # ---- Scenario 3: teleported cube appear/disappear under density ----
    for ri, room in enumerate(ROOMS_CUBE):
        occ = load_occ(room)
        pose = pick_start(occ, PREFERRED_START[room])
        # cube location: 6 voxels ahead, must be free in base occ
        cx, cy, cz = int(pose[0]) + 6, int(pose[1]), int(pose[2])
        assert not occ[cx - 1:cx + 2, cy - 1:cy + 2, cz - 1:cz + 2].any()
        occ_cube = occ.copy()
        occ_cube[cx - 1:cx + 2, cy - 1:cy + 2, cz - 1:cz + 2] = 1
        for di, dens in enumerate(DENSITIES):
            for mode, o_used, tag in (("appear", occ_cube, "cube"),
                                      ("disappear", occ, "base")):
                # converge WITHOUT the change, then slam WITH it
                o_pre = occ if mode == "appear" else occ_cube
                o_post = occ_cube if mode == "appear" else occ
                key = f"cube_{mode}/{room}/{dens}"
                lo = fresh_logodds(o_pre.shape)
                h_pre, _, _ = scan_into(lo, o_pre, pose, 7000 + ri * 100, dens)
                xlast, margs_p = [], []
                for s in range(N_SEEDS):
                    m, samples, _ = run_thrml(entry, h_pre, COLD_SWEEPS,
                                              COLD_BURN, seed=30 + s,
                                              x_init=None)
                    xlast.append(samples[-1]); margs_p.append(m)
                marg_pre = np.mean(margs_p, axis=0)
                h_post, _, _ = scan_into(lo, o_post, pose,
                                         8000 + ri * 100 + di, dens)
                ref, nf = refs_for(h_post, 9000 + ri * 100 + di * 10)
                ph_ref, mi_ref = phantom_miss_rates(ref, o_post, h_post)
                seeds = []
                for s in range(N_SEEDS):
                    marg_w, _, dt = run_thrml(entry, h_post, BUDGET, BURN,
                                              seed=9800 + ri * 100 + di * 10 + s,
                                              x_init=xlast[s])
                    L1_wr, L1_wp = l1(marg_w, ref), l1(marg_w, marg_pre)
                    trapped = bool((L1_wp < L1_wr) and (L1_wr > 2 * nf))
                    ph, mi = phantom_miss_rates(marg_w, o_post, h_post)
                    seeds.append({"L1_to_ref": round(L1_wr, 4),
                                  "L1_to_preshock": round(L1_wp, 4),
                                  "trapped": trapped,
                                  "phantom_rate": round(ph, 4),
                                  "miss_rate": round(mi, 4),
                                  "wall_s": round(dt, 1)})
                results["scenarios"][key] = {
                    "noise_floor_L1": round(nf, 4),
                    "ref_phantom_rate": round(ph_ref, 4),
                    "ref_miss_rate": round(mi_ref, 4),
                    "mean_abs_h": round(float(np.abs(h_post).mean()), 3),
                    "trapped": all(x["trapped"] for x in seeds),
                    "seeds": seeds}
                print(f"{key}: L1_ref={np.mean([x['L1_to_ref'] for x in seeds]):.4f} "
                      f"phantom={np.mean([x['phantom_rate'] for x in seeds]):.4f} "
                      f"trapped={results['scenarios'][key]['trapped']}", flush=True)

    out = os.path.join(OUTDIR, "SLAM_OBSCURANT.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
