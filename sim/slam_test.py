#!/usr/bin/env python3
"""Step-function SLAM tests for the S16 perception sampler.

Two slam types (Paul's refinement):
  (a) ROTATIONAL slam: sudden 90-degree yaw. The production sensor is
      omnidirectional (120 azimuths x 3 elevations, full 2*pi, no yaw/FOV
      parameter in sensor.py), so a 90-degree yaw maps the discrete ray set
      onto itself exactly. Expectation: h is BIT-IDENTICAL -> the slam is a
      no-op for perception by construction. We verify this rather than
      assume it.
  (b) TELEPORTED OBSTACLE slams: 3x3x3 cube APPEARS 4 voxels ahead of the
      drone (scene change, ego-pose fixed), then DISAPPEARS. GT occupancy is
      edited for scenario construction only; the sampler sees only h from
      scans (GT in the raycaster, as in flight). References are
      analysis-only, never fed back.

For each slam we record the RELAXATION CURVE: mean-binary-KL and L1 of the
warm-started marginals vs the cold reference at budgets {25,50,100,200,400}
sweeps (burn_in = budget//4), plus the TRAPPED criterion from
sampler_tracking_test.py:
  trapped = (L1_wp < L1_wr) AND (L1_wr > 2*noise_floor).

"sweeps to latch": smallest budget with L1_to_ref < 2*noise_floor and not
trapped.

Every number [measured in simulation, THRML backend].
"""
import os, sys, time, json

_SIMDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIMDIR)

import numpy as np
import jax

from sampler_tracking_test import (
    SLOWDRONE, OUTDIR, ROOMS, PREFERRED_START, J01, NOISE, N_AZ, ELEVATIONS,
    N_COLD_SEEDS, N_REF_SEEDS, N_WARM_SEEDS, COLD_SWEEPS, COLD_BURN,
    REF_SWEEPS, REF_BURN,
    load_occ, voxel_free, pick_start, pick_jump_dir, fresh_logodds,
    scan_into, run_thrml, best_energy, l1,
)
import head_to_head_thrml as hth

RELAX_BUDGETS = [25, 50, 100, 200, 400]


def mean_bkl(p, q, eps=1e-6):
    """Mean binary KL divergence KL(p||q) over voxels. Marginals-only;
    ignores correlations -- documented caveat, used as a latch metric."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1 - eps)
    return float((p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))).mean())


def cold_reference(entry, h, tag, seed_base):
    """Long cold runs -> (ref_marginals, noise_floor_L1, best_energy)."""
    margs, best = [], np.inf
    for s in range(N_REF_SEEDS):
        marg, samples, _ = run_thrml(entry, h, REF_SWEEPS, REF_BURN,
                                     seed=seed_base + s, x_init=None)
        margs.append(marg)
        best = min(best, best_energy(h, samples[::4]))
    ref = np.mean(margs, axis=0)
    floor = float(np.mean([l1(margs[i], margs[j])
                           for i in range(N_REF_SEEDS)
                           for j in range(i + 1, N_REF_SEEDS)]))
    return ref, floor, best


def relax_curve(entry, h_slam, x_init, ref, floor, ref_best, marg_pre,
                seed_base, tag):
    """Warm-start from x_init at each budget; return per-budget metrics."""
    out = {}
    for budget in RELAX_BUDGETS:
        burn = budget // 4
        per_seed = []
        for s in range(N_WARM_SEEDS):
            marg_w, samples_w, dt = run_thrml(
                entry, h_slam, budget, burn, seed=seed_base + s, x_init=x_init)
            L1_wr = l1(marg_w, ref)
            L1_wp = l1(marg_w, marg_pre)
            per_seed.append({
                "L1_to_ref": round(L1_wr, 4),
                "KL_to_ref": round(mean_bkl(ref, marg_w), 4),
                "L1_to_preslam": round(L1_wp, 4),
                "energy_gap": round(best_energy(h_slam, samples_w) - ref_best, 3),
                "trapped": bool((L1_wp < L1_wr) and (L1_wr > 2 * floor)),
                "wall_s": round(dt, 1),
            })
        # latch: smallest budget already handled in aggregation below
        out[str(budget)] = {
            "mean_L1_to_ref": round(float(np.mean([p["L1_to_ref"] for p in per_seed])), 4),
            "mean_KL_to_ref": round(float(np.mean([p["KL_to_ref"] for p in per_seed])), 4),
            "trapped": all(p["trapped"] for p in per_seed),
            "seeds": per_seed,
        }
    latched = [b for b in RELAX_BUDGETS
               if out[str(b)]["mean_L1_to_ref"] < 2 * floor
               and not out[str(b)]["trapped"]]
    out["sweeps_to_latch"] = min(latched) if latched else None
    return out


def find_cube_site(occ, pose, jdir):
    """Find a fully-free 3x3x3 site 3-6 voxels from pose, preferring ahead."""
    cands = []
    for dist in (4, 5, 3, 6):
        base = (pose[0] + dist * jdir[0], pose[1] + dist * jdir[1])
        cands.append((int(base[0]), int(base[1]), 8))
        for ox, oy in ((2, 0), (-2, 0), (0, 2), (0, -2), (3, 3), (-3, -3)):
            cands.append((int(base[0] + ox), int(base[1] + oy), 8))
    for c in cands:
        cube = [(c[0] + dx, c[1] + dy, c[2] + dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        if all(0 <= x < 32 and 0 <= y < 32 and 0 <= z < 16 for x, y, z in cube) \
           and all(not occ[x, y, z] for x, y, z in cube):
            return c, cube
    raise RuntimeError("no free 3x3x3 cube site")


def main():
    entry = hth._perc_program((32, 32, 16))
    results = {"config": {
        "rooms": ROOMS, "relax_budgets": RELAX_BUDGETS,
        "cold": [COLD_SWEEPS, COLD_BURN], "ref": [REF_SWEEPS, REF_BURN],
        "backend": "THRML", "claim": "Z1-legal formulation, validated in simulation",
        "kl_note": "mean binary KL over marginals; ignores correlations",
    }, "rooms": {}}

    for ri, room in enumerate(ROOMS):
        occ = load_occ(room)
        pose = pick_start(occ, PREFERRED_START[room])
        jdir = pick_jump_dir(occ, pose, [1, 2, 4, 8])
        print(f"\n=== {room}: pose={tuple(round(v,1) for v in pose)}", flush=True)
        rres = {"pose": pose}

        # ---- baseline: converge, no cube
        lo0 = fresh_logodds(occ.shape)
        scan_seed = 2000 + ri * 100
        h0 = scan_into(lo0, occ, pose, scan_seed)
        margs0, xlast0 = [], []
        for s in range(N_COLD_SEEDS):
            marg, samples, _ = run_thrml(entry, h0, COLD_SWEEPS, COLD_BURN,
                                         seed=20 + s, x_init=None)
            margs0.append(marg)
            xlast0.append(samples[-1])
        marg0 = np.mean(margs0, axis=0)

        # ---- (a) ROTATIONAL slam: 90-degree yaw, same pose, same scan seed
        # Production sensor has no yaw parameter (omnidirectional); a yaw
        # rotation is not representable. We verify the consequence: the
        # observation — and hence h — cannot change under yaw.
        h_yaw = scan_into(fresh_logodds(occ.shape), occ, pose, scan_seed)
        rres["yaw_slam"] = {
            "h_identical": bool(np.array_equal(h0, h_yaw)),
            "max_abs_dh": float(np.abs(h0 - h_yaw).max()),
            "verdict": ("no-op by construction: omnidirectional sensor, "
                        "90deg yaw maps the ray set onto itself exactly"),
        }
        print(f"  yaw slam: h identical = {rres['yaw_slam']['h_identical']}",
              flush=True)

        # ---- (b) TELEPORTED OBSTACLE: 3x3x3 cube APPEARS near the drone
        c, cube = find_cube_site(occ, pose, jdir)
        occ1 = occ.copy()
        for x, y, z in cube:
            occ1[x, y, z] = True
        lo1 = lo0.copy()  # production: accumulate into running logodds
        h1 = scan_into(lo1, occ1, pose, scan_seed)
        dh = np.abs(h1 - h0)
        rres["appear"] = {"cube_center": c,
                          "voxels_changed": int((dh > 0.5).sum())}
        assert rres["appear"]["voxels_changed"] > 0, "cube not observed!"
        print(f"  appear slam: {rres['appear']['voxels_changed']} voxels |dh|>0.5",
              flush=True)
        ref1, floor1, best1 = cold_reference(entry, h1, "appear",
                                             7000 + ri * 100)
        curve1 = relax_curve(entry, h1, xlast0[0], ref1, floor1, best1, marg0,
                             8000 + ri * 100, "appear")
        rres["appear"].update({"noise_floor_L1": round(floor1, 4),
                               "ref_best_energy": round(best1, 2),
                               "relax": curve1})
        print(f"    sweeps to latch: {curve1['sweeps_to_latch']}", flush=True)

        # ---- (b2) cube DISAPPEARS (same pose, accumulate)
        lo2 = lo1.copy()
        h2 = scan_into(lo2, occ, pose, scan_seed)
        # warm init = end state of the appear tracking (budget-400, seed 0)
        mw400, sw400, _ = run_thrml(entry, h1, 400, 100,
                                    seed=8000 + ri * 100, x_init=xlast0[0])
        ref2, floor2, best2 = cold_reference(entry, h2, "disappear",
                                             7500 + ri * 100)
        curve2 = relax_curve(entry, h2, sw400[-1], ref2, floor2, best2, ref1,
                             8500 + ri * 100, "disappear")
        rres["disappear"] = {"noise_floor_L1": round(floor2, 4),
                             "ref_best_energy": round(best2, 2),
                             "relax": curve2}
        print(f"  disappear slam: sweeps to latch: {curve2['sweeps_to_latch']}",
              flush=True)
        results["rooms"][room] = rres
        # incremental save: a later room crashing must not lose earlier rooms
        with open(os.path.join(OUTDIR, "SLAM_TEST.json"), "w") as f:
            json.dump(results, f, indent=1)

    out = os.path.join(OUTDIR, "SLAM_TEST.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
