"""Phase-3 curated driver: wind sweep (numpy) -> ablations (numpy) -> finals (THRML).

Same 10 curated missions as Phase 2b (8 GRADED + 2 zero-solution, imported
from curated_v2). Paired seeds/starts/goals/noise; only the physics config
changes between runs.

Phases:
  sweep    : wind 0/2/4 m/s x gust off/on, FULL Phase-3 physics
             (gravity envelope T/W=2.0 + attitude lag + 2b filter),
             8 graded, numpy. Finds the informative wind level.
  ablation : at the chosen wind W (default 4): a_wind / b_tilt / c_grav /
             d_all / d_nofilt (+ tw167 sensitivity on d_all), 8 graded.
  final    : THRML: a, b, c, d, d_nofilt at chosen wind, 8 graded +
             2 zero-solution under d_all.

Config naming -> mission_id: p3_<cfg>_<room>_s<seed>_n<noise>_s16.

Run examples:
  python3 curated_v3.py --phase sweep --backend numpy
  python3 curated_v3.py --phase ablation --backend numpy --wind 4
  XLA_PYTHON_CLIENT_PREALLOCATE=false python3 curated_v3.py --phase final --backend thrml --wind 4
"""
import os
import sys
import time
import json
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission_dynamics_v3 as md
from mission_dynamics_v3 import MANIFEST_FIELDS_V3
from curated_v2 import GRADED, make_zero_enclosed, make_zero_partition

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
HERE = os.path.dirname(os.path.abspath(__file__))


def base_cfg():
    return {"gravity_envelope": True, "attitude_lag": True,
            "tw_ratio": 2.0, "filter_tau": 0.4,
            "wind_steady": 0.0, "wind_gust": False}


def sweep_configs():
    cfgs = []
    for w in (0.0, 2.0, 4.0):
        for g in (False, True):
            c = base_cfg()
            c["wind_steady"] = w
            c["wind_gust"] = g
            name = f"w{int(w)}" + ("g" if g else "")
            c["config_name"] = name
            cfgs.append((name, c))
    return cfgs


def ablation_configs(wind):
    cfgs = []
    # (a) wind only: old physics + wind
    c = {"gravity_envelope": False, "attitude_lag": False,
         "tw_ratio": 2.0, "filter_tau": 0.4,
         "wind_steady": wind, "wind_gust": True, "config_name": "a_wind"}
    cfgs.append(("a_wind", c))
    # (b) attitude lag only (no gravity envelope, no wind)
    c = {"gravity_envelope": False, "attitude_lag": True,
         "tw_ratio": 2.0, "filter_tau": 0.4,
         "wind_steady": 0.0, "wind_gust": False, "config_name": "b_tilt"}
    cfgs.append(("b_tilt", c))
    # (c) gravity/envelope only (no lag, no wind)
    c = {"gravity_envelope": True, "attitude_lag": False,
         "tw_ratio": 2.0, "filter_tau": 0.4,
         "wind_steady": 0.0, "wind_gust": False, "config_name": "c_grav"}
    cfgs.append(("c_grav", c))
    # (b2) PHYSICAL attitude-lag isolation: gravity envelope ON (thrust
    # direction is the physical lagged quantity) but non-binding
    # (T/W=10, tilt 89 deg), no wind. Replaces the naive (b), which proved
    # unsafe for an implementation reason (180-degree direction flips lerp
    # through perpendicular -> sideways braking transients).
    c = {"gravity_envelope": True, "attitude_lag": True,
         "tw_ratio": 10.0, "tilt_max_deg": 89.0, "filter_tau": 0.4,
         "wind_steady": 0.0, "wind_gust": False, "config_name": "b2_phys"}
    cfgs.append(("b2_phys", c))
    # (d) all three
    c = base_cfg()
    c["wind_steady"] = wind
    c["wind_gust"] = True
    c["config_name"] = "d_all"
    cfgs.append(("d_all", c))
    # (d) all three, filter bypassed -> tilt-lag-only test
    c = base_cfg()
    c["wind_steady"] = wind
    c["wind_gust"] = True
    c["filter_tau"] = None
    c["config_name"] = "d_nofilt"
    cfgs.append(("d_nofilt", c))
    # T/W sensitivity on d_all
    c = base_cfg()
    c["wind_steady"] = wind
    c["wind_gust"] = True
    c["tw_ratio"] = 1.67
    c["config_name"] = "d_tw167"
    cfgs.append(("d_tw167", c))
    return cfgs


def run_set(missions, cfgs, backend, out_json, manifest_path, label):
    results = []
    manifest_rows = []
    _orig_psg = md.pick_start_goal
    total = len(missions) * len(cfgs)
    done_n = 0
    for cfg_name, cfg in cfgs:
        for occ, room, seed, noise, mid_base, info, sp, gp in missions:
            mid = f"p3_{cfg_name}_{mid_base}"
            t0 = time.time()
            if sp is not None:
                md.pick_start_goal = (lambda o, s, _sp=sp, _gp=gp: (_sp, _gp))
            try:
                row = md.run_mission_dynamics(
                    occ, room, seed, noise, mid, backend=backend,
                    verbose=False, cfg=dict(cfg))
                outcome = row["outcome"]
                row["tier"] = info.get("tier")
                manifest_rows.append(row)
            except Exception as e:  # noqa: BLE001 — record, don't crash
                outcome = f"ERROR:{type(e).__name__}:{e}"
                row = {}
            finally:
                md.pick_start_goal = _orig_psg
            dt = time.time() - t0
            done_n += 1
            rec = {"mission": mid, "config": cfg_name,
                   "tier": info.get("tier"), "outcome": outcome,
                   "n_steps": row.get("n_steps", -1),
                   "wall_s": round(dt, 1)}
            results.append(rec)
            print(f"[{done_n}/{total}] {mid}: {outcome} "
                  f"steps={rec['n_steps']} ({dt:.0f}s)", flush=True)
    n_ok = sum(1 for r in results if r["outcome"] == "success")
    n_coll = sum(1 for r in results if r["outcome"] == "collision")
    print(f"=== {label}: {n_ok}/{len(results)} success, {n_coll} collisions "
          f"[{backend}] ===")
    with open(out_json, "w") as f:
        json.dump({"backend": backend, "label": label,
                   "results": results}, f, indent=1)
    # manifest (append; v3 schema)
    new_file = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS_V3)
        if new_file:
            w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS_V3})
    print(f"wrote {out_json} + {manifest_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["sweep", "ablation", "final"])
    ap.add_argument("--backend", default="numpy",
                    choices=["thrml", "numpy"])
    ap.add_argument("--wind", type=float, default=4.0,
                    help="steady wind for ablation/final phases")
    ap.add_argument("--only-zero", action="store_true")
    args = ap.parse_args()

    import planner  # noqa: F401

    missions = []
    for room, seed, noise in GRADED:
        occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
        mid = f"{room}_s{seed}_n{noise:.2f}_s16"
        missions.append((occ, room, seed, noise, mid, {"tier": "graded"},
                         None, None))
    zero_missions = []
    for maker in (make_zero_enclosed, make_zero_partition):
        occ, room, seed, noise, mid, info, sp, gp = maker()
        info["tier"] = "zero-solution"
        print(f"constructed p3_..._{mid}: {info}", flush=True)
        zero_missions.append((occ, room, seed, noise, mid, info, sp, gp))

    mpath = os.path.join(HERE, "RUN_MANIFEST_DYN_V3.csv")
    if args.phase == "sweep":
        cfgs = sweep_configs()
        ms = [m for m in missions
              if not args.only_zero]  # graded only for the sweep
        out = os.path.join(HERE, f"curated_v3_sweep_{args.backend}.json")
        run_set(ms, cfgs, args.backend, out, mpath, "p3 wind sweep")
    elif args.phase == "ablation":
        cfgs = ablation_configs(args.wind)
        out = os.path.join(HERE, f"curated_v3_ablation_{args.backend}.json")
        run_set(missions, cfgs, args.backend, out, mpath,
                f"p3 ablation @wind={args.wind}")
    elif args.phase == "final":
        cfgs = [c for c in ablation_configs(args.wind)
                if c[0] in ("a_wind", "b2_phys", "c_grav", "d_all", "d_nofilt")]
        out = os.path.join(HERE, f"curated_v3_final_{args.backend}.json")
        run_set(missions, cfgs, args.backend, out, mpath,
                f"p3 FINAL @wind={args.wind}")
        # zero-solution under d_all
        d_all = [c for c in ablation_configs(args.wind)
                 if c[0] == "d_all"]
        out0 = os.path.join(HERE, f"curated_v3_final_zero_{args.backend}.json")
        run_set(zero_missions, d_all, args.backend, out0, mpath,
                f"p3 FINAL zero-solution @wind={args.wind}")


if __name__ == "__main__":
    main()
