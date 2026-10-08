#!/usr/bin/env python3
"""Phase-4 curated driver: obscurant density sweep (numpy) -> archetype check
-> THRML finals -> zero-solution.

Same 8 graded + 2 zero-solution missions as Phase 2b/3 (imported from
curated_v2). Paired seeds/starts/goals/noise; only the obscurant changes.

Vehicle: Phase-3 physics MINUS wind (gravity envelope + attitude lag + 2b
filter on, wind_steady=0, wind_gust=False) — this phase isolates SENSING
degradation; wind force would confound the break-mechanism analysis.

Config naming -> mission_id: p4_<archetype>_<density>_<room>_s<seed>_n<noise>_s16.

Run examples:
  python3 curated_v4.py --phase sweep --backend numpy
  python3 curated_v4.py --phase archetype --backend numpy
  XLA_PYTHON_CLIENT_PREALLOCATE=false python3 curated_v4.py --phase final --backend thrml --densities clear,moderate,heavy
  XLA_PYTHON_CLIENT_PREALLOCATE=false python3 curated_v4.py --phase zero --backend thrml
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

import mission_dynamics_v4 as md
from mission_dynamics_v4 import MANIFEST_FIELDS_V4
from curated_v2 import GRADED, make_zero_enclosed, make_zero_partition
import obscurant

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
HERE = os.path.dirname(os.path.abspath(__file__))


def base_cfg():
    # Phase-3 physics minus wind: isolate sensing.
    return {"gravity_envelope": True, "attitude_lag": True,
            "tw_ratio": 2.0, "filter_tau": 0.4,
            "wind_steady": 0.0, "wind_gust": False}


def density_configs(archetype="fog", densities=None):
    densities = densities or obscurant.DENSITIES
    cfgs = []
    for d in densities:
        c = base_cfg()
        c["obscurant"] = {"archetype": archetype, "density": d}
        c["config_name"] = f"{archetype}_{d}"
        cfgs.append((c["config_name"], c))
    return cfgs


def run_set(missions, cfgs, backend, out_json, manifest_path, label):
    results = []
    manifest_rows = []
    _orig_psg = md.pick_start_goal
    # resume: skip missions already recorded in this phase's results json.
    # (The manifest mixes backends under identical mission_ids, so it is
    # NOT a valid resume source.)
    if os.path.exists(out_json):
        try:
            results = json.load(open(out_json)).get("results", [])
        except Exception:  # noqa: BLE001
            results = []
    done_ids = {r["mission"] for r in results}
    new_file = not os.path.exists(manifest_path)
    total = len(missions) * len(cfgs)
    done_n = 0

    def flush():
        with open(out_json, "w") as f:
            json.dump({"backend": backend, "label": label,
                       "results": results}, f, indent=1)
        with open(manifest_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS_V4)
            for r in manifest_rows:
                w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS_V4})
        manifest_rows.clear()

    if new_file:
        with open(manifest_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS_V4).writeheader()

    for cfg_name, cfg in cfgs:
        for occ, room, seed, noise, mid_base, info, sp, gp in missions:
            mid = f"p4_{cfg_name}_{mid_base}"
            done_n += 1
            if mid in done_ids:
                print(f"[{done_n}/{total}] {mid}: SKIPPED (in manifest)",
                      flush=True)
                continue
            t0 = time.time()
            if sp is not None:
                md.pick_start_goal = (lambda o, s, _sp=sp, _gp=gp: (_sp, _gp))
            try:
                row = md.run_mission_dynamics(
                    occ, room, seed, noise, mid, backend=backend,
                    verbose=False, cfg=dict(cfg))
                outcome = row["outcome"]
                row["tier"] = info.get("tier")
                row["backend"] = backend  # extra col (DictWriter ignores)
                manifest_rows.append(row)
            except Exception as e:  # noqa: BLE001 — record, don't crash
                outcome = f"ERROR:{type(e).__name__}:{e}"
                row = {}
            finally:
                md.pick_start_goal = _orig_psg
            dt = time.time() - t0
            rec = {"mission": mid, "config": cfg_name,
                   "tier": info.get("tier"), "outcome": outcome,
                   "n_steps": row.get("n_steps", -1),
                   "n_d5_refusals": row.get("n_d5_refusals_s16", -1),
                   "wall_s": round(dt, 1)}
            results.append(rec)
            print(f"[{done_n}/{total}] {mid}: {outcome} "
                  f"steps={rec['n_steps']} d5ref={rec['n_d5_refusals']} "
                  f"({dt:.0f}s)", flush=True)
            flush()
            done_ids.add(mid)
    n_ok = sum(1 for r in results if r["outcome"] == "success")
    n_coll = sum(1 for r in results if r["outcome"] == "collision")
    print(f"=== {label}: {n_ok}/{len(results)} success, {n_coll} collisions "
          f"[{backend}] ===")
    flush()
    print(f"wrote {out_json} + {manifest_path}")
    return results


def load_missions():
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
        print(f"constructed p4_..._{mid}: {info}", flush=True)
        zero_missions.append((occ, room, seed, noise, mid, info, sp, gp))
    return missions, zero_missions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["sweep", "archetype", "final", "zero"])
    ap.add_argument("--backend", default="numpy", choices=["thrml", "numpy"])
    ap.add_argument("--densities", default=None,
                    help="comma-separated subset for final phase")
    args = ap.parse_args()

    missions, zero_missions = load_missions()
    mpath = os.path.join(HERE, "RUN_MANIFEST_DYN_V4.csv")

    if args.phase == "sweep":
        cfgs = density_configs("fog")
        out = os.path.join(HERE, f"curated_v4_sweep_{args.backend}.json")
        run_set(missions, cfgs, args.backend, out, mpath, "p4 fog sweep")
    elif args.phase == "archetype":
        cfgs = density_configs("rain", ["moderate", "heavy"])
        out = os.path.join(HERE, f"curated_v4_archetype_{args.backend}.json")
        run_set(missions, cfgs, args.backend, out, mpath, "p4 rain check")
    elif args.phase == "final":
        dens = (args.densities.split(",") if args.densities
                else ["clear", "moderate", "heavy"])
        cfgs = density_configs("fog", dens)
        out = os.path.join(HERE, f"curated_v4_final_{args.backend}.json")
        run_set(missions, cfgs, args.backend, out, mpath, "p4 FINAL")
    elif args.phase == "zero":
        cfgs = density_configs("fog", ["clear", "heavy"])
        out = os.path.join(HERE, f"curated_v4_zero_{args.backend}.json")
        run_set(zero_missions, cfgs, args.backend, out, mpath, "p4 zero-sol")


if __name__ == "__main__":
    main()
