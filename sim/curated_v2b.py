"""Curated Phase-2b test set: same 10 missions as Phase 2 (8 graded + 2 zero-solution).

Phase 2b changes ONLY the turning: slew-rate-limited commanded direction +
continuous slow-for-corners. Same rooms/seeds/noise, same THRML perception,
S2/A*/D5/CBF. Paired with Phase 2 for the turn-smoothness comparison.

Zero-solution constructors are imported from curated_v2 (identical
construction + flood-fill verification); only the mission loop differs.

Run: python3 curated_v2b.py [--backend thrml|numpy]
Writes: curated_v2b_results.json, traj2b/p2b_<mission_id>.json (demo contract).
"""
import os
import sys
import time
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission_dynamics_v2b as md
from curated_v2 import GRADED, make_zero_enclosed, make_zero_partition

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "curated_v2b_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="thrml", choices=["thrml", "numpy"])
    ap.add_argument("--only-zero", action="store_true",
                    help="re-run only the zero-solution missions")
    args = ap.parse_args()

    import planner  # noqa: F401  (ensures pipeline importable)

    missions = []
    for room, seed, noise in GRADED:
        occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
        mid = f"p2b_{room}_s{seed}_n{noise:.2f}_s16"
        missions.append((occ, room, seed, noise, mid, {"tier": "graded"},
                         None, None))

    for maker in (make_zero_enclosed, make_zero_partition):
        occ, room, seed, noise, mid, info, sp, gp = maker()
        info["tier"] = "zero-solution"
        mid = "p2b_" + mid
        print(f"constructed {mid}: {info}", flush=True)
        missions.append((occ, room, seed, noise, mid, info, sp, gp))

    if args.only_zero:
        missions = [m for m in missions if m[5].get("tier") == "zero-solution"]

    results = []
    manifest_rows = []
    _orig_psg = md.pick_start_goal
    for occ, room, seed, noise, mid, info, sp, gp in missions:
        t0 = time.time()
        if sp is not None:
            md.pick_start_goal = (lambda o, s, _sp=sp, _gp=gp:
                                  (_sp, _gp))
        try:
            row = md.run_mission_dynamics(
                occ, room, seed, noise, mid, backend=args.backend,
                verbose=False)
            outcome = row["outcome"]
            row["tier"] = info.get("tier")
            manifest_rows.append(row)
        except Exception as e:  # noqa: BLE001 — record, don't crash the set
            outcome = f"ERROR:{type(e).__name__}:{e}"
            row = {}
        finally:
            md.pick_start_goal = _orig_psg
        dt = time.time() - t0
        rec = {"mission": mid, "tier": info.get("tier"),
               "outcome": outcome,
               "n_steps": row.get("n_steps", -1),
               "wall_s": round(dt, 1), "info": info}
        results.append(rec)
        print(f"{mid} [{info.get('tier')}]: {outcome} "
              f"steps={rec['n_steps']} ({dt:.0f}s)", flush=True)

    n_ok = sum(1 for r in results if r["outcome"] == "success")
    n_coll = sum(1 for r in results if r["outcome"] == "collision")
    print(f"=== {n_ok}/{len(results)} success, {n_coll} collisions "
          f"[{args.backend}] ===")
    with open(OUT, "w") as f:
        json.dump({"backend": args.backend, "results": results}, f, indent=1)
    print(f"wrote {OUT}")
    # manifest CSV (same schema as Phase 2's RUN_MANIFEST_DYN_V2.csv)
    import csv
    from head_to_head_v2 import MANIFEST_FIELDS_V2
    mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "RUN_MANIFEST_DYN_V2B.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS_V2) + ["tier"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r.get(k, "") for k in list(MANIFEST_FIELDS_V2) + ["tier"]})
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
