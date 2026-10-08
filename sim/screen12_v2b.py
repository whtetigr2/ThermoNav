"""12-mission dynamics screen — Phase 2b (smooth turning).

Same 11 round-3 regression + tall wall as Phase 1 screen12.py.
Hard gate: 0 collisions before any full run.

Run: python3 screen12_v2.py [--backend thrml|numpy]
Writes screen12_v2b_results.json. Trajectories -> traj2b/ (p2b_ prefix).
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

MISSIONS = [
    ("indoorcd_128", 0, 0.15), ("indoorcd_128", 0, 0.30),
    ("indoorcd_128", 1, 0.15), ("indoorcd_128", 1, 0.30),
    ("indoorcd_128", 2, 0.15), ("indoorcd_128", 2, 0.30),
    ("indoorcd_128", 3, 0.15), ("indoorcd_128", 3, 0.30),
    ("indoorcd_128", 5, 0.15), ("indoorcd_128", 5, 0.30),
    ("indoorcd_071", 1, 0.30),
]
REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "screen12_v2b_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="thrml", choices=["thrml", "numpy"])
    args = ap.parse_args()

    results = []
    # 11 real missions
    for room, seed, noise in MISSIONS:
        occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
        mid = f"p2b_{room}_s{seed}_n{noise:.2f}_s16"
        t0 = time.time()
        try:
            row = md.run_mission_dynamics(
                occ, room, seed, noise, mid, backend=args.backend,
                verbose=False)
            outcome, steps = row["outcome"], row["n_steps"]
        except Exception as e:
            outcome, steps = f"ERROR:{type(e).__name__}:{e}", -1
        dt = time.time() - t0
        results.append({"mission": mid, "outcome": outcome, "n_steps": steps,
                        "wall_s": round(dt, 1)})
        print(f"{mid}: {outcome} steps={steps} ({dt:.0f}s)", flush=True)

    # tall wall (synthetic)
    occ = np.zeros((32, 32, 16), dtype=np.uint8)
    occ[16, :, 0:13] = 1
    md.pick_start_goal = lambda o, s: ((4.5, 16.5, 4.5), (28.5, 16.5, 4.5))
    mid = "p2b_synth_wall2_s1_n0.30_s16"
    t0 = time.time()
    try:
        row = md.run_mission_dynamics(occ, "synth_wall2", 1, 0.30, mid,
                                      backend=args.backend, verbose=False)
        outcome, steps = row["outcome"], row["n_steps"]
    except Exception as e:
        outcome, steps = f"ERROR:{type(e).__name__}:{e}", -1
    dt = time.time() - t0
    results.append({"mission": mid, "outcome": outcome, "n_steps": steps,
                    "wall_s": round(dt, 1)})
    print(f"{mid}: {outcome} steps={steps} ({dt:.0f}s)", flush=True)

    n_ok = sum(1 for r in results if r["outcome"] == "success")
    print(f"=== {n_ok}/12 success [{args.backend}] ===")
    with open(OUT, "w") as f:
        json.dump({"backend": args.backend, "results": results}, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
