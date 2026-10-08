"""Tune Phase-2b turning params on numpy (fast): sweep TAU_CMD x
TURN_SHARPNESS on 2 turn-heavy missions, pick by (0 collisions, min
yaw-flip rate, then max mean speed).

Run: python3 tune_turn.py
Writes: tune_turn_results.json
"""
import os
import sys
import time
import json
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission_dynamics_v2b as md
from turn_metrics import turn_metrics

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "tune_turn_results.json")

# turn-heavy missions: worst V6 path ratio + a damn-near-impossible one
MISSIONS = [("indoorcd_128", 0, 0.30), ("indoorcd_071", 1, 0.15)]
GRID = list(itertools.product([0.15, 0.25, 0.4], [3.0, 4.0]))


def main():
    results = []
    for tau, sharp in GRID:
        md.TAU_CMD = tau
        md.TURN_SHARPNESS = sharp
        combo = {"tau": tau, "sharpness": sharp, "missions": []}
        for room, seed, noise in MISSIONS:
            occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
            mid = f"tune_{room}_s{seed}_n{noise:.2f}_t{tau}_p{sharp}"
            t0 = time.time()
            try:
                row = md.run_mission_dynamics(
                    occ, room, seed, noise, mid, backend="numpy",
                    verbose=False)
                outcome = row["outcome"]
            except Exception as e:  # noqa: BLE001
                outcome = f"ERROR:{type(e).__name__}:{e}"
            dt = time.time() - t0
            m = {"mission": mid, "outcome": outcome, "wall_s": round(dt, 1)}
            if outcome in ("success", "max_steps"):
                tj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "traj2b", mid + ".json")
                if os.path.exists(tj):
                    d = json.load(open(tj))
                    m.update(turn_metrics(d["xyz"], d["dt"]))
            combo["missions"].append(m)
            print(f"t={tau} p={sharp} {room}_s{seed}: {outcome} "
                  f"flip={m.get('yaw_flip_rate_hz', '-')} "
                  f"lat={m.get('lat_acc_std', '-')} ({dt:.0f}s)", flush=True)
        n_coll = sum(1 for m in combo["missions"]
                     if m["outcome"] == "collision")
        succ = [m for m in combo["missions"] if "yaw_flip_rate_hz" in m]
        combo["n_collisions"] = n_coll
        combo["mean_flip_hz"] = round(
            float(np.mean([m["yaw_flip_rate_hz"] for m in succ])), 4) if succ else -1
        combo["mean_lat_std"] = round(
            float(np.mean([m["lat_acc_std"] for m in succ])), 4) if succ else -1
        combo["mean_speed"] = round(
            float(np.mean([m["mean_speed"] for m in succ])), 4) if succ else -1
        results.append(combo)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {OUT}")
    print("\n=== summary (numpy, 2 missions) ===")
    for c in sorted(results, key=lambda c: (c["n_collisions"], c["mean_flip_hz"])):
        print(f"t={c['tau']} p={c['sharpness']}: coll={c['n_collisions']} "
              f"flip={c['mean_flip_hz']}Hz lat_std={c['mean_lat_std']} "
              f"speed={c['mean_speed']}m/s")


if __name__ == "__main__":
    main()
