"""Phase 2 vs Phase 2b turn-smoothness analysis (paired curated missions).

Reads Phase-2 trajectories from traj2/ and Phase-2b from traj2b/,
computes turn metrics + standard metrics on both, writes
turn_comparison.json and prints the comparison table.

Run: python3 analyze_turn.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turn_metrics import turn_metrics

HERE = os.path.dirname(os.path.abspath(__file__))

# (phase2 traj id, phase2b traj id, tier)
PAIRS = [
    ("indoorcd_071_s2_n0.15_s16", "p2b_indoorcd_071_s2_n0.15_s16", "easy"),
    ("indoorcd_004_s0_n0.15_s16", "p2b_indoorcd_004_s0_n0.15_s16", "easy"),
    ("indoorcd_131_s1_n0.15_s16", "p2b_indoorcd_131_s1_n0.15_s16", "medium"),
    ("indoorcd_185_s2_n0.15_s16", "p2b_indoorcd_185_s2_n0.15_s16", "medium"),
    ("indoorcd_128_s0_n0.30_s16", "p2b_indoorcd_128_s0_n0.30_s16", "hard"),
    ("indoorcd_071_s1_n0.30_s16", "p2b_indoorcd_071_s1_n0.30_s16", "hard"),
    ("indoorcd_128_s3_n0.30_s16", "p2b_indoorcd_128_s3_n0.30_s16", "damn"),
    ("indoorcd_071_s1_n0.15_s16", "p2b_indoorcd_071_s1_n0.15_s16", "damn"),
]


def load_metrics(traj_dir, mid):
    p = os.path.join(HERE, traj_dir, mid + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    m = turn_metrics(d["xyz"], d["dt"])
    m["outcome"] = d["outcome"]
    xyz = np.asarray(d["xyz"], dtype=float)
    m["path_len"] = round(float(np.linalg.norm(np.diff(xyz, axis=0),
                                               axis=1).sum()), 2)
    m["duration_s"] = round(len(xyz) * d["dt"], 1)
    return m


def main():
    rows = []
    for mid2, mid2b, tier in PAIRS:
        a = load_metrics("traj2", mid2)
        b = load_metrics("traj2b", mid2b)
        rows.append({"mission": mid2, "tier": tier,
                     "phase2": a, "phase2b": b})
        if a and b:
            print(f"{mid2} [{tier}] P2:{a['outcome']} P2b:{b['outcome']} | "
                  f"flip {a['yaw_flip_rate_hz']}->{b['yaw_flip_rate_hz']}Hz | "
                  f"lat_std {a['lat_acc_std']}->{b['lat_acc_std']} | "
                  f"maxyaw {a['max_yaw_rate']}->{b['max_yaw_rate']} | "
                  f"speed {a['mean_speed']}->{b['mean_speed']}m/s")

    def agg(key, phase):
        vals = [r[phase][key] for r in rows
                if r[phase] and r[phase]["outcome"] == "success"
                and key in r[phase]]
        return round(float(np.mean(vals)), 4) if vals else None

    print("\n=== aggregate (paired successes) ===")
    for key in ["yaw_flip_rate_hz", "lat_acc_std", "lat_acc_zc_hz",
                "max_yaw_rate", "mean_speed", "slow_frac", "duration_s",
                "path_len"]:
        a, b = agg(key, "phase2"), agg(key, "phase2b")
        if a is not None and b is not None:
            d = (b - a) / a * 100 if a else 0
            print(f"{key}: {a} -> {b} ({d:+.1f}%)")

    with open(os.path.join(HERE, "turn_comparison.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote turn_comparison.json")


if __name__ == "__main__":
    main()
