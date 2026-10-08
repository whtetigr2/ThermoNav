"""Phase-3 analysis: paired metrics vs Phase 2b baseline.

Reads curated_v3_*.json + RUN_MANIFEST_DYN_V3.csv + traj3/*.json,
computes turn_metrics on every trajectory, and builds paired comparison
tables against the Phase-2b baseline (traj2b/p2b_*.json).

Usage: python3 analyze_v3.py  (run after each phase; prints tables)
Writes: analysis_v3.json
"""
import os
import sys
import json
import csv
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turn_metrics import turn_metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def load_traj_metrics(traj_dir, prefix):
    out = {}
    for fn in sorted(os.listdir(traj_dir)):
        if not fn.startswith(prefix) or not fn.endswith(".json"):
            continue
        with open(os.path.join(traj_dir, fn)) as f:
            doc = json.load(f)
        xyz = doc["xyz"]
        m = turn_metrics(xyz)
        m["outcome"] = doc.get("outcome")
        m["mission_id"] = doc["mission_id"]
        # control effort proxy: mean |accel| from 2nd differences
        a = np.diff(np.diff(np.asarray(xyz, dtype=float), axis=0),
                    axis=0) / (0.05 ** 2)
        m["mean_a_mag"] = round(float(np.linalg.norm(a, axis=1).mean()), 3)
        m["path_len"] = round(float(
            np.linalg.norm(np.diff(np.asarray(xyz, dtype=float), axis=0),
                           axis=1).sum()), 2)
        m["duration_s"] = round(len(xyz) * 0.05, 1)
        out[doc["mission_id"]] = m
    return out


def base_of(mid):
    """p3_<cfg>_<room>_s<seed>_n<noise>_s16 -> <room>_s<seed>_n<noise>_s16 ;
       p2b_<...> -> same base."""
    for pfx in ("p3_", "p2b_"):
        if mid.startswith(pfx):
            mid = mid[len(pfx):]
            break
    # strip the config token for p3 (first token before room id)
    parts = mid.split("_")
    # room ids look like indoorcd_071 or zero_enclosed_indoorcd_004
    if parts[0] in ("a", "b", "c", "d", "b2") or parts[0].startswith("w"):
        # config token(s): a_wind | b_tilt | c_grav | d_all | d_nofilt |
        # d_tw167 | w0 | w2g ...
        if parts[0] == "d" and parts[1] in ("all", "nofilt", "tw167"):
            mid = "_".join(parts[2:])
        elif parts[0] in ("a", "b", "c", "b2"):
            mid = "_".join(parts[2:])
        else:
            mid = "_".join(parts[1:])
    return mid


def main():
    m2b = load_traj_metrics(os.path.join(HERE, "traj2b"), "p2b_")
    m3 = load_traj_metrics(os.path.join(HERE, "traj3"), "p3_")
    print(f"2b trajs: {len(m2b)}, v3 trajs: {len(m3)}")

    # group v3 by config
    by_cfg = defaultdict(list)
    for mid in m3:
        # config = token(s) between p3_ and room
        rest = mid[len("p3_"):]
        parts = rest.split("_")
        if parts[0] == "d" and parts[1] in ("all", "nofilt", "tw167"):
            cfg = parts[0] + "_" + parts[1]
        elif parts[0] in ("a", "b", "c"):
            cfg = parts[0] + "_" + parts[1]
        elif parts[0] == "b2":
            cfg = "b2_phys"
        else:
            cfg = parts[0]
        by_cfg[cfg].append(mid)

    # paired table vs 2b for each config (graded only: skip zero_*)
    keys = ["yaw_flip_rate_hz", "lat_acc_zc_hz", "max_yaw_rate",
            "lat_acc_std", "mean_speed", "slow_frac",
            "mean_a_mag", "path_len", "duration_s"]
    report = {}
    for cfg, mids in sorted(by_cfg.items()):
        pairs = []
        for mid in mids:
            b = base_of(mid)
            b2b_mid = "p2b_" + b
            if b2b_mid in m2b and not b.startswith("zero"):
                pairs.append((m2b[b2b_mid], m3[mid]))
        if not pairs:
            continue
        agg = {"n_paired": len(pairs), "metrics": {}}
        for k in keys:
            v2 = np.mean([p[0][k] for p in pairs])
            v3 = np.mean([p[1][k] for p in pairs])
            d = (v3 - v2) / v2 * 100 if v2 else 0.0
            agg["metrics"][k] = {"p2b": round(v2, 3), "p3": round(v3, 3),
                                 "delta_pct": round(d, 1)}
        # outcome counts from the v3 side
        outs = defaultdict(int)
        for _, m in pairs:
            outs[m["outcome"]] += 1
        agg["outcomes"] = dict(outs)
        report[cfg] = agg

    with open(os.path.join(HERE, "analysis_v3.json"), "w") as f:
        json.dump(report, f, indent=1)

    for cfg, agg in sorted(report.items()):
        print(f"\n=== {cfg} (n={agg['n_paired']}) outcomes={agg['outcomes']} ===")
        for k, v in agg["metrics"].items():
            print(f"  {k:18s} 2b={v['p2b']:<8} v3={v['p3']:<8} "
                  f"{v['delta_pct']:+.1f}%")


if __name__ == "__main__":
    main()
