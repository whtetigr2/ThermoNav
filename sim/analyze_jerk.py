"""Jerk metrics: Phase 1 (traj/) vs Phase 2 (traj2/).

Per mission (20 Hz trajectories, dt=0.05):
  - mean_speed, max_speed (m/s)
  - frac_slow: fraction of time with speed < 0.2 m/s
  - sign_changes: velocity-component sign flips (direction wobbles)
  - mean_abs_accel (m/s^2)
Plus mission-level: outcome, sim duration, path length.

Writes jerk_comparison.json + prints aggregate table.
All numbers [measured in simulation] from the recorded trajectories;
deltas are [calculated].
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "traj")
P2 = os.path.join(HERE, "traj2")
OUT = os.path.join(HERE, "jerk_comparison.json")
DT = 0.05


def metrics(path):
    d = json.load(open(path))
    p = np.asarray(d["xyz"], dtype=float)
    out = {"mission_id": d["mission_id"], "outcome": d["outcome"],
           "n_pts": len(p)}
    if len(p) < 3:
        out.update(mean_speed=0, max_speed=0, frac_slow=1.0,
                   sign_changes=0, mean_abs_accel=0.0, duration_s=0.0,
                   path_len_m=0.0)
        return out
    v = np.diff(p, axis=0) / DT
    sp = np.linalg.norm(v, axis=1)
    a = np.diff(v, axis=0) / DT
    dirs = np.sign(v)
    # ignore zero-velocity flicker: only count a flip when both sides nonzero
    flips = 0
    for k in range(3):
        s = dirs[:, k]
        nz = s[s != 0]
        flips += int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    out.update(
        mean_speed=round(float(sp.mean()), 3),
        max_speed=round(float(sp.max()), 3),
        frac_slow=round(float((sp < 0.2).mean()), 3),
        sign_changes=int(flips),
        mean_abs_accel=round(float(np.linalg.norm(a, axis=1).mean()), 3),
        duration_s=round(len(p) * DT, 1),
        path_len_m=round(float(np.linalg.norm(np.diff(p, axis=0),
                                              axis=1).sum()), 2),
    )
    return out


def main():
    recs = []
    p1 = {f[:-5]: os.path.join(P1, f) for f in os.listdir(P1)
          if f.endswith(".json")}
    p2 = {f[:-5]: os.path.join(P2, f) for f in os.listdir(P2)
          if f.endswith(".json")}
    # pair by canonical mission id (strip dyn_/dyn2_ screen prefixes)
    def canon(mid):
        for pre in ("dyn2_", "dyn_"):
            if mid.startswith(pre):
                return mid[len(pre):]
        return mid
    c1 = {canon(k): v for k, v in p1.items()}
    c2 = {canon(k): v for k, v in p2.items()}
    common = sorted(set(c1) & set(c2))
    print(f"phase1 trajs: {len(p1)}, phase2 trajs: {len(p2)}, "
          f"paired: {len(common)}")
    for mid in common:
        m1, m2 = metrics(c1[mid]), metrics(c2[mid])
        rec = {"mission_id": mid, "phase1": m1, "phase2": m2}
        for k in ("mean_speed", "max_speed", "frac_slow", "sign_changes",
                  "mean_abs_accel", "duration_s", "path_len_m"):
            rec[f"delta_{k}"] = round(m2[k] - m1[k], 3)
        recs.append(rec)

    # aggregate over paired missions with success in both phases
    ok = [r for r in recs if r["phase1"]["outcome"] == "success"
          and r["phase2"]["outcome"] == "success"]
    print(f"paired successes: {len(ok)}")
    with open(OUT, "w") as f:
        json.dump({"records": recs}, f, indent=1)

    def agg(key, phase):
        xs = [r[phase][key] for r in ok]
        return round(float(np.mean(xs)), 3) if xs else None

    print("\n=== aggregate (paired successes) [measured in simulation] ===")
    for k in ("mean_speed", "max_speed", "frac_slow", "sign_changes",
              "mean_abs_accel", "duration_s", "path_len_m"):
        a1, a2 = agg(k, "phase1"), agg(k, "phase2")
        d = round(a2 - a1, 3) if a1 is not None else None
        pct = round(100 * d / a1, 1) if a1 else None
        print(f"  {k:16s} phase1={a1}  phase2={a2}  delta={d} ({pct}%)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
