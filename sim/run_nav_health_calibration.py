"""Nav-health OFFLINE calibration (roadmap Phase 1 item 3; round-1 fixes).

Question: does the ONLINE signal (nav_health.py — estimator self-reports
only) separate healthy estimator segments from unhealthy ones?

Channels (red-team round 1, F3): cmd_speed_mps is the controller
SETPOINT (what the host commanded: a constant 1.0 m/s traverse, weave
setpoint includes the lateral component). It is NEVER the achieved or
true motion. Between setpoint and report sits an explicit achieved-
motion model [design]: achieved = 0.9 x setpoint + N(0, 0.02) per step
(first-order lag). The estimator report is then generated from ACHIEVED
motion per FAST-LIO2's measured behavior:
  healthy : est = achieved + N(0, 0.03) (eval range noise class)
  freeze  : est = ratio x achieved, ratio swept OFF the derivation
            point (0.05 / 0.125 / 0.18 / 0.26) [F4]
  partial : est = ratio x achieved, ratio in {0.30, 0.40, 0.55} [F4/F2]
  weave   : est = achieved + N(0, 0.30); residual ~0.30 m class
  residmid: healthy est; residual ~0.20 m (between the x3 and x10
            thresholds — tests graded response, F4)
  stale   : healthy est; map_age 2.0 s constant (staleness channel, F4)
Feature counts come from REAL obscurant scans of the two rooms.
Offline truth (NEVER fed to the monitor): per-step report error
|est - achieved|; a step is "truly unhealthy" if its cumulative 5-step
report error exceeds 0.5 m (the estimator bar).

Provenance: exec_provenance.begin/finish — the run ABORTS on a dirty
tree, and the commit column is verified at write time (F1 system fix).
Dual-purpose CSV: every lever explicit + the executed commit.
Claim: host-side classical, validated in simulation.

ThermoNav port (graduation 2026-10-11): ported from the research
repo's runner (slow-drone/dynamics_sim/run_nav_health_calibration.py).
Difference from the research run: occupancy grids load from
../rooms (ThermoNav layout) instead of real_lidar/.
"""
import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exec_provenance  # noqa: E402
import obscurant  # noqa: E402
import nav_health as nh  # noqa: E402
from fly import pick_start_goal  # noqa: E402

ROOMS = ["indoorcd_117", "indoorcd_122"]
STEPS = 24
OUT = os.path.join(HERE, "nav_health_calibration.csv")
CONDITIONS = ["clear_healthy", "heavy_healthy", "whiteout_freeze",
              "freeze_r0.05", "freeze_r0.18", "freeze_r0.26",
              "partial_r0.30", "partial_r0.40", "partial_r0.55",
              "clear_weave", "resid_mid", "stale_map"]
FIELDS = ["condition", "room", "seed", "step", "n_features",
          "cmd_speed_mps", "achieved_speed_mps", "est_speed_mps",
          "residual_m", "map_age_s",
          "health_state", "pose_trustworthy", "step_report_err_m",
          "cum5_report_err_m", "truth_unhealthy", "git_commit",
          "provenance_note"]
PROV_NOTE = ("round-1 corrected re-issue: executed-tree commit verified "
             "by exec_provenance (original issue named launch-time HEAD)")


def run_condition(cond, room, seed, commit):
    occ = np.load(os.path.join(HERE, "..", "rooms", f"{room}_occ.npy"))
    start, goal = pick_start_goal(occ, seed)
    _d = np.array(goal, dtype=float) - np.array(start, dtype=float)
    _d[2] = 0.0
    heading = _d / np.linalg.norm(_d)
    perp = np.array([-heading[1], heading[0], 0.0])
    density = {"clear_healthy": "clear", "heavy_healthy": "heavy",
               "whiteout_freeze": "whiteout"}.get(cond, "clear")
    obscurant.configure("fog", density)
    rng = np.random.default_rng(5000 + seed)
    mon = nh.NavHealthMonitor()
    rows, errs = [], []
    pos = np.array(start, dtype=float)
    for k in range(STEPS):
        # SETPOINT for this step (host-side command, known exactly).
        step_vec = heading * 1.0
        if cond == "clear_weave":
            step_vec = heading * 1.0 + perp * (0.6 if k % 2 == 0 else -0.6)
        cmd_speed = float(np.linalg.norm(step_vec))
        nxt = pos + step_vec
        if not all(1.0 <= nxt[i] < occ.shape[i] - 1 for i in range(3)):
            break
        scans = obscurant.scan_pose(
            occ, tuple(pos), n_az=120, elevations=(-0.26, 0.0, 0.26),
            max_range=40.0, range_noise=0.15, seed=1000 + seed * 100 + k)
        n_feat = int(sum(1 for s in scans if s[3] and s[4]))
        # Achieved motion [design model]: lag + noise off the setpoint.
        achieved = cmd_speed * 0.9 + float(rng.normal(0, 0.02))
        achieved_speed = achieved  # per-step dt = 1 s
        # Estimator report from ACHIEVED motion.
        if cond == "whiteout_freeze":
            est = 0.125 * achieved
            resid = 0.03 + abs(rng.normal(0, 0.01))
        elif cond.startswith("freeze_r"):
            est = float(cond.split("freeze_r")[1]) * achieved
            resid = 0.03 + abs(rng.normal(0, 0.01))
        elif cond.startswith("partial_r"):
            est = float(cond.split("partial_r")[1]) * achieved
            resid = 0.03 + abs(rng.normal(0, 0.01))
        elif cond == "clear_weave":
            est = max(0.0, achieved + rng.normal(0, 0.30))
            resid = 0.30 + abs(rng.normal(0, 0.05))
        elif cond == "resid_mid":
            est = max(0.0, achieved + rng.normal(0, 0.03))
            resid = 0.20 + abs(rng.normal(0, 0.02))
        else:
            est = max(0.0, achieved + rng.normal(0, 0.03))
            resid = 0.03 + abs(rng.normal(0, 0.01))
        map_age = 2.0 if cond == "stale_map" else 0.0
        err = abs(est - achieved)
        errs.append(err)
        cum5 = float(np.sum(errs[-5:]))
        state, _info = mon.update(
            {"cmd_speed_mps": cmd_speed, "est_speed_mps": est,
             "est_jump_m": est, "n_features": n_feat,
             "residual_m": float(resid), "map_age_s": map_age,
             "dt_s": 1.0}, t=float(k))
        rows.append(dict(zip(FIELDS, [
            cond, room, seed, k, n_feat, round(cmd_speed, 4),
            round(achieved_speed, 4), round(est, 4),
            round(float(resid), 4), map_age, state,
            int(mon.pose_trustworthy), round(err, 4), round(cum5, 4),
            int(cum5 > 0.5), commit, PROV_NOTE])))
        pos = nxt
    return rows


def main():
    token = exec_provenance.begin(HERE, [OUT])
    all_rows = []
    for cond in CONDITIONS:
        for room in ROOMS:
            all_rows += run_condition(cond, room, 0, token)
    commit = exec_provenance.finish(HERE, token, [OUT])
    for r in all_rows:
        r["git_commit"] = commit
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {OUT} ({len(all_rows)} rows, executed commit "
          f"{commit[:8]})")
    for cond in CONDITIONS:
        rs = [r for r in all_rows if r["condition"] == cond]
        c = Counter(r["health_state"] for r in rs)
        untr = next((r["step"] for r in rs
                     if r["health_state"] == "UNTRUSTED"), None)
        print(f"{cond:18s} states={dict(c)} first_UNTRUSTED={untr} "
              f"truth_unhealthy={sum(r['truth_unhealthy'] for r in rs)}"
              f"/{len(rs)}")


if __name__ == "__main__":
    main()
