"""PID tuning: velocity step (inner loop) + trajectory tracking + hover.

Run: python3 tune_pid.py
Tries a small gain grid, reports overshoot / settling / tracking error,
saves the winning step-response time series as JSON (matplotlib is broken
in this env — numpy 2.x vs 1.x build — so series data is saved instead of
PNGs; the report carries the numbers).

Gains are chosen from these measurements, not hand-picked.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vehicle import PointMass, V_MAX, A_MAX, DT
from pid import CascadedPID
from traj import plan_segment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning")
os.makedirs(OUT, exist_ok=True)


def _settle(t, y, target, band=0.02):
    y = np.asarray(y)
    b = band * abs(target)
    for i in range(len(t) - 1, -1, -1):
        if abs(y[i] - target) > b:
            return round(float(t[min(i + 1, len(t) - 1)]), 3)
    return 0.0


def velocity_step_test(Kp_p, Kp_v, Ki_v, Kd_v, v_target=3.0, T=4.0):
    """Inner loop only: Kp_p=0, p_des=pos (no position correction)."""
    pid = CascadedPID(Kp_p, Kp_v, Ki_v, Kd_v)
    pid.Kp_p = 0.0
    drone = PointMass(pos=(0, 0, 0))
    ts, vs = [], []
    t = 0.0
    while t < T:
        a = pid.command(drone.pos, (v_target, 0, 0), drone.pos, drone.vel)
        drone.step(a)
        t += DT
        ts.append(t)
        vs.append(drone.vel[0])
    vs = np.array(vs)
    os_ = max(0.0, (vs.max() - v_target) / v_target * 100.0)
    return (dict(overshoot_pct=round(float(os_), 2),
                 settle_s=_settle(ts, vs, v_target),
                 sse=round(float(abs(vs[-1] - v_target)), 4)),
            (ts, [float(v) for v in vs]))


def traj_track_test(Kp_p, Kp_v, Ki_v, Kd_v, dist=5.0):
    """Track a minimum-jerk trajectory; the real mission-loop scenario."""
    pid = CascadedPID(Kp_p, Kp_v, Ki_v, Kd_v)
    drone = PointMass(pos=(0, 0, 0))
    seg = plan_segment((0, 0, 0), (0, 0, 0), (dist, 0, 0), (0, 0, 0),
                       V_MAX, A_MAX)
    ts, ps, errs = [], [], []
    t = 0.0
    T = seg.T + 3.0
    while t < T:
        p_des, v_des, _ = seg.eval(min(t, seg.T))
        a = pid.command(p_des, v_des, drone.pos, drone.vel)
        drone.step(a)
        t += DT
        ts.append(t)
        ps.append(float(drone.pos[0]))
        errs.append(float(abs(drone.pos[0] - p_des[0])))
    ps = np.array(ps)
    overshoot = max(0.0, ps.max() - dist)  # meters past target (wall risk)
    return (dict(max_track_err_m=round(float(np.max(errs)), 4),
                 overshoot_m=round(float(overshoot), 4),
                 settle_s=_settle(ts, ps, dist),
                 sse_m=round(float(abs(ps[-1] - dist)), 4),
                 traj_T_s=round(seg.T, 3)),
            (ts, ps))


def hover_test(Kp_p, Kp_v, Ki_v, Kd_v, T=5.0):
    pid = CascadedPID(Kp_p, Kp_v, Ki_v, Kd_v)
    drone = PointMass(pos=(0, 0, 0))
    worst = 0.0
    t = 0.0
    while t < T:
        a = pid.command((0, 0, 0), (0, 0, 0), drone.pos, drone.vel)
        drone.step(a)
        t += DT
        worst = max(worst, float(np.linalg.norm(drone.pos)))
    return round(worst, 5)


def main():
    grid = []
    for Kp_p in (0.8, 1.2, 1.6):
        for Kp_v in (4.0, 6.0, 8.0):
            for Ki_v in (0.5, 1.5):
                for Kd_v in (0.6, 1.1, 1.6):
                    grid.append((Kp_p, Kp_v, Ki_v, Kd_v))
    rows = []
    for g in grid:
        mv, _ = velocity_step_test(*g)
        mt, _ = traj_track_test(*g)
        hov = hover_test(*g)
        # safety first: tracking error + overshoot dominate
        score = (mt["max_track_err_m"] * 50 + mt["overshoot_m"] * 100
                 + mt["settle_s"] + mv["settle_s"] + mv["overshoot_pct"] * 0.1
                 + hov * 10)
        rows.append((score,) + g + (mv, mt, hov))
        print(f"Kp_p={g[0]:.1f} Kp_v={g[1]:.1f} Ki_v={g[2]:.1f} Kd_v={g[3]:.1f} "
              f"| vel os={mv['overshoot_pct']}% st={mv['settle_s']}s | "
              f"trk err={mt['max_track_err_m']}m os={mt['overshoot_m']}m "
              f"st={mt['settle_s']}s | hov={hov}m | score={score:.2f}",
              flush=True)
    rows.sort(key=lambda r: r[0])
    b = rows[0]
    print(f"\nBEST: Kp_p={b[1]} Kp_v={b[2]} Ki_v={b[3]} Kd_v={b[4]} "
          f"score={b[0]:.2f}\n  vel={b[5]}\n  track={b[6]}\n  hover_drift={b[7]}m")

    # save winning series (for the report; no matplotlib in this env)
    _, vseries = velocity_step_test(b[1], b[2], b[3], b[4])
    _, tseries = traj_track_test(b[1], b[2], b[3], b[4])
    with open(os.path.join(OUT, "winning_series.json"), "w") as f:
        json.dump({
            "gains": {"Kp_p": b[1], "Kp_v": b[2], "Ki_v": b[3], "Kd_v": b[4]},
            "velocity_step": {"t": vseries[0][::10], "v": vseries[1][::10],
                              "target": 3.0},
            "traj_track": {"t": tseries[0][::10], "x": list(tseries[1][::10]),
                           "target": 5.0},
        }, f)
    with open(os.path.join(OUT, "tuning_results.txt"), "w") as f:
        for r in rows:
            f.write(f"score={r[0]:.2f} Kp_p={r[1]} Kp_v={r[2]} Ki_v={r[3]} "
                    f"Kd_v={r[4]} vel={r[5]} track={r[6]} hover={r[7]}\n")
    print("saved tuning/winning_series.json + tuning_results.txt")
    return b[1], b[2], b[3], b[4]


if __name__ == "__main__":
    main()
