"""Turn-smoothness metrics from a 20 Hz trajectory. Shared by the Phase-2b
tuning sweep and the final Phase-2-vs-2b analysis.

The sawtooth signature: commanded heading jumps with each S2 decision, so
the flown yaw rate oscillates — sign flips near the decision rate, and
lateral acceleration rings. Smooth gradients show up as: low flip rate,
low lateral-accel std, bounded max yaw rate.
"""
import numpy as np


def turn_metrics(xyz, dt=0.05):
    xyz = np.asarray(xyz, dtype=float)
    v = np.diff(xyz, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    moving = speed > 0.3

    # yaw (horizontal heading) and yaw rate
    hdg = np.arctan2(v[:, 1], v[:, 0])
    dh = np.diff(np.unwrap(hdg)) / dt  # len n-2 effectively vs speed[1:]
    mv = moving[1:]
    dh_m = dh[mv]
    max_yaw = float(np.abs(dh_m).max()) if len(dh_m) else 0.0
    nz = np.abs(dh_m) > 0.15
    s = np.sign(dh_m[nz])
    flips = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0
    t_move = float(mv.sum()) * dt
    flip_rate = flips / t_move if t_move > 0 else 0.0

    # lateral acceleration (horizontal, perpendicular to velocity)
    a = np.diff(v, axis=0) / dt
    vh = v[:-1].copy()
    vh[:, 2] = 0.0
    spd = np.linalg.norm(vh, axis=1, keepdims=True) + 1e-9
    tangent = vh / spd
    normal = np.stack([-tangent[:, 1], tangent[:, 0],
                       np.zeros(len(tangent))], axis=1)
    a_lat = (a * normal).sum(axis=1)
    m2 = speed[:-1] > 0.3
    lat_std = float(a_lat[m2].std()) if m2.sum() > 1 else 0.0
    z = np.sign(a_lat[m2])
    zc = int((np.diff(z) != 0).sum()) if len(z) > 1 else 0
    lat_zc_hz = zc / (2.0 * m2.sum() * dt) if m2.sum() > 0 else 0.0
    return {
        "n_points": len(xyz),
        "mean_speed": round(float(speed.mean()), 3),
        "max_yaw_rate": round(max_yaw, 3),
        "yaw_flip_rate_hz": round(flip_rate, 3),
        "yaw_flips": flips,
        "lat_acc_std": round(float(lat_std), 3),
        "lat_acc_zc_hz": round(float(lat_zc_hz), 3),
        "slow_frac": round(float((speed < 0.2).mean()), 4),
    }
