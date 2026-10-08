"""S16 mission loop with real dynamics (Phase 1).

Mirrors head_to_head_thrml.run_mission_s16_thrml EXACTLY except for the
movement: instead of controller.move_toward teleporting the drone toward
the accepted voxel, each accepted S2 decision becomes a minimum-jerk
trajectory segment tracked by a cascaded PID at 100 Hz.

Unchanged: 1 Hz outer loop (scan from TRUE pose -> perception -> D5 ->
S2 at 10 substeps), D5 guards, S16Planner logic, gate modes, escapes,
peek, oscillation detection, manifest schema, outcome definitions.

Changed:
  - vehicle: PointMass double integrator (100 Hz), v_max=5 m/s,
    a_max=8 m/s^2 [assumed, X500-class].
  - per accepted substep: min-jerk quintic from current (pos, vel) to
    the accepted voxel center (v_end=0), time-stretched to limits;
    receding horizon (replanned every substep).
  - CBF supervisor (host-side classical): v_allow = sqrt(2*a_brake*
    (clearance - margin)) from a believed-map raycast along velocity;
    if current speed exceeds v_allow the voxel move is not executed and
    the drone brakes. The kinematic sim could stop instantly; inertia
    means D5's voxel check alone cannot prevent overshoot into walls.
  - trajectory-level D5: the 0.1 s swept path must pass safe_to_enter,
    not just the endpoint voxel.
  - refusal/brake: velocity-PID braking (v_sp=0), not a quintic to self
    (which swerves forward then back).
  - collision: CONTINUOUS — 3D DDA segment traversal per 100 Hz step
    against GT occupied voxels (not voxel-center sampling).
  - sensor reads from the true continuous pose.
  - trajectory: per-mission JSON at dynamics_sim/traj/<mission_id>.json
    (20 Hz downsample, float32-ish, demo-renderer contract).

Backend: "thrml" (default, matches V6) or "numpy" (fast iteration).
No pipeline files modified. Dynamics + control are host-side classical;
no TSU claim there.
"""
import os
import sys
import time
import json
import csv
import math

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env  # noqa: F401
import sensor
import planner
import controller
import classical
import s16 as s16_module

from vehicle import PointMass, V_MAX, A_MAX, DT
from traj import plan_segment
from pid import CascadedPID

from fly import (
    pick_start_goal, optimal_len_to_tolerance,
    GOAL_TOL, GOAL_BLOCKED_RADIUS, MAX_MOVE_REFUSALS, START_GOAL_MIN_DIST,
    N_AZ, ELEV, MAX_STEPS, J_SMOOTH,
    N_SWEEPS_FULL, BURN_IN_FULL, N_SWEEPS_INCR, BURN_IN_INCR,
)
from head_to_head import (
    effective_observed_mask, map_hash, manifest_row,
    MANIFEST_FIELDS, WEAK_OBS_MAX, STRONG_L_THRESH,
    REAL_LIDAR_DIR,
)
from head_to_head_v2 import (
    MANIFEST_FIELDS_V2, TSU_SUBSTEPS, MAX_MOVE_REFUSALS_S16,
)

TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj")
INNER_PER_SUBSTEP = 10          # 100 Hz dynamics, 10 Hz decisions
TRAJ_RECORD_EVERY = 5           # 20 Hz demo feed (contract dt=0.05)


def _voxels_on_segment(p0, p1, shape):
    """All voxels a segment passes through (Amanatides & Woo), INCLUDING
    start and end voxels, bounded by the segment parameter t<=1.

    NOTE: planner._intermediate_voxels omits the t<=1 cutoff, which is
    correct for its center-to-center line-of-sight use but yields phantom
    voxels for sub-voxel segments (it steps to the nearest boundary even
    when the segment never reaches it). At 100 Hz our steps are ~0.05
    voxels, so we need the cutoff here. Pipeline code untouched.
    """
    import math
    x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    nx, ny, nz = shape
    ix, iy, iz = math.floor(x0), math.floor(y0), math.floor(z0)
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        yield (ix, iy, iz)
    else:
        return  # start outside: caller treats as crash separately
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    if dx == 0 and dy == 0 and dz == 0:
        return
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

    def _tmax_tdelta(p, d, step):
        if step == 0:
            return float("inf"), float("inf")
        boundary = math.floor(p) + (1 if step > 0 else 0)
        return (boundary - p) / d, abs(1.0 / d)

    tmax_x, tdelta_x = _tmax_tdelta(x0, dx, step_x)
    tmax_y, tdelta_y = _tmax_tdelta(y0, dy, step_y)
    tmax_z, tdelta_z = _tmax_tdelta(z0, dz, step_z)
    guard = abs(math.floor(x1) - ix) + abs(math.floor(y1) - iy) \
        + abs(math.floor(z1) - iz) + 8
    for _ in range(guard):
        if tmax_x < tmax_y and tmax_x < tmax_z:
            if tmax_x > 1.0:
                return
            ix += step_x
            tmax_x += tdelta_x
        elif tmax_y < tmax_z:
            if tmax_y > 1.0:
                return
            iy += step_y
            tmax_y += tdelta_y
        else:
            if tmax_z > 1.0:
                return
            iz += step_z
            tmax_z += tdelta_z
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            yield (ix, iy, iz)
        else:
            return


def _segment_hits_occ(p0, p1, occ):
    """Continuous collision: does segment p0->p1 touch a GT occupied voxel
    (or leave the map)?"""
    nx, ny, nz = occ.shape
    # endpoint outside the map: crash (same rule as kinematic check_collision)
    if not (0 <= int(p1[0]) < nx and 0 <= int(p1[1]) < ny
            and 0 <= int(p1[2]) < nz):
        return True
    for v in _voxels_on_segment(p0, p1, occ.shape):
        if bool(occ[v]):
            return True
    return False


def _write_traj_json(mission_id, room_id, xyz_list, outcome):
    """Demo-renderer contract: dynamics_sim/traj/<mission_id>.json."""
    os.makedirs(TRAJ_DIR, exist_ok=True)
    path = os.path.join(TRAJ_DIR, f"{mission_id}.json")
    doc = {
        "mission_id": mission_id,
        "room_id": room_id,
        "dt": 0.05,
        "xyz": [[round(float(x), 4), round(float(y), 4), round(float(z), 4)]
                for x, y, z in xyz_list],
        "outcome": outcome,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)
    return path


def _traj_passes_d5(seg, pos_v, marg, obs_mask, shape, n_samples=10):
    """Trajectory-level D5: every voxel the executed 0.1 s will sweep must
    pass safe_to_enter. Extends D5's invariant ("never execute a step into
    a voxel that fails this check") from the endpoint voxel to the actual
    swept path, which is what inertia makes relevant."""
    t_end = min(0.1, seg.T)
    for k in range(1, n_samples + 1):
        p, _, _ = seg.eval(t_end * k / n_samples)
        v = (int(p[0]), int(p[1]), int(p[2]))
        if not (0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]
                and 0 <= v[2] < shape[2]):
            return False
        if not planner.safe_to_enter(v, marg, obs_mask, pos_v):
            return False
    return True


# Control barrier function: never outrun the stopping distance.
A_BRAKE = 0.7 * A_MAX   # braking decel with safety factor [assumed]
CBF_MARGIN = 0.5        # extra clearance buffer, m [assumed]


def _blocked_believed(v, marg, obs_mask):
    """Believed-map blocked test (mirrors D5's core deny logic, fast)."""
    if bool(obs_mask[v]):
        return float(marg[v]) > planner.P_OBSTACLE
    return float(marg[v]) > planner.P_UNOBSERVED_DENY


def _clearance(pos, vel, marg, obs_mask, shape, max_d=15.0):
    """Distance to nearest believed-blocked voxel along velocity, or to the
    map boundary (the boundary is a hard wall), whichever is closer."""
    sp = float(np.linalg.norm(vel))
    if sp < 0.1:
        return float("inf")
    d = np.asarray(vel, dtype=float) / sp
    p = np.asarray(pos, dtype=float)
    # distance to map exit along the ray
    d_exit = float("inf")
    for i in range(3):
        if d[i] > 1e-9:
            d_exit = min(d_exit, (shape[i] - p[i]) / d[i])
        elif d[i] < -1e-9:
            d_exit = min(d_exit, (0.0 - p[i]) / d[i])
    best = min(d_exit, max_d)
    p1 = p + d * max_d
    for v in _voxels_on_segment(pos, p1, shape):
        if _blocked_believed(v, marg, obs_mask):
            c = np.array([v[0] + 0.5, v[1] + 0.5, v[2] + 0.5])
            best = min(best, max(0.0, float(np.linalg.norm(c - p)) - 0.9))
            break
    return float(best)


def _get_backend(backend):
    """Lazy backend imports (THRML pulls in JAX)."""
    if backend == "thrml":
        from head_to_head_thrml import (
            compute_marginals_thrml as _marg, ThrmlS2Agent as _S2)
        def make_s2(seed):
            return _S2(T=3, n_sweeps=200, seed=seed)
        def marg(logodds, h, x_last, seed, step):
            return _marg(logodds, h, x_last, seed, step)
    elif backend == "numpy":
        from head_to_head import compute_marginals as _marg
        from harness import S2Agent as _S2
        def make_s2(seed):
            return _S2(T=3, n_sweeps=200, seed=seed)
        def marg(logodds, h, x_last, seed, step):
            return _marg("tsu", logodds, h, x_last, seed, step)
    else:
        raise ValueError(f"unknown backend {backend}")
    return marg, make_s2


def run_mission_dynamics(occ, room_id, seed, noise_val, mission_id,
                         backend="thrml", telemetry_dir=None, verbose=False,
                         progress=False, write_traj=True, debug_log=None):
    """Fly one S16 mission with dynamics. Returns manifest row dict."""
    t_mission = time.time()
    shape = occ.shape
    compute_marginals, make_s2 = _get_backend(backend)

    start_pose, goal_pose = pick_start_goal(occ, seed)
    vehicle = PointMass(pos=start_pose)
    pid = CascadedPID()  # tuned gains are the defaults
    logodds = np.zeros(shape)
    obs_count = np.zeros(shape, dtype=np.int32)

    # Boundary prior (same as fly.py / thrml runner)
    logodds[0, :, :] = 2.0
    logodds[-1, :, :] = 2.0
    logodds[:, 0, :] = 2.0
    logodds[:, -1, :] = 2.0
    logodds[:, :, 0] = 2.0
    logodds[:, :, -1] = 2.0
    _sv = planner.pose_to_voxel(start_pose, shape)
    _gv = planner.pose_to_voxel(goal_pose, shape)
    logodds[_sv] = 0.0
    logodds[_gv] = 0.0

    p_occ_gt = occ.astype(float)
    gt_full = planner.astar(p_occ_gt,
                            planner.pose_to_voxel(start_pose, shape),
                            planner.pose_to_voxel(goal_pose, shape))
    optimal_len = optimal_len_to_tolerance(gt_full, goal_pose, GOAL_TOL) \
        if gt_full else -1

    log = {
        "mission_id": mission_id, "config": "s16-dyn", "room_id": room_id,
        "seed": seed, "noise_val": noise_val,
        "start": start_pose, "goal": goal_pose,
        "optimal_path_len": optimal_len, "steps": [], "outcome": None,
        "n_plan_calls": 0, "n_planning_failures": 0, "n_move_refusals": 0,
        "n_tsu_decisions": 0, "total_tsu_time": 0.0, "path_voxels": [],
        "collision_p_occ": None, "total_gibbs_time": 0.0,
        "total_plan_time": 0.0,
    }

    s16p = s16_module.S16Planner(
        waypoint_lookahead=2, stuck_threshold=40,
        s2_kwargs={"seed": seed, "n_sweeps": 200})
    s16p.s2 = make_s2(seed)

    x_last = None
    goal_v = planner.pose_to_voxel(goal_pose, shape)
    marg = np.zeros(shape)
    t_sim = 0.0
    traj_xyz = []  # 20 Hz demo feed
    inner_tick = 0
    n_cbf_brakes = 0
    dbg = open(debug_log, "w") if debug_log else None

    def _record():
        traj_xyz.append((vehicle.pos[0], vehicle.pos[1], vehicle.pos[2]))

    _record()
    outcome_set = False

    for step in range(MAX_STEPS):
        pose = tuple(float(c) for c in vehicle.pos)
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] {mission_id} step={step} "
                  f"pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                  f"vel={np.linalg.norm(vehicle.vel):.2f}", flush=True)

        # ---- 1 Hz host: scan from TRUE continuous pose ----
        scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                 max_range=40.0, range_noise=noise_val,
                                 seed=1000 + seed * 100 + step)
        sensor.integrate_scan(logodds, shape, scans, max_range=40.0,
                              obs_count=obs_count)

        # ---- 1 Hz host: Gibbs -> P(occupied) ----
        h = sensor.quantize_8bit(logodds)
        marg, dt_gibbs, x_last = compute_marginals(
            logodds, h, x_last, seed, step)
        log["total_gibbs_time"] += dt_gibbs
        obs_mask = effective_observed_mask(logodds, obs_count, "tsu")

        # ---- D3: goal-blocked check ----
        if (controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
                and bool(obs_mask[goal_v])
                and float(marg[goal_v]) > planner.P_OBSTACLE):
            log["steps"].append({"step": step, "pose": pose,
                                 "event": "goal_blocked"})
            log["outcome"] = "goal_blocked"
            break

        # ---- 10 Hz TSU fast loop, 100 Hz dynamics inside ----
        for tsu_sub in range(TSU_SUBSTEPS):
            pose = tuple(float(c) for c in vehicle.pos)
            try:
                pos_v = planner.pose_to_voxel(pose, shape)
            except ValueError:
                log["outcome"] = "collision"  # left the map
                outcome_set = True
                break
            # Dynamics note: S16Planner._goal_fallback latches once the
            # waypoint-progress detector fires. Under kinematics that means
            # "truly stuck"; under dynamics, CBF braking (normal, temporary)
            # also trips it, and the latch then abandons A* waypoints for
            # the rest of the mission. Reset each outer step so it
            # re-evaluates from current progress. (Pipeline file untouched;
            # the dynamics layer manages this planner state.)
            if tsu_sub == 0:
                s16p._goal_fallback = False
            next_v, s2info = s16p.tsu_act(
                pos_v, goal_v, marg, obs_mask, pos_cont=pose)
            log["n_tsu_decisions"] += 1
            log["total_tsu_time"] += s2info["tsu_time"]
            _sf = float(s2info.get("speed_factor", 1.0))
            # CBF supervisor: never outrun the stopping distance to
            # believed-blocked voxels along the velocity vector.
            sp = float(np.linalg.norm(vehicle.vel))
            d_clear = _clearance(vehicle.pos, vehicle.vel, marg, obs_mask,
                                 shape)
            v_allow = math.sqrt(2 * A_BRAKE * max(d_clear - CBF_MARGIN, 0.0))
            v_max_eff = min(V_MAX * _sf, v_allow)
            cbf_brake = sp > v_allow + 0.05

            # D5: predictive guard on the proposed voxel.
            d5_ok = planner.safe_to_enter(next_v, marg, obs_mask, pos_v)
            if dbg:
                dbg.write(json.dumps({
                    "step": step, "sub": tsu_sub, "t": round(t_sim, 2),
                    "pos": [round(float(c), 3) for c in vehicle.pos],
                    "vel": [round(float(c), 3) for c in vehicle.vel],
                    "pos_v": pos_v, "next_v": next_v,
                    "d5_ok": bool(d5_ok),
                    "mode": s2info.get("mode"),
                    "wp": list(s2info.get("waypoint") or []),
                    "gfb": bool(s16p._goal_fallback),
                    "marg_next": round(float(marg[next_v]), 3)
                    if all(0 <= c < s for c, s in zip(next_v, shape)) else None,
                    "obs_next": bool(obs_mask[next_v])
                    if all(0 <= c < s for c, s in zip(next_v, shape)) else None,
                }) + "\n")
            _braking = False
            if cbf_brake:
                # CBF supervisor: already too fast to stop within
                # clearance -> brake now, don't execute the voxel move.
                # Not a D5 refusal (no escape logic); normal supervision.
                _braking = True
                n_cbf_brakes += 1
                if dbg:
                    dbg.write(json.dumps({
                        "step": step, "sub": tsu_sub, "event": "cbf_brake",
                        "sp": round(sp, 2), "v_allow": round(v_allow, 2),
                        "d_clear": round(d_clear, 2)}) + "\n")
            else:
                _refused = False
                if not d5_ok:
                    _refused = True
                else:
                    target = planner.voxel_to_pose(next_v)
                    seg = plan_segment(vehicle.pos, vehicle.vel, target,
                                       (0, 0, 0), v_max_eff, A_MAX)
                    # Trajectory-level D5: the swept path must be approved,
                    # not just the endpoint (inertia makes the path matter).
                    if not _traj_passes_d5(seg, pos_v, marg, obs_mask, shape):
                        _refused = True
                        if dbg:
                            dbg.write(json.dumps(
                                {"step": step, "sub": tsu_sub,
                                 "event": "traj_refused"}) + "\n")
                if _refused:
                    log["n_move_refusals"] += 1
                    s16p.notify_refusal(next_v, pos_v, marg, obs_mask)
                    if log["n_move_refusals"] > MAX_MOVE_REFUSALS_S16:
                        log["outcome"] = "blocked"
                        outcome_set = True
                        break
                    # hover: brake via velocity PID (v_sp=0 -> near
                    # time-optimal monotonic decel).
                    _braking = True
            # (accept case: seg already planned above)

            # execute 0.1 s at 100 Hz
            t_seg = 0.0
            for _ in range(INNER_PER_SUBSTEP):
                if _braking:
                    a_cmd = pid.command(vehicle.pos, (0, 0, 0),
                                        vehicle.pos, vehicle.vel)
                else:
                    p_des, v_des, _ = seg.eval(min(t_seg, seg.T))
                    a_cmd = pid.command(p_des, v_des, vehicle.pos,
                                        vehicle.vel)
                p_prev = vehicle.pos.copy()
                vehicle.step(a_cmd)
                t_seg += DT
                t_sim += DT
                inner_tick += 1
                if _segment_hits_occ(p_prev, vehicle.pos, occ):
                    v = planner.pose_to_voxel(
                        tuple(float(c) for c in vehicle.pos), shape) \
                        if all(0 <= int(c) < s for c, s in
                               zip(vehicle.pos, shape)) else pos_v
                    log["collision_p_occ"] = float(marg[v]) \
                        if all(0 <= int(c) < s for c, s in
                               zip(vehicle.pos, shape)) else -1.0
                    log["outcome"] = "collision"
                    outcome_set = True
                    break
                if inner_tick % TRAJ_RECORD_EVERY == 0:
                    _record()
            if outcome_set:
                break

            v = planner.pose_to_voxel(tuple(float(c) for c in vehicle.pos),
                                      shape)
            if v not in log["path_voxels"][-1:]:
                log["path_voxels"].append(v)

            if controller.distance(tuple(float(c) for c in vehicle.pos),
                                   goal_pose) < GOAL_TOL:
                log["outcome"] = "success"
                outcome_set = True
                break
        else:
            log["n_plan_calls"] = s16p.n_host_replans
            log["n_planning_failures"] = s16p.n_host_failures
            log["steps"].append({"step": step, "n_tsu": TSU_SUBSTEPS})
            continue
        log["n_plan_calls"] = s16p.n_host_replans
        log["n_planning_failures"] = s16p.n_host_failures
        break
    else:
        log["outcome"] = "max_steps"

    # --- mission metrics (same as V1/V2/V6) ---
    log["n_steps"] = len(log["steps"])
    log["wall_time"] = time.time() - t_mission
    # flown_path_len: TRUE continuous path length from the 20 Hz trajectory
    # (not len(path_voxels) — that counts voxel entries, which understates
    # the smooth dynamics path). Matches the demo JSON xyz.
    if len(traj_xyz) >= 2:
        _tp = np.asarray(traj_xyz, dtype=float)
        log["flown_path_len"] = round(
            float(np.linalg.norm(np.diff(_tp, axis=0), axis=1).sum()), 2)
    else:
        log["flown_path_len"] = 0
    log["final_map_hash"] = map_hash(marg)
    log["sim_time"] = round(t_sim, 2)

    if log["path_voxels"]:
        flown_occ = [bool(occ[v]) for v in log["path_voxels"]]
        log["flown_fn_rate"] = float(np.mean(flown_occ))
    else:
        log["flown_fn_rate"] = 0.0

    log["final_acc"] = classical.accuracy(marg, occ)

    obs_final = effective_observed_mask(logodds, obs_count, "tsu")
    base_final = classical.observed_mask(logodds)
    gt_free = (occ == 0)
    gt_occ = (occ == 1)
    log["phantom_count"] = int(
        ((marg > planner.P_OBSTACLE) & gt_free & obs_final).sum())
    log["phantom_count_base"] = int(
        ((marg > planner.P_OBSTACLE) & gt_free & base_final).sum())
    log["fn_count_observed"] = int(
        ((marg < planner.P_CAUTION) & gt_occ & obs_final).sum())
    log["fn_count_total"] = int(
        ((marg < planner.P_CAUTION) & gt_occ).sum())

    if log["optimal_path_len"] > 0 and log["outcome"] == "success":
        log["path_efficiency"] = log["optimal_path_len"] / log["flown_path_len"]
    else:
        log["path_efficiency"] = None

    if telemetry_dir:
        os.makedirs(telemetry_dir, exist_ok=True)
        tpath = os.path.join(telemetry_dir, f"{mission_id}.jsonl")
        with open(tpath, "w") as f:
            for s in log["steps"]:
                f.write(json.dumps(s, default=str) + "\n")
        log["telemetry_file"] = tpath

    # trajectory JSON (demo contract) — written even on failure
    traj_path = None
    if write_traj:
        traj_path = _write_traj_json(mission_id, room_id, traj_xyz,
                                     log["outcome"])
    log["traj_file"] = traj_path
    if dbg:
        dbg.close()

    if verbose:
        print(f"  [s16-dyn-{backend}] {mission_id}: {log['outcome']} "
              f"(wall {log['wall_time']:.1f}s, sim {log['sim_time']:.1f}s, "
              f"tsu={log['n_tsu_decisions']}, cbf_brakes={n_cbf_brakes})")

    row = manifest_row(log)
    row["n_escapes"] = s16p.n_escapes
    row["n_d5_refusals_s16"] = s16p.n_d5_refusals
    row["n_escape_replans"] = s16p.n_escape_replans
    row["n_tsu_decisions"] = log["n_tsu_decisions"]
    row["total_tsu_time"] = round(log["total_tsu_time"], 3)
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser(description="S16 missions with dynamics")
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--noise", type=float, nargs="*", default=[0.15, 0.30])
    ap.add_argument("--backend", default="thrml", choices=["thrml", "numpy"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--manifest", default="RUN_MANIFEST_DYN_V1.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--no-traj", action="store_true",
                    help="skip trajectory JSON writing")
    args = ap.parse_args()

    from head_to_head import list_rooms

    outdir = args.outdir or os.path.join(_BASE, "dynamics_sim")
    tdir = os.path.join(outdir, "telemetry_dyn")
    mpath = os.path.join(outdir, args.manifest)
    os.makedirs(tdir, exist_ok=True)

    rooms = args.rooms or list_rooms()
    print(f"rooms: {len(rooms)}, seeds: {args.seeds}, noise: {args.noise}, "
          f"backend: {args.backend}")

    done = set()
    if args.resume and os.path.exists(mpath):
        with open(mpath) as f:
            for row in csv.DictReader(f):
                done.add(row["mission_id"])
        print(f"resuming: {len(done)} missions already done")

    occ_cache = {}
    for r in rooms:
        p = os.path.join(REAL_LIDAR_DIR, f"{r}_occ.npy")
        occ_cache[r] = np.load(p)

    write_header = not (os.path.exists(mpath) and args.resume
                        and os.path.getsize(mpath) > 0)
    fout = open(mpath, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS_V2)
    if write_header:
        writer.writeheader()
        fout.flush()

    n_done = 0
    t_all = time.time()
    try:
        for room_id in rooms:
            occ = occ_cache[room_id]
            for seed in args.seeds:
                for noise_val in args.noise:
                    mid = f"{room_id}_s{seed}_n{noise_val:.2f}_s16"
                    if mid in done:
                        continue
                    row = run_mission_dynamics(
                        occ, room_id, seed, noise_val, mid,
                        backend=args.backend, telemetry_dir=tdir,
                        verbose=True, progress=args.progress,
                        write_traj=not args.no_traj)
                    writer.writerow(row)
                    fout.flush()
                    n_done += 1
                    done.add(mid)
                    if args.smoke and n_done >= args.smoke:
                        print(f"smoke: {n_done} missions done, stopping")
                        return
    finally:
        fout.close()

    print(f"\ndone: {n_done} missions in {time.time()-t_all:.0f}s")
    print(f"manifest: {mpath}")


if __name__ == "__main__":
    main()
