"""S16 mission loop with real dynamics — Phase 4: PARTICLE OBSCURANT.

Phase 3 changes THREE things vs Phase 2b (mission_dynamics_v2b.py), each
flag-gated so ablations can isolate them (wind only / attitude only /
gravity+envelope only / all three):

1. Vector low-pass on the velocity command (TAU_CMD=0.25 s). S2's 10 Hz
   decisions flicker, alternate, and rotate — the drone's inertia turned
   that into a ~1 Hz sawtooth in the flown path (yaw flips at 0.94 Hz,
   measured; Paul: "trying to balance way too hard"). The low-pass
   attenuates decision noise at the source frequency: flicker averages
   out, alternation cancels toward hover (indecision → caution), genuine
   turns track with ~0.25 s lag. (Two earlier designs failed loudly and
   were removed: a slew-rate limiter — command lag created an unvetted
   transient, collisions on the numpy sweep; a majority vote — cannot
   filter perfect alternation, corridor clip on THRML. Both documented in
   DYNAMICS_PHASE2B.md §2.)
2. Continuous slow-for-corners. Phase 2's bang-bang sharp-turn cap
   (>45° -> 0.5 m/s) is replaced by a smooth scale
   cos(angle/2)^TURN_SHARPNESS on the cruise speed — no thresholds, no
   chatter around a cutoff. Operating on the low-pass-stabilized command
   direction, it slows for real corners instead of flicker.

Everything else is identical to Phase 2: velocity servo cruise, quintic
terminal approach, THRML perception, S2, A*, D5 (+trajectory-level D5),
CBF supervisor, velocity-PID braking, continuous DDA collision checking
vs GT, 100 Hz cascaded PID. Phase-1 and Phase-2 files are NOT modified;
this file is the A/B comparison branch. No pipeline files modified.
Dynamics + control are host-side classical; no TSU claim there.
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
import obscurant
import planner
import controller
import classical
import s16 as s16_module

from vehicle_v3 import (Phase3Vehicle, V_MAX, DT, A_MAX_OLD, envelope,
                        TW_PRIMARY, TAU_TILT)
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

TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj4")
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


def _vel_swept_ok(pos, v_cmd, pos_v, marg, obs_mask, shape,
                  window=0.1, n_samples=10):
    """Trajectory-level D5 for the velocity servo: every voxel the
    constant-velocity extrapolation sweeps over the execution window must
    pass safe_to_enter. Same invariant as _traj_passes_d5, for the
    cruise (non-quintic) case."""
    pos = np.asarray(pos, dtype=float)
    v_cmd = np.asarray(v_cmd, dtype=float)
    for k in range(1, n_samples + 1):
        p = pos + v_cmd * (window * k / n_samples)
        v = (int(p[0]), int(p[1]), int(p[2]))
        if not (0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]
                and 0 <= v[2] < shape[2]):
            return False
        if not planner.safe_to_enter(v, marg, obs_mask, pos_v):
            return False
    return True


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
# A_BRAKE / A_PLAN are cfg-driven globals, set at the top of
# run_mission_dynamics (they depend on the gravity-envelope flag and T/W).
A_BRAKE = None  # set per run: 0.7 * lateral braking authority [assumed factor]
A_PLAN = None   # set per run: accel limit handed to the quintic planner
CBF_MARGIN = 0.5        # extra clearance buffer, m [assumed]

# Phase 2: fly-through waypoints.
TERMINAL_RADIUS = 3.0   # m [design choice]: inside this distance to the
                        # goal, a quintic to the real target with v_end=0
                        # gives a smooth terminal approach (Phase 1).
                        # Outside it, a velocity servo commands cruise
                        # directly (see accept branch).
V_MIN_CREEP = 0.15      # m/s [design choice]: floor on the cruise speed
                        # so a conservative clearance ray can't freeze
                        # the drone; the CBF supervisor still brakes for
                        # real danger independently.
TURN_ANGLE_DEG = 45.0   # [superseded in Phase 2b]: kept for reference;
TURN_V_MAX = 0.5        # the bang-bang sharp-turn cap is replaced by the
                        # continuous _turn_speed_scale below.
# Phase 2b (revised v2): the majority vote is REMOVED. Measured: it cannot
# filter perfect alternation (odd-W vote flips every substep on A,B,A,B),
# so in a corridor trap the drone followed the alternation at speed and
# clipped a wall (p2b_indoorcd_128_s3_n0.30_s16, THRML — a 2b regression
# vs Phase 2). Replacement: VECTOR LOW-PASS on the velocity command.
# S2's 10 Hz decisions flicker/alternate/rotate; the drone's inertia turned
# that into a ~1 Hz sawtooth in the flown path. A vector low-pass with
# TAU_CMD=0.25 s attenuates decision noise at the source frequency:
#  - irregular flicker averages out (command holds the persistent intent)
#  - alternation CANCELS to ~0 vector → the drone slows instead of
#    thrashing (indecision → caution, the safe response to confusion)
#  - genuine turns track with ~TAU lag (0.25 s, not the slew's 1-2 s)
# The continuous turn scale still slows sharp turns; CBF/D5 unchanged.
FILTER_TAU = 0.4       # s [tuned Phase 2b] or None: velocity-command
                        # low-pass time constant. None BYPASSES the filter
                        # (tilt-lag-only ablation: does physics replace it?).
                        # Set per run from cfg['filter_tau'].
DT_SUB_LEN = 0.1        # s: S2 decision substep (TSU_SUBSTEPS=10 at 1 Hz).
TURN_SHARPNESS = 3.0    # [tuned]: exponent in the continuous slow-for-
                        # corners scale cos(angle/2)^TURN_SHARPNESS.


def _turn_speed_scale(angle_rad, p):
    """Continuous slow-for-corners: 1 at 0°, decaying smoothly to 0 at
    180°. No thresholds, no chatter. The V_MIN_CREEP floor at the call
    site keeps a full reversal from commanding a dead stop."""
    return max(0.0, math.cos(angle_rad / 2.0)) ** p


def _blocked_believed(v, marg, obs_mask):
    """Believed-map blocked test (mirrors D5's core deny logic, fast)."""
    if bool(obs_mask[v]):
        return float(marg[v]) > planner.P_OBSTACLE
    return float(marg[v]) > planner.P_UNOBSERVED_DENY


def _clearance_along(pos, d_hat, marg, obs_mask, shape, max_d=15.0):
    """Distance to nearest believed-blocked voxel along unit direction
    d_hat, or to the map boundary (a hard wall), whichever is closer."""
    p = np.asarray(pos, dtype=float)
    d = np.asarray(d_hat, dtype=float)
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


def _v_allow_for(d_clear):
    """Stopping-distance speed cap for a clearance (CBF formula)."""
    return math.sqrt(2 * A_BRAKE * max(d_clear - CBF_MARGIN, 0.0))


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
                         progress=False, write_traj=True, debug_log=None,
                         cfg=None):
    """Fly one S16 mission with dynamics. Returns manifest row dict."""
    t_mission = time.time()
    shape = occ.shape
    compute_marginals, make_s2 = _get_backend(backend)

    # ---- Phase 3 per-run physics config ----
    global A_BRAKE, A_PLAN, FILTER_TAU
    cfg = dict(cfg or {})
    _grav = bool(cfg.get("gravity_envelope", False))
    _tw = float(cfg.get("tw_ratio", TW_PRIMARY))
    if _grav:
        _env = envelope(_tw)
        A_BRAKE = 0.7 * _env["a_lat_max"]   # lateral braking authority
        A_PLAN = min(_env["a_up_max"], _env["a_lat_max"])
    else:
        A_BRAKE = 0.7 * A_MAX_OLD
        A_PLAN = A_MAX_OLD
    FILTER_TAU = cfg.get("filter_tau", 0.4)  # None = bypass
    _obsc = cfg.get("obscurant") or {"archetype": "fog", "density": "clear"}
    obscurant.configure(_obsc.get("archetype", "fog"),
                        _obsc.get("density", "clear"))

    start_pose, goal_pose = pick_start_goal(occ, seed)
    vehicle = Phase3Vehicle(pos=start_pose, cfg=cfg, seed=seed)
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
    v_cmd_filt = None  # Phase 2b: low-passed velocity command (persists
                       # across substeps). None = snap to next command.
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
        scans = obscurant.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                    max_range=40.0, range_noise=noise_val,
                                    seed=1000 + seed * 100 + step)
        _, _, n_skipped = obscurant.integrate_scan(logodds, shape, scans,
                                                  max_range=40.0,
                                                  obs_count=obs_count)
        log["n_rays_skipped"] = log.get("n_rays_skipped", 0) + n_skipped

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
            # (Supervisor itself unchanged from Phase 1.)
            sp = float(np.linalg.norm(vehicle.vel))
            if sp < 0.1:
                d_clear = float("inf")
            else:
                d_clear = _clearance_along(vehicle.pos, vehicle.vel / sp,
                                           marg, obs_mask, shape)
            v_allow = _v_allow_for(d_clear)
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
            use_servo = False  # Phase 2: set in the accept branch below
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
                    # ---- Phase 2: fly-through waypoints ----
                    # Cruise: VELOCITY SERVO. Command the cruise velocity
                    # directly — the inner velocity PID (0.73 s settle,
                    # measured in Phase 1 tuning) tracks it with no ramp
                    # trap. p_des = pos so the outer P term contributes
                    # nothing; steering comes from v_cmd direction, updated
                    # every 0.1 s by S2. (Why not a quintic with
                    # v_end=cruise: under 10 Hz receding-horizon replanning
                    # the min-jerk profile back-loads its acceleration —
                    # each replan restarts the ramp and velocity never
                    # builds. Measured: 0.03 m/s after 6 s. The servo is
                    # the right tool for sustained cruise; the quintic is
                    # the right tool for the terminal stop.)
                    # Terminal: inside TERMINAL_RADIUS of the goal, quintic
                    # to the real target with v_end=0 (Phase 1 behavior).
                    d_goal = controller.distance(
                        tuple(float(c) for c in vehicle.pos), goal_pose)
                    dvec = np.asarray(target, dtype=float) - vehicle.pos
                    nd = float(np.linalg.norm(dvec))
                    use_servo = False
                    seg = None
                    v_cmd = None
                    if d_goal < TERMINAL_RADIUS or nd < 1e-9:
                        v_end = np.zeros(3)
                        seg = plan_segment(vehicle.pos, vehicle.vel, target,
                                           v_end, v_max_eff, A_PLAN)
                    else:
                        d_hat = dvec / nd
                        d_clear_dec = _clearance_along(
                            vehicle.pos, d_hat, marg, obs_mask, shape)
                        v_allow_dec = _v_allow_for(d_clear_dec)
                        # Phase 2b: continuous slow-for-corners + VECTOR
                        # LOW-PASS on the velocity command. The raw command
                        # d_hat*v_lim still jumps with S2's 10 Hz flicker;
                        # the low-pass (TAU_CMD) attenuates that noise at
                        # the source: flicker averages out, alternation
                        # cancels toward hover, genuine turns track with
                        # ~0.25 s lag. turn_scale uses the FILTERED
                        # direction (stable, no flicker in the scale).
                        v_raw = d_hat * max(
                            V_MIN_CREEP,
                            min(V_MAX * _sf, v_allow, v_allow_dec))
                        if FILTER_TAU is None:
                            v_cmd_filt = v_raw  # bypassed: tilt-lag-only
                        elif v_cmd_filt is None:
                            v_cmd_filt = v_raw
                        else:
                            _al = min(1.0, DT_SUB_LEN / FILTER_TAU)
                            v_cmd_filt = v_cmd_filt + _al * (v_raw - v_cmd_filt)
                        _spd_f = float(np.linalg.norm(v_cmd_filt))
                        if _spd_f > 1e-9 and sp > 0.3:
                            _dir_f = v_cmd_filt / _spd_f
                            cos_a = float(np.dot(vehicle.vel, _dir_f) / sp)
                            cos_a = max(-1.0, min(1.0, cos_a))
                            turn_scale = _turn_speed_scale(
                                math.acos(cos_a), TURN_SHARPNESS)
                        else:
                            turn_scale = 1.0
                        v_cmd = v_cmd_filt * turn_scale
                        # re-apply the creep floor after scaling (a full
                        # reversal must not command a dead stop)
                        _n = float(np.linalg.norm(v_cmd))
                        if _n < V_MIN_CREEP and _n > 1e-12:
                            v_cmd = v_cmd / _n * V_MIN_CREEP
                        use_servo = True
                        if dbg:
                            dbg.write(json.dumps({
                                "step": step, "sub": tsu_sub, "t": round(t_sim, 2),
                                "event": "v2b_cmd",
                                "d_hat": [round(float(c), 4) for c in d_hat],
                                "v_cmd": [round(float(c), 4) for c in v_cmd],
                                "turn_scale": round(float(turn_scale), 3),
                                "sp": round(float(sp), 3),
                            }) + "\n")
                    # Trajectory-level D5: the swept path must be approved,
                    # not just the endpoint (inertia makes the path matter).
                    if use_servo:
                        swept_ok = _vel_swept_ok(vehicle.pos, v_cmd, pos_v,
                                                marg, obs_mask, shape)
                    else:
                        swept_ok = _traj_passes_d5(seg, pos_v, marg,
                                                  obs_mask, shape)
                    if not swept_ok:
                        _refused = True
                        if dbg:
                            dbg.write(json.dumps(
                                {"step": step, "sub": tsu_sub,
                                 "event": "traj_refused"}) + "\n")
                if _refused:
                    log["n_move_refusals"] += 1
                    s16p.notify_refusal(next_v, pos_v, marg, obs_mask)
                    # Phase 2b: a refusal (hover/brake) is a command
                    # boundary — drop the filter history so the first
                    # post-hover command is obeyed promptly, not dragged
                    # by stale pre-refusal velocity.
                    v_cmd_filt = None
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
                elif use_servo:
                    # Pure velocity servo: p_des = pos zeroes the outer P
                    # term; the inner PID tracks v_cmd directly.
                    a_cmd = pid.command(vehicle.pos, v_cmd,
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
        print(f"  [s16-dyn4-{backend}] {mission_id}: {log['outcome']} "
              f"(wall {log['wall_time']:.1f}s, sim {log['sim_time']:.1f}s, "
              f"tsu={log['n_tsu_decisions']}, cbf_brakes={n_cbf_brakes})")

    row = manifest_row(log)
    row["n_escapes"] = s16p.n_escapes
    row["n_d5_refusals_s16"] = s16p.n_d5_refusals
    row["n_escape_replans"] = s16p.n_escape_replans
    row["n_tsu_decisions"] = log["n_tsu_decisions"]
    row["total_tsu_time"] = round(log["total_tsu_time"], 3)
    row["wind_steady"] = cfg.get("wind_steady", 0.0)
    row["wind_gust"] = int(bool(cfg.get("wind_gust", False)))
    row["attitude_lag"] = int(bool(cfg.get("attitude_lag", False)))
    row["gravity_env"] = int(bool(cfg.get("gravity_envelope", False)))
    row["tw_ratio"] = cfg.get("tw_ratio", TW_PRIMARY)
    row["tau_tilt"] = cfg.get("tau_tilt", TAU_TILT) \
        if cfg.get("attitude_lag") else 0.0
    row["filter_tau"] = "" if FILTER_TAU is None else FILTER_TAU
    row["config_name"] = cfg.get("config_name", "")
    _ob = cfg.get("obscurant") or {}
    row["obsc_archetype"] = _ob.get("archetype", "fog")
    row["obsc_density"] = _ob.get("density", "clear")
    row["obsc_sigma"] = obscurant.get_config()["sigma"]
    return row

MANIFEST_FIELDS_V3 = list(MANIFEST_FIELDS_V2) + ["tier", "config_name",
    "wind_steady", "wind_gust", "attitude_lag", "gravity_env", "tw_ratio",
    "tau_tilt", "filter_tau"]
MANIFEST_FIELDS_V4 = MANIFEST_FIELDS_V3 + ["obsc_archetype", "obsc_density",
    "obsc_sigma", "backend"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="S16 missions with dynamics")
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--noise", type=float, nargs="*", default=[0.15, 0.30])
    ap.add_argument("--backend", default="thrml", choices=["thrml", "numpy"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--manifest", default="RUN_MANIFEST_DYN_V3.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--no-traj", action="store_true",
                    help="skip trajectory JSON writing")
    args = ap.parse_args()

    from head_to_head import list_rooms

    outdir = args.outdir or os.path.join(_BASE, "dynamics_sim")
    tdir = os.path.join(outdir, "telemetry_dyn2")
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
    writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS_V3)
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
