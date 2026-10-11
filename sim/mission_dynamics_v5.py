"""S16 mission loop with real dynamics — Phase 5: OBSCURANT FIXES.

Three fixes for Phase 4's silent D5-blindness (see obscurant_fixes.py):
FIX 1 attenuation-aware D5 (trust evidence), FIX 2 phantom confirmation
gating, FIX 3 low-vis executive (stop-and-hover). Config flags:
fix_atten_d5, fix_confirm, fix_lowvis. v4 file untouched; this is the
A/B comparison branch.

Phase 4 docstring follows (v4 = PARTICLE OBSCURANT).

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
import obscurant_fixes as ofix
import planner
import controller
import classical
import s16 as s16_module
import escalation as esc_mod
import battery as batt_mod
import battery_safety as bsafe_mod

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

TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_fix")
_TRACKER = None      # per-run EvidenceTracker (obscurant fixes)
_FIX1 = False         # attenuation-aware D5
_FIX2 = False         # phantom confirmation gating
_FIX3 = False         # low-vis executive (hover)
_BATTERY = None       # per-run BatteryModel (battery layer, graduated
                      # 2026-10-11)
_SAFETY = None        # per-run BatterySafetyController (Paul's design)
INNER_PER_SUBSTEP = 10          # 100 Hz dynamics, 10 Hz decisions
TRAJ_RECORD_EVERY = 5           # 20 Hz demo feed (contract dt=0.05)


def nav_trustworthy_stub(cfg, t, pose):
    """EXPLICIT STUB for the nav-health trust signal (see
    battery_safety.py CHECKPOINT SPEC). The nav-health monitor
    defines the signal as `pose_trustworthy` — True only when
    nav-health is HEALTHY — but that monitor has NOT graduated to
    ThermoNav yet (it graduates separately; wiring it in place of
    this stub is a recorded follow-up). This slot takes
    cfg["nav_trustworthy"]: a bool, or a callable (t_s, pose) ->
    bool with exactly those semantics. Absent the slot it returns
    False — the shipped status quo (pose_trustworthy hardcoded
    False) — so no in-flight checkpoint is recorded and RTC
    coincides with RTB. This is an interface placeholder, NOT a
    health estimate."""
    src = (cfg or {}).get("nav_trustworthy")
    if callable(src):
        return bool(src(float(t), tuple(float(c) for c in pose)))
    if src is not None:
        return bool(src)
    return False


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
        if not ofix.safe_to_enter_fixes(v, marg, obs_mask, pos_v, _TRACKER, _FIX1):
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
        if not ofix.safe_to_enter_fixes(v, marg, obs_mask, pos_v, _TRACKER, _FIX1):
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


# Boundary-prior logodds written onto every boundary voxel (including
# the whole zz=0 floor layer) at mission start in run_mission_dynamics.
# A floor voxel whose accumulator still reads exactly this value has NO
# scan evidence either way; hits push it up (+0.60 confirmed / +0.18
# provisional per hit), free-space traversals pull it down (-0.73).
FLOOR_PRIOR_LOGODDS = 2.0


def _below_state(pose, marg, obs_mask, shape, radius=1, *, logodds):
    """State of the vertical column below the drone in the CURRENT
    perceived map: 0 = known-clear (every voxel observed AND
    P(occ) < P_OBSTACLE), 1 = unknown (some voxel unobserved),
    2 = occupied (some voxel believed occupied).

    Floor layer zz=0 is keyed on EVIDENCE BEYOND THE BOUNDARY PRIOR
    (battery RT-1, red-team round 1, 2026-10-11; graduated with the
    battery layer). History: the original guard read zz=0's marg at
    face value — the prior alone gives ~0.88, so every column read
    occupied and LAND was dead code (battery Finding 1; the same
    unsatisfiability LINK_MODEL.md records as known limit 6 for the
    link layer). The first fix excluded zz=0 entirely — blind to
    real obstacles that occupy ONLY zz=0 (159 GT columns in 6/30
    corpus rooms). Correct key: a zz=0 voxel counts as OCCUPIED only
    if its logodds accumulator exceeds the prior value, i.e. it
    carries hit support beyond the prior; at or below the prior it
    is the landing surface and is ignored (neither blocks nor
    clears). zz>=1 voxels are checked as before. The checked range
    includes zz=0, so at voxel z=1 the set is {zz=0} — never empty
    (the RT-1 endpoint hole).

    `logodds` is a REQUIRED keyword argument (RT2-4): the previous
    `logodds=None` default silently reverted to the RT-1-blind
    surface reading for any caller that omitted it. Pass None
    explicitly if a surface reading is really what is wanted.
    """
    try:
        v = planner.pose_to_voxel(pose, shape)
    except ValueError:
        return 1
    x, y, z = int(v[0]), int(v[1]), int(v[2])
    state = 0
    for dz in range(1, z + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx, yy, zz = x + dx, y + dy, z - dz
                if not (0 <= xx < shape[0] and 0 <= yy < shape[1]
                        and 0 <= zz < shape[2]):
                    continue
                vv = (xx, yy, zz)
                if zz == 0:
                    # Floor layer: evidence-keyed (see docstring). At
                    # or below the prior it is the landing surface —
                    # skip it entirely. Above the prior = hit support
                    # beyond the prior = a real floor-layer obstacle.
                    if logodds is not None and float(logodds[vv]) \
                            > FLOOR_PRIOR_LOGODDS + 1e-6:
                        return 2  # hit-supported floor obstacle
                    continue
                if not bool(obs_mask[vv]):
                    state = max(state, 1)  # unknown below
                elif float(marg[vv]) >= planner.P_OBSTACLE:
                    return 2  # occupied below
    return state


def _floor_evidence(pose, logodds, shape, radius=1):
    """True iff EVERY floor-layer (zz=0) voxel in the footprint under
    the drone carries scan evidence — its accumulator differs from the
    boundary prior (hits push up, free-space traversals pull down;
    either is evidence the floor there has actually been sensed). A
    footprint still reading exactly the prior has never been seen:
    the descent into it is unverified, however clear zz>=1 reads."""
    if logodds is None:
        return False
    try:
        v = planner.pose_to_voxel(pose, shape)
    except ValueError:
        return False
    x, y = int(v[0]), int(v[1])
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xx, yy = x + dx, y + dy
            if not (0 <= xx < shape[0] and 0 <= yy < shape[1]):
                continue
            if abs(float(logodds[xx, yy, 0]) - FLOOR_PRIOR_LOGODDS) \
                    <= 1e-6:
                return False
    return True


def _below_clear(pose, marg, obs_mask, shape, radius=1, logodds=None):
    """Guard for escalation land-in-place: is the vertical column below the
    drone known-clear in the CURRENT perceived map?

    Measured 2026-10-10: blind descent in whiteout collided 4/16 with
    unmapped obstacles directly below the hover point. The industry "land"
    rung assumes terrain awareness; without a downward sensor, we only
    descend into space the current map marks observed AND clear
    (P(occ) < P_OBSTACLE). Unknown or occupied below -> hold (HOVER).

    This is an actuation guard in the loop, not policy logic: the policy
    (red-team PASS) says "land"; the loop lands safely or holds if it
    can't verify the ground. Documented, not silent (esc_land_vetoed).

    Battery-critical / safety-layer exceptions live at the call site,
    not here: this function still means known-clear.
    """
    return _below_state(pose, marg, obs_mask, shape, radius,
                        logodds=logodds) == 0


def _touchdown_disarm_check(policy, tracker, map_age, cfg, t_sim, log):
    """Final DISARM check at touchdown — the loop's touchdown path.

    Called ONLY from the loop's touchdown branch, which has already
    verified:
      - _esc_land is True: the loop executed the land behavior THIS step
        (post below-clear veto), so the policy's _current is LAND.
        A stale LAND in the policy with the loop hovering can never reach
        here — the branch condition gates on the loop's own action.
      - vehicle.pos[2] <= TOUCHDOWN_Z_M: the vehicle is on the ground.

    Makes a final synchronous policy.evaluate(landed=True). If the policy
    emits DISARM, calls disarm_behavior (outcome "landed_disarmed") and
    returns True. Otherwise returns False and the caller falls back to
    outcome="landed".

    Defense in depth: even if this were called with _current != LAND, the
    policy vetoes DISARM loudly (disarm_vetoed_not_landing) and returns
    non-DISARM — the helper then returns False. DISARM cannot fire from a
    stale policy state. If the policy is absent (escalation off), returns
    False.
    """
    if policy is None:
        return False
    _d, _ = policy.evaluate(
        tracker.vis_smooth if tracker is not None else 1.0,
        tracker.far_smooth if tracker is not None else 1.0,
        map_age, bool(cfg.get("jitter")), 1.0,
        pose_trustworthy=False, t=t_sim, landed=True)
    if _d == esc_mod.DISARM:
        esc_mod.EscalationPolicy.disarm_behavior(log)
        return True
    return False


def _loop_touchdown_branch(_esc_land, vehicle, _ESCALATION, _TRACKER,
                           map_age, cfg, t_sim, log):
    """The loop's land-in-place touchdown branch — factored verbatim from
    the escalation hold-executive in run_mission_dynamics (was inline).

    Fires only when:
      - _esc_land is True: the loop executed the land behavior THIS step
        (post below-clear veto), so the policy's _current is LAND. A stale
        LAND in the policy with the loop hovering can never reach here —
        the branch condition gates on the loop's own action.
      - vehicle.pos[2] <= TOUCHDOWN_Z_M: touchdown reached.

    On fire: final synchronous DISARM check via _touchdown_disarm_check;
    outcome "landed_disarmed" if DISARM fires, else fallback "landed".
    Returns True when the branch fired (caller sets outcome_set and
    breaks), False otherwise.
    """
    if _esc_land and vehicle.pos[2] <= esc_mod.TOUCHDOWN_Z_M:
        # Land-in-place complete: the loop executed the land
        # behavior this step (_esc_land is True post below-clear
        # veto), so the policy's _current is LAND — no
        # stale-state hazard (see _touchdown_disarm_check).
        # Final synchronous DISARM check; fallback to "landed".
        if not _touchdown_disarm_check(
                _ESCALATION, _TRACKER, map_age, cfg, t_sim, log):
            log["outcome"] = "landed"
        return True
    return False


def run_mission_dynamics(occ, room_id, seed, noise_val, mission_id,
                         backend="thrml", telemetry_dir=None, verbose=False,
                         progress=False, write_traj=True, debug_log=None,
                         cfg=None):
    """Fly one S16 mission with dynamics. Returns manifest row dict."""
    t_mission = time.time()
    shape = occ.shape
    compute_marginals, make_s2 = _get_backend(backend)

    # ---- Phase 3 per-run physics config ----
    global A_BRAKE, A_PLAN, FILTER_TAU, _TRACKER, _FIX1, _FIX2, _FIX3, _FIX4
    global _ESCALATION, _BATTERY, _SAFETY
    cfg = dict(cfg or {})
    _FIX1 = bool(cfg.get("fix_atten_d5", False))
    _FIX2 = bool(cfg.get("fix_confirm", False))
    _FIX3 = bool(cfg.get("fix_lowvis", False))
    _FIX4 = bool(cfg.get("fix_stale", False))
    _TRACKER = ofix.EvidenceTracker(shape) \
        if (_FIX1 or _FIX2 or _FIX3 or _FIX4) else None
    # Escalation layer: Option A industry-mirror policy. Off unless
    # cfg["escalation"] is set. The policy is host-side classical
    # (red-team PASS 2026-10-10); the decider kernel is exercised by
    # test_escalation.py, not the mission loop.
    # pose_trustworthy is hardcoded False (estimator: open problem) —
    # RTH degrades to land-in-place inside the policy.
    _ESCALATION = esc_mod.EscalationPolicy() if cfg.get("escalation") else None
    # Battery model (battery.py): classical host model, simulation
    # only. Off unless cfg["battery"] is set. cfg["battery_start_soc"]
    # scripts low-charge missions (<1.0). NOTE (research-repo RT-7):
    # with cfg["battery"] set but cfg["escalation"] unset, the battery
    # still steps and fills the batt_* manifest fields, but nothing
    # consumes its state — telemetry only, no ladder action. The
    # escalation ladder / safety layer are the only consumers, by
    # design.
    _BATTERY = batt_mod.BatteryModel(
        start_soc=float(cfg.get("battery_start_soc", 1.0))) \
        if cfg.get("battery") else None
    # Battery safety layer (Paul's design 2026-10-11, battery_safety.py):
    # one flag at RESERVE, measured-draw countdown, RTB -> RTC ->
    # land-in-place cascade. Off unless cfg["battery_safety"] is set;
    # requires the battery model (it owns the battery rows when on —
    # the escalation policy is then called WITHOUT battery inputs).
    _SAFETY = None  # created after pick_start_goal (needs base pose)
    if cfg.get("battery_safety") and _BATTERY is None:
        raise ValueError("battery_safety requires cfg['battery']")
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
    if cfg.get("battery_safety"):
        # BASE = the takeoff pose (start_pose). The checkpoint starts
        # as base and is refreshed from the flight itself while the
        # nav-trust stub reports trustworthy (battery_safety.py spec).
        _SAFETY = bsafe_mod.BatterySafetyController(start_pose)
    _mission_goal_pose = goal_pose  # original goal (safety may retarget)
    _return_target = None           # active RTB/RTC target pose, if any
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
        if _TRACKER is not None:
            _TRACKER.update(scans, fix1=_FIX1, fix2=_FIX2, fix3=_FIX3,
                            n_refusals=log["n_move_refusals"])
            _, _, n_skipped = ofix.integrate_scan_fixes(
                logodds, shape, scans, _TRACKER, max_range=40.0,
                obs_count=obs_count, fix2=_FIX2)
        else:
            _, _, n_skipped = obscurant.integrate_scan(
                logodds, shape, scans, max_range=40.0, obs_count=obs_count)
        log["n_rays_skipped"] = log.get("n_rays_skipped", 0) + n_skipped
        hover_mode = bool(_TRACKER is not None and (
            (_FIX3 and _TRACKER.hover) or (_FIX4 and _TRACKER.stale_hover)))
        log["n_hover_scans"] = log.get("n_hover_scans", 0) + int(hover_mode)
        log["vis_min"] = min(log.get("vis_min", 1.0),
                             _TRACKER.vis_smooth if _TRACKER else 1.0)

        # ---- 1 Hz host: Gibbs -> P(occupied) ----
        h = sensor.quantize_8bit(logodds)
        marg, dt_gibbs, x_last = compute_marginals(
            logodds, h, x_last, seed, step)
        log["total_gibbs_time"] += dt_gibbs
        obs_mask = effective_observed_mask(logodds, obs_count, "tsu")

        # FIX 4: map staleness (sim-seconds since last fresh publish).
        # ThermoNav has no jitter controller (lab test infrastructure);
        # map_age is 0.0 here — the gate never fires, correctly. The
        # wiring is preserved so the escalation policy's stale input
        # stays live if a staleness source is added later.
        map_age = 0.0
        if _TRACKER is not None:
            _TRACKER.update_stale(map_age, fix4=_FIX4)

        # ---- Battery (graduated 2026-10-11) ----
        # Stepped once per outer step (dt=1.0 s) at the load implied by
        # the vehicle's CURRENT velocity — its own onboard state, no
        # ground truth. The 1 Hz power sample is an approximation of
        # the second just flown (battery state moves slowly vs 1 Hz).
        _batt_state = None
        if _BATTERY is not None:
            _p = _BATTERY.power_for_velocity(vehicle.vel)
            # Scripted extra load [scenario lever, e.g. a payload
            # heater switching on]: cfg["battery_extra_load_w"] is a
            # float or callable(t_s) -> W added to the REAL load the
            # pack delivers. Decision code never reads it — its only
            # path into decisions is the stepped gauge telemetry
            # (soc / Wh consumed) that the safety meter measures,
            # exactly as a real extra load would present.
            _xl = cfg.get("battery_extra_load_w", 0.0)
            _p += float(_xl(t_sim)) if callable(_xl) else float(_xl)
            _batt_state = _BATTERY.step(1.0, _p)
            log["batt_phase"] = batt_mod.classify_phase(vehicle.vel)
            if _BATTERY.soc < log.get("batt_min_soc", 1.0):
                log["batt_min_soc"] = _BATTERY.soc

        # ---- Battery safety layer (Paul's design, battery_safety.py) --
        # Observed EVERY step (meter window + checkpoint record are
        # flight-long), stepped after the battery. When ON it owns
        # the battery decision: the escalation policy below is called
        # without battery inputs so the two layers never both act.
        _safety_action = bsafe_mod.NONE
        _land_terminal = False
        if _SAFETY is not None:
            _SAFETY.observe(
                t_sim, _BATTERY.soc, _BATTERY.wh_used, pose,
                nav_trustworthy_stub(cfg, t_sim, pose))
            _safety_action, _sinfo = _SAFETY.step(
                t_sim, _BATTERY.soc, pose)
            log["safety_rung"] = _sinfo.get("rung") or ""
            log["safety_n_events"] = len(_SAFETY.events)

        # ---- Escalation policy (Option A, industry mirror) ----
        # Integration contract (escalation.py): evaluated per outer step,
        # AFTER tracker.update_stale(). Final action = max(shipped,
        # escalation) by severity — escalation only ADDS beyond hover,
        # never contradicts FIX 3 / FIX 4. dt = 1.0 s (outer step rate).
        _esc_action = esc_mod.HOVER if hover_mode else esc_mod.CONTINUE
        if _ESCALATION is not None:
            _a, _ = _ESCALATION.evaluate(
                _TRACKER.vis_smooth if _TRACKER is not None else 1.0,
                _TRACKER.far_smooth if _TRACKER is not None else 1.0,
                map_age, False, 1.0,
                pose_trustworthy=False, t=t_sim,
                battery_soc=None if _SAFETY is not None else (
                    _batt_state["soc"] if _batt_state else None),
                battery_ttr_s=None if _SAFETY is not None else (
                    _batt_state["time_to_reserve_s"]
                    if _batt_state else None))
            _esc_action = max(_esc_action, _a)
            if _a > (esc_mod.HOVER if hover_mode else esc_mod.CONTINUE):
                log["esc_engaged_scans"] = log.get("esc_engaged_scans", 0) + 1
        # ---- Safety-layer action mapping (Paul's cascade) -------------
        # RTB/RTC ride the existing return mechanism: the mission goal
        # is retargeted to the snapshot target and the normal S2/A*
        # stack flies it; arrival converts to LAND. The original goal
        # is NOT restored — the flag ends the mission, by design.
        # The mapping OVERWRITES the escalation action on terminal
        # steps: DISARM (severity 4) cannot outrank a terminal battery
        # landing — the descent happens first, and DISARM still fires
        # afterwards via the loop's touchdown branch once touchdown
        # is confirmed.
        if _SAFETY is not None and _safety_action != bsafe_mod.NONE:
            if _safety_action in (bsafe_mod.RTB, bsafe_mod.RTC):
                _tgt = _SAFETY.target_pose
                if _tgt is not None and _return_target != _tgt:
                    _return_target = _tgt
                    goal_pose = tuple(float(c) for c in _tgt)
                    goal_v = planner.pose_to_voxel(goal_pose, shape)
                    log["safety_return_legs"] = \
                        log.get("safety_return_legs", 0) + 1
                if _return_target is not None and \
                        controller.distance(pose, _return_target) \
                        <= GOAL_TOL:
                    _esc_action = esc_mod.LAND  # arrived: land there
                    log["safety_arrived"] = \
                        log.get("safety_arrived", 0) + 1
            elif _safety_action == bsafe_mod.HOLD_METERING:
                _esc_action = max(_esc_action, esc_mod.HOVER)
            elif _safety_action == bsafe_mod.LAND_IN_PLACE:
                _esc_action = max(_esc_action, esc_mod.LAND)
            elif _safety_action == bsafe_mod.LAND_TERMINAL:
                # Countdown end: NOTHING outranks the flag. The guard
                # block below is bypassed for this action entirely.
                _esc_action = esc_mod.LAND
                _land_terminal = True
                log["safety_terminal_steps"] = \
                    log.get("safety_terminal_steps", 0) + 1
        if _esc_action > log.get("esc_max_sev", 0):
            log["esc_max_sev"] = _esc_action
        # LAND precondition (actuation guard, not policy): the vertical
        # column below must be known-clear in the current perceived map.
        # Blind descent collided 4/16 in whiteout (measured 2026-10-10).
        # If below is unknown/occupied, degrade to HOVER (hold) — loudly.
        _land_rate = esc_mod.LAND_DESCENT_RATE
        if _esc_action == esc_mod.LAND and (
                _ESCALATION is not None or _SAFETY is not None):
            _bst = _below_state(pose, marg, obs_mask, shape,
                                logodds=logodds)
            # Descent profile (battery RT-1): fully verified (guard
            # clear AND floor evidence present) -> full rate at any
            # altitude. Anything less -> creep inside the
            # final-approach band.
            _verified = (_bst == 0
                         and _floor_evidence(pose, logodds, shape))
            if not _verified \
                    and pose[2] <= esc_mod.LAND_CREEP_ALT_M:
                _land_rate = esc_mod.LAND_CREEP_RATE
            # Battery-critical forced landing: at soc <= CRITICAL the
            # alternative to descending is hovering until the pack is
            # empty and dropping uncontrolled — the industry
            # critical-battery forced-land case. The veto is
            # therefore relaxed for UNKNOWN below only (descend,
            # logged as esc_batt_land_unverified). KNOWN-OCCUPIED
            # below still vetoes at any charge.
            # With the safety layer ON, its countdown rule supersedes
            # this soc-keyed relaxation: the veto (unknown OR
            # occupied) may hold throughout the countdown window, and
            # at countdown end _land_terminal bypasses the guard
            # entirely (Paul's design: veto may hold during the
            # countdown, never at its end). With the layer OFF, the
            # legacy relaxation below applies.
            _batt_crit = _BATTERY is not None and _SAFETY is None \
                and _BATTERY.soc <= batt_mod.CRITICAL_SOC
            if _land_terminal:
                log["esc_batt_terminal_land"] = \
                    log.get("esc_batt_terminal_land", 0) + 1
            elif _bst == 2 or (_bst == 1 and not _batt_crit):
                _esc_action = esc_mod.HOVER
                log["esc_land_vetoed"] = log.get("esc_land_vetoed", 0) + 1
            elif _bst == 1 and _batt_crit:
                log["esc_batt_land_unverified"] = \
                    log.get("esc_batt_land_unverified", 0) + 1

        # ---- D3: goal-blocked check ----
        # Skipped on a safety return leg: the retargeted goal is a
        # return target (base/checkpoint), not a mission goal — a
        # believed-occupied return voxel must not end the flight as
        # "goal_blocked" (measured in the research repo: whiteout_term
        # ended at step ~15 when phantom evidence marked the base
        # voxel occupied).
        if (_return_target is None
                and controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
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
            # ---- Escalation behavior (Option A) ----
            # _esc_action = max(shipped, escalation) by severity. DISARM is
            # terminal stand-down; LAND descends in place; HOVER holds
            # (shipped FIX 3/4 or escalation hover). RTH is unreachable —
            # the gate (pose_trustworthy=False) converts RTH->LAND inside
            # the policy; the fallback below is loud and fail-safe.
            if _esc_action == esc_mod.DISARM:
                esc_mod.EscalationPolicy.disarm_behavior(log)
                outcome_set = True
                break
            _esc_land = (_esc_action == esc_mod.LAND)
            # RTH is unreachable today: the policy's gate
            # (pose_trustworthy=False) converts RTH->LAND internally, so the
            # loop never sees RTH. Kept as defense-in-depth: if the gate ever
            # opens (estimator solved), an RTH reaching the loop means
            # "return home" but the loop has no homing behavior — land in
            # place (fail-safe) rather than hover forever. Loud in the log
            # so the unexpected path is visible.
            if _esc_action == esc_mod.RTH:
                log["steps"].append(
                    {"step": step, "event": "esc_rth_unreachable_fallback"})
                _esc_land = True
            if _esc_action >= esc_mod.HOVER:
                # Hold executive: hold position (FIX 3/4 hover or escalation
                # hover) or descend (escalation land-in-place). Keeps the
                # 100 Hz dynamics + collision check running, skips S2/D5.
                # Mapping (scan + Gibbs above) continues every outer step.
                # NOTE: with escalation off, _esc_action == HOVER iff
                # hover_mode, _esc_land is False, and _tgt == vehicle.pos —
                # bit-identical to the shipped hover branch.
                for _ in range(INNER_PER_SUBSTEP):
                    if _esc_land:
                        # Land-in-place: velocity-servo descent.
                        # p_des=pos zeroes the outer P term; the inner PID
                        # tracks v_des=(0,0,-rate) directly. Rate is the
                        # guard block's _land_rate: full rate when the
                        # descent is verified, creep rate in the
                        # final-approach band when it is not (RT-1).
                        a_cmd = pid.command(
                            vehicle.pos,
                            (0, 0, -_land_rate),
                            vehicle.pos, vehicle.vel)
                    else:
                        a_cmd = pid.command(vehicle.pos, (0, 0, 0),
                                            vehicle.pos, vehicle.vel)
                    p_prev = vehicle.pos.copy()
                    vehicle.step(a_cmd)
                    t_sim += DT
                    inner_tick += 1
                    if _segment_hits_occ(p_prev, vehicle.pos, occ):
                        log["outcome"] = "collision"
                        outcome_set = True
                        break
                    if inner_tick % TRAJ_RECORD_EVERY == 0:
                        _record()
                if outcome_set:
                    break
                if _loop_touchdown_branch(
                        _esc_land, vehicle, _ESCALATION, _TRACKER,
                        map_age, cfg, t_sim, log):
                    outcome_set = True
                    break
                if not _esc_land and _return_target is None \
                        and controller.distance(
                        tuple(float(c) for c in vehicle.pos),
                        goal_pose) < GOAL_TOL:
                    log["outcome"] = "success"
                    outcome_set = True
                    break
                continue
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
            d5_ok = ofix.safe_to_enter_fixes(next_v, marg, obs_mask, pos_v,
                                             _TRACKER, _FIX1)
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

            # On a safety return leg, reaching the (retargeted) goal
            # is NOT mission success: the safety mapping converts
            # arrival to LAND at the next outer step.
            if _return_target is None and controller.distance(
                    tuple(float(c) for c in vehicle.pos),
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
    row["fix_atten_d5"] = int(_FIX1)
    row["fix_confirm"] = int(_FIX2)
    row["fix_lowvis"] = int(_FIX3)
    row["fix_stale"] = int(_FIX4)
    row["n_hover_scans"] = log.get("n_hover_scans", 0)
    row["vis_min"] = round(log.get("vis_min", 1.0), 3)
    # ---- Escalation layer (Option A, industry mirror) ----
    row["esc_enabled"] = int(_ESCALATION is not None)
    row["esc_final_action"] = esc_mod.ACTION_NAMES[log.get("esc_max_sev", 0)]
    row["esc_engaged_scans"] = log.get("esc_engaged_scans", 0)
    row["esc_n_events"] = len(_ESCALATION.log) \
        if _ESCALATION is not None else 0
    row["esc_rth_gated"] = sum(
        1 for _, e, _ in _ESCALATION.log if e == "rth_gated_land_in_place") \
        if _ESCALATION is not None else 0
    row["esc_land_vetoed"] = log.get("esc_land_vetoed", 0)
    row["esc_batt_land_unverified"] = log.get(
        "esc_batt_land_unverified", 0)
    # ---- Battery layer (graduated 2026-10-11) ----
    row["batt_enabled"] = int(_BATTERY is not None)
    row["batt_start_soc"] = round(_BATTERY.start_soc, 4) \
        if _BATTERY is not None else -1
    row["batt_final_soc"] = round(_BATTERY.soc, 4) \
        if _BATTERY is not None else -1
    row["batt_min_soc"] = round(log.get("batt_min_soc", 1.0), 4) \
        if _BATTERY is not None else -1
    row["batt_min_voltage"] = round(_BATTERY.min_voltage, 2) \
        if _BATTERY is not None else -1
    row["batt_wh_used"] = round(_BATTERY.wh_used, 3) \
        if _BATTERY is not None else -1
    row["batt_final_ttr_s"] = round(_BATTERY.time_to_reserve_s(), 1) \
        if _BATTERY is not None else -1
    row["batt_reserve_crossed"] = int(
        _BATTERY is not None
        and log.get("batt_min_soc", 1.0) <= batt_mod.RESERVE_SOC)
    row["batt_critical_crossed"] = int(
        _BATTERY is not None
        and log.get("batt_min_soc", 1.0) <= batt_mod.CRITICAL_SOC)
    # ---- Battery safety layer (Paul's design 2026-10-11) ----
    row["batt_safety_enabled"] = int(_SAFETY is not None)
    row["safety_flag_fired"] = int(
        _SAFETY is not None and _SAFETY.flag_fired)
    row["safety_rung"] = log.get("safety_rung", "") \
        if _SAFETY is not None else ""
    row["safety_countdown_s"] = (
        round(_SAFETY.countdown_s, 1)
        if _SAFETY is not None and _SAFETY.countdown_s is not None
        else -1)
    row["safety_measured_draw_w"] = (
        round(_SAFETY.meter.measured_draw_w(), 1)
        if _SAFETY is not None
        and _SAFETY.meter.measured_draw_w() is not None else -1)
    row["safety_n_abort_down"] = _SAFETY.n_abort_down \
        if _SAFETY is not None else 0
    row["safety_terminal_steps"] = log.get("safety_terminal_steps", 0)
    row["safety_arrived_steps"] = log.get("safety_arrived", 0)
    row["safety_return_legs"] = log.get("safety_return_legs", 0)
    row["safety_checkpoint"] = ",".join(
        f"{c:.2f}" for c in _SAFETY.tracker.checkpoint_pose) \
        if _SAFETY is not None else ""
    row["esc_batt_terminal_land"] = log.get("esc_batt_terminal_land", 0)
    # Full event + calibration records (consumed by the safety
    # validation runner; fixed-field CSV writers ignore extras, as
    # with the esc_*/batt_* fields above).
    row["safety_events"] = _SAFETY.events if _SAFETY is not None else []
    row["safety_calibration_log"] = _SAFETY.calibration_log \
        if _SAFETY is not None else []
    return row

MANIFEST_FIELDS_V3 = list(MANIFEST_FIELDS_V2) + ["tier", "config_name",
    "wind_steady", "wind_gust", "attitude_lag", "gravity_env", "tw_ratio",
    "tau_tilt", "filter_tau"]
MANIFEST_FIELDS_V4 = MANIFEST_FIELDS_V3 + ["obsc_archetype", "obsc_density",
    "obsc_sigma", "backend"]
MANIFEST_FIELDS_V5 = MANIFEST_FIELDS_V4 + ["fix_atten_d5", "fix_confirm",
    "fix_lowvis", "fix_stale", "n_hover_scans", "vis_min",
    "esc_enabled", "esc_final_action", "esc_engaged_scans", "esc_n_events",
    "esc_rth_gated", "esc_land_vetoed"]


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
