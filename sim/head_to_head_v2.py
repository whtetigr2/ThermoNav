"""Head-to-head V2: control vs TSU (perception) vs S16 (perception+planning).

Extends head_to_head.py with the S16 hierarchical planner integrated into
the main flight loop.

Configs:
  control: classical log-odds + A* + waypoint controller (unchanged)
  tsu:     Gibbs + weak-evidence mask + A* + waypoint controller (unchanged)
  s16:     Gibbs + weak-evidence mask + S16 (host A* + TSU S2 at 10 Hz)

The S16 loop per 1 Hz host step:
  1. Scan -> log-odds -> Gibbs -> P(occupied) [same as tsu]
  2. TSU fast loop (10 sub-steps):
     a. S2 picks next voxel toward host waypoint
     b. D5 safe_to_enter guard on proposed voxel
     c. Move 0.1 voxels toward voxel center
     d. Collision + goal checks per sub-step

Manifest: same fields as V1 plus n_tsu_decisions, total_tsu_time.
Output: head_to_head/RUN_MANIFEST_V2.csv
"""
import time
import json
import csv
import os
import sys
import numpy as np

import env
import sensor
import gibbs
import planner
import controller
import classical
import s16 as s16_module

from fly import (
    pick_start_goal, check_collision, optimal_len_to_tolerance,
    GOAL_TOL, GOAL_BLOCKED_RADIUS, MAX_MOVE_REFUSALS, START_GOAL_MIN_DIST,
    N_AZ, ELEV, MAX_STEPS, J_SMOOTH,
    N_SWEEPS_FULL, BURN_IN_FULL, N_SWEEPS_INCR, BURN_IN_INCR,
)
from head_to_head import (
    effective_observed_mask, compute_marginals, map_hash, manifest_row,
    MANIFEST_FIELDS, WEAK_OBS_MAX, STRONG_L_THRESH,
    REAL_LIDAR_DIR,
)

# S16 runs S2 at 10 Hz: 10 sub-steps per 1 Hz host step, 0.1 voxels each.
TSU_SUBSTEPS = 10
TSU_SPEED = 0.1  # voxels per sub-step (1.0 voxel/sec total, matches SPEED)
# D5 refusal budget scales with sub-steps: V1 allows 10 consecutive host-step
# refusals; S16 gets 10x sub-steps per host step, so 100 sub-step refusals.
MAX_MOVE_REFUSALS_S16 = MAX_MOVE_REFUSALS * TSU_SUBSTEPS

CONFIGS_V2 = ("control", "tsu", "s16")

MANIFEST_FIELDS_V2 = MANIFEST_FIELDS + ["n_tsu_decisions", "total_tsu_time",
                                        "n_escapes", "n_d5_refusals_s16",
                                        "n_escape_replans"]


def _now():
    """HH:MM:SS timestamp for progress logging."""
    return time.strftime("%H:%M:%S")


def run_mission_s16(occ, room_id, seed, noise_val, mission_id,
                    telemetry_dir=None, verbose=False, progress=False):
    """Fly one S16 mission. Returns a manifest row dict.

    Args:
        progress: if True, print timestamped step-level progress lines
            (scan, gibbs, tsu, move phases per host step). Use this to
            distinguish "slow but working" from "stuck in a loop" — if
            the log keeps updating, the mission is progressing.
    """
    t_mission = time.time()
    shape = occ.shape

    start_pose, goal_pose = pick_start_goal(occ, seed)
    pose = start_pose
    logodds = np.zeros(shape)
    obs_count = np.zeros(shape, dtype=np.int32)

    # Boundary prior (same as fly.py)
    logodds[0, :, :] = 2.0
    logodds[-1, :, :] = 2.0
    logodds[:, 0, :] = 2.0
    logodds[:, -1, :] = 2.0
    logodds[:, :, 0] = 2.0
    logodds[:, :, -1] = 2.0
    # Start/goal exemption: pick_start_goal guarantees GT-free voxels.
    # The boundary prior (P~0.88 occupied) would otherwise contradict the
    # mission definition at boundary goals, causing spurious goal_blocked.
    # This corrects a known-bad prior at setup time, not GT leakage during
    # flight — the drone must still navigate there on its beliefs.
    # (2026-10-06: 128_s5_n0.30 went success→goal_blocked after the F1
    # fix changed the approach trajectory; the boundary prior at the
    # y=0 goal dominated the belief.)
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
        "mission_id": mission_id,
        "config": "s16",
        "room_id": room_id,
        "seed": seed,
        "noise_val": noise_val,
        "start": start_pose,
        "goal": goal_pose,
        "optimal_path_len": optimal_len,
        "steps": [],
        "outcome": None,
        "n_plan_calls": 0,          # host A* replans
        "n_planning_failures": 0,   # host A* returned None
        "n_move_refusals": 0,       # D5 refusals
        "n_tsu_decisions": 0,       # S2 decisions
        "total_tsu_time": 0.0,
        "path_voxels": [],
        "collision_p_occ": None,
        "total_gibbs_time": 0.0,
        "total_plan_time": 0.0,
    }

    # S16 planner (host A* + TSU S2)
    # waypoint_lookahead=2: waypoints every 2 voxels, within S2's T=3 horizon.
    # (4 was too sparse — S2 couldn't see the waypoint.)
    s16p = s16_module.S16Planner(
        waypoint_lookahead=2, stuck_threshold=40,  # 40 sub-steps = 4 voxels
        s2_kwargs={"seed": seed, "n_sweeps": 200})
    # Note: stuck_threshold in sub-steps (0.1 voxels each). 40 = ~4 voxels
    # of no progress before fallback. The thermopilot default (4) was for
    # full-voxel steps.

    x_last = None
    goal_v = planner.pose_to_voxel(goal_pose, shape)
    marg = np.zeros(shape)

    for step in range(MAX_STEPS):
        t_step = time.time()
        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=scan_start "
                  f"pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f})",
                  flush=True)

        # ---- 1 Hz host: scan ----
        scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                 max_range=40.0, range_noise=noise_val,
                                 seed=1000 + seed * 100 + step)
        sensor.integrate_scan(logodds, shape, scans, max_range=40.0,
                              obs_count=obs_count)

        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=gibbs_start",
                  flush=True)

        # ---- 1 Hz host: Gibbs -> P(occupied) ----
        h = sensor.quantize_8bit(logodds)
        marg, dt_gibbs, x_last = compute_marginals(
            "tsu", logodds, h, x_last, seed, step)
        log["total_gibbs_time"] += dt_gibbs
        obs_mask = effective_observed_mask(logodds, obs_count, "tsu")

        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=gibbs_done "
                  f"dt={dt_gibbs:.2f}s", flush=True)

        # ---- D3: goal-blocked check ----
        if (controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
                and bool(obs_mask[goal_v])
                and float(marg[goal_v]) > planner.P_OBSTACLE):
            log["steps"].append({"step": step, "pose": pose,
                                 "event": "goal_blocked"})
            log["outcome"] = "goal_blocked"
            break

        # ---- 10 Hz TSU fast loop ----
        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=tsu_start "
                  f"substeps={TSU_SUBSTEPS}", flush=True)
        host_replans_before = s16p.n_host_replans
        tsu_time_step = 0.0
        for tsu_sub in range(TSU_SUBSTEPS):
            pos_v = planner.pose_to_voxel(pose, shape)
            next_v, s2info = s16p.tsu_act(
                pos_v, goal_v, marg, obs_mask, pos_cont=pose)
            log["n_tsu_decisions"] += 1
            log["total_tsu_time"] += s2info["tsu_time"]
            tsu_time_step += s2info["tsu_time"]

            # D5: predictive guard on the proposed voxel.
            # "Look before you leap" (2026-10-05): strict line-of-sight
            # check through the believed map for unobserved voxels.
            if not planner.safe_to_enter(next_v, marg, obs_mask, pos_v):
                log["n_move_refusals"] += 1
                # D5-refusal feedback (2026-10-05): tell S16 so it stops
                # re-proposing the same blocked voxel and escapes instead.
                s16p.notify_refusal(next_v, pos_v, marg, obs_mask)
                if log["n_move_refusals"] > MAX_MOVE_REFUSALS_S16:
                    log["outcome"] = "blocked"
                    break
                continue  # hover this sub-step

            # Move toward the chosen voxel's center, scaled by S16 speed_factor
            # (cautious mode moves slower; 2026-10-05 fix for hover pull-back).
            target = planner.voxel_to_pose(next_v)
            _sf = float(s2info.get("speed_factor", 1.0))
            pose = controller.move_toward(pose, target, speed=TSU_SPEED * _sf)
            v = planner.pose_to_voxel(pose, shape)
            if v not in log["path_voxels"][-1:] :
                log["path_voxels"].append(v)

            # Collision check (ground truth)
            if check_collision(pose, occ):
                log["collision_p_occ"] = float(marg[v])
                log["outcome"] = "collision"
                break

            # Goal check
            if controller.distance(pose, goal_pose) < GOAL_TOL:
                log["outcome"] = "success"
                break
        else:
            # Completed all 10 sub-steps without break: next host step
            log["n_plan_calls"] = s16p.n_host_replans
            log["n_planning_failures"] = s16p.n_host_failures
            log["steps"].append({"step": step, "n_tsu": 10})
            if progress:
                d_goal = controller.distance(pose, goal_pose)
                print(f"[{_now()}] {mission_id} step={step} phase=step_done "
                      f"tsu_dt={tsu_time_step:.2f}s "
                      f"dist_goal={d_goal:.1f} "
                      f"replans={s16p.n_host_replans} "
                      f"refusals={log['n_move_refusals']}",
                      flush=True)
            continue
        # Broke out of TSU loop (success/collision/blocked)
        log["n_plan_calls"] = s16p.n_host_replans
        log["n_planning_failures"] = s16p.n_host_failures
        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=outcome "
                  f"outcome={log['outcome']}", flush=True)
        break
    else:
        log["outcome"] = "max_steps"
        if progress:
            print(f"[{_now()}] {mission_id} phase=outcome outcome=max_steps",
                  flush=True)

    # --- mission metrics (same as V1) ---
    log["n_steps"] = len(log["steps"])
    log["wall_time"] = time.time() - t_mission
    log["flown_path_len"] = len(log["path_voxels"])
    log["final_map_hash"] = map_hash(marg)

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

    if verbose:
        print(f"  [s16] {mission_id}: {log['outcome']} "
              f"({log['wall_time']:.1f}s, "
              f"tsu={log['n_tsu_decisions']} decisions)")

    row = manifest_row(log)
    # Escape-system counters (2026-10-05): expose S16-internal stats.
    row["n_escapes"] = s16p.n_escapes
    row["n_d5_refusals_s16"] = s16p.n_d5_refusals
    row["n_escape_replans"] = s16p.n_escape_replans
    row["n_tsu_decisions"] = log["n_tsu_decisions"]
    row["total_tsu_time"] = round(log["total_tsu_time"], 3)
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Head-to-head V2: +S16")
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--noise", type=float, nargs="*", default=[0.15, 0.30])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS_V2),
                    choices=CONFIGS_V2)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--manifest", default="RUN_MANIFEST_V2.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tsu-only", action="store_true",
                    help="Only run s16 config (control/tsu already in V1)")
    ap.add_argument("--progress", action="store_true",
                    help="Print timestamped step-level progress per mission "
                         "(scan/gibbs/tsu phases). Use to distinguish "
                         "'slow but working' from 'stuck in a loop'.")
    args = ap.parse_args()

    # Import here to avoid circular import at module load
    from head_to_head import run_mission, list_rooms

    base = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(base, "head_to_head")
    tdir = os.path.join(outdir, "TELEMETRY_V2")
    mpath = os.path.join(outdir, args.manifest)
    os.makedirs(tdir, exist_ok=True)

    configs = ["s16"] if args.tsu_only else args.configs
    rooms = args.rooms or list_rooms()
    print(f"rooms: {len(rooms)}, seeds: {args.seeds}, noise: {args.noise}, "
          f"configs: {configs}")
    total = len(rooms) * len(args.seeds) * len(args.noise) * len(configs)
    print(f"total missions: {total}")

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
                    for config in configs:
                        mid = (f"{room_id}_s{seed}_n{noise_val:.2f}_{config}")
                        if mid in done:
                            continue
                        if config == "s16":
                            row = run_mission_s16(
                                occ, room_id, seed, noise_val, mid,
                                telemetry_dir=tdir, verbose=True,
                                progress=args.progress)
                        else:
                            # Reuse V1 runner for control/tsu (adds empty
                            # s16 fields for schema consistency)
                            row = run_mission(
                                config, occ, room_id, seed, noise_val, mid,
                                telemetry_dir=tdir, verbose=True)
                            row["n_tsu_decisions"] = 0
                            row["total_tsu_time"] = 0.0
                        writer.writerow(row)
                        fout.flush()
                        n_done += 1
                        done.add(mid)
    finally:
        fout.close()

    print(f"\ndone: {n_done} missions in {time.time()-t_all:.0f}s")
    print(f"manifest: {mpath}")


if __name__ == "__main__":
    main()
