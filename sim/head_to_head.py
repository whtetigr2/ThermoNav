"""Head-to-head benchmark: TSU stack vs classical control on real IndoorCD lidar data.

Plain English: two drone "brains" fly the exact same missions in the exact
same rooms. Same body, same sensors, same start/goal, same noise. The ONLY
difference is the mapper:

  CONTROL ("classical"): log-odds occupancy grid -> sigmoid -> P(occupied).
      Textbook OctoMap-style mapping. No Gibbs, no pairwise prior.
  TSU ("ours"): log-odds -> Ising fields -> Gibbs sampler (J=0.25) ->
      P(occupied), plus the weak-evidence fix (voxels seen only 1-2 times
      are treated as unknown, not blocked).

Everything else (A* planner, guarded controller, D5 safe_to_enter, D3
goal-blocked check) is IDENTICAL. "Brain swapped," as Paul put it.

This is SYNTHETIC SENSOR VALIDATION on REAL GEOMETRY. The rooms come from
real iPhone LiDAR scans (IndoorCD, academic-only license). The lidar itself
is simulated (ray-cast + Gaussian range noise) because we don't have
recorded lidar streams with drone trajectories.

Traceability: every mission writes one row to RUN_MANIFEST.csv and one
telemetry file to TELEMETRY/{mission_id}.jsonl. See TRACEABILITY.md for how
every reported number is derived from the manifest.
"""
import time
import json
import csv
import hashlib
import os
import sys
import numpy as np

import env
import sensor
import gibbs
import planner
import controller
import classical

from fly import (
    pick_start_goal, check_collision, optimal_len_to_tolerance,
    GOAL_TOL, GOAL_BLOCKED_RADIUS, MAX_MOVE_REFUSALS, START_GOAL_MIN_DIST,
    N_AZ, ELEV, MAX_STEPS, J_SMOOTH,
    N_SWEEPS_FULL, BURN_IN_FULL, N_SWEEPS_INCR, BURN_IN_INCR,
)

# ---- head-to-head config ----
WEAK_OBS_MAX = 2      # TSU weak-evidence fix: obs_count <= 2 -> treat as unknown,
                       # UNLESS |logodds| is large (strong evidence overrides).
                       # (validated by Grok rev 4: 13->7 failures, p=0.031 [simulated])
STRONG_L_THRESH = 1.0  # |L| above this = strong evidence, respected regardless
                       # of count. Fix for the indoorcd_128_s3_n0.15_tsu collision
                       # (2026-10-05): a voxel with 2 HIT observations (L=+1.2,
                       # P=0.77) was treated as "unknown" and the drone flew
                       # into it. Strong sensor evidence must never be ignored.

REAL_LIDAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "rooms")
TELEMETRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "head_to_head", "TELEMETRY")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "head_to_head", "RUN_MANIFEST.csv")

CONFIGS = ("control", "tsu")


def effective_observed_mask(logodds, obs_count, config):
    """Observed mask used by planner + D5 check.

    CONTROL: standard |L| > 0.05 (textbook).
    TSU: same, BUT voxels with obs_count <= WEAK_OBS_MAX are treated as
        unknown (not blocked by the P>0.3 threshold) — UNLESS |logodds|
        exceeds STRONG_L_THRESH, in which case the sensor evidence is
        respected regardless of count.

        The weak-evidence fix addresses: a cell the lidar has seen free only
        once or twice should be explorable with caution, not sealed off.
        The strong-evidence override addresses the safety hole found 2026-10-05:
        2 hit observations (L=+1.2, P=0.77) is NOT weak evidence — ignoring it
        caused a collision. Strong sensor evidence always counts.
    """
    base = classical.observed_mask(logodds)
    if config == "tsu" and obs_count is not None:
        strong_count = obs_count > WEAK_OBS_MAX
        strong_evidence = np.abs(logodds) > STRONG_L_THRESH
        return base & (strong_count | strong_evidence)
    return base


def compute_marginals(config, logodds, h_quantized, x_last, seed, step):
    """Return (marginals, gibbs_time, new_x_last).

    CONTROL: sigmoid of log-odds (independent voxels, no pairwise prior).
    TSU: chromatic Gibbs posterior (warm-started after the first step).
    """
    if config == "control":
        t0 = time.time()
        marg = classical.classical_marginals(logodds)
        return marg, time.time() - t0, None
    # TSU path
    if x_last is None:
        marg, dt, x_new = gibbs.sample_posterior(
            h_quantized, J_SMOOTH, n_sweeps=N_SWEEPS_FULL,
            burn_in=BURN_IN_FULL, seed=seed, x_init=None)
    else:
        marg, dt, x_new = gibbs.sample_posterior(
            h_quantized, J_SMOOTH, n_sweeps=N_SWEEPS_INCR,
            burn_in=BURN_IN_INCR, seed=seed + step, x_init=x_last)
    return marg, dt, x_new


def map_hash(marg):
    """Short hash of the P(occupied) map for traceability."""
    return hashlib.sha256(np.ascontiguousarray(marg).tobytes()).hexdigest()[:16]


def run_mission(config, occ, room_id, seed, noise_val, mission_id,
                telemetry_dir=None, verbose=False):
    """Fly one head-to-head mission. Returns a manifest row dict.

    Determinism: start/goal from `seed` (via pick_start_goal), scan noise
    from `seed` (1000 + seed*100 + step). Identical seeds => identical
    missions across configs. The ONLY difference is compute_marginals and
    the observed mask.
    """
    assert config in CONFIGS
    t_mission = time.time()
    shape = occ.shape

    start_pose, goal_pose = pick_start_goal(occ, seed)
    pose = start_pose
    logodds = np.zeros(shape)
    obs_count = np.zeros(shape, dtype=np.int32)

    # Boundary prior (same as fly.py — generic indoor safety invariant)
    logodds[0, :, :] = 2.0
    logodds[-1, :, :] = 2.0
    logodds[:, 0, :] = 2.0
    logodds[:, -1, :] = 2.0
    logodds[:, :, 0] = 2.0
    logodds[:, :, -1] = 2.0

    p_occ_gt = occ.astype(float)
    gt_full = planner.astar(p_occ_gt,
                            planner.pose_to_voxel(start_pose, shape),
                            planner.pose_to_voxel(goal_pose, shape))
    optimal_len = optimal_len_to_tolerance(gt_full, goal_pose, GOAL_TOL) \
        if gt_full else -1

    log = {
        "mission_id": mission_id,
        "config": config,
        "room_id": room_id,
        "seed": seed,
        "noise_val": noise_val,
        "start": start_pose,
        "goal": goal_pose,
        "optimal_path_len": optimal_len,
        "steps": [],
        "outcome": None,
        "n_plan_calls": 0,
        "n_planning_failures": 0,
        "n_move_refusals": 0,
        "path_voxels": [],
        "collision_p_occ": None,
        "total_gibbs_time": 0.0,
        "total_plan_time": 0.0,
    }

    x_last = None
    goal_v = planner.pose_to_voxel(goal_pose, shape)
    marg = np.zeros(shape)  # placeholder for early-exit paths

    for step in range(MAX_STEPS):
        step_info = {"step": step, "pose": [round(float(c), 2) for c in pose]}

        # 1. Scan (identical across configs given same seed)
        scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                 max_range=40.0, range_noise=noise_val,
                                 seed=1000 + seed * 100 + step)
        sensor.integrate_scan(logodds, shape, scans, max_range=40.0,
                              obs_count=obs_count)
        n_obs = int((np.abs(logodds) > 0.05).sum())
        step_info["n_observed"] = n_obs

        # 2. Marginals — THE brain swap
        h = sensor.quantize_8bit(logodds)
        marg, dt_gibbs, x_last = compute_marginals(
            config, logodds, h, x_last, seed, step)
        log["total_gibbs_time"] += dt_gibbs
        step_info["gibbs_time"] = round(dt_gibbs, 4)
        step_info["mean_p_occ"] = round(float(marg.mean()), 4)
        step_info["map_hash"] = map_hash(marg)

        obs_mask = effective_observed_mask(logodds, obs_count, config)
        step_info["n_strong_observed"] = int(obs_mask.sum())

        # 2b. D3: goal-blocked check
        if (controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
                and bool(obs_mask[goal_v])
                and float(marg[goal_v]) > planner.P_OBSTACLE):
            step_info["event"] = "goal_blocked"
            step_info["goal_p_occ"] = round(float(marg[goal_v]), 4)
            log["steps"].append(step_info)
            log["outcome"] = "goal_blocked"
            break

        # 3. Plan
        t_plan = time.time()
        waypoint, path, pinfo = planner.plan_next_waypoint(
            marg, pose, goal_pose, observed_mask=obs_mask)
        plan_dt = time.time() - t_plan
        log["total_plan_time"] += plan_dt
        step_info["plan_time"] = round(plan_dt, 4)
        step_info["plan_status"] = pinfo["status"]
        log["n_plan_calls"] += 1

        if waypoint is None:
            log["n_planning_failures"] += 1
            step_info["event"] = "planning_failed"
            log["steps"].append(step_info)
            log["outcome"] = "planning_failed"
            break

        # 4. D5: predictive collision check BEFORE moving
        new_pose = controller.move_toward(pose, waypoint)
        try:
            dest_v = planner.pose_to_voxel(new_pose, shape)
            dest_safe = planner.safe_to_enter(dest_v, marg, obs_mask)
        except ValueError:
            dest_safe = False
            dest_v = None
        if not dest_safe:
            log["n_move_refusals"] += 1
            step_info["event"] = "move_refused"
            log["steps"].append(step_info)
            if log["n_move_refusals"] > MAX_MOVE_REFUSALS:
                log["outcome"] = "blocked"
                break
            continue

        step_info["waypoint"] = [round(float(c), 2) for c in waypoint]
        step_info["path_len"] = pinfo["path_len"]

        # 5a. Collision check (ground truth)
        if check_collision(new_pose, occ):
            v = planner.pose_to_voxel(new_pose, shape)
            log["collision_p_occ"] = float(marg[v])
            step_info["event"] = "collision"
            pose = new_pose
            log["path_voxels"].append(v)
            log["steps"].append(step_info)
            log["outcome"] = "collision"
            break

        # 5b. Goal check
        pose = new_pose
        v = planner.pose_to_voxel(pose, shape)
        log["path_voxels"].append(v)
        if controller.distance(pose, goal_pose) < GOAL_TOL:
            step_info["event"] = "goal_reached"
            log["steps"].append(step_info)
            log["outcome"] = "success"
            break

        log["steps"].append(step_info)

    else:
        log["outcome"] = "max_steps"

    # --- mission metrics ---
    n_steps = len(log["steps"])
    log["n_steps"] = n_steps
    log["wall_time"] = time.time() - t_mission
    log["flown_path_len"] = len(log["path_voxels"])
    log["final_map_hash"] = map_hash(marg)

    if log["path_voxels"]:
        flown_occ = [bool(occ[v]) for v in log["path_voxels"]]
        log["flown_fn_rate"] = float(np.mean(flown_occ))
    else:
        log["flown_fn_rate"] = 0.0

    log["final_acc"] = classical.accuracy(marg, occ)

    # Phantom / false-negative analysis on the FINAL map.
    # Two views:
    #  - "operational" (obs_final): what each config's planner actually sees
    #    (TSU uses the weak-evidence-aware mask; control uses the standard mask)
    #  - "base" (base_final): same |L|>0.05 mask for both configs —
    #    apples-to-apples map-quality comparison.
    obs_final = effective_observed_mask(logodds, obs_count, config)
    base_final = classical.observed_mask(logodds)
    gt_free = (occ == 0)
    gt_occ = (occ == 1)
    # Phantom: observed, actually free, but P(occupied) > 0.3
    log["phantom_count"] = int(((marg > planner.P_OBSTACLE) & gt_free & obs_final).sum())
    log["phantom_count_base"] = int(((marg > planner.P_OBSTACLE) & gt_free & base_final).sum())
    # False negative (observed): actually occupied, but P(occupied) < 0.1
    log["fn_count_observed"] = int(((marg < planner.P_CAUTION) & gt_occ & obs_final).sum())
    # False negative (total): actually occupied, P < 0.1, regardless of observation
    log["fn_count_total"] = int(((marg < planner.P_CAUTION) & gt_occ).sum())

    if log["optimal_path_len"] > 0 and log["outcome"] == "success":
        log["path_efficiency"] = log["optimal_path_len"] / log["flown_path_len"]
    else:
        log["path_efficiency"] = None

    # --- telemetry dump ---
    if telemetry_dir:
        os.makedirs(telemetry_dir, exist_ok=True)
        tpath = os.path.join(telemetry_dir, f"{mission_id}.jsonl")
        with open(tpath, "w") as f:
            for s in log["steps"]:
                f.write(json.dumps(s) + "\n")
        log["telemetry_file"] = tpath

    if verbose:
        print(f"  [{config}] {mission_id}: {log['outcome']} in {n_steps} steps "
              f"({log['wall_time']:.1f}s)")

    return manifest_row(log)


def manifest_row(log):
    """Flatten a mission log to a manifest row dict."""
    sx, sy, sz = log["start"]
    gx, gy, gz = log["goal"]
    return {
        "mission_id": log["mission_id"],
        "config": log["config"],
        "room_id": log["room_id"],
        "seed": log["seed"],
        "noise_val": log["noise_val"],
        "outcome": log["outcome"],
        "n_steps": log["n_steps"],
        "flown_path_len": log["flown_path_len"],
        "optimal_path_len": log["optimal_path_len"],
        "path_efficiency": ("" if log["path_efficiency"] is None
                            else round(log["path_efficiency"], 4)),
        "n_plan_calls": log["n_plan_calls"],
        "n_planning_failures": log["n_planning_failures"],
        "n_move_refusals": log["n_move_refusals"],
        "total_gibbs_time": round(log["total_gibbs_time"], 3),
        "total_plan_time": round(log["total_plan_time"], 3),
        "wall_time": round(log["wall_time"], 2),
        "final_acc": round(log["final_acc"], 4),
        "flown_fn_rate": round(log["flown_fn_rate"], 4),
        "phantom_count": log["phantom_count"],
        "phantom_count_base": log["phantom_count_base"],
        "fn_count_observed": log["fn_count_observed"],
        "fn_count_total": log["fn_count_total"],
        "final_map_hash": log["final_map_hash"],
        "start_x": round(sx, 2), "start_y": round(sy, 2), "start_z": round(sz, 2),
        "goal_x": round(gx, 2), "goal_y": round(gy, 2), "goal_z": round(gz, 2),
        "telemetry_file": log.get("telemetry_file", ""),
    }


MANIFEST_FIELDS = [
    "mission_id", "config", "room_id", "seed", "noise_val", "outcome",
    "n_steps", "flown_path_len", "optimal_path_len", "path_efficiency",
    "n_plan_calls", "n_planning_failures", "n_move_refusals",
    "total_gibbs_time", "total_plan_time", "wall_time",
    "final_acc", "flown_fn_rate",
    "phantom_count", "phantom_count_base",
    "fn_count_observed", "fn_count_total",
    "final_map_hash",
    "start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z",
    "telemetry_file",
]


def list_rooms():
    """All available IndoorCD rooms, sorted."""
    rooms = []
    for fn in sorted(os.listdir(REAL_LIDAR_DIR)):
        if fn.endswith("_occ.npy"):
            room_id = fn[:-len("_occ.npy")]
            rooms.append(room_id)
    return rooms


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Head-to-head: TSU vs classical")
    ap.add_argument("--rooms", nargs="*", default=None,
                    help="Room IDs (default: all in real_lidar/)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3],
                    help="Mission seeds per room")
    ap.add_argument("--noise", type=float, nargs="*", default=[0.15, 0.30],
                    help="Noise levels (clean=0.15, high=0.30)")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS),
                    choices=CONFIGS)
    ap.add_argument("--outdir", default=None,
                    help="Output dir (default: slow-drone/head_to_head)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip missions already in the manifest")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(base, "head_to_head")
    tdir = os.path.join(outdir, "TELEMETRY")
    mpath = os.path.join(outdir, "RUN_MANIFEST.csv")
    os.makedirs(tdir, exist_ok=True)

    rooms = args.rooms or list_rooms()
    print(f"rooms: {len(rooms)}, seeds: {args.seeds}, noise: {args.noise}, "
          f"configs: {args.configs}")
    total = len(rooms) * len(args.seeds) * len(args.noise) * len(args.configs)
    print(f"total missions: {total}")

    # Load existing manifest for resume
    done = set()
    if args.resume and os.path.exists(mpath):
        with open(mpath) as f:
            for row in csv.DictReader(f):
                done.add(row["mission_id"])
        print(f"resuming: {len(done)} missions already done")

    # Load rooms
    occ_cache = {}
    for r in rooms:
        p = os.path.join(REAL_LIDAR_DIR, f"{r}_occ.npy")
        occ_cache[r] = np.load(p)
        print(f"  loaded {r}: shape {occ_cache[r].shape}, "
              f"occupied {occ_cache[r].mean()*100:.1f}%")

    write_header = not (os.path.exists(mpath) and args.resume
                        and os.path.getsize(mpath) > 0)
    fout = open(mpath, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS)
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
                    for config in args.configs:
                        mid = (f"{room_id}_s{seed}_n{noise_val:.2f}_{config}")
                        if mid in done:
                            continue
                        row = run_mission(config, occ, room_id, seed,
                                          noise_val, mid, telemetry_dir=tdir,
                                          verbose=True)
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
