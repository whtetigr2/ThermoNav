"""Closed-loop slow drone: sense -> map -> plan -> move -> repeat.

Plain English: this is the whole drone flying in simulation.
Each "second" (loop iteration):
  1. The drone takes a lidar scan from where it is
  2. The scan updates a log-odds map (what we've seen)
  3. Log-odds become Ising fields -> Gibbs sampler -> P(occupied) for every voxel
     (this is the piece that would run on the Z1 chip)
  4. A* planner finds a safe path on P(occupied)
  5. Safety check: is the next step into a safe voxel? (Fix D5)
     Goal check: is the goal really occupied up close? (Fix D3)
  6. The drone moves one step toward the next waypoint
  7. Check: did we crash? did we reach the goal?

This is SYNTHETIC VALIDATION. No real sensors, no real drone, no real chip.
The Gibbs sampler here is a CPU surrogate for what the Z1 would do.

Config (validated in Phase 1):
  J = 0.25 (agreement-form MRF)
  max|b| = 2.0 (NOT 3.5 — that was a red-team error, corrected)
"""
import time
import numpy as np

import env
import sensor
import gibbs
import planner
import controller
import classical

# ---- config ----
J_SMOOTH = 0.25          # validated Phase 1 (NOT 1.0)
N_SWEEPS_FULL = 300      # full Gibbs run (first iteration)
BURN_IN_FULL = 100
N_SWEEPS_INCR = 200      # incremental (warm-started from previous state)
BURN_IN_INCR = 50
N_AZ = 60                # rays per elevation (reduced for speed; 180 rays/scan)
ELEV = (-0.26, 0.0, 0.26)
MAX_STEPS = 200
GOAL_TOL = 1.5           # voxels: close enough to count as "reached"
GOAL_BLOCKED_RADIUS = 5.0  # Fix D3: within this distance, trust the sensor —
                           # if the goal voxel is observed-occupied, abort.
MAX_MOVE_REFUSALS = 10   # Fix D5: consecutive refused moves before "blocked"
START_GOAL_MIN_DIST = 15.0  # Fix D2/D6: min distance for random start/goal pairs


def pick_start_goal(occ, seed, z=8, min_dist=START_GOAL_MIN_DIST,
                    max_tries=200):
    """Pick a random start/goal pair of free voxels, far apart, GT-feasible.

    Fix D2/D6: the old version always picked the same max-distance waypoint
    pair (2 routes x 5 noise samples, not 10 missions). This samples random
    free-voxel pairs so each seed gets genuinely different geometry. For real
    maps this samples the actual map topology instead of synthetic waypoints.

    Only returns pairs where a ground-truth path exists (impossible missions
    are a bad test, not a drone failure).
    """
    rng = np.random.default_rng(seed)
    shape = occ.shape
    nx, ny, nz = shape
    if not (0 <= z < nz):
        z = nz // 2
    free = np.argwhere(occ[:, :, z] == 0)
    if len(free) < 2:
        raise RuntimeError(f"not enough free space at height z={z}")

    def to_pose(v):
        return (float(v[0]) + 0.5, float(v[1]) + 0.5, float(z) + 0.5)

    for _ in range(max_tries):
        i, j = rng.choice(len(free), 2, replace=False)
        a, b = to_pose(free[i]), to_pose(free[j])
        if controller.distance(a, b) < min_dist:
            continue
        # Ground-truth feasibility: does a path exist?
        p_gt = occ.astype(float)
        gt_path = planner.astar(p_gt,
                                planner.pose_to_voxel(a, shape),
                                planner.pose_to_voxel(b, shape))
        if gt_path is not None:
            return a, b

    # Fallback: relax min_dist, keep feasibility requirement
    for _ in range(max_tries):
        i, j = rng.choice(len(free), 2, replace=False)
        a, b = to_pose(free[i]), to_pose(free[j])
        p_gt = occ.astype(float)
        gt_path = planner.astar(p_gt,
                                planner.pose_to_voxel(a, shape),
                                planner.pose_to_voxel(b, shape))
        if gt_path is not None and a != b:
            return a, b
    raise RuntimeError("could not find feasible start/goal pair")


def check_collision(pose, occ):
    """True if the pose center is inside a ground-truth occupied voxel."""
    ix, iy, iz = int(pose[0]), int(pose[1]), int(pose[2])
    nx, ny, nz = occ.shape
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return True  # left the map: crash
    return bool(occ[ix, iy, iz])


def optimal_len_to_tolerance(gt_path, goal_pose, tol):
    """Fix D1: length of the optimal path truncated at first voxel within
    `tol` of the goal. The old metric compared the drone's early-stopped path
    against the full optimal path — different endpoints, inflated ratio."""
    for i, v in enumerate(gt_path):
        vp = planner.voxel_to_pose(v)
        if controller.distance(vp, goal_pose) < tol:
            return i + 1
    return len(gt_path)


def fly_mission(seed=0, verbose=True, map_name=None, occ=None, map_info=None):
    """Fly one mission. Returns a dict of metrics.

    Args:
        seed: random seed.
        verbose: print progress.
        map_name: MovingAI map file (e.g. "orz100d.map") for REAL map.
            If None, uses synthetic build_room.
        occ: pre-built occupancy grid (overrides map_name and build_room).
        map_info: dict describing a pre-built occ (for logging).
    """
    t_mission = time.time()

    # --- setup: occupancy grid (Fix D7: use occ.shape, not hardcoded NX/NY/NZ) ---
    if occ is None:
        if map_name is not None:
            occ, loaded_info = env.load_map(map_name, seed=seed)
            map_info = loaded_info
        else:
            occ = env.build_room(32, 32, 16)
            map_info = {"source": "synthetic", "note": "box room"}
    if map_info is None:
        map_info = {"source": "synthetic", "note": "box room"}
    shape = occ.shape

    start_pose, goal_pose = pick_start_goal(occ, seed)
    pose = start_pose
    logodds = np.zeros(shape)

    # Structural prior: the map boundary is solid. Every map we fly has
    # occupied borders (floor, ceiling, containment walls), and flying outside
    # the operating volume is never correct. This is generic indoor knowledge,
    # not map-specific cheating — it's a safety invariant.
    # Log-odds 2.0 -> P ~= 0.88, well above the 0.3 obstacle threshold.
    logodds[0, :, :] = 2.0
    logodds[-1, :, :] = 2.0
    logodds[:, 0, :] = 2.0
    logodds[:, -1, :] = 2.0
    logodds[:, :, 0] = 2.0
    logodds[:, :, -1] = 2.0

    # Optimal path length on ground truth, truncated to GOAL_TOL (Fix D1)
    p_occ_gt = occ.astype(float)
    gt_full = planner.astar(p_occ_gt,
                            planner.pose_to_voxel(start_pose, shape),
                            planner.pose_to_voxel(goal_pose, shape))
    optimal_len = optimal_len_to_tolerance(gt_full, goal_pose, GOAL_TOL) \
        if gt_full else -1

    # --- flight log ---
    log = {
        "seed": seed,
        "map": map_info,
        "start": start_pose,
        "goal": goal_pose,
        "optimal_path_len": optimal_len,
        "steps": [],
        "outcome": None,
        "n_plan_calls": 0,        # Fix D10: was "n_replans", miscounted
        "n_planning_failures": 0,
        "n_move_refusals": 0,    # Fix D5: refused unsafe steps
        "path_voxels": [],
        "collision_p_occ": None,
    }

    x_last = None  # warm-start for Gibbs
    goal_v = planner.pose_to_voxel(goal_pose, shape)

    for step in range(MAX_STEPS):
        step_info = {"step": step, "pose": pose}

        # 1. Scan
        scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                 max_range=40.0, range_noise=0.15,
                                 seed=1000 + seed * 100 + step)
        sensor.integrate_scan(logodds, shape, scans, max_range=40.0)
        n_obs = int((np.abs(logodds) > 0.05).sum())
        step_info["n_observed"] = n_obs

        # 2. Log-odds -> Ising fields -> Gibbs -> P(occupied)
        h = sensor.quantize_8bit(logodds)  # 8-bit in [-4, 4]
        if x_last is None:
            marg, dt_gibbs, x_last = gibbs.sample_posterior(
                h, J_SMOOTH, n_sweeps=N_SWEEPS_FULL, burn_in=BURN_IN_FULL,
                seed=seed, x_init=None)
        else:
            marg, dt_gibbs, x_last = gibbs.sample_posterior(
                h, J_SMOOTH, n_sweeps=N_SWEEPS_INCR, burn_in=BURN_IN_INCR,
                seed=seed + step, x_init=x_last)
        step_info["gibbs_time"] = dt_gibbs
        step_info["mean_p_occ"] = float(marg.mean())
        obs_mask = classical.observed_mask(logodds)

        # 2b. Fix D3: goal-blocked check. The planner treats the goal as
        # traversable (so distant false positives don't strand the drone),
        # but if close-range sensing says the goal is really occupied,
        # abort instead of flying into it.
        if (controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
                and bool(obs_mask[goal_v])
                and float(marg[goal_v]) > planner.P_OBSTACLE):
            step_info["event"] = "goal_blocked"
            step_info["goal_p_occ"] = float(marg[goal_v])
            log["steps"].append(step_info)
            log["outcome"] = "goal_blocked"
            break

        # 3. Plan
        t_plan = time.time()
        waypoint, path, pinfo = planner.plan_next_waypoint(
            marg, pose, goal_pose, observed_mask=obs_mask)
        step_info["plan_time"] = time.time() - t_plan
        step_info["plan_status"] = pinfo["status"]
        log["n_plan_calls"] += 1

        if waypoint is None:
            log["n_planning_failures"] += 1
            step_info["event"] = "planning_failed"
            log["steps"].append(step_info)
            log["outcome"] = "planning_failed"
            break

        # 4. Fix D5: predictive collision check BEFORE moving.
        # The planner can suggest anything; the drone never executes a step
        # into an unsafe voxel.
        new_pose = controller.move_toward(pose, waypoint)
        try:
            dest_v = planner.pose_to_voxel(new_pose, shape)
            dest_safe = planner.safe_to_enter(dest_v, marg, obs_mask)
        except ValueError:
            dest_safe = False  # out of bounds
            dest_v = None
        if not dest_safe:
            log["n_move_refusals"] += 1
            step_info["event"] = "move_refused"
            step_info["refused_voxel"] = dest_v
            log["steps"].append(step_info)
            if log["n_move_refusals"] > MAX_MOVE_REFUSALS:
                log["outcome"] = "blocked"
                break
            continue  # hover this step; fresh scan + replan next iteration

        step_info["waypoint"] = waypoint
        step_info["path_len"] = pinfo["path_len"]

        # 5a. Collision check (against ground truth — belt and suspenders;
        # the D5 check above should make this unreachable)
        if check_collision(new_pose, occ):
            v = planner.pose_to_voxel(new_pose, shape)
            log["collision_p_occ"] = float(marg[v])
            step_info["event"] = "collision"
            step_info["collision_voxel"] = v
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

        if verbose and step % 20 == 0:
            print(f"  step {step}: pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                  f"obs={n_obs} plan={pinfo['status']} path_len={pinfo['path_len']}")

    else:
        log["outcome"] = "max_steps"

    # --- mission metrics ---
    n_steps = len(log["steps"])
    log["n_steps"] = n_steps
    log["wall_time"] = time.time() - t_mission
    log["flown_path_len"] = len(log["path_voxels"])

    if log["path_voxels"]:
        flown_occ = [bool(occ[v]) for v in log["path_voxels"]]
        log["flown_fn_rate"] = float(np.mean(flown_occ))
        log["flown_mean_p_occ"] = float(np.mean([marg[v] for v in log["path_voxels"]]))
    else:
        log["flown_fn_rate"] = 0.0
        log["flown_mean_p_occ"] = 0.0

    log["final_acc"] = classical.accuracy(marg, occ)

    if verbose:
        print(f"  outcome: {log['outcome']} in {n_steps} steps "
              f"({log['wall_time']:.1f}s wall)")
        print(f"  flown path: {log['flown_path_len']} voxels, "
              f"optimal: {optimal_len}, "
              f"FN rate on path: {log['flown_fn_rate']:.3f}")

    return log


def fly_mission_s16(seed=0, verbose=True, map_name=None, occ=None,
                    map_info=None, noise_val=0.15):
    """Fly one mission with the S16 hierarchical planner in the main loop.

    Same perception stack as fly_mission (lidar -> log-odds -> Gibbs),
    but planning uses S16: host A* (1 Hz, global waypoints) + TSU S2
    (10 Hz, local voxel actions via 18-spin Ising sampling).

    This is the integrated thermopilot loop. See s16.py and
    thermopilot/HARDENING.md for the S2/S16 validation.

    Args:
        seed, verbose, map_name, occ, map_info: same as fly_mission.
        noise_val: lidar range noise (default 0.15).

    Returns:
        Same log dict format as fly_mission, plus:
          n_tsu_decisions, total_tsu_time, n_host_failures.
    """
    import s16 as s16_module

    TSU_SUBSTEPS = 10
    TSU_SPEED = 0.1
    MAX_REFUSALS_S16 = MAX_MOVE_REFUSALS * TSU_SUBSTEPS

    t_mission = time.time()

    if occ is None:
        if map_name is not None:
            occ, loaded_info = env.load_map(map_name, seed=seed)
            map_info = loaded_info
        else:
            occ = env.build_room(32, 32, 16)
            map_info = {"source": "synthetic", "note": "box room"}
    if map_info is None:
        map_info = {"source": "synthetic", "note": "box room"}
    shape = occ.shape

    start_pose, goal_pose = pick_start_goal(occ, seed)
    pose = start_pose
    logodds = np.zeros(shape)
    obs_count = np.zeros(shape, dtype=np.int32)

    # Boundary prior (same as fly_mission)
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
        "seed": seed,
        "map": map_info,
        "start": start_pose,
        "goal": goal_pose,
        "optimal_path_len": optimal_len,
        "steps": [],
        "outcome": None,
        "n_plan_calls": 0,
        "n_planning_failures": 0,
        "n_move_refusals": 0,
        "n_tsu_decisions": 0,
        "total_tsu_time": 0.0,
        "path_voxels": [],
        "collision_p_occ": None,
        "planner_mode": "s16",
    }

    s16p = s16_module.S16Planner(
        waypoint_lookahead=2, stuck_threshold=40,
        s2_kwargs={"seed": seed, "n_sweeps": 200})

    x_last = None
    goal_v = planner.pose_to_voxel(goal_pose, shape)
    marg = np.zeros(shape)

    for step in range(MAX_STEPS):
        # 1 Hz host: scan
        scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                 max_range=40.0, range_noise=noise_val,
                                 seed=1000 + seed * 100 + step)
        sensor.integrate_scan(logodds, shape, scans, max_range=40.0,
                              obs_count=obs_count)

        # 1 Hz host: Gibbs -> P(occupied)
        h = sensor.quantize_8bit(logodds)
        if x_last is None:
            marg, dt_gibbs, x_last = gibbs.sample_posterior(
                h, J_SMOOTH, n_sweeps=N_SWEEPS_FULL, burn_in=BURN_IN_FULL,
                seed=seed, x_init=None)
        else:
            marg, dt_gibbs, x_last = gibbs.sample_posterior(
                h, J_SMOOTH, n_sweeps=N_SWEEPS_INCR, burn_in=BURN_IN_INCR,
                seed=seed + step, x_init=x_last)

        # Weak-evidence-aware observed mask (same as head-to-head TSU)
        base_mask = classical.observed_mask(logodds)
        strong_count = obs_count > 2
        strong_ev = np.abs(logodds) > 1.0
        obs_mask = base_mask & (strong_count | strong_ev)

        # D3: goal-blocked check
        if (controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
                and bool(obs_mask[goal_v])
                and float(marg[goal_v]) > planner.P_OBSTACLE):
            log["steps"].append({"step": step, "pose": pose,
                                 "event": "goal_blocked"})
            log["outcome"] = "goal_blocked"
            break

        # 10 Hz TSU fast loop
        for _ in range(TSU_SUBSTEPS):
            pos_v = planner.pose_to_voxel(pose, shape)
            next_v, s2info = s16p.tsu_act(
                pos_v, goal_v, marg, obs_mask, pos_cont=pose)
            log["n_tsu_decisions"] += 1
            log["total_tsu_time"] += s2info["tsu_time"]

            # D5: predictive guard. "Look before you leap" (2026-10-05):
            # strict line-of-sight check through the believed map.
            if not planner.safe_to_enter(next_v, marg, obs_mask, pos_v):
                log["n_move_refusals"] += 1
                # D5-refusal feedback (2026-10-05): tell S16 so it stops
                # re-proposing the same blocked voxel and escapes instead.
                s16p.notify_refusal(next_v, pos_v, marg, obs_mask)
                if log["n_move_refusals"] > MAX_REFUSALS_S16:
                    log["outcome"] = "blocked"
                    break
                continue

            target = planner.voxel_to_pose(next_v)
            # speed_factor from S16 gate: cautious mode moves slower.
            # (2026-10-05: alternating hover caused pull-back to voxel center;
            #  speed scaling achieves "proceed with caution" correctly.)
            _sf = float(s2info.get("speed_factor", 1.0))
            pose = controller.move_toward(pose, target, speed=TSU_SPEED * _sf)
            v = planner.pose_to_voxel(pose, shape)
            if not log["path_voxels"] or v != log["path_voxels"][-1]:
                log["path_voxels"].append(v)

            if check_collision(pose, occ):
                log["collision_p_occ"] = float(marg[v])
                log["outcome"] = "collision"
                break

            if controller.distance(pose, goal_pose) < GOAL_TOL:
                log["outcome"] = "success"
                break
        else:
            log["n_plan_calls"] = s16p.n_host_replans
            log["n_planning_failures"] = s16p.n_host_failures
            log["steps"].append({"step": step})
            continue
        log["n_plan_calls"] = s16p.n_host_replans
        log["n_planning_failures"] = s16p.n_host_failures
        break
    else:
        log["outcome"] = "max_steps"

    n_steps = len(log["steps"])
    log["n_steps"] = n_steps
    log["wall_time"] = time.time() - t_mission
    log["flown_path_len"] = len(log["path_voxels"])

    if log["path_voxels"]:
        flown_occ = [bool(occ[v]) for v in log["path_voxels"]]
        log["flown_fn_rate"] = float(np.mean(flown_occ))
    else:
        log["flown_fn_rate"] = 0.0

    log["final_acc"] = classical.accuracy(marg, occ)

    if verbose:
        print(f"  [s16] outcome: {log['outcome']} in {n_steps} host steps "
              f"({log['wall_time']:.1f}s wall, "
              f"{log['n_tsu_decisions']} TSU decisions)")

    return log


def main():
    import json
    N_MISSIONS = 10
    print("== closed-loop slow drone: 10 missions (fixed) ==")
    print(f"J={J_SMOOTH}, 1 Hz sim loop, diverse start/goal pairs")
    print()

    all_logs = []
    for seed in range(N_MISSIONS):
        print(f"-- mission {seed} --")
        log = fly_mission(seed=seed, verbose=True)
        all_logs.append(log)
        print()

    # --- aggregate stats ---
    outcomes = [l["outcome"] for l in all_logs]
    def count(o):
        return sum(1 for x in outcomes if x == o)
    n_success = count("success")

    print("=" * 60)
    print("AGGREGATE RESULTS (10 missions, synthetic, diverse start/goal)")
    print("=" * 60)
    print(f"  success:          {n_success}/10")
    print(f"  collision:        {count('collision')}/10")
    print(f"  planning_failed:  {count('planning_failed')}/10")
    print(f"  goal_blocked:     {count('goal_blocked')}/10")
    print(f"  blocked:          {count('blocked')}/10")
    print(f"  max_steps:        {count('max_steps')}/10")
    print()

    # Start/goal diversity check (Fix D2)
    pairs = [(tuple(l["start"]), tuple(l["goal"])) for l in all_logs]
    print(f"  unique start/goal pairs: {len(set(pairs))}/10")
    print()

    successes = [l for l in all_logs if l["outcome"] == "success"]
    if successes:
        # Path efficiency with MATCHED endpoints (Fix D1)
        ratios = []
        for l in successes:
            if l["optimal_path_len"] > 0:
                ratios.append(l["flown_path_len"] / l["optimal_path_len"])
        if ratios:
            print(f"  path efficiency (flown/optimal, matched endpoints): "
                  f"mean={np.mean(ratios):.2f}, min={np.min(ratios):.2f}, "
                  f"max={np.max(ratios):.2f}")
        steps = [l["n_steps"] for l in successes]
        print(f"  steps to goal: mean={np.mean(steps):.1f}")
        print(f"  plan calls per mission: mean={np.mean([l['n_plan_calls'] for l in successes]):.1f}")
        print(f"  move refusals (all missions): total={sum(l['n_move_refusals'] for l in all_logs)}")
        fn_rates = [l["flown_fn_rate"] for l in all_logs]
        print(f"  FN rate on flown path (all missions): mean={np.mean(fn_rates):.4f}")

    collisions = [l for l in all_logs if l["outcome"] == "collision"]
    if collisions:
        p_occs = [l["collision_p_occ"] for l in collisions
                  if l["collision_p_occ"] is not None]
        if p_occs:
            print(f"  P(occupied) at crash sites: mean={np.mean(p_occs):.3f}")

    print()
    print("NOTE: synthetic validation only. No real sensors, no real drone.")

    def clean(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        return o

    with open("flight_logs.json", "w") as f:
        json.dump([clean(l) for l in all_logs], f, indent=1)
    print("saved flight_logs.json")


if __name__ == "__main__":
    main()
