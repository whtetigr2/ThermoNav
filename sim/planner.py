"""A* path planner on the probabilistic occupancy map.

Plain English: the drone knows P(occupied) for every voxel (from the Gibbs
sampler). A* finds the cheapest path from where we are to where we want to
go. "Cheap" means short AND safe — voxels that are probably occupied cost
more to fly through.

Safety thresholds (from REDTEAM_FIXES.md, synthetic-data recommendations):
  P(occupied) > 0.3  -> obstacle (blocked, cannot enter)
  0.1 < P <= 0.3     -> caution (high cost)
  P <= 0.1           -> free (low cost)

Unobserved voxels (no sensor evidence yet):
  Explorable with caution cost (5.0, the caution-zone midpoint). Unknown is
  treated as "caution-level", not worse than caution.
  NOTE: we do NOT block unobserved voxels by P-value. Measured: 57% of
  unobserved voxels have P>0.5 due to Gibbs smoothing (not real evidence),
  so a P-threshold would block most explorable space. The controller's
  pre-move check (safe_to_enter) is the safety net for individual steps.
  (Fix D4: the old cost of 11.0/step created a perverse incentive to hug
  the floor; see REDTEAM_FLIGHT.md.)

This is SYNTHETIC VALIDATION. These thresholds are not flight-certified.
"""
import heapq
import numpy as np

# Safety thresholds — from red-team safety review (synthetic data only)
P_OBSTACLE = 0.3   # above this (when OBSERVED): blocked
P_CAUTION = 0.1    # above this (but <= 0.3): high cost
# Unobserved voxels with believed P(occupied) above this are denied even
# though never directly observed. The Gibbs marginal smooths unobserved
# voxels toward occupied neighbors — a high value means the prior is
# screaming "probably occupied".
#
# 0.5 is the maximum-entropy point: the belief leans occupied. If you have
# never observed a voxel and your belief says it's more likely occupied
# than not, you do not enter. (2026-10-05 red-team: synthetic
# forced-crossing room, unobserved voxel with P=0.813 was occupied →
# collision. D5 waved it through because it never thresholded P for
# unobserved voxels.)
# (2026-10-06 red-team round 2, F12: indoorcd_128 s5 n0.15, unobserved
# voxel with P=0.667 was occupied → collision. 0.7 was a cliff edge;
# 0.667 slipped under it. Battery 1 validated 0.5/0.6/0.7/0.8 all prevent
# the synth collision with zero false-positive mission failures — the
# mechanism matters, and 0.5 is the principled value.)
# F10 cliff-edge note: P is noisy (±0.05 Monte Carlo). The 3-refusal escape
# system is the designed hysteresis — flicker around the threshold
# accumulates refusals and triggers lateral exploration, which is the
# correct response to an uncertain voxel.
P_UNOBSERVED_DENY = 0.5

# Lidar elevation fan (must match fly.py ELEV = (-0.26, 0.0, 0.26)).
# A voxel outside this elevation angle from the drone can NEVER be observed
# by the sensor — entering it is flying blind, not exploring.
SENSOR_MAX_ELEV = 0.26
BLIND_ELEV_MARGIN = 0.04  # slop for ray voxel-crossing width

# Cost weights
LAMBDA_CAUTION = 20.0  # extra cost per unit P(occupied) in caution zone
LAMBDA_FREE = 5.0      # extra cost per unit P(occupied) in free zone
# Unobserved voxels: explorable with caution-level cost.
# 1.0 + LAMBDA_UNKNOWN * 0.5 = 5.0 = caution-zone midpoint (P=0.2 -> 1+20*0.2).
# Fix D4: was 20.0 (cost 11.0), which made A* prefer descending through cheap
# observed space over paying for altitude — the brc200d crash root cause.
LAMBDA_UNKNOWN = 8.0
COST_UNKNOWN = 1.0 + LAMBDA_UNKNOWN * 0.5  # = 5.0
# Blind-spot planning penalty (2026-10-05, indoorcd_128 fix refinement;
# 2026-10-06 tall-wall diagnosis: set to 0.0). A* is allowed to plan through
# Blind-spot penalty for A* pass 2 (see astar two-pass, 2026-10-06).
# History: 10.0 (2026-10-05) priced peek overhead but made vertical routes
# so expensive A* chased nonexistent lateral paths (tall-wall y-sweep).
# 0.0 (2026-10-06) fixed the tall wall but let A* route freely through
# blind voxels D5 denies, livelocking indoorcd_048. No scalar satisfies
# both — the tension is structural. Two-pass A* resolves it: pass 1 avoids
# blind voxels entirely (no penalty needed); pass 2 (blind is the only way)
# uses this small penalty to prefer shorter blind sequences.
BLIND_SPOT_PENALTY = 2.0


def pose_to_voxel(pose, shape):
    """Continuous (x,y,z) pose -> integer voxel index.

    Raises ValueError if the pose is outside the grid (Fix D9: no more
    silent clamping — out-of-bounds is a bug, not a rounding choice).
    """
    nx, ny, nz = shape
    ix, iy, iz = int(pose[0]), int(pose[1]), int(pose[2])
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        raise ValueError(f"pose {pose} out of bounds for grid {shape}")
    return (ix, iy, iz)


def voxel_to_pose(voxel):
    """Voxel index -> continuous pose at voxel center."""
    return (float(voxel[0]) + 0.5, float(voxel[1]) + 0.5, float(voxel[2]) + 0.5)


def in_bounds(voxel, shape):
    """True if voxel is inside the grid."""
    nx, ny, nz = shape
    return 0 <= voxel[0] < nx and 0 <= voxel[1] < ny and 0 <= voxel[2] < nz


def traversal_cost(p_occ, observed=True):
    """Cost to ENTER a voxel with given P(occupied).

    Args:
        p_occ: P(occupied) in [0, 1].
        observed: whether the sensor has actually seen this voxel.

    Returns float('inf') for blocked voxels.
    """
    if not observed:
        # No direct evidence: explorable at caution-level cost (Fix D4).
        # We deliberately do NOT threshold on P here: Gibbs smoothing pushes
        # 57% of unobserved voxels above P=0.5 (measured), so a P-block would
        # seal off most explorable space. Step-level safety is handled by
        # safe_to_enter() in the controller path.
        return COST_UNKNOWN
    if p_occ > P_OBSTACLE:
        return float('inf')
    elif p_occ > P_CAUTION:
        return 1.0 + LAMBDA_CAUTION * p_occ
    else:
        return 1.0 + LAMBDA_FREE * p_occ


def _intermediate_voxels(start_c, end_c, shape):
    """Voxel index tuples strictly between two continuous points.

    3D DDA voxel traversal (Amanatides & Woo). Yields each voxel the
    segment passes through, EXCLUDING the voxel containing start_c and
    the voxel containing end_c. Used by the "look before you leap" check
    to inspect the believed map along the line of sight.
    """
    import math
    x0, y0, z0 = float(start_c[0]), float(start_c[1]), float(start_c[2])
    x1, y1, z1 = float(end_c[0]), float(end_c[1]), float(end_c[2])
    ix, iy, iz = math.floor(x0), math.floor(y0), math.floor(z0)
    tx, ty, tz = math.floor(x1), math.floor(y1), math.floor(z1)
    nx, ny, nz = shape

    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
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

    # Walk until we reach the target voxel; yield intermediates only.
    guard = (abs(tx - ix) + abs(ty - iy) + abs(tz - iz)) + 8
    for _ in range(guard):
        if tmax_x < tmax_y and tmax_x < tmax_z:
            ix += step_x
            tmax_x += tdelta_x
        elif tmax_y < tmax_z:
            iy += step_y
            tmax_y += tdelta_y
        else:
            iz += step_z
            tmax_z += tdelta_z
        if (ix, iy, iz) == (tx, ty, tz):
            return
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            yield (ix, iy, iz)
        else:
            return


def safe_to_enter(voxel, p_occ, observed_mask, current_pos=None):
    """Conservative pre-move check: may the drone step into this voxel now?

    Fix D5: the controller was moving blindly. This is the safety invariant —
    the planner can suggest anything, but the drone never executes a step
    into a voxel that fails this check.

    Returns False for: out-of-bounds voxels, and observed voxels with
    P(occupied) > 0.3.

    Unobserved voxels: allowed ONLY with "look before you leap" verification
    when the caller's current position is provided (current_pos). Three checks
    run against the BELIEVED map (p_occ / observed_mask, never ground truth):

    0. BELIEVED P THRESHOLD: unobserved voxels with P(occupied) > 0.7 are
       denied outright. The Gibbs marginal smooths unobserved voxels toward
       occupied neighbors — if the belief says "probably occupied", D5
       listens even without direct observation. (2026-10-05 red-team:
       synthetic forced-crossing room, unobserved voxel with P=0.813 was
       occupied → collision.)
    1. SENSOR BLIND SPOT: the lidar's elevation fan is +/-0.26 rad. A voxel
       outside that fan (e.g. directly above/below the drone) can never be
       observed by this sensor. Entry into an unobserved blind-spot voxel is
       denied — the drone must first maneuver laterally to get a viewing
       angle, observe the voxel, and then enter it as observed space.
       (2026-10-05: indoorcd_071 collision — the drone moved +z into a
       voxel at 90° elevation with zero ray touches; it was occupied.)
    2. LINE OF SIGHT: a ray from the drone's voxel through the believed map;
       if any observed voxel with P > 0.3 blocks it, entry is denied.
    3. FULLY-UNSEEN POCKET: if the target AND all 6 neighbors are
       unobserved, entry is denied — zero local information is blind
       flight, not exploration. (2026-10-06 red-team round 2, F9.)

    (2026-10-05: the old code assumed "the lidar sees the floor/walls before
    the drone gets close enough to step into them" — false. The scan is
    omnidirectional in azimuth but a single noisy scan can leave the adjacent
    voxel unobserved, and the elevation fan has a hard blind cone overhead.
    The drone was stepping into unobserved-but-occupied voxels. See the
    indoorcd_071 collision and the 5 V1 classical collisions.)

    When current_pos is None (legacy callers: V1 baseline, classical demo),
    unobserved voxels are allowed as before — the planner prices them at
    caution cost. New S16-path callers must pass current_pos.
    """
    import math
    nx, ny, nz = p_occ.shape
    vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
    if not (0 <= vx < nx and 0 <= vy < ny and 0 <= vz < nz):
        return False
    v = (vx, vy, vz)
    if bool(observed_mask[v]):
        return float(p_occ[v]) <= P_OBSTACLE
    if current_pos is None:
        return True  # legacy path: planner prices unknown at caution cost
    # Strict "look before you leap" path (S16 callers).
    cp = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
    if v == cp:
        return True  # hover in place: already there, trivially safe
    # Check 0: believed P(occupied) for unobserved voxels. The Gibbs
    # marginal can pull an unobserved voxel high via neighbor smoothing —
    # if the prior is screaming "probably occupied", D5 must listen.
    # (2026-10-05 red-team: unobserved voxel with P=0.813 was occupied →
    # collision. The old code never thresholded P for unobserved voxels.)
    if float(p_occ[v]) > P_UNOBSERVED_DENY:
        return False
    sc = (float(cp[0]) + 0.5, float(cp[1]) + 0.5, float(cp[2]) + 0.5)
    ec = (vx + 0.5, vy + 0.5, vz + 0.5)
    # Check 1: sensor blind spot. If the sensor physically cannot see the
    # target voxel, the drone may not enter it unverified.
    _dz = ec[2] - sc[2]
    _dh = math.hypot(ec[0] - sc[0], ec[1] - sc[1])
    _elev = math.atan2(abs(_dz), _dh) if _dh > 1e-9 else math.pi / 2
    if _elev > SENSOR_MAX_ELEV + BLIND_ELEV_MARGIN:
        return False
    # Check 2: line of sight through the believed map.
    for w in _intermediate_voxels(sc, ec, p_occ.shape):
        if bool(observed_mask[w]) and float(p_occ[w]) > P_OBSTACLE:
            return False  # believed obstacle blocks the line of sight
    # Check 3: no stepping into fully-unseen pockets. If the target voxel
    # AND all 6 of its neighbors are unobserved, we have zero local
    # information about the geometry — entering is not exploration, it's
    # blind flight. Frontier voxels (the legitimate exploration target)
    # always have at least one observed neighbor by construction.
    # (2026-10-06 red-team round 2, F9: defense-in-depth against
    # unobserved thin walls in sensor shadows.)
    _fully_unseen = True
    for _d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        _nb = (vx+_d[0], vy+_d[1], vz+_d[2])
        if (0 <= _nb[0] < nx and 0 <= _nb[1] < ny and 0 <= _nb[2] < nz
                and bool(observed_mask[_nb])):
            _fully_unseen = False
            break
    if _fully_unseen:
        return False
    return True


def is_blind_spot(voxel, observed_mask, current_pos):
    """True if this voxel is unobserved AND in the lidar's blind cone.

    The lidar's elevation fan is +/-SENSOR_MAX_ELEV. A voxel outside that
    fan can never be observed from current_pos. Used by S16 to plan "peek"
    maneuvers: move laterally to bring the voxel into view before entering.
    """
    import math
    vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
    v = (vx, vy, vz)
    if bool(observed_mask[v]):
        return False  # observed (from some angle) — not blind
    nx, ny, nz = observed_mask.shape
    if not (0 <= vx < nx and 0 <= vy < ny and 0 <= vz < nz):
        return False
    cp = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
    if v == cp:
        return False
    sc = (float(cp[0]) + 0.5, float(cp[1]) + 0.5, float(cp[2]) + 0.5)
    ec = (vx + 0.5, vy + 0.5, vz + 0.5)
    dz = ec[2] - sc[2]
    dh = math.hypot(ec[0] - sc[0], ec[1] - sc[1])
    elev = math.atan2(abs(dz), dh) if dh > 1e-9 else math.pi / 2
    return elev > SENSOR_MAX_ELEV + BLIND_ELEV_MARGIN


def entry_caution(voxel, p_occ, observed_mask, current_pos):
    """True if the drone should enter this voxel slowly.

    "Look before you leap", second half: when the target voxel is unobserved,
    or the line of sight to it passes through other unobserved voxels, the
    caller should reduce speed so that a fresh scan is integrated BEFORE the
    pose center commits into the voxel. A single noisy scan can leave an
    adjacent voxel unobserved; repeated scans from closer range converge on
    the truth (log-odds accumulate across scans).

    NOTE (2026-10-05): wiring the 0.5x slowdown for ALL unobserved entries
    regressed indoorcd_048 (too slow to finish in MAX_STEPS). The horizontal
    sensor fan already covers adjacent-horizontal voxels, so the slowdown
    bought little there. Currently unused by callers; kept as a tested
    utility for future selective use (e.g. ray-chain unknowns only).
    """
    vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
    v = (vx, vy, vz)
    if not bool(observed_mask[v]):
        return True  # unverified target: approach slowly, re-scan en route
    sc = (float(current_pos[0]) + 0.5, float(current_pos[1]) + 0.5,
          float(current_pos[2]) + 0.5)
    ec = (vx + 0.5, vy + 0.5, vz + 0.5)
    for w in _intermediate_voxels(sc, ec, p_occ.shape):
        if not bool(observed_mask[w]):
            return True  # chain of unknowns along the path
    return False


def astar(p_occ, start_voxel, goal_voxel, observed_mask=None):
    """A* on the probabilistic grid.

    Args:
        p_occ: 3D array of P(occupied) in [0,1].
        start_voxel: (ix, iy, iz) tuple.
        goal_voxel: (ix, iy, iz) tuple.
        observed_mask: optional 3D bool array. Voxels with False are
            treated as unobserved (explorable with caution cost, not blocked).

    Returns:
        List of voxel tuples from start to goal (inclusive), or None if no path.
        Uses 6-connected neighborhood (matches MRF degree).

    Two-pass (2026-10-06): pass 1 treats blind-spot voxels as BLOCKED and
        looks for a path that avoids them entirely. If none exists, pass 2
        allows blind voxels with BLIND_SPOT_PENALTY. This separates "blind
        is a shortcut" (avoid it — the peek/D5 cycle livelocks in confined
        space, e.g. indoorcd_048) from "blind is the only way" (take it —
        e.g. the tall wall, where the vertical route is necessary). No scalar
        penalty satisfies both; the structural fix is to try without first.
    """
    def _astar(allow_blind):
        nx, ny, nz = p_occ.shape
        if observed_mask is None:
            _obs = np.ones_like(p_occ, dtype=bool)
        else:
            _obs = observed_mask

        def in_b(v):
            return 0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz

        def heuristic(v):
            # Manhattan distance (admissible: min cost per step is 1.0)
            return abs(v[0] - goal_voxel[0]) + abs(v[1] - goal_voxel[1]) + abs(v[2] - goal_voxel[2])

        def cost(v):
            # Goal voxel is always traversable for PLANNING (a false positive
            # there shouldn't strand the drone at a distance; sensor data refines
            # as we approach). Fix D3: fly.py now aborts with "goal_blocked" if
            # close-range sensing confirms the goal is really occupied — so this
            # can no longer turn into a crash.
            if v == goal_voxel:
                p = p_occ[v]
                return 1.0 + LAMBDA_CAUTION * max(p, 0.5)
            return traversal_cost(p_occ[v], _obs[v])

        if cost(start_voxel) == float('inf'):
            return None

        open_heap = [(heuristic(start_voxel), start_voxel)]
        g_score = {start_voxel: 0.0}
        came_from = {}
        closed = set()

        neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                     (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_voxel:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)

            for dx, dy, dz in neighbors:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not in_b(nb) or nb in closed:
                    continue
                # NOTE (2026-10-05, indoorcd_128 fix): A* is allowed to PLAN
                # through blind-spot voxels (unobserved, in the lidar's blind
                # cone). The planner must see vertical routes — e.g. over a
                # full-width wall where the only path is straight up. Safety is
                # enforced at EXECUTION by safe_to_enter (D5 denies blind-spot
                # entry) + the S16 peek maneuver (observe from the side, then
                # move). Separation of concerns: planner sees the route,
                # controller flies it carefully. (The 2026-10-05 blind-spot skip
                # here caused the indoorcd_128 regression: A* returned no path
                # when the only route was vertical.)
                #
                # Two-pass (2026-10-06): pass 1 (allow_blind=False) blocks
                # blind voxels outright, so A* prefers routes that avoid them.
                # Pass 2 (allow_blind=True) prices them with BLIND_SPOT_PENALTY.
                # The goal itself is never blocked.
                step_cost = cost(nb)
                if is_blind_spot(nb, _obs, current):
                    if not allow_blind and nb != goal_voxel:
                        continue
                    step_cost += BLIND_SPOT_PENALTY
                if step_cost == float('inf'):
                    continue
                tentative_g = g_score[current] + step_cost
                if nb not in g_score or tentative_g < g_score[nb]:
                    g_score[nb] = tentative_g
                    came_from[nb] = current
                    heapq.heappush(open_heap, (tentative_g + heuristic(nb), nb))

        return None  # no path found

    # Pass 1: avoid blind spots entirely.
    path = _astar(allow_blind=False)
    if path is not None:
        return path
    # Pass 2: blind is the only way — allow it with penalty.
    return _astar(allow_blind=True)


def plan_next_waypoint(p_occ, current_pose, goal_pose, observed_mask=None):
    """Plan a path and return the next waypoint to fly to.

    Args:
        p_occ: 3D array of P(occupied).
        current_pose: (x, y, z) continuous.
        goal_pose: (x, y, z) continuous.
        observed_mask: optional 3D bool array (see astar).

    Returns:
        (waypoint_pose, path_voxels, info_dict)
        waypoint_pose is None if no path found.
    """
    shape = p_occ.shape
    if observed_mask is None:
        observed_mask = np.ones(shape, dtype=bool)
    start_v = pose_to_voxel(current_pose, shape)
    goal_v = pose_to_voxel(goal_pose, shape)

    if start_v == goal_v:
        return goal_pose, [goal_v], {"status": "at_goal", "path_len": 1}

    path = astar(p_occ, start_v, goal_v, observed_mask)
    if path is None:
        return None, None, {"status": "no_path", "path_len": 0}

    next_voxel = path[1] if len(path) > 1 else path[0]
    waypoint = voxel_to_pose(next_voxel)

    def _c(v):
        if v == goal_v:
            return 1.0 + LAMBDA_CAUTION * max(float(p_occ[v]), 0.5)
        return traversal_cost(float(p_occ[v]), bool(observed_mask[v]))
    info = {
        "status": "ok",
        "path_len": len(path),
        "path_cost": sum(_c(v) for v in path[1:]),
    }
    return waypoint, path, info
