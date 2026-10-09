"""S16 hierarchical planner integrated with the slow-drone stack.

Plain English: two brains working together.
  Host (slow, 1 Hz): A* on the full P(occupied) map -> list of waypoints.
  TSU (fast, 10 Hz): S2 Ising sampler -> picks the next voxel to step into,
      aiming at the next waypoint while avoiding local obstacles.

This mirrors ROS move_base (global_planner + local_planner) and puts the TSU
in the fast safety-critical loop.

Adapted from thermopilot/s16_agent.py for the slow-drone stack:
  - Uses planner.astar (the real one, with observed_mask and P thresholds)
    instead of the simplified harness.astar_path.
  - Works with the slow-drone coordinate conventions.
  - S2Agent is imported from thermopilot (not rewritten).

Z1 fit (from thermopilot FORMULATION_SEARCH.md, red-teamed):
  S2: 18 spins (K=6 x T=3), degree 15, max|J|=1.0, max|h|=4.8. Fits.
"""
import sys
import os

# Import S2Agent from thermopilot without rewriting it.
_THERMOPILOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "thermopilot")
sys.path.insert(0, _THERMOPILOT_DIR)
from harness import S2Agent  # noqa: E402

import numpy as np
import planner


class S16Planner:
    """Hierarchical host + TSU planner.

    Host loop (call host_replan at ~1 Hz):
        A* on P(occupied) -> full path -> waypoints (every wpl-th voxel).
    TSU loop (call tsu_act at ~10 Hz):
        Three-mode gate picks the decision method:
          s2:       obstacles seen nearby -> S2 Ising sampler maneuvers.
          direct:   local map seen and clear -> greedy step, full speed.
          cautious: local map too sparse -> greedy step, reduced speed.
        See _local_obs_fraction / _local_obstacle_nearby for the gate.

    The D5 safe_to_enter guard is NOT inside this class — the caller
    (fly.py / head_to_head.py) applies it to the proposed voxel before
    moving, same as the existing stack.
    """

    def __init__(self, waypoint_lookahead=2, stuck_threshold=4,
                 s2_kwargs=None,
                 gate_radius=4, gate_engage=0.6, gate_disengage=0.4,
                 cautious_every=2, obstacle_check_radius=1):
        self.wpl = waypoint_lookahead
        self.stuck_threshold = stuck_threshold
        s2_kwargs = dict(s2_kwargs or {})
        # S2 defaults from thermopilot hardening (T=3 validated).
        # n_sweeps=300: ~60ms/decision in Python sim (hits 10 Hz budget);
        # 10/10 decision agreement with n_sweeps=600 on maze test (2026-10-05).
        # On Z1 hardware this would be microseconds (native sampling).
        s2_kwargs.setdefault("T", 3)
        s2_kwargs.setdefault("n_sweeps", 300)
        self.s2 = S2Agent(**s2_kwargs)
        # --- S2 observation-density gate (2026-10-05 fix, refined per Paul) ---
        # S2's ray-cast costs are meaningless in unexplored space (all P=0.5),
        # and the smoothness coupling turns flat landscapes into limit cycles
        # (see thermopilot/S16_FAILURE_ANALYSIS.md).
        #
        # Paul's refinement: "empty is safe" — don't treat "I see empty
        # space" the same as "I can't see". Three modes:
        #   s2:       obstacles OBSERVED nearby -> S2's Ising sampler
        #             maneuvers around them (TSU earns its keep). Engages on
        #             the obstacle signal alone; a visible obstacle breaks
        #             the cost symmetry that causes limit cycles.
        #   direct:   no obstacles seen AND local map well-observed (>=0.6)
        #             -> greedy step toward waypoint at full speed.
        #   cautious: no obstacles seen AND local map sparse (<0.6) -> truly
        #             unobserved. Greedy step at reduced speed, D5 guards
        #             every move.
        #
        #   gate_radius: local window is (2*r+1)^3 voxels around the drone.
        #       r=4 -> 9x9x9, covers S2's T=3 ray-cast horizon with margin.
        #   gate_engage / gate_disengage: hysteresis on the observation
        #       fraction for the cautious/direct boundary. Leave cautious
        #       at >=0.6, drop back at <0.4.
        #   cautious_every: in cautious mode, the caller's move speed is
        #       divided by N (via info["speed_factor"]). N=2 halves the
        #       effective speed — "proceed with caution". Implemented as speed
        #       scaling, NOT alternating hover: returning pos_v as "hover"
        #       causes move_toward to pull back to voxel center, netting zero
        #       progress (2026-10-05 bug).
        #   obstacle_check_radius: how far to look for obstacles that trigger
        #       S2. Default 1 (immediate neighbors only). S2's T=3 horizon
        #       cannot maneuver around large obstacles (2026-10-05: 5-voxel
        #       wall in indoorcd_048); for those, follow A* waypoints greedily
        #       instead of letting S2 wander. S2 handles "something right in
        #       front of me", A* handles "big wall ahead".
        self.obstacle_check_radius = obstacle_check_radius
        self.gate_radius = gate_radius
        self.gate_engage = gate_engage
        self.gate_disengage = gate_disengage
        self.cautious_every = max(1, int(cautious_every))
        # Three-mode gate state (Paul's refinement 2026-10-05):
        # "s2" (maneuver around seen obstacles), "direct" (clear, go fast),
        # "cautious" (unobserved, proceed slowly). Starts cautious — the map
        # is empty at mission start, and the gate upgrades as it fills in.
        self._mode = "cautious"
        # --- Mode dwell (2026-10-05 oscillation fix) ---
        # The obstacle_nearby signal has no hysteresis: as the drone moves
        # 1 voxel, a wall flickers in/out of the 1-voxel check radius,
        # ping-ponging s2 <-> direct. Each switch reset _wp_stale_count
        # (only increments in s2 mode), so the stale-waypoint replan never
        # fired and the drone netted zero progress (indoorcd_048: 1 voxel
        # in 200 steps). Fix: once we enter a mode, stay there for at
        # least min_mode_dwell decisions before allowing a switch AWAY.
        # Switching TO s2 is always immediate (safety: don't delay
        # obstacle avoidance).
        self.min_mode_dwell = 5
        self._mode_dwell = 0  # decisions spent in current mode
        # Mode transition log for debugging (list of (step, old, new)).
        self.mode_transitions = []
        # --- D5-refusal feedback / escape state (2026-10-05) ---
        # Paul's spec: "I don't keep slamming my face into the wall, I look
        # to my left or right to see if it's clear and how far. Sometimes I
        # have to take a bit of a step back... the way around can also be
        # up or down." Full 3D: the lateral scan evaluates ALL 5 other
        # cardinal directions (not just planar left/right) — up/down are
        # equally valid escape routes.
        #
        # The caller (fly.py / head_to_head_v2.py) applies D5 safe_to_enter
        # to every proposed voxel. When D5 refuses, the caller must call
        # notify_refusal() so the planner learns instead of re-proposing
        # the same blocked voxel forever (indoorcd_048: 101 refusals).
        #
        # Priority: lateral move (cheapest) -> step back -> host replan.
        self.refusal_escape_threshold = 3   # consecutive refusals -> escape
        self.escape_step_budget = 20        # max escape steps before replan
        self.refused_memory_window = 50     # decisions a refused voxel is
                                            # avoided by greedy proposals
        self._refusal_counts = {}  # (pos_v, refused_v) -> consecutive count
        self._refused_voxels = {}  # refused_v -> step of last refusal
        self._escape = None        # None or dict(phase, target,
                                   #   blocked_dir, steps)
        # Peek maneuver (2026-10-05): when D5 denies a blind-spot vertical
        # (sensor can't see there), don't just deny forever — move laterally
        # to a vantage where the denied voxel enters the elevation fan,
        # let the scan observe it, then retry. None or dict(goal_v,
        # peek_pos, tries).
        self._peek = None
        self.peek_max_tries = 3    # peek positions to try before giving up
        # Stuck-peek detector (2026-10-05 red-team fix): the peek branch
        # preempts the escape branch, so a peek toward an unreachable
        # vantage livelocks (greedy 2-cycles, tries never increments because
        # the vantage is never reached). Track recent positions; on
        # sustained oscillation, abort the peek and hand off to escape.
        self._peek_pos_hist = []   # recent pos_v during peek (max 4)
        self._peek_stuck_count = 0
        self.peek_stuck_threshold = 12  # 2-cycle repeats before abort
        # Global oscillation detector (2026-10-05 red-team fix, part 2):
        # greedy 2-cycles can happen OUTSIDE the peek branch too (local
        # minimum between two voxels when the direct path is blocked).
        # Track voxel history at the tsu_act level; on sustained A-B-A
        # oscillation in any mode, force escape/replan.
        self._osc_hist = []        # recent pos_v across all modes (max 6)
        self._osc_hist_cont = []   # continuous positions for displacement
        self._osc_count = 0
        self.osc_sustain_windows = 3  # consecutive low-displacement windows before forcing escape
        self._wp = None            # current waypoint (set in tsu_act; used
                                   #   to score escape directions)
        self.n_d5_refusals = 0
        self.n_escapes = 0          # lateral/stepback escapes triggered
        self.n_escape_replans = 0   # escapes that fell through to replan
        self._last_wp = None       # last waypoint targeted (stale detection)
        self._wp_stale_count = 0   # consecutive decisions on same waypoint
        self.wp_stale_threshold = 30  # S2 decisions on same wp -> force replan
        self._waypoints = []       # list of voxel tuples
        self._step = 0
        self._last_pos = None      # voxel tuple
        self._last_pos_cont = None   # continuous pose (for 10 Hz sub-stepping)
        self._best_wp_dist = None    # best distance to current waypoint
        self._wp_noprogress = 0      # sub-steps without waypoint progress
        self._stuck_count = 0
        self._goal_fallback = False
        self.n_host_replans = 0
        self.n_host_failures = 0   # A* returned None
        self.n_tsu_decisions = 0
        self.n_s2_decisions = 0        # S2 maneuvering (obstacles nearby)
        self.n_direct_decisions = 0    # direct fast path (clear space)
        self.n_cautious_decisions = 0  # cautious fallback (unobserved)
        self.tsu_time_total = 0.0

    def host_replan(self, pos_v, goal_v, p_occ, observed_mask):
        """Host slow loop: A* -> waypoint list. Returns waypoint list."""
        import time
        t0 = time.time()
        path = planner.astar(p_occ, pos_v, goal_v,
                             observed_mask=observed_mask)
        dt = time.time() - t0
        if path is None or len(path) < 2:
            self._waypoints = [goal_v]
            if path is None:
                self.n_host_failures += 1
        else:
            self._waypoints = path[1::self.wpl]
            if self._waypoints[-1] != goal_v:
                self._waypoints.append(goal_v)
        self.n_host_replans += 1
        return self._waypoints, dt

    def _need_replan(self, pos_v, goal_v):
        if not self._waypoints:
            return True
        if self._stuck_count >= self.stuck_threshold:
            return True
        # Waypoint reached (or adjacent): advance.
        wp = self._waypoints[0]
        if (abs(pos_v[0] - wp[0]) + abs(pos_v[1] - wp[1])
                + abs(pos_v[2] - wp[2]) <= 1):
            return True
        return False

    def _local_obs_fraction(self, pos_v, observed_mask):
        """Fraction of observed voxels in the local window around pos_v.

        Window is (2*gate_radius+1)^3, clipped to map bounds. Uses the
        caller-supplied observed_mask (weak-evidence-aware).
        """
        r = self.gate_radius
        nx, ny, nz = observed_mask.shape
        x0, x1 = max(0, pos_v[0] - r), min(nx, pos_v[0] + r + 1)
        y0, y1 = max(0, pos_v[1] - r), min(ny, pos_v[1] + r + 1)
        z0, z1 = max(0, pos_v[2] - r), min(nz, pos_v[2] + r + 1)
        window = observed_mask[x0:x1, y0:y1, z0:z1]
        if window.size == 0:
            return 0.0
        return float(window.mean())

    def _local_obstacle_nearby(self, pos_v, wp, p_occ, observed_mask):
        """True if an OBSERVED obstacle blocks the path toward the waypoint.

        Paul's refinement: "is there something nearby I need to avoid?" —
        not just "is there a wall somewhere in the window", but "is something
        IN THE WAY". Checks the straight-line ray from pos_v toward wp for
        observed voxels with P > p_block, up to gate_radius voxels out, plus
        the 6-neighborhood for immediate threats S2's ray-cast would see.

        Unobserved voxels don't count — "I can't see" is handled by the
        cautious mode, not by S2. Only observed obstacles trigger S2's
        maneuvering, because S2 needs informative costs to plan with.

        Uses the S2 agent's p_block threshold for consistency.

        obstacle_check_radius (default 1) limits how far we look: S2's T=3
        horizon can only maneuver around small obstacles. Large walls are
        A*'s job — S2 trying to maneuver around them causes wandering
        (2026-10-05: indoorcd_048 5-voxel wall).

        2026-10-05 fix: REMOVED the 6-neighborhood check. It caused S2 to
        engage whenever ANY adjacent voxel was a wall — e.g., flying
        alongside a wall for 20+ steps on A*'s detour (indoorcd_048).
        S2 then tried to "maneuver" around a wall that A* was already
        routing around, interfering with a correct plan. Now S2 only
        engages when an obstacle is IN THE WAY of the greedy step toward
        the waypoint (the ray direction only).
        """
        r = self.obstacle_check_radius
        nx, ny, nz = p_occ.shape
        # Only the ray toward the waypoint — is something IN THE WAY?
        # The 6-neighborhood check was removed (2026-10-05): flying
        # alongside a wall is not a maneuvering situation; A* handles it.
        dx = int(np.sign(wp[0] - pos_v[0]))
        dy = int(np.sign(wp[1] - pos_v[1]))
        dz = int(np.sign(wp[2] - pos_v[2]))
        if (dx, dy, dz) == (0, 0, 0):
            return False
        for s in range(1, r + 1):
            v = (pos_v[0] + s * dx, pos_v[1] + s * dy,
                 pos_v[2] + s * dz)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny
                    and 0 <= v[2] < nz):
                break
            if bool(observed_mask[v]) and float(p_occ[v]) > self.s2.p_block:
                return True
        return False

    def _greedy_step(self, pos_v, wp, p_occ, observed_mask=None):
        """Single greedy step toward the waypoint.

        Picks the 6-neighbor of pos_v minimizing manhattan distance to wp.
        Does NOT commit to multi-step plans — one voxel at a time.
        The caller's D5 safe_to_enter guard still applies to the result.

        Used by both DIRECT mode (full speed, clear space) and CAUTIOUS
        mode (reduced speed, unobserved space) — the difference is tempo,
        not direction.

        If observed_mask is given, it is used for refusal-memory only.

        NOTE (2026-10-05, indoorcd_128 fix): the greedy step PROPOSES
        blind-spot moves (unobserved voxels in the lidar's blind cone)
        instead of skipping them. D5 (safe_to_enter) denies the move, the
        caller reports via notify_refusal, and the S16 peek maneuver observes
        from the side before retrying. Separation of concerns: propose the
        route, let D5 + peek fly it carefully. (Skipping blind spots here
        prevented the peek from ever triggering.)
        Returns (next_voxel, info_dict).
        """
        nx, ny, nz = p_occ.shape
        # Two-pass: first avoid recently-refused voxels (D5 already said no
        # to them — don't propose them again until the map updates). If
        # every neighbor was recently refused, allow them on the second
        # pass; D5 still guards the actual move.
        best = None
        best_score = float("inf")
        for allow_refused in (False, True):
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                v = (pos_v[0] + dx, pos_v[1] + dy, pos_v[2] + dz)
                if not (0 <= v[0] < nx and 0 <= v[1] < ny
                        and 0 <= v[2] < nz):
                    continue
                if not allow_refused and self._is_recently_refused(v):
                    continue
                # NOTE (2026-10-05, indoorcd_128 fix): do NOT skip blind-spot
                # moves here. Propose them; D5 denies; notify_refusal plans
                # the peek maneuver. See docstring.
                d = (abs(v[0] - wp[0]) + abs(v[1] - wp[1])
                     + abs(v[2] - wp[2]))
                # D5 handles hard safety; keep scoring simple/honest.
                if d < best_score:
                    best_score = d
                    best = v
            if best is not None:
                break
        if best is None:
            best = pos_v  # nowhere to go: hover
        return best, {"method": "greedy-step"}

    def _is_recently_refused(self, v):
        """True if D5 refused this voxel within the memory window."""
        last = self._refused_voxels.get(v)
        return last is not None and (self._step - last) < self.refused_memory_window

    def notify_refusal(self, refused_v, pos_v, p_occ, observed_mask):
        """D5 refused the proposed voxel — learn from it. CALLER MUST CALL.

        Called by fly.py / head_to_head_v2.py in the safe_to_enter refusal
        branch. Tracks consecutive refusals per (position, direction); after
        `refusal_escape_threshold` consecutive refusals of the same blocked
        move, plans an escape: lateral scan (full 3D: all 5 other cardinals)
        -> step back -> forced host replan.

        Paul's mental model: "must navigate toward designated target, path
        may be occluded/blocked/unknown, I must figure out how to move to
        target safely" — not "must go straight".
        """
        refused_v = tuple(int(c) for c in refused_v)
        pos_v = tuple(int(c) for c in pos_v)
        self.n_d5_refusals += 1
        self._refused_voxels[refused_v] = self._step
        key = (pos_v, refused_v)
        self._refusal_counts[key] = self._refusal_counts.get(key, 0) + 1
        # Blind-spot denial: the sensor can't see there. Don't just deny
        # forever — plan a peek maneuver (lateral vantage -> observe ->
        # retry). Takes priority over the generic escape.
        if (self._peek is None
                and planner.is_blind_spot(refused_v, observed_mask, pos_v)):
            self._plan_peek(refused_v, pos_v, p_occ, observed_mask)
            self._refusal_counts[key] = 0
            return
        if self._refusal_counts[key] >= self.refusal_escape_threshold \
                and self._escape is None:
            self._plan_escape(refused_v, pos_v, p_occ, observed_mask)
            # Escape planned (or replan forced): reset the count so we
            # don't immediately re-trigger on the next refusal.
            self._refusal_counts[key] = 0

    def _plan_peek(self, goal_v, pos_v, p_occ, observed_mask, tries=1):
        """Plan a peek maneuver: move laterally so goal_v enters the lidar's
        elevation fan, let the host scan observe it, then retry the move.

        Paul's "look before you leap": the sensor can't see straight up/down,
        so look from the side first.

        (2026-10-05 red-team fix): the vantage must be REACHABLE. The old
        code picked the closest lateral vantage without checking the drone
        could get there — when the vantage was behind a wall, the greedy
        stepper 2-cycled toward it forever (tries never incremented because
        the vantage was never reached; the give-up branch was dead code).
        Candidates are now reachability-checked via A* before committing.
        """
        import math
        nx, ny, nz = p_occ.shape
        gx, gy, gz = goal_v
        # Collect candidates sorted by manhattan distance.
        cands = []
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                if dx == 0 and dy == 0:
                    continue
                qx, qy, qz = pos_v[0] + dx, pos_v[1] + dy, pos_v[2]
                if not (0 <= qx < nx and 0 <= qy < ny and 0 <= qz < nz):
                    continue
                q = (qx, qy, qz)
                # Must be enterable: observed-free, or unobserved but in
                # the horizontal fan (D5 allows those).
                if bool(observed_mask[q]) and float(p_occ[q]) > planner.P_OBSTACLE:
                    continue
                # Elevation from q to goal_v within the fan?
                _dz = (gz + 0.5) - (qz + 0.5)
                _dh = math.hypot((gx + 0.5) - (qx + 0.5),
                                 (gy + 0.5) - (qy + 0.5))
                if _dh < 1e-9:
                    continue
                _elev = math.atan2(abs(_dz), _dh)
                if _elev > planner.SENSOR_MAX_ELEV:
                    continue
                # Line of sight q -> goal_v clear of observed obstacles?
                _blocked = False
                _sc = (qx + 0.5, qy + 0.5, qz + 0.5)
                _ec = (gx + 0.5, gy + 0.5, gz + 0.5)
                for _w in planner._intermediate_voxels(_sc, _ec,
                                                      p_occ.shape):
                    if (bool(observed_mask[_w])
                            and float(p_occ[_w]) > planner.P_OBSTACLE):
                        _blocked = True
                        break
                if _blocked:
                    continue
                cands.append((abs(dx) + abs(dy), q))
        cands.sort(key=lambda c: c[0])
        # Pick the closest REACHABLE vantage (2026-10-05 red-team fix).
        best = None
        for _, q in cands[:8]:  # A* is cheap at this range; check top 8
            if q == pos_v:
                best = q
                break
            if planner.astar(p_occ, pos_v, q, observed_mask) is not None:
                best = q
                break
        if best is not None:
            self._peek = {"goal_v": goal_v, "peek_pos": best, "tries": tries}
            self.n_peeks = getattr(self, "n_peeks", 0) + 1
        else:
            # No reachable vantage: clear peek, force a host replan to
            # route around the unobservable voxel.
            self._peek = None
            self._waypoints = []
            self.n_peek_failures = getattr(self, "n_peek_failures", 0) + 1

    def _plan_escape(self, refused_v, pos_v, p_occ, observed_mask):
        """Choose an escape maneuver after repeated D5 refusals.

        Priority: lateral move (cheapest) -> step back -> host replan.
        The lateral scan evaluates ALL 5 other cardinal directions in full
        3D (left/right/forward-back-plane AND up/down) — per Paul, the way
        around can be up or down too.
        """
        nx, ny, nz = p_occ.shape
        # Blocked direction: unit cardinal from pos_v to refused_v.
        # Proposals are always 6-neighbors, so this is a clean cardinal.
        blocked_dir = (int(np.sign(refused_v[0] - pos_v[0])),
                       int(np.sign(refused_v[1] - pos_v[1])),
                       int(np.sign(refused_v[2] - pos_v[2])))
        cardinals = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                     (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        wp = self._wp  # may be None if tsu_act hasn't run yet

        # --- 1. Lateral scan: all 5 other cardinals ---
        best = None
        best_score = float("-inf")
        for d in cardinals:
            if d == blocked_dir:
                continue
            cand = (pos_v[0] + d[0], pos_v[1] + d[1], pos_v[2] + d[2])
            if not planner.safe_to_enter(cand, p_occ, observed_mask, pos_v):
                continue
            # Clearance: how far does the free path extend in direction d?
            # (Paul: "see if it's clear and how far".)
            clearance = 0
            for s in range(1, 5):
                v = (pos_v[0] + s * d[0], pos_v[1] + s * d[1],
                     pos_v[2] + s * d[2])
                if planner.safe_to_enter(v, p_occ, observed_mask, pos_v):
                    clearance += 1
                else:
                    break
            # Score: clearance matters most (room to maneuver), then
            # waypoint progress. Weights are heuristic, documented as such.
            prog_term = 0.0
            if wp is not None:
                prog_term = -0.5 * (abs(cand[0] - wp[0])
                                    + abs(cand[1] - wp[1])
                                    + abs(cand[2] - wp[2]))
            score = 2.0 * clearance + prog_term
            if score > best_score:
                best_score = score
                best = cand
        if best is not None and best != pos_v:
            self._escape = {"phase": "lateral", "target": best,
                            "blocked_dir": blocked_dir, "steps": 0}
            self.n_escapes += 1
            return

        # --- 2. Step back: reverse the blocked direction for a better view ---
        back = (pos_v[0] - blocked_dir[0], pos_v[1] - blocked_dir[1],
                pos_v[2] - blocked_dir[2])
        if back != pos_v and planner.safe_to_enter(back, p_occ,
                                                   observed_mask, pos_v):
            self._escape = {"phase": "stepback", "target": back,
                            "blocked_dir": blocked_dir, "steps": 0}
            self.n_escapes += 1
            return

        # --- 3. Nothing works locally: force a host replan ---
        # The map has new obstacle info now; A* can route around globally.
        self._waypoints = []
        self.n_escape_replans += 1

    def _escape_step(self, pos_v, p_occ, observed_mask=None):
        """Execute one step of the active escape. Returns (voxel, done)."""
        esc = self._escape
        target = esc["target"]
        if pos_v == target:
            self._escape = None
            return pos_v, True
        esc["steps"] += 1
        if esc["steps"] > self.escape_step_budget:
            # Escape isn't converging — give up and replan globally.
            self._escape = None
            self._waypoints = []
            self.n_escape_replans += 1
            return pos_v, True
        nxt, _ = self._greedy_step(pos_v, target, p_occ, observed_mask)
        return nxt, False

    def tsu_act(self, pos_v, goal_v, p_occ, observed_mask,
              pos_cont=None):
        """TSU fast loop: one S2 decision. Returns (next_voxel, info).

        Handles stuck detection, goal fallback, and waypoint management
        internally. The caller must still apply the D5 safe_to_enter guard
        to the returned voxel before moving.

        Args:
            pos_v: current voxel (integer tuple) — for S2 and host planning.
            goal_v: goal voxel (integer tuple).
            p_occ: P(occupied) array.
            observed_mask: bool array.
            pos_cont: optional continuous (x,y,z) pose. When given (10 Hz
                sub-stepping), stuck detection uses Euclidean distance on
                the continuous pose instead of voxel equality, because
                0.1-voxel sub-steps don't change the integer voxel for
                ~10 sub-steps during normal movement.
        """
        import time
        # --- Global oscillation detector (2026-10-05 red-team fix,
        # --- revised 2026-10-06 for F1) ---
        # Greedy 2-cycles (A-B-A-B...) happen when the direct path is
        # blocked and greedy has no memory — in ANY mode, not just peek.
        #
        # Detection via NET DISPLACEMENT, not unique-voxel count.
        # (2026-10-06 red-team round 2, F1: the old <=5-unique-voxels rule
        # false-fired every ~43 decisions because 40 decisions at 0.1
        # vox/sub-step covers ~4 voxels of steady flight, so <=5 unique
        # was almost always true — 10-25% wasted motion. Net displacement
        # distinguishes "moving" (displacement ~4) from "stuck"
        # (displacement ~0) regardless of speed.)
        _pv = tuple(int(c) for c in pos_v)
        self._osc_hist.append(_pv)
        # Continuous positions for net-displacement (F1): voxel
        # quantization loses slow progress (0.05 vox/step = 2 voxels/40
        # decisions, but int-truncation shows displacement ~1.0).
        _pc = tuple(float(c) for c in pos_cont) if pos_cont is not None \
            else (float(_pv[0]) + 0.5, float(_pv[1]) + 0.5,
                  float(_pv[2]) + 0.5)
        self._osc_hist_cont.append(_pc)
        if len(self._osc_hist) > 50:
            self._osc_hist.pop(0)
        if len(self._osc_hist_cont) > 50:
            self._osc_hist_cont.pop(0)
        if len(self._osc_hist_cont) >= 40:
            _w = self._osc_hist_cont[-40:]
            _dx = _w[-1][0] - _w[0][0]
            _dy = _w[-1][1] - _w[0][1]
            _dz = _w[-1][2] - _w[0][2]
            _net_disp = (_dx*_dx + _dy*_dy + _dz*_dz) ** 0.5
            if _net_disp < 1.5:
                self._osc_count += 1
            else:
                self._osc_count = 0
        if self._osc_count >= self.osc_sustain_windows:
            # Sustained local minimum: abort whatever we're doing.
            # Capture the locus (most-visited voxel) BEFORE clearing —
            # F11 needs it if retreat is blocked.
            from collections import Counter as _Ctr
            _locus = _Ctr(self._osc_hist).most_common(1)[0][0] \
                if self._osc_hist else _pv
            self._osc_count = 0
            self._osc_hist = []
            self._osc_hist_cont = []
            self._peek = None
            self._peek_pos_hist = []
            self._peek_stuck_count = 0
            self._waypoints = []  # force host replan
            self.n_osc_breaks = getattr(self, "n_osc_breaks", 0) + 1
            # Retreat: step directly away from the oscillation locus to
            # break the local-minimum geometry, then replan from there.
            # (The replan from a new position often finds what the stuck
            # position couldn't — different viewing angles, different
            # greedy tie-breaks.) Single step — the caller moves one
            # voxel per decision.
            _retreat = (_pv[0], _pv[1] + 1, _pv[2])  # +y = away from wall
            _nx, _ny, _nz = p_occ.shape
            if (0 <= _retreat[0] < _nx and 0 <= _retreat[1] < _ny
                    and 0 <= _retreat[2] < _nz
                    and planner.safe_to_enter(_retreat, p_occ,
                                              observed_mask, _pv)):
                # Override: go to retreat point this decision.
                self.n_tsu_decisions += 1
                return _retreat, {
                    "method": "osc_retreat",
                    "escape": True,
                    "speed_factor": 1.0,
                    "goal_fallback": self._goal_fallback,
                    "tsu_time": 0.0,
                    "waypoint": None,
                    "mode": self._mode,
                    "s2_engaged": False,
                    "local_obs_frac": self._local_obs_fraction(
                        _pv, observed_mask),
                    "obstacle_nearby": False,
                    "stale_wp_replan": True,
                }
            # Retreat blocked: don't just clear waypoints and hope A*
            # finds a new route — A* on the (possibly wrong) believed map
            # will route straight back through the trap. Instead, trigger
            # the escape system with the oscillation locus as the "blocked"
            # voxel: lateral exploration around the trap, not believed-map
            # replan into it. (2026-10-06 red-team round 2, F11: blocked
            # retreat + believed-map replan looped through the trap
            # indefinitely; only the D5 refusal budget terminated it.)
            # _plan_escape falls back to host replan (step 3) if no
            # lateral/stepback move is available, so this is a strict
            # improvement over the old fall-through.
            self._plan_escape(_locus, _pv, p_occ, observed_mask)
            # Fall through to normal planning; the escape (if planned)
            # will be picked up by _escape_step on the next decision.
        # Host replan if needed (uses voxel-equality stuck as preliminary).
        # The robust stuck detection (waypoint progress) runs after wp is known.
        prelim_stuck = False
        if pos_cont is not None:
            if self._last_pos_cont is not None:
                d = float(np.linalg.norm(np.array(pos_cont)
                                         - np.array(self._last_pos_cont)))
                if d < 0.05:
                    self._stuck_count += 1
                else:
                    self._stuck_count = 0
            self._last_pos_cont = tuple(pos_cont)
            self._last_pos = pos_v
        else:
            if self._last_pos is not None and pos_v == self._last_pos:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_pos = pos_v
            self._last_pos_cont = None

        if self._stuck_count >= self.stuck_threshold:
            prelim_stuck = True

        # --- host replan if needed ---
        # (temporarily force _goal_fallback for _need_replan check if prelim stuck)
        _fb_save = self._goal_fallback
        if prelim_stuck:
            self._goal_fallback = True
        need = self._need_replan(pos_v, goal_v)
        self._goal_fallback = _fb_save
        if need:
            self.host_replan(pos_v, goal_v, p_occ, observed_mask)
            def manhattan(a, b):
                return (abs(a[0] - b[0]) + abs(a[1] - b[1])
                        + abs(a[2] - b[2]))
            self._waypoints = [w for w in self._waypoints
                               if manhattan(pos_v, w) > 1] or [goal_v]
            self._stuck_count = 0
            self._wp_noprogress = 0
            self._best_wp_dist = None
            # Fresh waypoints: clear fallback (unless A* failed and we only
            # have the goal as waypoint, in which case fallback is moot).
            if not prelim_stuck:
                self._goal_fallback = False

        self._step += 1

        # --- determine current waypoint ---
        wp = goal_v if (self._goal_fallback or prelim_stuck) else self._waypoints[0]
        if prelim_stuck and not self._goal_fallback:
            self._goal_fallback = True

        # --- robust stuck detection: waypoint progress ---
        # If we're not getting closer to the waypoint, we're stuck
        # (catches oscillation that per-step movement checks miss).
        if pos_cont is not None:
            wp_pose = planner.voxel_to_pose(wp)
            wp_dist = float(np.linalg.norm(np.array(pos_cont)
                                           - np.array(wp_pose)))
            if self._best_wp_dist is None or wp_dist < self._best_wp_dist - 0.1:
                self._best_wp_dist = wp_dist
                self._wp_noprogress = 0
            else:
                self._wp_noprogress += 1
            if self._wp_noprogress >= self.stuck_threshold:
                self._goal_fallback = True
                # Reset to give the fallback a chance
                self._wp_noprogress = 0
                self._best_wp_dist = None
                wp = goal_v
        # Clear fallback when making progress toward goal
        if self._goal_fallback and pos_cont is not None:
            # (fallback clears when waypoint progress resumes; handled above
            # via _best_wp_dist reset on replan)
            pass
        # Remember the current waypoint for escape-direction scoring in
        # notify_refusal (lateral scan prefers directions with clearance
        # AND waypoint progress).
        self._wp = wp
        # --- Peek maneuver (2026-10-05): observe before ascending/descending.
        # If D5 denied a blind-spot vertical, we're navigating to a lateral
        # vantage. Once the goal voxel is observed, clear the peek and let
        # normal planning retry the vertical move.
        if self._peek is not None:
            _pg = self._peek["goal_v"]
            if bool(observed_mask[_pg]):
                self._peek = None  # observed: retry normally next decision
                self._peek_pos_hist = []
                self._peek_stuck_count = 0
            else:
                # Stuck-peek detector (2026-10-05 red-team fix): the peek
                # branch preempts the escape branch, so a peek toward an
                # unreachable vantage livelocks — greedy 2-cycles, tries
                # never increments (vantage never reached), escape never
                # fires. Detect A-B-A oscillation and hand off to escape.
                self._peek_pos_hist.append(pos_v)
                if len(self._peek_pos_hist) > 4:
                    self._peek_pos_hist.pop(0)
                if (len(self._peek_pos_hist) >= 3
                        and (self._peek_pos_hist[-1]
                             == self._peek_pos_hist[-3])):
                    self._peek_stuck_count += 1
                else:
                    self._peek_stuck_count = 0
                if self._peek_stuck_count >= self.peek_stuck_threshold:
                    # Abort the stuck peek; the escape system takes over.
                    # Falls through to the escape branch below this decision.
                    self._peek = None
                    self._peek_pos_hist = []
                    self._peek_stuck_count = 0
                    self.n_peek_failures = getattr(
                        self, "n_peek_failures", 0) + 1
                    self._plan_escape(_pg, pos_v, p_occ, observed_mask)
                else:
                    _pp = self._peek["peek_pos"]
                    if pos_v == _pp:
                        # Reached vantage but goal still unobserved (occluded
                        # or bad angle). Try once more from here, else give up.
                        self._peek["tries"] += 1
                        if self._peek["tries"] > self.peek_max_tries:
                            self._peek = None
                            self._waypoints = []  # force host replan
                            self.n_peek_failures = getattr(
                                self, "n_peek_failures", 0) + 1
                        else:
                            # Re-plan peek from the new vantage. Preserve
                            # the tries count — _plan_peek defaults to
                            # tries=1, which used to reset the counter and
                            # make the give-up branch dead code
                            # (2026-10-05 red-team).
                            _tries = self._peek["tries"]
                            self._plan_peek(_pg, pos_v, p_occ,
                                            observed_mask, tries=_tries)
                            if self._peek is None:
                                pass  # _plan_peek forced replan; fall thru
                            else:
                                _pp = self._peek["peek_pos"]
                    if self._peek is not None:
                        nxt, _pinfo = self._greedy_step(pos_v, _pp, p_occ,
                                                        observed_mask)
                        self.n_tsu_decisions += 1
                        return nxt, {
                            "method": "peek",
                            "peek": True,
                            "peek_goal": _pg,
                            "speed_factor": 1.0,
                            "goal_fallback": self._goal_fallback,
                            "tsu_time": 0.0,
                            "waypoint": wp,
                            "mode": self._mode,
                            "s2_engaged": False,
                            "local_obs_frac": self._local_obs_fraction(
                                pos_v, observed_mask),
                            "obstacle_nearby": False,
                            "stale_wp_replan": False,
                        }
        # --- D5-refusal escape override (2026-10-05) ---
        # If an escape is active (planned by notify_refusal after repeated
        # D5 refusals), it takes priority over the gate modes: execute one
        # deliberate escape step instead of re-proposing the blocked move.
        # Paul: "look left or right... step back to assess... the way around
        # can also be up or down." Full 3D — _plan_escape already scored
        # all 5 other cardinals.
        if self._escape is not None:
            import time as _etime
            _et0 = _etime.time()
            nxt, _done = self._escape_step(pos_v, p_occ, observed_mask)
            esc_phase = self._escape["phase"] if self._escape else "done"
            self.n_tsu_decisions += 1
            _edt = _etime.time() - _et0
            self.tsu_time_total += _edt
            return nxt, {
                "method": "escape",
                "escape": True,
                "escape_phase": esc_phase,
                "speed_factor": 1.0,  # deliberate maneuver, full speed
                "goal_fallback": self._goal_fallback,
                "tsu_time": _edt,
                "waypoint": wp,
                "mode": self._mode,
                "s2_engaged": False,
                "local_obs_frac": self._local_obs_fraction(
                    pos_v, observed_mask),
                "obstacle_nearby": False,
                "stale_wp_replan": False,
            }
        # --- S2 observation-density gate (2026-10-05 fix, refined per Paul) ---
        # Paul's nuance: "empty is safe" — don't treat "I see empty space"
        # the same as "I can't see". Three modes:
        #   s2:       obstacles OBSERVED nearby -> S2's Ising sampler
        #             maneuvers around them. This is where the TSU earns its
        #             keep. Engages on the obstacle signal alone — a visible
        #             obstacle breaks the cost symmetry that causes limit
        #             cycles, so S2 has informative costs to work with.
        #   direct:   no obstacles seen AND local map well-observed (>=0.6)
        #             -> greedy step toward waypoint at full speed. S2's
        #             complex planning is overkill for empty space.
        #   cautious: no obstacles seen AND local map sparse (<0.6) -> truly
        #             unobserved. Greedy step at reduced speed, D5 guards
        #             every move. "Proceed with caution" while sensors fill
        #             in the map.
        # The catastrophic failure (S16_FAILURE_ANALYSIS.md) was NO obstacles
        # visible + sparse map = flat costs = limit cycles. As long as S2
        # sees at least one obstacle OR the map is clear-and-observed, its
        # costs are informative. Only the truly-unknown case needs caution.
        # Hysteresis on the cautious/direct boundary (0.4/0.6) prevents
        # oscillation. The obstacle signal is stable within a host step
        # (p_occ/obs_mask only update per host step), so S2 doesn't flicker.
        import time
        obs_frac = self._local_obs_fraction(pos_v, observed_mask)
        obstacle_nearby = self._local_obstacle_nearby(
            pos_v, wp, p_occ, observed_mask)
        # --- Mode selection with dwell (2026-10-05 oscillation fix) ---
        # Compute the DESIRED mode, then enforce dwell before switching away.
        if obstacle_nearby:
            desired = "s2"
        elif self._mode == "cautious":
            if obs_frac >= self.gate_engage:
                desired = "direct"
            else:
                desired = "cautious"
        else:
            if obs_frac < self.gate_disengage:
                desired = "cautious"
            else:
                desired = "direct"
        # Dwell enforcement: switching TO s2 is immediate (safety).
        # Switching AWAY from any mode requires min_mode_dwell decisions.
        # This stops the s2<->direct ping-pong that trapped indoorcd_048.
        if desired != self._mode:
            if desired == "s2" or self._mode_dwell >= self.min_mode_dwell:
                self.mode_transitions.append(
                    (self._step, self._mode, desired))
                self._mode = desired
                self._mode_dwell = 0
            # else: hold current mode; dwell not yet satisfied.
        self._mode_dwell += 1

        t0 = time.time()
        if self._mode == "s2":
            # Blind-spot aware (2026-10-05): forbid moves D5 would deny for
            # unobserved targets (sensor blind cone / blocked line of sight).
            # S2's ray costs price OBSERVED obstacles; they don't know about
            # the blind spot, so S2 kept proposing blind verticals that D5
            # denied -> refusal loop -> max_steps. Now S2 plans around it.
            _fb = set()
            for _i, _d in enumerate(self.s2.moves):
                _cand = (pos_v[0] + _d[0], pos_v[1] + _d[1],
                         pos_v[2] + _d[2])
                if (0 <= _cand[0] < p_occ.shape[0]
                        and 0 <= _cand[1] < p_occ.shape[1]
                        and 0 <= _cand[2] < p_occ.shape[2]
                        and not bool(observed_mask[_cand])
                        and not planner.safe_to_enter(
                            _cand, p_occ, observed_mask, pos_v)):
                    _fb.add(_i)
            nxt, info = self.s2.act(pos_v, wp, p_occ,
                                    forbidden_moves=_fb or None)
            self.n_s2_decisions += 1
            info["speed_factor"] = 1.0
        elif self._mode == "direct":
            # Clear space, full speed: greedy step every decision.
            self.n_direct_decisions += 1
            nxt, info = self._greedy_step(pos_v, wp, p_occ, observed_mask)
            info["speed_factor"] = 1.0
        else:
            # Cautious: greedy step, but at reduced speed. The caller
            # multiplies its move speed by speed_factor. We do NOT alternate
            # hover/move here — returning pos_v as "hover" causes the
            # caller's move_toward to pull the drone BACK to the voxel
            # center, undoing progress (2026-10-05 bug: 1 voxel in 200 steps).
            # Reduced speed + D5 guards = "proceed with caution".
            self.n_cautious_decisions += 1
            nxt, info = self._greedy_step(pos_v, wp, p_occ, observed_mask)
            info["speed_factor"] = 1.0 / self.cautious_every
        dt = time.time() - t0
        self.n_tsu_decisions += 1
        self.tsu_time_total += dt
        info["goal_fallback"] = self._goal_fallback
        info["tsu_time"] = dt
        info["waypoint"] = wp
        info["mode"] = self._mode
        info["s2_engaged"] = (self._mode == "s2")
        info["local_obs_frac"] = obs_frac
        info["obstacle_nearby"] = obstacle_nearby

        # --- S2 stale-waypoint detection ---
        # If S2 keeps targeting the SAME waypoint without reaching it, it's
        # oscillating around an obstacle it can't plan around with its T=3
        # horizon (see indoorcd_071 trace: S2 circled a real wall for 40+
        # decisions). Don't let S2 spin forever — force a host replan so A*
        # can find a global detour. Clearing _waypoints triggers _need_replan
        # on the next call. Drop to direct mode meanwhile so we don't keep
        # oscillating while waiting for the replan.
        #
        # 2026-10-05 fix: count s2 decisions on the same waypoint even across
        # mode switches. Previously the count reset on ANY mode change, so
        # the s2<->direct ping-pong (now fixed by dwell) prevented this
        # from ever firing. Only reset when the waypoint actually changes.
        if wp == self._last_wp:
            if self._mode == "s2":
                self._wp_stale_count += 1
            # else: not in s2 mode; hold count (don't reset) so that
            # re-entering s2 on the same waypoint resumes the count.
        else:
            self._wp_stale_count = 0
        self._last_wp = wp
        if self._wp_stale_count >= self.wp_stale_threshold:
            self._waypoints = []  # force host replan next call
            self._wp_stale_count = 0
            self._mode = "direct"
            info["stale_wp_replan"] = True
        else:
            info["stale_wp_replan"] = False
        return nxt, info
