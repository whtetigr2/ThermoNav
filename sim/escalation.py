"""D5 escalation policy — Option A (industry mirror), ported to our sensor set.

Paul picked Option A (2026-10-10) from
surveys/escalation-policy/ESCALATION_OPTIONS.md.

Industry ladder (DJI / PX4 / ArduPilot, reported): Warning -> Hold/Hover ->
Return-to-home -> Land -> Disarm. Hold-first (operator override window) and
most-severe-wins are universal design rules. FAA Part 107 mandates procedures
exist; it prescribes zero thresholds.

Our port: triggers come from OUR measured signals (vis_smooth, far_smooth,
map_age, jitter). Battery / link / nav-health trigger slots exist but are
STUBBED as MISSING (unmodeled in sim — variable-totality: listed, not
invented).

RTH CAVEAT (documented, not silent): the industry RTH assumes a known home,
a navigable path, and working navigation. Our estimator is an open problem
(scan-matching LEGO killed, red-team pass) and our fog is 16x below the FAA
waiver floor. RTH is therefore GATED on pose trustworthiness. With no
estimator the gate stays closed and RTH degrades to land-in-place, loudly
logged. Paul picked A knowing this.

Claim label: host-side classical policy. No TSU claim.
"Z1-legal formulation, validated in simulation" applies to the decider
kernel (escalation_decider.py), not this module.

INTEGRATED BUILD (2026-10-10): D5 integration hooks are WIRED in
mission_dynamics_v5.py (branch muse/escalation-policy-2026-10-10, rebased
onto muse/tree-recon-phase2-2026-10-10). Call site: per outer step, after
tracker.update_stale(). pose_trustworthy hardcoded False. Final action =
max(shipped, escalation) by severity. See INTEGRATION CONTRACT below for
the wired mapping. This module exposes the contract; mission_dynamics_v5
implements the call site (numpy only here — no planner/s16 import).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds. Every number carries provenance:
#   [measured] — from our sweeps, file/line cited
#   [design]   — Option A proposal, not measured
#   MISSING    — unmodeled; stub only
# ---------------------------------------------------------------------------

# Perception-degraded band (shared with FIX 4 stale-map gate)
VIS_DEGRADED = 0.70    # [measured: obscurant_fixes.py STALE_VIS_ENTER]
FAR_DEGRADED = 0.10    # [measured: obscurant_fixes.py STALE_FAR_ENTER]
MAP_STALE_S = 1.0      # [measured: obscurant_fixes.py STALE_AGE_S]
# Hysteresis exits (mirror the tracker's band exits; those are [design]
# in obscurant_fixes.py too — not measured)
VIS_CLEAR = 0.75       # [design: mirrors STALE_VIS_EXIT]
FAR_CLEAR = 0.15       # [design: mirrors STALE_FAR_EXIT]

# Sustained-degradation timers (Option A trigger table)
DEGRADE_SUSTAIN_HOVER_S = 5.0    # [design: Option A, "sustained 5 s -> Hold"]
DEGRADE_SUSTAIN_RTH_S = 30.0    # [design: Option A, "30 s no recovery -> RTH"]

# Hold-first operator override window (industry rule 1)
HOLD_OVERRIDE_S = 10.0  # [design: PX4-style operator window before RTH/land]

# Land-in-place descent (Option A fallback when RTH gated)
LAND_DESCENT_RATE = 0.5  # m/s [design]

# Severity ladder (industry rule 2: most-severe-wins)
CONTINUE, HOVER, RTH, LAND, DISARM = 0, 1, 2, 3, 4
ACTION_NAMES = ["continue", "hover", "RTH", "land", "disarm"]


class TriggerStub:
    """A trigger slot with no sensor behind it. Always inactive; the
    provenance is the point (variable-totality: MISSING is listed)."""

    def __init__(self, name, reason):
        self.name = name
        self.provenance = "MISSING"
        self.reason = reason

    def evaluate(self):
        return False, f"{self.name}: MISSING ({self.reason})"


class EscalationPolicy:
    """Option A escalation policy. Classical, deterministic, host-side.

    Usage (per outer step, AFTER the evidence tracker update):
        action, info = policy.evaluate(vis_smooth, far_smooth, map_age_s,
                                      jitter_active, dt,
                                      pose_trustworthy=False,
                                      operator_override=False)
    """

    def __init__(self):
        # Sustained-degradation timers (sim-seconds of continuous degraded band)
        self._deg_timer = 0.0
        # Hold-first state: pending harsher action + override window
        self._pending = None          # action waiting out the hold window
        self._pending_timer = 0.0
        self._current = CONTINUE
        # MISSING trigger slots (unmodeled in sim)
        self.missing_triggers = {
            "battery": TriggerStub(
                "battery", "no power model in sim"),
            "link": TriggerStub(
                "link", "no comms model in sim"),
            "nav_health": TriggerStub(
                "nav_health", "no GPS indoors by assumption; "
                              "estimator is an open problem"),
        }
        self.log = []  # (t, event, detail) for the mission manifest

    # -- input conditioning -------------------------------------------------
    @staticmethod
    def _degraded(vis, far):
        return vis < VIS_DEGRADED and far < FAR_DEGRADED

    @staticmethod
    def _clear(vis, far):
        return vis > VIS_CLEAR and far > FAR_CLEAR

    # -- policy table (Option A) --------------------------------------------
    def _table_action(self, vis, far, map_age, jitter):
        """Desired severity from the Option A trigger table, ignoring the
        hold window and the RTH gate. Returns (severity, reason)."""
        degraded = self._degraded(vis, far)
        # Row 3: degradation sustained 30 s -> RTH (checked first: severest)
        if degraded and self._deg_timer >= DEGRADE_SUSTAIN_RTH_S:
            return RTH, "degraded sustained 30 s [design]"
        # Row 1: degraded sustained 5 s -> Hold
        if degraded and self._deg_timer >= DEGRADE_SUSTAIN_HOVER_S:
            return HOVER, "degraded sustained 5 s [design]"
        # Row 2: degraded AND map stale -> immediate Hold (ships as FIX 4)
        if degraded and map_age >= MAP_STALE_S:
            return HOVER, "degraded AND map_age>=1.0 s [measured: FIX 4]"
        # Jitter alone does not escalate beyond what staleness already
        # triggers (jitter's effect IS map_age in this sim); it is an input
        # to the decider kernel, not a separate policy row.
        return CONTINUE, "nominal"

    def evaluate(self, vis, far, map_age, jitter, dt,
                 pose_trustworthy=False, operator_override=False, t=0.0):
        """One policy step. Returns (action, info dict).

        pose_trustworthy: RTH gate. False until an estimator clears the
            <0.5 m/mission bar (open problem). RTH degrades to land-in-place.
        operator_override: industry hold-first rule — True cancels a pending
            harsher action back to hover during the override window.
        """
        # Missing slots are evaluated (always inactive) so the table is
        # complete and auditable.
        for stub in self.missing_triggers.values():
            fired, _ = stub.evaluate()
            assert not fired

        # Sustained-degradation timer with hysteresis reset.
        if self._degraded(vis, far):
            self._deg_timer += dt
        elif self._clear(vis, far):
            self._deg_timer = 0.0
        # else: hold timer (in-band hysteresis, neither accumulate nor reset)

        desired, reason = self._table_action(vis, far, map_age, jitter)

        # Industry rule 1: hold-first. Hover IS the hold: continue->hover
        # engages immediately (the shipped FIX 3 / FIX 4 behavior is not
        # delayed). Steps BEYOND hover (RTH/land/disarm) wait out the
        # operator override window AT hover level.
        holding = False
        overridden = False
        if desired <= HOVER:
            # Hover engages immediately; recovery steps down immediately.
            # A pending harsher action is cancelled (recovery wins).
            self._pending, self._pending_timer = None, 0.0
            action = desired
        else:
            # desired in (RTH, LAND, DISARM): hold window at hover level.
            if self._pending != desired:
                self._pending = desired
                self._pending_timer = 0.0
                self.log.append(
                    (t, "hold_window_open",
                     f"pending={ACTION_NAMES[desired]} reason={reason}"))
            self._pending_timer += dt
            if operator_override:
                self.log.append((t, "operator_override",
                                 f"cancelled pending "
                                 f"{ACTION_NAMES[self._pending]}"))
                self._pending, self._pending_timer = None, 0.0
                action = HOVER
                overridden = True
            elif self._pending_timer < HOLD_OVERRIDE_S:
                # Hold at hover level, but never de-escalate below a
                # committed harsher action (most-severe-wins).
                action = max(self._current, HOVER)
                holding = True
            else:
                # Window elapsed: commit the pending action.
                action = self._pending
                self._pending, self._pending_timer = None, 0.0

        # RTH GATE (the documented caveat). Applies to EVERY return path,
        # BEFORE _current is assigned. Invariant: no path emits RTH when
        # the gate is closed, on any step, with any _current history.
        # (Red-team 2026-10-10 caught the hold-window early return skipping
        # this check with a stale pre-gate RTH in _current.)
        gated = False
        if action == RTH and not pose_trustworthy:
            gated = True
            action = LAND  # land-in-place fallback
            self.log.append(
                (t, "rth_gated_land_in_place",
                 "RTH requested per industry ladder; pose trust gate "
                 "CLOSED (estimator: open problem, no <0.5 m/mission "
                 "estimator exists) -> land-in-place fallback"))

        # Commit post-gate state. _current never holds a pre-gate RTH.
        if action != self._current:
            if action < self._current:
                self.log.append((t, "deescalate",
                                 f"now={ACTION_NAMES[action]}"))
            self._current = action

        return action, self._info(reason, gated=gated,
                                 holding=holding, overridden=overridden)

    @staticmethod
    def _info(reason, gated, holding=False, overridden=False):
        return {"reason": reason, "rth_gated": gated,
                "holding_for_override": holding,
                "operator_overridden": overridden,
                "missing_triggers": ["battery", "link", "nav_health"]}

    # -- behavior primitives (integration contract) -------------------------
    # These are called from the mission loop AFTER the Phase 2 rebase.
    # Signatures mirror the loop's (vehicle, pid, ...) conventions.
    # hover: already exists in-loop (FIX 3 / FIX 4 hover_mode). Escalation
    # hover MUST reuse it (contract: final action = max(shipped, escalation)
    # by severity, so escalation never contradicts a shipped hover).

    @staticmethod
    def land_in_place_behavior(vehicle, pid, dt):
        """Vertical descent at LAND_DESCENT_RATE until ground contact.
        Velocity-servo: p_des=pos zeroes the outer P term; the inner PID
        tracks v_des directly. (A creeping position target descends ~100x
        too slow.) [design]"""
        # command(p_des, v_des, p, v): error = p_des - p.
        a_cmd = pid.command(vehicle.pos, (0, 0, -LAND_DESCENT_RATE),
                            vehicle.pos, vehicle.vel)
        vehicle.step(a_cmd)
        landed = vehicle.pos[2] <= 0.05
        return landed

    @staticmethod
    def rth_behavior(vehicle, pid, home_pose, plan_fn, dt):
        """Return-to-home: A* path to home_pose, flown waypoint by waypoint.
        ONLY called when the pose gate is open (pose_trustworthy=True).
        plan_fn: callable(start_xyz, goal_xyz) -> list of waypoints.
        Returns True when home reached."""
        raise NotImplementedError(
            "RTH path execution wires into the Phase 2 reconciled planner; "
            "standalone build provides the policy + gate only.")

    @staticmethod
    def disarm_behavior(log, outcome="stand_down"):
        """Terminal: mission abort. Outcome string matches the existing
        whiteout safe stand-down convention."""
        log["outcome"] = outcome
        return True


# ---------------------------------------------------------------------------
# INTEGRATION CONTRACT (for the Phase 2 rebase — NOT wired in this build)
# ---------------------------------------------------------------------------
# 1. Call site: mission loop, per outer step, AFTER tracker.update_stale().
#    Inputs: _TRACKER.vis_smooth, _TRACKER.far_smooth, map_age (from _jit),
#    jitter_active (from jitter config), dt (outer step seconds).
# 2. pose_trustworthy: False until an estimator clears <0.5 m/mission.
#    Currently hardcoded False; the day it flips True is a policy event.
# 3. Final action = max(shipped_action, escalation_action) by severity, where
#    shipped_action = HOVER if hover_mode (FIX 3/4) else CONTINUE. This
#    guarantees the regression property: escalation only ADDS behavior
#    beyond hover, never changes existing outcomes.
# 4. RTH behavior wires to the reconciled planner (two-pass A*). Land and
#    disarm are self-contained above.
# 5. Manifest: log policy.log entries + final action per mission.
