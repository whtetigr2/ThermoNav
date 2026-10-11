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

LINK UPDATE (2026-10-11, roadmap Phase 1 item 2): the link slot is now
MODELED by link_health.py (receiver-side HEALTHY/DEGRADED/LOST report from
delivered-packet observables only). When evaluate() is called with a
link_report, the link rows below fire for real and "link" leaves the
MISSING list. Calls without a link_report keep the old stub behavior
exactly (backwards compatible). Battery / nav-health remain MISSING.

BATTERY UPDATE (2026-10-11, roadmap Phase 1 item 1, graduated after
red-team round 2 PASS WITH CAVEATS): the battery slot is now MODELED
by battery.py (classical host model, simulation only, parameters
calculated/ASSUMED as labeled there). When evaluate() is called with
battery_soc / battery_ttr_s, the battery rows below fire for real and
"battery" leaves the MISSING list. Calls without battery inputs keep
the stub behavior exactly (backwards compatible). The full measured-
countdown + RTB/RTC/land safety cascade lives in battery_safety.py;
when that layer is on it owns the battery decision and the mission
loop calls this policy WITHOUT battery inputs. Nav-health remains
MISSING. See sim/BATTERY_MODEL.md.

RTH CAVEAT (documented, not silent): the industry RTH assumes a known home,
a navigable path, and working navigation. Our estimator is an open problem
(scan-matching LEGO killed, red-team pass) and our fog is 16x below the FAA
waiver floor. RTH is therefore GATED on pose trustworthiness. With no
estimator the gate stays closed and RTH degrades to land-in-place, loudly
logged. Paul picked A knowing this.

Claim label: host-side classical policy. No TSU claim.
"Z1-legal formulation, validated in simulation" applies to the decider
kernel (escalation_decider.py), not this module.

DISARM (wired 2026-10-11, Paul's approval): the terminal ladder state now
has a code path. DISARM fires only after LAND completes (touchdown
confirmed) — it is a post-landing state transition, NOT a decider output:
the 12-spin decider maps perception inputs to 4 actions and correctly has
no DISARM island (touchdown is not a perception input). See the DISARM
TRANSITION block in evaluate() for the guards.

INTEGRATED BUILD (2026-10-10): D5 integration hooks are WIRED in
mission_dynamics_v5.py (branch muse/escalation-policy-2026-10-10, rebased
onto muse/tree-recon-phase2-2026-10-10). Call site: per outer step, after
tracker.update_stale(). pose_trustworthy hardcoded False. Final action =
max(shipped, escalation) by severity. See INTEGRATION CONTRACT below for
the wired mapping. This module exposes the contract; mission_dynamics_v5
implements the call site (numpy only here — no planner/s16 import).
"""

import numpy as np

from battery import RESERVE_SOC, WARN_SOC, CRITICAL_SOC

# Battery warning also fires on time-to-reserve, not just charge
# fraction: a high-draw phase can eat the reserve band faster than the
# fraction alone suggests at low draw.
BATT_TTR_WARN_S = 300.0  # s [design]

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

# Link rows (2026-10-11; timer semantics revised after red-team round 1,
# F1/F2). The LOST *declaration* itself is grounded in link_health.py
# (5.0 s silence, cited ArduPilot FS_GCS_TIMEOUT default). The policy
# timers below are LEAKY BUCKETS — one semantics for both, replacing the
# original reset/hold mix (which was neither sustained nor honestly
# labeled: DEGRADED reset on one HEALTHY report, so a flapping link
# never fired; LOST held across DEGRADED gaps, so separated episodes
# summed). Each bucket accumulates at 1x dt while its bad state (or a
# worse one) is reported, decays at LINK_BUCKET_LEAK x dt on any better
# report, and is floored at 0 / capped at its firing threshold. A 50%-
# duty flapping link therefore accumulates net-positive and escalates;
# a recovered link drains. Thresholds remain [design]:
LINK_DEGRADED_BUCKET_S = 5.0  # [design: bucket threshold, DEGRADED-tier
                              #  evidence -> Hold. Mirrors the perception
                              #  5 s figure. Not measured.]
LINK_LOST_BUCKET_S = 10.0     # [design: bucket threshold, LOST evidence
                              #  -> RTH. The 10 s figure mirrors PX4's
                              #  data-link-loss test timeout (docs.px4.io
                              #  MC_04: return after 10 s, reported),
                              #  but as a policy timer it is a design
                              #  choice.]
LINK_BUCKET_LEAK = 0.5        # [design: decay rate on better reports,
                              #  as a fraction of the accumulation rate]

# Commit-path semantics (red-team round 2, N1/N2 fix). Round 1 made the
# buckets leaky but left the commit path brittle: the bucket is capped
# at its threshold, so ONE better report dipped it below threshold,
# the RTH desire fell, and the hold window hard-reset — a link at
# 90-95% LOST duty with zero HEALTHY reports NEVER committed, and one
# stray packet every <=15 s held a total-dropout link at HOVER
# forever; symmetrically, one DEGRADED report released a committed
# LAND outright. Two mechanisms, both leaky like the buckets:
#
# (1) Leaky hold progress + latched desire. Hold-window progress
#     (_pending_timer) no longer resets when the desire dips: it
#     accumulates at 1x dt while a harsher-than-hover action is desired
#     and LEAKS at LINK_HOLD_LEAK x dt while it is not, so brief dips
#     cost progress instead of cancelling the window. The leak is
#     deliberately steep (3.5x): the fast path's break-even is ~78%
#     desire duty across the window horizon (a dip must be paid back
#     3.5x over in bucket-full desire). That line sits between the
#     red team's near-miss shapes (LOST 10 s / DEGRADED 2 s: ~79%
#     desire duty — commits on this path in ~2-3 min) and the
#     builder's padded brief dropouts (LOST 8 s / DEGRADED 2 s: ~75%
#     desire duty — net-negative, never commits on this path; its
#     refill steps leak too, which is what keeps it under the line).
#     For links whose LOST runs are too broken even for that, a second
#     integrator latches the desire: net LOST exposure accumulates at
#     1x dt on LOST reports, HOLDS on DEGRADED (packets flowing is not
#     recovery, but it is not blackout either), and drains at 1x dt on
#     HEALTHY reports; at LINK_EXPOSURE_COMMIT_S the RTH desire latches
#     on and stays on until genuine (HEALTHY) evidence drains the
#     exposure back below threshold. Any link a human would call dead
#     — never healthy, blacked out half the time or more — therefore
#     commits in bounded time (~85-160 s of wall time in the red team's
#     straddle shapes), while links with real HEALTHY gaps drain the
#     exposure and stay uncommitted. NOTE (round-3 C1, measured): the
#     55%-duty 11 s/9 s LOST/HEALTHY shape does NOT latch in ~750 s or
#     at any horizon (tested to 2400 s) — under the 75 s cap its
#     cycle-start exposure equilibrates at 75 - gap = 66, below the
#     latch (the 60% shape equilibrates at 67, also never). The latch
#     fires as documented when gaps are DEGRADED, which holds
#     exposure: 1 s LOST / 9 s DEGRADED commits at 750.5 s.
LINK_HOLD_LEAK = 3.5          # [design: hold-progress leak rate on
                              #  desire dips, as a fraction of the
                              #  accumulation rate (round-2 N1 fix)]
LINK_EXPOSURE_COMMIT_S = 75.0  # [design: net LOST-exposure (LOST time
                              #  minus HEALTHY time, DEGRADED holds)
                              #  at which the RTH desire latches
                              #  (round-2 N1 fix)]

# Land-in-place descent (Option A fallback when RTH gated)
LAND_DESCENT_RATE = 0.5  # m/s [design]
# Final-approach creep for NOT-fully-verified descents (battery RT-1):
# when the column/guard state is unverified (unknown column, or the
# floor layer has never produced scan evidence), the last
# LAND_CREEP_ALT_M are flown at creep rate. Measured reason: floor
# evidence arrives late (a zz=0-only obstacle was first hit by a scan
# at pose z=1.05), and the hover PID's ~0.73 s settle sinks ~0.1 m
# after a veto — at 0.5 m/s entry the veto fired and the vehicle
# still sank into the patch. At creep entry speed the settle sink
# is ~0.03 m and the guard gets ~3x the scans. [design]
LAND_CREEP_RATE = 0.15   # m/s [design]
LAND_CREEP_ALT_M = 2.5   # m above ground [design]

# Touchdown confirmation threshold (mirrors land_in_place_behavior below).
# DISARM may fire only when the vehicle reports at-or-below this altitude.
# [design: matches the 0.05 m landed check; conservative vs rotor radius]
TOUCHDOWN_Z_M = 0.05

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
        # Link leaky-bucket timers (from receiver reports only)
        self._link_deg_bucket = 0.0
        self._link_lost_bucket = 0.0
        self._link_lost_exposure = 0.0  # net LOST exposure (N1 latch):
                                        # +dt on LOST, hold on DEGRADED,
                                        # -dt on HEALTHY, capped at the
                                        # latch threshold
        self._link_modeled = False
        self._committed = None      # desired action that survived a hold
                                    # window (F3: commit means commit)
        self._gated_prev = False    # RTH-gate log dedup
        self._link_fired_prev = 0   # link-row severity last step (logging)
        # Hold-first state: pending harsher action + override window
        self._pending = None          # action waiting out the hold window
        self._pending_timer = 0.0
        self._current = CONTINUE
        # Battery warning latch: warn once per flight, not per step.
        self._batt_warned = False
        self._batt_modeled = False
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
    @staticmethod
    def _battery_trigger(battery_soc, battery_ttr_s):
        """Battery band from charge fraction (+ TTR for the warning).
        Returns 'critical' / 'reserve' / 'warn' / 'none' / None (no
        battery input supplied — pre-graduation callers)."""
        if battery_soc is None:
            return None
        if battery_soc <= CRITICAL_SOC:
            return "critical"
        if battery_soc <= RESERVE_SOC:
            return "reserve"
        if battery_soc <= WARN_SOC or (
                battery_ttr_s is not None
                and battery_ttr_s <= BATT_TTR_WARN_S):
            return "warn"
        return "none"

    def _table_action(self, vis, far, map_age, jitter, link_state=None,
                      battery_soc=None, battery_ttr_s=None):
        """Desired severity from the Option A trigger table, ignoring the
        hold window and the RTH gate. Returns (severity, reason,
        immediate, link_severity) where link_severity is what the link
        rows alone desire (for combined-case logging, red-team F7) and
        `immediate` is True only for battery-critical: the hold-first
        override window is skipped because the energy state only
        worsens — hovering out the window burns the last of the pack
        and ends in an uncontrolled drop [design].

        link_state: receiver health state (link_health.HEALTHY/DEGRADED/
        LOST) or None when the link is unmodeled (stub behavior). Link
        rows read the policy's leaky buckets, updated in evaluate() from
        the report sequence — never the channel's hidden condition."""
        degraded = self._degraded(vis, far)
        candidates = [(CONTINUE, "nominal")]
        # Row 3: degradation sustained 30 s -> RTH (checked first: severest)
        if degraded and self._deg_timer >= DEGRADE_SUSTAIN_RTH_S:
            candidates.append((RTH, "degraded sustained 30 s [design]"))
        # Row 1: degraded sustained 5 s -> Hold
        if degraded and self._deg_timer >= DEGRADE_SUSTAIN_HOVER_S:
            candidates.append((HOVER, "degraded sustained 5 s [design]"))
        # Row 2: degraded AND map stale -> immediate Hold (ships as FIX 4)
        if degraded and map_age >= MAP_STALE_S:
            candidates.append(
                (HOVER, "degraded AND map_age>=1.0 s [measured: FIX 4]"))
        # Jitter alone does not escalate beyond what staleness already
        # triggers (jitter's effect IS map_age in this sim); it is an input
        # to the decider kernel, not a separate policy row.
        # Link rows (leaky-bucket semantics; see the constants block):
        link_candidates = [(CONTINUE, "nominal")]
        if link_state is not None:
            # LOST-evidence bucket full -> RTH (severest link row)
            if self._link_lost_bucket >= LINK_LOST_BUCKET_S:
                link_candidates.append(
                    (RTH, "link LOST-evidence bucket >=10 s [design: "
                          "leaky accumulation]"))
            # Net LOST exposure latched -> RTH desire persists across
            # DEGRADED dips (round-2 N1 fix; see constants block)
            if self._link_lost_exposure >= LINK_EXPOSURE_COMMIT_S:
                link_candidates.append(
                    (RTH, "link net LOST exposure >=75 s latched "
                          "[design: leaky accumulation]"))
            # LOST declared (5 s silence, cited threshold) -> Hold now:
            # no operator intent can arrive; hold position while the
            # LOST bucket fills.
            elif link_state == 2:  # LOST (link_health.LOST)
                link_candidates.append(
                    (HOVER, "link LOST declared (silence>=5 s) [cited]"))
            # DEGRADED-tier bucket full -> Hold
            if self._link_deg_bucket >= LINK_DEGRADED_BUCKET_S:
                link_candidates.append(
                    (HOVER, "link DEGRADED-evidence bucket >=5 s [design: "
                            "leaky accumulation]"))
        link_best = max(link_candidates, key=lambda c: c[0])
        # Battery rows (graduated 2026-10-11). Most-severe-wins across
        # sources. Critical is severest and immediate (see docstring);
        # reserve is an RTB intent riding the generic RTH mechanism.
        trig = self._battery_trigger(battery_soc, battery_ttr_s)
        if trig == "critical":
            return LAND, ("battery critical: soc<=%.2f [design: "
                          "battery.py CRITICAL_SOC]" % CRITICAL_SOC), \
                True, link_best[0]
        if trig == "reserve":
            candidates.append(
                (RTH, "battery reserve: soc<=%.2f -> RTB intent "
                      "(return to base, RTH mechanism) "
                      "[design: battery.py RESERVE_SOC]" % RESERVE_SOC))
        best = max(candidates + link_candidates, key=lambda c: c[0])
        return best[0], best[1], False, link_best[0]

    def evaluate(self, vis, far, map_age, jitter, dt,
                 pose_trustworthy=False, operator_override=False, t=0.0,
                 landed=False, link_report=None,
                 battery_soc=None, battery_ttr_s=None):
        """One policy step. Returns (action, info dict).

        pose_trustworthy: RTH gate. False until an estimator clears the
            <0.5 m/mission bar (open problem). RTH degrades to land-in-place.
        operator_override: industry hold-first rule — True cancels a pending
            harsher action back to hover during the override window.
        landed: touchdown confirmation from the actuation layer
            (land_in_place_behavior returns True when pos[2] <= TOUCHDOWN_Z_M).
            DISARM fires ONLY when _current == LAND and landed is True.
            This is the sole DISARM transition — there is deliberately NO
            timeout-to-DISARM path: disarming in the air is an uncontrolled
            fall. The ladder's terminal AIRBORNE state is LAND; DISARM
            follows confirmed touchdown only.
        link_report: dict from link_health.LinkHealthTracker.report()
            (receiver observables only), or None = link unmodeled (stub).
            An operator override can only be *delivered* via the link:
            callers wire the delivered-override flag, so a lost link
            cannot produce an override (enforced by the wiring, and the
            hold window still runs — it just cannot be cancelled).
            Known policy-surface fact (red-team round 3, C4): a call
            with link_report=None instantly releases a link-committed
            action to HOVER; no wired caller does this (the mission
            loop always passes a report when the link is modeled).
        battery_soc / battery_ttr_s: from battery.BatteryModel (host
            model, simulation only). None = no battery input; the policy
            then behaves exactly as the pre-graduation build.
        """
        # Missing slots are evaluated (always inactive) so the table is
        # complete and auditable. Once a link report is supplied the link
        # slot is modeled, not missing; same for battery inputs.
        self._link_modeled = link_report is not None
        self._batt_modeled = battery_soc is not None
        for name, stub in self.missing_triggers.items():
            if name == "link" and self._link_modeled:
                continue
            if name == "battery" and self._batt_modeled:
                continue
            fired, _ = stub.evaluate()
            assert not fired

        # Battery warning latch: the warn band is a host notification
        # level (warning is not a flight action), logged once when
        # first entered.
        batt_trig = self._battery_trigger(battery_soc, battery_ttr_s)
        if batt_trig in ("warn", "reserve", "critical") \
                and not self._batt_warned:
            self._batt_warned = True
            self.log.append(
                (t, "battery_warn",
                 f"soc={battery_soc:.3f} ttr_s={battery_ttr_s} "
                 f"band={batt_trig} [design thresholds: battery.py]"))

        # Sustained-degradation timer with hysteresis reset.
        if self._degraded(vis, far):
            self._deg_timer += dt
        elif self._clear(vis, far):
            self._deg_timer = 0.0
        # else: hold timer (in-band hysteresis, neither accumulate nor reset)

        # Link leaky buckets (report sequence only). One semantics for
        # both buckets: accumulate at 1x dt in the bad state (LOST also
        # fills the DEGRADED-tier bucket — it is worse than degraded),
        # decay at LINK_BUCKET_LEAK x dt on any better report, floor 0,
        # cap at the firing threshold.
        link_state = None
        if link_report is not None:
            link_state = int(link_report["state"])
            if link_state == 0:      # HEALTHY: both buckets drain
                self._link_deg_bucket = max(
                    0.0, self._link_deg_bucket - LINK_BUCKET_LEAK * dt)
                self._link_lost_bucket = max(
                    0.0, self._link_lost_bucket - LINK_BUCKET_LEAK * dt)
            elif link_state == 1:    # DEGRADED: deg fills, lost drains
                self._link_deg_bucket = min(
                    LINK_DEGRADED_BUCKET_S, self._link_deg_bucket + dt)
                self._link_lost_bucket = max(
                    0.0, self._link_lost_bucket - LINK_BUCKET_LEAK * dt)
            else:                    # LOST: both fill
                self._link_deg_bucket = min(
                    LINK_DEGRADED_BUCKET_S, self._link_deg_bucket + dt)
                self._link_lost_bucket = min(
                    LINK_LOST_BUCKET_S, self._link_lost_bucket + dt)
            # Net LOST exposure (round-2 N1 latch): +dt on LOST, hold
            # on DEGRADED, -dt on HEALTHY; capped at the latch
            # threshold so genuine recovery unlatches it promptly.
            if link_state == 2:      # LOST
                self._link_lost_exposure = min(
                    LINK_EXPOSURE_COMMIT_S, self._link_lost_exposure + dt)
            elif link_state == 0:    # HEALTHY drains the exposure
                self._link_lost_exposure = max(
                    0.0, self._link_lost_exposure - dt)
            # DEGRADED: exposure holds (neither blackout nor recovery)

        desired, reason, immediate, link_sev = self._table_action(
            vis, far, map_age, jitter, link_state=link_state,
            battery_soc=battery_soc, battery_ttr_s=battery_ttr_s)

        # Industry rule 1: hold-first. Hover IS the hold: continue->hover
        # engages immediately (the shipped FIX 3 / FIX 4 behavior is not
        # delayed). Steps BEYOND hover (RTH/land/disarm) wait out the
        # operator override window AT hover level.
        holding = False
        overridden = False
        commit_held = False
        if immediate:
            # Battery-critical: commit now, no hold window (see
            # _table_action). Pending state cleared so a later recovery
            # is evaluated fresh.
            self._pending, self._pending_timer = None, 0.0
            action = desired
            self.log.append(
                (t, "battery_critical_immediate",
                 f"action={ACTION_NAMES[desired]} reason={reason}"))
        elif desired <= HOVER:
            # Hover engages immediately. Commit-path semantics
            # (round-2 N1/N2 fix — see the constants block):
            #  * A PENDING harsher action is NOT cancelled by a desire
            #    dip: its hold progress leaks at LINK_HOLD_LEAK x dt
            #    and the pending action clears only when the progress
            #    fully drains (~3 s of absent desire at the 3.5x leak —
            #    a genuinely recovered link drains it; a dying link's
            #    brief dips only dent it).
            #  * A COMMITTED action is NOT released by a dip: while
            #    the link is modeled, release requires both link
            #    buckets fully drained to zero. The DEGRADED bucket
            #    drains only on HEALTHY reports and the LOST bucket
            #    needs 20 s of better reports from full (and any
            #    DEGRADED report refills the DEGRADED bucket), so in
            #    practice release requires ~20 s of sustained HEALTHY
            #    reports from FULL buckets (fast-path commits); latch
            #    commits release in 10-15 s because their buckets are
            #    part-drained at commit (round-3 C2, measured) —
            #    evidence comparable in weight to the 10 s bucket +
            #    10 s hold window that created the commit. Release
            #    ignores the exposure integrator (round-3 C3): a latch
            #    commit can release with exposure still at 60-65, a
            #    re-commit head start (same straddle re-commits in
            #    62 s vs 152 s cold) — conservative direction.
            #    NOTE (round-3 C4, policy surface): evaluate() called
            #    with link_report=None releases a link-committed action
            #    instantly on this branch ("no link model" is treated
            #    as "buckets drained"); no wired caller does this —
            #    the mission loop always passes a report.
            #    One DEGRADED report — all a single stray packet
            #    manufactures — releases nothing. When the evidence
            #    IS met, the release steps down to HOVER, never
            #    straight to mission resumption on the release step.
            if self._committed is not None and (
                    link_state is not None
                    and (self._link_lost_bucket > 0.0
                         or self._link_deg_bucket > 0.0)):
                action = self._committed
                commit_held = True
            elif self._committed is not None:
                self.log.append(
                    (t, "commit_released",
                     f"released {ACTION_NAMES[self._committed]}: "
                     f"sustained recovery evidence (link buckets "
                     f"drained to zero); step down to hover"))
                self._committed = None
                action = max(desired, HOVER)
            else:
                action = desired
            if self._pending is not None:
                self._pending_timer = max(
                    0.0, self._pending_timer - LINK_HOLD_LEAK * dt)
                if self._pending_timer <= 0.0:
                    self._pending = None
        elif self._committed == desired:
            # F3 fix (red-team round 1): this action already survived a
            # hold window and committed. Hold the commitment — do NOT
            # re-open the window. An operator override cancels a
            # PENDING action during its window; it does not un-commit
            # a committed one. Release on a desire dip goes through
            # the sustained-evidence branch above (round-2 N2 fix).
            action = desired
        else:
            # desired in (RTH, LAND): hold window at hover level.
            # (DISARM is never a table output — it fires only from the
            # post-landing transition below, never through this path.)
            # If the same action is still pending after a desire dip,
            # its leaked progress resumes accumulating — the window
            # does not re-open and no second log entry is written.
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
                self._committed = self._pending
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
            if not self._gated_prev:
                self.log.append(
                    (t, "rth_gated_land_in_place",
                     "RTH requested per industry ladder; pose trust gate "
                     "CLOSED (estimator: open problem, no <0.5 m/mission "
                     "estimator exists) -> land-in-place fallback"))
            self._gated_prev = True
        else:
            self._gated_prev = False

        # DISARM TRANSITION (wired 2026-10-11, Paul's approval).
        # DISARM = motors off. It fires ONLY here: _current == LAND and
        # the actuation layer confirms touchdown (landed=True).
        # This covers RTH->land too: the closed RTH gate already degrades
        # RTH to land-in-place, so every landing funnels through LAND.
        #
        # There is deliberately NO timeout-to-DISARM path. Sustained
        # degradation with landed=False holds at LAND (attempting landing),
        # never DISARM — disarming in the air is an uncontrolled fall.
        # The ladder's terminal AIRBORNE state is LAND.
        #
        # Safety chain (defense in depth):
        #   1. below-clear guard (_below_clear, mission loop): LAND descends
        #      only if the column below is observed-clear; else HOVER.
        #   2. touchdown confirmation: land_in_place_behavior returns True
        #      only when pos[2] <= TOUCHDOWN_Z_M (0.05 m).
        #   3. THIS transition: DISARM only from LAND + landed.
        #   4. Latch below: DISARM is terminal; once committed it sticks.
        disarmed = False
        if self._current == DISARM:
            # Terminal latch: DISARM sticks regardless of inputs.
            action = DISARM
            disarmed = True
        elif self._current == LAND and landed:
            action = DISARM
            disarmed = True
            self.log.append(
                (t, "disarm_touchdown_confirmed",
                 "LAND complete (touchdown confirmed) -> DISARM"))
        elif landed and self._current != LAND:
            # Guard: a landed flag outside LAND is ignored, loudly.
            # (Stale flag, or touchdown via a non-policy path — the policy
            # does not disarm what it did not land.)
            self.log.append(
                (t, "disarm_vetoed_not_landing",
                 f"landed=True but _current="
                 f"{ACTION_NAMES[self._current]}; DISARM requires LAND"))

        # Commit post-gate state. _current never holds a pre-gate RTH.
        if action != self._current:
            if action < self._current:
                self.log.append((t, "deescalate",
                                 f"now={ACTION_NAMES[action]}"))
            self._current = action

        # Link-escalation logging (F7 fix): log on the link rows' own
        # severity rising edge, whether or not the link reason won the
        # combined max() — the old reason-string check logged nothing
        # when perception and link fired together.
        if link_sev > self._link_fired_prev:
            self.log.append(
                (t, "link_escalation",
                 f"link rows desire {ACTION_NAMES[link_sev]} "
                 f"(combined reason: {reason})"))
        self._link_fired_prev = link_sev

        return action, self._info(
            reason, gated=gated, holding=holding, overridden=overridden,
            disarmed=disarmed, commit_held=commit_held,
            link_modeled=self._link_modeled, link_state=link_state,
            battery_soc=battery_soc, battery_ttr_s=battery_ttr_s,
            battery_trigger=batt_trig,
            batt_modeled=self._batt_modeled)

    @staticmethod
    def _info(reason, gated, holding=False, overridden=False,
              disarmed=False, commit_held=False, link_modeled=False,
              link_state=None, battery_soc=None, battery_ttr_s=None,
              battery_trigger=None, batt_modeled=False):
        missing = []
        if not batt_modeled:
            missing.append("battery")
        if not link_modeled:
            missing.append("link")
        missing.append("nav_health")
        return {"reason": reason, "rth_gated": gated,
                "holding_for_override": holding,
                "operator_overridden": overridden,
                "disarmed": disarmed,
                "commit_held": commit_held,
                "missing_triggers": missing,
                "link_modeled": link_modeled,
                "link_state": link_state,
                "battery_soc": battery_soc,
                "battery_ttr_s": battery_ttr_s,
                "battery_trigger": battery_trigger}

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
        too slow.) [design]

        Returns True on touchdown (pos[2] <= TOUCHDOWN_Z_M). The mission
        loop passes this as landed= to the next policy.evaluate(); that
        is the SOLE trigger for the DISARM transition."""
        # command(p_des, v_des, p, v): error = p_des - p.
        a_cmd = pid.command(vehicle.pos, (0, 0, -LAND_DESCENT_RATE),
                            vehicle.pos, vehicle.vel)
        vehicle.step(a_cmd)
        landed = vehicle.pos[2] <= TOUCHDOWN_Z_M
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
    def disarm_behavior(log, outcome="landed_disarmed"):
        """Terminal: motors off after confirmed touchdown. Called ONLY when
        the policy emits DISARM (which requires LAND + landed=True — see
        the DISARM TRANSITION block in evaluate). Never call in the air.
        Outcome string extends the existing safe stand-down convention."""
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
# 5. DISARM wiring (2026-10-11, fixed 2026-10-11): at touchdown the loop
#    makes a FINAL synchronous policy.evaluate(landed=True) via the
#    mission loop's _touchdown_disarm_check — not on the next step,
#    because the loop breaks on touchdown. The loop's own inline landing
#    code executes the descent (land_in_place_behavior above is a
#    standalone reference implementation, not called by the loop);
#    _esc_land (True only post below-clear veto) gates the check, so a
#    stale LAND in the policy can never fire DISARM while the loop is
#    hovering. When the policy emits DISARM, the loop calls
#    disarm_behavior(log) and ends the mission with
#    outcome="landed_disarmed"; otherwise outcome="landed". DISARM fires
#    only from LAND + landed=True; there is no timeout-to-DISARM path by
#    design (disarming in the air is an uncontrolled fall).
# 6. Manifest: log policy.log entries + final action per mission.
# 7. Link-health layer (graduated 2026-10-11): link_health.py models the
#    link slot; evaluate(link_report=...) enables the link rows (leaky
#    buckets, leaky hold progress, net-LOST-exposure latch, bucket-drained
#    release). Calls without a link_report keep the stub behavior above,
#    except the commit bookkeeping (committed actions now record
#    _committed; a no-report call releases a link commitment instantly —
#    round-3 C4, documented in LINK_MODEL.md; no wired caller does this).
#    See sim/LINK_MODEL.md for the full build report and known limits.
# 8. Battery layer (graduated 2026-10-11): battery.py models the battery
#    slot; evaluate(battery_soc=..., battery_ttr_s=...) enables the
#    battery rows (warn latch / reserve RTB intent / critical immediate
#    LAND). battery_safety.py owns the battery decision when the safety
#    layer is on — the mission loop then calls evaluate() WITHOUT
#    battery inputs and maps the safety action over the escalation
#    action (terminal battery landing overwrites, so DISARM cannot
#    outrank it; DISARM still fires only after that landing's
#    touchdown, via the loop's touchdown branch). See
#    sim/BATTERY_MODEL.md for the full build report and known limits.
