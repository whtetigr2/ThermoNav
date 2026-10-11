"""Battery safety layer — Paul's decided design (2026-10-11).

This layer sits ABOVE the battery gauge (battery.py) and the escalation
ladder (escalation.py). It implements, exactly as decided:

  * ONE strict low-battery FLAG. It fires at the RESERVE threshold.
    Nothing outranks it once it is terminal (countdown end, below).
  * At RESERVE a bounded COUNTDOWN starts. Its length is sized from
    MEASURED draw under real load — the countdown is a meter, not
    just a timer. "Measured" here means the gauge's own post-step
    telemetry (soc + Wh actually consumed, i.e. what a coulomb
    counter / power meter on a real aircraft reports), read back over
    a trailing window. It is NEVER sized from the power model's
    predicted draw (battery.power_for_velocity) and NEVER from the
    model's soc percentage alone: the red team measured the model's
    endurance arithmetic ~8% optimistic against the registry band
    (RT-6), and Paul's ruling was that a measured drain rate beats a
    modeled percentage. Paul explicitly rejected "raise CRITICAL to
    ~19%" as a substitute for this meter; CRITICAL_SOC is used here
    only as the soc the countdown is sized to REACH, not as a
    raised threshold.
  * During the countdown window the whiteout occupancy veto may
    still hold a landing (the mission loop's guard does that). At
    countdown END the layer is terminal: LAND regardless of any
    condition — the veto does not outrank the flag.
  * When the flag fires, the action is a REACHABILITY decision made
    with the measured meter — a three-rung cascade (Paul's
    correction, 2026-10-11):
        1. RTB — return to BASE, if energy-to-base is affordable;
        2. RTC — return to CHECKPOINT, if RTB is not affordable but
           energy-to-checkpoint is;
        3. emergency LAND IN PLACE, if neither is affordable.
    Land-in-place is the FALLBACK, not the default.
  * Reachability is re-checked CONTINUOUSLY during any return leg.
    If a leg becomes unaffordable mid-flight the cascade aborts
    DOWN (RTB -> RTC -> land). It never aborts up.

TERMINOLOGY (Paul, 2026-10-11): RTB = return to BASE. RTC = return
to CHECKPOINT. They are distinct targets. The escalation ladder's
generic RTH severity is the return MECHANISM both rungs ride on;
this module owns the battery-specific targets and names.

CHECKPOINT SPEC (smallest honest version, decided here and in
BATTERY_MODEL.md):
  * A checkpoint is a position the vehicle itself occupied during
    THIS flight, recorded while nav-health reported trustworthy.
  * Record/refresh: every outer step (1 Hz) whose nav-trust signal
    is trustworthy overwrites the checkpoint with the current pose
    (and its time). The checkpoint is therefore the LAST
    trustworthy pose of the flight.
  * Freeze: the first untrustworthy step freezes the checkpoint;
    it is not updated again until trust returns (a later trustworthy
    step refreshes it again — freezing is per-loss, not latching).
  * Takeoff: the start pose is recorded as the initial checkpoint
    at t=0 (the vehicle is on the ground, pose known by
    construction). Until a trustworthy in-flight refresh replaces
    it, checkpoint == base and RTC coincides with RTB.
  * Interface: the nav-trust signal is a single boolean with the
    semantics of the nav-health monitor's `pose_trustworthy`
    property — True ONLY when nav-health is HEALTHY (documented in
    nav_health.py / NAV_HEALTH.md: HEALTHY / DEGRADED / UNTRUSTED,
    gate open only on HEALTHY). GRADUATION UPDATE (2026-10-11): the
    real NavHealthMonitor now lives in this repo (sim/nav_health.py)
    and the mission loop feeds the checkpoint from it BY DEFAULT —
    resolve_nav_trustworthy() in mission_dynamics_v5.py returns
    monitor.pose_trustworthy, read after the monitor's 1 Hz update
    for the step. The explicit slot survives as a scripted scenario
    override only: cfg["nav_trustworthy"] may be a bool or a
    callable (t_s, pose) -> bool with exactly those semantics, and
    when a scenario supplies it, it wins (deterministic RTC
    constructions cut trust at an exact time — see
    nav_trustworthy_stub). With neither a monitor nor an override
    the signal is False, so no in-flight checkpoint is recorded and
    RTC == RTB.

AFFORDABILITY (measured meter, not model):
  energy-to-target, in soc units =
      measured drain rate (soc/s, trailing window)
      * (distance_to_target / RETURN_SPEED)
      * MARGIN_FACTOR
  plus the requirement that the projected soc on arrival stays at
  or above LANDING_FLOOR_SOC. LANDING_FLOOR_SOC is a PLAN-TIME
  threshold on that projection — NOT an outcome floor on the charge
  the vehicle actually lands with: the projection excludes path /
  speed reality, post-arrival veto holds, and descent cost, and
  measured finals land well below it (rtb_clear projected arrival
  soc 0.178, actual final 0.0693 — red-team round 2, RT2-2). No
  document should describe 0.05 as the charge landed with.
  Distance is Euclidean distance to the recorded target pose, from
  the vehicle's own current pose (onboard state — no ground-truth
  occupancy/pose channel; same leakage ruling as the battery model).
  Straight-line distance underprices a real path around obstacles;
  MARGIN_FACTOR is the (partial, labeled) compensation, and path
  lengthening beyond the margin is a known limit in the report.

COUNTDOWN SIZING (measured):
  countdown_s = (soc_at_flag - CRITICAL_SOC) / measured_drain_rate
  i.e. the measured time the remaining reserve band buys, at the
  draw actually observed in flight. The meter keeps measuring for
  the whole countdown: if the measured drain rate WORSENS, the
  countdown end is re-sized EARLIER (never later — the countdown
  is bounded by its first sizing). Measured reason (validation
  round 2, abort_spike at 52b3b3d): a load spike after sizing held
  the one-shot 68.7 s countdown while the spiked drain emptied
  the pack to soc 0.0 before it ended; with re-sizing, the end
  tracks the meter and the terminal landing fires while charge
  remains. The flag may fire before the
  meter has a usable window (scripted low-soc starts); then the
  layer hovers in HOLD_METERING while the window fills (a few
  seconds of countdown are spent buying the meter — Paul's words:
  "Count down will drain more battery but it buys us an actual
  meter"), and the countdown is sized the moment the meter is
  ready. If soc reaches CRITICAL before that, the countdown is
  zero-length and the layer is terminal immediately.

HOLD_METERING BOUND (red-team round 2, RT2-1 — the Wh channel was
measured/logged but never used by any decision, so a reported soc
that stops draining — a stuck gauge on hardware, unreachable with
the in-sim gauge — held HOLD_METERING forever and the "bounded
countdown" guarantee was silently conditional on the soc gauge
draining). The hold is now bounded two ways, whichever fires first:
  * CROSS-CHECK: if the meter window shows real energy leaving the
    pack (consumed Wh over the window >= WH_DISAGREE_MIN_WH) while
    the soc channel reports no drain at all (rate <= 0 — flat or
    rising), the channels disagree: the soc gauge is not trusted
    and the countdown is sized from the Wh-implied drain rate
    (consumed-Wh rate / pack capacity) instead. The Wh channel is
    used ONLY for this disagreement fallback — normal sizing still
    reads the soc channel, which on a real aircraft IS the coulomb
    counter's own output.
  * TIME BOUND: HOLD_METERING may last at most HOLD_METERING_MAX_S
    of flag time (plus a step-count backstop for callers whose
    clock does not advance). At the bound, if the Wh channel
    implies a positive drain rate the countdown is sized from it;
    if neither channel shows drain there is nothing to size from
    and the layer terminates to the safest available action —
    LAND_TERMINAL (landing now on whatever charge remains beats
    hovering on a dead gauge until the pack is actually empty).
Both paths are logged as events (soc_wh_disagreement /
hold_metering_bound) so a gauge fault is loud, never silent.

CALIBRATION LOGGING: while the countdown runs, every step appends
a row to `calibration_log` (t, soc, measured draw, drain rate,
countdown remaining, rung, both reachability computations). The
validation runner writes these rows to CSV with provenance
columns (script, seed, room, git commit captured at runtime).

Claim label: classical host safety layer, simulation only.
"Z1-legal formulation, validated in simulation" does not apply to
this host module (it applied to the escalation decider kernel,
untouched here). No silicon / aircraft claim.
"""
import math

from battery import RESERVE_SOC, CRITICAL_SOC, CAPACITY_WH

# ---------------------------------------------------------------------------
# Design constants. Every one is [design] — a policy choice, not a
# measurement — except the thresholds imported from battery.py.
# ---------------------------------------------------------------------------
RETURN_SPEED_MPS = 1.0     # [design] speed assumed for return legs
                           # (operational cruise in this stack is
                           # <= 2 m/s; planning the return at 1 m/s
                           # prices the leg at the slower, thirstier
                           # end of the momentum-theory bucket)
MARGIN_FACTOR = 1.25       # [design] reachability margin multiplier:
                           # energy-to-target must fit with 25% to
                           # spare (path lengthening, maneuvering)
LANDING_FLOOR_SOC = 0.05   # [design] PLAN-TIME threshold only: the
                           # projected arrival soc must stay >= this
                           # for a leg to be picked. NOT an outcome
                           # floor — measured finals land well below
                           # the projection (RT2-2; see docstring).
METER_WINDOW_S = 10.0      # [design] trailing window for the measured
                           # draw / drain-rate estimates (long enough
                           # to average phase changes, short enough to
                           # react to a load change mid-return)
METER_MIN_SPAN_S = 2.0     # [design] minimum window span before the
                           # meter may size a countdown or judge
                           # reachability
HOLD_METERING_MAX_S = 30.0  # [design] RT2-1 bound: maximum flag-time
                           # the layer may sit in HOLD_METERING before
                           # it must decide from the Wh channel or
                           # terminate (see docstring). Sized to be
                           # far above the normal few-second metering
                           # hold and far below any plausible reserve-
                           # band endurance at real loads.
HOLD_METERING_MAX_STEPS = 120  # [design] RT2-1 backstop for callers
                           # whose step clock does not advance: the
                           # hold also ends after this many steps.
WH_DISAGREE_MIN_WH = 0.5   # [design] RT2-1 cross-check: consumed Wh
                           # over the meter window at/above which a
                           # zero soc drain rate reads as channel
                           # disagreement, not a quiet pack (~18 W
                           # over the 10 s window — far below the
                           # ~282 W hover draw, far above noise).

# Actions (the layer's output vocabulary; the mission loop maps them)
NONE = "none"                      # flag not fired: no safety action
HOLD_METERING = "hold_metering"    # flag fired, meter not ready: hold
RTB = "RTB"                        # return to base
RTC = "RTC"                        # return to checkpoint
LAND_IN_PLACE = "land_in_place"    # emergency land where it is (guarded
                                   # by the loop's veto while the
                                   # countdown runs)
LAND_TERMINAL = "land_terminal"    # countdown end: land regardless of
                                   # any veto — nothing outranks it

_RUNG_ORDER = {RTB: 0, RTC: 1, LAND_IN_PLACE: 2}


def _dist(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


class MeasuredDrawMeter:
    """Measured draw from gauge telemetry: (t, soc, wh_used) samples.

    This is the "actual meter" of Paul's design. Its inputs are what
    the pack actually did (charge/energy consumed over elapsed time),
    never a power prediction. Both estimates are trailing-window
    differences, so a load change shows up within one window.
    """

    def __init__(self, window_s=METER_WINDOW_S):
        self.window_s = float(window_s)
        self.samples = []  # (t, soc, wh_used), monotone t

    def add(self, t, soc, wh_used):
        self.samples.append((float(t), float(soc), float(wh_used)))
        cutoff = float(t) - self.window_s
        # Keep at least the last two samples even if the window is
        # shorter than the sample interval.
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.pop(0)

    @property
    def ready(self):
        if len(self.samples) < 2:
            return False
        return (self.samples[-1][0] - self.samples[0][0]) \
            >= METER_MIN_SPAN_S

    def span_s(self):
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1][0] - self.samples[0][0]

    def measured_draw_w(self):
        """Mean electrical draw (W) over the window, from consumed
        Wh — the measured load, not the model's prediction."""
        if len(self.samples) < 2:
            return None
        (t0, _, w0), (t1, _, w1) = self.samples[0], self.samples[-1]
        if t1 <= t0:
            return None
        return (w1 - w0) * 3600.0 / (t1 - t0)

    def drain_rate_soc_per_s(self):
        """Measured soc drain rate (fraction/s) over the window."""
        if len(self.samples) < 2:
            return None
        (t0, s0, _), (t1, s1, _) = self.samples[0], self.samples[-1]
        if t1 <= t0:
            return None
        return max(0.0, (s0 - s1) / (t1 - t0))

    def wh_consumed_window(self):
        """Wh consumed over the window (the meter's second channel)."""
        if len(self.samples) < 2:
            return None
        return self.samples[-1][2] - self.samples[0][2]

    def wh_implied_drain_rate_soc_per_s(self):
        """Drain rate implied by the Wh channel alone: consumed-Wh
        rate divided by pack capacity. Used ONLY by the RT2-1
        disagreement / hold-bound fallback — normal decisions read
        the soc channel (see module docstring)."""
        if len(self.samples) < 2:
            return None
        (t0, _, w0), (t1, _, w1) = self.samples[0], self.samples[-1]
        if t1 <= t0:
            return None
        return max(0.0, (w1 - w0) / (t1 - t0) / CAPACITY_WH)

    def soc_wh_disagree(self):
        """RT2-1 cross-check: real energy is leaving the pack (Wh
        channel) while the soc channel reports no drain at all.
        That combination is a gauge fault, not a quiet pack."""
        if not self.ready:
            return False
        wh = self.wh_consumed_window()
        rate = self.drain_rate_soc_per_s()
        if wh is None or rate is None:
            return False
        return wh >= WH_DISAGREE_MIN_WH and rate <= 0.0


class CheckpointTracker:
    """The checkpoint of the cascade (see module docstring spec)."""

    def __init__(self, base_pose):
        self.base_pose = tuple(float(c) for c in base_pose)
        self.checkpoint_pose = self.base_pose
        self.checkpoint_t = 0.0
        self.n_refresh = 0
        self.frozen = False

    def update(self, pose, trustworthy, t):
        """Refresh while trustworthy; freeze when trust is lost."""
        if trustworthy:
            self.checkpoint_pose = tuple(float(c) for c in pose)
            self.checkpoint_t = float(t)
            self.n_refresh += 1
            self.frozen = False
        else:
            self.frozen = True
        return self.checkpoint_pose


class BatterySafetyController:
    """One flag, one countdown, one three-rung cascade.

    Per outer step the mission loop calls:
        observe(t, soc, wh_used, pose, nav_trustworthy)
        action, info = step(t, soc, pose)
    observe() feeds the meter and the checkpoint tracker even before
    the flag fires — the meter must already hold a measured window
    when RESERVE is crossed, and the checkpoint is a flight-long
    record, not something created at the flag.
    """

    def __init__(self, base_pose):
        self.meter = MeasuredDrawMeter()
        self.tracker = CheckpointTracker(base_pose)
        self.base_pose = self.tracker.base_pose
        self.flag_fired = False
        self.flag_t = None
        self.flag_soc = None
        self.countdown_s = None      # sized once the meter is ready
        self.countdown_end_t = None
        self.rung = None             # RTB / RTC / LAND_IN_PLACE
        self.target_pose = None      # snapshot for the current rung
        self.terminal = False
        self.events = []             # (t, event, detail)
        self.calibration_log = []    # one dict per countdown step
        self.n_abort_down = 0
        self._hold_start_t = None    # RT2-1: first HOLD_METERING step
        self._hold_steps = 0         # RT2-1: HOLD_METERING step count
        self.hold_bound_reason = None  # RT2-1: why the hold ended by
                                       # bound/cross-check, if it did

    # -- reachability (measured) -------------------------------------------
    def energy_soc_needed(self, pose, target):
        """Soc the leg to `target` costs at the MEASURED drain rate,
        margin included. None when the meter cannot judge yet."""
        rate = self.meter.drain_rate_soc_per_s()
        if rate is None or not self.meter.ready:
            return None
        t_leg = _dist(pose, target) / RETURN_SPEED_MPS
        return rate * t_leg * MARGIN_FACTOR

    def affordable(self, pose, target, soc):
        need = self.energy_soc_needed(pose, target)
        if need is None:
            return None
        return (soc - need) >= LANDING_FLOOR_SOC

    def _pick_rung(self, pose, soc):
        """The cascade, top down. Returns (rung, target_pose)."""
        if self.affordable(pose, self.base_pose, soc):
            return RTB, self.base_pose
        ckpt = self.tracker.checkpoint_pose
        if self.affordable(pose, ckpt, soc):
            return RTC, ckpt
        return LAND_IN_PLACE, tuple(float(c) for c in pose)

    # -- RT2-1: bounded metering hold ---------------------------------------
    def _force_metering_decision(self, t, soc, pose, reason):
        """The metering hold has hit its bound or the channels
        disagree: decide NOW from whatever the meter has.

        If the Wh channel implies a positive drain rate, size the
        countdown from it (the soc gauge is the broken channel —
        energy bookkeeping is the surviving measurement). Returns
        None when a countdown was sized (the caller falls through
        to the cascade), or the LAND_TERMINAL action when neither
        channel shows drain — nothing to size from, so terminate
        to the safest available action rather than hover on a dead
        gauge (see module docstring)."""
        self.hold_bound_reason = reason
        wh_rate = self.meter.wh_implied_drain_rate_soc_per_s()
        if wh_rate and wh_rate > 0.0:
            self.countdown_s = max(0.0, (soc - CRITICAL_SOC) / wh_rate)
            self.countdown_end_t = float(t) + self.countdown_s
            self.events.append(
                (float(t), reason,
                 f"HOLD_METERING ended by {reason}: countdown sized "
                 f"from the Wh channel (wh-implied drain "
                 f"{wh_rate:.6f} soc/s, measured_draw_w="
                 f"{self.meter.measured_draw_w()}) because the soc "
                 f"channel reports no drain [RT2-1]"))
            if self.countdown_s > 0.0:
                return None
        self.countdown_s = 0.0
        self.countdown_end_t = float(t)
        self.terminal = True
        self.events.append(
            (float(t), reason,
             f"HOLD_METERING ended by {reason} with no usable drain "
             f"measurement on either channel: LAND_TERMINAL — the "
             f"safest available action on a dead gauge [RT2-1]"))
        self._log_calibration(t, soc, pose, LAND_TERMINAL)
        return LAND_TERMINAL

    # -- per-step interface --------------------------------------------------
    def observe(self, t, soc, wh_used, pose, nav_trustworthy):
        self.meter.add(t, soc, wh_used)
        self.tracker.update(pose, bool(nav_trustworthy), t)

    def step(self, t, soc, pose):
        pose = tuple(float(c) for c in pose)
        # --- the flag ---------------------------------------------------
        if not self.flag_fired:
            if soc > RESERVE_SOC:
                return NONE, {"flag": False}
            self.flag_fired = True
            self.flag_t = float(t)
            self.flag_soc = float(soc)
            self.events.append(
                (float(t), "battery_flag_fired",
                 f"soc={soc:.4f} <= RESERVE {RESERVE_SOC} [Paul's "
                 f"design 2026-10-11: one strict flag]"))
        # --- meter keeps measuring: a worsening measured drain
        # re-sizes the countdown end EARLIER (never later). At a
        # constant rate the candidate end is invariant, so this is
        # a no-op until the load actually changes.
        if self.countdown_end_t is not None and not self.terminal:
            rate_now = self.meter.drain_rate_soc_per_s()
            if soc <= CRITICAL_SOC:
                self.countdown_end_t = min(self.countdown_end_t,
                                           float(t))
            elif self.meter.ready and rate_now and rate_now > 0.0:
                end_candidate = float(t) + (soc - CRITICAL_SOC) / rate_now
                if end_candidate < self.countdown_end_t - 1.0:
                    self.events.append(
                        (float(t), "countdown_resized_earlier",
                         f"end {self.countdown_end_t:.1f} -> "
                         f"{end_candidate:.1f} at measured drain "
                         f"{rate_now:.6f} soc/s (was sized from "
                         f"flag_soc {self.flag_soc:.4f})"))
                    self.countdown_end_t = end_candidate
        # --- terminal floor: countdown end outranks everything --------
        if self.countdown_end_t is not None and t >= self.countdown_end_t:
            if not self.terminal:
                self.terminal = True
                self.events.append(
                    (float(t), "countdown_end_terminal_land",
                     f"countdown {self.countdown_s:.1f}s elapsed; LAND "
                     f"regardless of veto [Paul's design]"))
            self._log_calibration(t, soc, pose, LAND_TERMINAL)
            return LAND_TERMINAL, self._info(t, soc, pose)
        # --- size the countdown from the MEASURED drain rate ----------
        if self.countdown_s is None:
            # RT2-1 bound/cross-check BEFORE the normal hold: a soc
            # channel that reports no drain while energy leaves the
            # pack (or a hold that has simply run too long) must not
            # park the layer in HOLD_METERING forever.
            if self._hold_start_t is None:
                self._hold_start_t = float(t)
            hold_elapsed = float(t) - self._hold_start_t
            if self.meter.soc_wh_disagree():
                forced = self._force_metering_decision(
                    t, soc, pose, "soc_wh_disagreement")
                if forced is not None:
                    return forced, self._info(t, soc, pose)
            elif hold_elapsed >= HOLD_METERING_MAX_S \
                    or self._hold_steps >= HOLD_METERING_MAX_STEPS:
                forced = self._force_metering_decision(
                    t, soc, pose, "hold_metering_bound")
                if forced is not None:
                    return forced, self._info(t, soc, pose)
        if self.countdown_s is None:
            rate = self.meter.drain_rate_soc_per_s()
            if self.meter.ready and rate and rate > 0.0:
                # Size from the CURRENT soc (the flag soc minus what
                # metering already spent), at the measured rate.
                self.countdown_s = max(
                    0.0, (soc - CRITICAL_SOC) / rate)
                self.countdown_end_t = float(t) + self.countdown_s
                self.events.append(
                    (float(t), "countdown_sized_from_measured_draw",
                     f"countdown_s={self.countdown_s:.1f} "
                     f"measured_draw_w="
                     f"{self.meter.measured_draw_w():.1f} "
                     f"drain_rate_soc_per_s={rate:.6f} "
                     f"flag_soc={self.flag_soc:.4f} "
                     f"floor_soc={CRITICAL_SOC} [measured, not "
                     f"model percentage]"))
                if self.countdown_s <= 0.0:
                    self.terminal = True
                    self._log_calibration(t, soc, pose, LAND_TERMINAL)
                    return LAND_TERMINAL, self._info(t, soc, pose)
            else:
                # Meter not ready yet: hold while it fills. The
                # countdown is running (soc keeps draining) — that
                # drain is exactly what buys the measurement. The
                # hold is BOUNDED (RT2-1, checked above): time,
                # step count, and the soc-vs-Wh cross-check all end
                # it with a decision, never an indefinite park.
                self._hold_steps += 1
                self._log_calibration(t, soc, pose, HOLD_METERING)
                return HOLD_METERING, self._info(t, soc, pose)
        # --- cascade: decide, then continuously re-check DOWN only ----
        if self.rung is None:
            self.rung, self.target_pose = self._pick_rung(pose, soc)
            self.events.append(
                (float(t), "cascade_decision",
                 f"rung={self.rung} target={self.target_pose} "
                 f"need_soc_base={self.energy_soc_needed(pose, self.base_pose)} "
                 f"need_soc_ckpt={self.energy_soc_needed(pose, self.tracker.checkpoint_pose)} "
                 f"soc={soc:.4f}"))
        else:
            # Continuous re-check of the CURRENT leg. Abort DOWN only.
            while self.rung in (RTB, RTC):
                ok = self.affordable(pose, self.target_pose, soc)
                if ok is None or ok:
                    break
                old = self.rung
                if old == RTB:
                    ckpt = self.tracker.checkpoint_pose
                    if self.affordable(pose, ckpt, soc):
                        self.rung, self.target_pose = RTC, ckpt
                    else:
                        self.rung = LAND_IN_PLACE
                        self.target_pose = pose
                else:  # RTC leg became unaffordable
                    self.rung = LAND_IN_PLACE
                    self.target_pose = pose
                self.n_abort_down += 1
                self.events.append(
                    (float(t), "cascade_abort_down",
                     f"{old}->{self.rung}: leg unaffordable at "
                     f"soc={soc:.4f} need_soc="
                     f"{self.energy_soc_needed(pose, self.target_pose)} "
                     f"measured_draw_w="
                     f"{self.meter.measured_draw_w()} [abort down "
                     f"only, never up]"))
        action = self.rung if self.rung in (RTB, RTC) else LAND_IN_PLACE
        self._log_calibration(t, soc, pose, action)
        return action, self._info(t, soc, pose)

    # -- records -------------------------------------------------------------
    def _info(self, t, soc, pose):
        return {
            "flag": self.flag_fired,
            "rung": self.rung,
            "target_pose": self.target_pose,
            "countdown_s": self.countdown_s,
            "countdown_remaining_s": (
                None if self.countdown_end_t is None
                else max(0.0, self.countdown_end_t - float(t))),
            "measured_draw_w": self.meter.measured_draw_w(),
            "drain_rate_soc_per_s": self.meter.drain_rate_soc_per_s(),
            "checkpoint_pose": self.tracker.checkpoint_pose,
            "checkpoint_t": self.tracker.checkpoint_t,
            "base_pose": self.base_pose,
            "terminal": self.terminal,
            "n_abort_down": self.n_abort_down,
            "hold_bound_reason": self.hold_bound_reason,
        }

    def _log_calibration(self, t, soc, pose, action):
        """Calibration row for this countdown step: the measured
        meter's raw material, kept so the countdown sizing can be
        audited / refit later (dual-purpose data doctrine)."""
        self.calibration_log.append({
            "t_s": round(float(t), 2),
            "soc": round(float(soc), 5),
            "measured_draw_w": (
                None if self.meter.measured_draw_w() is None
                else round(self.meter.measured_draw_w(), 2)),
            "drain_rate_soc_per_s": (
                None if self.meter.drain_rate_soc_per_s() is None
                else round(self.meter.drain_rate_soc_per_s(), 7)),
            "countdown_s": (
                None if self.countdown_s is None
                else round(self.countdown_s, 2)),
            "countdown_remaining_s": (
                None if self.countdown_end_t is None
                else round(max(0.0, self.countdown_end_t - float(t)), 2)),
            "action": action,
            "rung": self.rung,
            "dist_to_base_m": round(_dist(pose, self.base_pose), 3),
            "dist_to_checkpoint_m": round(
                _dist(pose, self.tracker.checkpoint_pose), 3),
            "need_soc_base": (
                None if self.energy_soc_needed(pose, self.base_pose) is None
                else round(self.energy_soc_needed(
                    pose, self.base_pose), 5)),
            "need_soc_checkpoint": (
                None if self.energy_soc_needed(
                    pose, self.tracker.checkpoint_pose) is None
                else round(self.energy_soc_needed(
                    pose, self.tracker.checkpoint_pose), 5)),
            "checkpoint_x": round(self.tracker.checkpoint_pose[0], 3),
            "checkpoint_y": round(self.tracker.checkpoint_pose[1], 3),
            "checkpoint_z": round(self.tracker.checkpoint_pose[2], 3),
            "checkpoint_t_s": round(self.tracker.checkpoint_t, 2),
        })
