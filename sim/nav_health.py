"""Navigation-health monitor — Phase 1, roadmap item 3 (2026-10-11).

A first-class escalation input: turns the estimator's OWN self-reports
into a categorical health state (HEALTHY / DEGRADED / UNTRUSTED) that
replaces the hardcoded pose_trustworthy boolean in the escalation
policy's RTH gate.

CRITICAL CONSTRAINT (the whole point of the build): every input is
computable ONLINE from quantities a real estimator reports about ITSELF
plus quantities the host already knows (its own commanded motion, its
own publish timestamps). Ground truth (ATE, true pose) is NEVER read
here. ATE is used OFFLINE ONLY, in run_nav_health_calibration.py, to
check that this online signal tracks real estimator health.

Inputs (per estimator update, nominal 1 Hz in this stack):
  est_speed_mps   estimator-reported speed magnitude: norm of the
                  estimator's own inter-update displacement / dt.
                  FAST-LIO2 publishes odometry; differencing consecutive
                  published poses is exactly this. [online]
  cmd_speed_mps   host-commanded speed magnitude (the flight controller
                  knows what it commanded). [online]
  n_features      effective tracked-feature count the estimator reports
                  (FAST-LIO2: points surviving voxel downsample / used in
                  the iterated update). In the mission loop the proxy is
                  the scan hit count — see NAV_HEALTH.md for the measured
                  caveat (whiteout backscatter INFLATES raw hits in this
                  sim, so this input is DEGRADED-only, never sufficient
                  for UNTRUSTED on its own). [online]
  residual_m      scan-matching residual / fitness the estimator reports
                  (FAST-LIO2 point-to-plane residuals). Optional: None
                  means "estimator does not report it" and the check is
                  skipped, not zero-filled. In the mission loop this is
                  MISSING (no scan matcher runs in the loop) — listed,
                  not invented. [online when reported]
  map_age_s       seconds since the last fresh published map/estimate
                  (D5 freshness signal, pipeline publish timestamps).
                  [online]
  est_jump_m      norm of the estimator's single-update pose jump.
                  Defaults to est_speed*dt when not supplied. [online]

The whiteout FREEZE is the load-bearing case: FAST-LIO2 in whiteout
outputs 3.4 m of motion for 27 m of true motion (12.5% motion ratio,
max inter-pose jump 0.061 m — FASTLIO2_BROADEN.md, [measured]) while
looking like a confident hover. The freeze rule below is built exactly
on that signature: commanded motion present, reported motion ~absent,
sustained. It does NOT need ground truth: the host compares what it
COMMANDED against what the estimator REPORTED.

RATE CONTRACT: the monitor assumes ~1 Hz updates (this stack's outer
step rate; all sustain/evidence counts are in UPDATES at that rate).
A faster native feed (FAST-LIO2 odometry ~10 Hz in the eval) MUST be
decimated to ~1 Hz before feeding this monitor — at 10 Hz the freeze
evidence would accumulate 10x faster in wall time and the jump
threshold (which scales with the report's dt_s) would correspond to a
different physical statement. dt_s is read per report; speeds are
per-second quantities and are rate-independent.

Claim label: host-side classical monitor, validated in simulation.
No TSU/Z1 claim (this is not a kernel).

CONSUMER INTERFACE (checkpoint gating — battery/escalation workers
build against this; Paul's battery safety design, 2026-10-11, defines
a CHECKPOINT as a position recorded while nav-health was trustworthy,
RTC = return to checkpoint):
  Name:      NavHealthMonitor.pose_trustworthy (bool property on the
             monitor instance; the same value is returned per update in
             info["pose_trustworthy"] and recorded per step in the
             calibration CSV's pose_trustworthy column).
  Semantics: True IFF the current health state is HEALTHY. DEGRADED
             and UNTRUSTED are both NOT trustworthy (False) — record a
             checkpoint only on True. This is deliberately the same
             semantics as the escalation RTH gate (gate open ==
             trustworthy), so checkpoint recording and RTH permission
             can never disagree about what "trustworthy" means.
  Cadence:   recomputed on EVERY monitor.update() call — nominal
             1 Hz in this stack (one call per outer step; a faster
             estimator feed must be decimated to ~1 Hz first, see
             RATE CONTRACT above). Consumers read the value AFTER the
             update for the interval they are about to act on; there
             is no asynchronous channel and no stale caching.
  Stickiness: SCOPE MATTERS (red-team round 2, N4 — an earlier version
             of this paragraph overgeneralized). (a) FREEZE LATCH:
             after a freeze latch, trustworthy stays False until 3
             consecutive COUNTED-clean updates UNDER COMMANDED MOTION
             release the latch (monitor level); the integrated policy
             HOVERs while UNTRUSTED, so in flight a True->False
             transition VIA THE LATCH is terminal for that flight leg.
             "Counted-clean" is narrower than "the reports look
             perfect": an update counts only when no freeze-family
             band fires on it (including the long-window aggregate,
             whose memory of the fault must first wash out) AND its
             score is 0 — and the partial-evidence contributor keeps
             the score nonzero until the UNCAPPED partial accumulator
             decays below 2.0 at 0.5/update. Release time therefore
             GROWS with fault duration; it is NOT a flat 3 updates
             after an arbitrary fault. Measured (NAV_HEALTH.md S2/S6):
             a 2-update freeze releases after 3 clean updates (6 to
             HEALTHY); a 10-update freeze mid-stream needs 49 clean
             updates to DEGRADED and 52 to HEALTHY; a 30-update
             freeze needs 176/179. (b) DEGRADED
             (non-latch): a True->False transition via DEGRADED is NOT
             final — DEGRADED recovers to HEALTHY at monitor level
             (one severity level per 3 consecutive clean updates) and
             pose_trustworthy returns True; a checkpoint consumer
             built to resume on trust return (the battery tracker's
             per-loss freeze) should do exactly that.
  Lag:       a lagging-but-consistent estimator is NOT a freeze
             (red-team round 2, N3). Reports that match a delayed
             setpoint (a single consistent lag of 1..LAG_TOLERANCE
             updates, armed at motion onsets / stream start / large
             setpoint steps) do not accumulate freeze-family evidence.
             Lag beyond LAG_TOLERANCE, or a lag that begins mid-motion
             with no onset, is treated as a freeze. See the LAG
             constants below and NAV_HEALTH.md S2/S6 for the exact
             tolerance, rationale, and limits.
             NOISE (red-team round 3, N3-noise): the match test is
             noise-robust. A report inside the noise band of a delayed
             setpoint (LAG_NOISE_EST_MAX against a hover-level delayed
             setpoint, LAG_NOISE_SLACK_MPS below the ratio threshold
             against a moving one) is UNRESOLVED — it neither confirms
             nor kills the hypothesis, and (hover-delayed case) the
             freeze family does not fire on it. A far mismatch still
             kills the hypothesis, but the hypothesis may RE-ARM
             inside the onset window if a later report strictly
             matches a delayed setpoint, so no single noisy update —
             in-band or far — can latch a healthy lagging estimator.
             Unresolved candidates expire with the onset window, so a
             sustained in-band freeze is still bounded by the
             tolerance + dwell (latch at the 7th onset update).
"""

from collections import deque

# --- thresholds, each with provenance -------------------------------------
# Freeze: reported/cmd motion ratio below this while cmd is real motion.
FREEZE_RATIO = 0.25      # [measured-derived: whiteout freeze ratio 0.125
                         #  (FASTLIO2_BROADEN.md, corrected version on
                         #  branch muse/fastlio2-fix-2026-10-11); healthy
                         #  ratio ~1.0 (straight-segment ATE 0.05-0.26 m
                         #  over metres of travel). 0.25 = 2x the frozen
                         #  ratio, 4x below healthy.]
FREEZE_CMD_MIN = 0.30    # m/s [design: instantaneous-rule floor — above
                         #  hover noise, below 1 m/s eval cruise / 2.0
                         #  V_MAX. Motion at/below this is covered by
                         #  the WINDOW rule below, not this rule.]
# Freeze evidence is LEAKY (red-team round 1, probe B): a frozen update
# adds 1.0, a clean update halves it; the latch needs FREEZE_EVIDENCE.
# Alternating frozen/clean therefore still accumulates (1, .5, 1.5, 1,
# 2 -> latch at the 5th update) while a single transient (1.0, decaying)
# never latches.
FREEZE_EVIDENCE = 2.0    # [design: = the old 2-consecutive dwell for a
                         #  solid freeze; longer for intermittent ones]
FREEZE_DECAY = 0.5       # [design: evidence SUBTRACTED on a clean update
                         #  (floor 0). Subtractive, not multiplicative:
                         #  a x0.5 leak has fixed point exactly at the
                         #  latch threshold for a 50%-duty intermittent
                         #  freeze and never fires; -0.5 reaches it in
                         #  5 updates (1, .5, 1.5, 1, 2).]
# Partial freeze / scale error (red-team probe A): reported motion a
# sustained fraction of commanded. Below PARTIAL_RATIO the estimator is
# losing real motion -> DEGRADED (gate closes). A SEVERE partial
# (<= SEVERE_PARTIAL_RATIO) sustained SEVERE_PARTIAL_SUSTAIN updates is
# treated as a freeze (pose error accrues >= 0.6 m/s at 1 m/s cmd).
PARTIAL_RATIO = 0.70     # [design: healthy achieved/setpoint ratio in
                         #  the calibration model is ~0.9; 0.70 leaves
                         #  margin below it and above the severe band]
SEVERE_PARTIAL_RATIO = 0.40  # [design]
SEVERE_PARTIAL_SUSTAIN = 10  # updates [design: 10 s at 1 Hz]
PARTIAL_EVIDENCE = 2.0   # leaky-evidence threshold for DEGRADED [design]
# Window rule (red-team probe C): over the last WINDOW updates, if the
# COMMANDED displacement sums to >= WINDOW_MIN_CMD_M while the REPORTED
# sums to < FREEZE_RATIO of it, that is a freeze regardless of how slow
# the motion is. At hover the commanded sum is ~0 and the rule cannot
# fire. The LONG_WINDOW rule below extends the same idea to slower
# motion (accrues 1 m of commanded travel over up to 120 updates).
WINDOW = 10              # updates [design]
WINDOW_MIN_CMD_M = 1.0   # m [design]
# Long-window rule (round-2 closure of the residual slow band): the
# same aggregate test over the last LONG_WINDOW updates. Detection is
# keyed to COMMANDED TRAVEL, not speed: once >= LONG_MIN_CMD_M of
# travel has been commanded inside the window, the aggregate
# reported/commanded ratio is classified with the same bands as the
# instantaneous rule (frozen < FREEZE_RATIO; partial < PARTIAL_RATIO;
# severe <= SEVERE_PARTIAL_RATIO). This catches total freeze down to
# ~0.05 m/s in ~21 updates and ~0.02 m/s in ~51 updates, plus slow
# partial and slow intermittent freezes the 10-update window dilutes.
# Principled floor: motion so slow that 120 s of it commands < 1.0 m
# (< ~0.008 m/s average) cannot fire it — but the undetected pose
# error in that band is bounded by the same 1.0 m of travel PER
# 120-update WINDOW, not cumulatively (red-team round 2, N2): bursts
# that are evicted from the window before reaching 1.0 m, repeated
# over many windows, accumulate UNBOUNDED total error while every
# single window stays under the bound. NAV_HEALTH.md S6 states the
# measured shapes; do not drop the per-window qualifier.
LONG_WINDOW = 120            # updates [design: 2 min at 1 Hz]
LONG_MIN_CMD_M = 1.0         # m of commanded travel [design: = WINDOW_MIN_CMD_M]
LONG_MIN_UPDATES = 5         # window must span at least this many updates
# Estimator lag (red-team round 2, N3): a pure pipeline delay makes
# est(t) track cmd(t-L) instead of cmd(t); at a motion onset the first
# L reports are ~the pre-onset (hover) reports and are indistinguishable
# from a freeze BY THE INSTANTANEOUS SIGNATURE. The monitor therefore
# keeps a CONSISTENT-LAG hypothesis: armed at a motion onset (stream
# start, a rise from hover-level cmd, or a setpoint step >=
# LAG_STEP_RESEED_MPS), it holds a set of candidate lags; each update
# keeps only the candidates whose delayed setpoint the report actually
# matches (ratio >= PARTIAL_RATIO against a moving delayed setpoint,
# or reported speed <= LAG_QUIET_EST_MAX against a hover-level delayed
# setpoint). While at least one candidate survives, the update is
# lag-consistent and the freeze family does not fire on it.
# Round-3 noise robustness (N3-noise): under the calibration harness's
# healthy noise model (achieved = 0.9*cmd + N(0, 0.02); report =
# achieved + N(0, 0.03); combined sigma ~0.036 m/s) a lagged HOVER
# report exceeds the strict 0.06 threshold on ~5% of updates (1.7
# sigma, one-sided), so the round-2 per-update kill latched healthy
# lagging estimators UNTRUSTED (Monte Carlo: 12-66 of 200 flights at
# lag 2-5). The kill is now graded by distance from the delayed
# setpoint, mirroring the freeze evidence's own leaky design:
#   match      — the strict test above; confirms the candidate.
#   unresolved — inside the noise band of SOME candidate's delayed
#                setpoint (hover-delayed: report <= LAG_NOISE_EST_MAX;
#                moving-delayed: report within LAG_NOISE_SLACK_MPS
#                below the ratio threshold). Unresolved neither
#                confirms nor kills: the candidate survives, and for
#                the hover-delayed case — where the expected report is
#                ~0 and the noise IS the signal scale — the freeze
#                family does not fire on the update. (Moving-delayed
#                unresolved only preserves the candidate; the update
#                is still judged normally, since at a moving setpoint
#                the ratio bands already carry the signal.)
#   far        — outside every candidate's noise band: kills the
#                hypothesis and the update is judged normally.
# A kill is not permanent inside the onset window: onsets/steps
# re-open the hypothesis (fresh candidate set + window age), and
# after a far kill a later report that STRICTLY matches a delayed
# setpoint re-arms it while the window age is <= LAG_TOLERANCE, so a
# single far spike (healthy-noise probability ~0.6% per update past
# the band edge, ~0.04% past 0.12) decays instead of latching.
# Unresolved candidates expire with the window age: a freeze whose
# report sits constantly inside the band stops being excused once
# its delayed setpoints have all turned moving / the window lapses,
# and latches at the 7th onset update (tolerance + dwell) — the
# noise band buys NO extra escape time beyond the disclosed price.
# CATCH-UP FORGIVENESS: the aggregate windows cannot see lag — their
# sums hold the lag dip as a freeze signature for updates after the
# lag resolves, and combined with one far spike's evidence that
# latched healthy lag-5 streams (extended Monte Carlo, pre-fix
# 15/2000). So: if the hypothesis STRICTLY MATCHED in the window
# AND the window holds at most LAG_FORGIVENESS_FAULT_BOUND counted
# faults, a far kill occurred inside the window, and within
# LAG_TOLERANCE of the kill a report matches the CURRENT setpoint
# (the delayed motion has demonstrably arrived), the window's OWN
# freeze/partial contribution is forgiven (contribution-scoped,
# round-4 C3: pre-window evidence and aggregate entries are
# restored as they would stand had the window's updates been
# clean, NOT wiped to zero). Round-4 fix: the gate was previously
# `matched OR faults <= 1` — one strictly-matched update (est 0.0
# on the onset update, the canonical first report of a freeze at
# an onset) set matched for the whole window and the fault count
# was never consulted, so matched + severe-partial x2..x9 (up to
# 6.85 m of error) was fully forgiven (REDTEAM4_NAV S2c, FAIL).
# The conjunctive gate closes that: a fault counted repeatedly
# in the window is never forgiven, matched or not. The bound (1)
# is the legitimate case the gate exists for: a lag-1/2 onset
# spike produces at most ONE counted fault in-window (the spike
# itself; the lagged hover reports around it are matched or
# unresolved, not counted), and the lag-5 aggregate-dip case the
# forgiveness was built for likewise holds exactly one counted
# fault (the far spike) beside its matched reports. Any second
# counted fault is, by the monitor's own bands, a repeated fault
# — not a spike. Boundary, measured (unchanged by the round-4
# gate for this shape, whose counted-fault count is 1): frozen
# reports spanning onset updates 1-6 followed by clean reports
# are forgiven (indistinguishable from a lag that marginally
# exceeds tolerance and catches up); a 7th frozen report latches
# at update 7, before any catch-up can arrive. That boundary is
# PER-ONSET and repeatable (round-4 C2, NAV_HEALTH.md S6).
LAG_TOLERANCE = 5        # updates [design: the largest lag the round-2
                         #  red team demonstrated as a false positive
                         #  (lag-5 onset + lag-5 speed ramp); at 1 Hz
                         #  this is a generous ceiling — the stack's
                         #  own command path lags are sub-second
                         #  (0.4 s filter tau, 1-step issued-setpoint
                         #  lag), so a healthy decimated feed should
                         #  sit far inside it]
LAG_QUIET_CMD_MAX = 0.05  # m/s [design: hover-level setpoint; at or
                         #  below this a delayed setpoint expects a
                         #  near-zero report]
LAG_QUIET_EST_MAX = 0.06  # m/s [design: ~2x the 0.03 m eval noise
                         #  sigma — a lagged hover report is hover
                         #  noise; deliberately BELOW the 0.10-0.1125
                         #  m/s a ratio-0.125 whiteout freeze reports
                         #  at cruise, so that signature is never
                         #  excused as lag, even at an onset]
LAG_NOISE_EST_MAX = 0.09  # m/s [design: noise-band edge for a
                         #  hover-delayed candidate, round-3 N3-noise:
                         #  2.5x the harness's combined healthy noise
                         #  sigma sqrt(0.02^2 + 0.03^2) ~= 0.036 m/s
                         #  (one-sided exceedance ~0.6% per update,
                         #  vs ~5% at the strict 0.06 edge). Still
                         #  BELOW the 0.10-0.1125 m/s whiteout
                         #  signature at cruise: that signature is a
                         #  FAR mismatch, never unresolved, so its
                         #  onset latch stays at update 2]
LAG_NOISE_SLACK_MPS = 0.11  # m/s [design: noise-band slack below the
                         #  PARTIAL_RATIO match threshold for a
                         #  moving delayed setpoint: ~3x the combined
                         #  healthy noise sigma 0.036 m/s]
LAG_STEP_RESEED_MPS = 0.30  # m/s [design: = FREEZE_CMD_MIN; a setpoint
                         #  step at least this large re-arms the lag
                         #  hypothesis — lag manifests at changes]
LAG_FORGIVENESS_FAULT_BOUND = 1  # counted faults [design, round-4:
                         #  the lag-1/2 onset spike — the legitimate
                         #  case catch-up forgiveness exists for —
                         #  produces at most 1 counted fault in-window;
                         #  a second counted fault is a repeated fault,
                         #  never a spike, and blocks forgiveness even
                         #  when the window also strictly matched]
# Pose jump: a single update moving the estimate further than the
# vehicle could physically fly in dt. Scales with the report's dt.
JUMP_MARGIN_M = 0.5      # m [design]
V_MAX_MPS = 2.0          # m/s [measured: vehicle_v3.py V_MAX]
# Feature starvation (DEGRADED contributor only — see module docstring).
FEATURES_DEGRADED = 10   # [measured anchor, corrected round 1: calibration
                         #  corpus clear-air means are 151.1 hits
                         #  (indoorcd_117, min 94) and 48.3 (indoorcd_122)
                         #  per 360-ray scan; a pose can transiently see
                         #  ~0, so this needs sustain and can never reach
                         #  UNTRUSTED alone]
FEATURE_SUSTAIN = 2      # updates [design]
# Residual (only when the estimator reports one).
RESIDUAL_DEGRADED_M = 0.10   # [design: ~3x the 0.03 m Mid-360 range noise
                             #  used in the FAST-LIO2 eval synthesis]
RESIDUAL_UNTRUSTED_M = 0.30  # [design: 10x eval range noise]
# Staleness (shares D5's measured anchor).
MAP_AGE_DEGRADED_S = 1.0     # [measured: D5 FIX 4 threshold, FRESHNESS_GATE.md]
MAP_AGE_UNTRUSTED_S = 3.0    # [design: 3x the D5 anchor, sustained absence
                             #  of any fresh estimate]
# Recovery hysteresis: consecutive fully-clean updates to step back up.
RECOVER_CLEAN = 3        # updates [design]

HEALTHY, DEGRADED, UNTRUSTED = "HEALTHY", "DEGRADED", "UNTRUSTED"
_SEV = {HEALTHY: 0, DEGRADED: 1, UNTRUSTED: 2}


class NavHealthMonitor:
    """Stateful per-update health classifier with dwell + hysteresis.

    update(report) -> (state, info). report is a dict; missing optional
    keys are treated as "not reported" (check skipped), never as zero.
    The monitor never sees a true pose.
    """

    def __init__(self):
        self.state = HEALTHY
        self._freeze_evidence = 0.0    # leaky freeze accumulator
        self._partial_evidence = 0.0   # leaky partial-freeze accumulator
        self._severe_partial_run = 0
        self._freeze_latched = False   # freeze fault latch (see update)
        self._motion_clean_run = 0     # clean updates UNDER MOTION
        self._feature_run = 0
        self._score_run = 0      # consecutive updates with score >= 3
        self._clean_run = 0
        self._window = deque(maxlen=WINDOW)  # (cmd_disp_m, est_disp_m)
        self._window_long = deque(maxlen=LONG_WINDOW)
        self._cmd_hist = deque(maxlen=LAG_TOLERANCE)  # past cmd speeds
        self._lag_candidates = set()  # consistent-lag hypothesis (N3)
        self._lag_onset_update = None  # n_updates at the last onset /
                                      # step that (re-)opened the lag
                                      # hypothesis; bounds unresolved
                                      # survival + post-kill re-arming
                                      # to the onset window (N3-noise)
        self._lag_window_matched = False  # hypothesis strictly matched
                                      # at least once in this window:
                                      # positive evidence the stream
                                      # really is lagging (gates catch-up
                                      # forgiveness, N3-noise)
        self._lag_kill_update = None  # n_updates of the last far kill
                                      # inside the onset window, if any
        self._lag_window_faults = 0   # freeze-family-counted updates
                                      # in this onset window (a second
                                      # forgiveness gate: a window whose
                                      # reports were repeatedly counted
                                      # as faults — e.g. a partial
                                      # freeze at an onset — is not a
                                      # lag window, matched or not)
        # Contribution-scoped forgiveness snapshot (round-4 C3):
        # evidence + aggregate entries as they stood BEFORE the
        # current onset window's first update. Forgiveness restores
        # these (evidence decayed by the window's pre-catch-up
        # updates, as if those updates had been clean) instead of
        # wiping the GLOBAL accumulators to zero, so evidence from
        # an independent earlier fault survives a lag forgiveness.
        self._lag_snap_freeze = 0.0
        self._lag_snap_partial = 0.0
        self._lag_snap_severe = 0
        self._lag_snap_window = []
        self._lag_snap_window_long = []
        self.n_updates = 0
        self.trace = []          # (t, state, reasons) for the corpus

    @property
    def pose_trustworthy(self):
        """THE checkpoint-gating signal (see module docstring,
        CONSUMER INTERFACE): True iff state == HEALTHY; DEGRADED and
        UNTRUSTED are both False. RTH-gate semantics are the same:
        only HEALTHY opens the gate [design: conservative — a
        degraded estimate is exactly when you do not fly a route home
        on it, and exactly when you do not record a checkpoint].
        Updated by every update() call (nominal 1 Hz)."""
        return self.state == HEALTHY

    def update(self, report, t=0.0):
        self.n_updates += 1
        reasons = []
        score = 0

        cmd = report.get("cmd_speed_mps")
        est = report.get("est_speed_mps")
        dt = float(report.get("dt_s", 1.0))
        jump = report.get("est_jump_m")
        if jump is None and est is not None:
            jump = est * dt

        # Hard rule 1 — pose jump beyond physical reach in dt: instant
        # UNTRUSTED (a teleporting estimate is not a degraded one).
        hard = False
        hard_jump = False
        jump_max = V_MAX_MPS * dt + JUMP_MARGIN_M
        if jump is not None and jump > jump_max:
            hard = True
            hard_jump = True
            reasons.append(f"pose_jump {jump:.2f} m > {jump_max:.2f} m "
                           f"[calculated: V_MAX*dt+margin]")

        # Freeze family (red-team round 1, probes A/B/C). All evidence
        # is LEAKY: frozen updates add, clean updates decay — no hard
        # reset on a single clean update.
        frozen_now = False     # ratio < FREEZE_RATIO under motion
        partial_now = False    # FREEZE_RATIO <= ratio < PARTIAL_RATIO
        severe_now = False     # ratio <= SEVERE_PARTIAL_RATIO sustained
        ratio = None
        lag_match = False      # report matches a consistent delayed
                               # setpoint (N3) — not a freeze signature
        lag_excused = False    # lag_match OR hover-delayed unresolved:
                               # the freeze family does not fire (N3-noise)
        if cmd is not None and est is not None:
            # Pre-update snapshot candidates (round-4 C3): if this
            # update turns out to be an onset (computed below), these
            # become the window's contribution-scope snapshot — the
            # state BEFORE this window's first update is counted.
            _pre_freeze = self._freeze_evidence
            _pre_partial = self._partial_evidence
            _pre_severe = self._severe_partial_run
            _pre_window = list(self._window)
            _pre_window_long = list(self._window_long)
            # Window bookkeeping (per-update displacements) for ALL
            # motion levels, including slow flight under the floor.
            self._window.append((cmd * dt, est * dt))
            self._window_long.append((cmd * dt, est * dt))
            # Consistent-lag hypothesis (N3, noise-robust round 3).
            # Armed at motion onsets: stream start, a rise from
            # hover-level cmd, or a setpoint step >=
            # LAG_STEP_RESEED_MPS; an onset/step RE-OPENS it (fresh
            # candidates + window age) even if a narrowed set is still
            # alive — lag manifests at changes. Delayed setpoints
            # before the stream start are taken as hover (0.0) — the
            # onset case: a lagging estimator entering motion reports
            # its pre-onset hover stream first. After a far kill the
            # hypothesis may RE-ARM inside the onset window (age <=
            # LAG_TOLERANCE): the full candidate set is re-tried and
            # survives only on a strict match / in-band report.
            prev_cmd = self._cmd_hist[-1] if self._cmd_hist else None
            is_onset = cmd > LAG_QUIET_CMD_MAX and (
                prev_cmd is None or prev_cmd <= LAG_QUIET_CMD_MAX
                or cmd - prev_cmd >= LAG_STEP_RESEED_MPS)
            rearm_attempt = False
            if is_onset:
                self._lag_candidates = set(range(1, LAG_TOLERANCE + 1))
                self._lag_onset_update = self.n_updates
                self._lag_window_matched = False
                self._lag_kill_update = None
                self._lag_window_faults = 0
                self._lag_snap_freeze = _pre_freeze
                self._lag_snap_partial = _pre_partial
                self._lag_snap_severe = _pre_severe
                self._lag_snap_window = _pre_window
                self._lag_snap_window_long = _pre_window_long
            elif not self._lag_candidates \
                    and self._lag_onset_update is not None \
                    and self.n_updates - self._lag_onset_update \
                    <= LAG_TOLERANCE:
                # Post-kill re-arm attempt inside the onset window.
                self._lag_candidates = set(range(1, LAG_TOLERANCE + 1))
                rearm_attempt = True
            if self._lag_candidates:
                matched, unres, unres_hover = set(), set(), set()
                explains = set()  # candidates whose delayed setpoint
                # differs from the CURRENT cmd: a match against them
                # is explained by lag and NOT by no-lag (a match
                # against delayed == cmd is vacuous — the report is
                # equally consistent with a healthy unlagged stream)
                for lag in self._lag_candidates:
                    delayed = self._cmd_hist[-lag] \
                        if lag <= len(self._cmd_hist) else 0.0
                    if delayed > LAG_QUIET_CMD_MAX:
                        if est >= PARTIAL_RATIO * delayed:
                            matched.add(lag)
                        elif est >= PARTIAL_RATIO * delayed \
                                - LAG_NOISE_SLACK_MPS:
                            unres.add(lag)
                        else:
                            continue
                    elif est <= LAG_QUIET_EST_MAX:
                        matched.add(lag)
                    elif est <= LAG_NOISE_EST_MAX:
                        unres.add(lag)
                        unres_hover.add(lag)
                    else:
                        continue
                    if abs(delayed - cmd) > LAG_QUIET_CMD_MAX:
                        explains.add(lag)
                window_age = None if self._lag_onset_update is None \
                    else self.n_updates - self._lag_onset_update
                if window_age is not None and window_age > LAG_TOLERANCE:
                    unres, unres_hover = set(), set()  # band expires
                if matched:
                    lag_match = True
                    lag_excused = True
                    self._lag_candidates = matched | unres
                    reasons.append(
                        f"lag-consistent: est matches setpoint delayed "
                        f"by {sorted(matched)} update(s) [N3 tolerance "
                        f"{LAG_TOLERANCE}]")
                elif unres:
                    # UNRESOLVED (N3-noise): inside the noise band of
                    # a delayed setpoint — the hypothesis survives
                    # unconfirmed. Only a hover-delayed unresolved
                    # excuses the update (there the expected report
                    # is ~0 and healthy noise alone crosses the
                    # strict edge on ~5% of updates); a moving-delayed
                    # unresolved merely keeps the candidate alive.
                    self._lag_candidates = unres
                    if unres_hover:
                        lag_excused = True
                    reasons.append(
                        f"lag-unresolved: est within noise band of "
                        f"setpoint delayed by {sorted(unres)} "
                        f"update(s) [N3-noise: hypothesis survives, "
                        f"not confirmed]")
                else:
                    # FAR mismatch: hypothesis dies; the update is
                    # judged normally below. Inside the onset window a
                    # later strict match can still re-arm it (above).
                    self._lag_candidates = set()
                    if window_age is not None \
                            and window_age <= LAG_TOLERANCE:
                        self._lag_kill_update = self.n_updates
                if rearm_attempt and not explains and (matched or unres):
                    # Vacuous re-arm (N4 guard): every surviving
                    # candidate's delayed setpoint equals the current
                    # cmd, so this report is equally explained by
                    # no-lag — it must not resurrect the hypothesis
                    # (on constant cmd that would let clean reports
                    # suppress the long-window washout the freeze
                    # release rule is measured on). Judged normally.
                    self._lag_candidates = set()
                    lag_match = False
                    lag_excused = False
                    reasons.append(
                        "lag re-arm vacuous: surviving delayed "
                        "setpoint(s) equal the current cmd — report "
                        "equally explained by no-lag; judged normally "
                        "[N4 guard]")
                if lag_match:
                    self._lag_window_matched = True
            self._cmd_hist.append(cmd)
            # Catch-up forgiveness (N3-noise; round-4 conjunctive
            # gate + contribution scope): inside this onset window
            # the hypothesis STRICTLY MATCHED at least once AND the
            # window holds at most LAG_FORGIVENESS_FAULT_BOUND
            # freeze-family-counted updates (the far spike itself —
            # the legitimate lag-1/2 spike produces exactly one),
            # a far update killed the hypothesis, and NOW the
            # delayed motion demonstrably arrives (the report matches
            # the CURRENT setpoint) within LAG_TOLERANCE of the kill
            # — the stream was lagging, not frozen, so the window's
            # OWN freeze-family contribution (including the aggregate
            # windows' memory of the lag dip) is forgiven. Round-4:
            # the pre-fix gate was `matched OR faults <= 1`; one
            # matched update unlocked forgiveness for up to 9
            # counted severe-partial faults (REDTEAM4 S2c) — now
            # BOTH conditions must hold, so a repeatedly-counted
            # fault is never forgiven, matched or not. Scope
            # (round-4 C3): forgiveness RESTORES the pre-window
            # snapshot — evidences decayed by the window's
            # pre-catch-up updates (as if those updates had been
            # clean; this catch-up update's own decay is applied by
            # the normal evidence step below) and aggregate deques
            # holding only pre-window entries plus this catch-up
            # entry — instead of wiping the global accumulators, so
            # evidence from an independent earlier fault survives.
            # (The severe run is a consecutive-run counter, not an
            # accumulator: this clean catch-up update resets it via
            # the normal severe step below regardless, so restoring
            # its snapshot here has no separate effect.)
            if not self._freeze_latched \
                    and self._lag_window_matched \
                    and self._lag_window_faults \
                    <= LAG_FORGIVENESS_FAULT_BOUND \
                    and self._lag_kill_update is not None \
                    and 1 <= self.n_updates - self._lag_kill_update \
                    <= LAG_TOLERANCE \
                    and cmd > FREEZE_CMD_MIN \
                    and est >= PARTIAL_RATIO * cmd:
                _n_pre = 0 if self._lag_onset_update is None else \
                    max(0, self.n_updates - self._lag_onset_update)
                self._freeze_evidence = max(
                    0.0, self._lag_snap_freeze - FREEZE_DECAY * _n_pre)
                self._partial_evidence = max(
                    0.0, self._lag_snap_partial - FREEZE_DECAY * _n_pre)
                self._severe_partial_run = self._lag_snap_severe
                self._window = deque(self._lag_snap_window,
                                     maxlen=WINDOW)
                self._window.append((cmd * dt, est * dt))
                self._window_long = deque(self._lag_snap_window_long,
                                          maxlen=LONG_WINDOW)
                self._window_long.append((cmd * dt, est * dt))
                self._lag_kill_update = None
                self._lag_window_matched = False
                self._lag_window_faults = 0
                reasons.append(
                    "lag catch-up: delayed motion arrived within "
                    "the onset window; window freeze evidence "
                    "forgiven [N3-noise, round-4 scoped]")
            if not lag_excused and cmd > FREEZE_CMD_MIN:
                ratio = est / cmd
                if ratio < FREEZE_RATIO:
                    frozen_now = True
                    reasons.append(
                        f"freeze ratio {ratio:.3f} < {FREEZE_RATIO} "
                        f"(cmd {cmd:.2f}, est {est:.2f} m/s)")
                elif ratio < PARTIAL_RATIO:
                    partial_now = True
                    reasons.append(
                        f"partial ratio {ratio:.3f} < {PARTIAL_RATIO} "
                        f"(cmd {cmd:.2f}, est {est:.2f} m/s)")
            # Window rule: slow-motion freeze (probe C). Fires on
            # cumulative commanded travel, not instantaneous speed.
            if not lag_excused and len(self._window) >= 3:
                w_cmd = sum(c for c, _ in self._window)
                w_est = sum(e for _, e in self._window)
                if w_cmd >= WINDOW_MIN_CMD_M and \
                        w_est < FREEZE_RATIO * w_cmd:
                    frozen_now = True
                    reasons.append(
                        f"window freeze: reported {w_est:.2f} m of "
                        f"commanded {w_cmd:.2f} m over "
                        f"{len(self._window)} updates [probe C rule]")
            # Long-window rule: same aggregate bands over up to
            # LONG_WINDOW updates — closes the sub-0.1 m/s total
            # freeze band and slow partial / slow intermittent
            # freezes (aggregate ratio is dilution-proof once the
            # commanded-travel floor is met).
            if not lag_excused and len(self._window_long) >= LONG_MIN_UPDATES:
                lw_cmd = sum(c for c, _ in self._window_long)
                lw_est = sum(e for _, e in self._window_long)
                if lw_cmd >= LONG_MIN_CMD_M:
                    lw_ratio = lw_est / lw_cmd
                    if lw_ratio < FREEZE_RATIO:
                        if not frozen_now:
                            reasons.append(
                                f"long-window freeze: reported "
                                f"{lw_est:.2f} m of commanded "
                                f"{lw_cmd:.2f} m over "
                                f"{len(self._window_long)} updates")
                        frozen_now = True
                    elif lw_ratio < PARTIAL_RATIO:
                        if not partial_now:
                            reasons.append(
                                f"long-window partial ratio "
                                f"{lw_ratio:.3f} < {PARTIAL_RATIO} over "
                                f"{len(self._window_long)} updates "
                                f"(cmd {lw_cmd:.2f} m)")
                        partial_now = True
                        if lw_ratio <= SEVERE_PARTIAL_RATIO:
                            severe_now = True
        if frozen_now:
            self._freeze_evidence += 1.0
        else:
            self._freeze_evidence = max(0.0,
                                        self._freeze_evidence - FREEZE_DECAY)
        if partial_now:
            self._partial_evidence += 1.0
        else:
            self._partial_evidence = max(
                0.0, self._partial_evidence - FREEZE_DECAY)
        if self._freeze_evidence >= FREEZE_EVIDENCE:
            hard = True
            if not self._freeze_latched:
                self._freeze_latched = True
                # The latch now holds the fault; clear the windows so
                # recovery is judged on POST-latch behavior only —
                # otherwise the windows' memory of the freeze itself
                # (aggregate ratio < PARTIAL_RATIO for tens of updates)
                # would block the clean-under-motion release accounting
                # long after the estimator demonstrably recovered.
                self._window.clear()
                self._window_long.clear()
            reasons.append(f"freeze evidence {self._freeze_evidence:.1f} "
                           f">= {FREEZE_EVIDENCE} [leaky dwell]")
        # Severe partial sustained -> freeze-class (latched). The
        # severe condition may come from the instantaneous ratio
        # (fast flight) or the long-window aggregate (slow flight).
        if partial_now and ((ratio is not None and
                             ratio <= SEVERE_PARTIAL_RATIO) or severe_now):
            self._severe_partial_run += 1
            if self._severe_partial_run >= SEVERE_PARTIAL_SUSTAIN:
                hard = True
                if not self._freeze_latched:
                    self._freeze_latched = True
                    self._window.clear()       # see freeze-evidence
                    self._window_long.clear()  # latch site above
                reasons.append(
                    f"severe partial sustained "
                    f"{self._severe_partial_run} updates [design]")
        else:
            self._severe_partial_run = 0
        # Onset-window fault accounting (N3-noise): updates the
        # freeze family actually counted, while the onset window (or
        # its catch-up horizon) is open — gates catch-up forgiveness.
        if (frozen_now or partial_now) \
                and self._lag_onset_update is not None \
                and self.n_updates - self._lag_onset_update \
                <= 2 * LAG_TOLERANCE:
            self._lag_window_faults += 1

        # Scored contributors.
        nf = report.get("n_features")
        if nf is not None and nf < FEATURES_DEGRADED:
            self._feature_run += 1
            if self._feature_run >= FEATURE_SUSTAIN:
                score += 1
                reasons.append(f"features {nf} < {FEATURES_DEGRADED} "
                               f"sustained {self._feature_run}")
        else:
            self._feature_run = 0

        res = report.get("residual_m")
        if res is not None:
            if res > RESIDUAL_UNTRUSTED_M:
                score += 2
                reasons.append(f"residual {res:.3f} m > "
                               f"{RESIDUAL_UNTRUSTED_M} [design]")
            elif res > RESIDUAL_DEGRADED_M:
                score += 1
                reasons.append(f"residual {res:.3f} m > "
                               f"{RESIDUAL_DEGRADED_M} [design]")

        age = report.get("map_age_s")
        if age is not None:
            if age >= MAP_AGE_UNTRUSTED_S:
                # NOTE (intended ceiling, probe E): staleness ALONE
                # caps at DEGRADED — a stale-but-consistent estimate is
                # D5's hover case, not proof of estimator failure.
                # UNTRUSTED needs a second channel (score >= 3 path).
                score += 2
                reasons.append(f"map_age {age:.1f} s >= "
                               f"{MAP_AGE_UNTRUSTED_S} [design]")
            elif age >= MAP_AGE_DEGRADED_S:
                score += 1
                reasons.append(f"map_age {age:.1f} s >= "
                               f"{MAP_AGE_DEGRADED_S} [measured: D5]")
        # Partial-freeze evidence is a DEGRADED contributor.
        if self._partial_evidence >= PARTIAL_EVIDENCE:
            score += 1
            reasons.append(f"partial-freeze evidence "
                           f"{self._partial_evidence:.1f} [probe A rule]")
        self._score_run = self._score_run + 1 if score >= 3 else 0

        # Freeze latch [design]: a frozen estimator does not earn trust
        # back by sitting still quietly (at hover both cmd and est are
        # ~0 and every check passes vacuously — without the latch the
        # policy would release, fly blind, re-freeze, and limit-cycle).
        # Unlatching requires RECOVER_CLEAN consecutive COUNTED-clean
        # updates UNDER COMMANDED MOTION: the estimator must demonstrate
        # agreement while actually moving. COUNTED-clean (N4) is the
        # clean_motion predicate below: score == 0 (which requires the
        # UNCAPPED partial-evidence accumulator — +1 per partial
        # update, -0.5 per non-partial update, floor 0 — to have
        # decayed below PARTIAL_EVIDENCE), no freeze-family band
        # firing on the update (the window/long-window aggregates keep
        # firing on their memory of the fault until clean travel washes
        # the aggregate ratio back over PARTIAL_RATIO), and no pose
        # jump. A fault of duration D therefore takes O(D) perfect
        # updates to release, NOT a flat RECOVER_CLEAN — the exact
        # measured release rule is in NAV_HEALTH.md S2/S6 and is
        # locked by test_long_freeze_release_latency_measured. The
        # accumulators are deliberately NOT capped: the uncapped tail
        # is what makes intermittent faults accumulate toward the
        # latch instead of being forgiven at a cap.
        # (Excludes hard_jump only: residual freeze EVIDENCE decaying
        # from a past latch still trips the evidence threshold for an
        # update or two and must not block the release accounting —
        # round-1 fix; a genuinely clean update is ratio-healthy,
        # score-free, and jump-free.)
        clean_motion = (score == 0 and not frozen_now and not partial_now
                        and not hard_jump
                        and cmd is not None and cmd > FREEZE_CMD_MIN)
        self._motion_clean_run = self._motion_clean_run + 1 \
            if clean_motion else 0
        if self._freeze_latched:
            if self._motion_clean_run >= RECOVER_CLEAN:
                # Release: clear the latch AND the evidence (the
                # estimator has demonstrated clean motion; evidence
                # starts fresh — otherwise this same update's residual
                # evidence re-latches instantly). Return early so the
                # generic classification cannot re-fire on stale state.
                self._freeze_latched = False
                self._freeze_evidence = 0.0
                self._partial_evidence = 0.0
                self._severe_partial_run = 0
                self._window.clear()
                self._window_long.clear()
                self.state = DEGRADED
                self._clean_run = 0
                reasons.append("freeze latch released: clean under "
                               "motion x3 [design]")
                info = {"state": self.state, "score": score,
                        "hard": False, "reasons": reasons,
                        "pose_trustworthy": self.pose_trustworthy}
                self.trace.append((t, self.state, "; ".join(reasons)))
                return self.state, info
            self.state = UNTRUSTED
            info = {"state": self.state, "score": score, "hard": hard,
                    "reasons": reasons,
                    "pose_trustworthy": self.pose_trustworthy}
            self.trace.append((t, self.state, "; ".join(reasons)))
            return self.state, info

        # Classification: escalation is immediate; de-escalation steps
        # down ONE level per RECOVER_CLEAN consecutive clean updates.
        if hard or self._score_run >= 2:
            self.state = UNTRUSTED
            self._clean_run = 0
        elif score >= 1:
            if _SEV[DEGRADED] > _SEV[self.state]:
                self.state = DEGRADED
            self._clean_run = 0
        else:
            self._clean_run += 1
            if self._clean_run >= RECOVER_CLEAN and self.state != HEALTHY:
                self.state = DEGRADED if self.state == UNTRUSTED else HEALTHY
                self._clean_run = 0
        info = {"state": self.state, "score": score, "hard": hard,
                "reasons": reasons, "pose_trustworthy": self.pose_trustworthy}
        self.trace.append((t, self.state, "; ".join(reasons)))
        return self.state, info
