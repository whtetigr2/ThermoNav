# Navigation-Health Model — Phase 1, roadmap item 3

> **ThermoNav port note (2026-10-11, graduation):** this report is the
> research-repo build record, ported from `slow-drone/dynamics_sim/`
> in thermo-work (branch `muse/roadmap-navhealth-2026-10-11` @
> `7828341`, fix commit `c06984f`). Red-team round 5 returned **PASS
> WITH CAVEATS (documentation-only)** on that tree and ruled the C2
> residual (§2(iv)/§6: repeated ×6-frozen onset bursts harvesting
> forgiveness — ×6 burst = 6.00 m forgotten, 10 bursts = 60.00 m,
> HEALTHY throughout) an **acceptable DISCLOSED limit** — no budget
> fix; it is carried here verbatim in substance and is not re-opened.
> In ThermoNav the layer lives in `sim/`: `sim/nav_health.py`,
> `sim/exec_provenance.py`, the nav rows merged into
> `sim/escalation.py` (preserving its DISARM + link + battery wiring —
> the decider kernel is byte-identical), the mission-loop integration
> in `sim/mission_dynamics_v5.py`, `sim/test_nav_health.py` (24 tests),
> `sim/test_exec_provenance.py` (7 tests), `sim/test_nav_checkpoint.py`
> (the checkpoint wiring), and the two harnesses
> `sim/run_nav_health_calibration.py` /
> `sim/run_nav_health_mission_validation.py`. Layout differences from
> the research repo: ThermoNav's mission loop has no jitter controller
> (map_age stays 0.0; the calibration harness is unaffected, the
> mission harness's paired-clear cells run without the jitter
> component) and occupancy grids load from `../rooms`. **The battery
> checkpoint wiring — the recorded follow-up of the battery
> graduation — is done here:** the checkpoint's default trust path
> is this monitor (`resolve_nav_trustworthy` in
> `mission_dynamics_v5.py`: monitor active by default whenever the
> battery safety layer is on and no scripted override is supplied;
> checkpoint semantics unchanged — a position recorded while
> `pose_trustworthy` was True). The old `nav_trustworthy_stub`
> survives only as the scripted scenario override (deterministic RTC
> constructions), documented in `battery_safety.py`. ThermoNav-side
> validation results are recorded in §8 at the end of this file.

**Status:** BUILT + VALIDATED in simulation; red-team round 1 FAIL
addressed (see §7 for the per-finding correction record), plus a
round-2 closure pass (§7): the residual sub-0.1 m/s freeze band is
closed by a long-window aggregate rule, the provenance guard is
hardened and unit-tested, and both re-issued CSVs carry
checkout-and-reproduce proof. Red-team round 2 returned PASS WITH
CAVEATS; its five caveats (N1a provenance rename hole, N1b guard
scope disclosure, N2 aggregate-band evasions, N3 estimator-lag false
positive, N4 stickiness/recovery timing) are addressed in the
round-2 fix pass recorded at the end of §7 — N1a and N3 in code,
N1b/N2 as explicit disclosures with measured numbers, N4 as a
documentation correction to the measured release rule (locked by a
new test). Red-team round 3 confirmed N1a/N1b/N2/N4 but returned
FAIL on the N3 fix under noise (healthy lag × the harness's own
report noise latched UNTRUSTED in up to 66/200 flights); the
round-3 fix pass (noise-robust lag hypothesis, locked by seeded
noise-on tests) is recorded at the end of §7. Red-team round 4
confirmed the round-3 noise fix but returned FAIL on catch-up
forgiveness (its OR gate forgave repeatedly-counted faults on the
strength of one matched update); the round-4 fix pass
(conjunctive forgiveness gate + contribution-scoped clearing,
locked by seeded RT4-shape tests) is recorded at the end of §7.
**Claim label:** host-side classical monitor + policy wiring, validated
in simulation. No TSU/Z1 claim (this is not a kernel), no silicon claim.
**Code:** `nav_health.py` (monitor), `escalation.py` (wiring),
`mission_dynamics_v5.py` (loop integration, cfg-gated, default OFF),
`exec_provenance.py` (executed-tree provenance guard for the harnesses).
**Tests:** `test_nav_health.py` (24 tests — the round-1/2 sixteen
plus lag-tolerance, lag-limit, measured release-latency, the
round-3 seeded noise-on lag Monte Carlo + freeze-escape audit,
and the round-4 conjunctive-gate / scoped-forgiveness /
burst-boundary tests),
`test_exec_provenance.py` (7 tests — the provenance guard seen to
bite, including the round-2 rename-onto-output attack) + the
pre-existing `test_escalation.py` suite (unchanged behavior on
legacy calls).
**Data:** `nav_health_calibration.csv`, `nav_health_mission_validation.csv`
(dual-purpose: every lever an explicit column, executed commit per row,
verified at write time — §7).

## 1. What it is

A categorical estimator-health signal — HEALTHY / DEGRADED / UNTRUSTED —
computed online from the estimator's own self-reports, wired into the
escalation ladder as the `nav_health` input. It replaces the
`pose_trustworthy` boolean that the mission loop hardcoded `False` and
retires the `nav_health` MISSING stub in `EscalationPolicy` (battery
and link remain MISSING — separate roadmap items).

Wiring (all in `escalation.EscalationPolicy.evaluate(nav_health=...)`;
omitting the argument reproduces legacy behavior exactly):

- RTH gate: `pose_trustworthy AND (nav_health == HEALTHY)`. DEGRADED
  closes the gate too [design: conservative].
- UNTRUSTED is a trigger on its own: HOVER immediately; after 30 s
  sustained [design: mirrors the degradation RTH row] it reaches the
  RTH row — which the (closed) gate converts to land-in-place, logged
  `rth_gated_land_in_place`. **Commit latency is ~39–40 s, not 30 s**
  (30 s timer + the 10 s hold-first override window; red-team probe G).
  RTH can never be emitted on an UNTRUSTED estimate; unit tests assert
  this on every step of a 45-step drive, including `_current` never
  holding a pre-gate RTH.

### Consumer interface — the trustworthiness signal (checkpoint gating)

Paul's battery safety design (2026-10-11) defines a CHECKPOINT as a
position recorded while nav-health was trustworthy (RTC = return to
checkpoint, RTB = return to base). The signal the battery/escalation
side gates checkpoint recording on is:

- **Name:** `NavHealthMonitor.pose_trustworthy` — a bool property on
  the monitor instance (`nav_health.py`). The same value is returned
  per update in `info["pose_trustworthy"]`, and recorded per step in
  the calibration CSV's `pose_trustworthy` column.
- **Semantics:** True **iff** the current state is HEALTHY. DEGRADED
  and UNTRUSTED are both False — record a checkpoint only on True.
  This is deliberately identical to the escalation RTH-gate semantics
  (gate open == trustworthy), so checkpoint recording and RTH
  permission can never disagree about what "trustworthy" means.
- **Cadence:** recomputed on every `monitor.update()` call — nominal
  1 Hz in this stack (one call per outer step; a faster estimator
  feed must be decimated to ~1 Hz first, §2 rate contract). Read it
  after the update for the interval you are about to act on; there
  is no async channel and no caching.
- **Stickiness (corrected, round-2 N4):** the two False paths are
  NOT the same. (a) *Freeze latch:* it stays False until 3
  consecutive **counted-clean** updates under commanded motion
  (monitor level) — and "counted-clean" is strictly narrower than
  "the reports look perfect again" (the actual release rule and
  measured latencies are in §2/§6: release time grows with fault
  duration; a 10-update freeze mid-stream needs 49 clean updates to
  DEGRADED, 52 to HEALTHY — measured with clean reports at ratio
  1.0; under the stack's 0.9 healthy report model the same freeze
  needs 74/77, §2). In flight the policy's HOVER response
  makes the latch release unreachable, so a True→False transition
  **via the latch** is terminal for the flight leg (§6).
  (b) *DEGRADED (non-latch):* a True→False transition via DEGRADED
  is **not** final — DEGRADED recovers to HEALTHY at monitor level
  and this signal returns True; a consumer built to resume on trust
  return (the battery CheckpointTracker's per-loss freeze) should
  do exactly that. An earlier version of this section (and of the
  `nav_health.py` docstring) stated the latch's finality for every
  True→False transition; that was wrong for DEGRADED and is fixed.

## 2. The signal — every input online-computable

| Input | Source on a real system | Used for |
|---|---|---|
| `est_speed_mps` | norm of consecutive published odometry poses / dt (FAST-LIO2 publishes odometry) | freeze family, jump |
| `cmd_speed_mps` | the controller's **setpoint** — what the host commanded, not what the vehicle achieved (round-1 F3: both harnesses now feed setpoint-derived cmd; see §3) | freeze family denominator |
| `est_jump_m` | single-update pose displacement | teleport rule (threshold scales with the report's `dt_s`) |
| `n_features` | estimator's effective tracked-feature count (FAST-LIO2 reports points used in its update) | starvation contributor |
| `residual_m` | estimator's scan-matching residual/fitness, when reported | scored contributor |
| `map_age_s` | pipeline publish timestamps (the D5 freshness signal) | staleness contributor |

Ground truth (true pose, ATE) appears NOWHERE in the decision path —
`test_no_truth_in_monitor` source-audits the monitor, and
`test_wired_path_audit` extends the audit to both harnesses' cmd
channels. ATE-class truth is used offline only (§4), and the achieved
motion there is a modeled quantity, not the cmd channel.

**Rate contract:** the monitor assumes ~1 Hz updates (this stack's
outer-step rate; all evidence/sustain counts are in updates at that
rate). A faster native feed (FAST-LIO2 odometry ~10 Hz in the eval)
MUST be decimated to ~1 Hz first — at 10 Hz the evidence would
accumulate 10× faster in wall time. The jump threshold scales with
each report's `dt_s`.

### Rules (thresholds + provenance)

- **FREEZE family (the load-bearing rules).** All freeze evidence is
  **leaky** (round-1 F2): a frozen update adds 1.0, a clean update
  subtracts 0.5 (floor 0) — no hard reset. (Subtractive, not
  multiplicative: a ×0.5 leak has its fixed point exactly at the latch
  threshold for a 50%-duty intermittent freeze and never fires.)
  - *Total freeze:* reported/commanded ratio < 0.25 under cmd >
    0.30 m/s accumulates evidence; evidence ≥ 2.0 → UNTRUSTED,
    latched. Ratio threshold [measured-derived]: the whiteout freeze
    measured 0.125 (3.4 m reported / 27 m true — FASTLIO2_BROADEN.md,
    **corrected version on branch `muse/fastlio2-fix-2026-10-11`,
    commit `64cf35e`**; the original broaden branch states the
    opposite characterization and must not be cited for this).
    0.25 = 2× the frozen ratio, 4× below healthy.
  - *Intermittent freeze:* the leaky accumulator catches patterns
    with no two consecutive frozen updates (alternating frozen/clean
    latches at the 5th update; tested).
  - *Slow freeze (window rule):* over the last 10 updates, if
    commanded displacement sums to ≥ 1.0 m while reported sums to
    < 25% of it, the update counts as frozen — at any speed. This
    covers total freeze down to ~0.1 m/s average commanded travel.
    At hover the commanded sum is ~0 and the rule cannot fire.
  - *Long-window aggregate (round-2 closure):* the same band test
    over the last 120 updates, keyed to commanded travel instead of
    speed: once ≥ 1.0 m has been commanded inside the window, the
    aggregate reported/commanded ratio is classified with the same
    bands (frozen < 0.25; partial < 0.70; severe ≤ 0.40). Total
    freeze is therefore caught at 0.05 m/s in ~21 updates and at
    0.02 m/s in ~51 updates — detection latency is bounded by 1.0 m
    of commanded travel, not by a speed floor — and slow partial /
    slow intermittent freezes (invisible to every per-update rule
    under the 0.30 floor) close the gate via the aggregate partial
    band **when their aggregate ratio is below 0.70**. That
    qualifier is load-bearing (round-2 N2): any sustained loss
    pattern whose aggregate reported/commanded ratio stays
    ≥ 0.70 — a sustained scale ratio of 0.71–0.85, a 25%-duty
    intermittent freeze at 0.20 m/s — is in the invisible band and
    does NOT close the gate; §6 states the measured consequences.
    When the freeze latch engages or releases, both windows
    clear: recovery is judged on post-latch behavior only, never on
    the windows' memory of the freeze itself.
  - *Estimator lag is not a freeze (round-2 N3, code; round-3
    N3-noise, code):* a pure pipeline delay makes the report track
    a DELAYED setpoint, `est(t) ≈ cmd(t−L)`, and at a motion onset
    its first L reports are the pre-onset hover reports —
    indistinguishable from a freeze by the instantaneous
    signature, and round 1/2 testing showed lag ≥ 2 latching
    UNTRUSTED (terminal in flight). The monitor keeps a
    **consistent-lag hypothesis**: armed at a motion onset (stream
    start, a rise from hover-level cmd ≤ 0.05 m/s, or a setpoint
    step ≥ 0.30 m/s — an onset/step re-opens it with a fresh
    candidate set and window age), it holds candidate lags
    1..**LAG_TOLERANCE = 5 updates** [design: the largest lag the
    red team demonstrated as a false positive; the stack's own
    command-path lags are sub-second, so 5 s at the 1 Hz rate is a
    generous ceiling, not a measurement of FAST-LIO2 latency].
    Each update grades every candidate's delayed setpoint against
    the report (round-3: the kill is graded, not binary — under
    the harness's healthy noise model, combined σ ≈ 0.036 m/s, a
    lagged hover report crosses the strict 0.06 edge on ~5% of
    updates, and the round-2 binary kill latched healthy lagging
    estimators in 12–66 of 200 Monte Carlo flights):
    **match** — ratio ≥ 0.70 against a moving delayed setpoint, or
    reported speed ≤ 0.06 m/s against a hover-level one (the strict
    edge is unchanged: ~2× the 0.03 m noise sigma, below the
    0.10–0.1125 m/s a ratio-0.125 freeze reports at cruise);
    **unresolved** — inside the noise band (≤ 0.09 m/s = 2.5σ
    against a hover-level delayed setpoint, or within 0.11 m/s
    = ~3σ below the ratio threshold against a moving one): the
    candidate survives unconfirmed, and for the hover-delayed case
    the freeze family does not fire on the update; unresolved
    candidates expire with the onset window; **far** — outside
    every candidate's band: the hypothesis dies and the update is
    judged normally, BUT a later report inside the window that
    strictly matches a delayed setpoint **re-arms** it — unless the
    match is vacuous (delayed setpoint == current cmd, so no-lag
    explains the report equally well; that guard is what keeps the
    N4 release rule below measured on schedule). Finally,
    **catch-up forgiveness**: if the hypothesis **strictly
    matched in the window AND the window holds at most one
    counted fault** (conjunctive gate, round-4 fix — see below),
    a far kill occurred in the window, and within 5 updates of
    the kill a report matches the CURRENT setpoint (the delayed
    motion has demonstrably arrived), the window's **own**
    freeze/partial contribution is forgiven — the aggregate sums
    otherwise hold the lag dip as a freeze signature for updates
    after the lag resolves. Forgiveness is **contribution-scoped**
    (round-4 C3): it restores the pre-window snapshot (evidence
    decayed as if the window's updates had been clean; aggregate
    deques keep their pre-window entries plus the catch-up entry)
    instead of wiping the global accumulators, so evidence from
    an independent earlier fault survives. A real freeze never
    catches up; a fault counted repeatedly in the window is
    **never forgiven, matched or not**. **[Corrected, round 4 —
    the previous sentence here claimed "a fault counted
    repeatedly in the window (a partial freeze at an onset: 4
    counted updates) is never forgiven," full stop; red-team
    round 4 (`REDTEAM4_NAV.md` §2c) falsified it: the gate was
    `matched OR faults ≤ 1`, so one strictly-matched onset update
    (est 0.0 — the canonical first report of a freeze at an
    onset) set `matched` for the whole window, the fault count
    was never consulted, and matched + severe-partial ×2–×9 was
    fully forgiven (6.85 m of error forgotten at ×9). The old
    claim was true only for faults whose first onset report is
    itself far. The gate is now conjunctive with bound 1 —
    justified by the legitimate case the gate exists for: a
    lag-1/2 onset spike produces at most ONE counted fault
    in-window (the spike itself; the lagged reports around it
    are matched or unresolved, not counted), and the lag-5
    aggregate-dip case likewise holds exactly one.]** Measured
    after the round-4 fix (seeded, locked by
    `test_lag_noise_monte_carlo` /
    `test_lag_noise_freeze_escape_audit` /
    `test_catchup_forgiveness_conjunctive_gate`): onset lag
    2/3/5, lag-5 on a speed ramp, hover-onset lag, and step lag
    all stay HEALTHY on every update noiselessly (previously
    3–30 UNTRUSTED); with the harness noise model ON, a healthy
    lagged estimator (lags 0–5) produces **zero UNTRUSTED
    latches in 200 Monte Carlo flights per lag on each of the
    three-onset profile and a single-onset profile** — on the
    builder seed base (31000) and on the red team's independent
    base (770000). The round-3 "zero non-HEALTHY" result does
    **not** survive the conjunctive gate at lag 1/2, and is
    corrected here: spike-only flights whose window never
    strictly matched (their far spike is the window's only
    lagged report and their catch-up match is vacuous at
    constant cmd) are no longer forgiven, and their retained
    spike evidence costs a transient DEGRADED — DEGRADED-only
    flights measure three-onset lag 1: **3/200**, single-onset
    lag 1: **3/200**, single-onset lag 2: **1/200** (builder
    base; red-team base: three-onset lag 1: 2/200, zero
    elsewhere), zero at every other lag including lag 4. The
    deviation errs toward longer distrust (safe direction),
    never toward a latch. **The noise envelope's independence
    assumption is load-bearing (round-4 C1):** every zero-latch
    number in this bullet is measured under the harness noise
    model, whose per-update noise draws are **iid** (combined
    σ ≈ 0.036 m/s). Real estimator noise is not iid: with
    AR(1)-correlated noise at ρ = 0.8 and the **same nominal
    σ**, healthy lag 4–5 flights latch UNTRUSTED in **2/200**
    per profile (red-team round-4 measurement, seed base
    770000; lag 4 and lag 5, three-onset and single-onset
    lag 4), and at σ × 1.5 (iid) latches appear at lag ≥ 3
    (≤ 4/200). The envelope claim is therefore scoped to
    iid noise at nominal σ, beside the σ it was already
    scoped to. **Limits, stated plainly:** (i) lag **beyond 5
    updates is treated as a freeze** — a 10-update lag latches
    UNTRUSTED at update 7 after the onset (5 tolerance + evidence
    dwell; stream update 13 on a hover-prefixed stream, ≤ 13
    under noise); (ii) a lag that *begins* mid-motion with no
    onset/step to arm the hypothesis is treated as a freeze;
    (iii) the price is onset ambiguity in the other direction —
    a true freeze that begins **exactly at an onset** latches by
    update 7 after the onset whatever it reports (under noise,
    +1 update — update 8 — when noise lifts a frozen report into
    the partial band, measured worst case over 200 seeds on the
    scaled-0.125 shape at cmd 0.4); a freeze reporting the
    measured 0.125 signature at cruise (0.1125 m/s, a far
    mismatch) latches at update 2 deterministically and at
    updates 2–7 under noise (median 2), and any freeze beginning
    during established motion still latches at its 2nd frozen
    update; (iv) frozen reports spanning only onset updates 1–6
    followed by clean reports are forgiven when the delayed
    motion arrives (indistinguishable from a lag marginally over
    tolerance that catches up) — a 7th frozen report latches at
    update 7 before any catch-up can arrive. **This boundary is
    per-onset and repeatable (round-4 C2, re-measured under the
    conjunctive gate — the shape holds exactly one counted
    fault, so the new gate does not move it): with a ≥ 5-update
    hover gap between onsets, the burst repeats with no
    cumulative accounting — measured 3 bursts = 18.00 m and
    10 bursts = 60.00 m of forgotten error, HEALTHY on every
    update, locked by `test_onset_burst_boundary_disclosed`.
    A second burst in the SAME cruise (no new onset) is not
    forgiven and latches at its 2nd frozen update.** (v) an alternating
    freeze beginning exactly at an onset now latches at update
    11, not 7 — the first catch-up cycle is consumed by
    forgiveness (mid-motion intermittent freezes are unchanged).
  - *Partial freeze / scale error:* ratio in [0.25, 0.70) sustained
    (leaky evidence ≥ 2) → DEGRADED, gate closed. A severe partial
    (ratio ≤ 0.40, instantaneous or long-window aggregate) sustained
    10 updates → latched UNTRUSTED (at 1 m/s cmd that is ≥ 0.6 m/s
    of pose-error accrual).
- **Freeze latch [design]:** UNTRUSTED-by-freeze does not recover on
  quiet-at-hover updates (there both speeds are ~0 and every check
  passes vacuously — unlatched, the loop limit-cycles: release →
  fly blind → re-freeze, exactly what round-0 did on the pre-latch
  tree). Monitor-level unlatch requires 3 consecutive
  **counted-clean** updates **under commanded motion**, where an
  update counts as clean only if ALL of the following hold (the
  actual release rule, round-2 N4): no freeze-family band fires on
  it — including the long-window aggregate, whose ratio must first
  climb back to ≥ 0.70 as post-fault clean travel washes the
  fault's travel out of the aggregate; its score is 0 — which
  requires the **uncapped** partial-evidence accumulator
  (+1.0 per partial update, −0.5 per non-partial update, floor 0)
  to have decayed below 2.0; and no pose jump. Because the
  evidence is uncapped and the aggregate must wash out, **release
  time grows with fault duration — it is not a flat 3 updates**.
  Measured at cmd 1.0 **with clean reports at ratio 1.0** (report
  == setpoint — the measurement condition for every constant in
  this bullet; round-3 caveat, locked by
  `test_long_freeze_release_latency_measured`): a 2-update freeze
  releases after 3 clean updates (HEALTHY at 6); a 10-update total
  freeze mid-stream needs **49** clean updates to DEGRADED and
  **52** to HEALTHY (the red team's probe: 10-update freeze + 35
  perfect updates is still UNTRUSTED — consistent with 49); a
  30-update freeze needs 176/179; a mere 4-update partial
  (DEGRADED, no latch) needs **13** clean updates to HEALTHY, not
  the 3 + 3 the hysteresis line alone suggests. **Under the
  stack's own healthy report model (clean reports at 0.9 ×
  setpoint) release is materially slower** — the long-window
  aggregate washes out at the report ratio: freeze-10 needs
  **74** clean updates to DEGRADED and **77** to HEALTHY,
  freeze-30 **258/261**, partial-4 only **7** (red-team round-3
  probes). Every deviation errs toward longer distrust (safe
  direction); the formula is exact, the constants are
  condition-dependent. The accumulators
  are deliberately not capped: the uncapped tail is what makes
  intermittent faults accumulate toward the latch instead of being
  forgiven at a cap, so the correction is to this documentation,
  not to the behavior. **At integration level this recovery is BY
  DESIGN unreachable in flight** — the policy answers UNTRUSTED
  with HOVER, under which commanded motion is zero (§6).
- **Pose jump (hard):** single-update jump > 2.0·dt + 0.5 m → instant
  UNTRUSTED [calculated: V_MAX 2.0 m/s (vehicle_v3.py) × dt + margin].
- **Residual:** > 0.10 m → +1 score [design: ~3× the eval's 0.03 m
  range noise]; > 0.30 m → +2 [design: 10×]. **Single-channel ceiling
  (intended, round-1 probe E): residual alone caps at DEGRADED.**
- **Map age:** ≥ 1.0 s → +1 [measured: D5 FIX 4 anchor]; ≥ 3.0 s →
  +2 [design]. **Staleness alone also caps at DEGRADED** — a
  stale-but-consistent estimate is D5's hover case, not proof of
  estimator failure. UNTRUSTED via score requires ≥ 3 sustained
  2 updates (e.g. residual +2 with map-age +1; behaviorally tested).
- **Feature starvation:** < 10 features sustained 2 updates → +1,
  DEGRADED-only by construction [measured anchor, corrected round 1:
  calibration corpus clear-air means are 151.1 hits (indoorcd_117,
  min 94) and 48.3 (indoorcd_122) per 360-ray scan; a pose can
  transiently see ~0, hence sustained, hence never sufficient for
  UNTRUSTED].
- **Recovery hysteresis [design]:** one severity level per 3
  consecutive clean updates (freeze latch excepted, above). Note
  the same evidence-decay caveat applies on the DEGRADED path: a
  "clean" update for hysteresis purposes still requires score 0,
  so residual partial evidence delays the first counted-clean
  update (the 4-update partial → 13 clean updates measurement
  above is exactly this).

## 3. What the sim's estimator feed is (honesty box)

The mission loop flies on true pose (no estimator runs in-loop — the
same situation as before this build). With `cfg["nav_health"]` set, a
report generator plays the estimator-hardware role: nominal reports =
flown displacement + N(0, 0.03 m) noise (the FAST-LIO2 eval's
Mid-360-class noise; the same truth+noise sensor-model pattern as the
IMU-prior LEGO's synthetic IMU). **The monitor never sees truth** —
it consumes the report dict; truth touches only the sensor model that
manufactures the report. **The cmd channel is NOT truth** (round-1
F3): it is the velocity setpoint the loop last **issued to the PID**,
recorded post-supervision at every command site (cruise setpoint on
servo steps, 0 on braking/hover steps, descent rate on land steps),
one-step-lagged as a real setpoint stream would be. The distinction
matters and was measured the hard way: an earlier revision fed the
pre-supervision cruise setpoint, and on CBF-braked steps the honest
estimator's smaller achieved motion read as a freeze — indoorcd_071
nav-ON stood down (UNTRUSTED, 120 scans) on a nominal mission. The
issued setpoint is the only defensible command channel. In calibration, cmd is the scripted setpoint and an explicit
achieved-motion model (0.9× setpoint + noise [design]) sits between
setpoint and report, so healthy ratios run ~0.9, not 1.0 by
construction. `residual_m` is None in-loop (MISSING: no scan matcher
runs in the loop — listed, not invented); the residual rules are
exercised in calibration and unit tests. `n_features` in-loop is the
raw scan hit count — a proxy with a measured caveat (§6).

## 4. Offline calibration (`run_nav_health_calibration.py`)

Goal-ward 1 m/s setpoint traverses in indoorcd_117/122 using the
stack's own obscurant model (feature counts from real scans); reports
per §3; offline truth = report error vs achieved motion (never fed to
the monitor). 576 rows, 12 conditions, executed commit in every row.

| Condition | Monitor result | Offline truth |
|---|---|---|
| clear, healthy report | 48/48 HEALTHY | 0/48 unhealthy |
| heavy fog, healthy report | 48/48 HEALTHY | 0/48 unhealthy |
| whiteout FREEZE (ratio 0.125) | UNTRUSTED from 2nd update, 46/48 | 48/48 unhealthy |
| freeze ratios 0.18 / 0.26 (off-point sweep) | UNTRUSTED from 2nd update, 46/48 each | 48/48 unhealthy each |
| freeze ratio 0.05 (off-point sweep) | UNTRUSTED from 7th update, 36/48 | 48/48 unhealthy |
| partial 0.30 / 0.40 | DEGRADED at ~2, UNTRUSTED at update 10 (severe sustain), 30/48 UNTRUSTED | 48/48 unhealthy |
| partial 0.55 | DEGRADED 46/48, never UNTRUSTED | 46/48 unhealthy — gate closed on all flagged steps |
| weave report (σ ×10 error) | UNTRUSTED 44/48 (combined channels: residual + partial evidence) | 48/48 unhealthy |
| residual mid-band (0.20 m), healthy report | DEGRADED 48/48 | 0/48 unhealthy (report accurate; gate closed conservatively) |
| stale map (map_age 2.0 s), healthy report | DEGRADED 48/48 | 0/48 unhealthy (same) |

Separation read: across 96 healthy steps, **zero** false flags — now
earned against a setpoint cmd with an achieved-motion lag model, not
a by-construction ratio. Freeze detection lag: 1 update after onset
for every freeze reporting ≥ 0.10 m/s. The ratio-0.05 row is the one
measured exception, and it is the N3 lag tolerance's disclosed
price (§2/§6): its reported speed is 0.045 m/s — under the 0.06 m/s
quiet threshold — and the condition starts at stream start, so its
first 5 updates are indistinguishable from a lagging estimator's
hover reports and are excused; it latches at the 7th update and
still ends UNTRUSTED on 36/48 steps (was 46/48 before the lag fix —
the only calibration number the N3 fix moves). Weave changed vs
round 0 (was DEGRADED-only): with the partial band added, weave's
low report draws accumulate partial evidence on top of the residual
channel and the combination crosses UNTRUSTED. Reported as measured,
not as originally designed.

## 5. Mission validation (`run_nav_health_mission_validation.py`)

13 missions, numpy backend, noise 0.15, fixes 1–4 + escalation ON in
both arms, executed commit `4e73601` (verified at begin and write
time by `exec_provenance`; the CSV's commit column is that commit).
**Result: PASS — 0 collisions, 0 errors.** Every substantive column
is identical to the previous (`00de3af`) issue — the round-4
forgiveness fix moves no mission number (0 non-wall-time diffs
across all 13 rows). The calibration CSV was likewise re-issued
(executed commit `c06984f`); against its previous issue there are
**zero** non-commit diffs — the round-4 fix moves no calibration
number either, and the healthy arms still read 96/96
zero-false-flag.

| Arm | n | Result |
|---|---|---|
| Paired clear (004/071/128/185) | 4 pairs | outcomes identical OFF vs ON (4/4 success); nominal feed never UNTRUSTED |
| Paired heavy fog (117/122) | 2 pairs | outcomes identical (2/2 max_steps — the shipped heavy-fog stand-down); nav feed ends HEALTHY, never UNTRUSTED |
| Scripted freeze (004, clear air, freeze from step 5) | 1 | health UNTRUSTED for 194 outer steps; ladder reached the land row with RTH gated 16 times; below-clear guard vetoed blind descent → **max_steps** safe stand-down, never success, never collision, never RTH |

Paired clear missions end at nav DEGRADED — the map-age contributor
on the jittered clear set, investigated in §6 (F5), not a defect.
Outcomes are identical between arms; wall times and nav counters
differ (round-0's "bit-identical" phrasing was an overstatement and
is corrected here).

Process note: the first round-1 re-run of this suite was DISCARDED
by the new provenance guard — a mid-run edit (this report) dirtied
the tree and `exec_provenance.finish` refused to issue the CSV.
That discarded run is also how the pre-supervision-setpoint defect
(§3) was caught: its indoorcd_071 nav-ON arm stood down UNTRUSTED.
The suite above is the re-run on the fixed, committed tree.

## 6. Known limits (read these before trusting it)

- **UNTRUSTED is terminal-for-mission BY DESIGN (round-1 F6).** The
  monitor-level unlatch (§2) exists for post-landing / ground
  re-evaluation; in flight the policy's HOVER response makes the
  release condition (clean updates under commanded motion)
  unreachable, so any freeze that latches — including a transient
  2-update stall — ends the mission in a stand-down. There is no
  in-flight recovery path and §2's unlatch must not be read as one.
  Consequence: the integrated drone can hover-forever on this path;
  in reality that state is bounded by **battery** — the battery
  health model is a parallel roadmap item (not wired here); when it
  lands, its trigger must route through this gate (a DEGRADED-closed
  gate converts it to land-in-place, the safe direction).
- **DEGRADED is an unbounded gate closure BY DESIGN (round-1 F5).**
  DEGRADED has no ladder row and no timeout: RTH stays denied for as
  long as the cause persists, while local flight (S2/D5, which do
  not navigate on the global pose the way RTH does) continues. It
  ends only by the cause clearing (recovery hysteresis) or worsening
  into UNTRUSTED. The paired clear missions in §5 END in DEGRADED —
  investigated, not a calibration defect: those arms carry periodic
  2 s jitter freezes (inherited from the escalation-validation set),
  the map-age contributor (D5 anchor) fires correctly on the stale
  windows (12 scans in the diagnostic re-run of indoorcd_004 s0), and
  the missions end inside the recovery-hysteresis window.
- **Ultra-slow freeze floor (round-1 F2c, residual after the
  round-2 long-window closure).** Detection is keyed to commanded
  travel: the freeze family fires once ≥ 1.0 m has been commanded
  inside a 120-update window. Sustained motion slower than
  ~0.008 m/s average (120 s of it commands < 1.0 m) is below that
  floor and cannot fire it. The bound is principled, not just
  empirical: undetected pose error in that band is itself < 1.0 m
  **per 120-update window — not cumulatively** (round-2 N2: a
  sustained total freeze at 0.008 m/s reads HEALTHY on 600/600
  updates while accruing 4.8 m of error over five windows; the
  per-window bound does not sum to a total bound). D5/map-age plus
  the pose-jump rule still apply. Unit tests lock both sides:
  total freeze at 0.05 and 0.02 m/s latches UNTRUSTED; healthy
  flight at those speeds stays HEALTHY (no false fire).
- **Aggregate-loss invisible band (round-2 N2, disclosure — the
  measured price of the 0.70 partial threshold).** Any sustained
  pattern whose aggregate reported/commanded ratio stays ≥ 0.70
  reads HEALTHY with the gate open while pose error accrues
  without bound. Measured shapes (red-team round 2, re-verified
  after the N3 fix — numbers unchanged): sustained scale ratio
  0.71 / 0.75 / 0.80 / 0.85 at cmd 1.0 m/s — HEALTHY on all 60
  updates, error accruing 0.15–0.29 m/s; 25%-duty intermittent
  freeze (7 frozen / 21 clean) at cmd 0.20 m/s with a clean
  prefix — HEALTHY on 310/310 updates, 14 m of commanded motion
  lost (the 4/12-duty variant ends HEALTHY with only transient
  DEGRADED, 20 m lost over 320 updates); total freeze in 0.99 m
  bursts separated by 121 hover updates (each burst is evicted
  from the sliding window before reaching the 1.0 m travel
  floor) — HEALTHY on 880/880 updates, 3.96 m of error.
  **Measurement condition (round-3 caveat):** every shape in
  this bullet was measured with clean reports at ratio 1.0;
  under the stack's own healthy report model (achieved = 0.9 ×
  setpoint) the same F7/C21 shape's aggregate is 0.675 < 0.70
  and it closes the gate instead (DEGRADED 193/310, never
  UNTRUSTED) — the ratio-1.0 numbers overstate the evasion for
  the stack's own report model (safe direction), and the band
  itself is real because ratio-1.0 reports occur whenever
  achieved ≈ setpoint. These
  live exactly on the published thresholds; closing the ≥ 0.70
  band would mean raising PARTIAL_RATIO toward the healthy
  achieved/setpoint ratio (~0.9), which the calibration's healthy
  arms do not leave room for without false flags — so the band is
  disclosed, not closed. This is the same "< 1.0 m" qualifier as
  the ultra-slow floor above: the error bound is **per
  120-update window**, never cumulative.
- **Estimator lag (round-2 N3, fixed in code; round-3 N3-noise,
  fixed in code; round-4 forgiveness gate, fixed in code —
  limits remain).** Pure lag at an onset no
  longer latches the freeze UNTRUSTED (§2's consistent-lag
  hypothesis, tolerance 5 updates), including under the harness's
  healthy report noise — the Monte Carlo (§2) measures zero
  false latches at lags 0–5 where the round-2 binary kill
  latched up to 66/200 flights; **[corrected, round 4]** the
  round-3 "zero transient DEGRADED" no longer holds under the
  conjunctive forgiveness gate: DEGRADED-only flights measure
  3/200 (three-onset lag 1), 3/200 (single-onset lag 1), 1/200
  (single-onset lag 2) on the builder seed base, zero elsewhere.
  **The catch-up boundary is per-onset and cumulative-blind
  (round-4 C2, disclosed residual — NOT closed):** frozen
  reports confined to onset updates 1–6 hold exactly one counted
  fault and are forgiven at each onset; repeated across onsets
  separated by ≥ 5 hover updates the forgotten error accumulates
  without bound while the monitor reads HEALTHY on every update
  — measured 3 bursts = 18.00 m, 10 bursts = 60.00 m (§2(iv),
  locked by test). There is no cumulative forgiven-error
  accounting anywhere in the monitor; closing this would mean
  either refusing forgiveness for the (indistinguishable)
  marginal-lag case or adding a cumulative budget — a design
  call for the PI, not taken here. **Forgiveness scope
  (round-4 C3, fixed):** forgiveness restores the pre-window
  snapshot, it does not clear the global accumulators — evidence
  from an independent fault before the window survives (§7
  round-4 record: snapshot partial evidence 6.0 → 3.5 retained
  at the forgiveness update, state DEGRADED there, where the
  pre-fix code zeroed it and returned HEALTHY). **Noise
  independence (round-4 C1):** the lag envelope's zero-latch
  claim assumes **iid** noise at combined σ ≈ 0.036 m/s; AR(1)
  ρ = 0.8 correlated noise at the same σ latches healthy
  lag 4–5 flights in 2/200 (§2). The remaining interactions: lag **beyond 5 updates is treated
  as a freeze** (latched, terminal in flight like any freeze); a
  lag beginning mid-motion with no onset/step is treated as a
  freeze; a true freeze beginning **exactly at an onset** is
  detected at update 7 after the onset, not update 2 (update 8
  in the measured worst noisy case) — named instance of that
  price: a sub-0.06 m/s freeze beginning at a **descent onset**
  costs ~3.5 m of blind descent (7 descent updates at the
  0.5 m/s LAND descent rate) before it latches, and since the
  round-3 noise band a freeze reporting the 0.125 signature
  *scaled to the descent setpoint* (0.0625 m/s at cmd 0.5 — inside
  the band, indistinguishable from lagged hover noise) pays the
  same 3.5 m; frozen reports confined to onset updates 1–6 that
  then recover are forgiven (§2, catch-up boundary), and an
  alternating freeze beginning exactly at an onset latches at
  update 11 instead of 7. This is the monitor's face of the
  roadmap's known "compute latency unmodeled" gap: the sim's
  estimator feed has no pipeline-delay model, so the tolerance
  is a design ceiling validated against synthetic lag streams,
  not against measured FAST-LIO2 latency (no telemetry —
  MISSING, as §6 notes below).
- **Corridor case — NOT covered.** The corridor degeneracy
  (along-axis unobservable, ATE 0.88–2.04 m, broaden eval) produces
  a smooth, self-consistent, wrong estimate: reported motion tracks
  commanded motion, residuals stay low, features stay plentiful.
  Catching it needs a degeneracy signal from inside the estimator
  (scan-matcher Hessian/observability eigenvalues) — FAST-LIO2 does
  not publish one in the eval harness. MISSING, not papered over.
- **Slow consistent drift — NOT covered.** A bias growing slowly
  under the jump/freeze/residual thresholds (cf. the pre-turn yaw
  drift in rooms 157/168, mechanism unknown) is invisible until it
  crosses a threshold. The monitor catches failures of
  *consistency*, not failures of *accuracy* that stay consistent.
  The latch's unlatch is magnitude-only for the same reason:
  speed-plausible updates from a wrong pose release it (to
  DEGRADED) once they satisfy the counted-clean rule (§2, N4).
- **Feature-count proxy in-loop is weak in fog.** Measured: this
  sim's obscurant model makes whiteout *inflate* raw hit counts
  (backscatter phantoms: ~250 hits vs ~61–151 clear). A real
  effective-feature count should fall instead — but in THIS harness
  the feature input cannot see whiteout. That is why it is
  DEGRADED-only and the freeze family (which needs no feature data)
  carries the whiteout case.
- **D5 overlap in real whiteout:** perception triggers (FIX 3/4 +
  escalation) fire alongside nav-health there; nav-health is not the
  sole coverage and this build does not claim it is. Its
  *independent* coverage is estimator faults under healthy
  perception (scripted-failure arm, §5).
- Thresholds marked [design] are engineering choices, not
  measurements; the freeze ratio and map-age anchors are the
  measured ones. The calibration's residual generator and thresholds
  share a 0.03 m base constant (round-1 F4) — the mid-band condition
  (0.20 m) shows graded response, but the residual channel's
  absolute calibration remains design-grade, not measured.
- Simulation only. The report generator is a model of FAST-LIO2's
  published behavior, not recorded FAST-LIO2 telemetry (no telemetry
  logs exist in this repo — MISSING).

## 7. Provenance correction record (round-1 F1)

Round 0's CSVs recorded `git rev-parse HEAD` captured at script
launch, in a workflow where code landed mid-run / pre-commit:
the mission CSV named `f994011` (pre-latch tree — re-running the
scripted arm there gives SUCCESS with limit-cycling, the opposite of
the reported stand-down) and the calibration CSV named `ac2bac6`
(a commit where `nav_health.py` did not exist). The numbers were
true of the executed working trees; the artifacts' origins were
false, and a CSV that cannot be trusted to name its code is a
defect in the artifact, not a footnote.

System fix (drift doctrine — change the system, not the number):
`exec_provenance.py` — both harnesses now **abort on a dirty tree**
and **re-verify HEAD at write time**; the commit column is the
executed commit **verified at begin and at write time** (the
guard's exact scope, including what it cannot see, is stated in
the round-2 fix record below and in the module docstring) or the
run produces no CSV at all. Both
CSVs in this revision are re-issued under that guard, and each row
carries a `provenance_note` stating the correction. Any future
harness in this stack should import the same guard.

**Round-2 hardening + proof (this pass).** (i) The guard's dirty
check now compares full repo-relative paths (it previously matched
basenames, so a dirty file elsewhere sharing an output's basename
could have slipped through) and handles renames;
`verify_csv_commit()` adds a post-hoc self-check (one full hash,
present in-repo, ancestor of HEAD), exercised by
`test_exec_provenance.py` — 6 tests that watch the guard abort on a
dirty tree, a mid-run dirtying, a HEAD that moves mid-run (the exact
round-1 failure), and a same-basename impostor file. The guard also
bit for real during this pass: a log file dropped inside a repro
clone aborted a launch with PROVENANCE ABORT, as designed.
(ii) Checkout-and-reproduce: the round-1 calibration CSV was
reproduced **bit-identically** by checking out its recorded commit
`3db0a04` in a clean clone and re-running; the round-1 mission CSV
was reproduced with **0 non-wall-time diffs across all 13 rows**
(including the commit and note columns) from its recorded commit
`4edaef7`. The round-2 re-issues carry the same proof: calibration
at `90659a5` bit-identical on checkout-and-rerun; mission at
`6173c1f` reproduced on checkout-and-rerun (verdict PASS, scripted
UNTRUSTED 194 / gated 16, 0 collisions).

**Red-team round-2 fix record (PASS WITH CAVEATS → this pass).**
The round-2 red team (`REDTEAM2_NAV.md`, worktree `navrt2`)
verified the round-1 fixes and raised five caveats; each is
resolved here as follows.
- **N1a (code):** the provenance guard resolved a rename to its
  destination only, so `git mv code.py out.csv` mid-run slipped
  `finish()` — the destination was an allowed output and the
  tracked source's disappearance went unchecked. `_dirty_paths`
  now checks BOTH rename endpoints: a rename is allowed only if
  its source is itself an allowed output (and likewise its
  destination). Regression test
  `test_rename_onto_output_aborts` reproduces the red team's A8
  attack at begin and at finish (plus an allowed-output control);
  re-running the red team's 17 provenance attacks against the
  fixed guard, A8 now aborts like every other dirty state.
- **N1b (disclosure):** the guard's scope limits are now stated in
  the `exec_provenance` docstring and here: it certifies the
  **tracked, non-ignored** tree at **two instants** (begin and
  write time). Git-ignored files are invisible to it (in this
  repo the ignored artifacts under the harness tree are
  `__pycache__/*.pyc` and an untracked kernel .npz neither harness
  loads — hypothetical here, structural in general); a
  commit-then-reset excursion between the two instants and edits
  hidden by `--assume-unchanged` / skip-worktree are undetectable
  by any git-status-based guard (red-team attacks A6/A11/A12
  still pass the guard, now as documented limits, not surprises).
  "Verified executed commit" means verified at begin and at
  write time, under those limits.
- **N2 (disclosure; claims corrected to measured coverage, no
  behavior change):** §2's aggregate-band sentence now carries
  its ≥ 0.70 qualifier, and §6 discloses the measured invisible
  band with its shapes (sustained ratio 0.71–0.85; 25%-duty
  intermittent at 0.20 m/s, 310/310 HEALTHY, 14 m lost; 0.99 m
  hover-evicted bursts, 3.96 m; sustained 0.008 m/s freeze,
  4.8 m over 600 updates). The "< 1.0 m error" bound is now
  qualified **per 120-update window, not cumulative** everywhere
  it is claimed, including the `nav_health.py` long-window
  comment that previously dropped the qualifier. The red team's
  probes were re-run after this pass: every N2 number is
  unchanged (the F4/C12 variant's transient-DEGRADED count moved
  69 → 45, final state unchanged — better, not worse).
- **N3 (code):** the estimator-lag false positive is fixed by the
  consistent-lag hypothesis (§2): onset lag 2/3/5 and the lag-5
  speed ramp now stay HEALTHY on every update (red-team probes
  previously: UNTRUSTED for 3 / 4 / 18 and 30 updates
  respectively), while a true freeze during established motion
  still latches at the 2nd frozen update and a freeze at an onset
  reporting the measured 0.125 signature still latches at update
  2. The chosen tolerance (5 updates, consistent single lag,
  armed only at onsets/steps) and its limits (lag beyond
  tolerance or beginning mid-motion = treated as a freeze; a
  sub-0.06 m/s freeze beginning exactly at an onset is detected
  at update 7) are disclosed in §2/§6 and locked by
  `test_estimator_lag_is_not_a_freeze` /
  `test_lag_beyond_tolerance_is_a_freeze`.
- **N4 (documentation corrected to measured behavior; no behavior
  change):** the stickiness text in §1/§2, §6, and the
  `nav_health.py` consumer docstring now states the actual
  release rule (counted-clean predicate + uncapped evidence decay
  at 0.5/update + aggregate washout) with measured latencies
  (freeze-2: 3 clean to DEGRADED; freeze-10 mid-stream: 49/52;
  freeze-30: 176/179; partial-4: 13 to HEALTHY), locked by
  `test_long_freeze_release_latency_measured`. The docstring's
  "treat a True→False transition as final for that flight leg" is
  now scoped to the freeze latch only; DEGRADED recovers and
  `pose_trustworthy` returns True. The accumulators were NOT
  capped: the uncapped decay is load-bearing for intermittent
  faults, so the docs moved to the behavior, not the reverse.
- **Nit (code):** the dead `_nh_prev_pose is None` branch at the
  mission loop's cmd channel (red-team §3) is removed; the value
  is identical on every step (`_issued_cmd` initializes to 0.0).
- **Re-issue + reproduction (this pass):** both CSVs re-issued
  under the guard at the new executed commits — calibration at
  `c07d2e9` (healthy arms still 96/96 zero-false-flag; the only
  diff vs the previous issue is the 10 freeze_r0.05 rows in §4)
  and mission validation at `afdb499` (PASS, 0 collisions,
  scripted UNTRUSTED 194 / gated 16; every substantive column
  identical to the previous issue). Checkout-and-rerun proof at
  those commits: calibration reproduced **bit-identically**
  (SHA-256 equal) from a clean clone at `c07d2e9`; mission
  reproduced with **0 non-wall-time diffs across all 13 rows**
  from a clean clone at `afdb499`.

**Red-team round-3 fix record (FAIL → this pass).** The round-3
red team (`REDTEAM3_NAV.md`, worktree `navrt3`) confirmed
N1a/N1b/N2/N4 and returned FAIL on one blocking defect:
- **N3-noise (code):** the round-2 lag hypothesis died
  permanently on a single report above 0.06 m/s against a
  hover-level delayed setpoint and never re-armed; under the
  calibration harness's healthy noise model (achieved =
  0.9·cmd + N(0, 0.02), report = achieved + N(0, 0.03), combined
  σ ≈ 0.036) a healthy lagging estimator therefore latched
  UNTRUSTED (terminal in flight) in 12/200 (lag 2), 40/200
  (lag 3), 66/200 (lag 5) of the red team's three-onset Monte
  Carlo flights, 5.5%/7%/21% per single onset, with a lag-0
  control at 0/200 — the product healthy × lag × noise had never
  been tested (lag tests were noiseless; the calibration harness
  has noise but no lag). The fix (§2) grades the lag kill
  (match / unresolved noise band / far), re-arms the hypothesis
  inside the onset window on a non-vacuous strict match, and
  forgives onset-window evidence on a demonstrated catch-up.
  Locked by two new seeded tests: `test_lag_noise_monte_carlo`
  (200 seeds × {three-onset, single-onset} × lags 0/1/2/3/5:
  **0 UNTRUSTED, 0 non-HEALTHY updates**, transient DEGRADED
  measured at zero on both protocols — an extended 2000-seed
  sweep on a different seed base also measures 0/0, after the
  aggregate-window residue it initially exposed, 15/2000 lag-5
  latches, was traced to the windows' lag-dip memory and closed
  by catch-up forgiveness) and
  `test_lag_noise_freeze_escape_audit` (deterministic + noisy
  escape shapes: onset freezes latch by update 7, +1 under
  partial-band noise; the 0.125 signature at cruise latches at
  2 deterministically, 2–7 under noise, median 2; descent-onset
  freezes by stream update 12; step/mid-motion freezes at the
  2nd frozen update; noisy lag-10 by stream update 13; the
  alternating-from-onset price moved 7 → 11 and the
  recover-after-6-onset-updates boundary are disclosed in
  §2/§6).
- **N2/N4 measurement conditions (disclosure, round-3 caveats):**
  §2's release constants and §6's aggregate-band shapes are now
  stated as measured with clean reports at ratio 1.0, with the
  0.9-model numbers beside them (freeze-10 release 74/77 vs
  49/52; F7/C21 closes the gate at 0.9) — both deviations err
  toward longer distrust / less evasion for the stack's own
  report model.
- **Descent-onset price (disclosure, round-3 caveat):** §6 now
  names the descent instance of the onset price — a sub-0.06
  m/s freeze at a descent onset costs ~3.5 m of blind descent
  before latching — and records that the 0.0625 m/s
  scaled-signature descent freeze moved from stream update 7
  to stream update 12 (the same 3.5 m) under the noise band.
- **No other behavior moved:** `pose_trustworthy` semantics are
  unchanged (True iff HEALTHY, verified on every update of the
  red team's probes and the new tests); both CSVs re-issued
  under the guard at the new executed commits — calibration at
  `cb75e9b` with **zero non-commit diffs** vs the previous issue
  (healthy arms still 96/96) and mission validation at
  `00de3af` (PASS, 0 collisions, scripted UNTRUSTED 194 / gated
  16; **0 non-wall-time diffs** across all 13 rows).
  Checkout-and-rerun proof: calibration reproduced
  **bit-identically** from a clean clone at `cb75e9b` (SHA-256
  `d6e8331668518e3e9d6bfc9602c6526c7e9834f58ce138fb30f4929056048e83`);
  mission reproduced with **0 non-wall-time diffs across all
  13 rows** from a clean clone at `00de3af` (verdict PASS,
  scripted UNTRUSTED 194 / gated 16, 0 collisions).

**Red-team round-4 fix record (FAIL → this pass).** The round-4
red team (`REDTEAM4_NAV.md`, worktree `navrt4`) confirmed the
round-3 N3-noise fix on an independent seed base (0/200 at every
lag 0–5, both profiles) and the graded band / re-arm / vacuity
results, and returned FAIL on one mechanism — catch-up
forgiveness — with three caveats. Resolved here:
- **Forgiveness gate (code, the blocking item):** the gate was
  `window_matched OR counted_faults ≤ 1`; one strictly-matched
  update (est 0.0 on the onset update — also the canonical first
  report of a freeze at an onset) set `matched` for the whole
  window and the fault count was never consulted, so matched +
  severe-partial 0.35 ×2–×9 was fully forgiven by one catch-up
  report (no latch, both aggregate windows wiped, back to
  HEALTHY, 2.30 m → 6.85 m of error forgotten; ×10 stopped only
  because the severe-sustain latch fired first). The gate is
  now **conjunctive**: `window_matched AND counted_faults ≤
  LAG_FORGIVENESS_FAULT_BOUND = 1`. Bound justification: the
  legitimate case the gate exists for — a lag-1/2 onset spike —
  produces at most one counted fault in-window (the spike; the
  surrounding lagged reports are matched/unresolved, not
  counted), and the lag-5 aggregate-dip case holds exactly one
  counted fault beside its matched reports; a second counted
  fault is a repeated fault by the monitor's own bands.
  Re-measured under the new gate (RT probe shapes, 15-clean
  tails unless noted): matched + sev ×1 forgiven at update 3,
  all HEALTHY (the bound case carries); **×2–×9: zero
  forgiveness events**, gate closed (not HEALTHY) on the
  catch-up update itself; ×2 ends HEALTHY only after 12
  DEGRADED updates of normal leaky decay; ×3–×8 end DEGRADED
  (×3 after 17 DEGRADED updates; with a 20-clean tail ×3 ends
  HEALTHY after 21, ×4–×8 still DEGRADED); **×9 now latches
  UNTRUSTED at update 11** (the retained long-window aggregate
  completes the severe sustain on the catch-up update) and
  ×10 latches at 11 as before. The frozen-band sibling
  (matched + frozen 0.20 ×2–×4, forgiven pre-fix at ×2/×3)
  now has zero forgiveness and its latch stands. The
  no-matched-prefix partial family (×1–×6) is unchanged in
  outcome (×1's spurious forgiveness event is gone; it never
  needed it). RT M3/M4 shapes: every first-UNTRUSTED, count,
  and final state identical to the pre-fix run (only spurious
  forgiveness-event counts drop). Locked by
  `test_catchup_forgiveness_conjunctive_gate`.
- **Monte Carlo re-run (round-3 protocol, after the gate
  change):** healthy lag 0–5 (lag 4 now included), harness
  noise, 200 seeds × both profiles — **0 UNTRUSTED latches at
  every lag on both seed bases** (builder 31000, red-team
  770000). DEGRADED-only flights, reported honestly (the
  round-3 zero-non-HEALTHY claim is corrected in §2/§6):
  builder base — three-onset lag 1: 3/200, single-onset
  lag 1: 3/200, single-onset lag 2: 1/200, zero elsewhere;
  red-team base — three-onset lag 1: 2/200, zero elsewhere.
  Cause: spike-only lag-1/2 windows never strictly match, so
  the conjunctive gate cannot forgive them; their retained
  spike evidence costs a transient DEGRADED (safe direction).
  `test_lag_noise_monte_carlo` now locks exactly these counts.
  The freeze-escape audit is unchanged (alternating-from-onset
  still latches at update 11 — its first cycle still holds
  ≤ 1 counted fault, so forgiveness still consumes it).
- **C3 forgiveness scope (code):** forgiveness cleared the
  GLOBAL accumulators pre-fix, wiping pre-window evidence from
  an independent fault (RT probe M2d-1: severe-partial ×8
  evidence zeroed in one step, back to HEALTHY). The monitor
  now snapshots evidence + aggregate deques at each onset;
  forgiveness restores that snapshot — evidences decayed by
  the window's pre-catch-up updates (as if they had been
  clean), deques holding pre-window entries plus the catch-up
  entry. Measured on the M2d-1 shape: snapshot partial
  evidence 6.0 at the onset; at the forgiveness update (27)
  the state is **DEGRADED with partial evidence 3.5 retained**
  (restored 4.0, minus the catch-up update's own 0.5 decay)
  and aggregate lengths 10 / 23 (not 1 / 1); forgiveness
  events 2 → 1; the stream spends 21 updates DEGRADED before
  recovering by normal decay. The no-prior-fault control
  forgives clean to HEALTHY with 0.0 retained, as before.
  The RT F3 repeated-burst shape (partial ×8 + laggy onset
  washout, ×3 cycles) now records **zero** forgiveness events
  and ends DEGRADED 72 — identical to its no-washout control.
  Locked by `test_catchup_forgiveness_scoped_to_window`.
- **C2 burst boundary (disclosure, re-measured — residual NOT
  closed):** the ×6-frozen-with-clean-tail burst holds exactly
  one counted fault, so the conjunctive gate leaves the
  boundary where it was: per-onset, repeatable across
  ≥ 5-update hover gaps with no cumulative accounting.
  Re-measured: 3 bursts = **18.00 m**, 10 bursts = **60.00 m**
  of forgotten error, HEALTHY on every update, 1 forgiveness
  per burst; a same-cruise second burst still latches. §2(iv)
  and §6 now state this with the numbers; locked by
  `test_onset_burst_boundary_disclosed`. Closing it needs a
  cumulative forgiven-error budget or refusing the
  indistinguishable marginal-lag case — a PI design call,
  disclosed here rather than taken unilaterally.
- **C1 iid noise assumption (disclosure):** §2 now names the
  noise model's **iid** assumption as load-bearing beside its
  σ, with the red team's measured sensitivity: AR(1) ρ = 0.8
  at nominal σ → healthy lag 4/5 latches in 2/200 flights
  (three-onset lag 4 and 5, single-onset lag 4; seed base
  770000); iid σ × 1.5 → latches at lag ≥ 3 (≤ 4/200). No
  behavior change: in-envelope (iid, nominal σ) the fix is
  exact.
- **§2 claim correction:** the falsified "a fault counted
  repeatedly in the window … is never forgiven" sentence (and
  the identical pre-fix comment in `nav_health.py`) is
  corrected in place in §2 to the measured truth under the new
  gate, with the falsification recorded — not silently
  rewritten.
- **No other behavior moved:** `pose_trustworthy` semantics
  unchanged (True iff HEALTHY, asserted on every update of the
  new tests and the RT probe shapes); all suites pass
  (`test_nav_health.py` 24, `test_exec_provenance.py` 7,
  `test_escalation.py` 9); calibration 96/96 zero-false-flag.
  Both CSVs re-issued under the guard at the new executed
  commits — calibration at `c06984f` with **zero non-commit
  diffs** vs the previous issue (healthy arms still 96/96) and
  mission validation at `4e73601` (PASS, 0 collisions,
  scripted UNTRUSTED 194 / gated 16; **0 non-wall-time diffs**
  across all 13 rows). Checkout-and-rerun proof: calibration
  reproduced **bit-identically** from a clean clone at
  `c06984f` (SHA-256
  `4f7d6a91f348a6338f2afbe73a344d2922d8322b756cff732adf5fa51efa296f`);
  mission reproduced with **0 non-wall-time diffs across all
  13 rows** (commit and note columns included) from a clean
  clone at `4e73601` (verdict PASS, scripted UNTRUSTED 194 /
  gated 16, 0 collisions).
