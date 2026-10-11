"""Tests for the nav-health monitor + its escalation wiring.

Behavioral, per lab doctrine: the guards must be SEEN to bite.
1. Healthy stream stays HEALTHY; pose_trustworthy True.
2. Whiteout FREEZE stream (measured signature: reported motion 12.5% of
   commanded) -> UNTRUSTED within FREEZE_SUSTAIN updates; a single bad
   update does NOT declare UNTRUSTED.
3. Recovery needs RECOVER_CLEAN clean updates, one level at a time.
4. Pose jump -> instant UNTRUSTED. One transient feature dip -> nothing.
5. No-truth audit: the monitor module imports no sim/GT module and its
   update() consumes only the report dict (checked by source scan).
6. Wiring: gate open + HEALTHY nav -> RTH; same inputs + UNTRUSTED nav
   -> LAND, RTH never emitted on any step; UNTRUSTED alone (clear
   perception) -> HOVER; legacy evaluate() without nav_health is
   unchanged (nav_health still MISSING, gate = pose flag alone).
"""
import inspect

import nav_health as nh
import escalation as esc


def rep(**kw):
    r = {"cmd_speed_mps": 1.0, "est_speed_mps": 1.0, "est_jump_m": 1.0,
         "n_features": 60, "residual_m": 0.03, "map_age_s": 0.0,
         "dt_s": 1.0}
    r.update(kw)
    return r


def test_healthy():
    m = nh.NavHealthMonitor()
    for k in range(10):
        s, _ = m.update(rep(), t=float(k))
    assert s == nh.HEALTHY and m.pose_trustworthy
    print("healthy stream: HEALTHY, gate open")


def test_freeze_bites_with_dwell():
    m = nh.NavHealthMonitor()
    s, _ = m.update(rep(est_speed_mps=0.125, est_jump_m=0.125))
    assert s == nh.HEALTHY, s  # one frozen update: dwell not met
    s, info = m.update(rep(est_speed_mps=0.125, est_jump_m=0.125))
    assert s == nh.UNTRUSTED and not m.pose_trustworthy, (s, info)
    assert any("freeze" in r for r in info["reasons"]), info
    print("freeze: UNTRUSTED at update 2, not at update 1")


def test_freeze_needs_commanded_motion():
    # Estimator quiet while the host commands ~hover is NOT a freeze.
    m = nh.NavHealthMonitor()
    for k in range(5):
        s, _ = m.update(rep(cmd_speed_mps=0.05, est_speed_mps=0.0,
                            est_jump_m=0.0))
    assert s == nh.HEALTHY, s
    print("quiet estimator at hover command: no false freeze")


def test_recovery_hysteresis():
    m = nh.NavHealthMonitor()
    for _ in range(2):
        m.update(rep(est_speed_mps=0.1, est_jump_m=0.1))
    assert m.state == nh.UNTRUSTED
    for _ in range(nh.RECOVER_CLEAN - 1):
        m.update(rep())
    assert m.state == nh.UNTRUSTED  # clean run not yet complete
    m.update(rep())
    assert m.state == nh.DEGRADED   # one level only
    for _ in range(nh.RECOVER_CLEAN):
        m.update(rep())
    assert m.state == nh.HEALTHY
    print("recovery: UNTRUSTED -> DEGRADED -> HEALTHY, 3 clean each")


def test_freeze_latch_no_hover_recovery():
    # After a freeze, quiet-at-hover updates must NOT restore trust:
    # the estimator has to prove itself under commanded motion.
    m = nh.NavHealthMonitor()
    for _ in range(2):
        m.update(rep(est_speed_mps=0.1, est_jump_m=0.1))
    assert m.state == nh.UNTRUSTED
    for _ in range(10):
        s, _ = m.update(rep(cmd_speed_mps=0.0, est_speed_mps=0.0,
                            est_jump_m=0.0))
    assert s == nh.UNTRUSTED, s  # latched: no limit-cycle release
    for _ in range(nh.RECOVER_CLEAN):
        m.update(rep())  # clean under motion (cmd 1.0, est 1.0)
    assert m.state == nh.DEGRADED, m.state
    print("freeze latch: hover-quiet never recovers; motion-clean does")


def test_partial_freeze_band():
    # Probe A (round 1): ratio 0.5 sustained must close the gate
    # (DEGRADED), ratio 0.35 sustained must reach UNTRUSTED (severe).
    m = nh.NavHealthMonitor()
    states = [m.update(rep(est_speed_mps=r, est_jump_m=r))[0]
              for r in [0.5] * 12]
    assert nh.DEGRADED in states and nh.UNTRUSTED not in states, states
    assert not m.pose_trustworthy
    m2 = nh.NavHealthMonitor()
    states2 = [m2.update(rep(est_speed_mps=r, est_jump_m=r))[0]
               for r in [0.35] * 12]
    assert states2[-1] == nh.UNTRUSTED, states2
    print("partial band: 0.50 -> DEGRADED (gate closed); "
          "0.35 sustained -> UNTRUSTED")


def test_intermittent_freeze_leaky_evidence():
    # Probe B (round 1): alternating frozen/clean must accumulate.
    m = nh.NavHealthMonitor()
    states = []
    for k in range(12):
        frozen = (k % 2 == 0)
        states.append(m.update(rep(
            est_speed_mps=0.0 if frozen else 1.0,
            est_jump_m=0.0 if frozen else 1.0))[0])
    assert nh.UNTRUSTED in states, states
    print("intermittent freeze: leaky evidence latches UNTRUSTED")


def test_slow_freeze_window_rule():
    # Probe C (round 1): total freeze at cmd 0.20 m/s (below the
    # instantaneous floor) is caught by the cumulative window rule.
    # Round-2 closure: the LONG-window rule keys detection to
    # commanded TRAVEL (>= 1.0 m inside 120 updates), so 0.05 m/s
    # latches at ~update 21 and 0.02 m/s at ~update 51; healthy slow
    # flight at the same speeds must NOT fire (ratio ~1 aggregate).
    m = nh.NavHealthMonitor()
    states = [m.update(rep(cmd_speed_mps=0.20, est_speed_mps=0.0,
                            est_jump_m=0.0))[0] for _ in range(15)]
    assert nh.UNTRUSTED in states, states
    for cmd, n, first_by in ((0.30, 10, 10), (0.10, 15, 15),
                             (0.05, 25, 22), (0.02, 60, 52)):
        m2 = nh.NavHealthMonitor()
        st = [m2.update(rep(cmd_speed_mps=cmd, est_speed_mps=0.0,
                             est_jump_m=0.0))[0] for _ in range(n)]
        assert nh.UNTRUSTED in st[:first_by], (cmd, st)
        assert not m2.pose_trustworthy
    for cmd in (0.20, 0.05, 0.02):
        m3 = nh.NavHealthMonitor()
        st = [m3.update(rep(cmd_speed_mps=cmd, est_speed_mps=cmd,
                             est_jump_m=cmd))[0] for _ in range(60)]
        assert all(s == nh.HEALTHY for s in st), (cmd, st)
        assert m3.pose_trustworthy
    print("slow freeze: window rules catch 0.30/0.20/0.10/0.05/0.02 "
          "m/s total freeze; healthy slow flight stays HEALTHY")


def test_slow_partial_and_intermittent():
    # Round-2: partial (ratio 0.5) and intermittent (alternating
    # frozen/clean, aggregate ratio 0.5) freezes at cmd 0.20 m/s —
    # under the instantaneous floor — must at least close the gate
    # (DEGRADED -> pose_trustworthy False) via the long-window
    # aggregate; a severe slow partial (0.35) must reach UNTRUSTED.
    m = nh.NavHealthMonitor()
    st = [m.update(rep(cmd_speed_mps=0.20, est_speed_mps=0.10,
                        est_jump_m=0.10))[0] for _ in range(20)]
    assert nh.DEGRADED in st and not m.pose_trustworthy, st
    m = nh.NavHealthMonitor()
    st = []
    for k in range(30):
        frozen = (k % 2 == 0)
        st.append(m.update(rep(
            cmd_speed_mps=0.20,
            est_speed_mps=0.0 if frozen else 0.20,
            est_jump_m=0.0 if frozen else 0.20))[0])
    assert nh.DEGRADED in st and not m.pose_trustworthy, st
    m = nh.NavHealthMonitor()
    st = [m.update(rep(cmd_speed_mps=0.20, est_speed_mps=0.07,
                        est_jump_m=0.07))[0] for _ in range(30)]
    assert st[-1] == nh.UNTRUSTED, st
    print("slow partial/intermittent at 0.20 m/s: gate closes; "
          "severe slow partial -> UNTRUSTED")


def test_estimator_lag_is_not_a_freeze():
    # N3 (red-team round 2): pure estimator LAG at motion onset —
    # est = 0 for `lag` updates (reporting the pre-onset hover), then
    # perfect — must NEVER latch UNTRUSTED, for lag up to the stated
    # tolerance LAG_TOLERANCE=5. Measured: all HEALTHY, all updates.
    for lag in (2, 3, 5):
        m = nh.NavHealthMonitor()
        reps = [rep(est_speed_mps=0.0, est_jump_m=0.0)] * lag \
            + [rep()] * 30
        states = [m.update(r)[0] for r in reps]
        assert all(s == nh.HEALTHY for s in states), (lag, states)
        assert m.pose_trustworthy
    # Lag 5 on a speed RAMP (est = cmd 5 updates ago): also never
    # UNTRUSTED — the lag is consistent, not just at onset.
    m = nh.NavHealthMonitor()
    cmds = [0.5 + 0.02 * k for k in range(50)]
    states = []
    for k, c in enumerate(cmds):
        e = cmds[k - 5] if k >= 5 else 0.0
        states.append(m.update(rep(cmd_speed_mps=c, est_speed_mps=e,
                                   est_jump_m=e))[0])
    assert all(s == nh.HEALTHY for s in states), states
    # Lag after a witnessed hover and after a mid-stream setpoint
    # step: also excused (the hypothesis re-arms at onsets/steps).
    m = nh.NavHealthMonitor()
    reps = [rep(cmd_speed_mps=0.0, est_speed_mps=0.0, est_jump_m=0.0)] * 10 \
        + [rep(est_speed_mps=0.0, est_jump_m=0.0)] * 3 + [rep()] * 20
    states = [m.update(r)[0] for r in reps]
    assert all(s == nh.HEALTHY for s in states), states
    print("lag: onset lag 2/3/5, lag-5 ramp, hover-onset and step lag "
          "all stay HEALTHY (no freeze latch)")


def test_lag_beyond_tolerance_is_a_freeze():
    # N3 limit: lag > LAG_TOLERANCE is treated as a freeze. A stream
    # reporting 0 for 10 updates at cmd 1.0 latches UNTRUSTED at
    # update index LAG_TOLERANCE+1 (tolerance updates of ambiguity +
    # the usual evidence dwell) — the SAME stream is what a total
    # freeze beginning exactly at an onset looks like, and this delay
    # is the disclosed price of the lag tolerance.
    m = nh.NavHealthMonitor()
    states = [m.update(rep(est_speed_mps=0.0, est_jump_m=0.0))[0]
              for _ in range(30)]
    first = next(i for i, s in enumerate(states) if s == nh.UNTRUSTED)
    assert first == nh.LAG_TOLERANCE + 1, (first, states)
    # A freeze beginning DURING established motion (no onset) is not
    # delayed at all: it latches at the 2nd frozen update, per the
    # pre-existing bands — including the measured 0.125 signature.
    for frozen_est in (0.0, 0.125):
        m = nh.NavHealthMonitor()
        for _ in range(10):
            m.update(rep())
        states = [m.update(rep(est_speed_mps=frozen_est,
                                est_jump_m=frozen_est))[0]
                  for _ in range(10)]
        assert states[0] == nh.HEALTHY and states[1] == nh.UNTRUSTED, \
            (frozen_est, states)
    print("lag limit: lag-10 / onset freeze latches at update "
          "LAG_TOLERANCE+2; mid-motion freeze still latches at 2")


def _noisy_lag_flight(rng, lag, profile):
    """Healthy estimator, pure pipeline lag, harness noise model:
    achieved = 0.9*cmd + N(0, 0.02); report = delayed achieved +
    N(0, 0.03), floored at 0 (the red-team round-3 construction)."""
    ach = [0.9 * c + rng.normal(0, 0.02) for c in profile]
    m = nh.NavHealthMonitor()
    states = []
    for k, c in enumerate(profile):
        src = ach[k - lag] if k - lag >= 0 else rng.normal(0, 0.02)
        est = max(0.0, src + rng.normal(0, 0.03))
        states.append(m.update(rep(cmd_speed_mps=c, est_speed_mps=est,
                                   est_jump_m=est))[0])
    return states


def test_lag_noise_monte_carlo():
    # N3-noise (red-team round 3): healthy + lag + noise was the
    # untested product — the round-2 lag tests are noiseless and the
    # calibration harness has noise but no lag. Seeded Monte Carlo on
    # the red team's own protocol (200 seeds/condition, seeds
    # 31000+seed, three-onset profile + a single-onset profile).
    # Round-3 result: ZERO UNTRUSTED and zero non-HEALTHY at every
    # lag (pre-round-3-fix this protocol latched 12/200 at lag 2,
    # 40/200 at lag 3, 66/200 at lag 5 on the three-onset profile).
    # Round-4 re-measure under the CONJUNCTIVE forgiveness gate
    # (matched AND counted faults <= 1): ZERO UNTRUSTED latches at
    # every lag 0-5 on both profiles — the safety target holds —
    # but the spike-only lag-1/2 flights whose window never strictly
    # matched (the far spike is their only lagged report; their
    # catch-up match is vacuous at constant cmd) are no longer
    # forgiven, and their retained spike evidence costs a transient
    # DEGRADED: measured DEGRADED-only flights are three-onset
    # lag 1: 3/200, single-onset lag 1: 3/200, single-onset lag 2:
    # 1/200, zero at every other lag (incl. lag 4, added round 4).
    # Those exact counts are locked here; the deviation errs toward
    # longer distrust (safe direction), never toward a latch.
    import numpy as np
    profiles = {
        "three-onset": ([0.0] * 6 + [1.0] * 30 + [0.6] * 10
                        + [0.0] * 6 + [0.5] * 10 + [1.0] * 20),
        "single-onset": [0.0] * 6 + [1.0] * 40,
    }
    expected_nonhealthy = {("three-onset", 1): 3,
                           ("single-onset", 1): 3,
                           ("single-onset", 2): 1}
    for pname, prof in profiles.items():
        for lag in (0, 1, 2, 3, 4, 5):
            n_untr = n_nonhealthy = 0
            for seed in range(200):
                rng = np.random.default_rng(31000 + seed)
                st = _noisy_lag_flight(rng, lag, prof)
                n_untr += nh.UNTRUSTED in st
                n_nonhealthy += any(s != nh.HEALTHY for s in st)
            assert n_untr == 0, (pname, lag, n_untr)
            assert n_nonhealthy == expected_nonhealthy.get(
                (pname, lag), 0), (pname, lag, n_nonhealthy)
    print("lag x noise Monte Carlo (200 seeds x 2 profiles, lags "
          "0-5): 0 UNTRUSTED latches; DEGRADED-only flights "
          "three-onset lag1 3/200, single-onset lag1 3/200, "
          "single-onset lag2 1/200, 0 elsewhere [round-4 gate]")


def test_lag_noise_freeze_escape_audit():
    # N3-noise, the other side: true freezes under noise must still
    # latch within the disclosed onset bound (7th onset update; +1
    # when noise lifts a frozen report into the partial band), and
    # the deterministic escape shapes keep their measured prices.
    import numpy as np

    def first_untrusted(states):
        return next((i + 1 for i, s in enumerate(states)
                     if s == nh.UNTRUSTED), None)

    # Deterministic shapes.
    m = nh.NavHealthMonitor()
    st = [m.update(rep(est_speed_mps=0.08, est_jump_m=0.08))[0]
          for _ in range(20)]
    assert first_untrusted(st) == 7, st  # in-band onset freeze
    m = nh.NavHealthMonitor()
    st = [m.update(rep(est_speed_mps=0.10, est_jump_m=0.10))[0]
          for _ in range(20)]
    assert first_untrusted(st) == 2, st  # far-band onset freeze
    m = nh.NavHealthMonitor()
    st = [m.update(rep())[0] for _ in range(10)] + \
         [m.update(rep(est_speed_mps=0.08, est_jump_m=0.08))[0]
          for _ in range(10)]
    assert first_untrusted(st) == 12, st  # mid-motion: 2nd frozen update
    m = nh.NavHealthMonitor()
    st = [m.update(rep(est_speed_mps=0.0 if k % 2 == 0 else 1.0,
                       est_jump_m=0.0 if k % 2 == 0 else 1.0))[0]
          for k in range(14)]
    assert first_untrusted(st) == 11, st  # alternating from onset:
    # forgiveness consumes the first catch-up cycle (was update 7)
    m = nh.NavHealthMonitor()
    st = [m.update(rep(est_speed_mps=0.0, est_jump_m=0.0))[0]
          for _ in range(6)] + [m.update(rep())[0] for _ in range(10)]
    assert nh.UNTRUSTED not in st, st  # frozen reports spanning only
    # onset updates 1-6, then clean: forgiven catch-up boundary
    m = nh.NavHealthMonitor()
    st = [m.update(rep(est_speed_mps=0.0, est_jump_m=0.0))[0]
          for _ in range(7)] + [m.update(rep())[0] for _ in range(10)]
    assert first_untrusted(st) == 7, st  # a 7th frozen report latches

    # Noisy shapes (200 seeds each; frozen report = pinned value +
    # N(0, 0.03) floored at 0; healthy prefix at the 0.9 model).
    def noisy(cmds, pinned, prefix=0, lag=None):
        firsts = []
        for seed in range(200):
            rng = np.random.default_rng(91000 + seed)
            if lag is None:
                ach = None
            else:
                ach = [0.9 * c + rng.normal(0, 0.02) for c in cmds]
            mon = nh.NavHealthMonitor()
            sts = []
            for k, c in enumerate(cmds):
                if lag is not None:
                    src = ach[k - lag] if k - lag >= 0 \
                        else rng.normal(0, 0.02)
                    est = max(0.0, src + rng.normal(0, 0.03))
                elif k < prefix:
                    est = max(0.0, 0.9 * c + rng.normal(0, 0.036))
                else:
                    est = max(0.0, pinned + rng.normal(0, 0.03))
                sts.append(mon.update(rep(cmd_speed_mps=c,
                                          est_speed_mps=est,
                                          est_jump_m=est))[0])
            firsts.append(first_untrusted(sts))
        return firsts

    f = noisy([1.0] * 20, 0.06)
    assert all(x is not None for x in f) and max(f) <= 7, max(f)
    f = noisy([0.4] * 20, 0.05)  # scaled 0.125 signature
    assert all(x is not None for x in f) and max(f) <= 8, max(f)
    f = noisy([1.0] * 20, 0.1125)  # the 0.125 whiteout signature
    assert all(x is not None for x in f) and min(f) == 2 \
        and max(f) <= 7, (min(f), max(f))
    f = noisy([0.0] * 5 + [0.5] * 25, 0.0, prefix=5)
    assert all(x is not None for x in f) and max(f) <= 12, max(f)
    f = noisy([0.0] * 5 + [0.5] * 25, 0.0625, prefix=5)
    assert all(x is not None for x in f) and max(f) <= 12, max(f)
    f = noisy([1.0] * 10 + [1.4] * 15, 0.0, prefix=10)
    assert all(x == 12 for x in f), (min(f), max(f))  # step freeze:
    # 2nd frozen update, noise or not
    f = noisy([0.0] * 6 + [1.0] * 40, None, lag=10)
    assert all(x is not None for x in f) and max(f) <= 13, max(f)
    print("freeze-escape audit (deterministic + noisy, 200 seeds/"
          "shape): every freeze class latches within disclosed "
          "bounds (onset 7, +1 partial-band noise; descent stream "
          "12; step/mid-motion 2nd frozen update; lag-10 stream 13)")


def _forgiveness_updates(infos):
    return [i + 1 for i, inf in enumerate(infos)
            if any("catch-up" in r for r in inf["reasons"])]


def test_catchup_forgiveness_conjunctive_gate():
    # Round-4 FAIL item (REDTEAM4 S2c): the pre-fix gate was
    # `window_matched OR counted_faults <= 1` — one strictly-matched
    # onset update (est 0.0, the canonical first report of a freeze
    # at an onset) unlocked forgiveness for arbitrarily many counted
    # faults: matched + severe-partial 0.35 x2..x9 was fully forgiven
    # (no latch, windows wiped, back to HEALTHY, up to 6.85 m of
    # error forgotten). The gate is now CONJUNCTIVE: matched AND
    # counted faults <= LAG_FORGIVENESS_FAULT_BOUND (= 1, the
    # lag-1/2 onset-spike case: exactly one counted fault).
    def run(stream):
        m = nh.NavHealthMonitor()
        states, infos = [], []
        for c, e in stream:
            s, info = m.update(rep(cmd_speed_mps=c, est_speed_mps=e,
                                   est_jump_m=e))
            assert m.pose_trustworthy == (s == nh.HEALTHY)
            states.append(s)
            infos.append(info)
        return m, states, _forgiveness_updates(infos)

    # The legitimate bound case still carries: matched + ONE counted
    # severe-partial fault + catch-up is forgiven, all HEALTHY.
    _, st, forg = run([(1.0, 0.0)] + [(1.0, 0.35)] * 1
                      + [(1.0, 1.0)] * 20)
    assert forg == [3] and all(s == nh.HEALTHY for s in st), (forg, st)
    # The RT4 reproducer family: x2..x9 is NEVER forgiven (zero
    # catch-up events), the gate is still closed (not HEALTHY) on
    # the catch-up update itself, and the evidence is retained:
    # x4..x8 are still DEGRADED after 20 clean updates; x9 — one
    # update short of the severe-sustain latch pre-fix — now
    # LATCHES at update 11 (the retained long-window aggregate
    # completes the severe sustain on the catch-up update).
    for n in range(2, 10):
        _, st, forg = run([(1.0, 0.0)] + [(1.0, 0.35)] * n
                          + [(1.0, 1.0)] * 20)
        assert forg == [], (n, forg)
        assert st[1 + n] != nh.HEALTHY, (n, st[1 + n])
        if 4 <= n <= 8:
            assert st[-1] == nh.DEGRADED, (n, st[-1])
        if n == 9:
            assert next(i + 1 for i, s in enumerate(st)
                        if s == nh.UNTRUSTED) == 11, st
    # x2/x3 do eventually recover — by the normal leaky decay +
    # hysteresis over 12 / 21 DEGRADED updates, NOT by forgiveness.
    _, st, _ = run([(1.0, 0.0)] + [(1.0, 0.35)] * 2 + [(1.0, 1.0)] * 20)
    assert st[-1] == nh.HEALTHY and st.count(nh.DEGRADED) == 12, st
    _, st, _ = run([(1.0, 0.0)] + [(1.0, 0.35)] * 3 + [(1.0, 1.0)] * 20)
    assert st[-1] == nh.HEALTHY and st.count(nh.DEGRADED) == 21, st
    # Frozen-band sibling (RT F2c): matched + frozen 0.20 x2..x4 was
    # forgiven pre-fix for x2/x3; now zero forgiveness, latch stands.
    for n in (2, 3, 4):
        _, st, forg = run([(1.0, 0.0)] + [(1.0, 0.20)] * n
                          + [(1.0, 1.0)] * 20)
        assert forg == [] and nh.UNTRUSTED in st, (n, forg, st)
    print("round-4 conjunctive gate: matched+sev x1 forgiven (bound "
          "case); x2..x9 zero forgiveness, gate closed at catch-up, "
          "x4..x8 still DEGRADED after 20 clean, x9 latches at 11")


def test_catchup_forgiveness_scoped_to_window():
    # Round-4 C3: pre-fix forgiveness cleared the GLOBAL accumulators,
    # wiping evidence from an independent earlier fault in one step
    # (RT probe M2d-1). Forgiveness is now contribution-scoped: it
    # restores the pre-window snapshot (evidence decayed as if the
    # window's updates had been clean; aggregate deques keep their
    # pre-window entries). Proof on the RT shape: a mid-motion severe
    # partial x8 (partial evidence 8.0), hover, a laggy onset, then a
    # catch-up. At the forgiveness update the monitor must still be
    # DEGRADED with the pre-window evidence RETAINED (measured:
    # snapshot 6.0 at the onset, restored 4.0, 3.5 after the
    # catch-up update's own decay — not 0.0), and the aggregate
    # windows must keep their pre-window entries (lengths 10 / 23,
    # not 1 / 1).
    stream = ([(1.0, 1.0)] * 10 + [(1.0, 0.35)] * 8 + [(0.0, 0.0)] * 4
              + [(1.0, 0.0)] * 3 + [(1.0, 0.2)] * 1 + [(1.0, 1.0)] * 20)
    m = nh.NavHealthMonitor()
    seen = None
    for i, (c, e) in enumerate(stream):
        s, info = m.update(rep(cmd_speed_mps=c, est_speed_mps=e,
                               est_jump_m=e))
        if any("catch-up" in r for r in info["reasons"]):
            seen = (i + 1, s, m._partial_evidence,
                    len(m._window), len(m._window_long))
    assert seen is not None, "laggy onset must still be forgiven"
    upd, s_at, pev, wlen, lwlen = seen
    assert upd == 27 and s_at == nh.DEGRADED, seen
    assert pev == 3.5 and pev >= nh.PARTIAL_EVIDENCE, seen
    assert (wlen, lwlen) == (10, 23), seen
    # Control: the SAME laggy onset with no prior fault forgives to
    # HEALTHY with zero retained evidence — scoping preserves only
    # evidence that was really there before the window.
    ctrl = ([(1.0, 1.0)] * 10 + [(0.0, 0.0)] * 4
            + [(1.0, 0.0)] * 3 + [(1.0, 0.2)] * 1 + [(1.0, 1.0)] * 20)
    m2 = nh.NavHealthMonitor()
    seen2 = None
    for i, (c, e) in enumerate(ctrl):
        s, info = m2.update(rep(cmd_speed_mps=c, est_speed_mps=e,
                                est_jump_m=e))
        if any("catch-up" in r for r in info["reasons"]):
            seen2 = (i + 1, s, m2._partial_evidence)
    assert seen2 is not None and seen2[1] == nh.HEALTHY \
        and seen2[2] == 0.0, seen2
    print("round-4 scoped forgiveness: independent severe-partial x8 "
          "evidence survives a lag forgiveness (partial evidence 3.5 "
          "retained at the forgiveness update, state DEGRADED, "
          "aggregate windows keep pre-window entries); no-prior-fault "
          "control forgives clean to HEALTHY")


def test_onset_burst_boundary_disclosed():
    # Round-4 C2, the disclosed residual the conjunctive gate does
    # NOT close (and must not silently grow): frozen reports confined
    # to onset updates 1-6 hold exactly ONE counted fault (the first
    # five reports are lag-matched), so the burst still satisfies
    # matched AND faults <= 1 and is forgiven — per onset. With a
    # >= 5-update hover gap (cmd history fully evicted) the shape
    # repeats with no cumulative accounting: measured x3 = 18.00 m
    # and x10 = 60.00 m of forgotten error, HEALTHY on every update.
    # This test LOCKS that measured boundary (NAV_HEALTH.md S6); if
    # a future change moves it, the disclosure must move with it.
    def run(stream):
        m = nh.NavHealthMonitor()
        states, infos = [], []
        for c, e in stream:
            s, info = m.update(rep(cmd_speed_mps=c, est_speed_mps=e,
                                   est_jump_m=e))
            states.append(s)
            infos.append(info)
        return states, _forgiveness_updates(infos)

    burst = [(0.0, 0.0)] * 5 + [(1.0, 0.0)] * 6 + [(1.0, 1.0)] * 8
    st, forg = run(burst * 3)
    assert len(forg) == 3 and all(s == nh.HEALTHY for s in st), forg
    assert sum(c - e for c, e in burst * 3) == 18.0
    st, forg = run(burst * 10)
    assert len(forg) == 10 and all(s == nh.HEALTHY for s in st), forg
    assert sum(c - e for c, e in burst * 10) == 60.0
    # The harvest needs a new onset: a second freeze burst in the
    # SAME cruise (no hover gap) is not forgiven — it latches.
    st, forg = run([(1.0, 0.0)] * 6 + [(1.0, 1.0)] * 1
                   + [(1.0, 0.0)] * 6 + [(1.0, 1.0)] * 10)
    assert len(forg) == 1 and nh.UNTRUSTED in st, (forg, st)
    print("onset burst boundary (disclosed, locked): hover>=5 + "
          "freeze x6 + clean repeats per onset — x3 = 18.00 m, "
          "x10 = 60.00 m, HEALTHY on every update; same-cruise "
          "repeat latches UNTRUSTED")


def test_long_freeze_release_latency_measured():
    # N4 (red-team round 2): release latency is NOT a flat 3 clean
    # updates — the uncapped evidence accumulators and the aggregate
    # windows' fault memory make it grow with fault duration. These
    # are the MEASURED numbers the documentation states; if a change
    # moves them, the docs must move too (drift doctrine).
    # (a) The documented case: 2-update freeze -> 3 clean to
    #     DEGRADED, 6 total to HEALTHY (test_recovery_hysteresis).
    # (b) 10-update total freeze MID-STREAM (healthy prefix, so no
    #     onset/lag ambiguity): 49 clean updates to DEGRADED,
    #     52 to HEALTHY.
    m = nh.NavHealthMonitor()
    for _ in range(10):
        m.update(rep())
    for _ in range(10):
        m.update(rep(est_speed_mps=0.0, est_jump_m=0.0))
    assert m.state == nh.UNTRUSTED
    to_deg = to_health = None
    for i in range(200):
        s, _ = m.update(rep())
        if to_deg is None and s != nh.UNTRUSTED:
            to_deg = i + 1
        if s == nh.HEALTHY:
            to_health = i + 1
            break
    assert (to_deg, to_health) == (49, 52), (to_deg, to_health)
    # (c) A 4-update partial (ratio 0.50, DEGRADED, no latch) needs
    #     13 clean updates to return to HEALTHY — not the 3 the old
    #     hysteresis line alone would suggest.
    m = nh.NavHealthMonitor()
    for _ in range(4):
        m.update(rep(est_speed_mps=0.5, est_jump_m=0.5))
    assert m.state == nh.DEGRADED and not m.pose_trustworthy
    n = 0
    for i in range(60):
        s, _ = m.update(rep())
        n = i + 1
        if s == nh.HEALTHY:
            break
    assert n == 13 and m.pose_trustworthy, n
    print("release latency measured: freeze-10 mid-stream 49 clean "
          "to DEGRADED / 52 to HEALTHY; partial-4 13 clean to HEALTHY")


def test_residual_and_map_age_channels():
    # F4 (round 1): each channel fires behaviorally; single-channel
    # ceilings are DEGRADED (intended, probe E); combined score >= 3
    # sustained reaches UNTRUSTED.
    m = nh.NavHealthMonitor()
    s = [m.update(rep(residual_m=0.20))[0] for _ in range(4)][-1]
    assert s == nh.DEGRADED, s
    m = nh.NavHealthMonitor()
    s = [m.update(rep(residual_m=0.90))[0] for _ in range(6)][-1]
    assert s == nh.DEGRADED, s  # +2 alone caps at DEGRADED
    m = nh.NavHealthMonitor()
    s = [m.update(rep(map_age_s=600.0))[0] for _ in range(6)][-1]
    assert s == nh.DEGRADED, s  # staleness alone caps at DEGRADED
    m = nh.NavHealthMonitor()
    states = [m.update(rep(residual_m=0.35, map_age_s=1.5))[0]
              for _ in range(4)]
    assert states[-1] == nh.UNTRUSTED, states  # 2+1=3, sustained 2
    print("channels: residual/map-age fire; single-channel ceiling "
          "DEGRADED; combined score>=3 -> UNTRUSTED")


def test_wired_path_audit():
    # F3 (round 1): the audit extends beyond the monitor file — the
    # wired harnesses must not feed achieved/true motion as cmd.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cal = open(os.path.join(here, "run_nav_health_calibration.py")).read()
    assert '"cmd_speed_mps": true_disp' not in cal
    assert '"cmd_speed_mps": cmd_speed' in cal
    loop = open(os.path.join(here, "mission_dynamics_v5.py")).read()
    assert '"cmd_speed_mps": float(np.linalg.norm(vehicle.vel))' \
        not in loop
    assert '"cmd_speed_mps": _cmd_speed' in loop
    print("wired-path audit: cmd is setpoint-derived in both harnesses")


def test_jump_and_transients():
    m = nh.NavHealthMonitor()
    s, _ = m.update(rep(est_jump_m=4.0, est_speed_mps=4.0))
    assert s == nh.UNTRUSTED, s
    m2 = nh.NavHealthMonitor()
    s, _ = m2.update(rep(n_features=0))
    assert s == nh.HEALTHY, s  # single starved scan: nothing
    print("pose jump: instant UNTRUSTED; transient feature dip: ignored")


def test_no_truth_in_monitor():
    src = inspect.getsource(nh)
    # Code-level references only: the module's docstrings legitimately
    # DISCUSS ground truth (to state it is never read); what must not
    # exist is any import of, or call into, a truth-carrying module.
    for banned in ("import vehicle", "from vehicle", "pick_start_goal",
                   "ground_truth", "true_pose", "import obscurant",
                   "mission_dynamics", "import sensor"):
        assert banned not in src, banned
    print("no-truth audit: monitor source carries no GT reference")


def _drive(policy, nav, steps=45):
    acts = []
    for k in range(steps):
        a, info = policy.evaluate(1.0, 1.0, 0.0, False, dt=1.0,
                                  pose_trustworthy=True, t=float(k),
                                  nav_health=nav)
        acts.append((a, info))
    return acts


def test_wiring_gate_flips_with_health():
    # Clear perception, gate input True, HEALTHY nav, but no trigger
    # rows fire -> continue (nav alone does not invent a trigger).
    p = esc.EscalationPolicy()
    a, info = p.evaluate(1.0, 1.0, 0.0, False, dt=1.0,
                         pose_trustworthy=True, nav_health="HEALTHY")
    assert a == esc.CONTINUE and info["nav_health"] == "HEALTHY"
    assert "nav_health" not in info["missing_triggers"]
    # UNTRUSTED nav alone -> HOVER immediately.
    p2 = esc.EscalationPolicy()
    a, info = p2.evaluate(1.0, 1.0, 0.0, False, dt=1.0,
                          pose_trustworthy=True, nav_health="UNTRUSTED")
    assert a == esc.HOVER, (a, info)
    print("wiring: HEALTHY+clear -> continue; UNTRUSTED alone -> hover")


def test_wiring_untrusted_never_rth():
    # Sustained degradation + gate input True: HEALTHY nav -> RTH path;
    # UNTRUSTED nav -> the gate closes, RTH is NEVER emitted, ends LAND.
    big = esc.DEGRADE_SUSTAIN_RTH_S + esc.HOLD_OVERRIDE_S + 1.0
    p = esc.EscalationPolicy()
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=big,
                      pose_trustworthy=True, nav_health="HEALTHY")
    assert a == esc.RTH, a
    p2 = esc.EscalationPolicy()
    a, info = p2.evaluate(0.5, 0.05, 0.0, False, dt=big,
                          pose_trustworthy=True, nav_health="UNTRUSTED")
    assert a == esc.LAND and info["rth_gated"], (a, info)
    # Multi-step: sustained UNTRUSTED over 45 s, RTH never leaks.
    p3 = esc.EscalationPolicy()
    for k in range(45):
        a, _ = p3.evaluate(0.5, 0.05, 0.0, False, dt=1.0,
                           pose_trustworthy=True, t=float(k),
                           nav_health="UNTRUSTED")
        assert a != esc.RTH, (k, a)
        assert p3._current != esc.RTH
    assert a == esc.LAND, a
    print("wiring: UNTRUSTED closes gate on every step; LAND, never RTH")


def test_legacy_unchanged():
    p = esc.EscalationPolicy()
    big = esc.DEGRADE_SUSTAIN_RTH_S + esc.HOLD_OVERRIDE_S + 1.0
    a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=big,
                         pose_trustworthy=True)
    assert a == esc.RTH and info["missing_triggers"] == \
        ["battery", "link", "nav_health"], (a, info)
    print("legacy evaluate() without nav_health: unchanged")


if __name__ == "__main__":
    test_healthy()
    test_freeze_bites_with_dwell()
    test_freeze_needs_commanded_motion()
    test_recovery_hysteresis()
    test_freeze_latch_no_hover_recovery()
    test_partial_freeze_band()
    test_intermittent_freeze_leaky_evidence()
    test_slow_freeze_window_rule()
    test_slow_partial_and_intermittent()
    test_estimator_lag_is_not_a_freeze()
    test_lag_beyond_tolerance_is_a_freeze()
    test_lag_noise_monte_carlo()
    test_lag_noise_freeze_escape_audit()
    test_catchup_forgiveness_conjunctive_gate()
    test_catchup_forgiveness_scoped_to_window()
    test_onset_burst_boundary_disclosed()
    test_long_freeze_release_latency_measured()
    test_residual_and_map_age_channels()
    test_wired_path_audit()
    test_jump_and_transients()
    test_no_truth_in_monitor()
    test_wiring_gate_flips_with_health()
    test_wiring_untrusted_never_rth()
    test_legacy_unchanged()
    print("ALL NAV-HEALTH TESTS PASS")
