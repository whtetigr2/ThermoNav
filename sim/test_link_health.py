"""Tests for the link-health model (link_health.py) + its escalation wiring.

Behavioral, per lab doctrine — the guards must be seen to bite:
1. Tracker: HEALTHY under nominal traffic; LOST declared only after
   5.0 s of silence (cited threshold), not before; recovery passes
   through DEGRADED; DEGRADED under sustained loss / high latency.
2. Dropout duration accounting: silence and time-in-state track the
   scripted dropout length; window loss matches delivered counts.
3. Channel latency application: a packet is never delivered before its
   sampled latency elapses; dropped packets never arrive.
4. No-leakage: tracker report exposes receiver observables only (no
   channel condition / hidden-state keys).
5. Policy wiring (escalation.EscalationPolicy with link_report):
   - short dropout (3 s < LOST declaration) never escalates past
     CONTINUE and the mission-level state recovers;
   - LOST declared -> HOVER; LOST-evidence bucket >=10 s -> RTH desired, held
     at HOVER through the override window, then committed — gated to
     LAND when pose is untrustworthy (logged), real RTH when trusted;
   - DEGRADED-evidence bucket >=5 s -> HOVER, recovery -> CONTINUE;
   - link recovery cancels a pending RTH (de-escalation);
   - without link_report the policy is the old stub (no link rows).
6. Markov driver: seeded run produces all three conditions' effects
   (deliveries + drops) and the tracker never crashes; deterministic
   under the same seed.
"""
import numpy as np

import link_health as lh
from link_health import LinkChannel, LinkHealthTracker
import escalation as esc
from escalation import EscalationPolicy, CONTINUE, HOVER, RTH, LAND


def _drive(channel, t_end, payload_fn=None):
    """Emit one packet per tick and advance the wire; returns reports
    sampled every 0.5 s as {t: report}."""
    reps = {}
    t = 0.0
    dt = 1.0 / lh.PACKET_RATE_HZ
    i = 0
    while t < t_end - 1e-9:
        channel.send(t, (payload_fn(t) if payload_fn else {}) or {})
        channel.step(t + dt)
        t += dt
        i += 1
        if i % 5 == 0:
            reps[round(t, 1)] = channel.tracker.report(t)
    return reps


def test_tracker_nominal_healthy():
    ch = LinkChannel({"mode": "nominal"}, seed=11)
    reps = _drive(ch, 10.0)
    for t, r in reps.items():
        assert r["state_name"] == "HEALTHY", (t, r)
        assert r["window_loss_frac"] <= 0.05, (t, r)
        assert r["latency_ewma_ms"] < 100.0, (t, r)
    print("tracker nominal: HEALTHY throughout, loss<=5%, EWMA<100 ms")


def test_tracker_lost_timing_and_recovery():
    ch = LinkChannel({"mode": "scripted",
                      "schedule": [(2.0, 12.0, "dropout")]}, seed=12)
    reps = _drive(ch, 20.0)
    # Silence starts ~2 s. LOST must NOT appear before 2+5=7 s...
    early = [t for t, r in reps.items() if t < 6.9
             and r["state_name"] == "LOST"]
    assert not early, early
    # ...and MUST appear by 7.5 s.
    assert any(r["state_name"] == "LOST" for t, r in reps.items()
               if 7.0 <= t <= 7.5), {t: r["state_name"] for t, r in reps.items()}
    # Dropout ends at 12 s: first report after is DEGRADED (recovering),
    # and HEALTHY returns once the window clears.
    post = [r["state_name"] for t, r in sorted(reps.items()) if t >= 12.5]
    assert post[0] == "DEGRADED", post[:3]
    assert "HEALTHY" in post, post
    print("tracker LOST timing: not before 6.9 s, declared by 7.5 s, "
          "recovery via DEGRADED -> HEALTHY")


def test_dropout_duration_accounting():
    tr = LinkHealthTracker()
    # 3 s of perfect traffic, then 4 s silence, checking at 0.1 s steps.
    for i in range(30):
        tr.note_window(i * 0.1, 1, [25.0])
    sil_max = 0.0
    for i in range(30, 70):
        tr.note_window(i * 0.1, 1, [])
        sil_max = max(sil_max, tr.silence_s(i * 0.1 + 0.1))
    assert abs(sil_max - 4.0) < 0.15, sil_max
    r = tr.report(7.0)
    # 4 s silence: degraded (window loss 100%) but NOT lost — the cited
    # 5 s declaration threshold has not been crossed.
    assert r["state_name"] == "DEGRADED", r
    assert r["window_loss_frac"] == 1.0, r  # trailing 2 s: all missed
    # one more second of silence crosses the cited threshold
    for i in range(70, 82):
        tr.note_window(i * 0.1, 1, [])
    r = tr.report(8.2)
    assert r["state_name"] == "LOST", r
    assert r["time_in_state_s"] < 0.3, r  # just entered LOST
    print("dropout accounting: silence tracks duration (+/-0.15 s), "
          "window loss exact, LOST at >=5 s only")


def test_degraded_classification():
    tr = LinkHealthTracker()
    rng = np.random.default_rng(0)
    # 25% loss, ~180 ms latency for 6 s -> DEGRADED (not LOST: packets flow)
    for i in range(60):
        lats = [] if rng.random() < 0.25 else [180.0]
        tr.note_window(i * 0.1, 1, lats)
    r = tr.report(6.0)
    assert r["state_name"] == "DEGRADED", r
    assert 0.15 <= r["window_loss_frac"] <= 0.45, r
    assert r["latency_ewma_ms"] > 100.0, r
    print("tracker DEGRADED: 25%-loss/180 ms stream classified DEGRADED")


def test_latency_application():
    ch = LinkChannel({"mode": "nominal"}, seed=13)
    # Force a known latency by monkeypatching the sampler draw: instead,
    # verify the invariant over many packets — nothing arrives in the
    # same tick it was sent when latency > tick, and arrival order times
    # are consistent: use a degraded channel (180 ms mean >> 100 ms tick).
    ch2 = LinkChannel({"mode": "scripted",
                       "schedule": [(0.0, 30.0, "degraded")]}, seed=14)
    sent_payload = {"marker": 1}
    ch2.send(0.0, sent_payload)
    got = ch2.step(0.1)  # first tick: latency ~180 ms cannot have arrived
    assert sent_payload not in got, got
    got_all = []
    t = 0.1
    while t < 2.0:
        got_all += ch2.step(t + 0.1)
        t += 0.1
    assert sent_payload in got_all
    print("latency application: degraded packet not delivered in tick 1, "
          "delivered within 2 s")


def test_loss_application_and_no_leakage():
    ch = LinkChannel({"mode": "scripted",
                      "schedule": [(0.0, 5.0, "dropout")]}, seed=15)
    for i in range(50):
        ch.send(i * 0.1, {"i": i})
    got = ch.step(5.0)
    assert got == [], got  # dropout: nothing delivered, ever
    assert ch.stats["dropped"] >= 49, ch.stats
    rep = ch.tracker.report(5.0)
    allowed = {"state", "state_name", "score", "time_in_state_s",
               "silence_s", "window_loss_frac", "latency_ewma_ms",
               "n_delivered", "n_expected"}
    assert set(rep.keys()) == allowed, set(rep.keys())
    print("loss application: dropout delivers nothing; report keys are "
          "receiver observables only")


def test_markov_seeded_determinism():
    def run():
        ch = LinkChannel({"mode": "markov"}, seed=42)
        _drive(ch, 120.0)
        return ch.stats
    a, b = run(), run()
    assert a == b, (a, b)
    assert a["delivered"] > 0 and a["sent"] == 1200, a
    print(f"markov: seeded runs identical, stats={a}")


# ---------------------------------------------------------------------------
# Policy wiring
# ---------------------------------------------------------------------------

def _rep(state, silence=0.0):
    return {"state": state, "state_name": lh.HEALTH_NAMES[state],
            "score": 1.0, "time_in_state_s": 0.0, "silence_s": silence,
            "window_loss_frac": 0.0, "latency_ewma_ms": 25.0,
            "n_delivered": 0, "n_expected": 0}


def _run_policy(reports, dt=0.5, pose=False):
    p = EscalationPolicy()
    actions = []
    t = 0.0
    for state in reports:
        a, info = p.evaluate(1.0, 1.0, 0.0, False, dt,
                             pose_trustworthy=pose, t=t,
                             link_report=_rep(state))
        actions.append((a, info))
        t += dt
    return p, actions


def test_policy_short_dropout_no_escalation():
    # HEALTHY 4 s, silence-driven DEGRADED ~3 s (never LOST), HEALTHY.
    seq = [lh.HEALTHY] * 8 + [lh.DEGRADED] * 6 + [lh.HEALTHY] * 8
    _, actions = _run_policy(seq)
    assert all(a == CONTINUE for a, _ in actions), actions
    print("policy short disturbance: never escalates, recovers clean")


def test_policy_long_dropout_escalates_gated():
    # LOST for 40 s of policy time (dt=0.5, 80 steps), pose untrusted.
    seq = [lh.HEALTHY] * 4 + [lh.LOST] * 80
    p, actions = _run_policy(seq, pose=False)
    acts = [a for a, _ in actions]
    assert acts[4] == HOVER, acts[:8]          # LOST declared -> hold
    assert RTH not in acts, acts               # gate: RTH never emitted
    assert acts[-1] == LAND, acts[-6:]         # ... gated land-in-place
    assert any(e[1] == "rth_gated_land_in_place" for e in p.log), p.log
    assert any(e[1] == "link_escalation" for e in p.log), p.log
    print("policy long dropout: HOVER -> (window) -> LAND via RTH gate, "
          "logged")


def test_policy_long_dropout_rth_when_trusted():
    seq = [lh.HEALTHY] * 4 + [lh.LOST] * 80
    _, actions = _run_policy(seq, pose=True)
    acts = [a for a, _ in actions]
    assert acts[-1] == RTH, acts[-6:]
    print("policy long dropout (pose trusted): commits real RTH")


def test_policy_degraded_sustained_hover_and_recovery():
    seq = [lh.DEGRADED] * 30 + [lh.HEALTHY] * 10
    _, actions = _run_policy(seq)
    acts = [a for a, _ in actions]
    # Bucket fills in 5 s at dt=0.5 -> HOVER by step ~10, never harsher.
    assert HOVER in acts and max(acts) == HOVER, acts
    assert acts[-1] == CONTINUE, acts[-4:]  # recovery de-escalates
    print("policy degraded: HOVER once bucket fills (~5 s), recovery de-escalates")


def test_policy_lost_recovery_cancels_pending():
    # LOST long enough to open the hold window, then link recovers
    # before RTH commits: pending must cancel back to CONTINUE/HOVER.
    seq = [lh.LOST] * 24 + [lh.HEALTHY] * 6  # 12 s LOST: RTH desired at 10 s
    p, actions = _run_policy(seq, pose=True)
    acts = [a for a, _ in actions]
    assert RTH not in acts, acts
    assert acts[-1] == CONTINUE, acts[-4:]
    print("policy recovery: pending RTH cancelled on link recovery")


def test_policy_stub_without_report():
    p = EscalationPolicy()
    a, info = p.evaluate(1.0, 1.0, 0.0, False, 1.0, t=0.0)
    assert a == CONTINUE and "link" in info["missing_triggers"], info
    a, info = p.evaluate(1.0, 1.0, 0.0, False, 1.0, t=1.0,
                         link_report=_rep(lh.HEALTHY))
    assert "link" not in info["missing_triggers"], info
    assert info["missing_triggers"] == ["battery", "nav_health"], info
    print("policy stub compat: no report -> link MISSING; report -> modeled")


def test_policy_flap_degraded_escalates():
    """Red-team F2 regression: a flapping link must escalate. Leaky
    bucket: DEGRADED reports accumulate 1x, HEALTHY drains 0.5x, so a
    50%-duty flap is net-positive and reaches HOVER."""
    for bad, good in ((6, 6), (4, 4)):  # reports at dt=0.5: 3s/3s, 2s/2s
        seq = ([lh.DEGRADED] * bad + [lh.HEALTHY] * good) * 25
        _, actions = _run_policy(seq)
        acts = [a for a, _ in actions]
        assert max(acts) == HOVER, (bad, good, max(acts))
        assert acts[-1] in (HOVER, CONTINUE), acts[-3:]
    print("policy flap (3s/3s and 2s/2s, 50% duty): leaky bucket "
          "reaches HOVER (F2 fixed)")


def test_policy_p1b_no_cumulative_rth():
    """Red-team F1 regression (their P1b, verbatim shape): LOST 6 s ->
    DEGRADED 20 s -> LOST 4 s -> DEGRADED. Old code committed RTH on
    cumulative 10 s. Leaky bucket: the LOST bucket drains during the
    DEGRADED gap, so RTH must NEVER commit; final action is HOVER."""
    seq = [lh.LOST] * 12 + [lh.DEGRADED] * 40 + [lh.LOST] * 8 \
        + [lh.DEGRADED] * 30
    _, actions = _run_policy(seq, pose=True)
    acts = [a for a, _ in actions]
    assert RTH not in acts, acts
    assert acts[-1] == HOVER, acts[-5:]
    print("policy P1b regression: no cumulative RTH across DEGRADED gap; "
          "final HOVER (F1 fixed)")


def test_policy_padded_dropouts_no_spurious_rth():
    """Round-2 audit shapes: brief LOST episodes padded by DEGRADED or
    HEALTHY gaps must never COMMIT RTH in these sequences. Under the
    round-2-fix commit semantics the bucket may transiently touch its
    threshold, but (a) the hold progress leaks at 3.5x on every desire
    dip and these shapes' desire duty (~75% for the worst, 8 s/2 s)
    sits under the fast path's ~78% break-even, and (b) their total
    net LOST exposure (48 s for the 8 s/2 s shape) stays under the
    75 s latch. Max action is HOVER in every shape, as sequenced."""
    shapes = [
        ([lh.LOST] * 8 + [lh.DEGRADED] * 8) * 10,    # 4 s LOST / 4 s DEG
        ([lh.LOST] * 6 + [lh.HEALTHY] * 6) * 15,     # 3 s LOST / 3 s OK
        ([lh.LOST] * 16 + [lh.DEGRADED] * 4) * 6,    # 8 s LOST / 2 s DEG
        [lh.LOST] * 18 + [lh.HEALTHY] * 60,          # single 9 s episode
    ]
    for seq in shapes:
        _, actions = _run_policy(seq, pose=True)
        acts = [a for a, _ in actions]
        assert RTH not in acts, acts
        assert max(acts) == HOVER, acts
    print("policy padded dropouts (4s/4s DEG, 3s/3s OK, 8s/2s DEG, "
          "single 9 s): no spurious RTH commit, max HOVER")


def test_policy_p3_flap_90pct_degraded_hovers():
    """Red-team P3 shape verbatim: 4.5 s DEGRADED + 0.5 s HEALTHY,
    90 % degraded duty for 60 s. Pre-fix this never escalated; the
    leaky bucket is net-positive per cycle and reaches HOVER."""
    seq = ([lh.DEGRADED] * 9 + [lh.HEALTHY] * 1) * 12
    _, actions = _run_policy(seq)
    acts = [a for a, _ in actions]
    assert max(acts) == HOVER, acts
    print("policy P3 flap (90% degraded duty, 60 s): reaches HOVER")


def test_policy_sustained_lost_commit_timing():
    """Sustained LOST still commits on schedule: at mission rate
    (dt=1.0) the bucket fills at 10 s of LOST reports and RTH commits
    after the 10 s hold window — first RTH at report 19-20, and it
    stays committed."""
    _, actions = _run_policy([lh.LOST] * 60, dt=1.0, pose=True)
    acts = [a for a, _ in actions]
    first = next(i for i, a in enumerate(acts) if a == RTH)
    assert 18 <= first <= 20, (first, acts[:22])
    assert acts[-1] == RTH and all(a == RTH for a in acts[first:]), acts
    print(f"policy sustained LOST (dt=1.0): RTH commits at report "
          f"{first + 1} (bucket 10 s + hold window), stays committed")


def test_fullpath_flap_escalates():
    """Red-team P11 shape, full path: scripted channel -> tracker ->
    policy at mission rate (1 Hz) with degraded/good cycles."""
    for bad, good in ((3, 3), (2, 2), (1, 1)):
        sched = []
        t = 8.0
        while t < 120.0:
            sched.append((t, t + bad, "degraded"))
            t += bad + good
        ch = LinkChannel({"mode": "scripted", "schedule": sched}, seed=5)
        pol = EscalationPolicy()
        lt, maxact = 0.0, 0
        for _step in range(120):
            for k in range(10):
                ch.send(lt + k * 0.1, {})
            ch.step(lt + 1.0)
            lt += 1.0
            r = ch.tracker.report(lt)
            a, _ = pol.evaluate(1.0, 1.0, 0.0, False, 1.0,
                                pose_trustworthy=False, t=lt, link_report=r)
            maxact = max(maxact, a)
        assert maxact == HOVER, (bad, good, maxact)
    print("full-path flap (3s/3s, 2s/2s, 1s/1s scripted): policy reaches HOVER")


def test_policy_committed_land_is_sticky():
    """Red-team F3 regression: once LAND commits (gated), the window
    must not re-open cyclically, and a later override step must not
    knock the committed action back to HOVER."""
    seq = [lh.LOST] * 60  # 30 s at dt=0.5: bucket 10 s + window 10 s
    p, actions = _run_policy(seq, pose=False)
    acts = [a for a, _ in actions]
    assert acts[-1] == LAND, acts[-5:]
    n_open = sum(1 for _, e, _ in p.log if e == "hold_window_open")
    n_gate = sum(1 for _, e, _ in p.log
                 if e == "rth_gated_land_in_place")
    assert n_open == 1, p.log
    assert n_gate == 1, p.log
    # Now: still LOST, but an override is (hypothetically) delivered.
    a, info = p.evaluate(1.0, 1.0, 0.0, False, 0.5, pose_trustworthy=False,
                         operator_override=True, t=99.0,
                         link_report=_rep(lh.LOST))
    assert a == LAND, (a, info)
    print("policy committed LAND: one window, one gate log, override "
          "cannot un-commit (F3 fixed)")


def test_tracker_loss_clamped_and_rng_isolation():
    tr = LinkHealthTracker()
    tr.note_window(0.1, 1, [20.0, 20.0])  # more arrivals than expected
    tr.note_window(0.2, 1, [])
    r = tr.report(0.2)
    assert 0.0 <= r["window_loss_frac"] <= 1.0, r
    assert 0.0 <= r["score"] <= 1.0, r
    # RNG isolation (red-team P7a shape): an external stream is
    # bit-identical whether or not a channel runs alongside it.
    def external_draws(with_channel):
        rng = np.random.default_rng(123)
        ch = LinkChannel({"mode": "markov"}, seed=9) if with_channel \
            else None
        out = []
        for i in range(100):
            if ch is not None:
                ch.send(i * 0.1, {})
                ch.step(i * 0.1 + 0.1)
            out.append(rng.random())
        return out
    assert external_draws(False) == external_draws(True)
    print("tracker loss/score clamped to [0,1]; channel RNG isolated "
          "from external streams")


def test_policy_n1_intermittent_lost_commits():
    """Red-team round-2 N1 regression: near-threshold intermittent
    links that NEVER committed under the hard-reset window must now
    commit in bounded time — LOST runs broken by single DEGRADED
    reports (the tracker-realizable blip), and LOST/DEGRADED tier
    straddles with zero HEALTHY reports ever. 600 s horizons."""
    shapes = [
        ([lh.LOST] * 19 + [lh.DEGRADED] * 1) * 60,   # 9.5 s / 0.5 s
        ([lh.LOST] * 19 + [lh.DEGRADED] * 2) * 57,   # 9.5 s / 1.0 s
        ([lh.LOST] * 20 + [lh.DEGRADED] * 4) * 50,   # 10 s / 2 s
        ([lh.LOST] * 10 + [lh.DEGRADED] * 10) * 60,  # 5 s / 5 s straddle
        ([lh.LOST] * 18 + [lh.DEGRADED] * 18) * 33,  # 9 s / 9 s straddle
        ([lh.LOST] * 20 + [lh.DEGRADED] * 10) * 40,  # 10 s / 5 s straddle
    ]
    for seq in shapes:
        _, actions = _run_policy(seq, pose=True)
        acts = [a for a, _ in actions]
        assert RTH in acts, max(acts)
        assert acts[-1] == RTH, acts[-5:]  # commits and stays committed
    print("policy N1 regression: intermittent 90-95% LOST duty and "
          "LOST/DEGRADED straddles all commit RTH within 600 s and "
          "stay committed (N1 fixed)")


def test_policy_n1_stray_packet_eventually_commits():
    """Red-team round-2 N1, tracker-realizable: total dropout with one
    stray 0.1 s good tick every P s, end-to-end at mission rate for
    240 s. P=12 s and P=15 s NEVER committed under the old semantics
    (HOVER forever on a ~99% loss link); all periods here must now
    commit the gated LAND."""
    for period in (12.0, 15.0, 17.0, 20.0):
        sched = []
        t = 0.0
        while t < 240.0:
            sched.append((t, t + period - 0.1, "dropout"))
            sched.append((t + period - 0.1, t + period, "good"))
            t += period
        ch = LinkChannel({"mode": "scripted", "schedule": sched}, seed=3)
        pol = EscalationPolicy()
        lt, maxact = 0.0, 0
        for _step in range(240):
            for k in range(10):
                ch.send(lt + k * 0.1, {})
            ch.step(lt + 1.0)
            lt += 1.0
            r = ch.tracker.report(lt)
            a, _ = pol.evaluate(1.0, 1.0, 0.0, False, 1.0,
                                pose_trustworthy=False, t=lt, link_report=r)
            maxact = max(maxact, a)
        assert maxact == LAND, (period, maxact)
    print("policy N1 stray-packet: dropout + 1 packet every 12/15/17/"
          "20 s all commit gated LAND within 240 s (N1 fixed)")


def test_policy_n2_committed_land_survives_single_degraded():
    """Red-team round-2 N2 regression (their probe E): a committed
    LAND must NOT release on one DEGRADED report. Release requires
    both link buckets drained to zero — ~20 s of sustained HEALTHY
    reports from full — and lands on HOVER, not mission resumption."""
    p, actions = _run_policy([lh.LOST] * 60, pose=False)  # commits LAND
    assert actions[-1][0] == LAND, actions[-3:]
    a, info = p.evaluate(1.0, 1.0, 0.0, False, 0.5, pose_trustworthy=False,
                         t=99.0, link_report=_rep(lh.DEGRADED))
    assert a == LAND and p._committed is not None, (a, info)
    assert info["commit_held"], info
    # Resumed LOST: stays LAND, no new hold window opens.
    for i in range(60):
        a, _ = p.evaluate(1.0, 1.0, 0.0, False, 0.5,
                          pose_trustworthy=False, t=99.5 + i * 0.5,
                          link_report=_rep(lh.LOST))
        assert a == LAND, (i, a)
    n_open = sum(1 for _, e, _ in p.log if e == "hold_window_open")
    assert n_open == 1, p.log
    # Sustained HEALTHY: LAND holds while the buckets drain (>=19 s),
    # then releases — to HOVER on the release step, CONTINUE after.
    rel = []
    for i in range(50):
        a, _ = p.evaluate(1.0, 1.0, 0.0, False, 0.5,
                          pose_trustworthy=False, t=130.0 + i * 0.5,
                          link_report=_rep(lh.HEALTHY))
        rel.append(a)
    first = next(i for i, a in enumerate(rel) if a != LAND)
    assert first * 0.5 >= 19.0, (first, rel)
    assert rel[first] == HOVER, rel[first - 2:first + 3]
    assert rel[first + 1] == CONTINUE, rel[first - 2:first + 3]
    assert any(e[1] == "commit_released" for e in p.log), p.log
    print("policy N2 regression: one DEGRADED report does not release "
          "a committed LAND; sustained HEALTHY releases after ~20 s "
          "of bucket drain, stepping down to HOVER (N2 fixed)")


def test_policy_breakeven_no_wrong_commit():
    """Red-team round-2 wrong-commit search, re-run as a regression:
    LOST/HEALTHY cycles at or just above the buckets' 33% leak
    break-even — including 55% duty (11 s blackouts every 20 s for
    600 s) — must NEVER commit RTH. The only shape here that commits
    is the 20 s continuous run, which commits inside its first run
    (20 s continuous LOST, by design)."""
    for lost_s, ok_s in ((3, 7), (4, 8), (11, 19), (6, 11), (11, 9)):
        n_l, n_o = int(lost_s / 0.5), int(ok_s / 0.5)
        seq = ([lh.LOST] * n_l + [lh.HEALTHY] * n_o) * (
            600 // (lost_s + ok_s))
        _, actions = _run_policy(seq, pose=True)
        acts = [a for a, _ in actions]
        assert RTH not in acts, (lost_s, ok_s)
    seq = ([lh.LOST] * 40 + [lh.HEALTHY] * 20) * 20  # 20 s / 10 s
    _, actions = _run_policy(seq, pose=True)
    acts = [a for a, _ in actions]
    assert RTH in acts  # 20 s continuous LOST commits by design
    print("policy break-even: 30-55% LOST duty with HEALTHY gaps never "
          "commits in 600 s; 20 s continuous run commits (no new "
          "wrong commits)")


if __name__ == "__main__":
    test_tracker_nominal_healthy()
    test_tracker_lost_timing_and_recovery()
    test_dropout_duration_accounting()
    test_degraded_classification()
    test_latency_application()
    test_loss_application_and_no_leakage()
    test_markov_seeded_determinism()
    test_policy_short_dropout_no_escalation()
    test_policy_long_dropout_escalates_gated()
    test_policy_long_dropout_rth_when_trusted()
    test_policy_degraded_sustained_hover_and_recovery()
    test_policy_lost_recovery_cancels_pending()
    test_policy_stub_without_report()
    test_policy_flap_degraded_escalates()
    test_policy_p1b_no_cumulative_rth()
    test_policy_padded_dropouts_no_spurious_rth()
    test_policy_p3_flap_90pct_degraded_hovers()
    test_policy_sustained_lost_commit_timing()
    test_fullpath_flap_escalates()
    test_policy_committed_land_is_sticky()
    test_tracker_loss_clamped_and_rng_isolation()
    test_policy_n1_intermittent_lost_commits()
    test_policy_n1_stray_packet_eventually_commits()
    test_policy_n2_committed_land_survives_single_degraded()
    test_policy_breakeven_no_wrong_commit()
    print("ALL LINK-HEALTH TESTS PASS")
