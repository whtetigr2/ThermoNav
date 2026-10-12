"""Tests for battery_safety.py — Paul's decided design (2026-10-11).

  1. Meter reads MEASURED draw: a BatteryModel stepped at hover
     reports ~hover power back through consumed Wh, and a faster
     soc drain yields a proportionally shorter countdown (sizing
     follows the measurement, not the model's soc percentage).
  2. Countdown: sized from the measured drain rate to reach
     CRITICAL; at countdown end the action is LAND_TERMINAL.
  3. Checkpoint: refreshes only while trustworthy, freezes on loss.
  4. Cascade rungs: RTB when base affordable; RTC when only the
     checkpoint is; LAND_IN_PLACE when neither is.
  5. Mid-leg abort DOWN only: RTB -> RTC -> LAND as the measured
     drain worsens; a rung never upgrades.
  6. Calibration log rows are produced during the countdown.
  7. RT2-1: HOLD_METERING is bounded — a flat soc with a screaming
     Wh channel ends the hold via the disagreement cross-check
     (countdown sized from the Wh channel), and a fully dead gauge
     terminates at the time bound.
"""
import battery_safety as bs
from battery import BatteryModel, RESERVE_SOC, CRITICAL_SOC


def _feed(ctrl, t0, t1, soc_fn, pose=(0.0, 0.0, 8.0), trust=True,
          wh_rate=0.08):
    """Drive observe()+step() over [t0, t1] at 1 Hz with a scripted
    soc trajectory (the gauge telemetry a real meter would report).
    wh_rate: Wh per second consumed (measured-draw channel)."""
    out = None
    wh = 0.0
    for k in range(int(t0), int(t1) + 1):
        wh += wh_rate
        ctrl.observe(float(k), soc_fn(k), wh, pose, trust)
        out = ctrl.step(float(k), soc_fn(k), pose)
    return out


def test_meter_measures_hover_draw():
    m = BatteryModel()
    meter = bs.MeasuredDrawMeter()
    p = m.power_for_phase("hover")
    for k in range(31):
        st = m.step(1.0, p)
        meter.add(float(k), st["soc"], m.wh_used)
    assert meter.ready
    w = meter.measured_draw_w()
    assert abs(w - p) / p < 0.02, (w, p)
    rate = meter.drain_rate_soc_per_s()
    assert rate > 0
    print(f"meter: measured {w:.1f} W vs stepped {p:.1f} W; "
          f"drain {rate:.6f} soc/s")


def test_countdown_sized_from_measured_rate():
    # Two controllers, same flag soc, drain rates differing 2x via the
    # MEASURED soc channel. With the descent reserve priced in
    # (whiteout fix), the sizing must match the formula exactly:
    # countdown = (soc_at_sizing - CRITICAL)/rate - descent_duration,
    # where descent_duration = altitude / 0.5 m/s = 16 s at z=8.
    # (The pre-fix "countdown halves when rate doubles" no longer
    # holds exactly: the reserve is rate*duration, so the countdown
    # is the reserve-band time MINUS a rate-independent duration.)
    for rate in (0.001, 0.002):
        c = bs.BatterySafetyController((0.0, 0.0, 8.0))
        soc = lambda k, r=rate: 0.30 - r * k  # noqa: E731
        soc_at_size = None
        for k in range(0, 200):
            c.observe(float(k), soc(k), 0.08 * k, (0.0, 0.0, 8.0), True)
            a, info = c.step(float(k), soc(k), (0.0, 0.0, 8.0))
            if c.countdown_s is not None:
                soc_at_size = soc(k)
                break
        assert c.flag_fired and c.countdown_s is not None, info
        expected = (soc_at_size - CRITICAL_SOC) / rate - 16.0
        assert abs(c.countdown_s - expected) < 2.0, (
            c.countdown_s, expected, rate)
        print(f"countdown from measured rate @{rate}/s: "
              f"{c.countdown_s:.0f}s (formula {expected:.0f}s incl. "
              f"16s descent reserve)")


def test_descent_reserve_and_envelope_boundaries():
    # Whiteout fix: reserve = measured rate x expected duration;
    # Tier A bound = CRITICAL + reserve (normal rate); Tier B bound
    # RE-STATED in fix round 2 (RT round 2 measured the round-1
    # flag-fire pricing false in the bottom of the band): primary
    # form on TERMINAL-START soc, r * whole descent steps; flag-fire
    # form adds the metering-hold allowance.
    c = bs.BatterySafetyController((0.0, 0.0, 8.5))
    pose = (0.0, 0.0, 8.5)
    r = 0.0045  # saturated drain, soc/s (RT-measured band)
    assert abs(c.descent_duration_s(pose) - 17.0) < 1e-9
    assert abs(c.descent_duration_s(pose, emergency=True) - 8.5) < 1e-9
    assert abs(c.descent_reserve_soc(pose, r) - 0.0765) < 1e-9
    assert abs(c.required_terminal_start_soc(pose, r)
               - (CRITICAL_SOC + 0.0765)) < 1e-9
    # Step granularity: 8.5 m at 1.0 m/s is spent as 9 whole 1 s
    # steps (measured, RT round 2); whole altitudes stay whole.
    assert c.tier_b_descent_steps(pose) == 9
    assert c.tier_b_descent_steps((0.0, 0.0, 9.0)) == 9
    assert c.tier_b_descent_steps((0.0, 0.0, 8.0)) == 8
    assert c.tier_b_descent_steps((0.0, 0.0, 0.4)) == 1
    # Terminal-start bound: r * 9 steps (the round-1 continuous
    # form was r * 8.5 = 0.03825 — underpriced by half a step
    # before the hold is even counted).
    assert abs(c.min_terminal_start_soc_tier_b(pose, r)
               - 0.0045 * 9) < 1e-9
    # Flag-fire form on a fresh controller (flag not fired -> the
    # worst-case metering hold is priced): + r * METER_MIN_SPAN_S.
    assert abs(c.min_flag_soc_tier_b(pose, r)
               - (0.0045 * 9 + 0.0045 * bs.METER_MIN_SPAN_S)) < 1e-9
    # The RT sub-CRITICAL case (flag soc 0.09, z 8.5, saturated):
    # below Tier A, above Tier B -> emergency guarantee band.
    assert c.min_flag_soc_tier_b(pose, r) <= 0.09 \
        < c.required_terminal_start_soc(pose, r)
    print("envelope: Tier A >= 0.1765, Tier B terminal-start >= "
          "0.0405 / flag-fire >= 0.0495 at z=8.5, r=0.0045; RT "
          "subcrit 0.09 sits in Tier B")


def test_tier_b_bound_rt2_regression():
    # Fix round 2 regression for the RT round-2 break, at RT2's own
    # measured numbers (tierb_044, seeds 0/1/2 identical): flag soc
    # 0.03976, sizing-time r = 0.0042329, terminal start 0.03130,
    # z = 8.5. The round-1 bound (0.0359) wrongly put it INSIDE
    # Tier B with a predicted touchdown >= 0.0039; it touched down
    # at soc 0.0. The re-stated bounds must put it OUTSIDE (no
    # claim), and must put tierb_052 (flag 0.04775, r = 0.0042462,
    # terminal start 0.03926, landed 0.0056) INSIDE with a
    # prediction the measured touchdown beats.
    pose = (12.5, 19.5, 8.5)
    c = bs.BatterySafetyController((12.5, 19.5, 0.0))
    r44, flag44, term44 = 0.0042329, 0.03976, 0.03130
    assert c.min_flag_soc_tier_b(pose, r44) > flag44
    assert c.min_terminal_start_soc_tier_b(pose, r44) > term44
    r52, flag52, term52 = 0.0042462, 0.04775, 0.03926
    assert c.min_flag_soc_tier_b(pose, r52) <= flag52
    assert c.min_terminal_start_soc_tier_b(pose, r52) <= term52
    pred52 = term52 - c.min_terminal_start_soc_tier_b(pose, r52)
    assert 0.0 < pred52 <= 0.0056, pred52
    print("tier B restated bound: tierb_044 outside (no claim), "
          "tierb_052 inside with predicted floor "
          f"{pred52:.4f} <= measured 0.0056")


def test_tier_b_hold_allowance_meter_ready():
    # Fix round 2: the hold allowance is priced only when the flag
    # fires with no usable meter window. An in-flight flag fire
    # (meter window long since filled) prices hold = 0, so the
    # flag-fire and terminal-start bounds coincide.
    c = bs.BatterySafetyController((0.0, 0.0, 8.5))
    pose = (0.0, 0.0, 8.5)
    wh = 0.0
    for k in range(60):
        soc = 0.30 - 0.0045 * k  # crosses RESERVE 0.20 at k=23
        wh += 0.35
        c.observe(float(k), soc, wh, pose, False)
        c.step(float(k), soc, pose)
        if c.flag_fired:
            break
    assert c.flag_fired and c.meter_ready_at_flag
    r = c.meter.drain_rate_soc_per_s()
    assert c.tier_b_hold_allowance_soc(r) == 0.0
    assert abs(c.min_flag_soc_tier_b(pose, r)
               - c.min_terminal_start_soc_tier_b(pose, r)) < 1e-12
    # And the scripted-start case: flag on the very first step,
    # meter empty -> not ready at flag, full hold priced.
    c2 = bs.BatterySafetyController((0.0, 0.0, 8.5))
    c2.observe(0.0, 0.044, 0.0, pose, False)
    c2.step(0.0, 0.044, pose)
    assert c2.flag_fired and not c2.meter_ready_at_flag
    assert abs(c2.tier_b_hold_allowance_soc(0.0045)
               - 0.0045 * bs.METER_MIN_SPAN_S) < 1e-12
    print("tier B hold allowance: 0 for in-flight flag fire, "
          "r*METER_MIN_SPAN_S for scripted start")


def test_subcritical_flag_arms_emergency_descent():
    # Whiteout fix 2a: flag fires already below CRITICAL (soc 0.09
    # trajectory), countdown is 0, terminal is immediate AND the
    # emergency descent rate is armed with the event logged.
    c = bs.BatterySafetyController((0.0, 0.0, 8.5))
    pose = (0.0, 0.0, 8.5)
    wh = 0.0
    action = None
    for k in range(30):
        soc = 0.13 - 0.0045 * k
        wh += 0.35
        c.observe(float(k), soc, wh, pose, False)
        action, info = c.step(float(k), soc, pose)
        if action == bs.LAND_TERMINAL:
            break
    assert action == bs.LAND_TERMINAL, (action, c.events)
    assert c.countdown_s == 0.0, c.countdown_s
    assert c.emergency_descent, c.events
    assert info["terminal_descent_rate_mps"] == \
        bs.EMERGENCY_TERMINAL_DESCENT_RATE_MPS
    assert any(e[1] == "emergency_terminal_descent" for e in c.events)
    print("sub-CRITICAL flag: countdown 0, terminal immediate, "
          "emergency descent armed at 1.0 m/s")


def test_granularity_dip_does_not_arm_emergency():
    # A normally-sized countdown end starts the descent up to ~1
    # step of drain below the required soc; that dip must NOT arm
    # the emergency rate (found in the first fix-round validation
    # run, where Tier A landings flew the emergency profile).
    c, a, _ = _ctrl_at_flag((5.0, 0.0, 8.0), rate=0.001)
    req = c.required_terminal_start_soc((5.0, 0.0, 8.0), 0.001)
    a, info = c.step(c.countdown_end_t + 1.0, req - 0.0005,
                     (5.0, 0.0, 8.0))
    assert a == bs.LAND_TERMINAL and not c.emergency_descent, c.events
    print("granularity dip (0.5 steps of drain below required): "
          "normal rate kept")


def test_reserve_covered_terminal_is_normal_rate():
    # A terminal start at/above the reserve-required soc must NOT
    # arm the emergency rate.
    c, a, _ = _ctrl_at_flag((5.0, 0.0, 8.0), rate=0.001)
    a, info = c.step(c.countdown_end_t + 1.0, 0.15, (5.0, 0.0, 8.0))
    assert a == bs.LAND_TERMINAL
    assert not c.emergency_descent, c.events
    assert info["terminal_descent_rate_mps"] == \
        bs.TERMINAL_DESCENT_RATE_MPS
    print("reserve-covered terminal: normal 0.5 m/s rate, no emergency")


def test_checkpoint_refresh_and_freeze():
    tr = bs.CheckpointTracker((1.0, 2.0, 8.0))
    assert tr.checkpoint_pose == (1.0, 2.0, 8.0)  # takeoff = base
    tr.update((5.0, 5.0, 8.0), True, 10.0)
    assert tr.checkpoint_pose == (5.0, 5.0, 8.0)
    tr.update((9.0, 9.0, 8.0), False, 11.0)  # trust lost: frozen
    assert tr.checkpoint_pose == (5.0, 5.0, 8.0) and tr.frozen
    tr.update((7.0, 7.0, 8.0), True, 12.0)   # trust back: refresh
    assert tr.checkpoint_pose == (7.0, 7.0, 8.0) and not tr.frozen
    print("checkpoint: refresh on trust, freeze on loss, base at t=0")


def _ctrl_at_flag(pose, base=(0.0, 0.0, 8.0), ckpt=(10.0, 0.0, 8.0),
                  rate=0.004, soc_at_flag=0.19):
    """Controller whose flag has just fired at `pose`, with a meter
    pre-filled at the given measured drain rate and a checkpoint
    frozen at `ckpt` (trust off so observe() cannot move it)."""
    c = bs.BatterySafetyController(base)
    # Pre-fill meter + checkpoint: fly trustworthy past the ckpt pose.
    wh = 0.0
    soc = 0.30
    for k in range(15):
        wh += rate * 5.0 * 74.0 / 3.6  # wh/s consistent-ish; meter only
        soc -= rate                     # needs the soc/wh channels
        c.observe(float(k), soc, wh, ckpt, True)
        c.step(float(k), soc, ckpt)
    # Now untrustworthy; move to `pose`, drain to the flag.
    for k in range(15, 400):
        wh += rate * 5.0 * 74.0 / 3.6
        soc -= rate
        c.observe(float(k), soc, wh, pose, False)
        a, info = c.step(float(k), soc, pose)
        if c.flag_fired and c.countdown_s is not None and c.rung:
            return c, a, info
    raise AssertionError("flag/cascade never decided")


def test_cascade_rtb_when_base_affordable():
    # pose 5 m from base, 15 m from checkpoint, gentle measured drain
    c, a, info = _ctrl_at_flag((5.0, 0.0, 8.0), rate=0.001)
    assert c.rung == bs.RTB and a == bs.RTB, (c.rung, a, info)
    assert c.target_pose == c.base_pose
    print(f"cascade RTB: base affordable (need "
          f"{c.energy_soc_needed((5,0,8), c.base_pose):.4f} soc)")


def test_cascade_rtc_when_only_checkpoint_affordable():
    # pose far from base (60 m) but 2 m from the frozen checkpoint,
    # steep measured drain: base leg unaffordable, checkpoint leg fits
    c, a, info = _ctrl_at_flag((60.0, 0.0, 8.0), ckpt=(58.0, 0.0, 8.0),
                               rate=0.004)
    assert c.rung == bs.RTC and a == bs.RTC, (c.rung, a, info)
    assert c.target_pose == (58.0, 0.0, 8.0)
    print("cascade RTC: base unaffordable, checkpoint affordable")


def test_cascade_land_when_neither_affordable():
    c, a, info = _ctrl_at_flag((80.0, 0.0, 8.0), rate=0.004)
    assert c.rung == bs.LAND_IN_PLACE and a == bs.LAND_IN_PLACE, (
        c.rung, a, info)
    print("cascade LAND: neither target affordable -> land in place")


def test_midleg_abort_down_only():
    # Start an RTB leg 40 m from base at gentle drain (affordable).
    # (Geometry widened in the whiteout fix round: at 20 m the
    # reserve-required terminal start — soc <= CRITICAL + reserve —
    # is now reached at about the same spiked rate as the abort
    # threshold, and the countdown-end terminal legitimately
    # preempts the abort re-check, since terminal outranks the
    # cascade inside its own window. At 40 m the leg goes
    # unaffordable well before the reserve start is reached, which
    # isolates the abort-down mechanism this test is about.)
    c, a, _ = _ctrl_at_flag((40.0, 0.0, 8.0), rate=0.001)
    assert c.rung == bs.RTB
    pose = (40.0, 0.0, 8.0)
    soc = c.flag_soc
    # Measured drain worsens 10x mid-leg (load spike in the telemetry):
    # after the meter window refills, RTB must abort DOWN.
    wh = c.meter.samples[-1][2]
    t = int(c.meter.samples[-1][0])
    for k in range(1, 30):
        t += 1
        soc -= 0.010
        wh += 0.8
        c.observe(float(t), max(soc, 0.11), wh, pose, False)
        a, info = c.step(float(t), max(soc, 0.11), pose)
        if c.n_abort_down >= 1:
            break
    assert c.n_abort_down >= 1, c.events
    assert bs._RUNG_ORDER[c.rung] > bs._RUNG_ORDER[bs.RTB], c.rung
    assert any(e[1] == "cascade_abort_down" for e in c.events)
    # Never up: drive it again with improving numbers — rung stays.
    a2, _ = c.step(float(t + 1), max(soc, 0.11), pose)
    assert bs._RUNG_ORDER[c.rung] >= bs._RUNG_ORDER[bs.RTC]
    print(f"abort-down: RTB -> {c.rung} after measured drain spike; "
          f"abort events={c.n_abort_down}")


def test_countdown_end_is_terminal():
    c, a, _ = _ctrl_at_flag((5.0, 0.0, 8.0), rate=0.001)
    assert c.countdown_end_t is not None
    a, info = c.step(c.countdown_end_t + 1.0, 0.15, (5.0, 0.0, 8.0))
    assert a == bs.LAND_TERMINAL and info["terminal"], (a, info)
    print("countdown end: LAND_TERMINAL (no veto outranks the flag)")


def test_countdown_resizes_earlier_on_drain_spike():
    """Validation round-2 negative (abort_spike at 52b3b3d): a load
    spike after sizing must pull the countdown end EARLIER — the
    meter keeps measuring — so the terminal landing fires while
    charge remains, not at soc 0."""
    c, a, _ = _ctrl_at_flag((5.0, 0.0, 8.0), rate=0.001)
    end0 = c.countdown_end_t
    pose = (5.0, 0.0, 8.0)
    soc = c.flag_soc
    wh = c.meter.samples[-1][2]
    t = int(c.meter.samples[-1][0])
    terminal_soc = None
    for k in range(1, 200):
        t += 1
        soc -= 0.005  # spiked measured drain (5x)
        wh += 0.4
        c.observe(float(t), max(soc, 0.0), wh, pose, False)
        a, info = c.step(float(t), max(soc, 0.0), pose)
        if a == bs.LAND_TERMINAL:
            terminal_soc = soc
            break
    assert any(e[1] == "countdown_resized_earlier" for e in c.events), \
        c.events
    assert c.countdown_end_t < end0, (c.countdown_end_t, end0)
    assert terminal_soc is not None and terminal_soc > 0.02, terminal_soc
    print(f"countdown re-size: end {end0:.0f} -> {c.countdown_end_t:.0f} "
          f"on drain spike; terminal at soc {terminal_soc:.3f}")


def test_calibration_log_rows():
    c, a, _ = _ctrl_at_flag((5.0, 0.0, 8.0), rate=0.001)
    assert len(c.calibration_log) >= 1
    row = c.calibration_log[-1]
    for col in ("t_s", "soc", "measured_draw_w", "drain_rate_soc_per_s",
                "countdown_s", "countdown_remaining_s", "action",
                "dist_to_base_m", "dist_to_checkpoint_m",
                "need_soc_base", "need_soc_checkpoint"):
        assert col in row, col
    assert row["measured_draw_w"] is not None
    print(f"calibration log: {len(c.calibration_log)} rows, cols ok")


def test_hold_metering_bounded_flat_soc_screaming_wh():
    """RT2-1 (red-team round 2): decisions read the soc channel;
    a flat soc at 0.19 with the Wh channel screaming parked the
    layer in HOLD_METERING for 600-1000 steps with no countdown
    and no terminal. The cross-check must now end the hold: the
    channels disagree, the countdown is sized from the Wh-implied
    drain rate, and the layer reaches a terminal in bounded time."""
    c = bs.BatterySafetyController((0.0, 0.0, 8.0))
    pose = (5.0, 0.0, 8.0)
    wh = 0.0
    actions = []
    terminal_t = None
    for k in range(120):
        wh += 10.0  # screaming Wh channel: 10 Wh/s = 36 kW reported
        c.observe(float(k), 0.19, wh, pose, True)
        a, info = c.step(float(k), 0.19, pose)
        actions.append(a)
        if a == bs.LAND_TERMINAL:
            terminal_t = k
            break
    n_hold = sum(1 for a in actions if a == bs.HOLD_METERING)
    assert c.flag_fired
    assert c.hold_bound_reason == "soc_wh_disagreement", (
        c.hold_bound_reason, c.events)
    assert any(e[1] == "soc_wh_disagreement" for e in c.events), c.events
    # Bounded: the disagreement fires as soon as the meter window
    # is ready (a few steps), NOT after hundreds of hold steps.
    assert n_hold <= 5, (n_hold, actions[:12])
    # Sized from the Wh channel: 10 Wh/s over a 74 Wh pack implies
    # ~0.135 soc/s; with the descent reserve priced in (whiteout
    # fix: reserve = 0.135 x 16 s = 2.16 soc at z=8) the countdown
    # is 0.0 and terminal is immediate — the honest answer at that
    # drain, not a short countdown to CRITICAL.
    assert c.countdown_s is not None and c.countdown_s == 0.0, \
        c.countdown_s
    assert terminal_t is not None and terminal_t <= 30, terminal_t
    print(f"RT2-1 disagreement: hold ended after {n_hold} steps, "
          f"countdown {c.countdown_s:.2f}s from Wh channel, "
          f"terminal at t={terminal_t}")


def test_hold_metering_bound_both_channels_dead():
    """RT2-1 backstop: soc flat AND Wh flat (a fully dead gauge).
    Nothing can be sized, so at the time bound the layer must
    terminate to the safest available action (LAND_TERMINAL)
    instead of holding forever."""
    c = bs.BatterySafetyController((0.0, 0.0, 8.0))
    pose = (5.0, 0.0, 8.0)
    terminal_t = None
    for k in range(int(bs.HOLD_METERING_MAX_S) + 30):
        c.observe(float(k), 0.19, 0.0, pose, True)
        a, info = c.step(float(k), 0.19, pose)
        if a == bs.LAND_TERMINAL:
            terminal_t = k
            break
        assert a == bs.HOLD_METERING, (k, a)
    assert terminal_t is not None, "hold never terminated"
    assert terminal_t <= bs.HOLD_METERING_MAX_S + 2, terminal_t
    assert c.hold_bound_reason == "hold_metering_bound", (
        c.hold_bound_reason, c.events)
    assert any(e[1] == "hold_metering_bound" for e in c.events), c.events
    print(f"RT2-1 dead gauge: HOLD_METERING bounded, terminal at "
          f"t={terminal_t} (bound {bs.HOLD_METERING_MAX_S:.0f}s)")


if __name__ == "__main__":
    test_meter_measures_hover_draw()
    test_countdown_sized_from_measured_rate()
    test_checkpoint_refresh_and_freeze()
    test_cascade_rtb_when_base_affordable()
    test_cascade_rtc_when_only_checkpoint_affordable()
    test_cascade_land_when_neither_affordable()
    test_midleg_abort_down_only()
    test_countdown_end_is_terminal()
    test_countdown_resizes_earlier_on_drain_spike()
    test_calibration_log_rows()
    test_descent_reserve_and_envelope_boundaries()
    test_tier_b_bound_rt2_regression()
    test_tier_b_hold_allowance_meter_ready()
    test_subcritical_flag_arms_emergency_descent()
    test_granularity_dip_does_not_arm_emergency()
    test_reserve_covered_terminal_is_normal_rate()
    test_hold_metering_bounded_flat_soc_screaming_wh()
    test_hold_metering_bound_both_channels_dead()
    print("ALL BATTERY-SAFETY TESTS PASS")
