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
    # MEASURED soc channel: countdown must halve — proof the sizing
    # reads the measurement, not a fixed model percentage.
    cds = []
    for rate in (0.001, 0.002):
        c = bs.BatterySafetyController((0.0, 0.0, 8.0))
        soc = lambda k, r=rate: 0.30 - r * k  # noqa: E731
        # fly above reserve until the meter is ready, then cross it
        _feed(c, 0, 99, soc)
        a, info = None, None
        for k in range(100, 140):
            c.observe(float(k), soc(k), 0.08 * k, (0.0, 0.0, 8.0), True)
            a, info = c.step(float(k), soc(k), (0.0, 0.0, 8.0))
            if c.countdown_s is not None:
                break
        assert c.flag_fired and c.countdown_s is not None, info
        cds.append(c.countdown_s)
    assert abs(cds[0] / cds[1] - 2.0) < 0.35, cds
    # And the slow-drain countdown matches (flag_soc-CRITICAL)/rate
    assert abs(cds[0] - (0.20 - CRITICAL_SOC) / 0.001) < 25.0, cds
    print(f"countdown from measured rate: {cds[0]:.0f}s @0.001/s vs "
          f"{cds[1]:.0f}s @0.002/s (ratio ~2)")


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
    # Start an RTB leg 20 m from base at gentle drain (affordable).
    c, a, _ = _ctrl_at_flag((20.0, 0.0, 8.0), rate=0.001)
    assert c.rung == bs.RTB
    pose = (20.0, 0.0, 8.0)
    soc = c.flag_soc
    # Measured drain worsens 10x mid-leg (load spike in the telemetry):
    # after the meter window refills, RTB must abort DOWN. The frozen
    # checkpoint sits at (10,0,8), 10 m away — also unaffordable at
    # the spiked rate given the floor, so the cascade lands.
    wh = c.meter.samples[-1][2]
    t = int(c.meter.samples[-1][0])
    actions = []
    for k in range(1, 30):
        t += 1
        soc -= 0.010
        wh += 0.8
        c.observe(float(t), max(soc, 0.11), wh, pose, False)
        a, info = c.step(float(t), max(soc, 0.11), pose)
        actions.append(a)
        if a == bs.LAND_IN_PLACE:
            break
    assert c.n_abort_down >= 1, c.events
    assert a == bs.LAND_IN_PLACE, actions
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
    # ~0.135 soc/s, so the countdown is short and terminal follows.
    assert c.countdown_s is not None and 0.0 < c.countdown_s < 5.0, \
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
    test_hold_metering_bounded_flat_soc_screaming_wh()
    test_hold_metering_bound_both_channels_dead()
    print("ALL BATTERY-SAFETY TESTS PASS")
