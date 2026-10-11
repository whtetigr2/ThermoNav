"""Tests for the battery model (battery.py) + its escalation wiring.

Model tests are sanity-of-physics, not fit-to-data (there is no data —
simulation only, parameters calculated/ASSUMED per battery.py):
  1. Discharge is monotone; soc never goes negative or rises.
  2. Voltage sags more under heavier load, and as soc falls.
  3. Time-to-reserve decreases as draw increases; zero at the floor.
  4. Phase ordering: climb > cruise ~ hover > descent at our speeds.
  5. Hover power lands near the registry's class-typical band
     (cross-check, not calibration — see BATTERY_MODEL.md).
  6. Energy bookkeeping: Wh drawn over a full hover discharge matches
     the usable capacity within the model's own approximations.

Wiring tests (EscalationPolicy):
  7. No battery input -> legacy behavior; battery listed MISSING
     until inputs are supplied (ThermoNav convention, as for link).
  8. Warn band: logs once, no flight action.
  9. Reserve: RTH intent; gate closed -> LAND (loud); gate open -> RTH.
 10. Critical: immediate LAND, hold window bypassed, gate irrelevant.
 11. Most-severe-wins across sources (perception + battery).
"""
import battery as batt
from battery import BatteryModel
from escalation import (EscalationPolicy, CONTINUE, HOVER, RTH, LAND)


def test_discharge_monotone():
    m = BatteryModel()
    p = m.power_for_phase("hover")
    socs = []
    for _ in range(120):
        st = m.step(5.0, p)
        socs.append(st["soc"])
    assert all(b < a for a, b in zip(socs, socs[1:])), "not monotone"
    assert all(0.0 <= s <= 1.0 for s in socs)
    # Empty floor: stepping past empty clamps at 0, never negative.
    m2 = BatteryModel(start_soc=0.05)
    for _ in range(100):
        st = m2.step(60.0, p)
    assert st["soc"] == 0.0
    print(f"discharge monotone: 120 steps hover, soc 1.0 -> {socs[-1]:.3f}")


def test_sag_increases_with_load_and_discharge():
    m = BatteryModel()
    v_light_full = m.terminal_voltage(100.0, soc=1.0)
    v_heavy_full = m.terminal_voltage(400.0, soc=1.0)
    v_heavy_low = m.terminal_voltage(400.0, soc=0.3)
    assert v_light_full > v_heavy_full > v_heavy_low, (
        v_light_full, v_heavy_full, v_heavy_low)
    # Sag is exactly I*R at the solved operating point.
    i = m.current(282.0, soc=1.0)
    sag = m.ocv(1.0) - m.terminal_voltage(282.0, soc=1.0)
    assert abs(sag - i * batt.R_PACK) < 1e-9, (sag, i)
    print(f"sag: 100W@{v_light_full:.2f}V > 400W@{v_heavy_full:.2f}V "
          f"> 400W@soc0.3 {v_heavy_low:.2f}V; sag(282W)={sag:.2f}V = I*R")


def test_ttr_decreases_with_draw():
    m = BatteryModel()
    ttrs = [m.time_to_reserve_s(p) for p in (150.0, 282.0, 450.0)]
    assert ttrs[0] > ttrs[1] > ttrs[2] > 0, ttrs
    m_low = BatteryModel(start_soc=batt.RESERVE_SOC)
    assert m_low.time_to_reserve_s(282.0) == 0.0
    m_below = BatteryModel(start_soc=0.15)
    assert m_below.time_to_reserve_s(282.0) == 0.0
    print(f"TTR(full): 150W {ttrs[0]:.0f}s > 282W {ttrs[1]:.0f}s "
          f"> 450W {ttrs[2]:.0f}s; TTR(reserve)=0")


def test_phase_ordering_and_hover_band():
    m = BatteryModel()
    p_hover = m.power_for_phase("hover")
    p_cruise = m.power_for_phase("cruise", speed=2.0)
    p_climb = m.power_for_phase("climb", climb_rate=1.0)
    p_descent = m.power_for_phase("descent", climb_rate=0.5)
    assert p_climb > p_hover > p_descent, (p_climb, p_hover, p_descent)
    # Cruise at the 2 m/s operational cap sits near hover (momentum
    # theory power bucket; parasite drag is tiny at 2 m/s).
    assert abs(p_cruise - p_hover) < 0.1 * p_hover, (p_cruise, p_hover)
    # Cross-check vs registry: propulsion-only hover (total minus the
    # avionics lump) within +/-20% of the registry's 280-320 W
    # class-typical PROPULSION band. Wide bounds on purpose: this is a
    # sanity fence between two independent estimates, not a fit.
    p_prop = p_hover - batt.P_AVIONICS
    assert 0.8 * 280 <= p_prop <= 1.2 * 320, p_prop
    print(f"phases: climb {p_climb:.0f} > hover {p_hover:.0f} "
          f"(cruise {p_cruise:.0f}) > descent {p_descent:.0f} W; "
          f"propulsion-only {p_prop:.0f} W vs registry band 280-320 W")


def test_energy_bookkeeping():
    m = BatteryModel()
    p = m.power_for_phase("hover")
    # Discharge at constant power to empty; Wh drawn should approximate
    # the charge removed x mean terminal voltage (the model's own
    # identity), and land under the 74 Wh nameplate (sag + the linear
    # OCV curve spend the pack below nominal voltage on average).
    while m.soc > 0.0:
        m.step(10.0, p)
    assert 55.0 < m.wh_used < batt.CAPACITY_WH, m.wh_used
    print(f"energy: full hover discharge drew {m.wh_used:.1f} Wh "
          f"(nameplate {batt.CAPACITY_WH:.0f} Wh)")


def test_policy_battery_absent_is_legacy():
    # ThermoNav convention (same as the link layer): the battery stub
    # stays in missing_triggers until battery inputs are supplied;
    # with none supplied the policy is bit-for-bit the legacy build.
    pol = EscalationPolicy()
    a, info = pol.evaluate(1.0, 1.0, 0.0, False, dt=1.0)
    assert a == CONTINUE and info["battery_trigger"] is None, (a, info)
    assert info["missing_triggers"] == ["battery", "link", "nav_health"], \
        info
    # With battery inputs supplied, battery leaves the MISSING list.
    a, info = pol.evaluate(1.0, 1.0, 0.0, False, dt=1.0,
                           battery_soc=0.28, battery_ttr_s=400.0)
    assert info["missing_triggers"] == ["link", "nav_health"], info
    print("battery absent: legacy CONTINUE, missing lists battery until "
          "inputs supplied")


def test_policy_battery_warn_logs_once_no_action():
    pol = EscalationPolicy()
    for k in range(3):
        a, info = pol.evaluate(1.0, 1.0, 0.0, False, dt=1.0, t=float(k),
                               battery_soc=0.28, battery_ttr_s=400.0)
        assert a == CONTINUE, (a, info)
    warns = [e for e in pol.log if e[1] == "battery_warn"]
    assert len(warns) == 1, pol.log
    assert info["battery_trigger"] == "warn", info
    print("battery warn: CONTINUE held, battery_warn logged exactly once")


def test_policy_battery_reserve_rth_intent_and_gate():
    # Gate closed (the shipped default): reserve -> RTH intent -> LAND,
    # after the hold window, loudly gated.
    pol = EscalationPolicy()
    a, info = pol.evaluate(1.0, 1.0, 0.0, False, dt=1.0, t=0.0,
                           battery_soc=0.19, battery_ttr_s=0.0)
    assert a == HOVER and info["holding_for_override"], (a, info)
    a, info = pol.evaluate(1.0, 1.0, 0.0, False, dt=11.0, t=1.0,
                           battery_soc=0.19, battery_ttr_s=0.0)
    assert a == LAND and info["rth_gated"], (a, info)
    assert "battery reserve" in info["reason"], info
    assert any(e[1] == "rth_gated_land_in_place" for e in pol.log)
    # Gate open: the same reserve crossing yields real RTH.
    pol2 = EscalationPolicy()
    pol2.evaluate(1.0, 1.0, 0.0, False, dt=1.0, t=0.0,
                  pose_trustworthy=True,
                  battery_soc=0.19, battery_ttr_s=0.0)
    a, info = pol2.evaluate(1.0, 1.0, 0.0, False, dt=11.0, t=1.0,
                            pose_trustworthy=True,
                            battery_soc=0.19, battery_ttr_s=0.0)
    assert a == RTH and not info["rth_gated"], (a, info)
    print("battery reserve: gated->LAND (logged), open->RTH, "
          "hold window respected")


def test_policy_battery_critical_immediate_land():
    pol = EscalationPolicy()
    # First-ever evaluation, already critical: LAND on THIS step — no
    # hold window, and the RTH gate is irrelevant (action is LAND).
    a, info = pol.evaluate(1.0, 1.0, 0.0, False, dt=1.0, t=0.0,
                           battery_soc=0.09, battery_ttr_s=0.0)
    assert a == LAND, (a, info)
    assert not info["holding_for_override"], info
    assert info["battery_trigger"] == "critical", info
    assert any(e[1] == "battery_critical_immediate" for e in pol.log)
    print("battery critical: immediate LAND, hold window bypassed")


def test_policy_battery_most_severe_wins():
    # Perception wants HOVER (degraded 5 s), battery is at reserve:
    # RTH intent (gated -> LAND after window) outranks the hover.
    pol = EscalationPolicy()
    a, _ = pol.evaluate(0.5, 0.05, 0.0, False, dt=6.0, t=0.0,
                        battery_soc=0.19, battery_ttr_s=0.0)
    assert a == HOVER, a  # window: at hover level, pending RTH intent
    a, info = pol.evaluate(0.5, 0.05, 0.0, False, dt=11.0, t=6.0,
                           battery_soc=0.19, battery_ttr_s=0.0)
    assert a == LAND and info["rth_gated"], (a, info)
    print("most-severe-wins: battery reserve outranks perception hover")


def test_below_state_guard():
    """The LAND actuation guard's three states (mission loop): clear /
    unknown / occupied must be distinguished, because battery-critical
    forced landing relaxes UNKNOWN only, never OCCUPIED. The floor
    layer zz=0 is keyed on evidence beyond the boundary prior (RT-1):
    prior-only must not block (Finding 1), hit-supported must veto
    (red-team synth_low), and the z=1 endpoint must check zz=0."""
    import numpy as np
    import mission_dynamics_v5 as md
    shape = (6, 6, 6)
    marg = np.zeros(shape)
    obs = np.ones(shape, dtype=bool)
    logodds = np.zeros(shape)
    logodds[:, :, 0] = md.FLOOR_PRIOR_LOGODDS  # boundary prior
    pose = (2.5, 2.5, 3.5)
    assert md._below_state(pose, marg, obs, shape, logodds=logodds) == 0
    assert md._below_clear(pose, marg, obs, shape, logodds=logodds)
    obs[2, 2, 1] = False  # one unknown voxel in the column
    assert md._below_state(pose, marg, obs, shape, logodds=logodds) == 1
    assert not md._below_clear(pose, marg, obs, shape, logodds=logodds)
    obs[2, 2, 1] = True
    marg[2, 2, 1] = 0.9  # believed occupied below (zz>=1)
    assert md._below_state(pose, marg, obs, shape, logodds=logodds) == 2
    assert not md._below_clear(pose, marg, obs, shape, logodds=logodds)
    marg[2, 2, 1] = 0.0
    # Direction 1 (Finding 1 regression): floor voxels reading marg
    # 0.88 from the PRIOR ALONE (logodds exactly 2.0, no hit support)
    # are the landing surface — they must not block.
    marg[2, 2, 0] = 0.88
    marg[3, 3, 0] = 0.88
    assert md._below_state(pose, marg, obs, shape, logodds=logodds) == 0
    assert md._below_clear(pose, marg, obs, shape, logodds=logodds)
    # Direction 2 (RT-1 regression): a zz=0-only obstacle WITH hit
    # support (logodds pushed past the prior) vetoes again.
    logodds[2, 2, 0] = 2.0 + 0.60  # one confirmed hit beyond the prior
    assert md._below_state(pose, marg, obs, shape, logodds=logodds) == 2
    assert not md._below_clear(pose, marg, obs, shape, logodds=logodds)
    # Endpoint hole (RT-1 case D): at voxel z=1 the checked set is
    # {zz=0} — hit-supported floor occupancy still vetoes, and
    # prior-only floor reads as the surface (state 0), never vacuous
    # in the occupied direction.
    pose_low = (2.5, 2.5, 1.5)
    assert md._below_state(pose_low, marg, obs, shape,
                           logodds=logodds) == 2
    logodds[2, 2, 0] = 2.0
    assert md._below_state(pose_low, marg, obs, shape,
                           logodds=logodds) == 0
    print("below-state guard: clear=0, unknown=1, occupied=2; floor "
          "prior-only=surface, floor hit-supported=veto, z=1 endpoint "
          "checks zz=0")


def test_floor_evidence():
    """RT-1 creep gate: a footprint whose floor voxels all still read
    exactly the boundary prior has never been sensed (unverified);
    every voxel off-prior (hit OR miss) counts as evidence."""
    import numpy as np
    import mission_dynamics_v5 as md
    shape = (8, 8, 6)
    logodds = np.zeros(shape)
    logodds[:, :, 0] = md.FLOOR_PRIOR_LOGODDS
    pose = (3.5, 3.5, 4.5)
    assert not md._floor_evidence(pose, logodds, shape)
    assert not md._floor_evidence(pose, None, shape)
    logodds[2:5, 2:5, 0] = 1.27  # misses pulled the footprint down
    assert md._floor_evidence(pose, logodds, shape)
    logodds[3, 3, 0] = md.FLOOR_PRIOR_LOGODDS  # one voxel back to prior
    assert not md._floor_evidence(pose, logodds, shape)
    print("floor evidence: all-prior=unverified, all-off-prior=seen, "
          "one prior voxel=unverified")


def test_ttr_stepped_matches_brute_force():
    """RT-5: stepped TTR must track a fine brute-force discharge to
    the reserve floor within 2% (the one-shot formula was +16-21%
    optimistic from full charge)."""
    for p in (282.5, 450.0):
        m = BatteryModel()
        est = m.time_to_reserve_s(p)
        brute = BatteryModel()
        t = 0.0
        while brute.soc > batt.RESERVE_SOC:
            brute.step(0.5, p)
            t += 0.5
        assert abs(est - t) / t < 0.02, (p, est, t)
        print(f"TTR @{p}W: stepped {est:.0f}s vs brute-force {t:.0f}s")


def test_saturation_bookkeeping():
    """RT-7: beyond the max-power point, wh_used books DELIVERED power
    (ocv^2/4R), not requested, so Wh and charge stay consistent."""
    m = BatteryModel()
    st = m.step(10.0, 5000.0)  # far past the max-power point
    assert m.saturated
    ocv = m.ocv(1.0)
    p_max = ocv * ocv / (4.0 * batt.R_PACK)
    assert abs(m.wh_used - p_max * 10.0 / 3600.0) < 1e-9, (
        m.wh_used, p_max)
    assert st["soc"] < 1.0
    print(f"saturation: requested 5000W, booked delivered {p_max:.0f}W, "
          f"wh_used={m.wh_used:.3f}")


if __name__ == "__main__":
    test_discharge_monotone()
    test_sag_increases_with_load_and_discharge()
    test_ttr_decreases_with_draw()
    test_phase_ordering_and_hover_band()
    test_energy_bookkeeping()
    test_policy_battery_absent_is_legacy()
    test_policy_battery_warn_logs_once_no_action()
    test_policy_battery_reserve_rth_intent_and_gate()
    test_policy_battery_critical_immediate_land()
    test_policy_battery_most_severe_wins()
    test_below_state_guard()
    test_floor_evidence()
    test_ttr_stepped_matches_brute_force()
    test_saturation_bookkeeping()
    print("ALL BATTERY TESTS PASS")
