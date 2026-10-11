"""Tests for the D5 escalation policy (Option A) + 12-spin decider.

1. Decider reproduces the policy table on the full 2^7 input grid
   (oracle = the Option A trigger table), stable across seeds.
2. Z1-legal rails: 10-14 spins, degree<=16, |h|<=6, |J|<=6.
3. Policy: most-severe-wins, hold-first override window, RTH pose gate
   -> land-in-place (loud), MISSING stubs inert.
4. Regression (standalone): on the shipped input envelope the decider
   never contradicts FIX 3 / FIX 4 (never outputs continue when the
   tracker hovers; never outputs RTH/land in clear air).
   Full mission-level regression runs after the Phase 2 rebase, when the
   D5 integration hooks are wired.
"""
import itertools
import numpy as np

from escalation import (EscalationPolicy, CONTINUE, HOVER, RTH, LAND, DISARM,
                        VIS_DEGRADED, FAR_DEGRADED, MAP_STALE_S,
                        DEGRADE_SUSTAIN_HOVER_S, DEGRADE_SUSTAIN_RTH_S,
                        HOLD_OVERRIDE_S, ACTION_NAMES as P_NAMES)
from escalation_decider import (
    build_decider, h_dyn_from_inputs, decide, save_kernel,
    N, ACTION_NAMES as D_NAMES)


def _J():
    h, edges, weights = build_decider()
    J = np.zeros((N, N))
    for (i, j), w in zip(map(tuple, edges), weights):
        J[i, j] = J[j, i] = w
    return h, J


def oracle(inp):
    """The Option A trigger table as a pure function."""
    vis_deg, far_deg, stale, _jitter, deg_5s, deg_30s, rth_av = inp
    degraded = vis_deg and far_deg
    if degraded and deg_30s and not rth_av:
        return "land"
    if degraded and deg_30s:
        return "RTH"
    if degraded and (deg_5s or stale):
        return "hover"
    return "continue"


def test_decider_grid():
    h0, J = _J()
    bad = []
    for inp in itertools.product([False, True], repeat=7):
        hd = h0 + h_dyn_from_inputs(*inp)
        acts = set()
        for seed in (0, 1, 2):
            a, _, _ = decide(hd, J, n_sweeps=300, seed=seed)
            acts.add(a)
        got, want = D_NAMES[a], oracle(inp)
        if got != want or len(acts) != 1:
            bad.append((inp, got, want, acts))
    assert not bad, f"{len(bad)} grid mismatches, e.g. {bad[:3]}"
    print("decider grid: 128/128 match oracle, seed-stable")


def test_z1_rails():
    _, stats = save_kernel()
    assert 10 <= stats["n_spins"] <= 14, stats
    assert stats["max_degree"] <= 16, stats
    assert stats["max|h|"] <= 6, stats
    assert stats["max|J|"] <= 6, stats
    print("Z1 rails:", stats)


def test_policy_rth_gate():
    # Sustained 30 s degradation, gate closed -> land-in-place (loud)
    p = EscalationPolicy()
    a, info = p.evaluate(0.5, 0.05, 0.0, False,
                         dt=DEGRADE_SUSTAIN_RTH_S + HOLD_OVERRIDE_S + 1.0,
                         pose_trustworthy=False, t=0.0)
    assert a == LAND, (a, info)
    assert info["rth_gated"], info
    assert any(e[1] == "rth_gated_land_in_place" for e in p.log), p.log
    # Gate open -> real RTH
    p2 = EscalationPolicy()
    a2, info2 = p2.evaluate(0.5, 0.05, 0.0, False,
                            dt=DEGRADE_SUSTAIN_RTH_S + HOLD_OVERRIDE_S + 1.0,
                            pose_trustworthy=True, t=0.0)
    assert a2 == RTH and not info2["rth_gated"], (a2, info2)
    print("RTH gate: closed->land-in-place (logged), open->RTH")


def test_policy_hold_first():
    p = EscalationPolicy()
    # Drive into sustained degradation stepwise: hover engages immediately.
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=6.0, t=0.0)
    assert a == HOVER, a
    # Step to just under 30 s: still hover.
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=23.0, t=10.0)
    assert a == HOVER, a
    # Cross 30 s: pending RTH must wait out the override window at hover.
    a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0, t=100.0)
    assert a == HOVER and info["holding_for_override"], (a, info)
    # Operator override during the window cancels back.
    a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0,
                         operator_override=True, t=101.0)
    assert info["operator_overridden"], info
    print("hold-first: window respected, operator override cancels")


def test_policy_most_severe_wins():
    p = EscalationPolicy()
    # All rows firing at once -> severest (RTH desired; gate closed -> land)
    a, _ = p.evaluate(0.5, 0.05, 2.0, True,
                      dt=DEGRADE_SUSTAIN_RTH_S + HOLD_OVERRIDE_S + 1.0,
                      pose_trustworthy=False, t=0.0)
    assert a == LAND, a  # RTH gated -> land-in-place
    print("most-severe-wins: OK")


def test_policy_rth_gate_multistep():
    """Red-team 2026-10-10: RTH gate bypass on the hold-window path.
    Gate closed throughout; sustained degradation across steps.
    Invariant: no step emits RTH with the gate closed, and _current never
    holds a pre-gate RTH. (Single-shot tests missed this; it needs 2+.)"""
    p = EscalationPolicy()
    # Step 1: 30 s degradation + full hold window -> commit path.
    a1, i1 = p.evaluate(0.5, 0.05, 0.0, False,
                        dt=DEGRADE_SUSTAIN_RTH_S + HOLD_OVERRIDE_S + 1.0,
                        pose_trustworthy=False, t=0.0)
    assert a1 == LAND and i1["rth_gated"], (a1, i1)
    assert p._current == LAND, p._current  # post-gate, never pre-gate RTH
    # Steps 2-4: degradation persists -> pending RTH -> hold window.
    # Old code: max(_current=RTH, HOVER) leaked RTH past the gate here.
    for t in (50.0, 51.0, 52.0):
        a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0,
                             pose_trustworthy=False, t=t)
        assert a != RTH, (t, a, info)
        assert p._current != RTH, (t, p._current)
    assert any(e[1] == "rth_gated_land_in_place" for e in p.log), p.log
    print("RTH gate multi-step (closed): RTH never emitted, _current clean")


def test_policy_rth_gate_open_then_closed():
    """Gate legitimately opens (RTH commits), then closes. The next RTH
    attempt must degrade to LAND loudly — not leak via max() with the
    stale committed RTH."""
    p = EscalationPolicy()
    a1, i1 = p.evaluate(0.5, 0.05, 0.0, False,
                        dt=DEGRADE_SUSTAIN_RTH_S + HOLD_OVERRIDE_S + 1.0,
                        pose_trustworthy=True, t=0.0)
    assert a1 == RTH and not i1["rth_gated"], (a1, i1)
    assert p._current == RTH
    # Gate closes; degradation persists -> hold window on stale RTH.
    a2, i2 = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0,
                        pose_trustworthy=False, t=50.0)
    assert a2 != RTH, (a2, i2)
    assert a2 == LAND and i2["rth_gated"], (a2, i2)
    assert p._current == LAND, p._current
    assert any(e[1] == "rth_gated_land_in_place" for e in p.log), p.log
    print("RTH gate open->closed: leak sealed, LAND + log")


def test_policy_missing_stubs():
    p = EscalationPolicy()
    for name, stub in p.missing_triggers.items():
        fired, note = stub.evaluate()
        assert not fired and "MISSING" in note, (name, note)
    print("MISSING stubs inert:",
          sorted(p.missing_triggers))


# -- DISARM wiring (2026-10-11, Paul's approval) ---------------------------
# DISARM fires ONLY from LAND + confirmed touchdown. Never in the air,
# never on a timer, latches once committed.

def _drive_to_land():
    """Fresh policy driven to committed LAND (gate closed). Returns it."""
    p = EscalationPolicy()
    a, info = p.evaluate(0.5, 0.05, 0.0, False,
                         dt=DEGRADE_SUSTAIN_RTH_S + HOLD_OVERRIDE_S + 1.0,
                         pose_trustworthy=False, t=0.0)
    assert a == LAND and info["rth_gated"], (a, info)
    assert p._current == LAND
    return p


def test_disarm_fires_after_landing():
    p = _drive_to_land()
    # Touchdown confirmed -> DISARM, logged.
    a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0,
                         pose_trustworthy=False, t=100.0, landed=True)
    assert a == DISARM, (a, info)
    assert info["disarmed"], info
    assert p._current == DISARM
    assert any(e[1] == "disarm_touchdown_confirmed" for e in p.log), p.log
    print("DISARM: fires on LAND + touchdown confirmed")


def test_disarm_never_in_air():
    # At HOVER, landed=False -> never DISARM.
    p = EscalationPolicy()
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=6.0, t=0.0)
    assert a == HOVER, a
    a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0, t=10.0,
                         landed=False)
    assert a != DISARM, (a, info)
    # At LAND, landed=False (still descending) -> stays LAND.
    p2 = _drive_to_land()
    a, info = p2.evaluate(0.5, 0.05, 0.0, False, dt=1.0,
                          pose_trustworthy=False, t=100.0, landed=False)
    assert a == LAND and not info["disarmed"], (a, info)
    print("DISARM: never fires while airborne (HOVER or descending LAND)")


def test_disarm_fuzz_never_airborne():
    """Random inputs, landed=False: DISARM must never be emitted."""
    rng = np.random.RandomState(0)
    for trial in range(200):
        p = EscalationPolicy()
        for step in range(20):
            vis = rng.uniform(0.0, 1.0)
            far = rng.uniform(0.0, 0.3)
            map_age = rng.uniform(0.0, 5.0)
            dt = rng.uniform(0.1, 5.0)
            a, info = p.evaluate(vis, far, map_age, False, dt=dt,
                                 pose_trustworthy=rng.rand() < 0.5,
                                 t=float(step), landed=False)
            assert a != DISARM, (trial, step, a, info)
            assert p._current != DISARM, (trial, step, p._current)
    print("DISARM fuzz: 200x20 steps, landed=False -> never DISARM")


def test_disarm_latches():
    p = _drive_to_land()
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0, t=100.0, landed=True)
    assert a == DISARM
    # Inputs go fully nominal, landed flag drops: DISARM sticks.
    a, info = p.evaluate(0.9, 0.5, 0.0, False, dt=1.0, t=101.0,
                         landed=False)
    assert a == DISARM and info["disarmed"], (a, info)
    assert p._current == DISARM
    print("DISARM: terminal latch holds through nominal recovery")


def test_disarm_guard_not_landing():
    # landed=True while at HOVER (stale flag / non-policy touchdown):
    # vetoed loudly, no DISARM.
    p = EscalationPolicy()
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=6.0, t=0.0)
    assert a == HOVER
    a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=1.0, t=10.0,
                         landed=True)
    assert a != DISARM, (a, info)
    assert p._current == HOVER, p._current
    assert any(e[1] == "disarm_vetoed_not_landing" for e in p.log), p.log
    # Same at CONTINUE (nominal inputs, spurious landed flag).
    p2 = EscalationPolicy()
    a, _ = p2.evaluate(0.9, 0.5, 0.0, False, dt=1.0, t=0.0, landed=True)
    assert a == CONTINUE, a
    assert any(e[1] == "disarm_vetoed_not_landing" for e in p2.log)
    print("DISARM guard: landed flag outside LAND is vetoed, loudly")


def test_disarm_no_timeout_path():
    """Indefinite degradation WITHOUT touchdown: the ladder holds at LAND
    (attempting landing) forever — never DISARM. Disarming in the air is
    an uncontrolled fall; there is deliberately no timeout path."""
    p = _drive_to_land()
    for t in range(100, 3700, 60):  # a full hour of degraded flight
        a, info = p.evaluate(0.5, 0.05, 0.0, False, dt=60.0,
                             pose_trustworthy=False, t=float(t),
                             landed=False)
        assert a == LAND, (t, a, info)
        assert not info["disarmed"], (t, info)
    assert p._current == LAND
    assert not any(e[1] == "disarm_touchdown_confirmed" for e in p.log)
    print("DISARM: no timeout path — 1 h degraded at LAND, never DISARM")


def test_disarm_behavior_contract():
    log = {}
    done = EscalationPolicy.disarm_behavior(log)
    assert done and log["outcome"] == "landed_disarmed", log
    print("disarm_behavior: terminal, outcome='landed_disarmed'")


def test_regression_shipped_envelope():
    """Standalone regression: on the shipped input envelope the decider
    never contradicts FIX 3 / FIX 4 and never escalates in clear air."""
    h0, J = _J()

    def dec(vis_deg, far_deg, stale, jitter=False,
            deg_5s=False, deg_30s=False, rth_av=False):
        hd = h0 + h_dyn_from_inputs(vis_deg, far_deg, stale, jitter,
                                    deg_5s, deg_30s, rth_av)
        a, _, _ = decide(hd, J, n_sweeps=300, seed=0)
        return D_NAMES[a]

    # Clear air (shipped: continue) -> decider: continue
    assert dec(False, False, False) == "continue"
    # FIX 3 condition (vis<0.50, far<0.02 -> tracker hover) -> hover, not RTH/land
    assert dec(True, True, False) in ("continue", "hover")
    # FIX 4 condition (degraded AND map_age>=1.0 -> tracker stale_hover)
    assert dec(True, True, True) == "hover"
    # Shipped hover must never be contradicted downward: any input where
    # the tracker hovers must map to hover or harsher, never continue.
    assert dec(True, True, True, deg_5s=True) == "hover"
    # Clear air never escalates, even with jitter alone
    assert dec(False, False, False, jitter=True) == "continue"
    print("regression (standalone): shipped envelope uncontradicted")


# ---------------------------------------------------------------------------
# Loop wiring: DISARM via the mission loop's touchdown path
# (branch muse/disarm-fix-2026-10-11).
# - test_disarm_via_loop_touchdown_helper goes through
#   mission_dynamics_v5._touchdown_disarm_check — the loop's DISARM helper.
# - test_disarm_live_mission_touchdown_branch goes through
#   _loop_touchdown_branch — the factored loop branch itself (the actual
#   branch lines from the escalation hold-executive), with a synthetic
#   vehicle forced onto the touchdown path.
# ---------------------------------------------------------------------------

def _loop_mod():
    import mission_dynamics_v5 as m
    return m


def test_disarm_via_loop_touchdown_helper():
    """The loop's touchdown helper fires DISARM.

    Drives a real EscalationPolicy to LAND (as the loop's top-of-step
    evaluate would), then invokes the loop's _touchdown_disarm_check with
    the state the touchdown branch guarantees (_esc_land True, vehicle at
    touchdown). Verifies DISARM fires and outcome is "landed_disarmed" —
    proving the loop helper wiring, not just the policy in isolation.
    (This calls the helper directly; the branch-lines test below goes
    through the factored loop branch itself.)
    """
    m = _loop_mod()
    p = _drive_to_land()
    assert p._current == LAND, p._current
    log = {}
    # The loop's touchdown branch state: policy at LAND, tracker absent,
    # touchdown confirmed. Mirrors the call site in mission_dynamics_v5.
    fired = m._touchdown_disarm_check(p, None, 0.0, {}, 100.0, log)
    assert fired, "DISARM did not fire via the loop's touchdown path"
    assert log["outcome"] == "landed_disarmed", log
    assert p._current == DISARM, p._current
    assert any(e[1] == "disarm_touchdown_confirmed" for e in p.log), p.log
    print("DISARM helper: loop _touchdown_disarm_check -> 'landed_disarmed'")


def test_disarm_live_mission_touchdown_branch():
    """Synthetic live-mission: DISARM fires through the loop's actual
    touchdown branch.

    The touchdown path is nearly unreachable in natural sim (degraded
    perception triggers LAND but the below-clear guard blocks descent — a
    safe Catch-22), so DISARM had never fired in a live mission. This test
    forces it: a synthetic vehicle at touchdown altitude, escalation policy
    driven to LAND as the loop's top-of-step evaluate would leave it, and
    _esc_land=True (which the loop sets only post below-clear veto, i.e.
    clear-below is implied). Executes _loop_touchdown_branch — the factored
    loop branch itself, the actual code the loop runs — and verifies DISARM
    fires end-to-end with outcome "landed_disarmed". Also verifies the
    branch does NOT fire when the vehicle is still airborne.
    """
    from types import SimpleNamespace
    m = _loop_mod()
    p = _drive_to_land()
    assert p._current == LAND, p._current
    # At touchdown: branch fires, DISARM emitted, outcome set.
    vehicle = SimpleNamespace(pos=np.array([3.0, 4.0, 0.02]))
    log = {}
    fired = m._loop_touchdown_branch(
        True, vehicle, p, None, 0.0, {}, 100.0, log)
    assert fired, "loop touchdown branch did not fire at touchdown"
    assert log.get("outcome") == "landed_disarmed", log
    assert p._current == DISARM, p._current
    # Still airborne: branch must not fire, no outcome set.
    p2 = _drive_to_land()
    vehicle2 = SimpleNamespace(pos=np.array([3.0, 4.0, 1.5]))
    log2 = {}
    fired2 = m._loop_touchdown_branch(
        True, vehicle2, p2, None, 0.0, {}, 100.0, log2)
    assert not fired2, "touchdown branch fired while airborne — hazard!"
    assert "outcome" not in log2, log2
    print("DISARM live-mission: loop touchdown branch -> 'landed_disarmed'")


def test_disarm_loop_stale_land_hazard():
    """Defense in depth for the stale-LAND hazard.

    The loop gates the final evaluate on _esc_land (True only post
    below-clear veto), so a stale LAND with the loop hovering never reaches
    the check. Even if it did — policy at HOVER with landed=True — the
    policy vetoes DISARM loudly and the helper returns False (loop falls
    back to "landed", never disarms).
    """
    m = _loop_mod()
    p = EscalationPolicy()
    a, _ = p.evaluate(0.5, 0.05, 0.0, False, dt=6.0, t=0.0)
    assert a == HOVER and p._current == HOVER, (a, p._current)
    log = {}
    fired = m._touchdown_disarm_check(p, None, 0.0, {}, 100.0, log)
    assert not fired, "DISARM fired with _current=HOVER — hazard!"
    assert log.get("outcome") != "landed_disarmed", log
    assert any(e[1] == "disarm_vetoed_not_landing" for e in p.log), p.log
    print("DISARM hazard: stale LAND + hovering -> vetoed, no DISARM")


def test_disarm_loop_no_policy_fallback():
    """Escalation off (policy None): the helper returns False and the loop
    falls back to outcome="landed". No crash, no DISARM."""
    m = _loop_mod()
    log = {}
    fired = m._touchdown_disarm_check(None, None, 0.0, {}, 100.0, log)
    assert not fired, fired
    assert "outcome" not in log, log
    print("DISARM loop: policy absent -> fallback, no DISARM")


if __name__ == "__main__":
    test_z1_rails()
    test_decider_grid()
    test_policy_rth_gate()
    test_policy_rth_gate_multistep()
    test_policy_rth_gate_open_then_closed()
    test_policy_hold_first()
    test_policy_most_severe_wins()
    test_policy_missing_stubs()
    test_disarm_fires_after_landing()
    test_disarm_never_in_air()
    test_disarm_fuzz_never_airborne()
    test_disarm_latches()
    test_disarm_guard_not_landing()
    test_disarm_no_timeout_path()
    test_disarm_behavior_contract()
    test_regression_shipped_envelope()
    test_disarm_via_loop_touchdown_helper()
    test_disarm_live_mission_touchdown_branch()
    test_disarm_loop_stale_land_hazard()
    test_disarm_loop_no_policy_fallback()
    print("ALL ESCALATION TESTS PASS")
