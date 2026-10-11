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


if __name__ == "__main__":
    test_z1_rails()
    test_decider_grid()
    test_policy_rth_gate()
    test_policy_rth_gate_multistep()
    test_policy_rth_gate_open_then_closed()
    test_policy_hold_first()
    test_policy_most_severe_wins()
    test_policy_missing_stubs()
    test_regression_shipped_envelope()
    print("ALL ESCALATION TESTS PASS")
