"""Tests for the nav-health -> battery checkpoint wiring (graduated
2026-10-11, roadmap Phase 1 item 3 closing the battery follow-up).

Behavioral, per lab doctrine: the wiring must be SEEN to bite.
1. resolve_nav_trustworthy: with a real NavHealthMonitor active the
   checkpoint trust is the monitor's pose_trustworthy (True while
   HEALTHY, False the step a freeze latches UNTRUSTED).
2. Scripted override fallback: with no monitor, an explicit
   cfg["nav_trustworthy"] (bool or callable) is honored — the
   deterministic scenario lever the battery validation's RTC
   constructions are built on. With neither, False (pre-graduation
   status quo: no in-flight checkpoint, RTC == RTB).
3. Mission level (numpy): a battery-safety flight with NO override
   and NO nav cfg runs the real monitor by default — nav_enabled=1,
   nominal feed stays HEALTHY, and the checkpoint is refreshed
   in flight (it is a flown pose, not the takeoff base).
4. Mission level: the same flight with the estimator report frozen
   from step 5 (measured whiteout signature) goes UNTRUSTED, never
   succeeds, never collides, and its checkpoint is frozen at an
   early in-flight pose (different from the nominal flight's
   late-flight checkpoint).

Claim label: host-side classical, validated in simulation.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nav_health as nh  # noqa: E402
import mission_dynamics_v5 as md  # noqa: E402
from fly import pick_start_goal  # noqa: E402

BASE_CFG = {"fix_atten_d5": True, "fix_confirm": True,
            "fix_lowvis": True, "fix_stale": True, "escalation": True,
            "battery": True, "battery_safety": True,
            "battery_start_soc": 0.5}


def _rep(**kw):
    r = {"cmd_speed_mps": 1.0, "est_speed_mps": 1.0, "est_jump_m": 1.0,
         "n_features": 60, "residual_m": 0.03, "map_age_s": 0.0,
         "dt_s": 1.0}
    r.update(kw)
    return r


def test_resolve_uses_real_monitor():
    mon = nh.NavHealthMonitor()
    mon.update(_rep(), t=0.0)
    assert md.resolve_nav_trustworthy({}, mon, 0.0, (0, 0, 0)) is True
    mon.update(_rep(est_speed_mps=0.125, est_jump_m=0.125), t=1.0)
    mon.update(_rep(est_speed_mps=0.125, est_jump_m=0.125), t=2.0)
    assert mon.state == nh.UNTRUSTED
    assert md.resolve_nav_trustworthy({}, mon, 2.0, (0, 0, 0)) is False
    print("resolve: monitor HEALTHY -> True; freeze-latched -> False")


def test_resolve_override_fallback():
    assert md.resolve_nav_trustworthy(
        {"nav_trustworthy": True}, None, 0.0, (0, 0, 0)) is True
    assert md.resolve_nav_trustworthy(
        {"nav_trustworthy": lambda t, p: t < 5.0}, None, 9.0,
        (0, 0, 0)) is False
    assert md.resolve_nav_trustworthy({}, None, 0.0, (0, 0, 0)) is False
    # The stub itself keeps its documented semantics.
    assert md.nav_trustworthy_stub({}, 0.0, (0, 0, 0)) is False
    assert md.nav_trustworthy_stub(
        {"nav_trustworthy": True}, 0.0, (0, 0, 0)) is True
    print("resolve: scripted override honored; absent both -> False")


def _fly(mid, cfg):
    occ = np.load(os.path.join(HERE, "..", "rooms",
                               "indoorcd_004_occ.npy"))
    return occ, md.run_mission_dynamics(
        occ, "indoorcd_004", 0, 0.15, mid, backend="numpy",
        write_traj=False, verbose=False, cfg=cfg)


def test_mission_default_monitor_feeds_checkpoint():
    occ, row = _fly("testnav_default_monitor", dict(BASE_CFG))
    start, _goal = pick_start_goal(occ, 0)
    base = ",".join(f"{c:.2f}" for c in start)
    assert row["nav_enabled"] == 1, row["nav_enabled"]
    assert row["batt_safety_enabled"] == 1
    assert row["nav_final_health"] == "HEALTHY", row["nav_final_health"]
    assert row["nav_untrusted_scans"] == 0
    assert row["outcome"] == "success", row["outcome"]
    # Checkpoint refreshed in flight by the real monitor: it is a
    # flown pose, not the takeoff base the stub era would have kept.
    assert row["safety_checkpoint"] not in ("", base), \
        row["safety_checkpoint"]
    print(f"mission default: monitor on, HEALTHY, checkpoint "
          f"{row['safety_checkpoint']} != base {base}")
    return row


def test_mission_freeze_freezes_checkpoint(nominal_row):
    cfg = dict(BASE_CFG, nav_health=True, nav_health_freeze_from_step=5)
    _occ, row = _fly("testnav_freeze_checkpoint", cfg)
    assert row["nav_enabled"] == 1
    assert row["nav_untrusted_scans"] > 0, row["nav_untrusted_scans"]
    assert row["nav_final_health"] == "UNTRUSTED"
    assert row["outcome"] != "success", row["outcome"]
    assert row["outcome"] != "collision", row["outcome"]
    # Frozen early: the checkpoint is an in-flight pose from before
    # the latch, not the nominal flight's late-flight checkpoint.
    assert row["safety_checkpoint"] not in (
        "", nominal_row["safety_checkpoint"]), row["safety_checkpoint"]
    print(f"mission freeze: UNTRUSTED x{row['nav_untrusted_scans']}, "
          f"outcome={row['outcome']}, checkpoint frozen at "
          f"{row['safety_checkpoint']}")


if __name__ == "__main__":
    test_resolve_uses_real_monitor()
    test_resolve_override_fallback()
    _nominal = test_mission_default_monitor_feeds_checkpoint()
    test_mission_freeze_freezes_checkpoint(_nominal)
    print("ALL NAV-CHECKPOINT WIRING TESTS PASS")
