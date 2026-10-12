"""Unit tests for the Option A bounded-hold bookkeeping
(_land_hold_update in mission_dynamics_v5.py; Paul, decided
2026-10-12). Pure-function tests: the mission-level behavior is
exercised by run_belowclear_validation.py; these pin the episode
rule itself.

Run: python3 -m pytest test_belowclear_hold.py -q
(or plain: python3 test_belowclear_hold.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mission_dynamics_v5 import _land_hold_update  # noqa: E402


def test_clear_column_never_starts_episode():
    start, expired = _land_hold_update(10.0, None, 100.0, 0)
    assert start is None and expired is False
    # ...and a clear step ends a running episode.
    start, expired = _land_hold_update(10.0, 90.0, 100.0, 0)
    assert start is None and expired is False


def test_first_vetoed_step_starts_episode_not_expired():
    start, expired = _land_hold_update(10.0, None, 100.0, 1)
    assert start == 100.0 and expired is False
    start, expired = _land_hold_update(10.0, None, 100.0, 2)
    assert start == 100.0 and expired is False


def test_episode_expires_exactly_at_window():
    # Vetoed continuously from t=100 with a 10 s window: t=109 is the
    # tenth vetoed step (held 9 s) — still held; t=110 expires.
    start, expired = _land_hold_update(10.0, 100.0, 109.0, 1)
    assert start == 100.0 and expired is False
    start, expired = _land_hold_update(10.0, 100.0, 110.0, 1)
    assert start == 100.0 and expired is True
    # Occupied (bst=2) obeys the same window — Option A expires
    # regardless of the veto state.
    start, expired = _land_hold_update(10.0, 100.0, 110.0, 2)
    assert start == 100.0 and expired is True


def test_episode_persists_while_vetoed():
    start = None
    for t in range(100, 110):
        start, expired = _land_hold_update(10.0, start, float(t), 1)
        assert start == 100.0 and expired is False
    start, expired = _land_hold_update(10.0, start, 110.0, 1)
    assert expired is True


def test_window_zero_expires_immediately():
    # Degenerate window 0: "the veto may hold for at most 0 s" — the
    # first vetoed step already satisfies held (0 s) >= window (0 s).
    # Note the mission cfg layer treats land_hold_window_s=0 as OFF
    # (falsy), so this case is unreachable from a mission; the rule
    # is pinned here so the helper has no undefined corner.
    start, expired = _land_hold_update(0.0, None, 5.0, 1)
    assert start == 5.0 and expired is True


if __name__ == "__main__":
    test_clear_column_never_starts_episode()
    test_first_vetoed_step_starts_episode_not_expired()
    test_episode_expires_exactly_at_window()
    test_episode_persists_while_vetoed()
    test_window_zero_expires_immediately()
    print("test_belowclear_hold: ALL PASS")
