"""Visibility test: dynamic fields vs the declared |h| cap.

S2Agent._Z1_MAX_H = 6.0 is a declared working cap (UNSOURCED — no
published Z1 field cap exists; see harness.py). The dynamic field in
S2Agent.act accumulates onto the static field with no bound, and on the
repo's own held-out data the sum exceeds the cap:

    max|h_static + h_dyn| = 7.050   (s2_data_v2.npz held_h_dyn +
                                     s2_theta_v3.npz biases)

This test asserts the cap anyway and is marked xfail(strict=True):

- The point is VISIBILITY, not compliance. Nothing is clipped: clipping
  fields to an unsourced threshold would silently distort the cost
  ordering to satisfy a number with no provenance, which is worse than
  exceeding it. S2Agent.act now surfaces max_abs_h per decision so the
  value is in the run record.
- strict=True means the suite FAILS if this test ever starts passing —
  i.e. if the violation disappears (fields re-derived, cap re-sourced,
  data regenerated), a human is forced to revisit the xfail and the
  cap's status instead of the marker quietly going stale.
- The test becomes a plain passing test only when a real field cap is
  sourced and the fields are shown to sit inside it.

Run under pytest for the xfail reporting; standalone it raises with
the observed value named.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DECLARED_CAP = 6.0  # S2Agent._Z1_MAX_H (UNSOURCED working number)


def observed_max_abs_h():
    """max|h_static + h_dyn| over the repo's held-out dynamic fields."""
    z = np.load(os.path.join(ROOT, "s2_theta", "s2_theta_v3.npz"))
    d = np.load(os.path.join(ROOT, "s2_theta", "s2_data_v2.npz"))
    biases = z["biases"].astype(float)
    h_dyn = d["held_h_dyn"].astype(float).reshape(-1, len(biases))
    return float(np.abs(h_dyn + biases).max())


def _assert_within_cap():
    observed = observed_max_abs_h()
    assert observed <= DECLARED_CAP, (
        f"max|h_static + h_dyn| = {observed:.3f} exceeds the declared "
        f"|h| cap of {DECLARED_CAP} (UNSOURCED working number)")


try:
    import pytest

    @pytest.mark.xfail(
        strict=True,
        reason="max|h_static + h_dyn| = 7.050 on held-out fields vs the "
               "declared 6.0 cap (UNSOURCED); visibility only — see "
               "module docstring",
    )
    def test_heldout_fields_within_declared_cap():
        _assert_within_cap()

except ImportError:  # pragma: no cover - pytest not installed
    def test_heldout_fields_within_declared_cap():
        _assert_within_cap()


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:
        _assert_within_cap()  # raises, naming the observed value
    else:
        sys.exit(pytest.main([os.path.abspath(__file__), "-rx"]))
