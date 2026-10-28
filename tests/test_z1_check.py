"""Tests for the z1_check screen in thermopilot/ising.py.

The essential set only: the parity test must be observed to reject
before it is trusted, and the screen must fail the shipped S2 kernel,
whose coupling graph is 6-chromatic (measured; see S2_THETA.md).

Run standalone (python3 tests/test_z1_check.py) or under pytest.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "thermopilot"))

from ising import two_colourable, z1_check  # noqa: E402


def _J(n, edges, w=1.0):
    J = np.zeros((n, n))
    for i, j in edges:
        J[i, j] = J[j, i] = w
    return J


def test_parity_accepts_path():
    # 3-node path 0-1-2: bipartite.
    assert two_colourable(_J(3, [(0, 1), (1, 2)])) is True


def test_parity_rejects_triangle():
    # 3-node triangle: odd cycle, must reject. A parity test never
    # observed to reject has not been observed to work.
    assert two_colourable(_J(3, [(0, 1), (1, 2), (0, 2)])) is False


def test_chain_screen_two_colourable():
    # A 6-node chain passes the colour test (and the whole screen:
    # degree 2, no fields or couplings near the caps).
    res = z1_check(np.zeros(6), _J(6, [(i, i + 1) for i in range(5)]),
                   name="chain")
    assert res["two_colourable"] is True
    assert res["fits_Z1"] is True


def test_shipped_kernel_fails_screen():
    # The shipped S2 kernel (s2_theta_v3.npz) passes the old degree/cap
    # screen (degree 15, max|h| 5.35, max|J| 1.138) but its graph is
    # 6-chromatic, so the screen must now fail it on colourability.
    z = np.load(os.path.join(ROOT, "s2_theta", "s2_theta_v3.npz"))
    h = z["biases"].astype(float)
    J = np.zeros((len(h), len(h)))
    for (i, j), w in zip(map(tuple, z["edges"]), z["weights"]):
        J[i, j] = J[j, i] = w
    res = z1_check(h, J, name="s2_theta_v3")
    assert res["two_colourable"] is False
    assert res["fits_Z1"] is False


if __name__ == "__main__":
    test_parity_accepts_path()
    print("PASS parity accepts path")
    test_parity_rejects_triangle()
    print("PASS parity rejects triangle")
    test_chain_screen_two_colourable()
    print("PASS chain two-colourable")
    test_shipped_kernel_fails_screen()
    print("PASS shipped kernel fails screen on colourability")
    print("ALL Z1 CHECK TESTS PASS")
