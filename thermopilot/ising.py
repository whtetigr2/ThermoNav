"""Ising construction kit for thermopilot formulation search.

Convention: E(s) = -sum_i h_i s_i - sum_{i<j} J_{ij} s_i s_j, s_i in {-1,+1}.
Lowering from binary x in {0,1}: x = (1+s)/2.

All functions return (h, J) with h: (n,) array, J: (n,n) symmetric array.
"""
import numpy as np


def one_hot(n, P):
    """One-hot penalty P*(sum x - 1)^2 lowered to Ising.

    Derived: h_i = (P/2)(2-n), J_{ij} = -P/2 for all i<j.
    Verified: minima exactly at one-hot configurations iff P > min_i c_i
    (when combined with linear costs c_i >= 0).

    Args:
        n: number of binary variables.
        P: penalty strength. Use P=2 with costs normalized to [0,1].
    Returns:
        (h, J)
    """
    h = np.full(n, (P / 2.0) * (2 - n))
    J = np.full((n, n), -P / 2.0)
    np.fill_diagonal(J, 0.0)
    return h, J


def linear_costs(costs):
    """Linear term sum_i c_i x_i lowered to Ising: h_i = -c_i/2."""
    return -np.asarray(costs, dtype=float) / 2.0


def bilinear(cost_matrix):
    """Bilinear term sum_{i,j} C_{ij} x_i y_j lowered to Ising.

    x_i y_j = (1 + s_i + s_j + s_i s_j)/4.
    Returns (h_x_extra, h_y_extra, J_xy) additions.
    """
    C = np.asarray(cost_matrix, dtype=float)
    n, m = C.shape
    # field on x_i from sum_j C_ij (1 + s_j)/4 -> (1/4) sum_j C_ij
    hx = C.sum(axis=1) / 4.0   # contributes -(hx) s_i? careful:
    # E contains +C_ij x_i y_j = C_ij/4 (1 + s_i^x + s_j^y + s_i^x s_j^y).
    # In E = -h s form: -(h_xi) s_i^x has h_xi = -C_ij/4 summed over j.
    hx = -C.sum(axis=1) / 4.0
    hy = -C.sum(axis=0) / 4.0
    Jxy = -C / 4.0
    return hx, hy, Jxy


def check_one_hot_optimal(costs, P):
    """Verify by brute force that exactly-one is always optimal.

    Returns True if for the QUBO E = sum c_i x_i + P(sum x - 1)^2,
    every minimizer is a one-hot vector.
    """
    from itertools import product
    n = len(costs)
    best = 1e18
    best_cfgs = []
    for bits in product([0, 1], repeat=n):
        b = np.array(bits)
        E = float(np.dot(costs, b) + P * (b.sum() - 1) ** 2)
        if E < best - 1e-9:
            best = E
            best_cfgs = [b]
        elif abs(E - best) < 1e-9:
            best_cfgs.append(b)
    return all(c.sum() == 1 for c in best_cfgs), best, best_cfgs


def degree_of(J, tol=1e-12):
    """Max degree (nonzero couplings per spin)."""
    return int((np.abs(J) > tol).sum(axis=1).max())


def max_abs_h(h):
    return float(np.abs(h).max())


def max_abs_J(J):
    return float(np.abs(J).max())


def z1_check(h, J, name=""):
    """Check against Z1 constraints. Returns dict."""
    d = degree_of(J)
    mh, mJ = max_abs_h(h), max_abs_J(J)
    ok = (d <= 16) and (mJ <= 6.0) and (mh <= 6.0)
    return {"name": name, "degree": d, "max|h|": round(mh, 3),
            "max|J|": round(mJ, 3), "n_spins": len(h), "fits_Z1": ok}
