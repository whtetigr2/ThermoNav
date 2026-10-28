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


def two_colourable(J, tol=1e-12):
    """True iff the support graph of J (nonzero couplings) is bipartite.

    Parity BFS: colour each spin by BFS-depth parity; an edge joining two
    spins of the same colour (or a self-loop) makes the graph
    non-bipartite.
    """
    J = np.asarray(J)
    n = J.shape[0]
    colour = np.full(n, -1, dtype=int)
    for start in range(n):
        if colour[start] != -1:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            u = stack.pop()
            for v in np.nonzero(np.abs(J[u]) > tol)[0]:
                if v == u:
                    return False
                if colour[v] == -1:
                    colour[v] = colour[u] ^ 1
                    stack.append(v)
                elif colour[v] == colour[u]:
                    return False
    return True


def z1_check(h, J, name=""):
    """Screen (h, J) against the Z1 working caps. Returns dict.

    This is a SCREEN, not a placement proof: passing is necessary, not
    sufficient. Degree <= 16 is weaker than the real rule — Z1 edges must
    be a subset of a fixed stencil, so a graph can pass the degree cap
    and still not place. The 6.0 caps are likewise working numbers, not
    datasheet specs: the |J| bound traces to a figure label in
    arXiv 2608.01615, and the |h| bound is UNSOURCED — no published Z1
    field cap exists (see the note on S2Agent._Z1_MAX_H in harness.py).

    `two_colourable` is the parity test on the support graph of J: the
    chip schedules spins in two colour blocks, so a graph that is not
    bipartite cannot be scheduled on it at all. (Correction 2026-10-28:
    before this date the screen did not test colourability and reported
    fits_Z1 True for the shipped S2 kernel, whose graph is 6-chromatic.)
    """
    d = degree_of(J)
    mh, mJ = max_abs_h(h), max_abs_J(J)
    tc = two_colourable(J)
    ok = (d <= 16) and (mJ <= 6.0) and (mh <= 6.0) and tc
    return {"name": name, "degree": d, "max|h|": round(mh, 3),
            "max|J|": round(mJ, 3), "n_spins": len(h),
            "two_colourable": bool(tc), "fits_Z1": ok}
