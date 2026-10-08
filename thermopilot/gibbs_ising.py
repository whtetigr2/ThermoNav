"""Generic sequential Gibbs sampler for arbitrary Ising models.

For the small dense action-selection problems (one-hot cliques), chromatic
updating doesn't apply — just do sequential single-site Gibbs.
"""
import numpy as np


def gibbs_ising(h, J, n_sweeps=2000, burn_in=500, seed=0, x_init=None):
    """Sample from p(s) propto exp(-E(s)), E = -h.s - s^T J s (J symmetric, 0 diag).

    Args:
        h: (n,) fields. J: (n,n) couplings.
    Returns:
        (marginals P(s_i=+1), best_state, best_energy, n_sweeps)
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    # work in s in {-1,+1}
    s = np.full(n, -1.0) if x_init is None else np.where(np.asarray(x_init) > 0, 1.0, -1.0)
    acc = np.zeros(n)
    n_acc = 0
    best_E = 1e18
    best_s = s.copy()

    def energy(sv):
        return float(-np.dot(h, sv) - np.dot(sv, J @ sv) / 2.0)

    for sweep in range(n_sweeps):
        order = rng.permutation(n)
        for i in order:
            # E(s_i=+1) - E(s_i=-1) = -2 h_i - 2 sum_j J_ij s_j
            dE = -2.0 * h[i] - 2.0 * np.dot(J[i], s)
            # P(s_i=+1) = sigmoid(-dE/2 * ... ) careful:
            # p(+1)/p(-1) = exp(-(E(+1)-E(-1))) = exp(-dE)
            p_plus = 1.0 / (1.0 + np.exp(np.clip(dE, -30, 30)))
            s[i] = 1.0 if rng.random() < p_plus else -1.0
        E = energy(s)
        if E < best_E:
            best_E = E
            best_s = s.copy()
        if sweep >= burn_in:
            acc += (s + 1.0) / 2.0
            n_acc += 1
    marg = acc / max(n_acc, 1)
    return marg, best_s, best_E, n_sweeps


def qubo_energy_binary(x, costs, P):
    """Reference QUBO energy for one-hot action selection (binary x)."""
    x = np.asarray(x, dtype=float)
    return float(np.dot(costs, x) + P * (x.sum() - 1.0) ** 2)
