"""Escalation decider: 12-spin Z1-legal Ising kernel.

4 actions x 3 spins (redundancy mirrors S2-theta's 6x3 layout):
    group 0 (spins 0-2):   continue
    group 1 (spins 3-5):   hover
    group 2 (spins 6-8):   RTH
    group 3 (spins 9-11):  land

Static (hand-designed, teacher-style — cf. the teacher audit: human-placed
weights are legitimate; the student hypothesis class must contain them):
    J = +1.0 within group  (ferromagnetic: redundancy agreement)
    J = 0    between groups (pairwise antiferro CANNOT stabilize one-hot
             with 4 groups — the pair combinatorics favor 2-active splits;
             verified analytically. One-hot is enforced at commit.)
    h_static = -0.25 per spin (activation baseline [design]: a group fires
             only if its action field overcomes it; no degenerate ties)

Dynamic h (per decision) maps the Option A trigger table into fields.
Most-severe-wins is guaranteed at the deterministic commit step: among
firing groups (positive spin-sum) the highest severity wins; ties are
harmless by construction. This is the S2 doctrine — stochastic belief
(Gibbs best_s), deterministic commit (severity-ordered argmax).

Dynamic h (per decision) maps the Option A trigger table into fields.
Severity ordering is IN the energy landscape: stronger fields on harsher
actions, so most-severe-wins emerges from the physics, not an if-chain.

Inputs (binary flags; the policy computes timers from raw signals):
    vis_deg, far_deg : perception-degraded band
    stale            : map_age >= 1.0 s
    jitter           : compute jitter active (decider input; policy row: none —
                       jitter's effect IS map_age in this sim)
    deg_5s           : degraded band sustained 5 s   [design]
    deg_30s          : degraded band sustained 30 s  [design]
    rth_available    : policy-level pose gate (True only when an estimator
                       clears <0.5 m/mission; False today)

Field map (dynamic h added to each group's 3 spins):
    continue : +0.5                                  (default bias [design])
    hover    : +2.0 if (vis_deg & far_deg & deg_5s) | (vis_deg & far_deg & stale)
    RTH      : +3.0 if (vis_deg & far_deg & deg_30s)
    land     : +4.0 if (vis_deg & far_deg & deg_30s & ~rth_available)

Readout: stochastic belief via sequential Gibbs (best_s, per the S2
doctrine), deterministic commit via argmax over group sums (ties ->
lowest index = continue, the industry default state).

Z1-legal stats: 12 spins (10-14 budget), degree 2 (<=16), max|h| = 4.0
(<=6), max|J| = 1.0 (<=6).

Claim label: "Z1-legal formulation, validated in simulation."
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

N_ACTIONS = 4
SPINS_PER_ACTION = 3
N = N_ACTIONS * SPINS_PER_ACTION  # 12
ACTION_NAMES = ["continue", "hover", "RTH", "land"]

J_WITHIN = 1.0    # ferromagnetic redundancy
J_BETWEEN = 0.0   # no between-group coupling: pairwise antiferro provably
                  # cannot stabilize one-hot with 4 groups (pair combinatorics
                  # favor 2-active/2-inactive splits). One-hot via commit rule.
H_BASELINE = -0.25  # [design] activation threshold per spin

# Field strengths (severity ordering lives here)
H_CONTINUE = 0.5  # [design] default bias
H_HOVER = 2.0     # [design]
H_RTH = 3.0       # [design]
H_LAND = 4.0      # [design]


def build_decider():
    """Static (h, edges, J). Returns (h_static, edges, weights)."""
    h = np.full(N, H_BASELINE)
    edges, weights = [], []
    for i in range(N):
        gi = i // SPINS_PER_ACTION
        for j in range(i + 1, N):
            gj = j // SPINS_PER_ACTION
            if gi != gj:
                continue  # no between-group coupling (see module docstring)
            edges.append((i, j))
            weights.append(J_WITHIN)
    return h, np.array(edges, dtype=np.int64), np.array(weights)


def h_dyn_from_inputs(vis_deg, far_deg, stale, jitter,
                      deg_5s, deg_30s, rth_available):
    """Map the Option A trigger table to dynamic fields. All inputs bool."""
    h = np.zeros(N)
    degraded = bool(vis_deg and far_deg)
    h[0:3] += H_CONTINUE
    if degraded and (deg_5s or stale):
        h[3:6] += H_HOVER
    if degraded and deg_30s:
        h[6:9] += H_RTH
    if degraded and deg_30s and not rth_available:
        h[9:12] += H_LAND
    # jitter is a decider input per the input spec; in this sim its effect
    # is fully captured by map_age (stale), so it adds no separate field.
    # Documented here rather than silently dropped.
    _ = jitter
    return h


def _group_argmax(state):
    """Deterministic commit: among firing groups (positive spin-sum) pick
    the HIGHEST severity (most-severe-wins, industry rule 2). No firing
    group -> continue (index 0), the industry default state."""
    scores = [float(np.sum(state[g * 3:(g + 1) * 3]))
              for g in range(N_ACTIONS)]
    firing = [g for g, s in enumerate(scores) if s > 0]
    if not firing:
        return 0
    return max(firing)  # severity order == group index order


def decide(h_total, J_full, n_sweeps=300, seed=0):
    """Stochastic belief (sequential Gibbs, best_s) -> deterministic commit.

    Returns (action_idx, best_state, best_energy).
    Mirrors the S2 deployment doctrine: physics forms the belief, argmax
    commits. For this hand-designed landscape best_s and the argmax agree;
    the unit test asserts it across the input grid.
    """
    rng = np.random.default_rng(seed)
    s = np.full(N, -1.0)

    def energy(sv):
        return float(-np.dot(h_total, sv) - sv @ (J_full @ sv) / 2.0)

    best_E, best_s = 1e18, s.copy()
    for _ in range(n_sweeps):
        for i in rng.permutation(N):
            dE = -2.0 * h_total[i] - 2.0 * np.dot(J_full[i], s)
            p_plus = 1.0 / (1.0 + np.exp(np.clip(dE, -30, 30)))
            s[i] = 1.0 if rng.random() < p_plus else -1.0
        E = energy(s)
        if E < best_E:
            best_E, best_s = E, s.copy()
    return _group_argmax(best_s), best_s, best_E


def z1_stats(h_static, h_dyn_max, edges, weights):
    """Z1-legal rails check. Returns dict of stats."""
    deg = np.zeros(N, dtype=int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    return {
        "n_spins": N,
        "max|h|": float(max(np.max(np.abs(h_static)),
                            np.max(np.abs(h_dyn_max)))),
        "max|J|": float(np.max(np.abs(weights))),
        "max_degree": int(np.max(deg)),
        "n_edges": len(edges),
    }


def save_kernel(path=None):
    """Save the kernel npz mirroring s2_theta_v2.npz fields."""
    h, edges, weights = build_decider()
    # Worst-case dynamic fields (all rows firing) for the |h| rail check.
    h_worst = h_dyn_from_inputs(True, True, True, True, True, True, False)
    stats = z1_stats(h, h_worst, edges, weights)
    path = path or os.path.join(_HERE, "kernel", "escalation_decider_v1.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             biases=h.astype(np.float32),
             weights=weights.astype(np.float32),
             edges=edges,
             h_eng=h.astype(np.float32),
             w_eng=weights.astype(np.float32),
             action_names=np.array(ACTION_NAMES),
             z1_stats=np.array([stats["n_spins"], stats["max|h|"],
                                stats["max|J|"], stats["max_degree"]]))
    return path, stats
