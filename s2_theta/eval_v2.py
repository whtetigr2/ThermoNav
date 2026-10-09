"""Round 2 eval: agreement, temporal-coupling diagnostic, Z1, head-to-head vs round 1.

Headline diagnostic: mean learned J on temporal edges (t0->t1, t1->t2 blocks)
vs teacher's. Round 1: learner +0.84 vs teacher -0.03 (WRONG SIGN).
If round 2 flips the sign, it was data. If not, it's architecture.
"""
import sys, os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'thermopilot'))
sys.path.insert(0, '.')

from harness import S2Agent
from gibbs_ising import gibbs_ising
from model_def import N, K, T, build_mask

_edges, h_eng_18, J_eng_18 = build_mask()
eidx = _edges
EIDX = [tuple(e) for e in eidx]


def temporal_edges():
    """Edge indices (into eidx) connecting consecutive timestep blocks."""
    out = []
    for k, (i, j) in enumerate(EIDX):
        ti, tj = i // K, j // K
        if abs(ti - tj) == 1:
            out.append(k)
    return out


def onehot_edges():
    out = []
    for k, (i, j) in enumerate(EIDX):
        if i // K == j // K:
            out.append(k)
    return out


def temporal_diagnostic(param_file, label):
    z = np.load(param_file)
    w = z['weights']
    w_eng = np.array([J_eng_18[i, j] for (i, j) in EIDX])
    te, oe = temporal_edges(), onehot_edges()
    print(f"--- {label} ---")
    print(f"  temporal edges: n={len(te)}")
    print(f"  teacher  mean J_temporal = {w_eng[te].mean():+.3f}  (onehot {w_eng[oe].mean():+.3f})")
    print(f"  learned  mean J_temporal = {w[te].mean():+.3f}  (onehot {w[oe].mean():+.3f})")
    # per-block-pair detail
    for (ta, tb) in [(0, 1), (1, 2)]:
        m = [k for k in te
             if sorted([EIDX[k][0] // K, EIDX[k][1] // K]) == [ta, tb]]
        print(f"    t{ta}->t{tb}: teacher {w_eng[m].mean():+.3f}, learned {w[m].mean():+.3f}")
    return w[te].mean()


def agreement_v2(param_file, n_sweeps=200, seed=123, max_n=400):
    """Learned t=0 action vs expert's realized a_t on held-out windows."""
    from s2_learned import LearnedS2Agent
    d = np.load('s2_data_v2.npz')
    hh, seq = d['held_h_dyn'][:max_n], d['held_seq'][:max_n]
    lrn = LearnedS2Agent(param_file=param_file, n_sweeps=n_sweeps, seed=seed)
    agree = 0
    for i in range(len(seq)):
        h = lrn.h_static + hh[i]
        _, best_s, _, _ = gibbs_ising(h, lrn.J, n_sweeps=n_sweeps,
                                      burn_in=n_sweeps // 4, seed=seed + i)
        row0 = ((best_s + 1) / 2).astype(int).reshape(T, K)[0]
        pred = int(np.argmax(row0)) if row0.sum() == 1 else -1
        if pred == seq[i, 0]:
            agree += 1
    return agree / len(seq)


def z1_check(param_file):
    z = np.load(param_file)
    b, w = z['biases'], z['weights']
    J = np.zeros((N, N))
    for (i, j), ww in zip(EIDX, w):
        J[i, j] = ww; J[j, i] = ww
    deg = (np.abs(J) > 1e-12).sum(axis=1).max()
    return dict(max_abs_h=float(np.abs(b).max()), max_abs_J=float(np.abs(w).max()),
                max_degree=int(deg), n_params=len(b) + len(w))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pfx', default='s2_theta_v2')
    args = ap.parse_args()
    for init in ['random', 'warm']:
        pf = f'{args.pfx}_{init}.npz'
        print(f"===== {pf} =====")
        temporal_diagnostic(pf, pf)
        ag = agreement_v2(pf)
        print(f"  held-out agreement (t=0): {ag*100:.1f}%")
        print(f"  Z1: {z1_check(pf)}")


if __name__ == '__main__':
    main()
