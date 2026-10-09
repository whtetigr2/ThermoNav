"""Evaluate learned S2-theta: agreement, Z1-legality, ablation."""
import sys, os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'thermopilot'))
sys.path.insert(0, '.')

from harness import S2Agent, MOVES_6
from s2_learned import LearnedS2Agent
from model_def import N, K


def agreement(held_h_dyn, held_act, param_file, n_sweeps=200, seed=123):
    """Learned action vs engineered action on held-out contexts."""
    eng = S2Agent(n_sweeps=n_sweeps, seed=seed)
    lrn = LearnedS2Agent(param_file=param_file, n_sweeps=n_sweeps, seed=seed)
    n = len(held_act)
    agree = 0
    # We need full contexts (pos, goal, p_occ) to call act(). But we only
    # saved h_dyn. So we compare differently: reconstruct h_total and sample.
    # Actually, for a fair comparison we need the expert's action on the same
    # h_dyn. The held_act IS the expert's action. We compute the learned
    # model's action by Gibbs sampling with h = h_learned + h_dyn.
    from gibbs_ising import gibbs_ising
    for i in range(n):
        h = lrn.h_static + held_h_dyn[i]
        _, best_s, _, _ = gibbs_ising(h, lrn.J, n_sweeps=n_sweeps,
                                      burn_in=n_sweeps // 4,
                                      seed=seed + i)
        x_best = ((best_s + 1) / 2).astype(int).reshape(3, K)
        row0 = x_best[0]
        pred = int(np.argmax(row0)) if row0.sum() == 1 else -1
        if pred == held_act[i]:
            agree += 1
    return agree / n


def teacher_noise_ceiling(held_h_dyn, held_act, n_sweeps=200):
    """Teacher vs teacher (different seeds): the sampling-noise ceiling."""
    from gibbs_ising import gibbs_ising
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, 'thermopilot'))
    from harness import S2Agent
    eng = S2Agent(n_sweeps=n_sweeps, seed=0)
    n = len(held_act)
    agree = 0
    for i in range(n):
        # teacher's distribution: sample with two different seeds
        h = eng.h_static + held_h_dyn[i]
        _, bs1, _, _ = gibbs_ising(h, eng.J, n_sweeps=n_sweeps,
                                   burn_in=n_sweeps // 4, seed=1000 + i)
        _, bs2, _, _ = gibbs_ising(h, eng.J, n_sweeps=n_sweeps,
                                   burn_in=n_sweeps // 4, seed=2000 + i)
        r1 = ((bs1 + 1) / 2).astype(int).reshape(3, K)[0]
        r2 = ((bs2 + 1) / 2).astype(int).reshape(3, K)[0]
        p1 = int(np.argmax(r1)) if r1.sum() == 1 else -1
        p2 = int(np.argmax(r2)) if r2.sum() == 1 else -1
        if p1 == p2:
            agree += 1
    return agree / n


def z1_check_param(param_file):
    z = np.load(param_file)
    b, w = z['biases'], z['weights']
    edges = z['edges']
    J = np.zeros((N, N))
    for (i, j), ww in zip(map(tuple, edges), w):
        J[i, j] = ww; J[j, i] = ww
    deg = (np.abs(J) > 1e-12).sum(axis=1).max()
    return dict(max_abs_h=float(np.abs(b).max()),
                max_abs_J=float(np.abs(w).max()),
                max_degree=int(deg),
                n_params=len(b) + len(w),
                bytes_fp32=(len(b) + len(w)) * 4)


def main():
    d = np.load('s2_data.npz')
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='s2_theta.npz')
    args = ap.parse_args()

    print("=== Z1 legality ===")
    z1 = z1_check_param(args.param)
    for k, v in z1.items():
        print(f"  {k}: {v}")
    assert z1['max_abs_h'] <= 6.0 and z1['max_abs_J'] <= 6.0 and z1['max_degree'] <= 16

    print("\n=== Agreement on held-out ===")
    agr = agreement(d['held_h_dyn'], d['held_act'], args.param)
    print(f"  learned vs expert: {agr*100:.1f}%")
    ceil = teacher_noise_ceiling(d['held_h_dyn'], d['held_act'])
    print(f"  teacher vs teacher (noise ceiling): {ceil*100:.1f}%")


if __name__ == '__main__':
    main()
