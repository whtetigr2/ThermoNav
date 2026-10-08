"""Collect (h_dyn, expert_action) training data from the engineered S2Agent.

Contexts: real room geometries + real perception model (noisy_perception_with_counts).
Expert: engineered S2Agent.act() (200 sweeps, same as deployed).
Rooms: train {071, 004, 131}, held-out {185, 128} (room-level holdout, no geometry leakage).
Saves incrementally per room.
"""
import sys, os
import numpy as np

SYSROOT = os.path.expanduser('~/workspace/flynav-redteam')
sys.path.insert(0, os.path.join(SYSROOT, 'thermopilot'))
sys.path.insert(0, os.path.join(SYSROOT, 'slow-drone', 'dynamics_sim'))
sys.path.insert(0, os.path.join(SYSROOT, 'slow-drone'))

from harness import S2Agent, noisy_perception_with_counts, MOVES_6
from model_def import N, K, T

TRAIN_ROOMS = ['indoorcd_071', 'indoorcd_004', 'indoorcd_131']
HELD_ROOMS = ['indoorcd_185', 'indoorcd_128']
REAL_DIR = os.path.join(SYSROOT, 'slow-drone', 'real_lidar')


def compute_h_dyn(agent, pos, goal, p_occ, forbidden_moves=None, obs_count=None):
    """Replicate S2Agent.act()'s dynamic field exactly. Returns additive h_dyn (18,)."""
    h_dyn = np.zeros(agent.n)
    _fb = set(forbidden_moves) if forbidden_moves else set()
    for t in range(agent.T):
        for i, d in enumerate(agent.moves):
            idx = t * agent.K + i
            c = agent._ray_cost(pos, d, t + 1, goal, p_occ, obs_count)
            if i in _fb:
                h_dyn[idx] = -agent._FORBID_PENALTY_H - agent.h_static[idx]
            else:
                h_dyn[idx] = -c / 2.0
    return h_dyn


def verify_h_dyn():
    rng = np.random.default_rng(0)
    occ = np.load(os.path.join(REAL_DIR, 'indoorcd_071_occ.npy'))
    agent = S2Agent(n_sweeps=50, seed=0)
    free = np.argwhere(occ < 0.5)
    pos = tuple(free[rng.integers(len(free))])
    goal = tuple(free[rng.integers(len(free))])
    p_occ, _ = noisy_perception_with_counts(occ, start=pos, goal=goal, seed=1)
    h_dyn = compute_h_dyn(agent, pos, goal, p_occ)
    h = agent.h_static.copy()
    for t in range(agent.T):
        for i, d in enumerate(agent.moves):
            idx = t * agent.K + i
            c = agent._ray_cost(pos, d, t + 1, goal, p_occ, None)
            h[idx] += -c / 2.0
    assert np.allclose(h, agent.h_static + h_dyn), "h_dyn mismatch!"
    print("h_dyn verification PASSED", flush=True)


def sample_contexts(room, n, seed, hard_frac=0.3):
    rng = np.random.default_rng(seed)
    occ = np.load(os.path.join(REAL_DIR, f'{room}_occ.npy'))
    free = np.argwhere(occ < 0.5)
    agent = S2Agent(n_sweeps=200, seed=seed)
    try:
        from scipy.ndimage import binary_dilation
        near = binary_dilation(occ > 0.5, iterations=2)
        near_free = [tuple(v) for v in np.argwhere((occ < 0.5) & near)]
    except ImportError:
        near_free = []
    h_dyn_list, act_list = [], []
    for s in range(n):
        if rng.random() < hard_frac and near_free:
            pos = near_free[rng.integers(len(near_free))]
        else:
            pos = tuple(free[rng.integers(len(free))])
        goal = tuple(free[rng.integers(len(free))])
        if max(abs(pos[d] - goal[d]) for d in range(3)) < 3:
            continue
        p_occ, _ = noisy_perception_with_counts(
            occ, start=pos, goal=goal, seed=1000 + seed * 10000 + s)
        fb = None
        if rng.random() < 0.2:
            fb = set(rng.choice(K, size=rng.integers(1, 3), replace=False).tolist())
        h_dyn = compute_h_dyn(agent, pos, goal, p_occ, fb)
        nxt, _ = agent.act(pos, goal, p_occ, forbidden_moves=fb)
        delta = (nxt[0] - pos[0], nxt[1] - pos[1], nxt[2] - pos[2])
        try:
            a = MOVES_6.index(delta)
        except ValueError:
            continue
        h_dyn_list.append(h_dyn)
        act_list.append(a)
        if (s + 1) % 500 == 0:
            print(f"  {room}: {s+1}/{n}", flush=True)
    print(f"{room}: {len(act_list)} samples", flush=True)
    return np.array(h_dyn_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def main():
    verify_h_dyn()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=2000)
    ap.add_argument('--n-held', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    train_h, train_a, held_h, held_a = [], [], [], []
    for i, room in enumerate(TRAIN_ROOMS):
        h, a = sample_contexts(room, args.n_train, seed=args.seed * 100 + i)
        train_h.append(h); train_a.append(a)
        np.savez('s2_data_partial.npz',
                 train_h_dyn=np.concatenate(train_h),
                 train_act=np.concatenate(train_a),
                 done_rooms=TRAIN_ROOMS[:i+1])
    for i, room in enumerate(HELD_ROOMS):
        h, a = sample_contexts(room, args.n_held, seed=args.seed * 100 + 50 + i)
        held_h.append(h); held_a.append(a)
    train_h = np.concatenate(train_h); train_a = np.concatenate(train_a)
    held_h = np.concatenate(held_h); held_a = np.concatenate(held_a)
    np.savez('s2_data.npz', train_h_dyn=train_h, train_act=train_a,
             held_h_dyn=held_h, held_act=held_a,
             train_rooms=TRAIN_ROOMS, held_rooms=HELD_ROOMS)
    print(f"TRAIN: {len(train_a)} samples, HELD: {len(held_a)} samples", flush=True)
    print(f"train hist: {np.bincount(train_a, minlength=6)}", flush=True)
    print(f"held hist: {np.bincount(held_a, minlength=6)}", flush=True)


if __name__ == '__main__':
    main()
