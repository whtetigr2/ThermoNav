"""Round 2 data: expert TRAJECTORY rollouts -> 3-step sliding windows.

At each decision step t of an expert trajectory we record (h_dyn(x_t), a_t).
Windows: (h_dyn_t, a_t, a_{t+1}, a_{t+2}) — the expert's REALIZED future
supervises all three timesteps. Last 2 steps of each trajectory are dropped.

Rooms: train {071, 004, 131}, held-out {185, 128} (same as round 1).
New files only; s2_data.npz (round 1) untouched.
"""
import sys, os
import numpy as np

SYSROOT = os.path.expanduser('~/workspace/flynav-redteam')
sys.path.insert(0, os.path.join(SYSROOT, 'thermopilot'))
sys.path.insert(0, os.path.join(SYSROOT, 'slow-drone', 'dynamics_sim'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import S2Agent, noisy_perception_with_counts, MOVES_6
from collect_data import compute_h_dyn, verify_h_dyn
from model_def import N, K, T

TRAIN_ROOMS = ['indoorcd_071', 'indoorcd_004', 'indoorcd_131']
HELD_ROOMS = ['indoorcd_185', 'indoorcd_128']
REAL_DIR = os.path.join(SYSROOT, 'slow-drone', 'real_lidar')

MAX_STEPS = 80
MIN_GOAL_DIST = 10


def rollout(room, traj_id, seed, max_steps=MAX_STEPS):
    """One expert trajectory. Returns (h_dyn_seq (L,18), act_seq (L,)) or None."""
    rng = np.random.default_rng(seed)
    occ = np.load(os.path.join(REAL_DIR, f'{room}_occ.npy'))
    free = np.argwhere(occ < 0.5)
    agent = S2Agent(n_sweeps=200, seed=seed)
    for _attempt in range(20):
        pos = tuple(free[rng.integers(len(free))])
        goal = tuple(free[rng.integers(len(free))])
        if sum(abs(pos[d] - goal[d]) for d in range(3)) >= MIN_GOAL_DIST:
            break
    h_list, a_list = [], []
    visits = {}
    nx, ny, nz = occ.shape
    for step in range(max_steps):
        if not (0 <= pos[0] < nx and 0 <= pos[1] < ny and 0 <= pos[2] < nz):
            break  # expert flew out of bounds; keep valid prefix
        p_occ, _ = noisy_perception_with_counts(
            occ, start=pos, goal=goal, seed=seed * 100000 + step)
        fb = None
        if rng.random() < 0.2:
            fb = set(rng.choice(K, size=int(rng.integers(1, 3)), replace=False).tolist())
        h_dyn = compute_h_dyn(agent, pos, goal, p_occ, fb)
        nxt, _ = agent.act(pos, goal, p_occ, forbidden_moves=fb)
        delta = (nxt[0] - pos[0], nxt[1] - pos[1], nxt[2] - pos[2])
        try:
            a = MOVES_6.index(delta)
        except ValueError:
            break
        h_list.append(h_dyn)
        a_list.append(a)
        pos = nxt
        visits[pos] = visits.get(pos, 0) + 1
        if pos == goal:
            break
        if visits[pos] >= 3:  # looping; stop, data still valid
            break
    if len(a_list) < 6:
        return None
    return np.array(h_list, dtype=np.float32), np.array(a_list, dtype=np.int64)


def collect_room(room, n_traj, seed0):
    hs, acts, ntraj = [], [], 0
    for t in range(n_traj):
        r = rollout(room, t, seed0 * 1000 + t)
        if r is None:
            continue
        h_seq, a_seq = r
        # 3-step sliding windows; drop last 2 steps
        for s in range(len(a_seq) - 2):
            hs.append(h_seq[s])
            acts.append((a_seq[s], a_seq[s + 1], a_seq[s + 2]))
        ntraj += 1
    print(f"{room}: {ntraj} trajs -> {len(acts)} windows", flush=True)
    return (np.array(hs, dtype=np.float32),
            np.array(acts, dtype=np.int64), ntraj)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train-traj', type=int, default=30)
    ap.add_argument('--n-held-traj', type=int, default=15)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=str, default='s2_data_v2.npz')
    args = ap.parse_args()
    verify_h_dyn()
    train_h, train_seq, held_h, held_seq = [], [], [], []
    for i, room in enumerate(TRAIN_ROOMS):
        h, s, _ = collect_room(room, args.n_train_traj, args.seed * 100 + i)
        train_h.append(h); train_seq.append(s)
        np.savez('s2_data_v2_partial.npz',
                 train_h_dyn=np.concatenate(train_h),
                 train_seq=np.concatenate(train_seq),
                 done_rooms=TRAIN_ROOMS[:i + 1])
    for i, room in enumerate(HELD_ROOMS):
        h, s, _ = collect_room(room, args.n_held_traj, args.seed * 100 + 50 + i)
        held_h.append(h); held_seq.append(s)
    train_h = np.concatenate(train_h); train_seq = np.concatenate(train_seq)
    held_h = np.concatenate(held_h); held_seq = np.concatenate(held_seq)
    np.savez(args.out,
             train_h_dyn=train_h, train_seq=train_seq,
             held_h_dyn=held_h, held_seq=held_seq,
             train_rooms=TRAIN_ROOMS, held_rooms=HELD_ROOMS)
    print(f"TRAIN windows: {len(train_seq)}, HELD windows: {len(held_seq)}", flush=True)
    print(f"train t0 hist: {np.bincount(train_seq[:, 0], minlength=6)}", flush=True)
    print(f"held  t0 hist: {np.bincount(held_seq[:, 0], minlength=6)}", flush=True)


if __name__ == '__main__':
    main()
