"""Novel-room generalization benchmark: paired teacher vs learned-v2 flights.

Usage: python3 bench_generalization.py --agent {teacher,learned} --out CSV room:seed [room:seed ...]
Paired by construction: pick_start_goal(occ, seed) gives identical start/goal
for the same (room, seed) regardless of agent.
"""
import sys, os, csv, time, argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2T = os.path.join(REPO_ROOT, 's2_theta')
sys.path.insert(0, os.path.join(REPO_ROOT, 'thermopilot'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'sim'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, S2T)

REAL_DIR = os.path.join(REPO_ROOT, 'rooms')
FIELDS = ['agent', 'room', 'seed', 'noise', 'outcome', 'collision',
          'n_steps', 'path_efficiency', 'n_d5_refusals_s16', 'n_move_refusals',
          'wall_s', 'mission_id']

CFG = {"gravity_envelope": True, "tw_ratio": 2.0, "filter_tau": 0.4,
       "obscurant": {"archetype": "fog", "density": "clear"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True, choices=['teacher', 'learned'])
    ap.add_argument('--out', required=True)
    ap.add_argument('pairs', nargs='+', help='room:seed')
    args = ap.parse_args()

    if args.agent == 'learned':
        from s2_learned import LearnedS2Agent
        import harness
        pf = os.path.join(S2T, 's2_theta_v3.npz')

        class PatchedLearnedV2(LearnedS2Agent):
            def __init__(self, **kw):
                super().__init__(param_file=pf, **kw)

        harness.S2Agent = PatchedLearnedV2

    import numpy as np
    import mission_dynamics_v5 as md

    new_file = not os.path.exists(args.out)
    fout = open(args.out, 'a', newline='')
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for pair in args.pairs:
        room, seed = pair.split(':')
        seed = int(seed)
        noise = 0.15
        occ = np.load(os.path.join(REAL_DIR, f'{room}_occ.npy'))
        mid = f'gen_{args.agent}_{room}_s{seed}_n0.15'
        t0 = time.time()
        try:
            row = md.run_mission_dynamics(occ, room, seed, noise, mid,
                                          backend='numpy', verbose=False,
                                          cfg=dict(CFG), write_traj=False)
            outcome = row['outcome']
            rec = {
                'agent': args.agent, 'room': room, 'seed': seed, 'noise': noise,
                'outcome': outcome,
                'collision': 1 if outcome == 'collision' else 0,
                'n_steps': row.get('n_steps', -1),
                'path_efficiency': row.get('path_efficiency', ''),
                'n_d5_refusals_s16': row.get('n_d5_refusals_s16', ''),
                'n_move_refusals': row.get('n_move_refusals', ''),
                'wall_s': round(time.time() - t0, 1),
                'mission_id': mid,
            }
        except Exception as e:  # noqa: BLE001
            rec = {'agent': args.agent, 'room': room, 'seed': seed,
                   'noise': noise, 'outcome': f'ERROR:{type(e).__name__}',
                   'collision': 0, 'n_steps': -1, 'path_efficiency': '',
                   'n_d5_refusals_s16': '', 'n_move_refusals': '',
                   'wall_s': round(time.time() - t0, 1), 'mission_id': mid}
            outcome = rec['outcome']
        w.writerow(rec)
        fout.flush()
        print(f"[{args.agent}] {room} s{seed}: {outcome} "
              f"steps={rec['n_steps']} ({rec['wall_s']}s)", flush=True)
    fout.close()


if __name__ == '__main__':
    main()
