"""Gate 3 (round 2): fly clear canonical 8 with v2 learned S2 via mission_dynamics_v5."""
import sys, os
import numpy as np

SYSROOT = os.path.expanduser('~/workspace/flynav-redteam')
sys.path.insert(0, os.path.join(SYSROOT, 'thermopilot'))
sys.path.insert(0, os.path.join(SYSROOT, 'slow-drone', 'dynamics_sim'))
sys.path.insert(0, os.path.join(SYSROOT, 'slow-drone'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s2_learned import LearnedS2Agent
import harness

PARAM_FILE = os.environ.get('S2_V2_PARAM',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 's2_theta_v2.npz'))


class PatchedLearnedV2(LearnedS2Agent):
    def __init__(self, **kw):
        super().__init__(param_file=PARAM_FILE, **kw)


harness.S2Agent = PatchedLearnedV2

import mission_dynamics_v5 as md
from curated_v2 import GRADED

REAL_DIR = os.path.join(SYSROOT, 'slow-drone', 'real_lidar')


def main():
    results = []
    for (room, seed, noise) in GRADED:
        occ = np.load(os.path.join(REAL_DIR, f'{room}_occ.npy'))
        cfg = {"gravity_envelope": True, "tw_ratio": 2.0, "filter_tau": 0.4,
               "obscurant": {"archetype": "fog", "density": "clear"}}
        mid = f'learnedv2_clear_{room}_s{seed}'
        row = md.run_mission_dynamics(occ, room, seed, noise, mid,
                                      backend='numpy', verbose=False,
                                      cfg=cfg, write_traj=False)
        results.append((mid, row['outcome'], row.get('path_efficiency', -1)))
        print(f"{mid}: {row['outcome']}", flush=True)
    n_ok = sum(1 for _, o, _ in results if o == 'success')
    n_coll = sum(1 for _, o, _ in results if o == 'collision')
    print(f"\nFLIGHT V2: {n_ok}/8 success, {n_coll} collisions")
    for mid, o, pe in results:
        print(f"  {mid}: {o} (path_eff={pe})")


if __name__ == '__main__':
    main()
