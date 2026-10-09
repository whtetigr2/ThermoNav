"""Drop-in S2 with learned (h_static, J). Same act() interface as S2Agent.

Loads s2_theta_v2.npz by default (biases=learned h_static (18,), weights=learned J on mask).
The dynamic h_dyn (ray costs, forbidden penalties) is computed identically
to S2Agent.act(); only the static (h, J) are learned.

NOTE: imports S2Agent at module level. If you monkey-patch harness.S2Agent,
import this module BEFORE patching.
"""
import os
import sys
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'thermopilot'))
from harness import S2Agent as _OrigS2Agent, MOVES_6
from gibbs_ising import gibbs_ising

_HERE = os.path.dirname(os.path.abspath(__file__))


class LearnedS2Agent(_OrigS2Agent):
    """S2Agent with learned static Ising params (h_static, J)."""

    def __init__(self, param_file=None, **kw):
        # Init engineered template for moves/constants/_ray_cost, but we
        # override h_static and J below. Pop param_file from kw.
        super().__init__(**kw)
        pf = param_file or os.path.join(_HERE, 's2_theta_v2.npz')
        z = np.load(pf)
        self.h_static = z['biases'].astype(float)
        edges = z['edges']
        J = np.zeros((self.n, self.n))
        for (i, j), w in zip(map(tuple, edges), z['weights']):
            J[i, j] = w
            J[j, i] = w
        self.J = J
        self.param_file = pf
