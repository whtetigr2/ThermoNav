# ThermoNav

A **trained thermodynamic navigation kernel** for drones — 123 parameters, 492 bytes, that flies.

ThermoNav's decision-maker is an 18-spin Ising model (6 directions × 3 steps of lookahead) trained with THRML, Extropic's JAX-based simulator for probabilistic computing hardware. It learns by imitating expert flight demonstrations through contrastive KL-gradient estimation — the same Gibbs sampling that will one day run natively on a thermodynamic chip.

## Headline results

| | |
|---|---|
| Parameters | **123** (18 biases + 105 couplings) — 492 bytes |
| Novel-room flights | **24/24 success, 0 collisions** (12 rooms never seen in training) |
| vs engineered baseline | **bit-identical** on all metrics across 16 paired clear-air flights; 96-flight stress test found one real gap — heavy fog: teacher 13/16, learned 11/13 (both fail safe, zero collisions either way) |
| Hardware legality | Z1-legal by construction: degree ≤ 15, \|h\| ≤ 5.35, \|J\| ≤ 1.14 |

**Claim label: Z1-legal formulation, validated in simulation.** This has not been executed on Z1 silicon and is not a validated aircraft — it is software-in-the-loop simulation of a custom navigation stack. Every number in the docs carries its provenance (measured / calculated / assumed).

## Layout

- `s2_theta/` — the trained kernel: `s2_theta_v2.npz` (the 123 params), human-readable `params.txt`, training scripts (`train_s2_theta_v2.py`), drop-in agent (`s2_learned.py`, defaults to v2), evaluation + generalization benchmark (`eval_v2.py`, `bench_generalization.py`), training data (`s2_data*.npz`, manifest in `CHECKPOINTS.md`), and full docs (`S2_THETA.md`, `GENERALIZATION.md` — including the honest failure log).
- `thermopilot/` — the navigation harness: `harness.py` (S2 teacher agent, A* planner, D5 safety veto), `ising.py` (Ising helpers), `gibbs_ising.py` (the sampler). See `thermopilot/README.md`.
- `sim/` — the dynamics validation suite (point-mass physics, wind, obscurants, fix validation; Phases 1–4). Entry points in `sim/README.md`.
- `rooms/` — 30 HM3D indoor occupancy grids used for training and benchmarking.
- `redteam/` — the red-team record: jitter envelope, real-world jitter research, multi-modal negative, both Evidence-θ negatives, D5 freshness gate. Start at [`REDTEAM.md`](REDTEAM.md).
- `requirements.txt`, `.gitignore` — dependencies and hygiene.

## Quickstart

```python
import numpy as np, sys
sys.path += ['s2_theta', 'thermopilot']
from s2_learned import LearnedS2Agent

agent = LearnedS2Agent(param_file='s2_theta/s2_theta_v2.npz')
# agent.act(pos, goal, p_occ) -> next waypoint, same interface as the engineered S2
```

Requirements: Python 3.10+, `numpy`, `jax[cpu]`, `optax`, `networkx`, `thrml` (for retraining). Training used `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

To reproduce the generalization benchmark:

```bash
cd s2_theta && python bench_generalization.py --agent learned   # 12 novel rooms
```

## The honest section

- The kernel **matches** its engineered teacher; it does not exceed it. Imitation has a ceiling — the next frontier is reward-driven training, not better demonstrations. (One measured gap: heavy-fog 96-flight stress test, teacher 13/16 vs learned 11/13 — both fail safe.)
- Single-decision agreement with the teacher is only 42.8%, yet closed-loop flight is identical: disagreements live in low-stakes states. Closed-loop behavior is the load-bearing metric.
- Validated in clear air (noise 0.15, HM3D indoor). Whiteout/fog is sensor-limited — the honest answer there is multi-modal sensing (radar), not more training.
- `max|h| = 5.35` sits close to the assumed Z1 rail of 6.0 — flagged for hardware watch.
- We publish our negatives: the full red-team record (killed lines, measured envelopes, known boundaries) is in [`REDTEAM.md`](REDTEAM.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
