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

- `s2_theta/` — the trained kernel: `s2_theta_v3.npz` (the 123 params, shipped default — v2 + vacancy hardening), `s2_theta_v2.npz` retained as fallback/baseline, human-readable `params.txt`, training scripts (`train_s2_theta_v2.py`), drop-in agent (`s2_learned.py`, defaults to v3), evaluation + generalization benchmark (`eval_v2.py`, `bench_generalization.py`), training data (`s2_data*.npz`, manifest in `CHECKPOINTS.md`), and full docs (`S2_THETA.md`, `GENERALIZATION.md` — including the honest failure log).
- `thermopilot/` — the navigation harness: `harness.py` (S2 teacher agent, A* planner, D5 safety veto), `ising.py` (Ising helpers), `gibbs_ising.py` (the sampler). See `thermopilot/README.md`.
- `sim/` — the dynamics validation suite (point-mass physics, wind, obscurants, fix validation; Phases 1–4). Entry points in `sim/README.md`. Includes the escalation ladder: `escalation.py` (Option A industry-mirror policy, host-side classical), `escalation_decider.py` (12-spin Z1-legal decider kernel), `test_escalation.py` (128-combo oracle test), `docs/ESCALATION_BUILD.md`, wired into `mission_dynamics_v5.py` via per-step evaluation and `max(shipped, escalation)`. Also includes the link-health layer (graduated 2026-10-11): `link_health.py` (receiver-side HEALTHY/DEGRADED/LOST tracker + channel model), the link rows in `escalation.py` (leaky buckets, leaky hold progress, net-LOST-exposure latch, bucket-drained release), `test_link_health.py` (25 tests), and `LINK_MODEL.md` (full build report + red-team dispositions). Also includes the battery layer (graduated 2026-10-11, after red-team round 2 PASS WITH CAVEATS): `battery.py` (4S 5000 mAh gauge: momentum-theory power, IR sag, stepped time-to-reserve), `battery_safety.py` (Paul's design: one strict flag at reserve, a countdown sized from *measured* draw, RTB → RTC → emergency-land reachability cascade with continuous re-check and abort-down only), the battery rows in `escalation.py` (warn/reserve/critical; warn 0.30 / reserve 0.20 / critical 0.10), `test_battery.py` (14 tests), `test_battery_safety.py` (12 tests), `run_battery_safety_validation.py` (9 missions, 6/6 gates, 0 collisions), and `BATTERY_MODEL.md` (full build report + red-team dispositions). Also includes the nav-health layer (graduated 2026-10-11, after red-team round 5 PASS WITH CAVEATS, documentation-only): `nav_health.py` (host-side classical monitor: HEALTHY/DEGRADED/UNTRUSTED from estimator self-reports only; `pose_trustworthy` True iff HEALTHY), `exec_provenance.py` (executed-tree provenance guard for the data harnesses), the nav rows in `escalation.py` (RTH gate = pose_trustworthy AND HEALTHY; UNTRUSTED is itself a trigger), `test_nav_health.py` (24 tests), `test_exec_provenance.py` (7 tests), `test_nav_checkpoint.py` (checkpoint wiring), `run_nav_health_calibration.py` + `run_nav_health_mission_validation.py`, and `NAV_HEALTH.md` (full build report + red-team dispositions, including the disclosed C2 burst-harvest residual — an accepted limit, not a fixed defect). The battery checkpoint's trust input is now the real monitor by default; the old `nav_trustworthy_stub` survives only as a scripted scenario override for deterministic RTC constructions. All three Phase-1 health slots (battery, link, nav-health) are now modeled — none remains a MISSING stub when its inputs are supplied.
- `rooms/` — 30 HM3D indoor occupancy grids used for training and benchmarking.
- `redteam/` — the red-team record: jitter envelope, real-world jitter research, multi-modal negative, both Evidence-θ negatives, D5 freshness gate. Start at [`REDTEAM.md`](REDTEAM.md).
- `requirements.txt`, `.gitignore` — dependencies and hygiene.

## Quickstart

```python
import numpy as np, sys
sys.path += ['s2_theta', 'thermopilot']
from s2_learned import LearnedS2Agent

agent = LearnedS2Agent()  # loads s2_theta_v3.npz by default
# agent = LearnedS2Agent(param_file='s2_theta/s2_theta_v2.npz')  # v2 fallback
# agent.act(pos, goal, p_occ) -> next waypoint, same interface as the engineered S2
```

Requirements: Python 3.10+, `numpy`, `jax[cpu]`, `optax`, `networkx`, `thrml` (for retraining). Training used `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

To reproduce the generalization benchmark:

```bash
cd s2_theta && python bench_generalization.py --agent learned   # 12 novel rooms
```

## The honest section

- The kernel **matches** its engineered teacher; it does not exceed it. Imitation has a ceiling — and reward-driven training has now been tried twice and ruled out (see below). The next frontier is architecture, not training. (One measured gap: heavy-fog 96-flight stress test, teacher 13/16 vs learned 11/13 — both fail safe.)
- Single-decision agreement with the teacher is only 42.8%, yet closed-loop flight is identical: disagreements live in low-stakes states. Closed-loop behavior is the load-bearing metric.
- Validated in clear air (noise 0.15, HM3D indoor). Whiteout/fog is sensor-limited — the honest answer there is multi-modal sensing (radar), not more training.
- `max|h| = 5.35` sits close to the assumed Z1 rail of 6.0 — flagged for hardware watch.
- We publish our negatives: the full red-team record (killed lines, measured envelopes, known boundaries) is in [`REDTEAM.md`](REDTEAM.md).
- **The stack is blind to walking-speed movers.** A person-scale mover paints one map voxel per scan and moves on; free-space carving (−3.40 log-odds from multi-ray carving vs +0.60 single hits) means the evidence never accumulates. 43/48 collisions in testing, D5 never vetoed a mover. Needs a motion-detection layer — architectural, not a training problem.
- **Land actuation under link loss is UNTESTED at mission level.** The link-health layer escalates a lost link to the gated LAND *action*. When that limit was recorded, descent was blocked by the below-clear guard, which was structurally unsatisfiable from altitude (floor layer pinned occupied by the boundary prior). That guard has since been replaced — in the battery graduation — by the evidence-keyed guard (a floor voxel blocks only with hit support beyond the prior, plus a creep final approach and a three-state below check), and battery-driven landings now actuate at mission level, including a whiteout terminal landing that ended `landed_disarmed`. What remains untested is the *link-loss scenario itself* at mission level: no lost-link mission has been run end-to-end in ThermoNav, so the stack-level claim that a lost-link vehicle ends up on the ground is still not demonstrated — the descent path it would take is now the mission-tested one. (Original analysis in `sim/LINK_MODEL.md`, known limit 6.)
- **The battery layer's "landing floor" is a plan-time threshold, not an outcome.** `LANDING_FLOOR_SOC` (0.05) gates which return leg the cascade picks, on a straight-line projection; measured validation missions land well below their projections (RTB mission: projected arrival soc 0.178, actual final 0.0693). Reachability is straight-line distance with a 1.25 margin factor, re-checked at 1 Hz with abort-down-only — that re-check is the mitigation for the projection's optimism, not a guarantee. The countdown is sized from measured gauge drain, bounded even if the soc gauge stalls (soc-vs-Wh cross-check + 30 s metering bound, red-team round-2 RT2-1).
- **The stack assumes pose is known.** Drift testing shows first collisions at 0.05 m/s drift (~1 m accumulated); a state estimator holding <0.5 m/mission is required. Unaided MEMS IMU cannot do this.

## What we tried after v2 (October 2026)

**Reward training: two clean negatives, both red-teamed.** REINFORCE on the kernel, one variable at a time:
- 100 episodes, coarse per-mission reward → tied the bar (11/13 heavy-fog pairs). Negative.
- 1,000 episodes → same tie. θ moved 3.6× more, margins tripled — the learning happened, the capability didn't. Sampling/time ruled out.
- 100 episodes, dense per-decision rewards (margin-weighted teacher agreement + D5-veto penalty + potential-based shaping) → θ moved 5.5× more than the coarse run, teacher agreement 0.22→0.33, margins tripled. Still 11/13. Credit assignment ruled out.
- Oracle certification of the 1,000-episode kernel: **FLAGGED** — 18% of states distributionally drifted, argmax changed on 49%. The bar's "wash" hid real churn underneath.
- Verdict: the kernel is at its architectural ceiling. v2 stays shipped.

**Teacher audit: clean.** The learned kernel subclasses the teacher — same inputs, same sampler, same fog. The student's hypothesis class even contains the teacher's exact parameters. The 11/13 gap is genuine, not a test leak.

**Fog grounding:** our "heavy fog" ≈ 50 m visibility (modeled, not measured) — roughly 16× below the FAA waiver floor of 800 m. We test harder than anything that legally flies.

## License

Apache-2.0 — see [LICENSE](LICENSE).
