# Multi-modal S2 test — does stochastic sampling earn its place?

**Verdict: NO. Argmax wins, decisively. Honest negative.**

## Protocol

Paired flights, novel rooms (outside training {071,004,131}, held-out {185,128},
and all previously benchmarked rooms), matched seeds/starts/goals, clear air,
noise 0.15. Two arms on the trained v2 kernel:

- **argmax** — deterministic: take the highest-probability S2 action
- **stochastic** — sample each action from the S2 posterior (`eval/s2_stochastic.py`)

Pre-registered bar: stochasticity earns its place ONLY with a behavioral win
(higher success or lower cost at equal safety), separated from noise on paired runs.

## Results [measured]

| Arm | Flights | Success | Collisions |
|---|---|---|---|
| argmax | 22 | **22/22** | 0 |
| teacher (control) | 22 | 22/22 | 0 |
| stochastic | 9 | **0/9** | 0 |

On the 9 paired rooms both arms flew: argmax succeeded in 12–28 steps every
time; stochastic hit `max_steps` (200, wandering) every time. Zero collisions
either way — stochastic fails by dithering, not by crashing.

## Diagnosis

Sampling the posterior every decision flip-flops between near-equal actions and
never commits to a coherent path. The agent wanders until the step cap. Argmax
commits. (Note: the benchmark agent errored on a runtime hiccup after 9
stochastic runs; 9/9 paired failures vs 9/9 paired successes needs no more n.)

## Decision

**Ship argmax.** The S2 posterior's stochasticity does not earn its place.
`kernel/s2_learned.py` stays deterministic. This was the last open question on
the S2-θ kernel — it is now closed.

Raw data: `results/multimodal_benchmark.csv`. Analysis: `eval/analyze_multimodal.py`.
Benchmark: `eval/bench_multimodal.py`. Stochastic variant (not shipped):
`eval/s2_stochastic.py`.

**Claim: Z1-legal formulation, validated in simulation.**
