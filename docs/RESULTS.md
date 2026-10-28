# Measured results

The digest. Every figure is **simulation-measured** unless labeled
otherwise, and each row points at the full record in this repository.
"Review" is the lab's own red-team process (doctrine-blind re-derivation
from raw artifacts); verdicts include their caveats by design.

## Flight kernel (S2-θ v3)

| Result | Value | Record |
|---|---|---|
| Parameters | 123 (18 biases + 105 couplings), 492 bytes of parameters, plus its fixed wiring | `s2_theta/S2_THETA.md` |
| Novel-room flights (12 HM3D rooms never in training) | 24/24 success, 0 collisions | `s2_theta/GENERALIZATION.md` |
| vs engineered teacher, 16 paired clear-air flights | bit-identical outcomes | `REDTEAM.md` |
| Substrate ablation of the trained parameters (2026-10-26) | trained-parameters claim **WITHDRAWN as filed** — decision-level only survives: T agreement 41.25% vs best control 21.0% (B-1 PASS); margin separation B-2 FAIL; closed-loop separation B-3 FAIL | `s2_theta/ABLATION.md` |
| Heavy-fog stress (96 paired flights) | teacher 13/16, learned 11/13 — both fail safe, 0 collisions | `REDTEAM.md` |
| Hardware caps and legality | Within Z1 caps, measured: degree ≤ 15, max\|h\| = 5.35, max\|J\| = 1.14 (caps: 16 / 6 / 6 — the \|h\| cap is [ASSUMED], the \|J\| cap is documented by Extropic). Chip legality is **not established**: the shipped kernel graph is 6-chromatic (measured 2026-10-27), so two-colouring is unavailable as shipped, and a mediated lowering of the minimized core was built and killed on sampler distortion | `s2_theta/S2_THETA.md` (correction of 2026-10-28) |
| Reward-driven retraining (REINFORCE, 3 variants) | negative — capability tie at up to 5.5× parameter movement; architecture ceiling, not a training gap | `README.md` §"What we tried" |

## Wavefront planner

| Result | Value | Record |
|---|---|---|
| 2D field vs true BFS distance (published pipeline) | corr **0.9991180978469811**, reproduced bit-identically in the 2026-10-22 verifiability re-run | `wavefront/` |
| Same field, raw, no host smoothing | corr 0.9982 — the field is the sampler's, not the smoother's | `wavefront/` |
| 3D ceilinged mazes (40×40×16, 31,752 spins each) | corr 0.980 / 0.984 / 0.994; goal reached 3/3 (65 / 65 / 83 steps) | `wavefront/` |
| J = 0 ablation | field is noise (corr 0.09), routing fails — the coupled dynamics are load-bearing | `wavefront/` |

## Safety machinery

| Line | Result | Record |
|---|---|---|
| D5 freshness gate | fog+jitter and whiteout+jitter collision cells → 0 (from 1 and 2); 28 clear-air missions, 0 false fires | `redteam/FRESHNESS_GATE.md` |
| Escalation decider | 12-spin decider kernel (degree 2, within Z1 caps), 128/128 vs policy oracle | `sim/docs/ESCALATION_BUILD.md` |
| Battery safety validation | 9 missions, 6/6 gates, 0 collisions; whiteout terminal landing disarmed at min soc 0.1014 | `sim/BATTERY_MODEL.md` |
| Link-health layer | 25/25 tests; red-team rounds 1–3 PASS WITH CAVEATS | `sim/LINK_MODEL.md` |
| Nav-health layer | healthy arms 96/96 zero false flags; one disclosed paired-cell FAIL (documented, conservative direction) | `sim/NAV_HEALTH.md` |
| Dynamic-object layer | paired 48-cell mover sweep: 43/48 collisions → **7/48**, 0 cells broken | `sim/DYNLayer_RESULTS.md` |
| Compute-jitter envelope | 182 missions: clear air holds to 5 s stalls; ≈0.5 Hz perception fails safe (wanders, never collides) | `redteam/JITTER.md` |

## Red-team totals

- Individual lines above: each carries ≥ 1 red-team round; several
  carry 3–5 (battery, link, nav-health), with FAIL rounds fixed and
  re-reviewed, all in `REDTEAM.md`.
- A separate capstone review of the wider research program (12 units,
  397 claims, October 2026) returned 11 units SURVIVE WITH CAVEATS and
  1 FAIL — the failure (an early full-stack integration whose pose
  input was not what its headline implied) is the origin of the
  self-location rules in `docs/OPEN_PROBLEMS.md`, and its revoked
  results are labeled revoked wherever they appear in the record.
- The substrate migration map (how much of the stack can live on the
  substrate, layer by layer) survived its red team with revisions;
  two of its claims stand as PARTIAL with scoped wording.

## Perception front-end (host-side classical, not thermodynamic)

Evaluated alongside the stack, not part of the substrate claim: a
standard published pipeline (ORB-SLAM3, monocular-inertial) on real
EuRoC MAV footage scored ATE 0.0708 m (MH_01_easy, 1.14× the published
median) and 0.0370 m (V1_01_easy, 0.76×) — inside the field-standard
bar the lab adopted (ATE ≤ 0.10 m and within 2× published). Single
executions, the two easiest sequences; the same pipeline in our
simulated worlds fragments (tracking losses) and does **not** yet meet
the sim gate — see `docs/OPEN_PROBLEMS.md`.

## Sibling demonstration (same substrate discipline, not a drone)

The warehouse-routing program applied the identical method —
substrate formulation, pre-registered bars, red-teamed LEGO stack —
to multi-robot warehouse traffic in THRML, and produced a complete
after-action package (October 2026): the substrate formulation is
legal and correct, with a hard measured cliff (no sampled trip of
length ≥ 38 ever solved, by any of four substrate methods tried);
the exact audit shows valid routes *are* the ground state but the
cold equilibrium is too entropic to harvest. It is mentioned here
because it is the same discipline producing a negative result at full
rigor — the method's failures are as measured as its successes. The
full package lives in the lab's research record, not this repository.
