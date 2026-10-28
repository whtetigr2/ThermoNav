# S2-θ Generalization Scoreboard

**Claim label:** Formulated for Z1-class thermodynamic hardware, within
measured caps; validated in simulation only. (Label corrected
2026-10-28 — see the correction note at the top of `S2_THETA.md`.)

## Question
Does the learned S2-θ v2 kernel (123 params, 492 bytes of parameters, plus
its fixed wiring) generalize to room
geometries it never trained on? The bar, per Paul: *"match the teacher where
the teacher flies"* — and only where a valid solution exists.

## Method [measured]
- **Paired flights:** learned v2 vs engineered S2, identical (room, seed, noise).
  `pick_start_goal(occ, seed)` is deterministic, so start/goal are identical
  across the pair by construction. Same physics cfg as rounds 1–2
  (gravity envelope, T/W 2.0, filter_tau 0.4, clear fog, noise 0.15),
  `mission_dynamics_v5`, numpy backend.
- **Novelty guarantee:** v2 trained on rooms {071, 004, 131}; held-out agreement
  eval used {185, 128}. All five excluded from this benchmark. The 12 rooms
  below were never touched by training or held-out eval.
- **Exclusion criterion (Paul's refinement):** a (room, seed) counts toward the
  scoreboard only if the ENGINEERED teacher succeeds there. If the teacher
  can't fly it, it's not a generalization gap — it's an unsolvable room.
  Teacher screening: 12 rooms × 2 seeds = 24/24 success → **zero exclusions.**
- **Benchmark set** (first 8 rooms passing screening, deterministic order):
  021, 023, 036, 040, 042, 048, 053, 063 × seeds {0, 1} = 16 pairs.
- **Extended bench:** +102, 105, 108, 112 (also teacher-screened 2/2) × 2 seeds.

## Paired results — benchmark 16 [measured]

| room | seed | teacher | learned | t_steps | l_steps | t_path_eff | l_path_eff | t_D5 | l_D5 |
|---|---|---|---|---|---|---|---|---|---|
| indoorcd_021 | 0 | success | success | 28 | 28 | 1.0544 | 1.0544 | 0 | 0 |
| indoorcd_021 | 1 | success | success | 42 | 42 | 0.9335 | 0.9335 | 0 | 0 |
| indoorcd_023 | 0 | success | success | 23 | 23 | 1.0690 | 1.0690 | 0 | 0 |
| indoorcd_023 | 1 | success | success | 28 | 28 | 0.9155 | 0.9155 | 0 | 0 |
| indoorcd_036 | 0 | success | success | 20 | 20 | 1.1413 | 1.1413 | 1 | 1 |
| indoorcd_036 | 1 | success | success | 34 | 34 | 0.9908 | 0.9908 | 0 | 0 |
| indoorcd_040 | 0 | success | success | 24 | 24 | 0.7229 | 0.7229 | 0 | 0 |
| indoorcd_040 | 1 | success | success | 27 | 27 | 1.0853 | 1.0853 | 0 | 0 |
| indoorcd_042 | 0 | success | success | 18 | 18 | 1.1242 | 1.1242 | 0 | 0 |
| indoorcd_042 | 1 | success | success | 34 | 34 | 0.9908 | 0.9908 | 0 | 0 |
| indoorcd_048 | 0 | success | success | 17 | 17 | 1.1146 | 1.1146 | 0 | 0 |
| indoorcd_048 | 1 | success | success | 44 | 44 | 0.7617 | 0.7617 | 1 | 1 |
| indoorcd_053 | 0 | success | success | 36 | 36 | 0.8703 | 0.8703 | 2 | 2 |
| indoorcd_053 | 1 | success | success | 33 | 33 | 1.0226 | 1.0226 | 0 | 0 |
| indoorcd_063 | 0 | success | success | 21 | 21 | 1.1101 | 1.1101 | 1 | 1 |
| indoorcd_063 | 1 | success | success | 25 | 25 | 1.1105 | 1.1105 | 0 | 0 |

**Totals [measured]:** teacher 16/16 success, 0 collisions; learned 16/16
success, 0 collisions. Mean step-count difference: 0.0. Mean path efficiency:
1.0011 both.

## Extended bench [measured]
Learned on 102, 105, 108, 112 × {0,1}: **8/8 success, 0 collisions**
(teacher screened 8/8 on the same pairs). Combined novel-room record:
**learned 24/24, teacher 24/24, zero collisions either side.**

## Verdict
**No generalization gap.** The learned kernel matches the teacher exactly on
every recorded metric across 16 paired novel-room flights, and succeeds on
all 24 novel-room flights total. Phase 2 (DAgger loop) is NOT triggered —
there is no gap to close.

## Honest notes
- The paired flights share identical S2 sampler seeds (mission seed +
  step_count), so the Gibbs noise realizations are identical across the pair.
  The finding is: under matched stochasticity, the learned kernel (492 bytes
  of parameters, plus its fixed wiring) is
  behaviorally indistinguishable from the engineered teacher on these rooms.
  The parameters genuinely differ (learned max|h|=5.35 vs engineered values) —
  this is parameter compression, not a copy.
- Single-decision agreement on held-out rooms was 42.0% (round 2;
  corrected 2026-10-26 — originally filed as 42.8% in the graduation
  record; re-measured with the shipped `eval_v2.py` at this commit =
  42.0%, 168/400 windows; the 42.8% figure is not reconstructible from
  the shipped record), yet
  closed-loop behavior is identical here: the disagreements live in
  low-stakes states that don't change flight outcomes. Closed-loop is the
  load-bearing metric; agreement is diagnostic.
- Scope: clear-air, noise 0.15, rooms from the same HM3D indoor distribution
  as training. Whiteout/heavy generalization not tested (sensor-limited regime;
  honest answer is multi-modal sensing, not more S2 training).

> **Correction (2026-10-28):** two sentences in the notes above are
> re-scoped by the 2026-10-26 substrate ablation (`ABLATION.md`).
> (1) "The parameters genuinely differ … this is parameter compression,
> not a copy" — the parameters do genuinely differ (measured), but
> parameter-level difference no longer carries a claim: the
> trained-parameters claim is withdrawn as filed, and what survives is
> decision-level. (2) "Closed-loop is the load-bearing metric;
> agreement is diagnostic" — on the shipped canonical-8 gate the
> ablation measured the opposite separation: every control arm also
> flew 8/8, so closed loop separated nothing (B-3 FAIL), while
> decision agreement is the metric that separated the trained kernel
> from destroyed-structure controls (41.25% vs 21.0% — B-1 PASS). The
> paired-flight record in this document stands as measured; what it
> evidences is behavioural equivalence on these rooms under matched
> stochasticity, not parameter-level credit.
- Raw rows: `gen_benchmark.csv` (paired scoreboard), `gen_screen.csv`
  (teacher screening incl. extended rooms).

## If a gap appears later
Re-screen novel rooms with the teacher, fly the learned kernel paired, and
only then consider DAgger — never train on the test set.
