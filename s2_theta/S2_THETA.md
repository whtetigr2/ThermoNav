# S2-θ: The Smallest Learnable Thermodynamic Decision Kernel

**Claim label:** Z1-legal formulation, validated in simulation.
(Never silicon, never real aircraft.)

## What this is

The first **trained** (not hand-engineered) thermodynamic kernel in the stack.
It learns the S2 decision Ising's static parameters — 18 biases + 105 couplings
— from expert demonstration, using THRML's official training path
(`IsingTrainingSpec` + `estimate_kl_grad`).

**Size:** 123 parameters, 492 bytes of parameters (fp32), plus its fixed
wiring (the 105 edges it runs on). Fits in L1 cache. Paul asked for
"triple digit kilobytes" — this is 0.48 KB.

## Design (frozen)

**Architecture:** 18-spin Ising (6 directions × 3 timesteps), Z1-legal mask
from the engineered S2 (max degree 15, |h|≤4.3, |J|≤1.0 — measured, already legal).

**Visible/latent split:**
- Visible: t=0's 6 spins, clamped to expert's one-hot action.
- Latent: t=1,2's 12 spins, sampled in positive phase.
- Negative phase samples all 18.

**Conditioning:** h_dyn(x) (18-dim ray-cost fields) added to learned biases
per-sample via vmap over the batch. The gradient flows to h_static only;
h_dyn is fixed input.

**Colorings:** DSATUR on the 18-node graph (6 colors; one-hot blocks are
6-cliques). Positive phase reuses coloring minus visible nodes (per THRML audit).

**Data:** 5,968 training samples from engineered S2Agent on rooms
{071, 004, 131}; 1,182 held-out samples from rooms {185, 128} (room-level
holdout, no geometry leakage). Contexts: real room geometry + real perception
model (`noisy_perception_with_counts`) + random positions/goals. Expert:
S2Agent.act() with 200 sweeps (deployed config).

**Training:** Adam 3e-3, 30 epochs, batch 64. Per-sample KL grad via vmap'd
`estimate_kl_grad`. Clip |h|,|J|≤6 after each step (Z1 projection).
Two inits: warm-start (engineered) and small-random. Kept the better.

## Results (measured)

| Metric | Value | Provenance |
|---|---|---|
| Params | 123 (18+105) | calculated |
| Bytes (fp32) | 492 bytes of parameters, plus its fixed wiring | calculated |
| Max |h| | 1.33 | measured |
| Max |J| | 1.30 | measured |
| Max degree | 15 | measured |
| Within Z1 caps | YES — degree 15 ≤ 16, \|h\| ≤ 6, \|J\| ≤ 6 | measured |
| Held-out agreement (learned vs expert) | 39.0% | measured |
| Teacher self-agreement (noise ceiling) | 51.4% | measured |
| Pure teacher agreement | 47.3% | measured |
| Closed-loop flight (clear canonical 8) | 7/8, 0 collisions | measured |
| Engineered baseline flight | 8/8, 0 collisions | measured |

Caps, stated exactly: the shipped kernel's measured maxima are degree 15,
max|h| = 5.35, max|J| = 1.14. The |h| ≤ 6 cap is [ASSUMED]; the |J| ≤ 6 cap
is documented by Extropic. **Colouring and placement are not claimed in
this repository.**

**Agreement calibration:** The teacher is highly stochastic (51% self-agreement
with different seeds). The learned model achieves 39% = 76% of the noise
ceiling, 82% of pure teacher.

**Flight:** 7/8 success, 0 collisions. The 1 miss was `goal_blocked` (safe,
not collision) on indoorcd_071_s1_n0.3. Five missions had identical paths to
engineered (S2 not called in direct mode); the learned S2 was exercised on 3.

**Ablation:** Random-init (39.0%) beats warm-init-trained (32.3%). Training
from scratch works; warm-start training drifted away from the teacher
(bias drift 0.60, weight drift 0.98) and hurt performance.

## Honesty section: what didn't work

**1. The 18-spin latent structure is not identifiable from action data.**
The learner correctly recovered the one-hot penalty (J=-1.2 vs teacher -1.0)
but got temporal couplings WRONG SIGN (+0.84 vs teacher -0.03). The t=1,2
latents are never observed; the data doesn't constrain them. The learner
found a different latent structure that matches t=0 decisions.

**Diagnostic:** A hybrid (learned one-hot/biases + teacher temporal) succeeds
on the failed mission, confirming temporal is the issue. But the hybrid is
not fully learned.

**2. Warm-start training hurt.** Starting from engineered params and training
made agreement WORSE (47.3% → 32.3%). The KL gradients pulled the model away
from a good solution. Random-init training (39.0%) was better.

**3. Frozen-temporal (63 params) didn't help.** Freezing temporal at zero gave
32.4% agreement — worse than full 123-param. The temporal couplings, even
with wrong sign, provide useful capacity.

**4. Flight gate missed (7/8 vs 8/8).** The learned model is safe (0 collisions)
but not quite at engineered reliability. The miss was a safe `goal_blocked`,
not a collision.

**5. The training data has a distribution shift.** Expert actions were collected
via offline context sampling, not closed-loop. The S2 is rarely called in
clear missions (0 calls in test), so the flight test barely exercises it.

## What this proves

1. **THRML training works on a real workload.** `estimate_kl_grad` with vmap'd
   per-sample conditioning trains a 123-param Ising to 76% of the noise ceiling.
2. **Tiny models can work.** 492 bytes of parameters, plus its fixed wiring; within Z1 caps (see the results table), flies safely.
3. **The pipeline is honest.** We report the negative (doesn't beat teacher)
   and the identifiability issue.

## What remains

- The temporal/latent structure needs a different training signal (or a
  smaller 6-spin model).
- Closed-loop data collection (DAgger-style) for on-policy contexts.
- Real HM3D/Habitat data (currently using engineered expert on real rooms).

## Files

- `train_s2_theta.py` — training script (THRML, vmap, both inits)
- `s2_theta.npz` — final learned params (random-init, 123 params)
- `s2_learned.py` — `LearnedS2Agent`, drop-in S2Agent replacement
- `collect_data.py` — expert data collection
- `eval.py` — agreement, Z1 checks
- `flight_test.py` — closed-loop validation
- `model_def.py` — graph, mask, colorings

---

# Round 2: n-step realized-future supervision (2026-10-08)

**Claim label:** Z1-legal formulation, validated in simulation.

## The diagnosis it tested

Round 1's testable prediction: the temporal couplings learned the wrong sign
(+0.84 vs teacher -0.03) because t=1,2 never touched data — "you can't learn
what you never see." Round 2 supervises the FULL 18-spin sequence with the
expert's realized future: at decision step t, clamp t=0 to a_t, t=1 to the
expert's actual a_{t+1}, t=2 to its actual a_{t+2}. The latents stop being latent.

## Design changes vs round 1

- **Data:** 4,337 training windows from expert TRAJECTORY rollouts (not offline
  contexts): 3-step sliding windows (h_dyn(x_t), a_t, a_{t+1}, a_{t+2}) over
  closed-loop S2Agent trajectories on rooms {071,004,131}; 1,537 held-out
  windows from rooms {185,128} (same room-level holdout). Drop last 2 steps
  per trajectory (no future). New: `collect_traj_data.py`, `s2_data_v2.npz`.
- **Model:** fully-visible conditional Ising — all 18 spins clamped in the
  positive phase. Positive moments EXACT from data (no positive-phase sampling).
  Negative phase: THRML block-Gibbs on all 18 spins, per-sample fields
  h_total = h_static(θ) + h_dyn(x), vmapped over batch.
- **Gradient:** manual fully-visible KL gradient —
  ∂KL/∂b_i = ⟨s_i⟩_model − ⟨s_i⟩_data,
  ∂KL/∂w_ij = ⟨s_i s_j⟩_model − ⟨s_i s_j⟩_data (on Z1 mask) —
  passed to optax Adam (descent), same convention as round 1's estimate_kl_grad.
  (estimate_kl_grad requires non-empty positive blocks; the manual gradient IS
  the KL gradient for fully-visible.) Z1 projection (|h|,|J|≤6) after each step.
- **Training:** Adam 3e-3, 30 epochs, batch 64, 32 negative samples/step
  (up from 16 — cleaner gradients for random-init). Warm-start AND random-init.
  New: `train_s2_theta_v2.py`. `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Results (measured)

**Headline diagnostic — THE SIGN FLIPPED (warm-init):**

| | teacher | round 1 (latent) | round 2 warm | round 2 random |
|---|---|---|---|---|
| mean J_temporal | -0.030 | +0.842 (t0→t1) | **-0.094** | +0.100 |
| mean J_onehot | -1.000 | -1.200 | **-1.054** | -0.528 |

Warm-init n-step supervision learns the CORRECT temporal sign. **It was data,
not architecture.** (Random-init still struggles — see honesty section.)

| Metric | Round 1 | Round 2 (warm) | Round 2 (random) | Provenance |
|---|---|---|---|---|
| Held-out agreement (t=0) | 39.0% | **42.0%** † | 38.4% | measured |
| Closed-loop (clear 8) | 7/8, 0 coll | **8/8, 0 coll** | — | measured |
| Engineered baseline | 8/8, 0 coll | 8/8, 0 coll | — | measured |
| Max \|h\| | 1.33 | 5.35 | 2.39 | measured |
| Max \|J\| | 1.30 | 1.14 | 0.68 | measured |
| Max degree | 15 | 15 | 15 | measured |
| Within Z1 caps (degree ≤ 16, \|h\| ≤ 6 [ASSUMED cap], \|J\| ≤ 6 documented by Extropic; colouring and placement not claimed in this repository) | YES | YES | YES | measured |
| Params / bytes | 123 / 492 bytes of parameters, plus fixed wiring | 123 / 492 bytes of parameters, plus fixed wiring | 123 / 492 bytes of parameters, plus fixed wiring | calculated |

> † **Correction (2026-10-26):** the round-2 (warm) held-out agreement in
> the table above was originally filed as **42.8%** (graduation record).
> Re-measured 2026-10-26 with the shipped `eval_v2.py` at this commit:
> **42.0% (168/400 windows)** for `s2_theta_v2.npz` (= `s2_theta_v2_warm.npz`).
> The 42.8% figure is not reconstructible from the shipped record. The
> derived sentence below is corrected with it: "the 42.8% is 83% of the
> noise ceiling" → "the 42.0% is 82% of the noise ceiling" (42.0/51.4).
> No other figure in this table changed.

**Flight gate PASSED:** 8/8 success, zero collisions — matches the engineered
baseline. (Agreement context: teacher self-agreement 51.4%, pure teacher 47.3%
— the 42.0% is 82% of the noise ceiling.)

## Honesty section: what didn't work / caveats

1. **Random-init still doesn't converge in 30 epochs** (38.4%, weak one-hot
   -0.528, temporal still +0.10). Fully-visible supervision NEEDS a good init
   (or more epochs) — the one-hot structure is hard to discover from scratch
   when the negative sampler starts flat. Round 1's "random beats warm" does
   NOT replicate here; the setups differ (latent vs fully-visible).
2. **Warm-init max|h| = 5.35** — legal (≤6.0) but close to the rail. Less
   headroom than round 1's 1.33. The biases drifted up during training.
   Worth watching if this goes near hardware.
3. **The "correct" temporal sign deserves a caveat.** The learner fits the
   MARGINAL distribution of realized action sequences, while the teacher's
   weak temporal couplings work through DYNAMIC imagined rollouts. The
   learned -0.094 matches the teacher's -0.030 in sign and order, but it's a
   different object doing a similar job. The flight result (8/8) is the real
   validation, not the parameter match.
4. **S2 call frequency in flight test:** same caveat as round 1 — S2 is not
   called on every mission step (mode gate). The 8/8 exercises the learned
   model where the stack actually invokes S2.
5. **No outcome weighting, no DAgger yet** — LEGO order. Those are round 3
   candidates if we want to push past the teacher.

## Files (round 2)

- `collect_traj_data.py` — trajectory rollouts → `s2_data_v2.npz`
- `train_s2_theta_v2.py` — fully-visible training (manual KL grad, THRML sampler)
- `s2_theta_v2.npz` — final learned params (warm-init, 123 params); retained as fallback/baseline
- `s2_theta_v3.npz` — SHIPPED DEFAULT (2026-10-11): v2 + uniform t=0 field shift (+1.0)
  for vacancy hardening. 204/204 decision agreement vs v2 (exact enumeration);
  min t=0-vacancy margin -0.900 -> +1.100. MD5 e67aaed07c7492f1f2239deecfd1c6cb.
  Win is fault tolerance, not better flight. Known tradeoffs (red-teamed):
  more decisive on marginal inputs; one-hot margin thins 2.91 -> 0.91.
- `s2_theta_v2_warm.npz`, `s2_theta_v2_random.npz` — per-init checkpoints
- `eval_v2.py` — agreement + temporal diagnostic + Z1
- `flight_test_v2.py` — closed-loop gate for v2 baseline (uses `s2_learned.LearnedS2Agent`)
- `flight_test_v3.py` — closed-loop gate for v3 SHIPPED DEFAULT
- `S2_THETA.md` — this file (round-2 section appended)
