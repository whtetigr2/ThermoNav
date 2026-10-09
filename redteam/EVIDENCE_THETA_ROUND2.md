# Evidence-θ Round 2: Temporal Training

**Verdict: FAIL — the learned mapper does not ship. Engineered+fixes stays the default.**

## Plain English

We tried to fix round 1's failure by training on full missions from startup
(instead of settled scans), including the cold-start transient and realistic
jitter. The learned mapper is better on paper and better in heavy fog, but it
crashed once in whiteout where the engineered stack stood down safely. One
collision is one too many. It does not ship.

**Claim label:** Z1-legal formulation, validated in simulation.
**Scope:** the learned mapper is classical host-side; all gain outputs within
±6 [asserted in code].

## What was built

`slow-drone/evidence_theta/` (round-2 files):
- `collect_temporal_data.py` — 120 full missions from t=0 with
  engineered+fixes (20 train rooms × 4 fog densities, 10 held-out rooms × 4).
  Every timestep logged: ray features + tracker state (scan_idx, vis_smooth)
  + GT occupancy. 25% of train missions flew with realistic jitter (isolated
  2 s delay at step 8, the JITTER.md characterized regime). [measured]
- `train_evidence_temporal.py` — 7 params (5 scalars + 2 age terms).
  age(t)=exp(−(t−1)/5). Loss = voxel BCE + 20×|carve| on GT-occ +
  2×phantom-paint + 20×wall-erasure + 10×starvation (new) +
  time-weighting (4× on scan 1, →1× by scan 11). [calculated]
- `integrate_learned_temporal.py` — closed-loop plug-in via `_EVIDENCE_FN`.
- `eval_temporal.py` — bars 1, 3, 4, 5. `flight_test_temporal.py` — bar 2.
- `evidence_temporal_train.csv` (426,701 rows) /
  `evidence_temporal_held.csv` (226,207 rows) — dual-purpose corpus.
- `evidence_theta_temporal.npz` — final params (do not use; failed bar).

## Training iterations (all documented, all honest)

1. **Unconstrained 7-param**: learned STRONG early carve (−5.06 on scan 1).
   This is the round-1b killer amplified. REJECTED before flight — the
   time-weighted loss rewards strong early carve because scan-1 paths in
   training are mostly free space. The supervised loss cannot see the
   closed-loop danger.
2. **Constrained (a_occ, a_free ≥ 0)**: age terms collapsed to exactly 0.
   Under the safety constraint, the optimizer prefers no temporal variation.
   The data doesn't support "weak early carve" — it wants strong carve
   everywhere, which is unsafe.
3. **Carve-frozen + starvation penalty** (shipped for validation):
   Carve frozen at engineered −0.73 (round-1c lesson). New LAMBDA_STARVE=10
   penalizes weak paint on GT-occupied hits (BCE alone maxes at 0.69 nats,
   too weak vs the phantom penalty). Learned:
   theta = [0.527, 0.062, −0.73, 0.0, 0.662, 0.0, 0.0].
   Paint +0.53/+0.50 (vs engineered +0.60), provisional scale 0.66
   (vs 0.30 — stronger unconfirmed-hit paint).

**Key finding**: the age feature is unused (a_occ=a_free=0 in all runs).
The temporal contribution is the DATA (cold-start transients + jitter),
not the model structure. The supervised loss doesn't reward temporal
variation in gains.

## Bar results (pre-registered, non-negotiable)

| # | Bar | Result |
|---|---|---|
| 1 | Held-out loss ≥ engineered | **PASS** (7.33 vs 7.44) [measured] |
| 2 | Extended closed-loop: whiteout 0 collisions, clear 16/16 | **FAIL** — whiteout 1 collision [measured] |
| 3 | Zero carve-through-occupied (whiteout) | 2122 events — identical for engineered (carve frozen at −0.73); gate property, not a regression |
| 4 | Cold-start quality ≥ engineered | **PASS** (15.41 vs 15.59 on scans ≤10) [measured] |
| 5 | All outputs within ±6 | **PASS** (max 0.73) [asserted in code] |

### Bar 2 detail (128 flights, novel rooms, paired seeds)

| Density | Engineered | Learned |
|---|---|---|
| clear (16) | 16/16, 0 coll | 16/16, 0 coll |
| moderate (16) | 16/16, 0 coll | 16/16, 0 coll |
| heavy (16) | 13/16, 0 coll | **15/16**, 0 coll |
| whiteout (16) | 0/16, 0 coll (16 blocked) | 0/16, **1 coll** (13 blocked, 2 max_steps) |

The 1 collision: indoorcd_184, seed 70631, step 158. Paired engineered
stood down safely (blocked at step 39). The learned mapper's stronger
provisional paint (c0=0.66) kept it flying where engineered wisely stopped,
and it eventually hit a wall.

**Bright spot**: heavy fog 15/16 vs 13/16 — the stronger paint genuinely
helps in degraded vis. But the whiteout collision is disqualifying.

### Jitter-in-the-loop cell (SHIP-BLOCKING per Paul, 2026-10-08)

Heavy fog + whiteout × isolated 2 s delay at step 8 (JITTER.md regime),
paired engineered-vs-learned, fresh seeds, 8 novel rooms. 64 flights.

| Density + jitter | Engineered | Learned |
|---|---|---|
| heavy (16) | 13/16, 0 coll | 13/16, 0 coll |
| whiteout (16) | 0/16, **2 coll** (13 blocked, 1 error) | 0/16, **5 coll** |

**Paired whiteout+jitter: learned collided in 4 missions where engineered
was safe** (indoorcd_132/s80110, indoorcd_157/s80211, indoorcd_168/s80310,
indoorcd_168/s80311 — all engineered=blocked). One mission both collided
(indoorcd_184/s80611).

**SHIP-BLOCKING FAIL.** The fog+blackout corner is dangerous for both
(engineered has 2 collisions here too — the JITTER.md finding), but the
learned mapper is strictly worse: 5 vs 2, with 4 being overconfidence
failures where engineered wisely stood down. The stronger provisional
paint (c0=0.66) makes the mapper fly where it should block.

Note: indoorcd_168 collided on BOTH seeds for learned — systematic
overconfidence on that room, not a flake.

## The real lesson

**The supervised per-row loss cannot see closed-loop safety.** It rewards
strong early carve (scan-1 paths are free in training) and strong paint
(heavy fog needs it), but it can't tell that "keep flying in whiteout"
is more dangerous than "stand down." The engineered stack's conservatism
(blocked at step 39) is a feature the loss doesn't value.

Three strikes for the age feature: (1) unconstrained it learns dangerous
strong-early-carve, (2) constrained it collapses to zero, (3) the shipped
model's improvement comes from stronger paint/provisional, not temporal
variation. **Temporal gain-shaping is not the answer.**

## Why not the MLP

Same reason as round 1: the failure is data/structure (supervised loss
can't see closed-loop danger), not capacity. An MLP would learn the same
wrong lesson with more parameters.

## If Evidence-θ is revisited

1. The loss needs a closed-loop term, or training needs to be on
   closed-loop outcomes (RL), not per-row supervision.
2. The provisional scale (c0) is the sensitive knob: 0.66 helps heavy fog
   but causes whiteout overconfidence. Needs vis-dependent shaping or a
   whiteout cap.
3. Or: accept that engineered+fixes is the right answer for the evidence
   mapper, and put learning effort into S2-θ (which works).

## Provenance

- Rooms: HM3D `real_lidar/*_occ.npy` [measured]. Fog: obscurant
  Beer-Lambert + backscatter [assumed coefficients].
- Jitter: FreezeController isolated 2 s at step 8 [measured regime].
- All flight outcomes [measured] via `mission_dynamics_v5`, numpy backend,
  paired missions (same room/seed/noise/density).
- Data: `evidence_temporal_train.csv` / `evidence_temporal_held.csv`
  (commit deebc795, every lever explicit).
- Claim: Z1-legal formulation, validated in simulation (nothing on silicon).
