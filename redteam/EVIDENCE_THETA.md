# Evidence-θ Round 1: Learning the Perception Evidence Mapping

**Verdict: FAIL — the learned mapper does not ship. Engineered+fixes stays the default.**

## Plain English

We tried to replace the hand-set "lidar ray → map evidence" numbers with learned ones. The learned version is better on paper (held-out accuracy 4.95 vs 5.41) but **crashes the drone in whiteout**: 6 collisions vs the engineered stack's 0. It does not ship. The failure taught us something real: you can't learn these numbers in isolation — they were co-tuned with the system's startup behavior, and our training didn't capture that.

**Claim label:** Z1-legal formulation, validated in simulation.
**Scope:** the learned mapper is classical host-side; all gain outputs within ±6 [asserted in code].

## What we built

`slow-drone/evidence_theta/`:
- `collect_evidence_data.py` — per-ray supervision from 30 HM3D rooms × 4 fog densities. 78k train rows (20 rooms), 39k held-out rows (10 rooms). Each row: visibility, carve/paint geometry, ground-truth occupancy labels. [measured]
- `train_evidence.py` — learns 5 scalars: clear-air hit-paint `cc`, hit-paint visibility-slope `s_occ`, clear-air carve `fc`, carve visibility-slope `s_free`, provisional scale `c0`. Loss = voxel BCE + `20×|carve|` on GT-occupied voxels + `2×positive-paint` on GT-free voxels + `20×negative-paint` on GT-occupied voxels (round 1b fix) + per-param L2 toward engineered. [calculated]
- `eval_evidence.py` — bars 1, 3, 4.
- `integrate_learned.py` — drop-in learned-gains wrapper (opt-in via `_EVIDENCE_FN` hook in `mission_dynamics_v5.py`; default `None` = engineered, verified unchanged by control run).
- `flight_test_learned.py` — bar 2: 8 GRADED missions × {clear, whiteout}.
- `evidence_theta.npz` — final (round 1c) params. `evidence_data_*.npz` — training data.

## Rounds and diagnoses

**Round 1a** (5-param, BCE + carve-safety only):
- Learned: hit-paint +0.44 (clear) / −0.61 (whiteout); carve −2.88/−2.82.
- Bar 2: clear 7/8 (1 collision), whiteout 1/8 (**7 collisions**). FAIL.
- Diagnosis: whiteout hits are 14:1 phantoms:real. BCE alone learned negative whiteout paint, erasing the few real close-range returns (p_survive(2m)=0.45). Loss bug: no penalty for negative-paint wall erasure.

**Round 1b** (+`20×` wall-erasure penalty on negative paint):
- Learned: hit-paint +0.28/+0.00 (whiteout defers to confirmation — correct); carve −2.88/−2.82.
- Bar 2: clear 7/8 (0 coll), whiteout 1/8 (**7 collisions**). FAIL.
- Diagnosis: the carve. `vis_smooth` initializes at 1.0, so the FIX-2 confirmation gate is open for exactly one scan. Engineered −0.73 is harmless there; learned −2.82 erases a wall voxel in a single scan. Training used pre-settled tracker state (3 burn-in scans) and never saw this transient. Fundamental conflict: one carve gain can't be both "strong for mapping" and "weak for transient safety."

**Round 1c** (paint only; carve frozen at engineered −0.73):
- Learned: hit-paint +0.28/+0.00; c0=0.26.
- Bars 1/4 pass (held-out 4.95 vs 5.41). Bar 2: clear **8/8** ✓, whiteout 0/8 (**6 collisions**). FAIL.
- Diagnosis: the weaker whiteout paint (0.00 vs engineered +0.18 provisional) starves the map during the cold-start transient before FIX-1 (visibility-gated D5) engages at ~scan 3. The engineered provisional paint was load-bearing for startup.

## The real lesson

**Evidence gains cannot be learned in isolation from the system's temporal dynamics.** The engineered numbers (+0.60/−0.73/0.3) were co-tuned with: `vis_smooth` init at 1.0, the 0.70 carve-gate threshold, confirmation timing, and FIX-1/FIX-3 engagement. Our training evaluated single scans with a settled tracker — a different dynamical regime than flight startup. Every round failed on a transient the training never saw.

## Why not the MLP

The task allowed escalating to a small MLP if the tiny version fell short. We are deliberately NOT building it: the failure is data/structure (missing cold-start transient), not capacity. A bigger model trained on the same settled-tracker data would learn the same wrong lesson, with more parameters. Building it would be training for its own sake.

## If Evidence-θ is revisited

1. Training data must include the **cold-start transient** (supervise scan 1 with a fresh tracker, not just settled state).
2. The mapper needs a **tracker-age** feature (or equivalent): `vis=1.0` on scan 1 means "untrustworthy," `vis=1.0` on scan 50 means "clear." Currently indistinguishable.
3. Or: learn the full temporal evidence policy (bigger project — the mapper as a sequential decision-maker, not a per-scan gain).

## Provenance

- Rooms: HM3D `real_lidar/*_occ.npy` [measured]. Fog: `obscurant.py` Beer-Lambert + backscatter [assumed coefficients].
- Room split: sorted names, first 20 train / last 10 held-out [design].
- Seeds: fixed per (room, density, pose) [design]. No GT at inference (mapper sees vis/hit/range only).
- All flight outcomes [measured] via `mission_dynamics_v5`, numpy backend, paired missions (same room/seed/noise).
- Baseline control re-run 2026-10-08: clear 8/8, whiteout 0/8 success 0 collisions — harness unchanged.
