# Specular sensing — the scoping record (imported)

**This file is why every mission-success and map-fidelity claim in this
repository is DIFFUSE-SENSING ONLY.**

Imported 2026-10-26 from the lab's research record — `SPECULAR.md`
(Phase 3(b) builder report, branch `muse/roadmap-specular-2026-10-11`)
and `SPECULAR_RT.md` (its independent red team, branch
`muse/roadmap-specularrt-2026-10-11`), both originally at
`slow-drone/dynamics_sim/` in the research repo — so that this
repository's public record stands alone. The scoping substance below
is ported faithfully from those two documents, including the red
team's corrections to the builder's first reading. The underlying
data files (`specular_static.csv`, `specular_missions.csv`,
`rt_static.csv`, `rt_missions.csv`) and scripts remain in the research
record; the numbers quoted here are as filed there, each row stamped
with its executed commit in those CSVs.

**Claim ceiling: simulation only — this is a model of specular
effects, not a measurement of real surfaces.**

## The caveat, stated where it is cited

Any "missions succeed in simulation" or "the occupancy map reflects
the room" claim in this repository is **diffuse-sensing only** until
re-measured under a specular arm. Under a specular-reflection sensor
model, long-route missions fail by map erasure at specular fractions
≥ ~0.1 of wall voxels with 50% engagement, via lost returns carving
real structure free — an interaction with the stack's
no-return-carves-free inverse sensor model (below). No claim in this
repository was re-measured under a specular arm; the annotation bounds
the headline numbers, it does not change them.

## What was measured (builder)

The baseline sensor model is diffuse-only by construction: every ray
returns the first surface along the ray, misses are perfectly
reliable, and the log-odds inverse model (l_occ = +0.60 /
l_free = −0.73) was calibrated from that reliability — which is
exactly what specular effects destroy. The smallest specular model
adds four parameters (fraction of occupied voxels designated
specular; engagement strength on oblique hits; a 25° near-normal
return cone; a scene salt). Per ray, an oblique hit on a specular
voxel either reflects and reports a GHOST (an obstacle behind the
mirror surface, at the total path length) or is LOST (no return —
and the free-space carving then runs the full ray, erasing the real
surface from the map at l_free = −0.73 per scan).

- **Diffuse-only baseline: 6/6 missions succeed**; final-map
  false-negative (believed-free) share of real structure = 0.00 in
  every room.
- **Low specular** (20% of wall voxels mirror-like, 50% engagement):
  3/4 missions succeed; the long mission (indoorcd_128) fails to
  finish in 200 steps. Final-map FN share 0.44–0.88.
- **High specular** (50% mirror-like, full engagement): 3/6 succeed.
  All three long missions (071, 117, 128; baseline lengths 50–69
  steps) run to the 200-step cap. FN share up to 0.97.
- Short missions (~20 steps) still complete at all levels — **the
  break regime is route length / room scale, not specular fraction
  alone.**
- **No collisions and zero D5 move refusals in every arm**: the
  failures are map-quality failures (wandering / livelock to the step
  cap), not near-miss safety events. The map underneath the safety
  layer is what breaks.
- Static sweep (all 30 rooms, 9 configs): ghost obstacles are a
  minor channel (false-positive cells never exceed ~9); **erasure is
  the damage** — even the mildest setting multiplies believed-free
  real structure ~50× over baseline, monotone in fraction, strength,
  and engagement cone.

## The red team's corrections (they bind this record)

Red-team verdict on the builder's report: PASS WITH CAVEATS — the
effect is real, exactly reproducible, specular-specific, and
map-caused. Two caveats move the claim materially:

1. **The plausible-regime floor is ≈ (frac 0.1, strength 0.5), on
   long routes, room-dependent** — not the builder's "low" (0.2, 0.5)
   as the lowest demonstrated point, and not "every specular
   setting". Mission sign-flips are demonstrated at (0.1, 0.5) in 1
   of 2 long rooms tested (117 fails; 128 degrades but completes).
   At the mildest plausible corner (0.05, 0.25) no mission effect was
   observed in the tested long room, and static FN share is 0.033.
   The builder's low/high settings are best read as stress settings:
   spec_frac is a fraction of *all* occupied voxels designated as
   perfect mirrors with zero partial/diffuse return, which real
   smooth surfaces are not.
2. **The headline is a property of specular × this stack's inverse
   sensor model, not of specular sensing alone.** Roughly 80% (low) /
   62% (high) of the FN damage is the no-return carving policy, not
   the lost photons: with identical specular scans but a
   no-return-carves-nothing integration variant, the high-setting
   long mission that fails at the 200-step cap succeeds in 87 steps.
   The claim, precisely: **specular loss breaks this stack's missions
   because its inverse sensor model carves full free rays on
   no-returns with an l_free calibrated for reliable misses.** A
   specular-robust inverse model is the experiment that decides how
   much of the verdict is physics and how much is policy; it is
   parked, not built.

Two further red-team findings travel with the record: what specular
adds over random dropout at the same loss rate is *persistence* (the
scene-fixed mask makes the same surfaces lose returns from every
pose, so their log-odds ratchet down and are never repaired — final-map
FN share alone is a poor failure predictor), and "ghosts are minor"
is room-dependent (indoorcd_071 at high is ghost-dominated).

Attack that failed: replacing the salt-and-pepper specular mask with
contiguous patches of exactly the same voxel count does not inflate
the effect away — contiguous surfaces carry ~90% of the damage.
