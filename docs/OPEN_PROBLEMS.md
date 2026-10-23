# Open problems and known pitfalls

First-class section, by design. This is what does **not** work yet,
what is measured-but-broken, and what is simply unmeasured — stated at
the same prominence as the results, because a proof of concept you
cannot see the edges of is not a proof of anything.

## 1. Self-location — the open problem

The stack consumes pose; it does not yet produce it well enough.

- In the current flight program, ground truth is **scoring-only**:
  never a flight input. Earlier integration flights that quietly used
  ground-truth pose as an input had their success claims **revoked**
  and are labeled revoked in the record. Nothing GT-fed counts as
  evidence here.
- The standard front-end (ORB-SLAM3 mono-inertial) performs at
  published level on real EuRoC footage, but in our simulated worlds
  it **fragments**: tracking is lost outright (3 losses in the current
  sim run), so the sim gate — pooled per-segment ATE ≤ 0.20 m *and
  zero tracking losses* — is **not met**, even though the pooled ATE
  number alone (0.141 m) passes.
- Directions measured dead so far: depth-ICP self-location; pure-RGB
  visual odometry (drifts ~0.5 m per meter flown); mapless dead
  reckoning on the quantized flight grid (closed by an oracle bound —
  even a perfect version of the approach cannot meet the bar).
- The live approach — a learned scorer inside the classical pipeline,
  arbitrating keyframe and pose commitment — is at the bench stage,
  pre-registered, not yet flown as evidence.

## 2. Collision counting is flight-phase only

Every collision count in this record is counted over the flight phase.
A ground-phase counting gap in the harness means collisions during
ground handling are not yet in any denominator. Treat every "0
collisions" as **0 flight-phase collisions** until the fix lands.

## 3. Vertical shafts

1×1-cell vertical shafts are **unclimbable** by the current kernel —
a hard geometric limit of the move set, measured, not a tuning item.

## 4. Unmeasured models: battery draw and wind

- The battery *safety cascade* is sized from measured drain *inside
  the simulator*, but the underlying power model is **calculated**
  (momentum theory), not measured against a real vehicle. The cascade
  logic is validated; its numbers are only as real as the model.
- **Wind is unmeasured.** A wind estimator was prototyped in a retired
  harness and its one PASS was corrected, on re-test, to a partial —
  the vertical-gust axis failed. In the current stack the estimator is
  **unbuilt** and wind response is **unmeasured**. A measurement-first
  charter exists in the research record (seeded wind, scorer-only
  ground truth, leakage tests); no results yet.

## 5. Field healing: first measurement landed, review pending

The wavefront planner's maps are static within a mission. When the map
changes mid-flight — a door closes, an obstacle appears — the field
must re-relax. The lab's first healing measurement landed on
2026-10-23 and is **UNREVIEWED** as of this writing; treat the numbers
below as preliminary:

- A stale field misbehaves exactly as feared while it is wrong: it
  routes through newly added walls and strands at a moved goal.
- Continued sampling heals a small added wall in the 2D world in
  ~1 sweep — but in the large 3D maze, a single-cell wall **never**
  heals by continuation (0 un-flip events in 8,320). Small worlds forgive;
  big ones don't.
- Discarding the stale field and re-forming it returns to the honest
  band at **~1× the cold-formation price** (ratios 0.87–1.09 across the
  four probe legs). Healing costs what forming costs: no shortcut, no
  penalty.

Until the red-team review lands, dynamic-world claims remain limited
to the dynamic-object layer's measured envelope, which handles movers
at the controller/safety level, not by re-planning.

## 6. Sensing-model boundary: diffuse only

Mission-success and map-fidelity claims are **diffuse-sensing only**.
Under a specular-reflection sensor model, long routes fail by map
erasure at specular fractions ≥ ~0.1 (lost returns carve real
structure free through the no-return-carves-free inverse model) —
red-team verified, full annotation in `README.md` and `REDTEAM.md`.

## 7. Named residuals in shipped lines

- **Lost-link landing is untested end-to-end.** The ladder reaches the
  gated LAND action and battery-driven landings actuate, but no
  lost-link mission has been run start to finish; "a lost-link vehicle
  ends up on the ground" is not yet demonstrated.
- **Movers:** an open-room wall-lane hugger is still invisible to the
  dynamic-object layer (its paints merge into wall flicker); 3
  motion-consistent noise paints can still manufacture a full-authority
  phantom (false dodges 2/24 clean missions, none mission-failing); one
  head-on cell fails on dodge geometry — there is no retreat behavior.
- **Whiteout** is safe but not operational: 0/8 flights complete, 8/8
  safely blocked. The honest fix is another modality (radar), not
  more training.
- **Rails:** the wavefront's `max|h| = 6.0` sits at the assumed Z1
  rail with zero headroom; S2-θ's 5.35 is close under it. Flagged for
  hardware watch since the original runs.
- **Pose assumption:** the shipped simulation stack assumes pose is
  known (item 1 is why that matters); drift testing shows first
  collisions at ~1 m accumulated error.

## 8. The ceiling of the whole artifact

Everything here is simulation. There is no silicon result, no hardware
flight, no measured energy figure — the substrate's energy advantage,
where the research record discusses it, is a **calculated projection**
from published device figures, not a measurement. ThermoNav's claim is
narrower and, we think, more useful: a complete navigation brain,
formulated for this substrate class, measured end to end in the
substrate's own simulator, with its failures documented at the same
standard as its successes.
