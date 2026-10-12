> **ThermoNav graduation copy (2026-10-12).** Ported verbatim from
> the research repo (thermo-work branch muse/belowclear-2026-10-12,
> commits 5cae995 + errata c8e406d + red-team fix round 5ef3763),
> where the full probe artifacts live
> (belowclear_phase0_probe.csv, probe_belowclear_phase0.py). The
> findings apply to ThermoNav unchanged: the record rooms' grids
> in ../rooms are byte-identical to the research real_lidar grids
> for the measured rooms (indoorcd_071 / indoorcd_131 md5-verified
> at graduation), and ThermoNav's loop uses the same sensor/mapper
> construction (obscurant.py scan_pose takes no attitude input;
> the boundary floor prior is written by the loop at mission
> start). Every ThermoNav price that touches a descent column —
> battery, whiteout, and the below-clear hold graduated alongside
> this document — is read under this caveat.

# BELOW-CLEAR PHASE 0 — Paul's ground-representation challenge: VERDICT

**VERDICT: MIXED — Paul is right on the load-bearing part.** In the
rooms that produced the validation record, the map model contains no
ground the sensor can hit: the ground-truth grids carry no floor at
all, and the only "ground" in the loop is a prior the mapper writes
onto its own grid at mission start. The sensor model also never
applies vehicle attitude to the fan — Paul's tilt mechanism is absent
from the sim. The part Paul is wrong about is narrow: it is not an
either/or, and a fan that paints the ground still would not satisfy
this guard, because the guard demands a filled cylinder of observed
cells, not a painted cone surface. Three stacked causes, floor
absence first. Detail and evidence below.

**Consequence for Phase 1 (stated up front, per the build brief):**
the "descent column UNKNOWN" finding on record is substantially a
modeling artifact in the record rooms. Option A (Paul, decided
2026-10-12) still stands as the fix for the actuation trap — a
landing that can be vetoed forever is wrong no matter why the column
is unknown, and in whiteout the same trap occurs under real
obscuration — but the flag-on collision price measured in Phase 1
must be read as "descending into columns that were unobservable
by construction," not "descending into columns the drone failed to
look at." A modeling-debt follow-up (floor geometry in the real-room
grids; tilt-aware fan) is parked, NOT folded into this build.

Claim label for everything below: measured in simulation / read from
committed code, at research commit 5827ab7. No silicon, no aircraft.

---

## Q1. The lidar model — fan shape, and is tilt applied?

- Three fixed rings, elevations **(-0.26, 0, +0.26) rad**, 120
  azimuths per ring, max range 40 m. `sensor.py:54` (scan_pose) and
  the drop-in obscurant replacement `obscurant.py:109`; the mission
  loop's ELEV constant is `fly.py:39`.
- Ray directions are built **only** from azimuth and elevation
  (`sensor.py:60-64`, `obscurant.py:124-127`). The pose contributes
  the ray ORIGIN and nothing else. There is no orientation parameter
  in either scan function.
- The mission loop passes `pose = vehicle.pos` (position tuple) to
  the scan (`mission_dynamics_v5.py`, per-step scan call ~line 673).
- The vehicle model DOES carry attitude — `vehicle_v3.py` keeps a
  thrust-direction state with `TILT_MAX_DEG = 35.0` (line 71) and an
  attitude-lag option — but that state is used by the dynamics only.
  **No pitch or roll is ever applied to a ray, in translation or in
  hover.** Paul's tilt mechanism is physically present in the vehicle
  model and absent from the sensor model.

Geometry of the fan as modeled: the -0.26 ring descends at
tan(0.26) ≈ 0.266 m per horizontal metre. From a hover at height h
it meets the floor plane on a RING at horizontal radius ≈ 3.76·h —
it paints a cone surface, never the nadir cylinder. A column cell at
depth d within the 3×3 footprint (horizontal offset ≤ ~1.4 m) lies
on a ray from the current pose only for d ≲ 0.4 m. From approach
positions the same cell lies on the cone only when the drone is
≈ 3.76·d away in a matching azimuth — painting is possible in
principle, one cell-ring at a time.

## Q2. The mapper — is there a ground?

- **In the real-room grids that produced the record: NO.** Measured
  directly from the committed grids (this probe, 2026-10-12):
  indoorcd_071 occupancy fraction by layer — zz=0: 0.0000, zz=1:
  0.0000 (first occupied layer zz=2); indoorcd_004 — zz=0..2 all
  0.0000; indoorcd_131 — zz=0 and zz=1 both 0.0000. The raycaster
  (`sensor.py:37` cast_ray) marches these grids: with no floor
  voxels, a down-ray never returns from the ground; it exits the
  grid and reports max range.
- Synthetic rooms differ: `env.py:29` and `env.py:146` DO set the
  floor layer occupied (`occ[:, :, 0] = 1`). Paul's branch (a) is
  room-dependent — true for the validation corpus, false for the
  synthetic builder.
- What the perceived map has instead is a **boundary prior**: at
  mission start the loop writes logodds +2.0 onto every boundary
  layer including the whole zz=0 floor
  (`mission_dynamics_v5.py`, run_mission_dynamics init;
  FLOOR_PRIOR_LOGODDS = 2.0, line 330). The floor in the perceived
  map is bookkeeping the mapper wrote itself, not sensed geometry.
- Floor RETURNS, where a floor exists, would be inserted as ordinary
  hit updates (+0.60) by the inverse sensor model; free-space
  traversals pull cells down (-0.73). In the record rooms there is
  nothing to return.

## Q3. The guard — what exactly is "the column"?

`_below_state` (`mission_dynamics_v5.py:333`): the 3×3 footprint
(radius 1) around the drone's voxel, every layer from zz = z-1 down
to zz = 0.

- zz ≥ 1: a cell is "observed" only under the TSU observed mask
  (`head_to_head.py:67` effective_observed_mask): base |logodds| >
  0.05 (`classical.py:19`) AND (obs_count > 2 OR |logodds| > 1.0)
  (WEAK_OBS_MAX = 2, STRONG_L_THRESH = 1.0, `head_to_head.py:48-51`).
  State 0 (clear) requires EVERY such cell observed AND its Gibbs
  marginal below P_OBSTACLE. One unobserved cell anywhere in the
  column → state 1 (unknown). One believed-occupied cell → state 2.
- zz = 0: evidence-keyed against the prior (the RT-1 fix): at or
  below +2.0 it is treated as the landing surface and ignored; only
  hit support ABOVE the prior makes it an obstacle. So the guard
  does not require the floor to be "clear" — it requires the air
  column above it to be fully observed, plus (for the full-rate
  descent profile) `_floor_evidence`: every footprint floor cell's
  accumulator moved off the prior by hits or traversals.

## Q4. Were the record columns ever painted during approach?

Probe: `probe_belowclear_phase0.py`, data
`belowclear_phase0_probe.csv`, run at commit 5827ab7 (clear air,
BASE_CFG = battery-safety validation configuration):

- rtb_arrival (indoorcd_071 seed 1, start_soc 0.26 — the RTB
  landing that the RT-3 scope correction describes): **46 guard
  consultations recorded in the probe CSV: state = 1 (UNKNOWN) on
  41 — every consultation at decision altitude (z ≥ 2.57 m) —
  and state = 0 on 5, all at z ≤ 2.07 m during the final descent,
  after the battery path had committed the landing.** The probe
  process's own mission row printed 13 vetoes / outcome
  landed_disarmed; the uninstrumented run of the same cell and
  cfg gives 28 vetoes / landed (established in
  BELOWCLEAR_VALIDATION.md, which supersedes the probe as the
  identity comparator). [Counts corrected 2026-10-12 in the
  red-team fix round: an earlier draft of this section said 64
  consultations, UNKNOWN on all 64 — the committed CSV says 46
  and 41; the verdict below rests on the decision-altitude
  subset and is unchanged.]
- At every consultation, the set of column cells observed EVER
  (union of the mission's own observed mask over all steps up to
  that point) equals the set observed NOW — **approach flight added
  zero observed cells to the descent column.** At the first
  consultation (z = 8.51 m) only 31 of 63 column cells were observed
  and only 3 of 7 layers fully observed; the unobserved cells were
  never observed at any earlier step either.
- Floor evidence under the footprint: **0 of 9 cells at 43 of 46
  consultations (and 0 of 9 at all 41 decision-altitude calls)** —
  consistent with Q2: there is no floor to sense and no ray
  reaches zz = 0 under the drone inside the room. The exceptions
  are instructive: at one final-descent call (z = 1.07 m, after
  the battery path had already committed the landing) the
  footprint read 9/9 — from ~1 m even this fan does register the
  floor layer — and at the last two calls (z ≤ 0.56 m) the column
  no longer includes a zz = 0 level (recorded −1). The guard only
  ever reports state 0 in this mission at z ≤ ~2 m, when the
  column has shrunk to its top level: at decision altitude the
  evidence is absent, full stop.
- The plain clear-air arrival mission (indoorcd_004 seed 0)
  consulted the guard **0 times** — in this configuration a normal
  goal arrival does not demand a guarded LAND; the guard bites on
  escalation/safety LAND demands (RTB arrival, land-in-place,
  whiteout arrival), which is where the record comes from.

Why approach painting fails even in principle-rich geometry: a cell
must be threaded by an actual ray sample point (the integrator
marches rays at 0.5 m steps, `sensor.py` integrate_scan /
`obscurant.py` integrate_scan) often enough to pass obs_count > 2
or |logodds| > 1.0. At the ranges involved (the cone reaches a
deep column cell only from ~3.76·d away, slant range up to ~28 m),
the 3° azimuth spacing spreads adjacent rays ~1.5 m apart — wider
than a voxel. The column is not merely unpainted from the hover;
it is statistically almost unpaintable by this fan under this mask,
floor or no floor.

## The verdict, claim by claim

| Paul's claim | Verdict | Basis |
|---|---|---|
| "No ground existed at all" (branch a) | **RIGHT for the record rooms**; wrong for synthetic rooms | GT grids: zz=0/1 occupancy 0.0000 in 071/004/131 (measured); env.py synthetic rooms do carry a floor |
| "We are modeling mechanics incorrectly" (branch b) — tilt | **RIGHT**: attitude is never applied to the fan | scan_pose signature and ray construction, sensor.py:54-64; vehicle attitude exists (vehicle_v3.py) but is dynamics-only |
| Fan-out should paint the ground | **PARTLY**: it paints a cone surface/ring, and returns would register — but the guard needs a filled cylinder under the mask thresholds, which the probe shows never accumulates | Q1 geometry; Q4 probe (ever == now ×46, floor evidence 0/9 at all 41 decision-altitude calls, the 9/9 exception at z = 1.07 m during final descent) |
| Implied: the UNKNOWN record is a sim artifact, not a sensor truth | **SUBSTANTIALLY RIGHT** in the record rooms | Q2 + Q4: the sensor is never given a ground to see in those rooms |

**Bottom line for Paul:** you were right to call it. The "drone
can't verify the ground" story the lab has been telling is, in the
rooms on record, mostly "the sim never gave the drone a ground to
verify, and never tilted its fan." The guard itself is honestly
built (it correctly reports that its evidence is absent) — the
absence is manufactured upstream of it. Option A proceeds, with
the Phase 1 numbers read under this caveat, and the modeling debt
(floor geometry in real-room grids; tilt-aware fan; mask-threshold
sensitivity) parked as its own follow-up card.
