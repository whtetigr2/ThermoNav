# Capabilities

What the drone can do, how it is told where to go, and what is not
established. Written 2026-11-11. Every figure below is from the flight
record in this repository's research digest (`docs/RESULTS.md`) and is
**simulation-measured** unless labeled otherwise. Scope, stated up
front: all of this is Gazebo physics simulation. Nothing has flown on
a physical vehicle.

## How it is told where to go

Nobody joysticks it. A mission is a list of destination points (hops)
on its map, plus a start. The drone builds a belief map from its own
camera as it flies — free, occupied, and unknown space kept separate —
lays an energy field (the wavefront planner) over that map, and chases
a "firefly" subgoal placed a short step ahead, re-placed at every
decision. The mission anchor never moves mid-flight.

Arrival counts at the **centre** of the goal zone, not the edge: on
the graded flights the drone held about 0.40 m of the target point
(hold RMS) inside a 1.5 m arrival radius.

Control underneath: an outer loop (P-only, kp 0.8/s) turns position
error into velocity commands at 50 Hz, and a vendor Lee controller
with untouched gains flies the motors.

## Speed

- **Flown cruise cap: 1.0 m/s.** That is a hand-set cap, not a found
  limit — it has never been searched. (A pre-registered computer
  search over this and the other hand-set control settings is in
  preparation.)
- The controller configuration allows up to 3.0 m/s and 2.0 m/s².
  That ceiling has never been flown or graded.
- Airframe: 2.40 kg; thrust-to-weight ≈ 3.1 (calculated); hover at
  about 57% throttle.

## Obstacle avoidance — and what has not been tested

Avoidance is map-based: obstacles enter the belief map from the
camera (848×480 at 10 Hz, 87° field of view, depth used over
0.3–10 m; mapper grid 88×48×12 at 0.5 m cells) and the field re-plans
around them.

- Measured against **fixed obstacles only**: zero contacts across all
  graded flights; minimum clearance 0.968 m on the 2026-11-11 graded
  set.
- **Moving obstacles and incoming objects: not tested.** No dodge
  behaviour and no reflex time has been measured, and none is claimed.
  The measured pieces of the reaction chain are listed so the gap is
  visible rather than papered over: camera at 10 Hz; sensor claims
  reaching the filter at p95 ≈ 0.18 s of age in course (after the
  delivery fix described below); commands at 50 Hz. Those pieces do
  not add up to an end-to-end reaction time, and this document does
  not state one.

## Does it know where it is

In simulation, now, yes — this was the drift problem, and it was the
subject of a dedicated unit. After the two fixes below, the
2026-11-11 graded set (four fresh sessions) held course position
error at p95 0.026–0.033 m against ground truth.

Two qualifications, both part of the record:

- The position estimate comes from a separate vision/LIO estimator —
  borrowed eyes. Its accuracy is evidence about the estimator, not
  about the thermodynamic decision layer, and the flight claims in
  this repository rest on the decision layer.
- Simulation sensors are kind (the flight IMU in the simulator is
  ideal). One failure mode is known with its cause found: an
  estimator death and restart once produced a 1.70 m re-anchoring
  offset. A fix (LIO-bridged re-anchoring) is suggested, not built.

## Flight record, in brief

- **School 3 — closed 2026-11-10.** The graded set landed 4 of 5 hops
  at 0.43–0.46 m arrival error, zero contacts, all five hops safe.
  Hop 1 was a false arrival at 1.0765 m.
- The drift behind that miss was run down to two measured causes:
  (1) sensor claims reaching the filter already stale while stamped
  as current — a delivery/timing defect; and (2) an acceleration bias
  injected in the body frame but tracked as a world-frame state.
  Both fixes were built and flown.
- **Drift unit — closed 2026-11-11** on the graded re-test: 4/4
  centre arrivals, 0 false arrivals, 0 contacts. One pre-registered
  bar did **not** pass as written — the claim-age tail; the miss was
  traced to the measuring instrument counting pre-flight pad time,
  and under a corrected course-window measure the set passes. Both
  statements stay on the record, the NOT PASSED included.

## The S2 kernel's colour problem, stated plainly

Not fixed. The shipped S2 kernel's coupling graph is 6-chromatic
(measured), so as shipped it cannot use the Z1 chip's two-colour
parallel update, and chip legality is not established. What changed
on 2026-10-29 (PR #6) is the screen: `z1_check` was fixed to test
colourability honestly, and the kernel now **fails** that screen on
all 12 shipped parameter files — visibly, instead of passing
unearned. Fixing the kernel itself would mean rebuilding it with a
two-colourable graph; that work has not been done.

This does not bear on the flight results above: the flying stack does
not use the S2 kernel (retired). It flies the constructed wavefront
planner.
