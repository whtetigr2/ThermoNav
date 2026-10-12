# sim/ — dynamics validation suite

Point-mass + approximate-attitude flight simulation (software-in-the-loop only —
not a validated aircraft). Phases 1–4: inertia fixes, obscurants, gravity/wind/
thrust-lag, and the safety-fix stack.

## Entry points (current — use these)

- `mission_dynamics_v5.py` — the mission runner (imports the modules below via
  repo-relative paths; run from this directory)
- `curated_fixes.py` — the validated fix stack (visibility-gated D5,
  confirmation-gated carving, low-vis executive, D5 freshness gate)
- `fly.py`, `head_to_head.py`, `head_to_head_v2.py`, `head_to_head_thrml.py` —
  flight/benchmark harnesses (THRML and NumPy backends)
- `env.py`, `sensor.py`, `planner.py`, `controller.py`, `classical.py`,
  `s16.py`, `gibbs.py`, `pid.py`, `traj.py`, `vehicle_v3.py`, `obscurant.py`,
  `obscurant_fixes.py` — supporting modules

## Version cruft (superseded — kept for provenance, not for use)

`mission_dynamics.py` → `v2` → `v2b` → `v3` → `v4` → **`v5` (current)**.
`curated.py` → `v2` → `v2b` → `v3` → `v4` → **`curated_fixes.py` (current)**.
`screen12.py` → `v2` → `v2b`. `vehicle.py` → **`vehicle_v3.py` (current)**.

## Research scripts (not entry points)

`link_health.py` + `test_link_health.py` + `LINK_MODEL.md` — the
link-health layer (graduated 2026-10-11): receiver-side link tracker and
the escalation policy's link rows. Run `python3 test_link_health.py`
(25 tests). The mission-loop link wiring from the research repo was not
ported; the layer is exercised at tracker + policy level here.

`battery.py` + `battery_safety.py` + `test_battery.py` (14 tests) +
`test_battery_safety.py` (18 tests) + `BATTERY_MODEL.md` — the battery
layer (graduated 2026-10-11): pack gauge, the escalation policy's battery
rows, and Paul's safety cascade (measured-draw countdown, RTB/RTC/land),
wired into `mission_dynamics_v5.py` (`cfg["battery"]` /
`cfg["battery_safety"]`, off by default; the checkpoint's nav-trust
input is the real NavHealthMonitor by default since the nav-health
graduation — see below — with the explicit `nav_trustworthy_stub`
kept only as a scripted scenario override). Whiteout update
(2026-10-12): the countdown now prices a descent reserve, the
terminal descent flies flat 0.5 m/s (emergency 1.0 m/s below the
reserve-required start), and the guarantee is stated as a two-tier
envelope (Tier A ~CRITICAL touchdown / Tier B soc > 0, terminal-start
form) — see `BATTERY_MODEL.md`. Run
`python3 run_battery_safety_validation.py` for the 9-mission safety
validation (6/6 gates at the whiteout graduation, executed
`018c841`; writes `battery_safety_validation.csv` +
`battery_safety_calibration.csv`), and
`python3 run_whiteout_tierb_spotcheck.py` for the TN-side Tier B
sub-CRITICAL spot check (writes `whiteout_tierb_spotcheck.csv`).

`nav_health.py` + `exec_provenance.py` + `test_nav_health.py`
(24 tests) + `test_exec_provenance.py` (7 tests) +
`test_nav_checkpoint.py` (checkpoint wiring) + `NAV_HEALTH.md` — the
nav-health layer (graduated 2026-10-11, after red-team round 5 PASS
WITH CAVEATS, documentation-only): categorical estimator-health
monitor (HEALTHY/DEGRADED/UNTRUSTED from estimator self-reports
only), the escalation policy's nav rows (RTH gate =
pose_trustworthy AND HEALTHY; UNTRUSTED triggers HOVER, RTH-desired
after 30 s sustained), and the battery checkpoint wiring
(`resolve_nav_trustworthy` in `mission_dynamics_v5.py`: the monitor
is active when `cfg["nav_health"]` is set, and by default whenever
the battery safety layer is on without a scripted override;
`cfg["nav_health_freeze_from_step"]` scripts the measured whiteout
freeze for validation). Run `python3 run_nav_health_calibration.py`
and `python3 run_nav_health_mission_validation.py` (both
provenance-guarded: they abort on a dirty tree and stamp the
executed commit into `nav_health_calibration.csv` /
`nav_health_mission_validation.csv`).

`dynlayer.py` + `test_dynlayer.py` (11 tests) + `motion.py` +
`movers.py` + `dodge.py` + `DYNLayer_RESULTS.md` — the dynamic-object
layer (graduated 2026-10-12, after red-team round 2 PASS WITH
CAVEATS): previous-scan differencing + accumulated-map appearances,
CV tracks with tiered authority (2 hits veto, 3rd full/dodge), F1
wall-candidate chains, projected-tube `blocks()`. Wired into
`mission_dynamics_v5.py`'s `_MOTION` slot as the MotionBuffer
successor (`cfg["dyn_layer"]`, DodgeController on top via
`cfg["dodge_layer"]`; movers staged via `cfg["movers"]`). Run
`python3 run_dynlayer_sweep.py off|on` then `merge` for the paired
48-cell validation (104 rows in `dynlayer_sweep.csv`, executed at
the graduation code commit: OFF 43/48 → ON 7/48, 36 fixed /
0 broken, 0 mismatched rows vs the research sweep). Scope
boundaries (open-room wall-lane blindness, 3-paint phantom bound,
131 dodge-geometry) are recorded in `DYNLayer_RESULTS.md` and
travel with the layer.

`slam_test.py`, `slam_obscurant.py`, `sampler_tracking_test.py`,
`tune_pid.py`, `tune_turn.py`, `analyze_*.py`, `turn_metrics.py` — analysis and
tuning experiments. They run, but they are not the validated path.
