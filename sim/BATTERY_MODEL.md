# Battery Model — Roadmap Phase 1, Item 1

> **ThermoNav port note (2026-10-11, graduation):** this report is the
> research-repo build record, ported with the RT2-1..RT2-4 corrections
> from `slow-drone/dynamics_sim/` in thermo-work (branch
> `muse/roadmap-battery-2026-10-11` @ `6272fc1`). In ThermoNav the layer
> lives in `sim/`: `sim/battery.py`, `sim/battery_safety.py`, the battery
> rows merged into `sim/escalation.py` (preserving its DISARM + link
> wiring — the decider kernel is byte-identical), the mission-loop
> integration in `sim/mission_dynamics_v5.py`, `sim/test_battery.py`,
> `sim/test_battery_safety.py`, and `sim/run_battery_safety_validation.py`.
> Layout differences from the research repo: ThermoNav's mission loop
> has no jitter/movers/dodge layers, so the whiteout validation cell
> runs as whiteout fog without the jitter component; the escalation
> policy keeps its ThermoNav MISSING-list convention (a slot is listed
> until its inputs are supplied, as with link). One deliberate
> consequence of the port: the evidence-keyed landing guard
> (`_below_state` / `_floor_evidence` / creep) replaces ThermoNav's
> old face-value `_below_clear`, whose floor-prior unsatisfiability is
> LINK_MODEL.md known limit 6 — that limit's premise no longer holds
> on this code (see README/REDTEAM updates). The checkpoint's
> nav-trust input remains the EXPLICIT stub (`nav_trustworthy_stub`):
> the real NavHealthMonitor graduates separately, and wiring it in
> place of the stub is a recorded follow-up. **Superseded
> 2026-10-11:** that follow-up is done — the nav-health graduation
> made the real NavHealthMonitor the checkpoint's default trust
> path (`resolve_nav_trustworthy` in `mission_dynamics_v5.py`);
> the stub survives only as a scripted scenario override. See
> `NAV_HEALTH.md`. ThermoNav-side validation
> results are recorded at the end of the safety-layer validation
> section below.

**Branch:** `muse/roadmap-battery-2026-10-11` (research repo) · **Date:** 2026-10-11
**Status:** BUILT + VALIDATED; **red-team round 1: FAIL** (two load-bearing findings, RT-1/RT-2) → fixes built and re-validated in the round-2 response below; **red-team round 2: PASS WITH CAVEATS** (verdict: closable; RTC gap judged CLOSED) — report `REDTEAM_BATTERY_R2.md` on branch `muse/roadmap-battrt2-2026-10-11`. Its four follow-ups (RT2-1..RT2-4) are implemented in the "Red-team round 2 — follow-ups" section below. The round-1 red team's report is `REDTEAM_BATTERY.md` on branch `muse/roadmap-battredteam-2026-10-11`. What the red team verified TRUE stands: physics, bookkeeping, CSV provenance, drain-arm reproduction (4/4 identical), legacy identity.
**Addendum (2026-10-11, later): Paul's battery safety design is BUILT on top of this model** — `battery_safety.py`: one strict flag, a countdown sized from *measured* draw, and the RTB → RTC → land-in-place reachability cascade. See "Paul's safety layer" below; it also carries the PI dispositions for Q1/Q2 (both questions were decided by Paul, not by this build). Safety-layer validation was first executed at commit `f5f0ddf752f5e5994d7ad14e5c32996ba84d1e6d` and re-executed in full at `7cb44b4` with the red team's rtc_v3/rtc_v4 scenarios ported in (every current CSV row stamps `7cb44b4`; all prior rows reproduced identically).
**Claim label:** classical host model, **simulation only**. Parameters are calculated from the assumed 2.4 kg mass, sourced from the repo variable registry, or labeled ASSUMED. **Nothing in this model is measured** — there is no hardware and no bench data. (The "Z1-legal, validated in simulation" ceiling applies to substrate pieces like the escalation decider, which this work does not touch.)

## Why this exists

The escalation ladder (Option A, merged 2026-10-10) had three MISSING trigger slots: battery, link, nav-health. Battery was a stub that could never fire — the ladder literally could not run out of energy. This build supplies the power model, wires its state into the policy, and removes battery from the MISSING list. (In the research repo at build time, link and nav-health remained MISSING stubs; in ThermoNav, link has since graduated — see `LINK_MODEL.md` — and nav-health remains MISSING.)

## The model (`battery.py`)

Smallest honest version (LEGO principle): momentum-theory propulsion power, a linear open-circuit-voltage curve, one lumped internal resistance. No Peukert effect, no temperature, no ageing.

### Equations

- Thrust in hover: `T = m g` = 23.54 N at m = 2.4 kg.
- Hover induced velocity (momentum theory): `v_h = sqrt(T / (2 rho A))`, with total disk area `A = 4 * pi * (0.254/2)^2 = 0.2027 m²` → `v_h = 6.89 m/s`, ideal induced power `P_i = T v_h = 162.1 W`.
- Forward flight induced velocity (textbook momentum relation): `v_i² = (-v² + sqrt(v⁴ + 4 v_h⁴)) / 2`; induced power `T * v_i`.
- Electrical load: `P = (P_induced + P_profile + P_parasite + P_climb) / eta_drive + P_avionics`, where `P_profile = 0.35 * P_i,hover` [ASSUMED], `P_parasite = 0.5 rho CdA v³` (CdA from vehicle_v3), `P_climb = m g v_z` (descent credit floored at 40% of hover propulsion [ASSUMED]). Shaft power is floored, never negative.
- Open-circuit voltage: `OCV(soc) = 4 * (3.0 + 1.2 * soc)` → 16.8 V full, 12.0 V empty [ASSUMED linear simplification of the LiPo curve].
- Terminal voltage under load solves `P = V_t * I`, `V_t = OCV - I * R_pack` exactly: `V_t = (OCV + sqrt(OCV² - 4 R P)) / 2`. Sag is therefore exactly `I * R_pack` at the operating point (unit-tested to 1e-9).
- Discharge: `soc -= I * dt / (3600 * 5.0 Ah)`, floored at 0.
- Time-to-reserve: **stepped integration** over soc in 0.002 increments, recomputing current at each step's voltage, so the rise of current as voltage falls is priced in [calculated estimate]. (RT-5 correction: the original one-shot formula `TTR = (soc - RESERVE) * Capacity_Ah / I` was measured by the red team as optimistic by **+16.4% from full at hover draw and +21.2% at 450 W**; its "<5%" claim held only within ~0.1 soc of the floor. The stepped version matches a fine brute-force discharge to <1%: 670 s from full at 282.5 W, 377 s at 450 W — the red team's exact figures are 671/377.)

### State outputs

Charge fraction (`soc`), terminal voltage under load, current, load power, time-to-reserve, time-to-empty, Wh drawn, minimum voltage seen.

### Calculated phase powers (from the equations above, not measured)

| Phase | Power | Note |
|---|---|---|
| Hover | **282.5 W** | propulsion-only 257.5 W |
| Cruise @ 2 m/s (operational cap) | 279.4 W | momentum-theory power bucket; parasite ≈ 0.7 W at CdA 0.15 |
| Cruise @ 5 m/s (capability cap) | 272.7 W | induced drop still dominates parasite at this CdA |
| Climb @ 1 m/s | 310.2 W | adds m g v_z / eta |
| Descent @ 0.5 m/s (LAND rate) | 268.6 W | climb credit, above the 40% floor |

**Cross-check (not calibration):** the variable registry independently records ~280–320 W as the class-typical *propulsion* hover band for this airframe (eCalc-class 2216/1045 4S curve, unmeasured). The model's propulsion-only figure, 257.5 W, sits ~8% below that band's bottom; total electrical including the avionics lump (282.5 W) sits inside the band's range numerically, but the band excluded avionics, so the honest statement is: **two independent estimates agree within ~8–20%**, and the model is the lower of the two (it would predict slightly longer endurance — the non-conservative direction, flagged here).

**Endurance sanity:** full pack at hover draw → time-to-reserve **670 s** (stepped integration), time-to-empty **801 s** (~13 min). Plausible for a 2.4 kg X500-class quad on 4S 5000 mAh; labeled estimate, not measurement. (The 783 s figure in the round-1 report was the one-shot formula RT-5 corrected.)

### Parameter table — provenance per parameter

| Parameter | Value | Provenance |
|---|---|---|
| Vehicle mass | 2.4 kg | ASSUMED — Paul's ruling 2026-10-08 (registry §1); not changed |
| Pack configuration | 4S1P, 5000 mAh, 74.0 Wh, 520 g | Registry bottom-up AUW battery line ("520 g 4S 5000 mAh", assumed-typical within listing range); Wh calculated |
| Nominal voltage | 14.8 V (3.7 V/cell) | Standard LiPo convention |
| OCV endpoints | 3.0 / 4.2 V per cell, linear in soc | Standard endpoints; linearity ASSUMED simplification |
| Cell internal resistance | 20 mΩ (pack 80 mΩ) | ASSUMED (typical LiPo 15–25 mΩ); bench measurement MISSING |
| Prop diameter | 0.254 m (10 in, 1045) | Registry prop class (2216/1045 on 4S) |
| Rotors | 4 | Registry (X500-class quad) |
| Air density | 1.225 kg/m³ | Standard (matches vehicle_v3) |
| Drag area CdA | 0.15 m² | Registry / vehicle_v3 [NAPKIN] |
| Profile power fraction | 0.35 × hover induced | ASSUMED |
| Drive efficiency (motor+ESC) | 0.85 | ASSUMED |
| Avionics lump | 25 W | ASSUMED lump (compute + lidar + FC); itemized draw MISSING |
| Descent power floor | 40% of hover propulsion | ASSUMED |
| Reserve floor | soc 0.20 | **DESIGN CHOICE** (industry low-battery RTH bands are typically 20–30%, reported; we picked the low end — see limits) |
| Warn threshold | soc 0.30, or TTR ≤ 300 s | DESIGN CHOICE |
| Critical threshold | soc 0.10 | DESIGN CHOICE |

## Wiring (`escalation.py`, `mission_dynamics_v5.py`)

- `EscalationPolicy.evaluate()` gains optional `battery_soc` / `battery_ttr_s`. **No battery input → behavior identical to the pre-battery build** (all pre-existing policy tests pass unmodified; the decider grid is still 128/128).
- Trigger rows (most-severe-wins across perception and battery):
  - **Warn** (soc ≤ 0.30 or TTR ≤ 300 s): host notification only — a `battery_warn` log entry, latched once per flight. Warning is not a flight action (per ESCALATION_BUILD's spin-budget scoping).
  - **Reserve** (soc ≤ 0.20): desired **RTH** — intent goes through the existing hold-first override window (10 s at hover), then the existing **pose-trust gate**: gate closed (the shipped default, estimator open problem) → **LAND in place**, logged as `rth_gated_land_in_place`. Battery does not weaken the gate.
  - **Critical** (soc ≤ 0.10): **LAND immediately — hold window bypassed** [design]. Justification: the energy state only worsens; hovering out a 10 s window burns the last of the pack and converts a controlled descent into an uncontrolled drop. Logged as `battery_critical_immediate`.
- Battery graduated out of `missing_triggers` (research-repo wording; in ThermoNav's convention the slot is listed until battery inputs are supplied — with inputs supplied, `info["missing_triggers"]` returns `["link", "nav_health"]`, and `["nav_health"]` when link is also modeled).
- Mission loop: the battery is stepped once per outer step (dt = 1.0 s) at the power implied by the vehicle's **own velocity** (onboard state — no ground-truth leakage), phase-classified (hover/cruise/climb/descent) for the record. Manifest rows gain `batt_enabled, batt_start_soc, batt_final_soc, batt_min_soc, batt_min_voltage, batt_wh_used, batt_final_ttr_s, batt_reserve_crossed, batt_critical_crossed`.
- The 12-spin decider kernel is **not** extended with a battery input (its input set is perception + timers; it is exercised standalone, not in the mission loop — same status as before this build). Battery acts through the host policy, which is where the ladder's wired path lives.

### Two guard findings this build forced (measured, not hypothetical)

**Finding 1 — escalation LAND was dead code (floor-layer guard bug), and its two-sided fix.** The loop's LAND actuation guard `_below_clear` checked the column down to *and including* the floor layer zz=0. That layer carries the boundary prior (logodds 2.0 on every boundary voxel), so its believed occupancy is ~0.88 everywhere by construction — every column read "occupied", every LAND was vetoed. Evidence: instrumented drain mission, offending voxel (11,16,0), marg 0.881, **GT occ = 0**; historically consistent with the 2026-10-10 whiteout validation (esc_land_vetoed 156, zero `landed` outcomes, all max_steps).

Round-1 fix (exclude zz=0) was **overbroad — red-team RT-1 FAIL**: obstacles occupying *only* zz=0 are real (159 GT columns across 6/30 corpus rooms; 147 in indoorcd_117; zero in this build's 4 validation rooms, so the set structurally couldn't see them), and the red team's synth_low mission collided with one post-fix (pre-fix it held). The exclusion also left an endpoint hole: at voxel z=1 the check range was empty, certifying the final ~2 m CLEAR vacuously.

**Current fix (evidence-keyed, round-2):** zz=0 is back in the checked column (the range includes it at every altitude, closing the z=1 hole), but a floor voxel counts as occupied **only if its logodds accumulator exceeds the boundary-prior value 2.0** — hit support beyond the prior (+0.60 confirmed / +0.18 provisional per hit; free-space traversals pull it below). At or below the prior it is the landing surface: ignored, neither blocking nor clearing. Plus a **creep final approach**: a descent that is not fully verified (guard clear *and* every floor voxel in the footprint carrying scan evidence) flies its last 2.5 m at 0.15 m/s instead of 0.5 m/s [design]. Measured reason creep exists: in synth_low the evidence-keyed check alone was one step too late — the patch was first hit by a scan at pose z=1.05, the guard vetoed, and the hover PID's ~0.73 s settle still sank the vehicle 1.053 → 0.998 into the patch. At creep entry speed the settle sink is ~0.03 m and the guard gets ~3× the scans. Regression tests cover both directions (prior-only floor does not block; hit-supported floor vetoes; z=1 endpoint checks zz=0); mission re-validation is in the Validation section (synth_low and the 117 patch now **hold**, no collision).

**Finding 2 — the veto trap at low battery, and the forced-landing rule (wording corrected per RT-4; scope per RT-3).** A battery LAND can be vetoed step after step while the pack drains — the pre-fix runs ended airborne at max_steps with soc = 0.0 (in-sim artifact wording: soc = 0 has no physical consequence in the sim; see Known Limits). Rule implemented at the LAND call site (guard refactored to `_below_state`: 0 clear / 1 unknown / 2 occupied). **The switching is soc-keyed, not trigger-keyed** — there is no trigger-provenance check anywhere in the call chain, and this report's round-1 sentence claiming otherwise was wrong:
- While soc > CRITICAL, every LAND — whatever started it — is strict-guarded (unknown or occupied → hold).
- The step soc crosses CRITICAL, the battery row *independently demands LAND*, and from that step the guard relaxes for that landing: **occupied still vetoes**, but **unknown no longer vetoes** — the vehicle descends into unverified space, logged per step as `esc_batt_land_unverified`. A perception-triggered landing that was holding on veto therefore switches to forced-unverified descent at the crossing — coherent, because at soc ≤ 0.10 the battery is also demanding the landing, but it is a soc-keyed switch, not a provenance rule. Verified invariants the other way (red team): **no path descends unverified above critical; no path descends into believed-occupied at any soc.**
- **RT-3 scope correction:** in practice the occupied-veto's reach in-suite is narrow. The lidar's elevations are ±0.26 rad, so the column directly below a hover is essentially unobservable — in *every* clear-air mission run (builder's and red team's), the descent column at decision time was **UNKNOWN, never believed-occupied**. The operative protection at critical is therefore the unknown-column rule, full stop; "occupied still vetoes" mostly bites in whiteout phantom conditions (RT-2), not in the clear-air geometry that matters.

### Which timers are now grounded?

**None of the three.** Honest accounting:

| Timer | Status after this build |
|---|---|
| 5 s degraded → hover | Still [design]. Battery gives no evidence about perception-degradation timing. |
| 30 s degraded → RTH | Still [design]. Unchanged. |
| 10 s hold-first override window | Still [design] — with one scoped exception: the battery model *motivates bypassing* it at critical soc (above), but does not ground its length. |

What the model does ground: the **battery trigger rows are state thresholds, not timers** — they fire on charge fraction / time-to-reserve, which are calculated from the pack model rather than chosen as durations. TTR is now a computed quantity the ladder can see (it drives the warn band), where before there was no energy clock at all.

## Validation

Scripts: `test_battery.py`, `test_escalation.py` (regression), `run_battery_validation.py` → `battery_validation.csv`, `run_battery_rt1_validation.py` → `battery_rt1_validation.csv` (dual-purpose: every lever a column, git commit per row). Backend numpy, noise 0.15, paired arms in one process (same seeds/noise/cfg; the only lever is the battery).

### Unit tests — 14/14 pass (`test_battery.py`)

Discharge monotone over 120 hover steps and clamped at empty · sag strictly increases with load (100 W @ 16.31 V > 400 W @ 14.61 V > 400 W @ soc 0.3 @ 10.35 V) and equals I·R exactly · TTR falls with draw and is 0 at/below reserve · **stepped TTR matches brute-force discharge within 2% (670 s @ 282.5 W, 377 s @ 450 W)** · phase ordering climb 310 > hover 282 > descent 269 W, cruise within 10% of hover · full hover discharge draws 63.6 Wh vs 74 Wh nameplate · **saturation bookkeeping: past the max-power point, Wh books delivered power (ocv²/4R), not requested** · policy rows: warn logs once with no action, reserve → gated LAND / open-gate RTH with window respected, critical → immediate LAND, reserve outranks perception hover, no-input → legacy · **below-state guard: clear/unknown/occupied distinguished; floor prior-only = surface (no block), floor hit-supported = veto, z=1 endpoint checks zz=0** · **floor-evidence: all-prior footprint = unverified, all-off-prior = seen**. Full `test_escalation.py` also passes unmodified (decider grid 128/128, Z1 rails, all RTH-gate tests).

### Mission validation — round-2 code (`battery_validation.csv` + `battery_rt1_validation.csv`, commit `ab8eb1c21e8c80a27c5b01494c5b6668fa480040`)

**Nominal paired regression — PASS** (unchanged from round 1): outcomes identical per pair, reserve never crossed, battery never engaged (min soc 0.914–0.981).

**Drain arm (start 0.215):** 004 s0 and 071 s1 **landed**, min soc **0.0692** (the creep final approach costs ~0.008 more soc than round 1's 0.0769 — the price of the slower last 2.5 m). 128 s1000 and 185 s2 **collision**, min soc 0.0928, 5 unverified steps — *unchanged from round 1*: those columns hide obstacles at zz ≥ 1 that the stack has never observed (the RT-3 geometry), a different class from the floor-layer class RT-1 fixed. **Critical arm (start 0.09):** 004/071 both **landed**, min soc **0.0575**, 13 unverified steps, zero vetoes.

**RT-1 re-validation (constructions replicate the red team's runner exactly):**
| Scenario | Round 1 (post-fix) | Round 2 (evidence-keyed + creep) |
|---|---|---|
| synth_low (zz=0-only 3×3 patch under start, soc 0.19) | **collision** (51 vetoes, 13 unverified) | **max_steps — held, no collision** (174 vetoes, 13 unverified, in-sim: soc 0.0 at the step cap) |
| rt117_patch (same patch in indoorcd_117, soc 0.19) | (not run) | **max_steps — held, no collision** (174 vetoes) |
| rt117_crit (indoorcd_117 natural, soc 0.09) | (not run) | **landed**, min soc 0.0575, 13 unverified steps |
| synth_tall (column zz 0–7 under start, soc 0.09) | collision (RT-3's run) | collision (1 unverified step) — the RT-3 unobservable-column case; recorded, no pass claimed |

**How to read the remaining drain collisions (the honest negative).** The forced-landing rule trades between two measured outcomes: without it, all 4 drain missions in the pre-fix run stayed airborne until soc = 0.0 at max_steps (in-sim artifact: no empty-pack physics; the physical reading is an uncontrolled drop at altitude). With it, 2/4 land clean and 2/4 descend onto an obstacle the map never observed — in a column that is *unknown*, not floor-layer. That class is not fixed by RT-1's change and is not claimed fixed; it motivates the parked follow-up: **landing-site selection at critical** (step sideways to the nearest verified-clear column before descending). No drain/critical mission remained airborne at empty; the runner's veto-trap gate passes.

**Whiteout spot-check (guard-fix blast radius):** 2 whiteout+jitter escalation-ON missions (indoorcd_131 s80010, indoorcd_173 s80510, full physics, no battery) re-run on the round-2 guard: both **max_steps, esc land, 156 vetoes, 0 collisions — identical to the 2026-10-10 recorded baseline and to round 1**. Log: `whiteout_spot.log`.

## Paul's safety layer (`battery_safety.py`, decided 2026-10-11)

Paul's design, implemented exactly as decided (this section is the disposition of Q1/Q2 below — the PI ruled; the build followed):

- **One strict low-battery FLAG** at RESERVE. The battery *model* above stays a continuous gauge; the safety layer is the flag. Nothing outranks the flag once it is terminal (countdown end).
- **A bounded countdown starts at RESERVE, sized from MEASURED draw under real load.** The meter (`MeasuredDrawMeter`) reads the gauge's own post-step telemetry — soc and Wh actually consumed, what a coulomb counter / power meter reports — over a trailing 10 s window, and the countdown is `(soc_now − CRITICAL) / measured drain rate`. It is **not** sized from the model's soc percentage or its predicted draw, and "raise CRITICAL to ~19%" was **not** substituted (Paul rejected that lever; CRITICAL stays 0.10 as the soc the countdown is sized to *reach*). The meter **keeps measuring**: if the measured drain worsens mid-countdown, the end is re-sized *earlier* (never later). If the flag fires before the meter has a window (scripted low-soc starts), the layer holds in `HOLD_METERING` while the window fills — Paul's words: the countdown "will drain more battery but it buys us an actual meter." Every countdown step is logged as a calibration row (measured draw, drain rate, countdown remaining, both reachability computations) → `battery_safety_calibration.csv`, with provenance (script, seed, room, runtime-captured git commit) on every row.
- **Veto scope:** during the countdown window the loop's occupancy veto (unknown *or* believed-occupied) may still hold a landing. **At countdown end the action is LAND_TERMINAL: the guard is bypassed entirely** — the veto does not outrank the flag (`esc_batt_terminal_land` in the manifest).
- **Reachability cascade (Paul's addendum + terminology correction):** when the flag fires, the action is decided by the measured meter — **RTB** (return to **BASE** = takeoff pose) if affordable, else **RTC** (return to **CHECKPOINT**) if affordable, else **emergency land in place**. Affordability, in soc units: `measured drain rate × (distance / 1.0 m/s [design return speed]) × 1.25 [design margin]`, and the projected arrival soc must stay ≥ 0.05 — a **[design] plan-time threshold on the projection, NOT an outcome floor** (RT2-2: projections exclude path/speed reality, post-arrival veto holds, and descent cost; measured finals land well below them — rtb_clear projected arrival 0.178, actual final 0.0693). Land-in-place is the fallback, not the default.
- **Continuous re-check, abort DOWN only:** every step of a return leg re-evaluates affordability at the current measured rate; a leg that becomes unaffordable aborts RTB → RTC → land, never upward.
- **Checkpoint spec (smallest honest version, chosen here):** a checkpoint is *a position the vehicle itself occupied during this flight while nav-health reported trustworthy*. It is refreshed every outer step (1 Hz) whose trust signal is trustworthy (so it is the *last* trustworthy pose), frozen the moment trust is lost, and refreshed again if trust returns. At takeoff the start pose is the initial checkpoint, so until a trustworthy in-flight refresh replaces it, checkpoint == base and RTC coincides with RTB. **Interface:** the signal is one boolean with the semantics of the nav-health monitor's `pose_trustworthy` property — True only when nav-health is HEALTHY — as documented in `nav_health.py` / `NAV_HEALTH.md` on branch `muse/roadmap-navhealth-2026-10-11`. That monitor does not exist in this branch (parallel build), so the mission loop exposes an **explicit stub slot**, `cfg["nav_trustworthy"]` (bool or callable `(t, pose) -> bool`; `nav_trustworthy_stub` in `mission_dynamics_v5.py`). Absent the slot it returns False — this branch's shipped status quo — and no in-flight checkpoint is ever recorded. The stub is an interface placeholder, not a health estimate.
- **Integration:** `cfg["battery_safety"]`, default OFF (legacy behavior bit-identical; paired gate below). When ON, the safety layer owns the battery decision — the escalation policy is called *without* battery inputs so the layers never both act — and RTB/RTC ride the ladder's existing return mechanism: the mission goal is retargeted to the snapshot target and the normal S2/A* stack flies it; arrival converts to LAND. The D3 goal-blocked check is skipped on return legs (measured reason: it ended the first whiteout_term attempt as `goal_blocked` when phantom evidence marked the *base* voxel occupied — a return target is not a mission goal). A scenario lever, `cfg["battery_extra_load_w"]` (float or callable), adds a scripted real load to the pack step (e.g. a payload heater); decision code never reads it — it reaches decisions only through the gauge telemetry the meter measures, and delivered power saturates at the pack's max-power point (~530–880 W).

### Safety-layer validation (`run_battery_safety_validation.py`, executed at `f5f0ddf`; re-executed at `7cb44b4` with rtc_v3/v4 added — all prior rows identical, all rows now stamp `7cb44b4`)

Unit: `test_battery_safety.py` **12/12 pass** (10 original + 2 RT2-1 tests) — meter reads back stepped hover power (282.5 W measured vs 282.5 W stepped); countdown halves when the measured drain doubles (100 s @ 0.001 soc/s vs 50 s @ 0.002 — sizing follows the measurement); checkpoint refresh/freeze; **each cascade rung decided deterministically** (RTB / RTC / land) at synthetic distances; **mid-leg abort DOWN** (RTB → RTC → land under a drain spike, never up); countdown-end terminal; countdown re-size on spike (end 199 → 119, terminal at soc 0.100); calibration-row schema; **RT2-1: flat-soc/screaming-Wh ends the hold via the disagreement cross-check (countdown 0.67 s sized from the Wh channel, terminal at t=3), and a fully dead gauge terminates at the 30 s bound**.

Missions (numpy, noise 0.15; `battery_safety_validation.csv` + 355 calibration rows, all stamped `7cb44b4`):

| Scenario | Result |
|---|---|
| rtb_clear (071 s1, start 0.26, trust on) | Flag at t=42, 11.3 m from base, need 0.020 soc → **RTB**; flew back (min dist-to-base 1.27 m), LAND vetoed 46 steps during countdown, **terminal landing at soc 0.098, landed, min soc 0.0693** |
| whiteout_term (131 s80010, whiteout+jitter, FULL_PHYSICS, start 0.215) | **The RT-2 composition, re-run under the decided design: vetoed 67 steps during the countdown, countdown end at t=77 / soc 0.0999, LAND_TERMINAL → landed, min soc 0.0691.** (Round-1 measured: airborne at soc 0.0, vetoed 181, rule never fired.) |
| abort_spike (071 s1, +3000 W requested from t=48, delivered capped) | Round 2 (52b3b3d, `*_r2` artifacts): **abort-down 1×** (RTB → land; checkpoint also unaffordable), then soc hit 0.0 in the guarded hold — the negative that motivated re-sizing. Round 3+ (current code): the re-sized countdown end (t=69, soc 0.098) preempted the abort re-check — a downward response to the same spike — **landed, min soc 0.0127** (the spiked drain ate most of the descent budget; recorded, not hidden) |
| rtc_attempt (071 s1, +2500 W, trust t<28) | **Picked RTB, landed, min soc 0.0144** — base was affordable in this room too. (RTC itself is demonstrated by rtc_v3/v4 below.) |
| rtc_v3 (004 s2, start 0.335, +2500 W, trust t<20 — red-team round-2 construction, ported RT2-3) | Flag at t=29, 30.85 m from base: RTB correctly rejected (projected arrival 0.022 < the 0.05 plan-time threshold), checkpoint 15.19 m / need 0.0861 → **RTC, landed via countdown-end terminal, min soc 0.0142** — matches the red team's independent run exactly |
| rtc_v4 (004 s2, start 0.32, +2500 W, trust t<24 — same source) | **RTC, landed via terminal (6 vetoes during countdown), min soc 0.0134** — matches the red team's independent run exactly |
| abort_synth (empty 48 m room) | Mission **succeeded in ~17 steps, before the flag could fire** (min soc 0.2362), in both attempts. Recorded, not gated; a longer synthetic leg needs a much larger room and Gibbs cost grows with volume — not chased. |
| paired OFF vs ON @ full pack (004 s0) | Outcomes **identical** (success/success, min soc 0.9785 both) — the layer is inert until the flag |

**ThermoNav-side validation (graduation run, `sim/run_battery_safety_validation.py` → `sim/battery_safety_validation.csv` + 356 calibration rows, runtime-stamped with the executed ThermoNav commit): all 6 gates PASS, 0 collisions.** The ThermoNav stack reproduces the research numbers essentially exactly: rtb_clear landed via RTB (cd 68.7 s, 46 vetoes, min soc 0.0693); abort_spike landed (min soc 0.0127); rtc_attempt landed via RTB (min soc 0.0144); rtc_v3/v4 landed via RTC (min soc 0.0142 / 0.0134); paired OFF/ON identical (min soc 0.9785). Two ThermoNav-specific notes: (1) whiteout_term ends **landed_disarmed** (cd 69.0 s, 67 vetoes, 23 terminal steps, min soc 0.0675) — the countdown-end terminal landing fired and ThermoNav's DISARM then fired after confirmed touchdown, demonstrating the required ordering end-to-end (DISARM does not outrank a terminal battery landing; it follows it). The whiteout cell runs without the research loop's jitter component (ThermoNav has no jitter controller; see port note). (2) The first ThermoNav run exposed a porting omission — the loop's end-of-step success check lacked the research loop's return-leg guard, so return legs ended as airborne "success" at the retargeted goal; the guard (`_return_target is None`) was ported and the re-run above is with it (run-1 CSVs preserved as `*_run1.csv`).

### Red-team round 2 — follow-ups (implemented for graduation, 2026-10-11)

Independent report: `REDTEAM_BATTERY_R2.md` (branch `muse/roadmap-battrt2-2026-10-11`) — **PASS WITH CAVEATS**, verdict closable, RTC gap judged CLOSED. Dispositions:

- **RT2-1 — HOLD_METERING unbounded on a stuck soc gauge: FIXED, both mechanisms, smallest honest bound.** Design choice: implement the cross-check AND the time bound (either alone leaves a hole — the cross-check alone cannot help a fully dead gauge whose Wh channel is also flat; the time bound alone would waste up to the bound on an obviously-lying soc channel). (a) *Cross-check:* if the meter window shows ≥ 0.5 Wh consumed while the soc drain rate reads ≤ 0, the channels disagree — the countdown is sized from the Wh-implied drain rate (consumed-Wh rate / 74 Wh capacity). The Wh channel is used **only** for this fallback; normal sizing still reads the soc channel, which on a real aircraft is the coulomb counter's own output. (b) *Bound:* HOLD_METERING ends after 30 s of flag time (plus a 120-step backstop) — sizing from the Wh channel if it implies drain, else LAND_TERMINAL, the safest available action on a dead gauge. Both paths log loud events (`soc_wh_disagreement` / `hold_metering_bound`). Unit evidence (`test_battery_safety.py`, 12/12): flat soc 0.19 + 10 Wh/s Wh channel → hold ends after 2 steps, countdown 0.67 s sized from the Wh channel, terminal at t=3 (red team's probe held 600–1000 steps pre-fix); flat soc + flat Wh → terminal at exactly t=30.
- **RT2-2 — LANDING_FLOOR_SOC language: CORRECTED everywhere.** It is a plan-time threshold on the projected arrival soc, not an outcome floor; measured finals land well below projections (rtb_clear projected 0.178 vs final 0.0693; abort_spike final 0.0127; rtc_v3 projected 0.111 vs final 0.0142). Wording fixed in `battery_safety.py` (module docstring + constant comment) and in this report (cascade bullet above). Nothing dies in-model because of the gap — bounded drain rate + 1 Hz re-check + abort-down (red team's cascade analysis) — but no document may quote 0.05 as the charge landed with.
- **RT2-3 — rtc_v3/v4 ported into the build's runner: DONE.** The red team's constructions (indoorcd_004 s2, +2500 W, trust cut before the flag) now run as gated scenarios in `run_battery_safety_validation.py`; at `7cb44b4` both picked RTC and landed with the red team's exact numbers (cd 21.3 / 21.1 s, min soc 0.0142 / 0.0134, vetoes 0 / 6), and all 7 prior mission rows + 269 prior calibration rows reproduced identically (355 calibration rows total now).
- **RT2-4 — `_below_state(logodds=None)` latent trap: FIXED (trivial, no behavior change).** `logodds` is now a required keyword argument in `mission_dynamics_v5.py`; every existing call site already passed it explicitly, and the guard tests pass unmodified. (The red team's A2 note — a hit-count floor key so backscatter carving cannot erase a floor hit — is a design change, not a nit; recorded here as a parked follow-up, not built.)

## Known limits

- **Linear OCV curve.** Real LiPo voltage has a plateau and an end knee; at the reserve floor the model's loaded terminal voltage is ~10.9 V (2.7 V/cell under load) — the real pack would be at least as unhappy. The reserve floor is a design choice, not a measured safe limit.
- **No Peukert / rate-capacity effect, no temperature, no ageing, no per-cell imbalance.** At our ~3.7 C hover draw the Peukert penalty for LiPo is small but real and unmodeled — endurance estimates err optimistic.
- **Power is velocity-only.** The model reads the vehicle's velocity, not its acceleration or attitude; aggressive maneuvering transients are smoothed away by the 1 Hz step. Wind rejection work (Phase-3 wind model) is not priced in.
- **Propulsion estimate sits ~8% below the registry's class-typical band** (see cross-check). If the registry band is right, every endurance/TTR number here is optimistic — and the stack multiplies worse than any single ingredient suggests. **See "Open questions for the PI" (RT-6) for the multiplied table; thresholds were NOT changed in this round.**
- **Indefinite whiteout, legacy configuration (safety layer OFF): NOT bounded by the battery (RT-2 — round-1 claim corrected; the claim as originally written was false).** Both halves measured by the red team and accepted: (a) from a full pack, whiteout + battery ends at **soc 0.787 at MAX_STEPS = 200** — critical would need ~850 steps, 4.2× the mission cap, so in any normal hold the battery never engages; (b) scripted into the drain band in whiteout (start 0.215), the mission ends **airborne at soc 0.0, vetoed 181, unverified 0** — whiteout phantom evidence reads the column below believed-OCCUPIED, the occupied-veto never relaxes, and the forced-land rule never fires. **Safety-layer configuration (ON): BOUNDED, by Paul's decision** — the veto may hold only during the measured countdown; at countdown end LAND is terminal regardless (see "Paul's safety layer": whiteout_term landed at min soc 0.0691 after 67 vetoed steps). Roadmap item 2 (indefinite whiteout) is therefore bounded *by the countdown* in the safety-on configuration and stays open in the legacy one; both statements are scoped to their configuration, and the safety layer is cfg-gated OFF by default like every layer in this stack.
- **Forced descent into unverified space collides in 2/4 drain cases** (measured above; zz ≥ 1 obstacles in never-observed columns). Landing-site selection at critical is the parked fix, not built here.
- **soc = 0 has no physical consequence in-sim** (RT-7): the vehicle flies on at soc 0.0 until the step cap. Every "airborne at empty" outcome in this report is that artifact — read it as "in-sim: max_steps at soc 0"; the physical reading (an uncontrolled drop) is interpretation, not simulation.
- **Battery without escalation is telemetry only** (RT-7): with `cfg["battery"]` set but escalation unset, the model steps and fills the `batt_*` manifest fields, but nothing consumes its state — no ladder action. The escalation ladder is the only consumer, by design; this is now documented in the loop code as well.
- **`battery_start_soc` must be in (0, 1]** (RT-7): 0 raises `ValueError` by design (a modeling error, not a scenario) and surfaces as an ERROR row in mission manifests.
- **Safety-layer reachability uses straight-line distance** to the target pose: an obstacle-avoiding path is longer, and the 1.25 margin factor [design] only partially compensates. A leg judged affordable can therefore be unaffordable as flown — the continuous re-check + abort-down is the mitigation, and it is why the re-check exists.
- **Safety layer: RTC is demonstrated at mission level** (rtc_v3/v4, red-team round-2 constructions ported into the build's runner, RT2-3) and the checkpoint is only as good as the nav-trust signal fed through the stub slot; with the default stub (False) RTC == RTB by construction. Wiring the real nav-health monitor in place of the stub is a separate roadmap dependency, not a battery defect (red-team round 2, §5).
- **Safety layer: the countdown's boundedness rests on the gauge channels (RT2-1, fixed — see follow-ups).** Decisions read the soc channel; the Wh channel is the cross-check. A soc gauge stuck flat is bounded by the disagreement cross-check + the 30 s metering bound, but a gauge that drains *plausibly but wrongly* (slow drift, wrong scale) is not detectable by this layer — that is a gauge-calibration problem, out of scope here.
- Battery does not yet feed the decider kernel, mission planning (no energy-aware goal feasibility), or the variable registry — registry update is a follow-up for the parent/PI to schedule.

## Open questions for the PI — BOTH DECIDED by Paul (2026-10-11), dispositions recorded

**Q1 — DECIDED: the measured countdown, not a threshold move.** Paul's ruling (2026-10-11): at RESERVE a bounded countdown starts, sized by *measured* draw under real load — measured time beats the model's ~8%-optimistic percentage — and at countdown end the vehicle lands regardless. Neither RT-6 lever was taken as framed: CRITICAL was **not** raised to ≳0.19 and RESERVE was **not** raised to 25%. The bounded-hold lever survives in measured form: the hold at reserve is bounded by the *measured* time-to-critical, re-sized as the meter updates. The optimism stack is answered by measuring drain in flight (consumed Wh / soc actually flown) instead of trusting modeled soc timing. Residual risk, honestly stated: in simulation the meter reads the stepped pack itself, so this validates the *mechanism*, not a real gauge; on hardware the same code reads the real meter, which is the point of the design — unmeasured here. The RT-6 table below is retained as the record the decision was made against.

**Q2 — DECIDED: the veto holds only during the countdown window.** Paul's ruling supersedes options (i)/(ii)/(iii) below: the whiteout occupancy veto may hold during the countdown, and **at countdown end LAND regardless of any condition** — no N-step count, no whiteout-only carve-out; the flag is strict and terminal. Implemented as `LAND_TERMINAL` (guard bypass) in `battery_safety.py` and validated in whiteout_term (above). Options (i)–(iii) are retained below as the record.

## Open questions for the PI (as posed in the round-2 response; decided above)

**Q1 — Thresholds against the multiplied optimism stack (RT-6).** The model counts charge at its own (possibly optimistic) draw. If the registry band is the truth (it is itself unmeasured — this is conditional arithmetic), with avionics held at 25 W the true/model draw ratio is k = 1.080 / 1.150 / 1.221 at band bottom/mid/top, and coulomb counting gives:

| Registry truth | True soc when model says reserve (0.20) | True soc when model says critical (0.10) | Model soc at true empty |
|---|---|---|---|
| 280 W (bottom) | 0.136 | 0.028 | 0.074 |
| 300 W (mid) | 0.080 | −0.035 (**already empty**) | 0.131 |
| 320 W (top) | 0.023 | −0.099 (**already empty**) | 0.181 |

In the guard-held path (reserve LAND vetoed on an unknown column — the drain missions' actual path, 58 vetoes before critical in round 1), at band mid/top **the pack is truly empty before the model's critical trigger can fire**: the vehicle drops during the hold, at model soc 0.13–0.18. Lever analysis (red team's, verified against the table): **raising RESERVE to 25% does not move that drop point** — the drop is set by when critical fires relative to true empty. The levers that move it are **CRITICAL ≳ 0.19 at band top** (so the forced landing triggers while true charge remains) or a **bounded hold at reserve** (cap the vetoed hold by time/charge, then force the landing). Decide reserve/critical against this table, not against the model's own soc.

**Q2 — Should battery-critical override a believed-occupied veto in whiteout? (RT-2.)** Options:
- **(i) Occupied veto absolute (current build).** Clear-air believed-occupied is honored absolutely; in whiteout the composed system hovers to empty (in-sim: max_steps at soc 0) because whiteout phantom evidence reads the column occupied.
- **(ii) Occupied veto relaxes at critical after N held steps** (a bounded phantom gamble): if the landing has been veto-held for N steps at soc ≤ critical, descend anyway. Bounds the whiteout ending, but also relaxes against *real* believed-occupied columns in clear air if the hold came from there.
- **(iii) Occupied veto relaxes at critical only when perception is D5-degraded / whiteout-flagged** (phantoms likely), never in clear air. Targets the phantom mechanism specifically; keeps clear-air protection absolute.
Context from RT-3 that bears on the choice: in-suite, the sub-vehicle column is essentially unobservable (lidar ±0.26 rad), so clear-air "believed-occupied" below is rare — the synthetic tall-column collision came through the UNKNOWN path, not the occupied path. The occupied-veto's practical bite is concentrated in exactly the whiteout/phantom regime option (iii) names.
