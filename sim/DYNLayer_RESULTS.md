# Dynamic-object layer — results (v3 fix round)

**Date:** 2026-10-11 (v2 build + red-team + fix round, same day).
**Branch:** `muse/roadmap-movers-2026-10-11`.
**Claim label:** Z1-legal formulation, validated in simulation
(host-side classical layer; no TSU/Z1 surface, no silicon claim).
**Red-team round 2:** PASS WITH CAVEATS (report:
`DYNLayer_REDTEAM2.md`, branch `muse/roadmap-moversrt2-2026-10-11`
@ `8c3929e`, executed commit `44befad`) — Phase 2(a) ruled
closable pending graduation. The RT-2 corrections below (the 071
head-on relabel, the 071 dual-mover correction, the fpwide median,
the `sim_time_s` dead column, and the SCOPE BOUNDARIES section)
are documentation-only: no code or CSV was changed by them.

## Verdict: FIXED + RE-VALIDATED — 43/48 → 7/48, diagnoses corrected

The v2 build scored 43/48 → 18/48 paired (25 fixed / 0 broken). The
red-team (`DYNLayer_REDTEAM.md`, branch
`muse/roadmap-moversrt-2026-10-11`) returned PASS WITH CAVEATS: the
headline survived independent reproduction exactly, but two of the
three failure-class diagnoses in the v2 doc were wrong about
mechanism, and it filed findings F1–F5. This fix round (v3, code
commit `a658150`) addresses all five; the full paired sweep was
re-run at that commit:

| arm | collisions (48 paired mover cells) |
|---|---|
| OFF (no layer, re-run at `a658150`) | **43/48** (baseline reproduced exactly) |
| ON v2 (own commit `06f6ae9`, context) | 18/48 |
| ON v3 (dynlayer v3 + dodge) | **7/48** |

Paired cell flips vs OFF: **36 fixed, 0 broken.** No cell that
survived without the layer collides with it. All 4 ON baselines and
all 12 wide-seed no-mover missions (below) succeed.

By geometry (OFF → v3 ON): crossing 16/16 → 1/16; head-on 16/16 →
4/16; overtake 11/16 → 2/16.
By speed: 0.5 m/s 9/12 → 2/12; 1.0: 10/12 → 1/12; 1.5: 12/12 → 1/12;
2.0: 12/12 → 3/12.
By room: 004: 12/12 → **0/12**; 071: 11/12 → **0/12** (but one
timeout — see below); 128: 10/12 → 4/12; 131: 10/12 → 3/12.

Outcome-level changes vs v2 (not just collision flips): 12 cells
improved collision → success (all four 128 crossing, all four 004
head-on, 128 head-on v1.0, 071 head-on v2.0, 071 overtake
v1.5/v2.0); **2 cells degraded vs v2** and are reported as such:

- **071 head-on v1.0: v2 success → v3 `max_steps`** (469 motion
  refusals). **Corrected per RT-2 (§3): the drone did not
  "avoid" the mover.** The real mover was raw-painted in 10 scans
  (13 hits) and produced 1 appearance and no confirmed track;
  all 378 `blocks()` calls occurred under phantom-only veto, and
  the mover's closest approach (0.98 m) was incidental. Relabel:
  FP-driven starvation with incidental closest approach — the
  veto field of false-positive tracks kept the drone from
  completing before the step cap (see FP tax and class 4).
- **131 head-on v0.5: v2 success → v3 collision** (detected,
  confirmed, 1 dodge, still collided; OFF also collides on this
  cell, so it is not an OFF-paired break — it is a v2-relative
  regression). **Diagnosis corrected per RT-2 (§4): this is a
  dodge-geometry + no-retreat failure, not late engagement —
  see class 3 and SCOPE BOUNDARIES (iii).**

## What the fix round changed (per red-team finding)

**F1 — Wall-adjacency blindness (was HIGH).** Adjacency suppression
no longer deletes hits: a hit suppressed *only* by wall adjacency
becomes a wall-candidate, and candidates are chained across scans
(track-before-detect). Promotion to an appearance requires a 3-scan
chain of motion-consistent candidate clusters (steps 0.4–2.2 m,
successive-step direction cosine ≥ 0.7, 3rd point within 0.75 m of
the constant-velocity prediction) plus a flicker filter (a
candidate voxel that was a hit 2 scans ago is dropped). The
promoted track is seeded with the chain points — a real velocity.
Once hot, a chain keeps promoting while the hugger persists.
*Design history, honestly:* a 2-scan movement test was implemented
first and measured in closed loop — 145 spurious promotions in one
004 crossing mission (the sparse rotating ray set makes static wall
voxels flicker, and flicker centroids drift), so single-step motion
is not evidence in this sensor. Four other candidate
discriminators were measured and rejected before settling on
chains: next-scan voxel fate, suppressor shadowing, cluster
compactness, t-3 hit history — none separates flicker from mover.
*Result:* corridor 004 goes 4/12 → 0/12 collisions; the head-on
corridor cells now detect the mover (real tracks, vetoes, dodges).

**F2 — Phantom confirmation.** Authority is now tiered:
- *Veto authority* (`blocks()` tube): 2 appearances **and** fitted
  speed ≤ 2.6 m/s (the layer's own mover envelope: staged movers
  cap at 2.0 + centroid slop). The red-team's U4 phantom
  (+2.0/+2.0 m/s fabricated, 2.83 m/s fit) now gains **no tube at
  all** — verified by re-running the RT unit probes against v3.
- *Full authority* (the dodge-facing `confirmed` set): a 3rd
  appearance within 1.5 m of the track's CV prediction. Two noise
  hits can still buy a local veto for one TTL window (bounded,
  measured below) but cannot trigger a dodge.
- *Mover-cell cost of the extra scan:* none net — collisions fell
  18 → 7 and 22 dodges still fired. Confirmation delay did not
  give back the gains on the paired cells.

**F3 — Two-mover merge corruption.** (i) A cluster contended by
≥ 2 tracks whose predictions are ≥ 0.75 m apart is split by
nearest prediction before association (piece centroids kept in
the hit-point frame via the parent cluster's sub-voxel offset).
Re-running RT probe U1: both tracks now survive the crossing with
velocities exactly +1.00/−1.00 at every scan (v2: one corrupted
−1.25 track, the other starved). (ii) `prev_appeared` decays over
2.5 s instead of being emptied by one zero-appearance scan
(self-lockout): in U1 the post-merge return scan fires again.
*Residual (documented, not fixed):* the scan in which two movers
swap voxels in place produces no differencing evidence at all
(U1 t=4: 0 appearances) — there is nothing to associate; and
closed-loop dual movers in 071 (fresh set) now end in `max_steps`
instead of v2's success — veto-taxed into a timeout (and, per
RT-2 §5, that cell effectively tests one mover — see the fresh
conditions section below).

**F4 — Tube offset.** `blocks()` no longer truncates the fitted
position with `int()`; it compares against voxel centers
(v + 0.5, radius + 0.5) — same Chebyshev coverage, continuous, so a
fit value of 9.9999999 for a mover at 10.0 no longer shifts the
tube a voxel toward −x/−y. Unit test asserts symmetric ±1 coverage.

**F5 — CSV defect.** `n_dyn_tracks_confirmed` is now the lifetime
count (`n_confirmed_total`); the end-of-mission snapshot survives
under its own name, `n_dyn_tracks_active_end`, alongside new
`n_dyn_tracks_full` / `n_dyn_wall_promoted` columns and the new
lever columns. Example of why it mattered: v3's 071 baseline reads
lifetime 17 vs active-at-end 0 — the v2 CSV would have published 0.
**Correction to the v2 record:** the v2 sweep's aggregates used the
correct lifetime column (`n_motion_confirmed`), but the published
`n_dyn_tracks_confirmed` column itself was the snapshot and is
wrong in `dynlayer_sweep_v2.csv` (preserved, renamed — do not cite
that column from the v2 file). The v2 doc's class-1 text also
misread it (see corrected classes below).

## Corrected failure classes (supersede the v2 doc's three classes)

The v2 doc's classes 1 and 2 were wrong about mechanism (red-team
instrumentation, §2 of their report). Corrected, and re-based on
the v3 sweep's remaining 7 collisions + 1 timeout:

1. **Corridor head-on (004) — WAS misdiagnosed; now fixed.** v2
   story ("detection works, drone hovers, needs retreat") was
   wrong: the mover was *never detected* (wall-adjacency
   suppression killed every hit, F1) and the 40 refusals were the
   clean baseline's phantom tracks. With F1, all four cells
   succeed. A retreat behavior may still be wanted for a *detected*
   head-on mover with no dodge clearance, but it was not the fix
   here; the parked card
   (`~/workspace/parking-lot/corridor-retreat-behavior.md`) has
   been annotated accordingly.
2. **Sparse-evidence cells — split into two unlike mechanisms.**
   (i) *No scan evidence before impact:* the sensor never paints
   the mover associably (RT's instrumented 128 crossing: raw hit
   in exactly the collision scan). Not a layer problem; widening
   the gate cannot help and was not done. Several v3 survivors
   still look like this (131 crossing v1.0: 3 appearances, 0
   confirmed; 131 head-on v2.0: 4/0). Note the four 128 crossing
   cells nevertheless flipped to success in v3 — via earlier
   flight-path divergence from the (few) vetoes, not via a
   confirmed track on the instrumented v2 trajectory; do not read
   them as "128 solved" (128 head-on/overtake survivors remain).
   (ii) *Evidence exists but confirms late/never:* 128 head-on
   v0.5 (5 appearances, 1 confirmed, 1 full — too late), 128
   overtake v1.5/v2.0 (4/1, 3/1). This is the confirmation-latency
   class F2 deliberately trades against phantom authority; the
   residual cells are its price.
3. **Late-engagement cells.** 128 head-on v2.0: detected,
   full-authority, 1 dodge, still collided (min-dist 0.84) — a
   2.0 m/s closing mover engaged too late.
   **131 head-on v0.5 does NOT belong in this class (RT-2 §4):**
   instrumented, the track reached full authority at t=8, 7.2 m
   out, ~4.5 s before impact, with fitted speed 0.49 vs truth
   0.5; the single dodge moved the drone laterally *into* the
   mover's line (origin y 3.5 → 5.8 across the mover's y≈5.3),
   after which the veto tube pinned it there while the mover
   walked the last 2.4 m in. That is a dodge-geometry +
   no-retreat failure; tiering cost 1 scan (veto t=7 → full
   t=8) and is exonerated. See SCOPE BOUNDARIES (iii).
   (v2 doc's minor figure corrected: the three surviving v2 128
   head-on cells had mref 0,1,1 — not "1–2".)
4. **NEW (v3): FP veto starvation.** 071 head-on v1.0 ends in
   `max_steps` with 469 motion refusals — spurious tracks (071 is
   candidate-dense) veto the drone into a timeout. No collision,
   but no mission either. This is the v3 sensitivity increase
   showing its cost in the worst room; see FP tax.
   **RT-2 (§3) confirms and strengthens the mechanism:** phantom-
   veto scans 82, real-veto scans 1, both 0; all 378 `blocks()`
   calls were phantom-only. The real mover never formed a track
   in this cell (13 raw hits → 1 appearance → no track), so the
   0.98 m closest approach is incidental, not an avoidance.

## False-positive cost (v3, honestly bounded)

- Protocol baselines (4): all succeed, 0 dodges. Tax: motion
  refusals 41 (004), **274 (071)**, 1 (128), 0 (131); v2 read
  40 / 0 / 0 / 0. Mean appearances/mission rose 17.4 → 35.2 —
  the layer is more sensitive everywhere, and 071 pays for it.
- Wide set (`run_dynlayer_fp_fresh.py fpwide`, 12 fresh no-mover
  seeds — rooms × seeds 2/3/4, paired OFF/ON, commit `a658150`):
  **12/12 ON succeed** (12/12 OFF succeed). **False dodges: 2 of 12
  missions, 1 dodge each** (071 s3, 128 s4) — the v2-era claim
  "zero false dodges" does NOT survive the wider seed set, in
  either version (the red team already found one at seed 2 against
  v2). Phantom-veto tax across the 12: **median 17.0 motion
  refusals** (standard median of the committed CSV; sorted ON
  taxes 0,0,0,0,1,2,32,40,40,56,99,100 — corrected per RT-2 §7:
  the previously published "median 32" was the upper-middle
  value), max 100 (131 s2), total 370. Full-authority phantom tracks
  appeared in 8/12 missions. Bounded — no clean mission failed —
  but materially higher than v2's 4-baseline profile suggested.
- RT unit probes re-run against v3 (`rt_unit.py` from the
  red-team branch, unmodified): U1 fixed (both velocities exact),
  U2 PASS (wall-hugger produces appearances), U4's fabricated
  tube no longer blocks. U3 residual, **corrected per RT-2 (§3,
  probe U8): a *stationary* repeated noise hit is veto-only at
  2 hits but reaches FULL authority at the 3rd hit (residual 0
  ≤ 1.5 m trivially)** — the v3-era "veto-only phantom" wording
  was one hit out of date. The phantom remains indistinguishable
  from a genuinely stopped mover, which the fresh set shows
  blocking correctly (071 stop-at). Accepted as the
  stopping-mover trade; the general 3-paint bound is recorded in
  SCOPE BOUNDARIES (ii).

## Fresh conditions (red-team cells re-run, `dynlayer_fresh.csv`)

The red team's 12 paired fresh cells re-run at `a658150`: **ON
never worse than OFF on outcome in any pair** (the v2 graceful-
degradation property holds). Vs the red team's v2-era ON outcomes:
three 128 cells improved collision → success (speed-1.25
crossing, diagonal crossing, dual movers); 071 dual movers
degraded success → `max_steps` (timed out — class 4 above).
**Correction per RT-2 (§5): the 071 dual cell effectively tests
ONE mover. The crossing mover records 0 raw hits in 200 scans;
the dual row is identical to the head-on-only control on all 10
diagnostics, and the crossing-only control is identical to the
no-mover baseline on all diagnostics. The earlier wording
"avoided both movers" overstates the cell: one mover was never
seen, and the other was not avoided (see the 071 head-on
relabel above). Duals in 004/128/131 are genuine — their
signatures differ from both singles.** 128 stop-at crossing
still collides in both arms
(class 2(i): no evidence). Parallel-offset: no veto starvation in
either room, as in v2.

## What was built (v2 recap + v3 deltas)

`dynlayer.py` — `DynLayer`, drop-in successor to
`motion.MotionBuffer` in the mission loop's `_MOTION` slot; the
existing `DodgeController` runs on top unchanged (its trail input
is now the full-authority set).

- **Detection:** two appearance channels as in v2 — (a)
  previous-scan differencing, (b) accumulated-map deeply-free —
  plus (v3) wall-candidate chains (F1) and the `prev_appeared`
  decay (F3). Mean appearances/mission 35.2 (15.3 prev-scan +
  15.8 map, union-counted once; wall promotions included in the
  union).
- **Tracking:** CV association, greedy one-to-one, 3.0 m gate,
  3.0 s TTL — plus (v3) contended-cluster splitting (F3) and
  tiered authority (F2).
- **Blocking:** projected tube over [0, 2.0 s], Chebyshev radius
  1, voxel-center comparison (F4).

Unit tests: `test_dynlayer.py` — 11 cases (5 original v2 cases +
wall-hug promotion, stationary-candidate silence, phantom-velocity
authority, tiered authority, two-mover split/decay, tube symmetry).
All pass at `a658150`.

## Condensation justification (why a separate layer)

Unchanged from v2: the static occupancy map cannot hold this state
(its inverse sensor model erases exactly this evidence, measured
−3.40 pinned log-odds), and tracks need object state a per-voxel
field cannot store. The layer sits beside the map, reads only raw
scans (+ log-odds for channel (b)), exports only `blocks()` and the
trail interface, and never writes the map.

## Provenance

- Sweep: `run_dynlayer_sweep.py` → `dynlayer_sweep.csv`, **104
  rows**, every lever a column (including the v3 levers:
  `dyn_speed_envelope`, `dyn_full_hits`, `dyn_full_gate_m`,
  `dyn_wall_min_move_m`, `dyn_appear_decay_s`), `git_commit`
  resolved from the executing worktree; the runner refuses a dirty
  tree. Executed commit: `a6581501c773b4e7b93dc1fc23a90a3085c3f970`
  (all 104 rows). The v2 sweep is preserved as
  `dynlayer_sweep_v2.csv` (its `n_dyn_tracks_confirmed` column
  carries the F5 defect — see above).
- Wide FP set: `run_dynlayer_fp_fresh.py fpwide` →
  `dynlayer_fpwide.csv`, 24 rows (12 seeds × 2 arms), same commit.
- Fresh set: `run_dynlayer_fp_fresh.py fresh` →
  `dynlayer_fresh.csv`, 24 rows (12 paired red-team cells), same
  commit.
- **Known CSV defect (RT-2 §7, documentation-only — CSVs are
  NOT rewritten): `sim_time_s` is a dead column.** It reads −1
  in all 152 v3 CSV rows (104 sweep + 24 fpwide + 24 fresh),
  because the runners read `row["sim_time"]`, which
  `manifest_row` never emits. Do not cite `sim_time_s` from any
  v3 CSV; the fix (correct the key or drop the column) is owed
  by the next sweep runner, not by editing committed data.
- Analyzer: `analyze_dynlayer.py` (paired flips + breakdowns).
- Assumptions unchanged from v2: 1 Hz scan rate, voxelized mover
  rendering (r=0.5 m), clear air, noise 0.15, numpy backend — same
  staging code and traj_fix files, so the OFF arm's exact 43/48
  reproduction is the pairing check.

## SCOPE BOUNDARIES (red-team round 2 residuals, recorded verbatim in substance)

RT-2's verdict is PASS WITH CAVEATS: the headline survives, and
the residuals below are scope boundaries of the closable claim,
not breaks in it. They are recorded here so the claim is never
read wider than the evidence.

(i) **Open-room wall-lane huggers remain blind.** The F1 fix is
corridor-scoped. A mover walking a wall-adjacent lane in an open
room (128: 32 free voxels, 100% wall-adjacent, 1.0 m from the
drone's route) is as invisible in v3 as in v2: at 0.25 m/s, 46
painted scans → 1 appearance → 0 tracks; at 1.0 m/s (in-band),
11 painted scans → 3 appearances → 0 tracks. Mechanism: every
mover hit is map-adjacency-suppressed, and the surviving mover
candidates merge into (are 26-connected to) the wall's own
flicker candidate clusters along the lane, so chain centroids
and the step/cosine/residual tests measure the wall segment,
not the mover. Parked card:
`~/workspace/parking-lot/dynlayer-wall-lane-blindness.md`.
Companion unit-level boundary: idealized sub-band huggers
(≤0.35 m/s, steps below the 0.4 m chain minimum) are blind in
unit probe U6 (0 appearances in 12 scans at 0.2 and 0.35 m/s),
though closed-loop corridor huggers at 0.2 m/s ARE detected via
voxel churn (the sphere's rendered voxel set steps in-band even
when the mover does not).

(ii) **Tiered-authority bound is 3 consistent paints.** Tiering
raises the phantom cost from 2 paints (v2) to 3 *consistent*
paints — at 3, a full-authority phantom exists (RT-2 probes
U7/U8/U9: 3 marching open-space paints, 3 marching wall-flicker
hits, and a stationary repeated paint all reach full authority;
a stationary paint's residual to prediction is 0). In closed
loop the combined false-dodge bound is 2/24 clean missions
(builder fpwide 2/12 + RT-2 extension 0/12, noise up to 0.25);
both false dodges trace to chain-born phantoms marching at
1.1–1.5 m/s along linear features — flicker centroids on a
static linear feature translating under the drone's ego-motion,
kinematically indistinguishable from a real wall-hugger. No
clean mission failed from a false dodge in 24 tries.
Voxel-persistence (whether the track's underlying voxels are
re-painted, not just its centroid) is the open discriminator;
parked, not chased.

(iii) **131 head-on v0.5 is dodge-geometry + no-retreat.**
(RT-2 §4; see class 3.) The dodge moved the drone into the
mover's line and the veto pinned it there; tiering is
exonerated (1 scan). A track now exists in this geometry, so
the parked retreat card's premise ("no track to trigger on")
is satisfiable — the missing piece is the retreat behavior
itself, not detection.

## What red-team round 2 should attack first

1. **071 veto starvation (class 4).** 469 refusals → `max_steps`
   in a mover cell; 274 in the clean baseline. Where do 071's
   spurious tracks live, and can a wall-born / flicker-born track
   be denied *interior* tube authority (it only ever earned
   wall-adjacent evidence) without losing the corridor fix?
2. **The 2/12 false-dodge rate on fresh seeds.** Both fired once,
   both missions succeeded — try to make one *fail* a mission, and
   instrument whether the triggering track was chain-born or
   channel-born.
3. **Chain-promotion audit.** 117 promoted voxels across the 12
   no-mover seeds means flicker chains complete at a measurable
   rate. Is there a 4th-scan property (persistence of the *same
   voxels* under the track, not just the centroid) that real
   huggers have and chains don't?
4. **131 head-on v0.5 (v2-relative regression).** Detected and
   dodged, still collided — instrument whether the dodge fired
   late, into the mover's line, or was vetoed mid-execution.

## ThermoNav graduation record (2026-10-12)

Ported from thermo-work `muse/roadmap-movers-2026-10-11` @
`30e2746` (the RT-2-corrected doc above travels verbatim; research
code commit `a658150`, results commit `44befad`, RT-2 report
`DYNLayer_REDTEAM2.md` on `muse/roadmap-moversrt2-2026-10-11` @
`8c3929e`, verdict PASS WITH CAVEATS — Phase 2(a) closable pending
graduation). **Claim label unchanged: Z1-legal formulation,
validated in simulation only** (host-side classical layer; no
TSU/Z1 surface, no silicon claim).

**What was ported.** `dynlayer.py` + `test_dynlayer.py` verbatim;
support modules `movers.py`, `motion.py`, `dodge.py` verbatim
(DodgeController unchanged); `DynLayer` wired into
`mission_dynamics_v5.py`'s `_MOTION` slot as the MotionBuffer
successor (cfg `dyn_layer` takes precedence over `motion_layer`,
same `_MOTION` interface; `DodgeController` on top via
`dodge_layer`). Wiring points, same as research: layer update on
the raw scan before Gibbs; dodge `observe` at 1 Hz; motion veto
in the D5 check and in both swept-path checks (`_vel_swept_ok` /
`_traj_passes_d5`, flagged via `_MOTION_VETO_HIT`); motion
refusals counted separately and exempt from the static stuck
budget; movers rendered into the scan GT at 1 Hz and the
collision GT at 10 Hz; mover/motion/dodge/dyn manifest rows.
ThermoNav divergences preserved: DISARM, battery, and nav-health
wiring untouched (decider kernel untouched); the PID cmd channel
still goes through `_pid_issue`; the escalation call still
receives battery/nav inputs. `sim_time` is now emitted as a
manifest row — the RT-2 §7 dead-column fix applied at the source
on this side (research CSVs were not rewritten).

**Not ported (documented, not silent).** (a) The research
`fpwide` / `fresh` sets (12 wide-seed no-mover + 12 red-team
fresh paired cells): the TN validation below covers the FP tax
on the 4 protocol baselines; the wider-seed false-dodge bound
(2/24 clean missions combined with RT-2's extension) remains a
research-side number, cited in SCOPE BOUNDARIES (ii). (b) The
threat-analytic continuous-sphere overlay and the jitter
controller: ThermoNav has no jitter controller (pre-existing,
documented in prior graduations; `map_age` stays 0.0) and the
dynlayer protocol uses voxelized movers, so neither is needed
for this validation. (c) RT-2's lane/hug instrumented probes
(U6, `rt2_lane`) remain research-side evidence; their findings
travel as SCOPE BOUNDARIES (i).

**TN-side validation (executed commit `851908e`, ThermoNav code
commit of this graduation; script `run_dynlayer_sweep.py`,
per-arm shards `dynlayer_sweep_off.csv` / `dynlayer_sweep_on.csv`
merged into `dynlayer_sweep.csv`, 104 rows, provenance verified
with `exec_provenance` begin/finish + `verify_csv_commit`).**
The FULL 48-cell paired set ran on ThermoNav — no subset was
needed. Staging trajectories are ThermoNav-native
(`sim/traj_fix/dyn_baseline_*.json`, flown by this loop, 4/4
success, committed as staging inputs) because the TN loop has
diverged from the research loop; despite the divergence the
results match the research sweep **exactly**:

- OFF (no layer): **43/48 collisions** — the blindness pattern
  reproduces on TN's loop.
- ON (dynlayer + dodge): **7/48 collisions; 36 fixed / 0 broken**
  vs OFF. Breakdowns identical to research: crossing 1/16,
  head-on 4/16, overtake 2/16; by speed 2/1/1/3 of 12; by room
  004 0/12, 071 0/12, 128 4/12, 131 3/12.
- **0 mismatched rows** vs the research `dynlayer_sweep.csv`
  across 104 rows on 18 compared fields (outcome, collision,
  mover_min_dist, move/motion refusals, motion events/confirmed,
  all dyn counters, dodges/fallbacks/wall incidents/motion
  blocks/substeps). Mean appearances 35.2 (15.3 prev + 15.8
  map); 22 dodges, 1 fallback, 0 wall incidents; ON successes
  41/48.
- The RT-2 relabeled cell reproduces: 071 head-on v1.0 ON =
  `max_steps`, 469 motion refusals, min-dist 0.98 m (FP-driven
  starvation with incidental closest approach — not an
  avoidance). ON baseline tax identical to research: 41 (004),
  274 (071), 1 (128), 0 (131); all 4 ON baselines succeed,
  0 dodges.
- `sim_time_s` is live in every TN row (min 8.75 s) — contrast
  the research dead column (−1 in all 152 rows, RT-2 §7).

**Unit + regression (all on the wired tree).** `test_dynlayer.py`:
11/11 PASS. Existing suites unregressed: `test_escalation.py`
ALL PASS (incl. DISARM live-mission + stale-LAND hazard);
`test_link_health.py` 25/25; `test_battery.py` +
`test_battery_safety.py` ALL PASS; `test_nav_health.py` ALL
PASS; `test_exec_provenance.py` ALL PASS;
`test_nav_checkpoint.py` ALL PASS. Battery safety validation
re-run at `851908e`: **6/6 gates PASS, 0 collisions**, every
behavioral field identical to the graduation record (RTB landed
0.0693; whiteout `landed_disarmed` 0.0675 with 67 countdown
vetoes; RTC landed 0.0142 / 0.0134); its CSVs are re-stamped
`851908e` (calibration: 356/356 differing rows are the commit
stamp only; validation: stamp + wall-times, plus `sim_time_s`
now live instead of −1 from the same source fix above).

**Caveats carried into ThermoNav.** The SCOPE BOUNDARIES above
travel with this layer, verbatim in substance: (i) open-room
wall-lane huggers remain blind (parked card
`dynlayer-wall-lane-blindness.md`); idealized sub-band huggers
(≤0.35 m/s) blind at unit level, corridor huggers at 0.2 m/s
detected via voxel churn; (ii) tiered authority's bound is 3
consistent paints → a full-authority phantom exists (false
dodges 2/24 clean missions combined, both chain-born phantoms
marching 1.1–1.5 m/s along linear features; voxel-persistence is
the open discriminator, parked); (iii) 131 head-on v0.5 is
dodge-geometry + no-retreat (dodge moved the drone into the
mover's line, veto pinned it; tiering exonerated; a track now
exists so the parked retreat card's premise is satisfiable).
The README's "blind to walking-speed movers" line is superseded
by this graduation for the staged protocol — the residual
blindness is the scope boundaries, not the original 43/48.
