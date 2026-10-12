# Below-clear guard, Option A — validation report (ThermoNav)

Paul decided Option A on 2026-10-12: for non-battery LAND demands
(lost-link landing, whiteout-arrival landing), the below-clear veto
may hold for a bounded window, after which descent proceeds at
creep speed (0.15 m/s) regardless of the veto. This report covers
the ThermoNav graduation of the research build (thermo-work branch
muse/belowclear-2026-10-12 @ 5ef3763; research red-team verdict
PASS WITH CAVEATS — every number the red team reran reproduced
exactly; caveats were disclosure-grade and are carried below).

Claim label for every number below: **validated in simulation
only.** No aircraft, no silicon. The Phase 0 caveat (next section)
attaches to every flag-on price in this report.

## The Phase 0 caveat, attached to every price

Phase 0 (BELOWCLEAR_PHASE0.md, ported alongside this report) found
the record is substantially a modeling artifact in the record
rooms: their ground-truth grids contain no floor (zz=0/1 occupancy
0.0000 in indoorcd_071/004/131, measured), the only "ground" in
the loop is the mapper's own boundary prior, and the sensor fan
never applies vehicle attitude. Descent columns in these rooms are
therefore unobservable by construction, in both of the guard's
non-clear modes: in clear air they are never observed (UNKNOWN at
every decision altitude), and in whiteout the s4 columns were
filled with phantom obscurant evidence and read as
believed-OCCUPIED (state 2 — see the price section; the research
red team established this from the committed Phase-2c diagnostic
and its own instrumented rerun). Read every collision price below
as "columns unobservable by construction," not as a measured
property of a real vehicle over real ground. The modeling debt
(floor geometry in real-room GT grids, tilt-aware fan,
mask-threshold sensitivity) is parked as its own card; it is not
folded into this build.

## What was built (ThermoNav port)

- `sim/mission_dynamics_v5.py`: pure helper `_land_hold_update`
  (episode bookkeeping, unit-testable without a mission) plus a
  call-site integration gated on cfg key `land_hold_window_s`.
  Unset, 0, or negative = OFF (a negative window must not arm
  instant expiry — research red-team catch); when OFF the hold
  state is never read and behavior is legacy bit-for-bit.
- A step that does not demand LAND resets the episode: the window
  bounds a CONTINUOUS hold, not a cumulative budget.
- On expiry the veto lapses for both guard states (1 UNKNOWN and
  2 believed-OCCUPIED) and descent proceeds at LAND_CREEP_RATE
  (0.15 m/s) at all altitudes. This bst==2 choice is Paul's Option
  A rule as written, disclosed here: after the window, a
  believed-occupied column is treated the same as an unknown one.
- The battery terminal bypass and the soc-keyed battery-critical
  relaxation OUTRANK the hold: episodes neither accrue nor expire
  under them. Paul's battery countdown design is untouched.
- New mission-row fields: `land_hold_window_s`,
  `esc_land_hold_episodes`, `esc_land_hold_expired_steps`,
  `esc_land_hold_max_s`.
- ThermoNav has a single below-clear guard call site, so this one
  port covers every LAND demand the composed TN loop produces —
  battery-cascade arrivals and escalation-ladder demands
  (including link-triggered land-in-place) alike. The research
  repo needed a second, separate port for its lost-link branch
  (older boolean guard, native 0.5 m/s rate); no such second site
  exists in ThermoNav.
- Tests: `test_belowclear_hold.py` ALL PASS;
  `test_escalation.py`, `test_battery.py`,
  `test_battery_safety.py`, `test_link_health.py`,
  `test_nav_health.py`, `test_dynlayer.py`,
  `test_exec_provenance.py`, `test_nav_checkpoint.py` ALL PASS.

## Reading the counters precisely

- `esc_land_vetoed` counts steps where the veto actually blocked
  the descent. Per continuous episode, blocked steps are bounded
  by the window plus one step of granularity.
- `esc_land_hold_max_s` is the accrued span of the longest
  episode measured at its last expired step. After expiry the
  vehicle is DESCENDING at creep rate, not holding; if the guard
  keeps returning non-clear during that descent, the episode span
  keeps accruing. So max_s (13.0–17.0 s in the TN cells below,
  window 10 s) is an episode span including expired descent
  steps, not a hold duration. The hold itself never exceeded the
  window in time span; the blocked STEP count can read window+1
  because floating-point t_sim accumulation can leave the
  +window step ~1e-12 s short, pushing expiry one step late —
  observed in research and confirmed by its red team.

## ThermoNav validation — legacy first (flag OFF)

Doctrine order: reproduce legacy behavior before trusting any
flag-on number. TN legacy baselines were derived on unpatched TN
main @ a2cb278 with this same runner (pre-port code: the
`land_hold_window_s` key is ignored there, and the ON arms came
out identical to OFF — determinism + flag-inertness on legacy
code). CSV `belowclear_validation.csv`, commit provenance in
every row; the graduation run below is stamped with the TN code
commit d651823.

| Cell | TN OFF result @ d651823 | Baseline it must match | Match |
|---|---|---|---|
| s4_clears (whiteout arrival, indoorcd_131 s80010, soc 0.215) | landed, 78 steps, vetoed 51, terminal 17, soc 0.1014 | TN baseline @ a2cb278 (= committed research whiteout row) | exact |
| s4_paired_indef (whiteout never lifts) | landed_disarmed, 78 steps, vetoed 51, terminal 17, soc 0.1014 | TN baseline @ a2cb278 | exact |
| rtb_clear (clear-air RTB, indoorcd_071 s1, soc 0.26) | landed, 110 steps, vetoed 28, terminal 18, soc 0.1016 | TN baseline @ a2cb278 (= research base-code reference @ 5827ab7; also TN's committed battery_safety_validation.csv row) | exact |

Recursive composition note (from the research red team, applies
unchanged): the s4/rtb LAND demands arrive via the battery
cascade's own RTB arrival, so the precise statement is: the hold
bounds the veto INSIDE the countdown for cascade-arrival
landings, and the terminal landing at countdown end outranks it —
the same effect Paul's rule intends, reached through the cascade
rather than a non-battery trigger. In the research validation
the purely non-battery trigger was the lost-link cell (next
section).

## ThermoNav validation — flag ON (window 10 s)

| Cell | TN ON result @ d651823 | Collision |
|---|---|---|
| s4_clears | landed, 77 steps, vetoed 10 (was 51), 1 episode, 4 expired steps, max span 13.0 s, terminal 15, soc 0.1029 | 0 |
| s4_paired_indef | landed_disarmed, 76 steps, vetoed 21 (was 51), 2 episodes, 11 expired steps, max span 17.0 s, terminal 14, soc 0.1045 | 0 |
| rtb_clear | landed, 107 steps, vetoed 20 (was 28), 2 episodes, 6 expired steps, max span 15.0 s, terminal 15, soc 0.1061 | 0 |

All six runner gates PASS. TN composition divergences from the
research ON numbers, stated plainly: s4_paired_indef and
rtb_clear match the research ON cells field-for-field except
expired steps in s4_paired_indef (11 vs research 13); s4_clears
composes differently in TN (1 episode / 4 expired steps / vetoed
10, vs research 2 / 7 / 21) — the TN loop has no jitter
controller and its composed flight profile differs, so the veto
episodes segment differently. Headline outcomes agree in every
cell: landed, 0 collisions, vetoes bounded per episode, final soc
in the same band (0.1029–0.1061 TN vs 0.1044–0.1061 research).

Battery safety validation re-run on the wired tree at d651823:
6/6 gates PASS, 0 collisions, behavioral fields identical to the
graduation record (CSVs re-stamped d651823).

## The research link cell, and what ThermoNav does not claim

The research validation carried a fourth cell on a separate
research branch (muse/belowclear-linkcell-2026-10-12), whose loop
carries the lost-link LAND wiring: long_dropout (indoorcd_004 s0)
went from **max_steps, airborne at the 200-step cap, vetoed 170**
to **landed in 57 steps, vetoed 11, 1 episode, 16 expired steps,
0 collisions** with the window armed. That is research-branch
evidence for the purely non-battery trigger.

ThermoNav's ladder reaches the same gated LAND action through the
guard ported here, but **no lost-link mission has been run
end-to-end in ThermoNav**: the mission-level claim "a lost-link
vehicle ends up on the ground" remains undemonstrated in TN, and
the link-health row's known limit stands. What graduates here is
the veto bound at the single guard every TN LAND demand passes
through — with the flag OFF by default, so shipped behavior is
unchanged until a mission arms the window.

## The price, next to the battery comparison column

Battery terminal bypass (Paul's land-regardless at countdown
end, graduated Phase 2): **2 collisions on record** — a tall
column entered at soc 0.1677 and a floor patch at soc 0.1061.
That is the measured price of landing regardless of the veto on
the battery path.

Option A flag-on price in these cells: **0 collisions in the 3/3
TN cells** (and 0 in the research repo's 4/4 cells, including the
link cell), with touchdowns at final soc 0.1029–0.1061. This is a
small cell count, not a rate: one room each, one seed each. The
honest statement is "no collision observed in the validation
cells," with the Phase 0 caveat attached.

Which guard state the expiries actually overrode matters (from
the research red team's instrumented rerun): descending into a
believed-OCCUPIED column after the window is not a hypothetical
— it was the actual mechanism in both research s4 cells (every
pre-terminal non-clear consultation state 2; 66/66 state-2 calls
in the committed Phase-2c diagnostic). The belief was positive
because whiteout evidence filled the column; ground truth was
empty (no floor, no obstacle). In rtb_clear both expiries were
against state 1 (UNKNOWN) only. TN's s4 cells fly the same rooms
and the same whiteout construction, so the same reading applies;
TN ran no separate state instrumentation, and none is claimed.

## What this closes, and what it does not

Closes: the Phase 1 carried limit as built — with the window
armed, a non-battery LAND in ThermoNav can no longer be vetoed
forever; vetoes are bounded per episode by the window, and the
countdown-end terminal landing still outranks the hold.

Does not close: landing-site selection (parked card — Option A
descends in place after the window; it does not search for a
better column), the modeling debt from Phase 0 (parked card),
the mission-level lost-link demonstration in ThermoNav (above),
and any claim about real ground — there is no floor in the
record rooms' ground truth, so all touchdown claims are
simulation-internal.

Untested corner, disclosed by the research red team and carried
for the PI: with the safety layer OFF and battery at/below
CRITICAL over a believed-OCCUPIED column (bst == 2), the hold's
reset condition does not apply (it covers the terminal case and
the bst == 1 critical case), so an armed hold would accrue and
expire there too, bounding the legacy occupied veto at the window
where legacy vetoes indefinitely. That is the Option A rule
applied uniformly, but no validation cell exercises it; it is
named here for the PI rather than priced.

## Provenance

- Research: Phase 0 commits 5cae995, c8e406d; build ec16d8f;
  validation 9e0e1e7; red-team fix round 5ef3763 (branch
  muse/belowclear-2026-10-12, thermo-work). Link port f7c408e
  (branch muse/belowclear-linkcell-2026-10-12).
- ThermoNav: code commit d651823 (branch
  muse/belowclear-grad-2026-10-12); TN legacy baselines derived
  on unpatched main @ a2cb278 with the shipped runner;
  belowclear_validation.csv + battery safety re-run stamped
  d651823.
- Research red-team outcome (2026-10-12): PASS WITH CAVEATS,
  all numbers reproduced exactly; the TN port was verified by
  the OFF-identity gates above plus the full TN regression set.
