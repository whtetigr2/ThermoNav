# D5 Freshness Gate (FIX 4) — stale-map stand-down

**Status:** built 2026-10-08, branch `muse/freshness-gate-2026-10-08`.
**Claim:** Z1-legal formulation, validated in simulation.
Host-layer classical gate; no TSU/kernel involvement, no Z1-constraint
implications. All numbers labeled [measured]/[calculated]/[design].

## 1. The problem (measured)

D5's vetoes are only as good as the map they see. The compute-jitter test
(JITTER.md) found the one corner where the stack collides: **heavy fog +
periodic 2 s perception freezes**. The map was BOTH degraded (fog, sparse
confirmation) AND stale (jitter) — D5 approved a move the stale map said
was safe.

- `jit_fog_per_2s_e3_indoorcd_128_s0_n0.15`: collided step 15 inside freeze
  window (14, 16). Replicated 2× — real, not a flake. [measured]
- Evidence-θ round-2 whiteout+jitter: engineered mapper collided 2/16
  (whiteout × isolated 2 s delay). [measured]

The existing FIX 3 low-vis executive keys on *live sensor blindness*
(`vis_smooth`/`far_smooth` from the true scan every step). In the jitter
corner the sensor is FINE — the tracker was never blind (heavy fog measures
vis 0.61 / far 0.04, deliberately above the FIX 3 blindness thresholds
0.50/0.02). The *published* `(marg, obs_mask)` was up to 2 s old. No
blindness trigger can catch a pipeline-staleness failure. That's the gap
this gate closes; the design is in section 2 below.

## 2. The gate (LEGO: smallest change that bites)

Two signals, conjunctive. Either alone is wrong:

- **Staleness alone** would fire on clear-air `died_8` (2 s freezes every
  2nd step: 6/14, **0 collisions** — already fails safe via max_steps
  wandering). Pure false-positive cost, zero safety gain. [measured]
- **Degradation alone** is FIX 3's job (blindness), already built.

**Signal 1 — degraded perception band** (live, from `EvidenceTracker`;
the sensor keeps scanning during a freeze, so these see reality):
`vis_smooth < 0.70 AND far_smooth < 0.10` → latch band; unlatch above
0.75 / 0.15 (hysteresis). Anchors [measured]: moderate 0.86/0.15 (stays
out), heavy 0.61/0.04 (enters). Whiteout 0.39/0.00 also enters — harmless
overlap, FIX 3 fires first there and the behavior (hover) is identical;
`trigger_source` distinguishes. Thresholds [design].

**Signal 2 — pipeline staleness**: `map_age ≥ 1.0 s`, where map_age is
sim-seconds since the last non-frozen publish (tracked in
`FreezeController.publish()`; in a real pipeline this comes from publish
timestamps). Anchor [measured]: the collision freeze window was 2.0 s;
1.0 s is a 2× engineering margin under it [design]. Fires at the *first*
sustained freeze — on the collision mission it triggers at step 8 (first
window [8,10)), pre-empting the step-15 collision with margin.

**Behavior:** `stale_hover` — stop-and-hover reusing the FIX 3 hover
machinery (hold position, keep 100 Hz dynamics + collision check running,
skip S2/D5, mapping continues every step). Direct signal-following: hover
chains through sustained freezes, releases on the first fresh publish.
Separate hover reason (`trigger_source`: none/blindness/staleness/both)
for logging and the corpus.

**What it does NOT touch:** FIX 1/2/3, `planner.py`, Gibbs/THRML path,
the D5-health quiet gate. Config flag `fix_stale` (default off).

## 3. Validation (behavioral)

Sweep: `run_freshness_sweep.py` → `freshness_sweep.csv` (58 missions,
dual-purpose corpus: every lever an explicit column + producing commit
hash). All cells: fixes 1–3 on + `fix_stale`, numpy backend, noise 0.15,
paired (room, seed) with the original findings. Per-cell physics matches
the original: JITTER.md cells run without gravity/attitude-lag (as the
sweep did); the whiteout_jit cell runs the full Phase-3 physics
(gravity_envelope + attitude_lag, tw_ratio 2.0) of the evidence-theta
round-2 cell it replicates.

| Cell | n | Collisions | Outcomes | Gate fired? |
|---|---|---|---|---|
| `fog_per_2s_e3` (JITTER.md collision cell) | 14 | **0** (was 1) | 6 success, 1 goal_blocked, 7 max_steps | yes, all 14 (10 both, 4 staleness-only) |
| `whiteout_jit` (evidence-θ round-2 engineered set) | 16 | **0** (was 2) | 15 blocked, 1 pre-existing error | yes, 15/15 (5 staleness-only, 10 both) |
| `clear_per_2s_e3` (regression) | 14 | 0 | 14/14 success — identical to JITTER.md | **never** (14/14 `trigger_source=none`) |
| `clear_died_8` (regression) | 14 | 0 | 6 success, 8 max_steps — identical to JITTER.md | **never** (14/14 none), even with map_age reaching 192 s |

Key behavioral reads [all measured]:

1. **The JITTER.md collision is gone.** `indoorcd_128_s0` (collided step 15
   inside freeze window 14–16): with the gate, the mission hovers through
   the freezes (128 stale-hover scans) and ends max_steps — safe
   stand-down, zero collisions across the full 14-mission cell.
2. **Tonight's 2 whiteout+jitter collisions are gone.** `indoorcd_184_s80611`
   and `indoorcd_185_s80710` (collided steps 30/23): both now block safely.
   The gate fires on the 2 s freeze (2 stale-hover scans each) and the
   missions stand down instead of flying the degraded post-freeze map.
3. **The conjunction holds — no clear-air false positives.** The gate
   never fired on 28 clear-air jitter missions, including `died_8` where
   the map went 192 s stale. Staleness without degradation is correctly
   ignored (that cell already fails safe).
4. **Cost is bounded and safe-direction.** `fog_per_2s_e3` success went
   9/14 → 6/14; the 3 missions became max_steps stand-downs, not
   collisions. The gate trades operational success for safety, never the
   reverse.
5. **The whiteout `IndexError`** (`indoorcd_173_s80510`) is pre-existing —
   identical mission, identical error in the evidence-θ round-2 run without
   the gate. Unrelated to this change; filed as a separate bug.

## 4. Red-team notes

- **Does the gate bite?** Yes — 40–128 stale-hover scans per fog mission;
   fires at the first freeze window (step 8), pre-empting the step-15
   collision with margin. [measured]
- **False positives?** None on 28 clear-air missions. An earlier
   vis/far-only band version DID false-fire on 5/28 clear missions
   (motion dips vis to 0.69, tight rooms dip far to 0.045 [measured]) —
   killed in testing; the skip-rate conjunct (clear 0.005 vs fog 0.26+)
   fixed it. The false-positive version never shipped. [measured]
- **Whiteout overlap:** FIX 3 (blindness) and FIX 4 both latch in whiteout;
   behavior identical (hover); `trigger_source` distinguishes. On the
   replicated cell FIX 4 fired standalone on 5 missions (`staleness` only)
   — the gate protects even where FIX 3's D5-quiet precondition isn't met.
- **Hover-forever:** chained stale-hovers end in max_steps/blocked — the
   safe outcome the fog baseline already produces. Escalation beyond hover
   (RTH/land) is NOT built; open question for Paul.
- **Sub-step granularity:** map_age is tracked per outer step (1 s
   resolution). A 0.5 s freeze reports age 1.0 at the next step → hovers
   one step. Conservative direction, bounded cost. [design]
- **What the gate does NOT fix:** collisions on a *fresh* but degraded
   map (e.g., late-whiteout navigation failures unrelated to staleness).
   The gate is a staleness guard, not a general fog fix.

## 5. Verdict

**SHIPS.** The gate bites exactly where the measured collision lived
(fog + stale map), never fires where the stack already fails safe
(clear air), and converts both known collision corners into safe
stand-downs. Cost is 3 fewer fog successes traded for zero collisions —
the safe direction. One honest boundary: it guards staleness, not
degraded-but-fresh maps.

**Claim:** Z1-legal formulation, validated in simulation. Host-layer
classical gate; no TSU/kernel/Z1-silicon claims.
