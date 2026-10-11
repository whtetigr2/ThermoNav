# Escalation Policy Build — Option A (industry mirror)

**Branch:** `muse/escalation-policy-2026-10-10` (rebased onto `muse/tree-recon-phase2-2026-10-10`). **Status:** D5 INTEGRATION WIRED (2026-10-10); mission-level validation running.
**Claim label:** "Z1-legal formulation, validated in simulation" (decider kernel). The policy module is host-side classical — no TSU claim.

## What was built

**1. `escalation.py` — the Option A policy (host-side, classical, numpy-only).**
Trigger table (every threshold carries provenance):

| Trigger | Threshold | Action | Provenance |
|---|---|---|---|
| Perception degraded (vis<0.70 AND far<0.10) | sustained 5 s | Hold (hover) | [design] (Option A) |
| Degraded AND map_age ≥ 1.0 s | immediate | Hold (hover) | [measured] (ships as FIX 4) |
| Degradation sustained | 30 s, no recovery | RTH | [design] (Option A) |
| RTH requested, pose gate closed | at decision time | Land in place | [design] + measured estimator gap |
| Battery / link / nav-health | — | STUB (MISSING) | unmodeled in sim |

Industry rules implemented: **hold-first** (RTH/land/disarm wait out a 10 s operator-override window at hover level; hover itself engages immediately — the shipped FIX 3/4 behavior is never delayed) and **most-severe-wins** (severity: continue < hover < RTH < land < disarm).

**2. The RTH gate (documented, not silent).** `pose_trustworthy` defaults False. RTH requested with the gate closed degrades to land-in-place with an explicit log entry (`rth_gated_land_in_place`, reason: estimator open problem, no <0.5 m/mission estimator exists). Paul picked A knowing this caveat; the implementation makes the gating visible in the mission log, not silent.

**3. `escalation_decider.py` — 12-spin Z1-legal decider kernel.**
4 actions × 3 spins (redundancy mirrors S2-θ's 6×3). Hand-designed h/J (teacher-style). Inputs {vis_deg, far_deg, stale, jitter} + timer-derived {deg_5s, deg_30s} + policy gate {rth_available} → {continue, hover, RTH, land}. Severity ordering lives in field strengths (0.5/2.0/3.0/4.0); most-severe-wins guaranteed at the deterministic commit step (severity-ordered argmax over firing groups).

**Z1-legal stats** (measured): 12 spins (10–14 budget) · max|h|=4.0 · max|J|=1.0 · degree 2 · 66→12 edges. All rails pass.

**Structural finding during build:** pairwise antiferro between groups CANNOT stabilize one-hot with 4 groups — the pair combinatorics favor 2-active/2-inactive splits (verified analytically; the first prototype failed exactly this way on the grid test). Between-group J set to 0; one-hot enforced at commit. Documented in the module.

**Spin-budget expressiveness (honest scoping):** the 12-spin budget expresses the 4 flight actions {continue, hover, RTH, land}. Warning is a host-side notification level (policy log), not a flight action; Disarm is terminal mission-abort (outcome `stand_down`, matching the whiteout convention), executed host-side. Nothing silently dropped.

## Tests (all pass)

- **Decider grid:** 128/128 input combos match the policy-table oracle, seed-stable across 3 seeds (`escalation_grid.csv`, 128 rows, commit per row).
- **Z1 rails:** asserted in-test.
- **RTH gate:** closed → land-in-place + log entry; open → RTH.
- **Hold-first:** window respected at hover; operator override cancels.
- **Most-severe-wins:** all rows firing → land (RTH gated).
- **MISSING stubs:** battery/link/nav-health inert, provenance MISSING.
- **Regression (standalone):** on the shipped input envelope the decider never contradicts FIX 3/FIX 4 (clear air → continue; FIX 4 condition → hover; jitter alone → continue). Full mission-level regression (20-mission close-out set) runs after the Phase 2 rebase, when D5 hooks are wired — the integration contract in `escalation.py` guarantees it structurally (final action = max(shipped, escalation) by severity).

## Integration — WIRED (2026-10-10, post-Phase-2 rebase)

Wired in `mission_dynamics_v5.py` per the contract (all 5 points):

1. Call site: per outer step, after `_TRACKER.update_stale()`; inputs `_TRACKER.vis_smooth` / `far_smooth`, `map_age` (from `_jit`), jitter-active flag, dt=1.0 s.
2. `pose_trustworthy=False` hardcoded (estimator: open problem).
3. Final action = max(shipped, escalation) by severity (`_esc_action`).
4. Behavior mapping in the 10 Hz loop: DISARM → `disarm_behavior` (terminal `stand_down`); LAND → velocity-servo descent at 0.5 m/s with the same 100 Hz collision checks as hover, outcome `landed` on touchdown; HOVER → shipped hover branch (reused); RTH unreachable (gate) with a loud fail-safe fallback. With escalation off, `_esc_action == HOVER iff hover_mode` — the branch is bit-identical to shipped.
5. Manifest: `esc_enabled`, `esc_final_action`, `esc_engaged_scans`, `esc_n_events`, `esc_rth_gated` per mission row.

**Two integration bugs found and fixed during wiring** (not gate/decider logic — both were behavior-primitive arg-order / control mistakes):
- `pid.command(p_des, v_des, p, v)` arg order: the first LAND attempt passed the descending target as `p` (current), which the PID read as "I am below target" and climbed. Fixed in both the loop and `land_in_place_behavior`.
- A creeping position target descends ~100x too slow (P term sees 0.005 m error/step). LAND now uses the velocity servo (`p_des=pos`, `v_des=(0,0,-0.5)`), matching the loop's existing servo convention. Verified: 2 m → touchdown in ~4 s.

**Safety finding during validation (measured 2026-10-10): blind LAND collides.** The first integrated validation run (20-mission close-out, escalation ON) produced **4/16 collisions** in the whiteout cell — all during LAND descent into ground-truth-occupied voxels directly below the hover point (e.g. indoorcd_132_s80110: hover at z=5.0, obstacle at z=4). The industry "land" rung assumes terrain awareness; our whiteout map can't verify the ground. Fix (actuation guard, not policy change): `_below_clear()` in `mission_dynamics_v5.py` — LAND descends only if the vertical column below is observed AND P(occ) < P_OBSTACLE (0.3) in the current perceived map; otherwise the step degrades to HOVER, logged as `esc_land_vetoed`. The policy (red-team PASS) still says "land"; the loop lands safely or holds. Re-validation running.

Validation: `run_escalation_validation.py` — the 20-mission R1 close-out set, run twice (escalation OFF for parity, ON for integration). Results in `escalation_integration_validation.csv`.

## Open / pending

- Red-team the integrated build (parent spawns after this report).
- Battery/link/nav-health: MISSING until power + comms models exist (unscoped).
- The 5 s / 30 s / 10 s thresholds are [design] — need validation runs before they ship as more than the LEGO.
