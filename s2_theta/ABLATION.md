# S2-θ substrate ablation (2026-10-26): the trained-parameters claim, tested and withdrawn

**Claim label:** validated in simulation. [SIM] throughout; nothing in
this file states or implies a silicon, placement, energy, or
real-aircraft claim.

## Why this file exists

This record is imported into this repository on 2026-10-28 from the
lab's research record, so that the claim correction below stands alone
here — the same pattern as the `sim/SPECULAR.md` import (2026-10-26).
Source: `RESULTS_KERNEL_ABLATION_BUILD2.md` (thermo-work @ `1105747`),
executed against charter `PLAN_KERNEL_ABLATION.md` (lab research record;
not present in the archived thermo-work branches) as conformed by
errata #1–#3; cold results review **PASS** (@ `965bdec`). The runs were
made from a checkout of this repository at `bd36881`, with every input
hash-pinned before scoring.

## The question

The shipped record credited **the trained static parameters of the
shipped S2 kernel as load-bearing for its decisions**. The ablation
tested exactly that claim, as filed, under the shipped encoder and the
deterministic argmax readout: destroy the trained structure — holding
scale, wiring, and protocol fixed — and see what the performance does.

## Arms

- **T** — the trained kernel as shipped (`s2_theta_v3.npz`).
- **A** — couplings shuffled across edges (same values, wrong places).
- **B** — biases shuffled across nodes.
- **C** — untrained Gaussian parameters, moments matched to T.
- **D** — couplings zeroed (biases + encoder fields only).

Controls were generated once, from T, with edges bit-identical across
arms, and the generation manifest was filed before any scoring.

## Protocol

- **Metric (i):** held-out decision agreement + mean decision margin,
  shipped `eval_v2` protocol verbatim — first 400 of the 1,537 held-out
  windows (rooms 185/128), 200 sweeps, burn-in 50, seed 123+*i*.
- **Metric (ii):** closed loop — the shipped canonical-8 flight gate
  (`flight_test_v3.py` pattern), per-mission outcomes filed.
- Each arm scored **once**, in the registered order T, A, B, C, D, under
  bars and outcome cells pre-registered before the run.

## Results [measured]

| Arm | Agreement (i) | Mean margin (i) | Closed loop (ii) |
|---|---|---|---|
| **T** (trained v3) | **41.25%** (165/400) | 0.0744 | 8/8, 0 collisions |
| A (couplings shuffled) | 21.0% (84/400) | 0.2456 | 8/8, 0 collisions |
| B (biases shuffled) | 21.0% (84/400) | 0.1972 | 8/8, 0 collisions |
| C (untrained, matched) | 0.5% (2/400) | 0.2217 | 8/8, 0 collisions |
| D (couplings zeroed) | 0.0% (0/400) | 0.0013 | 8/8, 0 collisions |

## Bars

- **B-1 (agreement separation):** T − best control = 41.25 − 21.0 =
  **+20.25 pts ≥ 10.0 → PASS.**
- **B-2 (margin separation):** T − best control = 0.0744 − 0.2456 =
  **−0.1712 < 0.05 → FAIL.** Three controls post *higher* mean decision
  margins than T while agreeing far less — and A and C also fail to
  decide at all on 110 and 385 of the 400 windows (the margin is
  computed on every window). The margin metric points the wrong way.
- **B-3 (closed-loop separation):** every arm flies 8/8 with
  0 collisions, so T − best control = **0 missions < 2 → FAIL** (T's
  zero-collision clause holds; the separation clause does not).

## Verdict

The governing outcome cell, verbatim from the charter's outcome table:

> "the trained-parameters claim is WITHDRAWN as filed — a measured
> negative, reported whole."

Scoped exactly as the charter scopes it: the claim that the trained
static parameters of the shipped S2 kernel are load-bearing for its
decisions, [SIM], under its shipped encoder and deterministic argmax
readout, on the filed evaluation sets. Nothing broader is claimed or
withdrawn: not placement, not silicon, not any other component.

## What survives

**Decision-level only.** The trained kernel's decisions agree with the
teacher at 41.25%, where destroyed-structure controls agree at 21.0% or
less — and the zeroed-couplings arm decides nothing at all (400/400
no-decision). Destroying the trained structure collapses decision
agreement; that is the load-bearing evidence, and it lives at the level
of *which decisions come out*. It does not live at the level of
parameter values (the sign/magnitude readings in `S2_THETA.md` are
measurements of the fitted values, not the claim), not in decision
margins (B-2 reversed), and not in mission outcomes on the shipped
gate, where nothing separated any arm from any other (B-3).

## What this does not say

- It does not say the kernel doesn't fly: T's flight record stands —
  and so, on this gate, does every control's. That the canonical-8 gate
  separates nothing is a finding about the gate, filed as such.
- It does not touch the wavefront planner or the safety stack.
- The fallback rate of every arm is **UNMEASURED** under this charter;
  no S2-decision or fallback count is filed anywhere in the unit.

## Where the correction is applied in this repository

Dated correction notes at the point of claim, originals preserved:
`S2_THETA.md` (claim label; "What this proves"; the round-2 headline
diagnostic), `GENERALIZATION.md` (honest notes), `REDTEAM.md` (header
label; S2-θ rows), `docs/RESULTS.md` (flight-kernel table),
`docs/ARCHITECTURE.md`, `README.md`.
