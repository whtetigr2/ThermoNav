# Red-Team Record

We kill our own lines. Every experiment below either shipped or died honestly —
the negatives are published with the same care as the wins, because a failure
mode you document is a failure mode you understand.

**Claim label (all): Z1-legal formulation, validated in simulation.**
Nothing here ran on Z1 silicon or a real aircraft. Every number carries its
provenance (measured / calculated / assumed).

---

## Shipped

| Line | Verdict | Key numbers |
|---|---|---|
| S2-θ v2 kernel | **SHIPS** | 123 params, degree 15, max\|h\|=5.35, max\|J\|=1.14 [measured]. Tight geometry 16/16, noise 0.30 16/16, heavy fog 11/13 (fails safe) — 96 paired flights [measured] |
| Deterministic argmax readout | **SHIPS** | 22/22 vs stochastic 0/9 paired [measured] — [`MULTIMODAL.md`](redteam/MULTIMODAL.md) |
| D5 freshness gate (FIX 4) | **SHIPS** | Fog+jitter collision cell 0 collisions (was 1); whiteout+jitter 0 (was 2); 28 clear-air missions zero false fires [measured] — [`FRESHNESS_GATE.md`](redteam/FRESHNESS_GATE.md) |

## Killed (honest negatives)

| Line | Verdict | Why it died |
|---|---|---|
| S2-θ v4 retrain (heavy-fog targeted) | **KILLED** | Bought nothing: worse teacher agreement (56.9% vs 67.9%), max\|h\| pinned at the 6.00 rail vs v2's 5.35, no outcome gap on fresh heavy rooms (9/9 all three) [measured]. v2 stays. |
| Stochastic S2 action sampling | **KILLED** | 0/9 — dithered to max_steps every time vs argmax 22/22 paired [measured]. Stochasticity stays inside Gibbs/posterior formation; the readout is deterministic. |
| Evidence-θ round 1 (per-ray scalars) | **KILLED** | Held-out beat engineered (4.95 vs 5.41) but whiteout closed-loop crashed 6× vs engineered 0 [measured]. Training never saw transients. |
| Evidence-θ round 2 (temporal) | **KILLED** | Held-out PASSED (7.33 vs 7.44) but 1 whiteout collision on bar-2, and 4 whiteout+jitter collisions where engineered stood down [measured]. Heavy fog genuinely improved (15/16 vs 13/16) but whiteout killed it. Mechanism: stronger provisional paint (0.66 vs 0.30) → overconfidence. Lesson: supervised loss can't see closed-loop safety. |

## Measured envelopes and boundaries

- **Compute-jitter envelope** — 182 missions: clear air holds to 5 s single delays, bursts, periodic misses; ≈0.5 Hz perception fails safe (wanders, never collides); heavy fog + severe jitter produced 1 honest collision (replicated, not a flake) — the corner the freshness gate now closes. Validated at noise 0.15 throughout [measured] — [`JITTER.md`](redteam/JITTER.md)
- **Jitter in the real world** — what actually causes multi-second stalls (Python GC pauses measured to 2.2 s; thermal throttling; sensor dropout) and how hobbyist / mil-spec / racing / legged platforms survive it (fast inner loop never stalls; freshness-gated outer loop). Web research, gaps marked MISSING — [`JITTER_REAL_WORLD.md`](redteam/JITTER_REAL_WORLD.md)
- **Known limitation: heavy fog** — learned 11/13 vs teacher 13/16; both fail safe (goal_blocked), zero collisions [measured]
- **Known limitation: whiteout** — safe but not operational (0/8 flights, 8/8 safely blocked). Needs another modality (radar), not more training.

## Method

Paired conditions and seeds throughout. Same starts/goals, same sensor noise,
no ground-truth leakage. Behavioral rejection beats test counts: a line ships
only if it wins where it matters (closed-loop flight), not where it's easy
(held-out loss).
