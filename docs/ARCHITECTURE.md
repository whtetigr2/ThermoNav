# Architecture

ThermoNav is a complete drone navigation stack in which every decision
that can be expressed as an energy landscape is expressed that way, and
formulated to run on a thermodynamic sampling substrate of the class
Extropic is building (Z1-generation: pairwise Ising programs, local
fields |h| ≤ 6, couplings |J| ≤ 6, node degree ≤ 16, on the order of
200k sampling elements). Everything in this repository was built and
measured in **simulation** — THRML, Extropic's JAX-based simulator of
that substrate — on ordinary CPUs. Nothing has run on silicon, and
nothing here is a validated aircraft.

We did not build the chip. This work is cartography of its landscape:
which navigation behaviors the substrate supports, which it cannot,
and where it breaks — measured, with the breaks published.

## The brain, layer by layer

```
 sensors ──► occupancy evidence ──► WAVEFRONT PLANNER (substrate)
                                        │  geodesic field
                                        ▼
                              S2-θ FLIGHT CONTROLLER (substrate)
                              18 spins: 6 directions × 3 lookahead
                                        │  one move per decision
                                        ▼
 SAFETY MACHINERY (substrate decider + host-classical policy)
   D5 freshness gate · escalation ladder · battery cascade
   link health · nav health · dynamic-object layer
                                        │
                                        ▼
                                   vehicle dynamics
```

### Wavefront planner (`wavefront/`)

The map becomes local fields (obstacle cells pinned, free cells biased,
goal clamped) and uniform ferromagnetic couplings between neighbors.
Sampling grows a flipped region out of the goal; each cell's first-flip
time is its geodesic distance, and the controller flies downhill. The
field is recomputed per mission by the sampler's dynamics.

**Credit, stated precisely:** the settled field's *values* agree with a
classical BFS/Dijkstra distance transform to correlation 0.98–0.999 —
the planner does not beat classical planning at its own game, and we do
not claim it does. What the substrate formulation buys is that the map,
the plan, and the flight decisions below live in **one physical system
with one programming model**, priced in the substrate's native currency
(sweeps), and that the same machinery that plans also absorbs noise,
holds state, and votes on safety. The non-replaceable parts of this
stack are the stochastic flight controller and the safety/memory
machinery — not the distance field itself.

### S2-θ flight controller (`s2_theta/`)

An 18-spin Ising kernel — 6 move directions × 3 steps of lookahead,
123 fitted parameters total (18 biases + 105 couplings), 492 bytes of
parameters, plus its fixed wiring —
that picks each move by sampling. The parameters were fitted by
imitation (behavior cloning of an engineered teacher via contrastive
KL-gradient estimation in THRML); training is host-side, the fitted
program is what would be flashed. A deterministic argmax readout ships;
stochastic action sampling was measured, and killed (0/9 vs 22/22
paired) — stochasticity lives inside the sampling that forms decisions,
not in the readout. Full record: `s2_theta/S2_THETA.md`.

### Safety machinery (`sim/`)

The part of the stack we trust least to improvisation, and the part
with the deepest test record:

- **D5 freshness gate** — refuses moves when perception is degraded
  *and* the map is stale; closed the one honest fog+jitter collision
  corner in the jitter envelope.
- **Escalation ladder** — an industry-mirror policy (warn → hold →
  return-to-home → land → disarm) with a 12-spin Z1-legal decider
  kernel verified 128/128 against the policy oracle.
- **Battery layer** — a gauge plus a return cascade sized from
  *measured* draw (never the model's predicted draw), with a descent
  reserve and a two-tier touchdown envelope, all stated with its
  plan-time-threshold caveat (`sim/BATTERY_MODEL.md`).
- **Link-health and nav-health monitors** — receiver-side link state
  and estimator-health state built only from observables the vehicle
  itself has; ground truth is never an input (source-audited in
  tests). Nav health exists because of a measured failure: a standard
  estimator reporting 12.5% of true motion while looking like a
  confident hover.
- **Dynamic-object layer** — scan-differencing tracks for moving
  obstacles with tiered veto authority; took the paired mover sweep
  from 43/48 collisions to 7/48, with its remaining blindnesses named
  in `sim/DYNLayer_RESULTS.md`.

### What is deliberately not here

- **No state estimator of our own.** Self-location is the program's
  open problem (see `docs/OPEN_PROBLEMS.md`). The stack consumes pose;
  the perception front-end evaluated alongside it is a standard
  published pipeline (ORB-SLAM3), host-side classical computer vision,
  not a thermodynamic component.
- **No learned perception.** The only fitted parameters in the entire
  stack are S2-θ's 123.
- **THRML itself** is a dependency, not vendored code: it is Extropic's
  library, installed per their instructions (`requirements.txt`).

## The discipline (how any of this can be checked)

Every shipped line passed the same gauntlet: a pre-registered bar
written before the run, paired conditions and seeds against the
incumbent, a red-team review that re-derives the numbers from raw
artifacts, and a published verdict — including the FAILs, which stay in
`REDTEAM.md` next to the wins. Claims carry labels throughout:
**measured** (ran, counted), **calculated** (derived from measured
inputs), **assumed** (a design choice or an unmeasured model). Energy
and performance projections for real hardware, where they appear in
the research record, are calculated — no hardware measurement exists.
