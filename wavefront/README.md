# wavefront/ — the substrate wavefront planner

The planner half of the ThermoNav brain. A navigation problem is written
onto an Ising system as local fields and couplings; a goal cell is pinned;
Gibbs sampling starts; and the region of flipped spins **grows outward
from the goal** through the ferromagnetic couplings. The per-cell time at
which its spin first flips is a geodesic distance field — the wavefront —
and the flight controller descends it. The Gibbs dynamics are the
computation: not the equilibrium magnetization, and not a host-side
shortest-path search. (An honest scoping of what that does and does not
mean is in `docs/ARCHITECTURE.md` — the settled field's values agree with
a classical distance transform to correlation 0.98–0.999, and this
repository does not claim otherwise.)

## Files

- `sim_2n_wave.py` — the experiment: 2D bar world + three 3D ceilinged
  mazes (40 × 40 × 16, roof at z = 15, gaps forced across ≥ 2 levels),
  plus the J = 0 ablation. Real THRML `IsingEBM` + `sample_states`;
  zero training. Two curation patches (marked in the file) replace
  lab-internal paths with this directory; the sampling, field
  construction, smoothing, and routing code is otherwise the research
  original.
- `maze_gen.py` — the maze generator, vendored verbatim (two functions)
  from the research script `sim_2m.py`, sha256
  `ca64dc5670e8a3cf62a8496455963e8af21733ae03533cf17611560bd0744f34`.
- `analyze_rerun.py` — independent analysis: recomputes every headline
  number from the saved fields alone, never from live sampling.
- `data/` — the saved fields and results of the 2026-10-22 verifiability
  re-run: `bar1_2d.npz`, `bar4_j0.npz` (the ablation), `maze{0,1,2}_3d.npz`,
  `results_rerun.json`, `analysis_rerun.json`, `summary_rerun.csv`.
  Every npz carries its program hash and the exact library versions
  that produced it.

## Measured results (simulation, THRML 0.1.4, CPU)

- **2D headline:** field vs true BFS distance, correlation
  **0.9991180978469811** through the published pipeline — and **0.9982**
  on the raw field with no host smoothing at all. The smoothing step is
  worth +0.0009; the field is the sampler's.
- **3D mazes:** 0.980 / 0.984 / 0.994 (mean field, smoothed) across the
  three mazes; gradient-following reached the goal in all three
  (65 / 65 / 83 steps). 31,752 spins per maze.
- **Ablation:** with the coupling switched off (J = 0) the field is
  structureless noise (correlation 0.09) and routing fails. The coupled
  dynamics are load-bearing; this is the control that proves it.
- Full record: `docs/RESULTS.md`. The fields were re-analyzed from the
  shipped npz files during this repository's curation (2026-10-23):
  summary CSV identical, analysis JSON agrees to floating-point dust.

## Run it

```bash
pip install -r ../requirements.txt   # numpy, jax[cpu], thrml, ...
python sim_2n_wave.py               # writes fields + results to ./runs/
python analyze_rerun.py data data/results_rerun.json ./check --commit any
```

The original run took minutes per case on a desktop CPU (Ryzen
9 9800X3D under WSL2, one core class, nice 19 — wall-clock figures in
the record are that machine's, and measure the *simulator*, never a
chip). Sampling is seeded; the 2D headline reproduced bit-identically
across a JAX version bump (0.9.1) in the re-run.

## Limits (also in `docs/OPEN_PROBLEMS.md`)

- **Static maps only.** The map is written into the fields before
  sampling; how fast the field heals when the map changes mid-flight is
  an open measurement in the lab, not a result.
- `max|h| = 6.0` sits **at** the assumed Z1 rail — zero headroom,
  flagged since the original run.
- The field is a per-mission program: a new goal means a new field
  write and a fresh relaxation, priced in sweeps, not in reuse.
