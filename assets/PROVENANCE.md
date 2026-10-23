# Figure provenance

Every figure in this directory is rendered from the lab's own saved
simulation data — no generative image model, no invented lines. Each
filament is a descent path numerically traced through a real, saved
wavefront field (the fields ship in `wavefront/data/`), and each trace
set was audited after tracing: no path step ever climbs, and no path
crosses a wall.

## `wavefront_maze2_ember.png` — NAVIGATION

- **Field:** layer z = 2 of `tmean_world` in `wavefront/data/maze2_3d.npz`
  (3D ceilinged maze 2; THRML 0.1.4; field-vs-truth correlation 0.993).
- **Drawn:** 4,747 descent paths through the field on the goal's own
  floor. Color is field value (first-flip time); the bright knot is the
  pinned goal. Chambers the wave reaches late drain to their own local
  knots — in the full 3D field those are the points where the wave
  entered the chamber from another floor.

## `wavefront_maze0_ember.png` — NAVIGATION

- **Field:** layer z = 2 of `tmean_world` in `wavefront/data/maze0_3d.npz`
  (maze 0; correlation 0.980).
- **Drawn:** 4,827 descent paths, same method and audit as above.

## Method

The field is the mean time at which each cell's spin first flipped as
the sampler's wavefront grew outward from the pinned goal, so field
value *is* geodesic distance, measured by the sampler's own dynamics.
Descent paths follow the field downhill from seeded points. Rendering
(ember-on-black) is display only; it adds no structure to the data.
