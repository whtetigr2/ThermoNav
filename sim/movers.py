"""Dynamic obstacles for the close-out sweep: slow walking-speed movers.

A Mover is a person-scale sphere traveling at constant velocity on a
straight line. The experiment harness renders movers into a per-step copy
of the GT occupancy grid BEFORE the 1 Hz lidar scan and the DDA collision
check, so the stack sees the mover only through sensor evidence — it never
receives the mover's velocity vector (no ground-truth leakage).

Fast incoming threats are DEFERRED to the radar layer (Paul's call);
mover speeds here are capped at 2.0 m/s.

Claim label: Z1-legal formulation, validated in simulation (movers are
host-side classical; no TSU claim).
"""

import math

import numpy as np

MOVER_RADIUS = 0.5  # m: person-scale sphere [assumed]
# Threat-envelope study (branch muse/threat-envelope-2026-10-10, Paul's
# counter-UAS scenario): cap raised from 2.0 to 20.0 m/s. The old cap was
# a task-scoping decision ("fast threats deferred to radar"), not physics.
MAX_MOVER_SPEED = 20.0  # m/s: study cap for the threat-envelope sweep


class Movers:
    """A set of constant-velocity spherical movers.

    specs: list of dicts with keys:
      p0: (x, y, z) position at t=0 [m, continuous]
      v:  (vx, vy, vz) velocity [m/s]
      r:  sphere radius [m] (default MOVER_RADIUS)
      geometry: label string, e.g. "crossing" (carried into the CSV)
    """

    def __init__(self, specs):
        self.specs = []
        for s in specs or []:
            sp = float(np.linalg.norm(s["v"]))
            if sp > MAX_MOVER_SPEED + 1e-9:
                raise ValueError(
                    f"mover speed {sp:.2f} exceeds cap {MAX_MOVER_SPEED} "
                    "(fast threats deferred to radar)")
            self.specs.append({
                "p0": np.asarray(s["p0"], dtype=float),
                "v": np.asarray(s["v"], dtype=float),
                "r": float(s.get("r", MOVER_RADIUS)),
                "geometry": str(s.get("geometry", "")),
                "speed": sp,
            })

    def __bool__(self):
        return bool(self.specs)

    def positions(self, t):
        """(n_movers, 3) continuous positions at sim time t."""
        return np.array([s["p0"] + s["v"] * t for s in self.specs])

    def render(self, occ, t):
        """GT occupancy copy with mover spheres voxelized at time t.

        Voxel centers within r of the mover center are marked occupied.
        Out-of-bounds movers simply render nothing (walked out of view).
        """
        if not self.specs:
            return occ
        out = occ.copy()
        nx, ny, nz = occ.shape
        for s, p in zip(self.specs, self.positions(t)):
            r = s["r"]
            lo = [max(0, int(math.floor(p[i] - r))) for i in range(3)]
            hi = [min((nx, ny, nz)[i] - 1, int(math.ceil(p[i] + r)))
                  for i in range(3)]
            for ix in range(lo[0], hi[0] + 1):
                for iy in range(lo[1], hi[1] + 1):
                    for iz in range(lo[2], hi[2] + 1):
                        c = np.array([ix + 0.5, iy + 0.5, iz + 0.5])
                        if float(np.linalg.norm(c - p)) <= r:
                            out[ix, iy, iz] = True
        return out

    def min_dist(self, pos, t):
        """Min over movers of ||pos - mover_pos(t)|| [m]."""
        if not self.specs:
            return float("inf")
        p = np.asarray(pos, dtype=float)
        return float(np.min(np.linalg.norm(self.positions(t) - p, axis=1)))


def stage_encounter(occ, traj_xyz, dt_sample, speed, geometry,
                    frac=0.45):
    """Build one mover spec staged to meet the drone mid-route.

    Uses the BASELINE (no-mover) trajectory so the encounter is choreographed
    from where the drone actually flies — the experimenter staging the test,
    not the stack seeing the future. The stack still only ever sees scan
    evidence.

    occ: static GT occupancy (used only to sanity-check the cross point is
      free space).
    traj_xyz: (N,3) baseline trajectory samples at dt_sample [s].
    speed: mover speed [m/s]. geometry: "crossing" | "head-on" | "overtake".
    Returns a mover spec dict, or None if the trajectory is too short or the
    cross point is not free space.
    """
    traj = np.asarray(traj_xyz, dtype=float)
    if len(traj) < 10:
        return None
    idx = int(len(traj) * frac)
    c = traj[idx]
    # cross point must be free space in the static map (sanity)
    ci = tuple(int(min(max(0, int(c[i])), occ.shape[i] - 1)) for i in range(3))
    if bool(occ[ci]):
        return None
    t_cross = idx * dt_sample
    # local path tangent from central difference (horizontal)
    a = traj[max(0, idx - 10)]
    b = traj[min(len(traj) - 1, idx + 10)]
    tang = np.array([b[0] - a[0], b[1] - a[1], 0.0])
    n = float(np.linalg.norm(tang))
    if n < 1e-6:
        return None
    tang /= n
    perp = np.array([-tang[1], tang[0], 0.0])
    z = float(c[2])
    if geometry == "crossing":
        v = perp * speed
        p0 = np.array([c[0], c[1], z]) - v * t_cross
    elif geometry == "head-on":
        # walks toward the drone along its path direction
        v = -tang * speed
        p0 = np.array([c[0], c[1], z]) + tang * speed * t_cross
    elif geometry == "overtake":
        # same direction as the drone, from behind; catches up only if
        # faster than the drone's cruise
        v = tang * speed
        p0 = np.array([c[0], c[1], z]) - tang * speed * t_cross
    else:
        raise ValueError(f"unknown geometry {geometry}")
    return {"p0": tuple(float(x) for x in p0),
            "v": tuple(float(x) for x in v),
            "r": MOVER_RADIUS,
            "geometry": geometry,
            "speed": float(speed),
            "cross_point": tuple(float(x) for x in (c[0], c[1], z)),
            "t_cross": float(t_cross)}
