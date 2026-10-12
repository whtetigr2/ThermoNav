"""Lidar scan-differencing motion layer (LEGO, 2026-10-10).

The static occupancy map asks "is this voxel occupied?" — this layer asks a
different question: "did this voxel CHANGE?" A scan hit landing in a voxel
the static map believes is deeply free is treated as a motion event:
something moved into previously-mapped free space. Events live in a
short-lived buffer the static map is not allowed to carve away, and D5
treats fresh events near the planned path as forbidden.

Why this targets the measured failure (branch
muse/dynamic-obstacles-2026-10-10, red-teamed PASS): the mover paints one
voxel per scan at +0.60 log-odds while the corridor sits at -3.40 from
multi-ray carving, so the evidence never accumulates and Gibbs erases the
isolated hit. Scan-differencing sidesteps the carving asymmetry entirely:
it detects the CHANGE (free -> hit), not the absolute occupancy.

Ego-motion discrimination: static geometry voxelizes to the same world
voxels regardless of drone pose (each scan is voxelized in its own pose
frame), so ego-motion does not shift the static world in voxel space.
Occlusion changes from ego-motion reveal previously-UNKNOWN voxels, not
deeply-carved-free ones — the FREE_THRESH requirement filters them.
Range-noise hits that land one voxel short of a wall (into carved free
space) are filtered by the isolation rule: a real mover in open space is
surrounded by free voxels; a noise-flipped wall hit neighbors an occupied
voxel. If the false-positive rate on no-mover baselines is still high,
that is the finding — do not add a tracker, report it.

No ground-truth leakage: inputs are scans + logodds + sim time only.
Mover truth is never consulted.

Claim label: Z1-legal formulation, validated in simulation (host-side
classical component; no TSU claim).
"""

import numpy as np

FREE_THRESH = -1.0   # log-odds below this = observed free (measured mover
                     # voxels sit at -3.40; walls at +3.60; unknown at 0.0)
OCC_LO = 1.0         # log-odds above this in any neighbor = adjacent to a
                     # real obstacle -> suppress (noise-flipped wall hit)
DECAY_S = 2.0        # motion events expire after this many sim-seconds.
                     # Short on purpose: a false event must not blockade a
                     # corridor. A real mover refreshes its trail every scan.
BLOCK_RADIUS = 1     # Chebyshev radius around a fresh event treated as
                     # forbidden. 0 was too tight: the mover leaves the voxel
                     # before the 10 Hz decision loop acts, so the planned
                     # path never intersects the exact event voxel. 1 gives
                     # the trail a 3-wide tube the flight path can hit.


class MotionBuffer:
    """Short-lived store of free->hit change events from scan differencing.

    An event becomes CONFIRMED (actionable for D5 vetoes) only when a
    second event lands nearby in a recent scan — a real mover paints a
    trail of events as it walks; range-noise artifacts are one-offs.
    Confirmation costs one scan (1 s) of latency.
    """

    TRAIL_RADIUS = 2   # voxels: partner event must be within this
    TRAIL_DT = 2.0     # s: partner event must be this recent

    def __init__(self, shape, free_thresh=FREE_THRESH, occ_lo=OCC_LO,
                 decay_s=DECAY_S, block_radius=BLOCK_RADIUS):
        self.shape = tuple(int(s) for s in shape)
        self.free_thresh = free_thresh
        self.occ_lo = occ_lo
        self.decay_s = decay_s
        self.block_radius = block_radius
        self.events = {}  # (ix, iy, iz) -> event time [sim s]
        self.confirmed = set()  # voxels with a trail partner (actionable)
        self.history = []  # (t, voxel) for trail confirmation
        self.t_now = 0.0
        self.n_events_total = 0  # lifetime counter (diagnostic only)
        self.n_confirmed_total = 0

    def update(self, scans, t, logodds):
        """Ingest one 1 Hz scan; flag deeply-free voxels that got hit."""
        self.t_now = float(t)
        nx, ny, nz = self.shape
        for tup in scans:
            if len(tup) == 5:
                origin, direction, meas_r, sensor_hit, has_return = tup
            else:  # 4-tuple production format: every ray has a return
                origin, direction, meas_r, _true_r = tup
                sensor_hit, has_return = True, True
            if not (sensor_hit and has_return):
                continue
            o = np.asarray(origin, dtype=float)
            d = np.asarray(direction, dtype=float)
            p = o + d * float(meas_r)
            v = (int(p[0]), int(p[1]), int(p[2]))
            if not (0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz):
                continue
            if logodds[v] >= self.free_thresh:
                continue  # wall / unknown / already suspicious: not our signal
            if self._adjacent_occupied(v, logodds):
                continue  # noise-flipped wall hit, not a mover in open space
            if v not in self.events:
                self.n_events_total += 1
                # Trail confirmation: a partner event in a previous scan,
                # nearby in space and time, marks this as real motion.
                for (ht, hv) in self.history:
                    if ht < self.t_now and self.t_now - ht <= self.TRAIL_DT \
                            and abs(hv[0] - v[0]) <= self.TRAIL_RADIUS \
                            and abs(hv[1] - v[1]) <= self.TRAIL_RADIUS \
                            and abs(hv[2] - v[2]) <= self.TRAIL_RADIUS:
                        if v not in self.confirmed:
                            self.n_confirmed_total += 1
                        self.confirmed.add(v)
                        break
                self.history.append((self.t_now, v))
            self.events[v] = self.t_now
        # decay expired events
        cutoff = self.t_now - self.decay_s
        for v in [v for v, te in self.events.items() if te < cutoff]:
            del self.events[v]
            self.confirmed.discard(v)
        self.history = [(ht, hv) for (ht, hv) in self.history
                        if ht >= cutoff]

    def _adjacent_occupied(self, v, logodds):
        nx, ny, nz = self.shape
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    w = (v[0] + dx, v[1] + dy, v[2] + dz)
                    if 0 <= w[0] < nx and 0 <= w[1] < ny and 0 <= w[2] < nz:
                        if logodds[w] > self.occ_lo:
                            return True
        return False

    def blocks(self, voxel):
        """True if a fresh CONFIRMED motion event is within block_radius.

        Only confirmed events (trail-backed) are actionable — one-off
        noise artifacts never veto. The query voxel is never used for
        indexing — out-of-bounds safe.
        """
        if not self.confirmed:
            return False
        r = self.block_radius
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        for (ex, ey, ez) in self.confirmed:
            if (ex, ey, ez) not in self.events:
                continue  # confirmed but decayed: not fresh
            if abs(ex - vx) <= r and abs(ey - vy) <= r and abs(ez - vz) <= r:
                return True
        return False

    @property
    def n_active(self):
        return len(self.events)
