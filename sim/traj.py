"""Minimum-jerk trajectory generation (Phase 1).

Plain English: given where the drone is (position + velocity) and where
it should go, build the smoothest possible path between them — a quintic
(minimum-jerk) polynomial — then stretch its duration until velocity and
acceleration stay within the vehicle's limits.

Two uses:
  1. smooth_waypoints(points): the full A* voxel path -> chained smooth
     trajectory (the "trajectory smoothing" requirement). Unit-tested on a
     real A* path with limit verification.
  2. plan_segment(p0, v0, p1, v1): one receding-horizon segment, used by
     the mission loop each S2 substep (current state -> accepted voxel).

No TSU claim here: trajectory generation is host-side classical.
"""
import numpy as np


def _quintic_coeffs(p0, v0, a0, p1, v1, a1, T):
    """Solve for quintic coefficients. Returns 6x3 array (one row per power)."""
    # p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
    # boundary: p(0)=p0, v(0)=v0, a(0)=a0, p(T)=p1, v(T)=v1, a(T)=a1
    c0 = np.asarray(p0, dtype=float)
    c1 = np.asarray(v0, dtype=float)
    c2 = np.asarray(a0, dtype=float) / 2.0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [T3, T4, T5],
        [3*T2, 4*T3, 5*T4],
        [6*T, 12*T2, 20*T3],
    ])
    b = np.stack([
        np.asarray(p1, dtype=float) - (c0 + c1*T + c2*T2),
        np.asarray(v1, dtype=float) - (c1 + 2*c2*T),
        np.asarray(a1, dtype=float) - 2*c2,
    ])
    # b has shape (3,3); solve per axis
    c345 = np.linalg.solve(M, b)  # (3,3): rows=c3,c4,c5; cols=x,y,z
    return np.stack([c0, c1, c2, c345[0], c345[1], c345[2]])  # (6,3)


def _eval_quintic(c, t):
    """Evaluate quintic + derivatives. c: (6,3). Returns (pos, vel, acc)."""
    t = float(t)
    tp = np.array([t**k for k in range(6)])
    tp1 = np.array([k * t**(k-1) if k > 0 else 0.0 for k in range(6)])
    tp2 = np.array([k*(k-1) * t**(k-2) if k > 1 else 0.0 for k in range(6)])
    return tp @ c, tp1 @ c, tp2 @ c


class Segment:
    """One quintic segment with duration T."""

    def __init__(self, coeffs, T):
        self.c = coeffs
        self.T = float(T)

    def eval(self, t):
        t = min(max(float(t), 0.0), self.T)
        return _eval_quintic(self.c, t)

    def max_acc(self, n=64):
        ts = np.linspace(0, self.T, n)
        return max(np.linalg.norm(self.eval(t)[2]) for t in ts)

    def max_vel(self, n=64):
        ts = np.linspace(0, self.T, n)
        return max(np.linalg.norm(self.eval(t)[1]) for t in ts)


def plan_segment(p0, v0, p1, v1, v_max, a_max, T0=None, a0=None):
    """Minimum-jerk segment from (p0,v0) to (p1,v1), time-stretched until
    |v|<=v_max and |a|<=a_max. Returns Segment."""
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a0 = np.zeros(3) if a0 is None else np.asarray(a0, dtype=float)
    dist = float(np.linalg.norm(p1 - p0))
    if T0 is None:
        T0 = max(0.25, 2.0 * dist / max(v_max, 1e-6))
    T = T0
    for _ in range(40):
        c = _quintic_coeffs(p0, v0, a0, p1, v1, np.zeros(3), T)
        seg = Segment(c, T)
        if seg.max_acc() <= a_max and seg.max_vel() <= v_max:
            return seg
        T *= 1.25
    # Fallback: return the last (limits approximately respected); caller
    # saturates anyway. Should not happen for sane inputs.
    return seg


def _prune_collinear(points):
    """Drop intermediate voxels on straight runs (6-connected A* paths are
    full of them). Keeps first, last, and direction-change points."""
    pts = [tuple(p) for p in points]
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        d1 = (pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1], pts[i][2]-pts[i-1][2])
        d2 = (pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1], pts[i+1][2]-pts[i][2])
        if d1 != d2:
            out.append(pts[i])
    out.append(pts[-1])
    return out


class Trajectory:
    """Chained minimum-jerk segments. eval(t) -> (pos, vel)."""

    def __init__(self, segments):
        self.segments = segments
        self._t = np.cumsum([0.0] + [s.T for s in segments])
        self.T_total = float(self._t[-1]) if len(self._t) else 0.0

    def eval(self, t):
        if not self.segments:
            raise ValueError("empty trajectory")
        t = min(max(float(t), 0.0), self.T_total)
        i = int(np.searchsorted(self._t[1:], t, side="right"))
        i = min(i, len(self.segments) - 1)
        p, v, _ = self.segments[i].eval(t - self._t[i])
        return p, v


def smooth_waypoints(points, v_cruise, v_max, a_max):
    """Full A* voxel path -> smooth flyable trajectory.

    points: sequence of voxel tuples (or continuous points at voxel
        centers). Via-velocities use central differences scaled to
        v_cruise; start/end velocities are zero. Each segment is
        time-stretched to respect v_max/a_max.
    """
    pts = _prune_collinear(points)
    pc = [np.array(p, dtype=float) for p in pts]
    n = len(pc)
    if n < 2:
        raise ValueError("need >= 2 waypoints")
    vels = [np.zeros(3)]
    for i in range(1, n - 1):
        d = pc[i+1] - pc[i-1]
        nd = np.linalg.norm(d)
        vels.append(d / nd * v_cruise if nd > 1e-9 else np.zeros(3))
    vels.append(np.zeros(3))
    segs = []
    for i in range(n - 1):
        segs.append(plan_segment(pc[i], vels[i], pc[i+1], vels[i+1],
                                v_max, a_max))
    return Trajectory(segs)
