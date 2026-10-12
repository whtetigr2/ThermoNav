"""Dodge LEGO: linear threat projection + reactive sidestep (2026-10-10).

Built on the motion layer's confirmed trail (branch
muse/motion-lego-2026-10-10). The red-team's key correction on that
branch: velocity is NOT a sensor problem — the confirmed motion trail
already yields a velocity vector for free. What's missing is a DODGE.
Veto-only protection guards voxels the drone *plans* to enter, but in a
lateral crossing the threat arrives at the drone's *current* position
(diagnosed 071/crossing: evented t=62, confirmed t=63, collision with
zero vetoes). This LEGO extracts the velocity from the trail, projects
the mover's position forward, and commands a reactive lateral sidestep
at the D5 level (like the existing veto/hover — not a planner rewrite).

Why linear projection is enough for the LEGO: the staged movers are
constant-velocity spheres (see movers.py). A Kalman tracker is the
unbuilt next stage; if linear projection + sidestep can't beat the
motion layer's 30/48, that is the finding — do not build the tracker,
report it.

No ground-truth leakage: inputs are the motion buffer's confirmed trail
(scan evidence only), sim time, and the drone's own pose. The mover's
true velocity is never consulted.

Claim label: Z1-legal formulation, validated in simulation (host-side
classical component; no TSU claim).
"""

import math

import numpy as np

TRAIL_WINDOW_S = 6.0   # keep per-scan trail points this long [design].
                       # The 1 m voxel grid makes the latest-voxel trail a
                       # staircase; a longer window averages the quantization
                       # out of the least-squares velocity fit.
MIN_TRAIL_DT = 1.0     # need >= this time span for a velocity estimate.
                       # Costs ~1 extra scan of latency past confirmation.
MIN_DISPLACEMENT = 0.5  # m: below this the "mover" is noise/stationary
SPEED_LO = 0.15        # m/s sanity band (movers capped at 2.0 m/s)
SPEED_HI = 5.0
PROJ_T = 2.0           # s: project the mover forward this far (linear)
THREAT_RADIUS = 2.0    # m: closest-approach distance that counts as a
                       # threat (mover r=0.5 + drone r~0.3 + ~1.2 m margin)
SIDESTEP_DIST = 2.5    # m: lateral move distance (clears THREAT_RADIUS
                       # with margin on both sides of the mover's line)
SIDESTEP_SPEED = 2.0   # m/s: sidestep servo speed cap [design]
DODGE_DURATION = 2.0   # s: servo toward the sidestep target this long
DODGE_COOLDOWN = 4.0   # s: min time between dodges (no oscillation)
SIDESTEP_CLEAR_MARGIN = 1.0  # m: sidestep needs DIST + this clearance


class DodgeController:
    """Reactive sidestep driven by the motion layer's confirmed trail."""

    def __init__(self, threat_radius=THREAT_RADIUS, proj_t=PROJ_T,
                 sidestep_dist=SIDESTEP_DIST, sidestep_speed=SIDESTEP_SPEED,
                 dodge_duration=DODGE_DURATION,
                 dodge_cooldown=DODGE_COOLDOWN):
        self.threat_radius = threat_radius
        self.proj_t = proj_t
        self.sidestep_dist = sidestep_dist
        self.sidestep_speed = sidestep_speed
        self.dodge_duration = dodge_duration
        self.dodge_cooldown = dodge_cooldown
        self.trail = []  # (t, centroid_xyz): one snapshot per 1 Hz scan
        self.active = False
        self.target = None
        self.t_end = 0.0
        self.cooldown_until = 0.0
        self.n_dodges = 0
        self.n_fallbacks = 0       # threat seen but no safe sidestep
        self.n_wall_incidents = 0  # dodge path failed D5 mid-execution
        self.last_threat = None    # (mover_pos, mover_vel) diagnostic

    def observe(self, motion_buffer, t):
        """1 Hz: snapshot the trail as (t, latest-voxel) points.

        The trail point is the LATEST confirmed voxel (max event time),
        not the centroid of all confirmed voxels: the confirmed set
        accumulates ~2 s of trail, so its centroid is the trail's
        midpoint and moves at ~half the mover's speed. The latest voxel
        is the mover's most recent painted position; differencing those
        gives the true velocity. (Measured 2026-10-10: centroid version
        estimated 0.54 m/s for a 1.0 m/s mover.)
        """
        t = float(t)
        conf = [(motion_buffer.events[v], v)
                for v in motion_buffer.confirmed
                if v in motion_buffer.events]
        if conf:
            _, v = max(conf, key=lambda te_v: te_v[0])
            c = np.array([v[0] + 0.5, v[1] + 0.5, v[2] + 0.5])
            if not self.trail or self.trail[-1][0] < t:
                self.trail.append((t, c))
        cutoff = t - TRAIL_WINDOW_S
        self.trail = [(tt, cc) for (tt, cc) in self.trail if tt >= cutoff]

    def estimate(self):
        """Least-squares velocity from the trail.

        Returns (pos_at_latest_t, vel) or None. The confirmed buffer holds
        ~2 s of trail, so the centroid lags the mover; the LS fit
        extrapolates to the latest scan time — that is the current-position
        estimate, and the slope is the velocity. Free, no new sensor.
        """
        if len(self.trail) < 2:
            return None
        ts = np.array([tt for tt, _ in self.trail])
        ps = np.array([cc for _, cc in self.trail])
        dt = float(ts[-1] - ts[0])
        if dt < MIN_TRAIL_DT:
            return None
        if float(np.linalg.norm(ps[-1] - ps[0])) < MIN_DISPLACEMENT:
            return None
        A = np.vstack([np.ones_like(ts), ts]).T
        vel = np.zeros(3)
        pos = np.zeros(3)
        t1 = float(ts[-1])
        for i in range(3):
            coef, *_ = np.linalg.lstsq(A, ps[:, i], rcond=None)
            vel[i] = coef[1]
            pos[i] = coef[0] + coef[1] * t1
        speed = float(np.linalg.norm(vel))
        if not (SPEED_LO <= speed <= SPEED_HI):
            return None
        return pos, vel

    def check_threat(self, drone_pos, drone_vel):
        """(mover_pos, mover_vel) if a lateral threat is imminent, else None.

        Threat = mover closing on the drone whose linear projection comes
        within THREAT_RADIUS of the drone's CURRENT position within PROJ_T.
        This is exactly the case the veto can't cover (threat arrives at
        current position, not planned voxels).

        Uses RELATIVE velocity (v_mover - v_drone): the drone is usually
        cruising toward the crossing point, so a stationary-drone
        assumption wildly overestimates the time to closest approach.
        (Measured 2026-10-10: stationary assumption gave t_star=3.9 s for
        a collision 1.5 s out.)
        """
        est = self.estimate()
        if est is None:
            return None
        p_m, v_m = est
        p_d = np.asarray(drone_pos, dtype=float)
        v_d = np.asarray(drone_vel, dtype=float)
        v_rel = v_m - v_d
        r = p_d - p_m
        if float(np.dot(v_rel, r)) <= 0:
            return None  # not closing on the drone
        # time of closest approach of the relative motion r - v_rel*t
        vr2 = float(np.dot(v_rel, v_rel))
        if vr2 < 1e-9:
            return None
        t_star = float(np.dot(r, v_rel)) / vr2
        if not (0.0 < t_star <= self.proj_t):
            return None
        d_close = float(np.linalg.norm(r - v_rel * t_star))
        if d_close > self.threat_radius:
            return None
        self.last_threat = (p_m, v_m)
        return p_m, v_m

    def plan_dodge(self, drone_pos, mover_pos, mover_vel,
                   clearance_fn, blocked_fn):
        """Pick a lateral sidestep target, or None if none is safe.

        Sidestep is perpendicular to the mover's velocity (horizontal
        plane — movers are at drone altitude; vertical dodges risk
        floor/ceiling). Both perpendicular directions clear the mover's
        line geometrically; obstacle clearance picks the side. If neither
        side is safe, return None and the caller falls back to the motion
        layer's hover-veto.
        """
        v = np.asarray(mover_vel, dtype=float)
        vh = np.array([v[0], v[1], 0.0])
        n = float(np.linalg.norm(vh))
        if n < 1e-9:
            return None
        perp = np.array([-vh[1], vh[0], 0.0]) / n
        p_d = np.asarray(drone_pos, dtype=float)
        best = None
        for sign in (+1.0, -1.0):  # deterministic: +perp preferred on ties
            d = sign * perp
            clr = float(clearance_fn(p_d, d))
            if clr < self.sidestep_dist + SIDESTEP_CLEAR_MARGIN:
                continue
            tgt = p_d + d * self.sidestep_dist
            tv = (int(tgt[0]), int(tgt[1]), int(tgt[2]))
            if bool(blocked_fn(tv)):
                continue
            if best is None or clr > best[1]:
                best = (tgt, clr)
        return best[0] if best is not None else None

    def trigger(self, drone_pos, drone_vel, t, clearance_fn, blocked_fn):
        """10 Hz: start a dodge if a new threat appears. Returns the
        sidestep target xyz on the triggering substep, else None."""
        t = float(t)
        if self.active or t < self.cooldown_until:
            return None
        th = self.check_threat(drone_pos, drone_vel)
        if th is None:
            return None
        p_m, v_m = th
        tgt = self.plan_dodge(drone_pos, p_m, v_m, clearance_fn, blocked_fn)
        if tgt is None:
            self.n_fallbacks += 1
            self.cooldown_until = t + 2.0  # don't spam fallbacks
            return None
        self.active = True
        self.target = np.asarray(tgt, dtype=float)
        self.t_end = t + self.dodge_duration
        self.n_dodges += 1
        return self.target

    def update_active(self, t):
        """Expire a finished dodge and start the cooldown."""
        t = float(t)
        if self.active and t >= self.t_end:
            self.active = False
            self.target = None
            self.cooldown_until = t + self.dodge_cooldown

    def finish_early(self, t):
        """Arrived at the sidestep target — end the dodge now."""
        self.active = False
        self.target = None
        self.cooldown_until = float(t) + self.dodge_cooldown
