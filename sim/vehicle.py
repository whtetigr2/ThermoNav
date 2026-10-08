"""Point-mass double-integrator vehicle model (Phase 1 dynamics).

Plain English: the drone is a point with mass. It has a position and a
velocity. You command an acceleration; the velocity changes, then the
position changes. Unlike the old kinematic sim (which teleported the
drone between voxel centers), this one has inertia — it cannot stop or
turn instantly.

State: pos (m), vel (m/s), 3D. 1 voxel = 1 m.
Integration: semi-implicit Euler at 100 Hz (dt=0.01) — stable for the
stiffness in this loop.

Parameters [assumed, X500-class quad]:
  mass   = 1.5 kg   (Holybro X500 V2 airframe ~1.6 kg with battery;
                     1.5 kg used as round nominal)
  v_max  = 5.0 m/s  (conservative indoor; X500 open-sky ~12-15 m/s)
  a_max  = 8.0 m/s² (snappy quad; well below motor saturation)

No TSU claim here: dynamics + control are host-side classical.
"""
import numpy as np

MASS = 1.5    # kg [assumed]
# V_MAX_CAP: X500-class capability [assumed]. V_MAX: operational limit.
# Measured (Phase 1): at 5 m/s the 1 Hz planner cannot keep up
# (overshoot/replan cycles, wandering -> max_steps); at 2 m/s navigation
# is clean. The vehicle CAN do 5 m/s, but safe indoor operation with the
# 1 Hz planning loop is ~2 m/s.
V_MAX_CAP = 5.0  # m/s [assumed]
V_MAX = 2.0      # m/s [measured: max safe with 1 Hz planner]
A_MAX = 8.0   # m/s^2 [assumed]
DT = 0.01     # s (100 Hz)


class PointMass:
    """Double integrator with accel/velocity saturation."""

    def __init__(self, pos, vel=None, v_max=V_MAX, a_max=A_MAX, dt=DT):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3) if vel is None else np.array(vel, dtype=float)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt = float(dt)
        self.t = 0.0

    def step(self, accel_cmd):
        """Advance one dt. Returns (pos, vel). Saturates accel and velocity."""
        a = np.array(accel_cmd, dtype=float)
        n = np.linalg.norm(a)
        if n > self.a_max:
            a = a / n * self.a_max
        # semi-implicit Euler: velocity first, then position
        self.vel = self.vel + a * self.dt
        vn = np.linalg.norm(self.vel)
        if vn > self.v_max:
            self.vel = self.vel / vn * self.v_max
        self.pos = self.pos + self.vel * self.dt
        self.t += self.dt
        return self.pos.copy(), self.vel.copy()
