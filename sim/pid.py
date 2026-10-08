"""Cascaded PID controller for trajectory tracking (Phase 1).

Plain English: the trajectory says where the drone should be and how fast
it should be going. The controller compares that to reality and commands
an acceleration to close the gap.

Structure (all host-side classical, no TSU claim):
  outer (position): v_corr = Kp_p * (p_des - p)          -> velocity correction
  inner (velocity): e_v = (v_des + v_corr) - v
                    a_cmd = Kp_v*e_v + Ki_v*∫e_v + Kd_v*d(e_v)/dt
                    (D on measurement to avoid setpoint kick; integrator
                    clamped = anti-windup)
  saturate |a_cmd| <= a_max (vehicle does this too — belt and suspenders).

Tuning is done empirically in tune_pid.py (hover + step response) and the
gains below are the tuned values — not hand-picked.
"""
import numpy as np


class CascadedPID:
    def __init__(self, Kp_p=0.8, Kp_v=8.0, Ki_v=0.5, Kd_v=0.6,
                 v_max=5.0, a_max=8.0, dt=0.01, i_clamp=3.0):
        self.Kp_p = float(Kp_p)
        self.Kp_v = float(Kp_v)
        self.Ki_v = float(Ki_v)
        self.Kd_v = float(Kd_v)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt = float(dt)
        self.i_clamp = float(i_clamp)
        self.reset()

    def reset(self):
        self._i = np.zeros(3)
        self._v_prev = None

    def command(self, p_des, v_des, p, v):
        """Return accel command (3,) to track the trajectory point."""
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        # outer: position error -> velocity correction, saturated
        v_corr = self.Kp_p * (np.asarray(p_des, float) - p)
        nc = np.linalg.norm(v_corr)
        if nc > self.v_max:
            v_corr = v_corr / nc * self.v_max
        v_sp = np.asarray(v_des, float) + v_corr
        ns = np.linalg.norm(v_sp)
        if ns > self.v_max:
            v_sp = v_sp / ns * self.v_max
        # inner: PID on velocity error (D on measurement)
        e = v_sp - v
        self._i = self._i + e * self.dt
        ni = np.linalg.norm(self._i)
        if ni > self.i_clamp:
            self._i = self._i / ni * self.i_clamp
        d_meas = np.zeros(3) if self._v_prev is None else (v - self._v_prev) / self.dt
        self._v_prev = v.copy()
        a = self.Kp_v * e + self.Ki_v * self._i - self.Kd_v * d_meas
        na = np.linalg.norm(a)
        if na > self.a_max:
            a = a / na * self.a_max
        return a
