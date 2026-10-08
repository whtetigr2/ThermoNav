"""Phase-3 vehicle model: wind + attitude lag + explicit gravity/thrust envelope.

Plain English: three things the old point-mass assumed away for free —

1. WIND. The air moves and the drone didn't plan for it. Modeled as
   aerodynamic drag from the *relative* airflow (wind minus ground
   velocity): a_wind = (0.5*rho*CdA/m) * |W - v| * (W - v). Steady part
   (0/2/4 m/s, test values [assumed]) + gust part (Ornstein-Uhlenbeck,
   theta=0.5/s, sigma=1.0 -> ~1 m/s std [assumed]). The controller never
   sees the wind; the PID + CBF must reject it.

2. ATTITUDE LAG (Paul's distributed-mass point). A real X-quad cannot
   instantly redirect its thrust — it must pitch/roll its body, and
   swinging ~2.4 kg on 0.25 m arms takes time. The commanded thrust
   direction chases the desired direction through a first-order lag with
   TAU_TILT=0.2 s [NAPKIN-derived, see below]. This is the physics the
   Phase-2b tuned filter (TAU_CMD=0.4 s) was implicitly standing in for.

3. GRAVITY + THRUST ENVELOPE. g=9.81 is now explicit [standard]. Hover
   thrust m*g is allocated FIRST; what remains is maneuver authority,
   derived from an assumed thrust-to-weight ratio — not an assumed a_max:
     a_up_max   = (T/W - 1) * g          (9.81 m/s^2 at T/W=2.0;
                                          6.57 at the 1.67 sensitivity)
     a_lat_max  = g * tan(tilt_max)       (6.87 m/s^2 at tilt_max=35 deg)
     a_down_max = g                        (motors idle; quads can't push
                                            down — the old symmetric +-8
                                            was optimistic here too)
   The old A_MAX=8.0 is now a DERIVED quantity, and at T/W=1.67 the old
   sim was optimistic upward (8.0 > 6.57) and lateral (8.0 > 6.87).

TAU_TILT derivation [NAPKIN]:
  X500-class: 500 mm wheelbase -> ~0.25 m motor arm [assumed geometry].
  Motor mass ~0.12 kg at each arm end [assumed] ->
    I ~= 4 * m_motor * L^2 = 4 * 0.12 * 0.0625 = 0.03 kg*m^2 per axis.
  Max differential torque: T_max = T/W * m * g = 2.0*2.4*9.81 = 47.1 N
  total, ~11.8 N/motor -> tau ~= 2 motors * 11.8 N * 0.25 m = 5.9 N*m.
  Bang-bang time for a 0.5 rad tilt step: t = 2*sqrt(theta*I/tau)
    = 2*sqrt(0.5*0.03/5.9) ~= 0.10 s.
  TAU_TILT = 0.2 s [assumed]: the 0.11 s rigid-body number plus margin
  for the attitude controller's own loop; typical closed-loop attitude
  response is 0.15-0.3 s. If this is wrong it errs conservative (slower).

Config flags let each change run alone (ablation order: wind only,
attitude only, gravity/envelope only, all three):
  gravity_envelope: False -> old behavior (no g, symmetric A_MAX=8.0,
                    MASS=1.5). True -> thrust/tilt envelope, MASS=2.4.
  attitude_lag:     False -> thrust direction snaps to commanded.
                    True  -> first-order chase with TAU_TILT.
  wind_steady:      m/s steady wind (0, 2, 4). wind_gust: OU on/off.
  tw_ratio:         thrust-to-weight (1.67 primary, 2.0 sensitivity).
  filter_tau:       passed through to the mission loop's 2b velocity-
                    command filter (None = filter bypassed -> tilt-lag-
                    only test). Not used inside the vehicle.

No TSU claim: dynamics + control are host-side classical.
"""
import math
import numpy as np

G = 9.81          # m/s^2 [standard]
RHO = 1.225       # kg/m^3, sea-level air [standard]
MASS_OLD = 1.5    # kg [assumed — Phase 1/2 test value]
MASS_NEW = 2.4    # kg [assumed — laden X500-class, background research]
A_MAX_OLD = 8.0   # m/s^2 [assumed — Phase 1/2 symmetric maneuver authority]
V_MAX = 2.0       # m/s [measured Phase 1: max safe with 1 Hz planner]
DT = 0.01         # s (100 Hz)

TW_PRIMARY = 2.0   # [assumed — typical camera quad]
TW_SENS = 1.67     # [assumed — background-research estimate at 2.4 kg,
                    #  sensitivity check: a_up_max = 6.57 < old 8.0]
TILT_MAX_DEG = 35.0  # deg [assumed]: max tilt before the attitude loop
                     # is out of its linear regime / camera unusable.
TAU_TILT = 0.2      # s [NAPKIN-derived, see module docstring]

CDA = 0.15         # m^2 [NAPKIN]: drag area of an X500-class quad.
                   # 4 m/s wind on 2.4 kg -> 0.61 m/s^2 disturbance.
OU_THETA = 0.5     # 1/s [assumed]: gust correlation rate (~2 s gusts)
OU_SIGMA = 1.0     # m/s/sqrt(s) [assumed]: -> ~1.0 m/s stationary gust std


def envelope(tw_ratio=TW_PRIMARY, tilt_max_deg=TILT_MAX_DEG):
    """Derived acceleration envelope. Returns dict [calculated]."""
    a_up = (tw_ratio - 1.0) * G
    a_lat = G * math.tan(math.radians(tilt_max_deg))
    return {"a_up_max": a_up, "a_lat_max": a_lat, "a_down_max": G,
            "t_max": tw_ratio * MASS_NEW * G}


class Phase3Vehicle:
    """Point-mass translation + optional wind, attitude lag, gravity/thrust
    envelope. The PID still commands *net* acceleration; this class turns
    it into achieved acceleration through the physical chain."""

    def __init__(self, pos, vel=None, cfg=None, seed=0):
        self.cfg = dict(cfg or {})
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3) if vel is None else np.array(vel, dtype=float)
        self.dt = DT
        self.t = 0.0
        self.v_max = V_MAX

        self.grav = bool(self.cfg.get("gravity_envelope", False))
        self.lag = bool(self.cfg.get("attitude_lag", False))
        self.mass = MASS_NEW if self.grav else MASS_OLD
        self.tw = float(self.cfg.get("tw_ratio", TW_PRIMARY))
        env = envelope(self.tw)
        self.a_up_max = env["a_up_max"]
        self.a_lat_max = env["a_lat_max"]
        self.t_max = env["t_max"]
        self.tilt_max = math.radians(
            float(self.cfg.get("tilt_max_deg", TILT_MAX_DEG)))
        self.tau_tilt = float(self.cfg.get("tau_tilt", TAU_TILT))

        # wind: steady vector (fixed direction per mission seed) + OU gust
        self.wind_steady_mag = float(self.cfg.get("wind_steady", 0.0))
        self.wind_gust = bool(self.cfg.get("wind_gust", False))
        self.rng = np.random.default_rng(seed)
        ang = self.rng.uniform(0.0, 2.0 * math.pi)
        self.wind_steady_vec = self.wind_steady_mag * np.array(
            [math.cos(ang), math.sin(ang), 0.0])
        self.gust_vec = np.zeros(3)
        self.wind_vec = self.wind_steady_vec.copy()  # current, for logging

        # attitude state: actual thrust direction (unit vector)
        self.d_actual = np.array([0.0, 0.0, 1.0])
        self._d_init = False

    def _update_gust(self):
        if not self.wind_gust:
            self.gust_vec[:] = 0.0
        else:
            th = OU_THETA
            sg = OU_SIGMA
            dt = self.dt
            # OU on horizontal components only (vertical gusts out of scope)
            for i in (0, 1):
                self.gust_vec[i] += (-th * self.gust_vec[i] * dt
                                     + sg * math.sqrt(dt)
                                     * self.rng.standard_normal())
        self.wind_vec = self.wind_steady_vec + self.gust_vec

    def _wind_accel(self):
        """Quadratic drag from relative airflow, per unit mass."""
        v_air = self.wind_vec - self.vel
        sp = float(np.linalg.norm(v_air))
        if sp < 1e-9:
            return np.zeros(3)
        k = 0.5 * RHO * CDA / self.mass
        return k * sp * v_air

    def step(self, accel_cmd):
        """One 100 Hz step. Returns (pos, vel)."""
        a_cmd = np.array(accel_cmd, dtype=float)
        self._update_gust()

        if self.grav:
            # Desired thrust vector: counter gravity + achieve a_cmd.
            t_des = self.mass * (a_cmd + np.array([0.0, 0.0, G]))
            # Thrust magnitude cap.
            tn = float(np.linalg.norm(t_des))
            if tn > self.t_max and tn > 1e-9:
                t_des = t_des / tn * self.t_max
                tn = self.t_max
            # Tilt cap: rotate the thrust vector back onto the tilt_max cone
            # SURFACE, preserving |T| (the motors are still at their
            # capped thrust; only the direction is limited). This caps
            # *sustained* level lateral accel at g*tan(tilt_max); a full-
            # thrust 35° tilt also climbs (see unit test §5).
            tz = t_des[2]
            if tz <= 1e-6:
                # commanded harder down than motors-idle allows: thrust
                # goes to zero, the vehicle falls at ~g.
                t_vec = np.zeros(3)
                d_des = self.d_actual  # keep last direction; magnitude 0
            else:
                h = t_des.copy()
                h[2] = 0.0
                hn = float(np.linalg.norm(h))
                if hn > tz * math.tan(self.tilt_max) and hn > 1e-9:
                    tn = float(np.linalg.norm(t_des))
                    h_dir = h / hn
                    t_vec = tn * (math.sin(self.tilt_max) * h_dir
                                  + np.array([0.0, 0.0,
                                              math.cos(self.tilt_max)]))
                else:
                    t_vec = t_des
                tn2 = float(np.linalg.norm(t_vec))
                d_des = t_vec / tn2 if tn2 > 1e-9 else self.d_actual
        else:
            # Old physics: a_cmd direction, symmetric accel cap.
            n = float(np.linalg.norm(a_cmd))
            if n > A_MAX_OLD and n > 1e-9:
                a_cmd = a_cmd / n * A_MAX_OLD
            d_des = a_cmd / n if n > 1e-9 else self.d_actual
            t_vec = None  # achieved accel computed directly below

        # Attitude lag: actual thrust direction chases desired.
        if self.lag:
            if not self._d_init:
                self.d_actual = d_des.copy()
                self._d_init = True
            else:
                al = min(1.0, self.dt / self.tau_tilt)
                self.d_actual = self.d_actual + al * (d_des - self.d_actual)
                dn = float(np.linalg.norm(self.d_actual))
                if dn > 1e-9:
                    self.d_actual = self.d_actual / dn
        else:
            self.d_actual = d_des.copy()
            self._d_init = True

        # Achieved acceleration.
        if self.grav:
            t_mag = float(np.linalg.norm(t_vec))
            a_ach = (t_mag / self.mass) * self.d_actual \
                - np.array([0.0, 0.0, G])
        else:
            a_ach = a_cmd.copy()
            if self.lag:
                # lag-only ablation: direction lags, magnitude from a_cmd
                n = float(np.linalg.norm(a_cmd))
                a_ach = self.d_actual * n

        # Wind: unmodeled acceleration disturbance.
        a_ach = a_ach + self._wind_accel()

        # integrate (semi-implicit Euler), velocity cap unchanged
        self.vel = self.vel + a_ach * self.dt
        vn = float(np.linalg.norm(self.vel))
        if vn > self.v_max:
            self.vel = self.vel / vn * self.v_max
        self.pos = self.pos + self.vel * self.dt
        self.t += self.dt
        return self.pos.copy(), self.vel.copy()
