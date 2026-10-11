"""Battery model — classical host model for the 2.4 kg vehicle.

Roadmap Phase 1, item 1 (2026-10-11): the escalation ladder's battery
trigger slot was a MISSING stub ("no power model in sim"). This module
is that power model. It is the SMALLEST HONEST VERSION (LEGO principle):
momentum-theory induced power + blade-profile allowance + parasite drag
+ climb power, a linear open-circuit-voltage curve, and a single
internal resistance for voltage sag. No Peukert effect, no temperature
dependence, no ageing — listed as known limits in BATTERY_MODEL.md.

CLAIM LABEL: simulation only. Every parameter is calculated from the
assumed 2.4 kg mass, sourced from the repo's own variable registry, or
labeled ASSUMED. NOTHING here is measured — we have no hardware and no
bench data. The hover-power output is cross-checked (not calibrated)
against the registry's independently recorded class-typical band.

Provenance key (repo convention):
  [standard]   physics constant / textbook value
  [registry]   recorded in ~/workspace/thermo-work/VARIABLE_REGISTRY.md
  [calculated] derived here from the above
  ASSUMED      our choice, no source — treat as a lever, not a fact
  [design]     policy/design choice, not a physical claim
"""
import math

# ---------------------------------------------------------------------------
# Vehicle / air (shared with vehicle_v3.py — same values, same provenance)
# ---------------------------------------------------------------------------
MASS = 2.4        # kg [registry: Paul's ruling 2026-10-08; do not change]
G = 9.81          # m/s^2 [standard]
RHO = 1.225       # kg/m^3 sea-level air [standard]
CDA = 0.15        # m^2 drag area [registry/NAPKIN, vehicle_v3.py CDA]

# ---------------------------------------------------------------------------
# Pack: 4S LiPo, 5000 mAh
# The registry's bottom-up AUW build already carries "520 g 4S 5000 mAh"
# as its battery line [registry: assumed-typical within listing range].
# We adopt exactly that pack so the mass budget and the energy budget
# describe the same vehicle.
# ---------------------------------------------------------------------------
CELLS_SERIES = 4              # 4S [registry pack line]
CAPACITY_AH = 5.0             # Ah [registry pack line]
V_NOMINAL = 14.8              # V = 4 x 3.7 V/cell [standard LiPo nominal]
CAPACITY_WH = V_NOMINAL * CAPACITY_AH   # 74.0 Wh [calculated]
PACK_MASS_G = 520             # g [registry: assumed-typical]

# Open-circuit voltage: linear in charge fraction between the standard
# LiPo per-cell endpoints 3.0 V (empty) and 4.2 V (full).
# ASSUMED simplification: the real curve has a plateau and a knee; the
# linear map is the smallest honest version and is monotone, which is
# what the ladder consumes. Pack OCV = 12.0 + 4.8 * soc.
V_CELL_EMPTY = 3.0            # V [standard LiPo cutoff convention]
V_CELL_FULL = 4.2             # V [standard LiPo full-charge]

# Internal resistance: one lumped pack resistance.
R_CELL = 0.020                # ohm ASSUMED (typical LiPo cell 15-25 mOhm;
                              # no bench measurement exists — MISSING)
R_PACK = CELLS_SERIES * R_CELL  # 0.080 ohm [calculated from ASSUMED]

# ---------------------------------------------------------------------------
# Propulsion chain
# Rotor: 1045 props (10 inch diameter) [registry: 2216/1045 on 4S class
# data]. Disk area from the prop diameter [calculated].
# ---------------------------------------------------------------------------
PROP_DIAMETER_M = 0.254       # m = 10 in [registry prop class]
N_ROTORS = 4                  # X500-class quad [registry]
DISK_AREA = N_ROTORS * math.pi * (PROP_DIAMETER_M / 2.0) ** 2  # m^2 [calc]

PROFILE_FRAC = 0.35   # ASSUMED: blade profile power = 35% of hover
                      # induced power, held constant at our low speeds.
ETA_DRIVE = 0.85      # ASSUMED: motor+ESC electrical->shaft efficiency.
P_AVIONICS = 25.0     # W ASSUMED lump: compute + lidar + FC + receiver.
                      # Itemized bench draw is MISSING.

# ---------------------------------------------------------------------------
# Policy thresholds (consumed by escalation.py; defined here so the
# battery module owns every battery number). ALL [design].
# ---------------------------------------------------------------------------
RESERVE_SOC = 0.20    # [design] reserve floor: at/below -> RTH intent
WARN_SOC = 0.30       # [design] warning band: log loudly, keep flying
CRITICAL_SOC = 0.10   # [design] at/below -> LAND now (hold window
                      # bypassed: the energy state only gets worse, and
                      # hover-to-empty ends in an uncontrolled drop)

# Phase-classification thresholds for logging [design]
CLIMB_VZ = 0.3        # m/s vertical speed above -> "climb"
CRUISE_VH = 0.5       # m/s horizontal speed above -> "cruise"
DESCENT_FLOOR_FRAC = 0.40  # ASSUMED: descent power never credited below
                           # 40% of hover propulsion (vortex-ring /
                           # control margin; no descent power model)


def classify_phase(vel):
    """Flight phase from the vehicle's OWN velocity (onboard state —
    no ground truth). Returns hover / cruise / climb / descent."""
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    if vz >= CLIMB_VZ:
        return "climb"
    if vz <= -CLIMB_VZ:
        return "descent"
    if math.hypot(vx, vy) >= CRUISE_VH:
        return "cruise"
    return "hover"


class BatteryModel:
    """Stateful pack: charge fraction, terminal voltage under load,
    time-to-reserve. Step it with a load power and a dt."""

    def __init__(self, start_soc=1.0):
        # start_soc must be in (0, 1]: a 0 start is a modeling error,
        # not a scenario — it raises here and surfaces as an ERROR row
        # in mission manifests (documented, RT-7).
        if not 0.0 < start_soc <= 1.0:
            raise ValueError(f"start_soc {start_soc} outside (0, 1]")
        self.soc = float(start_soc)
        self.start_soc = float(start_soc)
        self.wh_used = 0.0        # electrical energy drawn, Wh [calc]
        self.min_voltage = self.ocv()
        self.last_power_w = 0.0
        self.last_current_a = 0.0
        self.saturated = False    # load exceeded max-power point (never
                                  # expected at our ~300 W; flagged, not
                                  # silently clamped)

    # -- electrochemistry (the simple version) ----------------------------
    def ocv(self, soc=None):
        """Open-circuit pack voltage at a charge fraction [calculated
        from the ASSUMED linear curve]."""
        s = self.soc if soc is None else float(soc)
        v_cell = V_CELL_EMPTY + (V_CELL_FULL - V_CELL_EMPTY) * s
        return CELLS_SERIES * v_cell

    def terminal_voltage(self, power_w, soc=None):
        """Terminal voltage while delivering power_w [calculated].

        Solves P = V_t * I with V_t = OCV - I*R_PACK exactly:
        V_t = (OCV + sqrt(OCV^2 - 4 R P)) / 2  (the physical root).
        Sag therefore grows with load and with falling soc."""
        ocv = self.ocv(soc)
        disc = ocv * ocv - 4.0 * R_PACK * max(power_w, 0.0)
        if disc <= 0.0:
            return ocv / 2.0  # max-power point; caller checks `saturated`
        return (ocv + math.sqrt(disc)) / 2.0

    def current(self, power_w, soc=None):
        """Draw current (A) at a load power [calculated]."""
        ocv = self.ocv(soc)
        disc = ocv * ocv - 4.0 * R_PACK * max(power_w, 0.0)
        if disc <= 0.0:
            return ocv / (2.0 * R_PACK)
        v_t = (ocv + math.sqrt(disc)) / 2.0
        return max(power_w, 0.0) / v_t if v_t > 0 else 0.0

    # -- propulsion power (momentum theory) -------------------------------
    @staticmethod
    def hover_induced_power():
        """Ideal induced power in hover [calculated]: T * v_h with
        v_h = sqrt(T / (2 rho A)), T = m g."""
        t = MASS * G
        v_h = math.sqrt(t / (2.0 * RHO * DISK_AREA))
        return t * v_h, v_h

    def power_for_velocity(self, vel):
        """Electrical load power (W) for a velocity [calculated from
        ASSUMED chain parameters]. Components:
          induced   — momentum theory; forward speed reduces the
                      induced velocity via the textbook relation
                      v_i^2 = (-v^2 + sqrt(v^4 + 4 v_h^4)) / 2
          profile   — PROFILE_FRAC of hover induced, constant [ASSUMED]
          parasite  — 0.5 * rho * CdA * v^3 [calculated, CdA registry]
          climb     — m g vz added (credit on descent, floored)
          avionics  — constant lump [ASSUMED]
        """
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        v = math.hypot(vx, vy)
        p_ind_hover, v_h = self.hover_induced_power()
        vi2 = (-v * v + math.sqrt(v ** 4 + 4.0 * v_h ** 4)) / 2.0
        p_ind = MASS * G * math.sqrt(max(vi2, 0.0))
        p_profile = PROFILE_FRAC * p_ind_hover
        p_parasite = 0.5 * RHO * CDA * v ** 3
        p_climb = MASS * G * vz
        p_shaft = p_ind + p_profile + p_parasite + p_climb
        floor = DESCENT_FLOOR_FRAC * (p_ind_hover + p_profile)
        p_shaft = max(p_shaft, floor)
        return p_shaft / ETA_DRIVE + P_AVIONICS

    def power_for_phase(self, phase, speed=0.0, climb_rate=0.0):
        """Power for a named phase (test / scripting convenience)."""
        if phase == "hover":
            vel = (0.0, 0.0, 0.0)
        elif phase == "cruise":
            vel = (speed, 0.0, 0.0)
        elif phase == "climb":
            vel = (0.0, 0.0, abs(climb_rate))
        elif phase == "descent":
            vel = (0.0, 0.0, -abs(climb_rate))
        else:
            raise ValueError(f"unknown phase {phase}")
        return self.power_for_velocity(vel)

    # -- state outputs ------------------------------------------------------
    def time_to_reserve_s(self, power_w=None):
        """Seconds of flight left before the reserve floor at the given
        (or last) draw [calculated estimate].

        Stepped integration over soc in 0.002 increments, recomputing
        current at each step's voltage — this prices in that current
        rises as voltage falls. RT-5 (red-team 2026-10-11) measured the
        previous one-shot formula (charge / current-at-current-soc) as
        optimistic by +16.4% from full at hover draw and +21.2% at
        450 W, with the <5% claim holding only within ~0.1 soc of the
        floor. This stepped version tracks a fine brute-force
        integration to <1% (unit-tested against one)."""
        p = self.last_power_w if power_w is None else power_w
        if self.soc <= RESERVE_SOC:
            return 0.0
        if p is None or p <= 0.0:
            return float("inf")
        return self._time_to_soc(p, RESERVE_SOC)

    def _time_to_soc(self, p, target):
        """Stepped-integration time to reach a soc target at constant
        power (see time_to_reserve_s for why one-shot is wrong)."""
        t, s = 0.0, self.soc
        ds = 0.002
        while s > target:
            step = min(ds, s - target)
            i = self.current(p, soc=s - step / 2.0)
            if i <= 0.0:
                return float("inf")
            t += step * CAPACITY_AH / i * 3600.0
            s -= step
        return t

    def time_to_empty_s(self, power_w=None):
        p = self.last_power_w if power_w is None else power_w
        if p is None or p <= 0.0:
            return float("inf")
        return self._time_to_soc(p, 0.0)

    def step(self, dt, power_w):
        """Discharge for dt seconds at power_w. Returns a state dict:
        soc, voltage (terminal, under this load), current,
        time_to_reserve_s. Soc is floored at 0 (empty is a state the
        ladder must have acted long before)."""
        p = max(float(power_w), 0.0)
        ocv = self.ocv()
        self.saturated = (ocv * ocv - 4.0 * R_PACK * p) <= 0.0
        i = self.current(p)
        v_t = self.terminal_voltage(p)
        self.soc = max(0.0, self.soc - i * dt / (3600.0 * CAPACITY_AH))
        # Bookkeeping (RT-7): beyond the max-power point the pack can
        # only DELIVER ocv^2/(4R); book the delivered power (v_t * i at
        # the clamped operating point), not the requested power, so Wh
        # and charge never diverge. Unreachable in the flight envelope
        # (max draw 310 W vs >=450 W max-power point); latent only.
        p_delivered = p if not self.saturated else v_t * i
        self.wh_used += p_delivered * dt / 3600.0
        self.last_power_w, self.last_current_a = p, i
        self.min_voltage = min(self.min_voltage, v_t)
        return {
            "soc": self.soc,
            "voltage": v_t,
            "current_a": i,
            "power_w": p,
            "time_to_reserve_s": self.time_to_reserve_s(p),
        }

    def state(self):
        """Current outputs at the last stepped load."""
        return {
            "soc": self.soc,
            "voltage": self.terminal_voltage(self.last_power_w),
            "current_a": self.last_current_a,
            "power_w": self.last_power_w,
            "time_to_reserve_s": self.time_to_reserve_s(),
            "wh_used": self.wh_used,
        }
