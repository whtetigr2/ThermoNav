"""TN-side Tier B spot check (whiteout battery graduation, 2026-10-12).

Adapted from the research red-team sub-CRITICAL construction
(`atk1_synth_subcrit`, thermo-work branch
`muse/roadmap-whiteout-2026-10-11` @ `5827ab7`, WHITEOUT_VALIDATION.md
S3/S4): an empty synthetic 32x32x16 room, whiteout fog for the whole
mission, saturated scripted load (+2500 W requested, delivered capped
at the pack max-power point), and a scripted start soc of 0.09 — the
battery flag fires already BELOW CRITICAL (0.10), so the countdown is
0 and the terminal descent starts below the reserve-required start
soc. Under the ported envelope this is Tier B: the CRITICAL touchdown
floor is NOT guaranteed, but a touchdown with soc > 0 is, provided the
terminal-start soc is above r * n(z) at the sizing-time rate r.

TN divergences from the research construction, stated plainly:
  * TN's mission loop has no jitter controller, so the research
    jitter spec is omitted (the whiteout_term cell in TN's battery
    validation carries the same divergence).
  * TN's cfg obscurant is static, so whiteout is set once for the
    whole mission (no DensitySchedule wrapper is needed here —
    the research subcrit arm is also whiteout for its whole run).
  * A completed landing in TN may end "landed_disarmed" (DISARM
    fires after confirmed touchdown); both strings mean on the
    ground.

Gate: outcome is a landing, no collision, safety_emergency_descent
is armed, terminal steps >= 1, and batt_min_soc > 0.

Output: whiteout_tierb_spotcheck.csv — one row, runtime-stamped
with the executed git commit (TN provenance discipline).

Claim label: validated in simulation only. No silicon / aircraft
claim. The 1.0 m/s emergency rate is a velocity-envelope claim only;
the sim has no touchdown-impact model at any rate.
"""
import os
import sys
import csv
import time
import traceback
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mission_dynamics_v5 as md

BASE_CFG = {"fix_atten_d5": True, "fix_confirm": True,
            "fix_lowvis": True, "fix_stale": True,
            "escalation": True, "battery": True,
            "battery_safety": True}
FULL_PHYSICS = {"gravity_envelope": True, "attitude_lag": True,
                "tw_ratio": 2.0, "filter_tau": 0.4,
                "wind_steady": 0.0, "wind_gust": False}
NOISE = 0.15
SCRIPT = os.path.basename(os.path.abspath(__file__))
START_SOC = 0.09

FIELDS = ["scenario", "room", "seed", "noise", "start_soc",
          "extra_load_spec", "outcome", "collision", "sim_time_s",
          "safety_flag_fired", "safety_rung", "safety_countdown_s",
          "safety_terminal_steps", "safety_emergency_descent",
          "esc_land_vetoed", "esc_batt_terminal_land",
          "batt_final_soc", "batt_min_soc",
          "tier_b_note", "wall_s", "git_commit", "backend",
          "script", "note"]


def git_commit():
    out = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=10)
    return out.stdout.strip() or "UNKNOWN"


def main():
    git_hash = git_commit()
    print(f"code commit: {git_hash}", flush=True)
    occ = np.zeros((32, 32, 16), dtype=np.uint8)
    cfg = dict(BASE_CFG, nav_trustworthy=False,
               battery_extra_load_w=2500.0,
               battery_start_soc=START_SOC)
    cfg.update(FULL_PHYSICS)
    cfg["obscurant"] = {"archetype": "fog", "density": "whiteout"}
    mid = "tierb_subcrit_synthetic32_s0"
    t0 = time.time()
    note = ""
    try:
        row = md.run_mission_dynamics(
            occ, "synthetic32", 0, NOISE, mid, backend="numpy",
            write_traj=False, verbose=False, cfg=cfg)
    except Exception as e:  # noqa: BLE001
        row = {}
        note = traceback.format_exc(limit=3).replace("\n", " | ")
    outcome = row.get("outcome")
    events = row.get("safety_events") or []
    emerg = any(e[1] == "emergency_terminal_descent" for e in events)
    rec = {
        "scenario": "tierb_subcrit", "room": "synthetic32",
        "seed": 0, "noise": NOISE, "start_soc": START_SOC,
        "extra_load_spec": "+2500W constant (saturated)",
        "outcome": outcome,
        "collision": 1 if outcome == "collision" else 0,
        "sim_time_s": row.get("sim_time", -1),
        "safety_flag_fired": row.get("safety_flag_fired", -1),
        "safety_rung": row.get("safety_rung", "?"),
        "safety_countdown_s": row.get("safety_countdown_s", -1),
        "safety_terminal_steps": row.get("safety_terminal_steps", -1),
        "safety_emergency_descent": row.get(
            "safety_emergency_descent", -1),
        "esc_land_vetoed": row.get("esc_land_vetoed", -1),
        "esc_batt_terminal_land": row.get("esc_batt_terminal_land", -1),
        "batt_final_soc": row.get("batt_final_soc", -1),
        "batt_min_soc": row.get("batt_min_soc", -1),
        "tier_b_note": (
            f"emergency event logged: {emerg}; flag fired below "
            "CRITICAL, countdown 0, Tier B (soc>0 guaranteed, "
            "CRITICAL floor not guaranteed)"),
        "wall_s": round(time.time() - t0, 1),
        "git_commit": git_hash, "backend": "numpy",
        "script": SCRIPT, "note": note,
    }
    print(f"  {mid}: {outcome} rung={rec['safety_rung']} "
          f"cd={rec['safety_countdown_s']} "
          f"terminal={rec['safety_terminal_steps']} "
          f"emergency={rec['safety_emergency_descent']} "
          f"min_soc={rec['batt_min_soc']} ({rec['wall_s']}s)",
          flush=True)
    out_path = os.path.join(HERE, "whiteout_tierb_spotcheck.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(rec)
    landed = outcome in ("landed", "landed_disarmed")
    ok = (landed and not rec["collision"]
          and rec["safety_emergency_descent"] == 1
          and rec["safety_terminal_steps"] >= 1
          and isinstance(rec["batt_min_soc"], (int, float))
          and rec["batt_min_soc"] > 0)
    print(f"GATE tierb_subcrit emergency descent lands soc>0: "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"wrote {out_path}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
