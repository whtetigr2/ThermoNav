"""Below-clear Option A validation — ThermoNav port (graduation
2026-10-12). Ported from the research runner
(slow-drone/dynamics_sim/run_belowclear_validation.py, thermo-work
branch muse/belowclear-2026-10-12 @ 5ef3763; research red-team
verdict PASS WITH CAVEATS) against the ported
sim/mission_dynamics_v5.py.

Option A (Paul, decided 2026-10-12): for NON-battery LAND demands,
the below-clear veto may hold for at most cfg["land_hold_window_s"]
seconds of continuous veto; then the descent proceeds at creep rate
regardless of the veto. Flag default OFF (key unset = legacy
bit-identical).

ThermoNav divergences from the research runner, stated plainly:
  * Occupancy grids load from ../rooms (byte-identical to the
    research real_lidar grids for 071/131 — md5-verified at
    graduation).
  * The s4 cells' jitter spec is accepted by the cfg but
    ThermoNav's loop has no jitter controller (map_age stays 0.0),
    so those cells run the whiteout density schedule without the
    jitter component — the same divergence the battery safety
    validation carries for its whiteout_term cell.
  * A completed landing in TN may end "landed_disarmed" (DISARM
    fires after confirmed touchdown); both strings mean on the
    ground. The research loop has no DISARM and ends "landed".
  * BASE_CFG / FULL_PHYSICS are imported from
    run_battery_safety_validation (identical values to the research
    whiteout/battery runners' cfgs).

Doctrine order — legacy FIRST, then the new behavior:

Cells (paired OFF / ON, window 10 s):
  s4_clears        whiteout-arrival cell (research whiteout s4:
                   indoorcd_131 s80010, soc 0.215, FULL_PHYSICS,
                   whiteout steps<40 then clear). TN legacy baseline
                   derived on unpatched TN main @ a2cb278 (this
                   runner, pre-port code): landed, 78 scans, vetoed
                   51, terminal 17, final soc 0.1014 — the research
                   committed numbers exactly.
  s4_paired_indef  same, whiteout never lifts. TN baseline @
                   a2cb278: landed_disarmed (the TN DISARM outcome
                   string), same numbers (78/51/17/0.1014).
  rtb_clear        clear-air RTB arrival (battery safety
                   validation's cell: indoorcd_071 s1, soc 0.26).
                   TN baseline @ a2cb278: landed, 110 scans, vetoed
                   28, terminal 18, final soc 0.1016 (matches the
                   research base-code reference @ 5827ab7 and TN's
                   committed battery_safety_validation.csv row).

ON expectations: vetoes per episode bounded by the window; the
landing happens; collision price reported next to the battery
land-regardless price (2 terminal-bypass collisions on record,
BATTERY_MODEL.md).

Outputs: belowclear_validation.csv (one row per mission, every
lever a column, git commit of the executed code in every row).

Claim label: classical host safety behavior, validated in
simulation only. Phase 0 caveat (see BELOWCLEAR_PHASE0.md): the
record rooms' ground-truth grids carry no floor and the sensor fan
never applies vehicle attitude, so descent columns are
unobservable-by-construction UNKNOWN at decision altitude in these
cells — every price below is simulation-only under that caveat.
No silicon / aircraft claim.
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

import mission_dynamics_v5 as md  # noqa: E402
import obscurant  # noqa: E402
from run_battery_safety_validation import (  # noqa: E402
    BASE_CFG, FULL_PHYSICS)

REAL_DIR = os.path.join(HERE, "..", "rooms")
NOISE = 0.15
SCRIPT = os.path.basename(os.path.abspath(__file__))
WINDOW_S = 10.0
JITTER = {"mode": "isolated", "delay": 2.0, "at": 8}
LANDED = ("landed", "landed_disarmed")  # TN: DISARM after touchdown

FIELDS = ["cell", "arm", "room", "seed", "noise", "start_soc",
          "window_s", "outcome", "collision", "outer_steps",
          "safety_rung", "safety_terminal_steps", "safety_arrived_steps",
          "esc_land_vetoed", "esc_land_hold_episodes",
          "esc_land_hold_expired_steps", "esc_land_hold_max_s",
          "esc_batt_terminal_land", "esc_batt_land_unverified",
          "batt_final_soc", "batt_min_soc", "wall_s", "git_commit",
          "backend", "script", "note"]


def git_commit():
    out = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=10)
    return out.stdout.strip() or "UNKNOWN"


class DensitySchedule:
    """Runner-side time-varying obscurant (ported verbatim from the
    research whiteout runner). bands: list of (from_step,
    density_label), ascending. Installs a scan_pose wrapper on the
    obscurant module for one mission and restores the original
    afterwards. Also counts outer steps. The step index is derived
    from the loop's scan seed (1000 + mission_seed*100 + step),
    identical in the ThermoNav loop.
    """

    def __init__(self, mission_seed, bands):
        self.mission_seed = mission_seed
        self.bands = bands
        self.n_scans = 0
        self.current = None
        self._orig = None

    def _density_at(self, step):
        d = self.bands[0][1]
        for from_step, label in self.bands:
            if step >= from_step:
                d = label
        return d

    def __enter__(self):
        self._orig = obscurant.scan_pose
        sched = self

        def wrapped(occ, pose, n_az=120, elevations=(-0.26, 0.0, 0.26),
                    max_range=40.0, range_noise=0.15, seed=0):
            step = int(seed) - 1000 - sched.mission_seed * 100
            sched.n_scans += 1
            want = sched._density_at(max(step, 0))
            if want != sched.current:
                obscurant.configure("fog", want)
                sched.current = want
            return sched._orig(occ, pose, n_az=n_az,
                               elevations=elevations,
                               max_range=max_range,
                               range_noise=range_noise, seed=seed)

        obscurant.scan_pose = wrapped
        return self

    def __exit__(self, *exc):
        obscurant.scan_pose = self._orig
        return False


class StepCounter:
    """Counts obscurant.scan_pose calls = outer steps (the research
    runner's disclosed trick), for cells without a DensitySchedule."""

    def __init__(self):
        self.n = 0
        self._orig = None

    def __enter__(self):
        self._orig = obscurant.scan_pose

        def counting(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        obscurant.scan_pose = counting
        return self

    def __exit__(self, *exc):
        obscurant.scan_pose = self._orig
        return False


def run_cell(cell, room, seed, cfg, start_soc, bands, git_hash):
    occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
    rows = []
    for arm in ("off", "on"):
        c = dict(cfg)
        if start_soc is not None:
            c["battery_start_soc"] = start_soc
        if bands:
            c["obscurant"] = {"archetype": "fog",
                               "density": bands[0][1]}
        if arm == "on":
            c["land_hold_window_s"] = WINDOW_S
        mid = f"belowclear_{cell}_{arm}_{room}_s{seed}"
        t0 = time.time()
        note = ("SIMULATION ONLY. Phase 0 caveat: record rooms have "
                "no GT floor, fan never tilt-corrected; descent "
                "columns unobservable by construction. See "
                "BELOWCLEAR_VALIDATION.md.")
        row = {}
        steps = -1
        try:
            if bands:
                with DensitySchedule(seed, bands) as sched:
                    row = md.run_mission_dynamics(
                        occ, room, seed, NOISE, mid, backend="numpy",
                        write_traj=False, verbose=False, cfg=c)
                steps = sched.n_scans
            else:
                with StepCounter() as sc:
                    row = md.run_mission_dynamics(
                        occ, room, seed, NOISE, mid, backend="numpy",
                        write_traj=False, verbose=False, cfg=c)
                steps = sc.n
            outcome = row.get("outcome")
        except Exception as e:  # noqa: BLE001
            outcome = f"ERROR:{type(e).__name__}:{e}"
            note = traceback.format_exc(limit=3).replace("\n", " | ")
        rec = {
            "cell": cell, "arm": arm, "room": room, "seed": seed,
            "noise": NOISE, "start_soc": start_soc,
            "window_s": WINDOW_S if arm == "on" else -1,
            "outcome": outcome,
            "collision": 1 if outcome == "collision" else 0,
            "outer_steps": steps,
            "safety_rung": row.get("safety_rung", "?"),
            "safety_terminal_steps": row.get("safety_terminal_steps", -1),
            "safety_arrived_steps": row.get("safety_arrived_steps", -1),
            "esc_land_vetoed": row.get("esc_land_vetoed", -1),
            "esc_land_hold_episodes": row.get("esc_land_hold_episodes", -1),
            "esc_land_hold_expired_steps": row.get(
                "esc_land_hold_expired_steps", -1),
            "esc_land_hold_max_s": row.get("esc_land_hold_max_s", -1),
            "esc_batt_terminal_land": row.get("esc_batt_terminal_land", -1),
            "esc_batt_land_unverified": row.get(
                "esc_batt_land_unverified", -1),
            "batt_final_soc": row.get("batt_final_soc", -1),
            "batt_min_soc": row.get("batt_min_soc", -1),
            "wall_s": round(time.time() - t0, 1), "git_commit": git_hash,
            "backend": "numpy", "script": SCRIPT, "note": note,
        }
        print(f"  {mid}: {outcome} steps={steps} "
              f"vetoed={rec['esc_land_vetoed']} "
              f"episodes={rec['esc_land_hold_episodes']} "
              f"expired={rec['esc_land_hold_expired_steps']} "
              f"terminal={rec['safety_terminal_steps']} "
              f"final_soc={rec['batt_final_soc']} "
              f"({rec['wall_s']}s)", flush=True)
        rows.append(rec)
    return rows


def main():
    gh = git_commit()
    print(f"code commit: {gh}", flush=True)
    recs = []

    print("=== s4_clears (whiteout arrival, lifts at 40) ===", flush=True)
    cfg = dict(BASE_CFG, nav_trustworthy=False)
    cfg.update(FULL_PHYSICS)
    cfg["jitter"] = dict(JITTER)
    recs += run_cell("s4_clears", "indoorcd_131", 80010, cfg, 0.215,
                     [(0, "whiteout"), (40, "clear")], gh)

    print("=== s4_paired_indef (whiteout never lifts) ===", flush=True)
    cfg = dict(BASE_CFG, nav_trustworthy=False)
    cfg.update(FULL_PHYSICS)
    cfg["jitter"] = dict(JITTER)
    recs += run_cell("s4_paired_indef", "indoorcd_131", 80010, cfg,
                     0.215, [(0, "whiteout")], gh)

    print("=== rtb_clear (clear-air RTB arrival) ===", flush=True)
    cfg = dict(BASE_CFG, nav_trustworthy=True)
    recs += run_cell("rtb_clear", "indoorcd_071", 1, cfg, 0.26, None, gh)

    out_path = os.path.join(HERE, "belowclear_validation.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {out_path}")

    by = {(r["cell"], r["arm"]): r for r in recs}
    print("\n=== GATES ===")
    gates = {}
    s4off = by[("s4_clears", "off")]
    gates["s4_clears OFF reproduces TN legacy baseline @ a2cb278 "
          "(landed*/78 scans/vetoed51/terminal17/soc0.1014)"] = (
        s4off["outcome"] in LANDED
        and (s4off["outer_steps"], s4off["esc_land_vetoed"],
             s4off["safety_terminal_steps"], s4off["batt_final_soc"])
        == (78, 51, 17, 0.1014))
    sioff = by[("s4_paired_indef", "off")]
    gates["s4_paired_indef OFF reproduces TN legacy baseline"] = (
        sioff["outcome"] in LANDED
        and (sioff["outer_steps"], sioff["esc_land_vetoed"],
             sioff["safety_terminal_steps"], sioff["batt_final_soc"])
        == (78, 51, 17, 0.1014))
    rtoff = by[("rtb_clear", "off")]
    gates["rtb_clear OFF reproduces TN legacy baseline @ a2cb278 "
          "(landed/110 scans/vetoed28/terminal18/soc0.1016)"] = (
        rtoff["outcome"] in LANDED
        and (rtoff["outer_steps"], rtoff["esc_land_vetoed"],
             rtoff["safety_terminal_steps"], rtoff["batt_final_soc"])
        == (110, 28, 18, 0.1016))
    for cell in ("s4_clears", "s4_paired_indef", "rtb_clear"):
        on = by[(cell, "on")]
        off = by[(cell, "off")]
        gates[f"{cell} ON lands, 0 collisions, hold exercised, "
              f"vetoes bounded below OFF"] = (
            on["outcome"] in LANDED and on["collision"] == 0
            and on["esc_land_hold_episodes"] >= 1
            and on["esc_land_hold_expired_steps"] >= 1
            and 0 <= on["esc_land_vetoed"] < off["esc_land_vetoed"])
        print(f"{cell} ON: outcome={on['outcome']} "
              f"collision={on['collision']} steps={on['outer_steps']} "
              f"vetoed={on['esc_land_vetoed']} "
              f"episodes={on['esc_land_hold_episodes']} "
              f"expired_steps={on['esc_land_hold_expired_steps']} "
              f"max_hold_s={on['esc_land_hold_max_s']} "
              f"terminal={on['safety_terminal_steps']} "
              f"final_soc={on['batt_final_soc']}")
    ok_all = True
    for name, ok in gates.items():
        print(f"GATE {name}: {'PASS' if ok else 'FAIL'}", flush=True)
        ok_all = ok_all and ok
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
