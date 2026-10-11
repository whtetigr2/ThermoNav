"""Nav-health mission validation (roadmap Phase 1 item 3).

A. PAIRED NOMINAL (no-regression): clear-air + heavy-fog missions flown
   with escalation ON, nav-health OFF vs ON (nominal simulated estimator
   feed), same rooms/seeds/noise. Outcomes must be IDENTICAL per pair
   and the nominal feed must never reach UNTRUSTED. (Heavy fog is in
   FAST-LIO2's measured healthy band — ATE 0.08-0.25 m — so the nominal
   feed is the honest one there.)
B. SCRIPTED NAV FAILURE: one clear-air mission (perception healthy, D5
   silent) with the estimator report frozen from step 5 (12.5% motion
   ratio, the measured whiteout signature transplanted to clear air —
   an estimator fault, not a perception event). Required: health goes
   UNTRUSTED, the ladder takes the safe path (land-in-place via the
   gated RTH row, or a safe stand-down), NEVER success, NEVER collision,
   and RTH is never emitted (esc_final_action != RTH).

Dual-purpose CSV: every lever explicit + git commit.
Claim: host-side classical, validated in simulation.

ThermoNav port (graduation 2026-10-11): ported from the research
repo's runner (slow-drone/dynamics_sim/
run_nav_health_mission_validation.py) against the ported
sim/mission_dynamics_v5.py. Differences from the research run:
occupancy grids load from ../rooms; the paired-clear cells' jitter
spec is accepted by the cfg but ThermoNav's loop has no jitter
controller (map_age stays 0.0), so those cells run as clear air
without the jitter component — the gates (paired outcomes identical,
nominal feed never UNTRUSTED, scripted freeze never RTH / never
success / never collision) are unchanged. Gate wording is adapted
to ThermoNav semantics in exactly one place: a completed landing
ends as "landed" OR "landed_disarmed" (ThermoNav's DISARM wiring
fires after confirmed touchdown); both mean the vehicle is on the
ground.
"""
import csv
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exec_provenance  # noqa: E402
import mission_dynamics_v5 as md  # noqa: E402

OUT_PATH = os.path.join(HERE, "nav_health_mission_validation.csv")
PROV_NOTE = ("round-1 corrected re-issue: executed-tree commit verified "
             "by exec_provenance (original issue named launch-time HEAD "
             "f994011, a pre-latch tree on which the scripted arm "
             "succeeds — see NAV_HEALTH.md S7)")

NOISE = 0.15
BASE_CFG = {"fix_atten_d5": True, "fix_confirm": True, "fix_lowvis": True,
            "fix_stale": True, "escalation": True}
FULL_PHYSICS = {"gravity_envelope": True, "attitude_lag": True,
                "tw_ratio": 2.0, "filter_tau": 0.4,
                "wind_steady": 0.0, "wind_gust": False}
CLEAR_PAIRS = [("indoorcd_004", 0), ("indoorcd_071", 1),
               ("indoorcd_128", 1000), ("indoorcd_185", 2)]
FOG_PAIRS = [("indoorcd_117", 0), ("indoorcd_122", 1)]
FIELDS = ["cell", "mission_id", "room", "seed", "nav_enabled",
          "freeze_from_step", "outcome", "collision",
          "nav_final_health", "nav_untrusted_scans",
          "nav_degraded_scans", "esc_final_action", "esc_engaged_scans",
          "esc_rth_gated", "wall_s", "git_commit", "backend", "note"]


def run_one(cell, room, seed, nav, freeze_from, extra_cfg, commit):
    mid = f"navval_{cell}_{room}_s{seed}_nav{int(nav)}"
    if freeze_from is not None:
        mid += f"_frz{freeze_from}"
    occ = np.load(os.path.join(HERE, "..", "rooms", f"{room}_occ.npy"))
    cfg = dict(BASE_CFG)
    cfg.update(extra_cfg)
    cfg["nav_health"] = bool(nav)
    if freeze_from is not None:
        cfg["nav_health_freeze_from_step"] = int(freeze_from)
    t0 = time.time()
    note = ""
    try:
        row = md.run_mission_dynamics(occ, room, seed, NOISE, mid,
                                      backend="numpy", write_traj=False,
                                      cfg=cfg)
        outcome = row.get("outcome")
    except Exception as e:  # noqa: BLE001
        outcome = f"ERROR:{type(e).__name__}:{e}"
        row = {}
        note = traceback.format_exc(limit=3).replace("\n", " | ")
    rec = {"cell": cell, "mission_id": mid, "room": room, "seed": seed,
           "nav_enabled": int(nav),
           "freeze_from_step": -1 if freeze_from is None else freeze_from,
           "outcome": outcome,
           "collision": 1 if outcome == "collision" else 0,
           "nav_final_health": row.get("nav_final_health", ""),
           "nav_untrusted_scans": row.get("nav_untrusted_scans", -1),
           "nav_degraded_scans": row.get("nav_degraded_scans", -1),
           "esc_final_action": row.get("esc_final_action", "?"),
           "esc_engaged_scans": row.get("esc_engaged_scans", -1),
           "esc_rth_gated": row.get("esc_rth_gated", -1),
           "wall_s": round(time.time() - t0, 1), "git_commit": commit,
           "backend": "numpy", "note": note}
    print(f"  {mid}: {outcome} nav={rec['nav_final_health']} "
          f"untrusted={rec['nav_untrusted_scans']} "
          f"esc={rec['esc_final_action']} ({rec['wall_s']}s)", flush=True)
    return rec


def main():
    token = exec_provenance.begin(HERE, [OUT_PATH])
    commit = token
    print(f"executed-tree commit (verified): {commit}", flush=True)
    recs = []
    print("=== paired nominal: clear ===", flush=True)
    for room, seed in CLEAR_PAIRS:
        for nav in (False, True):
            recs.append(run_one("paired_clear", room, seed, nav, None,
                                {"jitter": {"mode": "periodic", "delay": 2.0,
                                            "every": 3, "start": 8,
                                            "until": 500}}, commit))
    print("=== paired nominal: heavy fog ===", flush=True)
    for room, seed in FOG_PAIRS:
        for nav in (False, True):
            cfg = dict(FULL_PHYSICS)
            cfg["obscurant"] = {"archetype": "fog", "density": "heavy"}
            recs.append(run_one("paired_fog", room, seed, nav, None,
                                cfg, commit))
    print("=== scripted nav failure: clear air, freeze from step 5 ===",
          flush=True)
    recs.append(run_one("scripted_freeze", "indoorcd_004", 0, True, 5,
                        {}, commit))

    commit = exec_provenance.finish(HERE, token, [OUT_PATH])
    for r in recs:
        r["git_commit"] = commit
        r["note"] = (r["note"] + " | " if r["note"] else "") + PROV_NOTE
    out = OUT_PATH
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)

    # ---- verdicts ----
    ok = True
    for cell in ("paired_clear", "paired_fog"):
        for room, seed in (CLEAR_PAIRS if cell == "paired_clear"
                           else FOG_PAIRS):
            pair = [r for r in recs if r["cell"] == cell
                    and r["room"] == room and r["seed"] == seed]
            off, on = pair[0], pair[1]
            same = off["outcome"] == on["outcome"]
            clean = on["nav_untrusted_scans"] == 0
            print(f"{cell} {room} s{seed}: OFF={off['outcome']} "
                  f"ON={on['outcome']} identical={same} "
                  f"nominal_never_untrusted={clean}", flush=True)
            ok = ok and same and clean
    fz = [r for r in recs if r["cell"] == "scripted_freeze"][0]
    fz_ok = (fz["nav_untrusted_scans"] > 0
             and fz["outcome"] in ("landed", "landed_disarmed",
                                   "blocked", "max_steps")
             and fz["esc_final_action"] != "RTH")
    print(f"scripted_freeze: outcome={fz['outcome']} "
          f"untrusted_scans={fz['nav_untrusted_scans']} "
          f"esc={fz['esc_final_action']} rth_gated={fz['esc_rth_gated']} "
          f"-> {'PASS' if fz_ok else 'FAIL'}", flush=True)
    ok = ok and fz_ok
    n_col = sum(r["collision"] for r in recs)
    print(f"RESULT: {len(recs)} missions, {n_col} collisions, "
          f"paired+scripted {'PASS' if ok and n_col == 0 else 'FAIL'}",
          flush=True)
    print(f"wrote {out}", flush=True)
    sys.exit(0 if (ok and n_col == 0) else 1)


if __name__ == "__main__":
    main()
