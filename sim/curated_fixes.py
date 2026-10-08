#!/usr/bin/env python3
"""Obscurant-fixes validation driver (Phase 5).

LEGO order: FIX 1 -> validate -> FIX 2 -> validate -> FIX 3 -> validate.
Each phase: numpy screening, then THRML finals for the load-bearing claims.

Phases:
  f1-numpy / f1-thrml : FIX 1 only (atten-aware D5)
  f2-numpy / f2-thrml : FIX 1+2 (add phantom confirmation)
  f3-numpy / f3-thrml : FIX 1+2+3 (add low-vis executive)
  zero                : zero-solution missions, all fixes, {clear, heavy}

Gates:
  REGRESSION: Phase-4 clear set (8 graded) must not get worse than 8/8.
  FIX GATE:   Phase-4 whiteout set — the 6 THRML collisions must become safe
              (success/goal_blocked/max_steps/hover all OK; collisions not).
  ZERO-SOL:   must not get worse than Phase-4 baseline
              (clear: 1 collision + 1 max_steps; heavy: 1 goal_blocked + 1 collision).
  FALSE-POS:  FIX 3 must not hover on clear/heavy/moderate (report rate).

Mission IDs: pfix_<tag>_<room>_s<seed>_n<noise>_s16 (traj_fix/ demo contract).

Run examples:
  python3 curated_fixes.py --phase f1-numpy --backend numpy
  XLA_PYTHON_CLIENT_PREALLOCATE=false python3 curated_fixes.py --phase f3-thrml --backend thrml
"""
import os
import sys
import time
import json
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission_dynamics_v5 as md
from mission_dynamics_v5 import MANIFEST_FIELDS_V5
from curated_v2 import GRADED, make_zero_enclosed, make_zero_partition

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
HERE = os.path.dirname(os.path.abspath(__file__))

FIX_SETS = {
    "f1": {"fix_atten_d5": True, "fix_confirm": False, "fix_lowvis": False},
    "f2": {"fix_atten_d5": True, "fix_confirm": True, "fix_lowvis": False},
    "f3": {"fix_atten_d5": True, "fix_confirm": True, "fix_lowvis": True},
}


def base_cfg():
    return {"gravity_envelope": True, "attitude_lag": True,
            "tw_ratio": 2.0, "filter_tau": 0.4,
            "wind_steady": 0.0, "wind_gust": False}


def run_set(missions, densities, fix_tag, backend, out_json, manifest_path,
            label):
    fixes = dict(FIX_SETS[fix_tag])
    results = []
    manifest_rows = []
    _orig_psg = md.pick_start_goal
    if os.path.exists(out_json):
        try:
            results = json.load(open(out_json)).get("results", [])
        except Exception:  # noqa: BLE001
            results = []
    done_ids = {r["mission"] for r in results}
    new_file = not os.path.exists(manifest_path)
    if new_file:
        with open(manifest_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS_V5).writeheader()
    total = len(missions) * len(densities)
    done_n = 0

    def flush():
        with open(out_json, "w") as f:
            json.dump({"backend": backend, "label": label, "fixes": fixes,
                       "results": results}, f, indent=1)
        with open(manifest_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS_V5)
            for r in manifest_rows:
                w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS_V5})
        manifest_rows.clear()

    for density in densities:
        for occ, room, seed, noise, mid_base, info, sp, gp in missions:
            mid = f"pfix_{fix_tag}_{density}_{mid_base}"
            done_n += 1
            if mid in done_ids:
                print(f"[{done_n}/{total}] {mid}: SKIPPED", flush=True)
                continue
            cfg = base_cfg()
            cfg.update(fixes)
            cfg["obscurant"] = {"archetype": "fog", "density": density}
            cfg["config_name"] = f"fix_{fix_tag}_{density}"
            t0 = time.time()
            if sp is not None:
                md.pick_start_goal = (lambda o, s, _sp=sp, _gp=gp: (_sp, _gp))
            try:
                row = md.run_mission_dynamics(
                    occ, room, seed, noise, mid, backend=backend,
                    verbose=False, cfg=dict(cfg))
                outcome = row["outcome"]
                row["tier"] = info.get("tier")
                manifest_rows.append(row)
            except Exception as e:  # noqa: BLE001
                outcome = f"ERROR:{type(e).__name__}:{e}"
                row = {}
            finally:
                md.pick_start_goal = _orig_psg
            dt = time.time() - t0
            rec = {"mission": mid, "density": density, "tier": info.get("tier"),
                   "outcome": outcome, "n_steps": row.get("n_steps", -1),
                   "n_hover": row.get("n_hover_scans", -1),
                   "vis_min": row.get("vis_min", -1),
                   "wall_s": round(dt, 1)}
            results.append(rec)
            print(f"[{done_n}/{total}] {mid}: {outcome} "
                  f"steps={rec['n_steps']} hover={rec['n_hover']} "
                  f"vis_min={rec['vis_min']} ({dt:.0f}s)", flush=True)
            flush()
            done_ids.add(mid)
    n_ok = sum(1 for r in results if r["outcome"] == "success")
    n_coll = sum(1 for r in results if r["outcome"] == "collision")
    n_hover_any = sum(1 for r in results
                      if isinstance(r["n_hover"], int) and r["n_hover"] > 0)
    print(f"=== {label}: {n_ok}/{len(results)} success, {n_coll} collisions, "
          f"{n_hover_any} hovered [{backend}] ===")
    flush()
    return results


def load_missions():
    missions = []
    for room, seed, noise in GRADED:
        occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
        mid = f"{room}_s{seed}_n{noise:.2f}_s16"
        missions.append((occ, room, seed, noise, mid, {"tier": "graded"},
                         None, None))
    zero_missions = []
    for maker in (make_zero_enclosed, make_zero_partition):
        occ, room, seed, noise, mid, info, sp, gp = maker()
        info["tier"] = "zero-solution"
        zero_missions.append((occ, room, seed, noise, mid, info, sp, gp))
    return missions, zero_missions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["f1-numpy", "f1-thrml", "f2-numpy", "f2-thrml",
                             "f3-numpy", "f3-thrml", "zero"])
    ap.add_argument("--backend", default="numpy", choices=["thrml", "numpy"])
    ap.add_argument("--densities", default=None,
                    help="comma-separated subset")
    args = ap.parse_args()

    missions, zero_missions = load_missions()
    mpath = os.path.join(HERE, "RUN_MANIFEST_FIXES.csv")
    fix_tag = args.phase.split("-")[0]

    if args.phase == "zero":
        dens = ["clear", "heavy"]
        out = os.path.join(HERE, f"curated_fixes_zero_{args.backend}.json")
        run_set(zero_missions, dens, "f3", args.backend, out, mpath,
                "fixes zero-sol")
    else:
        dens = (args.densities.split(",") if args.densities
                else ["clear", "whiteout"])
        out = os.path.join(HERE,
                           f"curated_fixes_{args.phase}_{args.backend}.json")
        run_set(missions, dens, fix_tag, args.backend, out, mpath,
                f"fixes {args.phase}")


if __name__ == "__main__":
    main()
