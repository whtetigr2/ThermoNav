"""Dynamic-layer v3 sweep (ThermoNav port): paired OFF vs ON mover missions.

Phase 2(a) roadmap burn-down (2026-10-11). Same protocol as
run_dynamic_sweep.py / run_motion_sweep.py / run_dodge_sweep.py: same
4 (room, seed) pairs, same staged encounters (traj_fix baselines),
same speeds/geometries, same noise — so the OFF arm must reproduce
the 43/48 no-layer baseline and the ON arm is directly comparable to
motion-only (30/48) and motion+dodge (26/48).

Arms:
  OFF: no dynamic layer (established baseline configuration).
  ON:  cfg dyn_layer=True (dynlayer.DynLayer in the _MOTION slot) +
       dodge_layer=True (unchanged DodgeController on top).

PROVENANCE (exec_provenance discipline, nav-health lesson): git_commit
is resolved from THIS checkout (git -C <this file's dir>) — the commit
that actually executed — not from any other checkout of the repo.
The sweep refuses to run on a dirty tree, so the recorded commit is
exactly the code that ran.

DUAL-PURPOSE DATA DOCTRINE: every lever is an explicit column.

Claim label: Z1-legal formulation, validated in simulation (host-side
classical layer; no TSU claim).

ThermoNav port (dynlayer graduation, 2026-10-12): ported from the
research repo's runner (slow-drone/dynamics_sim/run_dynlayer_sweep.py)
against the ported sim/mission_dynamics_v5.py. Differences from the
research run:
  * occupancy grids load from ../rooms (ThermoNav layout);
  * staging trajectories are ThermoNav-native: sim/traj_fix/
    dyn_baseline_<room>_s<seed>.json, flown by THIS loop (no-mover,
    OFF cfg, write_traj=True) and committed as staging inputs,
    because ThermoNav's loop has diverged from the research loop
    (DISARM / battery / nav wiring; no jitter controller);
  * provenance uses ThermoNav's exec_provenance guard (begin/finish)
    instead of the research runner's local clean-tree check;
  * sim_time_s is read from the row's sim_time field, which the
    ThermoNav mission loop now emits (research RT-2 S7: the column
    was -1 in all 152 research v3 rows because manifest_row never
    emits sim_time — fixed at the source here, CSVs not rewritten
    on the research side);
  * an arm argument writes dynlayer_sweep_<arm>.csv so the two arms
    can run as parallel shards; merge_shards() assembles
    dynlayer_sweep.csv from the shards.
"""

import os
import sys
import csv
import time
import json
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exec_provenance
import mission_dynamics_v5 as md
from movers import stage_encounter, MOVER_RADIUS
import dynlayer as dyn_mod
import dodge as dodge_mod

ROOMS_DIR = os.path.join(HERE, "..", "rooms")
OUT = os.path.join(HERE, "dynlayer_sweep.csv")
TRAJ_DIR = os.path.join(HERE, "traj_fix")

FIELDS = ["mission_id", "room", "seed", "cell", "arm", "mover_geometry",
          "mover_speed", "mover_radius", "outcome", "collision",
          "mover_min_dist", "n_move_refusals", "n_motion_refusals",
          "n_motion_events", "n_motion_confirmed",
          "n_dyn_prev_events", "n_dyn_map_events",
          "n_dyn_tracks_confirmed", "n_dyn_tracks_active_end",
          "n_dyn_tracks_full", "n_dyn_wall_promoted",
          "n_dodges", "n_dodge_fallbacks", "n_dodge_wall_incidents",
          "n_dodge_motion_blocks", "n_dodge_substeps",
          "sim_time_s", "path_efficiency", "noise", "fog_density",
          "dyn_gate_m", "dyn_confirm_hits", "dyn_track_ttl_s",
          "dyn_block_radius", "dyn_proj_horizon_s",
          "dyn_free_thresh", "dyn_occ_lo",
          "dyn_speed_envelope", "dyn_full_hits", "dyn_full_gate_m",
          "dyn_wall_min_move_m", "dyn_appear_decay_s",
          "dodge_threat_radius", "dodge_proj_t",
          "wall_s", "git_commit", "backend", "claim"]

FULL_PHYSICS = {"gravity_envelope": True, "attitude_lag": True,
                "tw_ratio": 2.0, "filter_tau": 0.4,
                "wind_steady": 0.0, "wind_gust": False}
FIXES = {"fix_atten_d5": True, "fix_confirm": True, "fix_lowvis": True,
         "fix_stale": True}
NOISE = 0.15
BACKEND = "numpy"
SPEEDS = [0.5, 1.0, 1.5, 2.0]
GEOMETRIES = ["crossing", "head-on", "overtake"]
CLAIM = "Z1-legal formulation, validated in simulation"
PAIRS = [("indoorcd_004", 0), ("indoorcd_071", 1),
         ("indoorcd_128", 0), ("indoorcd_131", 1)]


ALL_OUTPUTS = [os.path.join(HERE, n) for n in (
    "dynlayer_sweep.csv", "dynlayer_sweep_off.csv",
    "dynlayer_sweep_on.csv", "dynlayer_sweep.log")]


def merge_shards():
    """Assemble dynlayer_sweep.csv from the per-arm shards."""
    rows = []
    for arm in ("off", "on"):
        sp = os.path.join(HERE, f"dynlayer_sweep_{arm}.csv")
        if os.path.exists(sp):
            with open(sp) as f:
                rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (r["arm"], r["mission_id"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows -> {OUT}", flush=True)


def cfg_for(arm):
    cfg = dict(FIXES)
    cfg.update(FULL_PHYSICS)
    if arm == "on":
        cfg["dyn_layer"] = True
        cfg["dodge_layer"] = True
    return cfg


def run_one(occ, room, seed, mid, cfg):
    t0 = time.time()
    try:
        row = md.run_mission_dynamics(
            occ, room, seed, NOISE, mid, backend=BACKEND,
            write_traj=False, verbose=False, cfg=dict(cfg))
        outcome = row["outcome"]
    except Exception as e:  # noqa: BLE001 - record, don't crash
        outcome = f"ERROR:{type(e).__name__}:{e}"
        row = {}
    return outcome, row, time.time() - t0


def rec(mid, room, seed, cell, arm, outcome, row, dt, git_hash):
    return {
        "mission_id": mid, "room": room, "seed": seed, "cell": cell,
        "arm": arm,
        "mover_geometry": row.get("mover_geometry", ""),
        "mover_speed": row.get("mover_speed", 0.0),
        "mover_radius": MOVER_RADIUS if row.get("mover_geometry") else 0.0,
        "outcome": outcome,
        "collision": 1 if outcome == "collision" else 0,
        "mover_min_dist": row.get("mover_min_dist", -1.0),
        "n_move_refusals": row.get("n_move_refusals", -1),
        "n_motion_refusals": row.get("n_motion_refusals", -1),
        "n_motion_events": row.get("n_motion_events", -1),
        "n_motion_confirmed": row.get("n_motion_confirmed", -1),
        "n_dyn_prev_events": row.get("n_dyn_prev_events", -1),
        "n_dyn_map_events": row.get("n_dyn_map_events", -1),
        # F5: lifetime count (column name kept; semantics repaired —
        # previously an end-of-mission active-track snapshot).
        "n_dyn_tracks_confirmed": row.get("n_dyn_tracks_confirmed", -1),
        "n_dyn_tracks_active_end": row.get("n_dyn_tracks_active_end", -1),
        "n_dyn_tracks_full": row.get("n_dyn_tracks_full", -1),
        "n_dyn_wall_promoted": row.get("n_dyn_wall_promoted", -1),
        "n_dodges": row.get("n_dodges", -1),
        "n_dodge_fallbacks": row.get("n_dodge_fallbacks", -1),
        "n_dodge_wall_incidents": row.get("n_dodge_wall_incidents", -1),
        "n_dodge_motion_blocks": row.get("n_dodge_motion_blocks", -1),
        "n_dodge_substeps": row.get("n_dodge_substeps", -1),
        "sim_time_s": row.get("sim_time", -1),
        "path_efficiency": row.get("path_efficiency", ""),
        "noise": NOISE, "fog_density": "clear",
        "dyn_gate_m": dyn_mod.GATE_M, "dyn_confirm_hits": dyn_mod.CONFIRM_HITS,
        "dyn_track_ttl_s": dyn_mod.TRACK_TTL_S,
        "dyn_block_radius": dyn_mod.BLOCK_RADIUS,
        "dyn_proj_horizon_s": dyn_mod.PROJ_HORIZON_S,
        "dyn_free_thresh": dyn_mod.FREE_THRESH, "dyn_occ_lo": dyn_mod.OCC_LO,
        "dyn_speed_envelope": dyn_mod.SPEED_ENVELOPE,
        "dyn_full_hits": dyn_mod.FULL_HITS,
        "dyn_full_gate_m": dyn_mod.FULL_GATE_M,
        "dyn_wall_min_move_m": dyn_mod.WALL_MIN_MOVE_M,
        "dyn_appear_decay_s": dyn_mod.APPEAR_DECAY_S,
        "dodge_threat_radius": dodge_mod.THREAT_RADIUS,
        "dodge_proj_t": dodge_mod.PROJ_T,
        "wall_s": round(dt, 1), "git_commit": git_hash,
        "backend": BACKEND, "claim": CLAIM,
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_shards()
        return
    token = exec_provenance.begin(HERE, ALL_OUTPUTS)
    git_hash = token
    only_arm = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = os.path.join(
        HERE, f"dynlayer_sweep_{only_arm}.csv") if only_arm else OUT
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f):
                done.add(r["mission_id"])
    fout = open(out_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if not done:
        writer.writeheader()

    for arm in ("off", "on"):
        if only_arm and arm != only_arm:
            continue
        # Phase 0: no-mover baselines (static regression + FP cost).
        for room, seed in PAIRS:
            mid = f"dyn2_{arm}_baseline_{room}_s{seed}"
            if mid in done:
                continue
            occ = np.load(os.path.join(ROOMS_DIR, f"{room}_occ.npy"))
            outcome, row, dt = run_one(occ, room, seed, mid, cfg_for(arm))
            writer.writerow(rec(mid, room, seed, "baseline", arm,
                                outcome, row, dt, git_hash))
            fout.flush()
            print(f"[{arm} baseline] {room} s{seed}: {outcome} "
                  f"events={row.get('n_motion_events')} "
                  f"mref={row.get('n_motion_refusals')} "
                  f"dodges={row.get('n_dodges')} ({dt:.0f}s)", flush=True)
        # Phase 1: mover matrix (clear air), paired across arms.
        for room, seed in PAIRS:
            occ = np.load(os.path.join(ROOMS_DIR, f"{room}_occ.npy"))
            tpath = os.path.join(TRAJ_DIR, f"dyn_baseline_{room}_s{seed}.json")
            if not os.path.exists(tpath):
                print(f"[skip] no baseline traj for {room} s{seed}",
                      flush=True)
                continue
            with open(tpath) as f:
                doc = json.load(f)
            traj = np.array(doc["xyz"])
            dt_samp = float(doc.get("dt", 0.05))
            for geom in GEOMETRIES:
                for speed in SPEEDS:
                    mid = f"dyn2_{arm}_{geom}_{room}_s{seed}_v{speed}"
                    if mid in done:
                        continue
                    spec = stage_encounter(occ, traj, dt_samp, speed, geom)
                    if spec is None:
                        print(f"[skip] staging failed {mid}", flush=True)
                        continue
                    cfg = cfg_for(arm)
                    cfg["movers"] = [spec]
                    outcome, row, dt = run_one(occ, room, seed, mid, cfg)
                    writer.writerow(rec(mid, room, seed, geom, arm,
                                        outcome, row, dt, git_hash))
                    fout.flush()
                    print(f"[{arm} {geom} v={speed}] {room} s{seed}: "
                          f"{outcome} min_dist={row.get('mover_min_dist')} "
                          f"mref={row.get('n_motion_refusals')} "
                          f"trk={row.get('n_dyn_tracks_confirmed')} "
                          f"dodges={row.get('n_dodges')} ({dt:.0f}s)",
                          flush=True)
    fout.close()
    exec_provenance.finish(HERE, token, ALL_OUTPUTS)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
