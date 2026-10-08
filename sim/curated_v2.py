"""Curated Phase-2 test set (Paul's change of plan — replaces the full 240).

8 graded missions (2 easy / 2 medium / 2 hard / 2 damn-near-impossible),
picked from the V6 manifest on measured difficulty signals (path ratio,
steps, D5 refusals, escapes), PLUS 2 zero-solution missions where the goal
is genuinely unreachable (verified by flood fill).

THE KEY QUESTION: is the stack smart enough to NOT go when the endpoint is
blocked? Expected smart behavior: detect impossibility, abort gracefully
(no_path/blocked) — not wander to max_steps, NEVER collide. Whatever it
actually does is the finding; this script does not patch anything.

Run: python3 curated_v2.py [--backend thrml|numpy]
Writes: curated_v2_results.json, traj2/<mission_id>.json (demo contract).
"""
import os
import sys
import time
import json
import argparse
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission_dynamics_v2 as md

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_lidar")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "curated_v2_results.json")

# (room, seed, noise) — canonical mission IDs, paired with V6/Phase 1.
GRADED = [
    # EASY — clean, near-optimal in V6
    ("indoorcd_071", 2, 0.15),
    ("indoorcd_004", 0, 0.15),
    # MEDIUM — 1.00x ratio, moderate length
    ("indoorcd_131", 1, 0.15),
    ("indoorcd_185", 2, 0.15),
    # HARD — triage outliers
    ("indoorcd_128", 0, 0.30),   # worst V6 path ratio 3.61x
    ("indoorcd_071", 1, 0.30),   # 3.08x, 9 D5 refusals, 1 escape
    # DAMN-NEAR-IMPOSSIBLE
    ("indoorcd_128", 3, 0.30),   # 2.80x, 10 D5 refusals
    ("indoorcd_071", 1, 0.15),   # 2.83x, 7 refusals, 2 escapes
]


def flood_fill_free(occ, start_v):
    """BFS over free voxels from start. Returns set of reached voxels."""
    occ = np.asarray(occ)
    shape = occ.shape
    seen = set()
    dq = deque([tuple(int(c) for c in start_v)])
    while dq:
        v = dq.popleft()
        if v in seen:
            continue
        if not (0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]
                and 0 <= v[2] < shape[2]):
            continue
        if bool(occ[v]):
            continue
        seen.add(v)
        dq.extend([(v[0] + 1, v[1], v[2]), (v[0] - 1, v[1], v[2]),
                   (v[0], v[1] + 1, v[2]), (v[0], v[1] - 1, v[2]),
                   (v[0], v[1], v[2] + 1), (v[0], v[1], v[2] - 1)])
    return seen


def make_zero_enclosed():
    """Zero-solution 1: goal voxel sealed inside a 3x3x3 occupied shell.
    Room indoorcd_004 (otherwise an EASY mission) — the only change is the
    shell. Returns (occ, room_id, seed, noise, mission_id)."""
    import planner
    from fly import pick_start_goal
    room_id, seed, noise = "indoorcd_004", 0, 0.15
    occ = np.load(os.path.join(REAL_DIR, f"{room_id}_occ.npy")).copy()
    start_pose, goal_pose = pick_start_goal(occ, seed)
    gv = tuple(int(c) for c in planner.pose_to_voxel(goal_pose, occ.shape))
    sv = tuple(int(c) for c in planner.pose_to_voxel(start_pose, occ.shape))
    # seal the 26 neighbors (goal voxel itself stays free)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                w = (gv[0] + dx, gv[1] + dy, gv[2] + dz)
                if 0 <= w[0] < 32 and 0 <= w[1] < 32 and 0 <= w[2] < 16:
                    occ[w] = 1
    # VERIFY: flood fill from start must NOT reach the goal voxel
    reached = flood_fill_free(occ, sv)
    assert gv not in reached, "ZERO-SOLUTION BROKEN: goal reachable!"
    assert len(reached) > 100, "start region trivially small"
    assert not bool(occ[sv]), "start voxel blocked"
    # NOTE: the mission loop's pick_start_goal enforces GT-feasibility and
    # would SILENTLY RE-PICK a reachable goal on this modified grid. The
    # constructor's start/goal are returned so the driver can pin them.
    return (occ, room_id, seed, noise, "zero_enclosed_indoorcd_004_s0",
            {"shell_center": gv, "start_reachable": len(reached)},
            start_pose, goal_pose)


def make_zero_partition():
    """Zero-solution 2: a full YZ partition wall bisects the room; start
    and goal end up on opposite sides. Returns (occ, room_id, seed, noise,
    mission_id)."""
    import planner
    from fly import pick_start_goal
    room_id, noise = "indoorcd_128", 0.15
    base = np.load(os.path.join(REAL_DIR, f"{room_id}_occ.npy"))
    for seed in range(8):
        occ = base.copy()
        start_pose, goal_pose = pick_start_goal(occ, seed)
        sv = tuple(int(c) for c in planner.pose_to_voxel(start_pose, occ.shape))
        gv = tuple(int(c) for c in planner.pose_to_voxel(goal_pose, occ.shape))
        # partition plane between them in x (leave z=15 open? NO — full
        # partition: the point is unsolvability)
        xw = (sv[0] + gv[0]) // 2
        if xw <= 0 or xw >= 31 or xw == sv[0] or xw == gv[0]:
            continue
        occ[xw, :, :] = 1
        # carve back the start/goal voxels if the wall covered them
        # (it can't — xw differs from both — but be safe)
        occ[sv] = 0
        occ[gv] = 0
        reached = flood_fill_free(occ, sv)
        if gv not in reached and len(reached) > 100 and not bool(occ[sv]):
            return (occ, room_id, seed, noise,
                    f"zero_partition_indoorcd_128_s{seed}",
                    {"wall_x": xw, "start_reachable": len(reached)},
                    start_pose, goal_pose)
    raise RuntimeError("could not construct a partitioned zero-solution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="thrml", choices=["thrml", "numpy"])
    ap.add_argument("--only-zero", action="store_true",
                    help="re-run only the zero-solution missions")
    args = ap.parse_args()

    import planner  # noqa: F401  (ensures pipeline importable)

    missions = []
    for room, seed, noise in GRADED:
        occ = np.load(os.path.join(REAL_DIR, f"{room}_occ.npy"))
        mid = f"{room}_s{seed}_n{noise:.2f}_s16"
        missions.append((occ, room, seed, noise, mid, {"tier": "graded"},
                         None, None))

    for maker in (make_zero_enclosed, make_zero_partition):
        occ, room, seed, noise, mid, info, sp, gp = maker()
        info["tier"] = "zero-solution"
        print(f"constructed {mid}: {info}", flush=True)
        missions.append((occ, room, seed, noise, mid, info, sp, gp))

    if args.only_zero:
        missions = [m for m in missions if m[5].get("tier") == "zero-solution"]

    results = []
    _orig_psg = md.pick_start_goal
    for occ, room, seed, noise, mid, info, sp, gp in missions:
        t0 = time.time()
        # Pin the constructor's start/goal for zero-solution missions:
        # the loop's pick_start_goal enforces GT-feasibility and would
        # otherwise silently re-pick a reachable goal, invalidating the test.
        if sp is not None:
            md.pick_start_goal = (lambda o, s, _sp=sp, _gp=gp:
                                  (_sp, _gp))
        try:
            row = md.run_mission_dynamics(
                occ, room, seed, noise, mid, backend=args.backend,
                verbose=False)
            outcome = row["outcome"]
        except Exception as e:  # noqa: BLE001 — record, don't crash the set
            outcome = f"ERROR:{type(e).__name__}:{e}"
            row = {}
        finally:
            md.pick_start_goal = _orig_psg
        dt = time.time() - t0
        rec = {"mission": mid, "tier": info.get("tier"),
               "outcome": outcome,
               "n_steps": row.get("n_steps", -1),
               "wall_s": round(dt, 1), "info": info}
        results.append(rec)
        print(f"{mid} [{info.get('tier')}]: {outcome} "
              f"steps={rec['n_steps']} ({dt:.0f}s)", flush=True)

    n_ok = sum(1 for r in results if r["outcome"] == "success")
    n_coll = sum(1 for r in results if r["outcome"] == "collision")
    print(f"=== {n_ok}/{len(results)} success, {n_coll} collisions "
          f"[{args.backend}] ===")
    with open(OUT, "w") as f:
        json.dump({"backend": args.backend, "results": results}, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
