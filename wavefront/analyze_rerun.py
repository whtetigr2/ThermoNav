#!/usr/bin/env python3
"""R3 analysis — recompute the 2n correlations from the SAVED fields.

Answers, from runs_rerun/*.npz only (never from live sampling):
  (a) 2D flip-time/BFS correlation, replicated published pipeline;
  (b) raw-vs-host-smoothed attribution for the published 0.999;
  (c) the 3D flip-time/BFS correlation, measured for the first time
      (the 2026-10-17 code contains only a comment; U1-F6).

Definitions frozen in PLAN_RERUN.md: cell set = free + BFS-reachable;
field variants mean/median x raw/smoothed; Pearson (published metric)
+ Spearman (flip-ORDER metric). smooth_field / BFS below are faithful
copies of the original sim functions; the bar-1 self-check (saved
fields -> analysis corr == the patched sim's own printed bar-1 corr)
is the proof the copies are faithful.

Usage: analyze_rerun.py <fields_dir> <sim_results_json> <out_dir>
       --commit <git hash recorded in the CSV header>
"""
import sys, os, json, argparse
import numpy as np
from collections import deque


def smooth_field(t, obs_set, ws):
    """Faithful copy of sim_2n_wave.smooth_field (3x3(x3) mean, no inf)."""
    dim = len(ws)
    ts = t.copy()
    if dim == 2:
        W, H = ws
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if (x, y) in obs_set:
                    continue
                w = t[y - 1:y + 2, x - 1:x + 2]
                f = w[np.isfinite(w)]
                if len(f):
                    ts[y, x] = f.mean()
    else:
        W, H, D = ws
        for z in range(1, D - 1):
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if (x, y, z) in obs_set:
                        continue
                    w = t[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
                    f = w[np.isfinite(w)]
                    if len(f):
                        ts[z, y, x] = f.mean()
    return ts


def bfs_dist(obs_set, goal, ws):
    dim = len(ws)
    if dim == 2:
        W, H = ws
        dist = np.full((H, W), -1)
        dq = deque([goal])
        dist[goal[1], goal[0]] = 0
        while dq:
            x, y = dq.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in obs_set \
                        and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    dq.append((nx, ny))
        return dist
    W, H, D = ws
    dist = np.full((D, H, W), -1)
    dq = deque([goal])
    dist[goal[2], goal[1], goal[0]] = 0
    while dq:
        x, y, z = dq.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D \
                    and (nx, ny, nz) not in obs_set \
                    and dist[nz, ny, nx] < 0:
                dist[nz, ny, nx] = dist[z, y, x] + 1
                dq.append((nx, ny, nz))
    return dist


def spearman(a, b):
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        i = 0
        sv = v[order]
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def analyze_case(path):
    z = np.load(path, allow_pickle=False)
    ws = tuple(int(v) for v in z["world_shape"])
    dim = len(ws)
    obs_mask = z["obs_mask"]
    goal = tuple(int(v) for v in z["goal"])
    n_sweeps = int(z["n_sweeps"])
    if dim == 2:
        obs_set = {(x, y) for y in range(ws[1]) for x in range(ws[0])
                   if obs_mask[y, x]}
    else:
        obs_set = {(x, y, zz) for zz in range(ws[2]) for y in range(ws[1])
                   for x in range(ws[0]) if obs_mask[zz, y, x]}
    dist = bfs_dist(obs_set, goal, ws)
    mask = (~obs_mask) & (dist >= 0)
    cube = z["flip_cube"]  # halo coords, per-chain
    if dim == 2:
        inner = cube[:, 1:-1, 1:-1]
    else:
        inner = cube[:, 1:-1, 1:-1, 1:-1]
    never = int(((inner == n_sweeps + 1).all(axis=0) & mask).sum())
    rows = []
    fields = {
        "mean_raw": z["tmean_world"],
        "mean_smoothed": smooth_field(z["tmean_world"], obs_set, ws),
        "median_raw": z["tmed_world"],
        "median_smoothed": smooth_field(z["tmed_world"], obs_set, ws),
    }
    dm = dist[mask].astype(float)
    for variant, f in fields.items():
        vals = f[mask]
        fin = np.isfinite(vals)
        v, d = vals[fin], dm[fin]
        rows.append({
            "field_variant": variant, "n_cells": int(mask.sum()),
            "n_finite": int(fin.sum()),
            "pearson": float(np.corrcoef(v, d)[0, 1]),
            "spearman": spearman(v, d),
            "never_flipped_cells": never,
        })
    meta = {k: (z[k].item() if z[k].ndim == 0 else str(z[k]))
            for k in ("J", "H_FREE", "H_OBS", "beta", "n_sweeps",
                      "n_chains", "prog_hash", "code_sha256",
                      "thrml_version", "jax_version", "numpy_version")}
    meta["goal"] = goal
    meta["start"] = [int(v) for v in z["start"]]
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fields_dir")
    ap.add_argument("sim_results")
    ap.add_argument("out_dir")
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sim = json.load(open(a.sim_results))
    cases = ["bar1_2d", "bar4_j0", "maze0_3d", "maze1_3d", "maze2_3d"]
    all_rows, metas = [], {}
    per = {}
    for c in cases:
        rows, meta = analyze_case(os.path.join(a.fields_dir, c + ".npz"))
        metas[c] = meta
        per[c] = {r["field_variant"]: r for r in rows}
        for r in rows:
            all_rows.append({"case": c, "dim": 2 if c.endswith("2d") or
                             c == "bar4_j0" else 3, **r})

    # --- self-check: saved fields -> analysis == sim's own bar-1 corr
    selfcheck = abs(per["bar1_2d"]["mean_smoothed"]["pearson"]
                    - sim["bar1"]["corr"])
    if selfcheck > 1e-9:
        print("SELF-CHECK FAIL: analysis bar1 mean_smoothed pearson "
              "differs from sim bar1 corr by %g" % selfcheck)
        sys.exit(1)
    print("self-check OK: |analysis - sim bar1 corr| = %.2e" % selfcheck)

    # --- verdicts (rules frozen in PLAN_RERUN.md)
    sm = per["bar1_2d"]["mean_smoothed"]["pearson"]
    rw = per["bar1_2d"]["mean_raw"]["pearson"]
    if sm >= 0.995 and rw >= 0.95:
        c1 = "CONFIRMED"
    elif sm < 0.95:
        c1 = "DEAD"
    else:
        c1 = "CORRECTED"
    m3 = [per["maze%d_3d" % i]["mean_smoothed"]["pearson"] for i in range(3)]
    m3_mean = float(np.mean(m3))
    c2 = "CONSISTENT" if abs(m3_mean - 0.981) <= 0.03 else "CORRECTED"
    b4 = sim["bar4"]
    steps = [m["steps"] for m in sim["mazes"]]
    reached = [m["reached"] for m in sim["mazes"]]
    if not b4["reached"] and abs(b4["field_std"] - 0.04283522131518765) \
            <= 0.02 and steps == [65, 65, 83] and all(reached):
        c3 = "CONFIRMED"
    elif not b4["reached"] and all(reached):
        c3 = "CORRECTED"
    else:
        c3 = "DEAD"
    verdicts = {
        "C1_2d_headline": {"verdict": c1,
                           "published": 0.9991180978469811,
                           "rerun_mean_smoothed_pearson": sm,
                           "rerun_mean_raw_pearson": rw,
                           "rerun_median_raw_pearson":
                           per["bar1_2d"]["median_raw"]["pearson"],
                           "rerun_mean_raw_spearman":
                           per["bar1_2d"]["mean_raw"]["spearman"],
                           "rerun_mean_smoothed_spearman":
                           per["bar1_2d"]["mean_smoothed"]["spearman"]},
        "C2_3d_headline": {"verdict": c2,
                           "as_published": "UNVERIFIABLE-AS-PUBLISHED",
                           "published": 0.981,
                           "rerun_per_maze_mean_smoothed_pearson": m3,
                           "rerun_mean_of_mazes": m3_mean,
                           "rerun_per_maze_mean_raw_pearson":
                           [per["maze%d_3d" % i]["mean_raw"]["pearson"]
                            for i in range(3)]},
        "C3_ablation_and_mazes": {"verdict": c3,
                                  "published_std": 0.04283522131518765,
                                  "rerun_std": b4["field_std"],
                                  "rerun_ablation_reached": b4["reached"],
                                  "published_steps": [65, 65, 83],
                                  "rerun_steps": steps,
                                  "rerun_reached": reached},
    }
    with open(os.path.join(a.out_dir, "summary_rerun.csv"), "w") as f:
        f.write("# R3 2n verifiability re-run — correlation summary\n")
        f.write("# commit: %s\n" % a.commit)
        f.write("# source artifacts: runs_rerun/*.npz (saved fields); "
                "analysis: lego2n/analyze_rerun.py; plan: "
                "lego2n/PLAN_RERUN.md\n")
        f.write("case,dim,field_variant,metric_or_count,n_cells,n_finite,"
                "pearson,spearman,never_flipped_cells\n")
        for r in all_rows:
            f.write("%s,%d,%s,,%d,%d,%.10f,%.10f,%d\n" % (
                r["case"], r["dim"], r["field_variant"], r["n_cells"],
                r["n_finite"], r["pearson"], r["spearman"],
                r["never_flipped_cells"]))
    with open(os.path.join(a.out_dir, "analysis_rerun.json"), "w") as f:
        json.dump({"verdicts": verdicts, "rows": all_rows,
                   "meta": {k: {kk: str(vv) for kk, vv in v.items()}
                            for k, v in metas.items()},
                   "self_check_abs_diff": selfcheck}, f, indent=1)
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
