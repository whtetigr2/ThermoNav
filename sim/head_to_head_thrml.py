"""Head-to-head V5-THRML: S16 missions with BOTH TSU workloads on THRML.

Mirrors head_to_head_v2.py::run_mission_s16 EXACTLY, except:
  - perception Gibbs -> THRML (verified mappings from thrml_port/:
      b = h/2, J = 0.125/edge, beta = 1.0, 2-block chromatic, all-False cold
      init, x_last warm init after step 0)
  - S2 decisions    -> THRML (b = h, J = J, beta = 1.0, 18 sequential
      single-spin blocks; same greedy fallback as S2Agent.act)

Everything else (sensor model, host A*, D5 guards, controller, S16Planner
logic, gate modes, escape system) is byte-identical to the NumPy V4/V5 path.

THRML is a CPU/GPU sampler — NOT Z1 silicon. Nothing here implies silicon.

PERFORMANCE NOTE (2026-10-06): THRML 0.1.4's sample_states re-traces on
every call (~16 s fixed cost per S2 call) because sample_with_observation
has no jit boundary. This runner works around it WITHOUT changing the
math: each program is built ONCE per structure; per call only the unary
bias weights are swapped via eqx.tree_at (bit-identical to a fresh build —
verified), and sampling goes through one eqx.filter_jit per program.
Per-call cost drops ~500x (16 s -> 0.03 s for S2). The executed Gibbs
updates are THRML's own SpinGibbsConditional math, unchanged.

Paul's OOM rule: XLA_PYTHON_CLIENT_PREALLOCATE=false is set below before
any JAX import, and must also be exported in the shell.
"""
import os
# Must be set before JAX is first imported/used (Paul's OOM rule).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
import json
import csv
import sys

import numpy as np

import env
import sensor
import planner
import controller
import classical
import s16 as s16_module
from harness import S2Agent  # noqa: E402  (s16.py puts thermopilot on sys.path)

import jax
import jax.numpy as jnp
import equinox as eqx
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from fly import (
    pick_start_goal, check_collision, optimal_len_to_tolerance,
    GOAL_TOL, GOAL_BLOCKED_RADIUS, MAX_MOVE_REFUSALS, START_GOAL_MIN_DIST,
    N_AZ, ELEV, MAX_STEPS, J_SMOOTH,
    N_SWEEPS_FULL, BURN_IN_FULL, N_SWEEPS_INCR, BURN_IN_INCR,
)
from head_to_head import (
    effective_observed_mask, map_hash, manifest_row,
    MANIFEST_FIELDS, WEAK_OBS_MAX, STRONG_L_THRESH,
    REAL_LIDAR_DIR,
)
from head_to_head_v2 import (
    MANIFEST_FIELDS_V2, TSU_SUBSTEPS, TSU_SPEED, MAX_MOVE_REFUSALS_S16,
)

_J01 = 0.25  # agreement-form {0,1} coupling (J_SMOOTH)
_BETA_F = 1.0


def _jitted_sampler(observe_blocks):
    """One eqx.filter_jit per program. Static structure is shared across
    calls (via eqx.tree_at bias swaps); only array values change, so the
    trace compiles once and every later call is a cache hit."""
    @eqx.filter_jit
    def _run(key, prog, sched, init):
        return sample_states(key, prog, sched, init, [], observe_blocks)
    return _run


# ---------------------------------------------------------------------------
# THRML perception (surgical: program built once per shape, biases swapped)
# ---------------------------------------------------------------------------

_perc_cache = {}


def _perc_program(shape):
    """Build the 16k-spin perception program ONCE per shape. Returns dict
    with prog0 (placeholder biases), block index maps, observe, sampler."""
    key = tuple(shape)
    if key in _perc_cache:
        return _perc_cache[key]
    nx, ny, nz = (int(v) for v in shape)
    n = nx * ny * nz
    nodes = [SpinNode() for _ in range(n)]
    pairs = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                i = (x * ny + y) * nz + z
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    x2, y2, z2 = x + dx, y + dy, z + dz
                    if x2 < nx and y2 < ny and z2 < nz:
                        j = (x2 * ny + y2) * nz + z2
                        pairs.append((i, j))
    edges = [(nodes[i], nodes[j]) for i, j in pairs]
    weights = jnp.full(len(pairs), _J01 / 2.0, dtype=jnp.float32)
    beta = jnp.asarray(_BETA_F, dtype=jnp.float32)
    color = np.fromfunction(lambda x, y, z: (x + y + z) % 2,
                            (nx, ny, nz), dtype=int).ravel()
    black_idx = np.where(color == 0)[0]
    white_idx = np.where(color == 1)[0]
    black = [nodes[i] for i in black_idx]
    white = [nodes[i] for i in white_idx]
    observe = [Block(nodes)]
    model = IsingEBM(nodes, edges,
                     jnp.zeros(n, dtype=jnp.float32), weights, beta)
    prog0 = IsingSamplingProgram(model, [Block(black), Block(white)], [])
    entry = {
        "prog0": prog0, "observe": observe,
        "black_idx": black_idx, "white_idx": white_idx,
        "n": n, "shape": (nx, ny, nz),
        "sampler": _jitted_sampler(observe),
        "sched_cold": SamplingSchedule(int(BURN_IN_FULL),
                                       int(N_SWEEPS_FULL - BURN_IN_FULL), 1),
        "sched_warm": SamplingSchedule(int(BURN_IN_INCR),
                                       int(N_SWEEPS_INCR - BURN_IN_INCR), 1),
    }
    _perc_cache[key] = entry
    return entry


def _swap_perc_biases(entry, h_quantized):
    """Return program with unary bias weights = beta*(h/2), bit-identical to
    a fresh build (verified 2026-10-06)."""
    bf = np.asarray(h_quantized, dtype=np.float64).ravel() / 2.0
    b32 = bf.astype(np.float32)
    new_w = [
        jnp.asarray((_BETA_F * b32[entry["black_idx"]]).reshape(-1, 1),
                    dtype=jnp.float32),
        jnp.asarray((_BETA_F * b32[entry["white_idx"]]).reshape(-1, 1),
                    dtype=jnp.float32),
    ]
    return eqx.tree_at(
        lambda p: [p.per_block_interactions[0][0].weights,
                   p.per_block_interactions[1][0].weights],
        entry["prog0"], new_w)


def compute_marginals_thrml(logodds, h_quantized, x_last, seed, step):
    """THRML mirror of head_to_head.compute_marginals("tsu", ...).

    Returns (marginals, gibbs_time, new_x_last). Step 0: full cold run;
    later steps: incremental warm run from x_last — same schedule as NumPy.
    """
    entry = _perc_program(h_quantized.shape)
    prog = _swap_perc_biases(entry, h_quantized)
    if x_last is None:
        sched = entry["sched_cold"]
        x0 = np.zeros(entry["n"], dtype=bool)
        key = jax.random.key(int(seed))
    else:
        sched = entry["sched_warm"]
        x0 = np.asarray(x_last).ravel().astype(bool)
        key = jax.random.key(int(seed + step))
    init = [jnp.asarray(x0[entry["black_idx"]]),
            jnp.asarray(x0[entry["white_idx"]])]
    t0 = time.time()
    sm = entry["sampler"](key, prog, sched, init)
    jax.block_until_ready(sm)
    dt = time.time() - t0
    arr = np.asarray(sm[0], dtype=np.float64)  # (n_samples, n) booleans
    marg = arr.mean(axis=0).reshape(h_quantized.shape)  # P(s=+1) == P(x=1)
    x_new = arr[-1].astype(np.int32).reshape(h_quantized.shape)
    return marg, dt, x_new


# ---------------------------------------------------------------------------
# THRML S2 (program built once; biases swapped per act())
# ---------------------------------------------------------------------------

class ThrmlS2Agent(S2Agent):
    """S2Agent with the NumPy gibbs_ising call replaced by THRML sampling.

    The 18-spin program is built ONCE; per act() only the unary bias
    weights are swapped (bit-identical to a fresh build — verified
    2026-10-06). h/J construction, forbidden-move handling, decision rule,
    and greedy fallback are replicated exactly from S2Agent.act.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n = self.n
        nodes = [SpinNode() for _ in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if self.J[i, j] != 0.0]
        edges = [(nodes[i], nodes[j]) for i, j in pairs]
        weights = jnp.asarray(
            np.array([self.J[i, j] for i, j in pairs], dtype=np.float64),
            dtype=jnp.float32)
        beta = jnp.asarray(_BETA_F, dtype=jnp.float32)
        blocks = [Block([nd]) for nd in nodes]
        self._s2_observe = [Block(nodes)]
        model = IsingEBM(nodes, edges,
                         jnp.zeros(n, dtype=jnp.float32), weights, beta)
        # 18 single-spin blocks, separate superblocks -> sequential Gibbs.
        self._s2_prog0 = IsingSamplingProgram(
            model, [Block([nd]) for nd in nodes], [])
        self._s2_sampler = _jitted_sampler(self._s2_observe)
        self._s2_sched = SamplingSchedule(int(self.n_sweeps // 4),
                                          int(self.n_sweeps
                                              - self.n_sweeps // 4), 1)

    def _build_h(self, pos, goal, p_occ, obs_count, forbidden_moves):
        # Replicated from S2Agent.act (thermopilot/harness.py), F4/F2 incl.
        h = self.h_static.copy()
        _fb = set(forbidden_moves) if forbidden_moves else set()
        for t in range(self.T):
            for i, d in enumerate(self.moves):
                idx = t * self.K + i
                c = self._ray_cost(pos, d, t + 1, goal, p_occ, obs_count)
                h[idx] += -c / 2.0
                if i in _fb:
                    h[idx] = -self._FORBID_PENALTY_H
        return h, _fb

    def _swap_s2_biases(self, h):
        b32 = np.asarray(h, dtype=np.float64).astype(np.float32)
        new_w = [jnp.asarray([[_BETA_F * float(b32[bi])]], dtype=jnp.float32)
                 for bi in range(self.n)]
        return eqx.tree_at(
            lambda p: [p.per_block_interactions[bi][0].weights
                       for bi in range(self.n)],
            self._s2_prog0, new_w)

    def act(self, pos, goal, p_occ, obs_count=None, forbidden_moves=None):
        nx, ny, nz = p_occ.shape
        h, _fb = self._build_h(pos, goal, p_occ, obs_count, forbidden_moves)
        prog = self._swap_s2_biases(h)
        init = [jnp.zeros(1, dtype=jnp.bool_) for _ in range(self.n)]
        key = jax.random.key(int(self.seed + self.step_count))
        sm = self._s2_sampler(key, prog, self._s2_sched, init)
        jax.block_until_ready(sm)
        self.step_count += 1
        samples = np.asarray(sm[0], dtype=bool)  # (n_samples, 18)
        # Decision: best-energy sample -> argmax(row0).
        s_pm = np.where(samples, 1.0, -1.0)
        energies = np.array([
            float(-np.dot(h, s) - np.dot(s, self.J @ s) / 2.0) for s in s_pm
        ])
        bi = int(np.argmin(energies))
        x_best = ((s_pm[bi] + 1.0) / 2.0).astype(int).reshape(self.T, self.K)
        row0 = x_best[0]
        if row0.sum() != 1:
            # Greedy fallback — replicated exactly from S2Agent.act.
            max_d = float(nx + ny + nz)
            costs = [self._ray_cost(pos, d, 1, goal, p_occ, obs_count)
                     for d in self.moves]
            for i in _fb:
                costs[i] += self._FORBID_PENALTY_H * 2.0
            i = int(np.argmin(costs))
        else:
            i = int(np.argmax(row0))
        dx, dy, dz = self.moves[i]
        return (pos[0] + dx, pos[1] + dy, pos[2] + dz), {"method": "tsu-s2"}


# ---------------------------------------------------------------------------
# Mission loop: byte-identical to head_to_head_v2.run_mission_s16 except the
# two THRML call sites (marked [THRML]).
# ---------------------------------------------------------------------------

def _now():
    return time.strftime("%H:%M:%S")


def run_mission_s16_thrml(occ, room_id, seed, noise_val, mission_id,
                          telemetry_dir=None, verbose=False, progress=False):
    """Fly one S16 mission with perception + S2 on THRML. Returns manifest row."""
    t_mission = time.time()
    shape = occ.shape

    start_pose, goal_pose = pick_start_goal(occ, seed)
    pose = start_pose
    logodds = np.zeros(shape)
    obs_count = np.zeros(shape, dtype=np.int32)

    # Boundary prior (same as fly.py)
    logodds[0, :, :] = 2.0
    logodds[-1, :, :] = 2.0
    logodds[:, 0, :] = 2.0
    logodds[:, -1, :] = 2.0
    logodds[:, :, 0] = 2.0
    logodds[:, :, -1] = 2.0
    _sv = planner.pose_to_voxel(start_pose, shape)
    _gv = planner.pose_to_voxel(goal_pose, shape)
    logodds[_sv] = 0.0
    logodds[_gv] = 0.0

    p_occ_gt = occ.astype(float)
    gt_full = planner.astar(p_occ_gt,
                            planner.pose_to_voxel(start_pose, shape),
                            planner.pose_to_voxel(goal_pose, shape))
    optimal_len = optimal_len_to_tolerance(gt_full, goal_pose, GOAL_TOL) \
        if gt_full else -1

    log = {
        "mission_id": mission_id,
        "config": "s16",
        "room_id": room_id,
        "seed": seed,
        "noise_val": noise_val,
        "start": start_pose,
        "goal": goal_pose,
        "optimal_path_len": optimal_len,
        "steps": [],
        "outcome": None,
        "n_plan_calls": 0,
        "n_planning_failures": 0,
        "n_move_refusals": 0,
        "n_tsu_decisions": 0,
        "total_tsu_time": 0.0,
        "path_voxels": [],
        "collision_p_occ": None,
        "total_gibbs_time": 0.0,
        "total_plan_time": 0.0,
    }

    # S16 planner (host A* + TSU S2) — same params as V4/V5 NumPy runs.
    s16p = s16_module.S16Planner(
        waypoint_lookahead=2, stuck_threshold=40,
        s2_kwargs={"seed": seed, "n_sweeps": 200})
    # [THRML] swap the NumPy S2 sampler for the THRML one (same params).
    s16p.s2 = ThrmlS2Agent(T=3, n_sweeps=200, seed=seed)

    x_last = None
    goal_v = planner.pose_to_voxel(goal_pose, shape)
    marg = np.zeros(shape)

    for step in range(MAX_STEPS):
        t_step = time.time()
        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=scan_start "
                  f"pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f})",
                  flush=True)

        # ---- 1 Hz host: scan ----
        scans = sensor.scan_pose(occ, pose, n_az=N_AZ, elevations=ELEV,
                                 max_range=40.0, range_noise=noise_val,
                                 seed=1000 + seed * 100 + step)
        sensor.integrate_scan(logodds, shape, scans, max_range=40.0,
                              obs_count=obs_count)

        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=gibbs_start",
                  flush=True)

        # ---- 1 Hz host: Gibbs -> P(occupied)  [THRML] ----
        h = sensor.quantize_8bit(logodds)
        marg, dt_gibbs, x_last = compute_marginals_thrml(
            logodds, h, x_last, seed, step)
        log["total_gibbs_time"] += dt_gibbs
        obs_mask = effective_observed_mask(logodds, obs_count, "tsu")

        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=gibbs_done "
                  f"dt={dt_gibbs:.2f}s", flush=True)

        # ---- D3: goal-blocked check ----
        if (controller.distance(pose, goal_pose) < GOAL_BLOCKED_RADIUS
                and bool(obs_mask[goal_v])
                and float(marg[goal_v]) > planner.P_OBSTACLE):
            log["steps"].append({"step": step, "pose": pose,
                                 "event": "goal_blocked"})
            log["outcome"] = "goal_blocked"
            break

        # ---- 10 Hz TSU fast loop ----
        if progress:
            print(f"[{_now()}] {mission_id} step={step} phase=tsu_start "
                  f"substeps={TSU_SUBSTEPS}", flush=True)
        host_replans_before = s16p.n_host_replans
        tsu_time_step = 0.0
        for tsu_sub in range(TSU_SUBSTEPS):
            pos_v = planner.pose_to_voxel(pose, shape)
            next_v, s2info = s16p.tsu_act(
                pos_v, goal_v, marg, obs_mask, pos_cont=pose)
            log["n_tsu_decisions"] += 1
            log["total_tsu_time"] += s2info["tsu_time"]
            tsu_time_step += s2info["tsu_time"]

            # D5: predictive guard on the proposed voxel.
            if not planner.safe_to_enter(next_v, marg, obs_mask, pos_v):
                log["n_move_refusals"] += 1
                s16p.notify_refusal(next_v, pos_v, marg, obs_mask)
                if log["n_move_refusals"] > MAX_MOVE_REFUSALS_S16:
                    log["outcome"] = "blocked"
                    break
                continue  # hover this sub-step

            target = planner.voxel_to_pose(next_v)
            _sf = float(s2info.get("speed_factor", 1.0))
            pose = controller.move_toward(pose, target, speed=TSU_SPEED * _sf)
            v = planner.pose_to_voxel(pose, shape)
            if v not in log["path_voxels"][-1:]:
                log["path_voxels"].append(v)

            if check_collision(pose, occ):
                log["collision_p_occ"] = float(marg[v])
                log["outcome"] = "collision"
                break

            if controller.distance(pose, goal_pose) < GOAL_TOL:
                log["outcome"] = "success"
                break
        else:
            log["n_plan_calls"] = s16p.n_host_replans
            log["n_planning_failures"] = s16p.n_host_failures
            log["steps"].append({"step": step, "n_tsu": 10})
            if progress:
                d_goal = controller.distance(pose, goal_pose)
                print(f"[{_now()}] {mission_id} step={step} phase=step_done "
                      f"tsu_dt={tsu_time_step:.2f}s "
                      f"dist_goal={d_goal:.1f} "
                      f"replans={s16p.n_host_replans} "
                      f"refusals={log['n_move_refusals']}",
                      flush=True)
            continue
        log["n_plan_calls"] = s16p.n_host_replans
        log["n_planning_failures"] = s16p.n_host_failures
        if progress:
            print(f"[{_now()}] {mission_id} phase=outcome "
                  f"outcome={log['outcome']}", flush=True)
        break
    else:
        log["outcome"] = "max_steps"
        if progress:
            print(f"[{_now()}] {mission_id} phase=outcome outcome=max_steps",
                  flush=True)

    # --- mission metrics (same as V1/V2) ---
    log["n_steps"] = len(log["steps"])
    log["wall_time"] = time.time() - t_mission
    log["flown_path_len"] = len(log["path_voxels"])
    log["final_map_hash"] = map_hash(marg)

    if log["path_voxels"]:
        flown_occ = [bool(occ[v]) for v in log["path_voxels"]]
        log["flown_fn_rate"] = float(np.mean(flown_occ))
    else:
        log["flown_fn_rate"] = 0.0

    log["final_acc"] = classical.accuracy(marg, occ)

    obs_final = effective_observed_mask(logodds, obs_count, "tsu")
    base_final = classical.observed_mask(logodds)
    gt_free = (occ == 0)
    gt_occ = (occ == 1)
    log["phantom_count"] = int(
        ((marg > planner.P_OBSTACLE) & gt_free & obs_final).sum())
    log["phantom_count_base"] = int(
        ((marg > planner.P_OBSTACLE) & gt_free & base_final).sum())
    log["fn_count_observed"] = int(
        ((marg < planner.P_CAUTION) & gt_occ & obs_final).sum())
    log["fn_count_total"] = int(
        ((marg < planner.P_CAUTION) & gt_occ).sum())

    if log["optimal_path_len"] > 0 and log["outcome"] == "success":
        log["path_efficiency"] = log["optimal_path_len"] / log["flown_path_len"]
    else:
        log["path_efficiency"] = None

    if telemetry_dir:
        os.makedirs(telemetry_dir, exist_ok=True)
        tpath = os.path.join(telemetry_dir, f"{mission_id}.jsonl")
        with open(tpath, "w") as f:
            for s in log["steps"]:
                f.write(json.dumps(s, default=str) + "\n")
        log["telemetry_file"] = tpath

    if verbose:
        print(f"  [s16-thrml] {mission_id}: {log['outcome']} "
              f"({log['wall_time']:.1f}s, "
              f"tsu={log['n_tsu_decisions']} decisions)")

    row = manifest_row(log)
    row["n_escapes"] = s16p.n_escapes
    row["n_d5_refusals_s16"] = s16p.n_d5_refusals
    row["n_escape_replans"] = s16p.n_escape_replans
    row["n_tsu_decisions"] = log["n_tsu_decisions"]
    row["total_tsu_time"] = round(log["total_tsu_time"], 3)
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Head-to-head V5-THRML: S16 on THRML")
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--noise", type=float, nargs="*", default=[0.15, 0.30])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--manifest", default="RUN_MANIFEST_V5_THRML.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Run only the first N missions (smoke test).")
    args = ap.parse_args()

    from head_to_head import list_rooms

    base = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(base, "head_to_head")
    tdir = os.path.join(outdir, "TELEMETRY_V5_THRML")
    mpath = os.path.join(outdir, args.manifest)
    os.makedirs(tdir, exist_ok=True)

    rooms = args.rooms or list_rooms()
    print(f"rooms: {len(rooms)}, seeds: {args.seeds}, noise: {args.noise}")
    total = len(rooms) * len(args.seeds) * len(args.noise)
    print(f"total missions: {total}")
    print(f"XLA preallocate: "
          f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '<unset>')}")

    done = set()
    if args.resume and os.path.exists(mpath):
        with open(mpath) as f:
            for row in csv.DictReader(f):
                done.add(row["mission_id"])
        print(f"resuming: {len(done)} missions already done")

    occ_cache = {}
    for r in rooms:
        p = os.path.join(REAL_LIDAR_DIR, f"{r}_occ.npy")
        occ_cache[r] = np.load(p)

    write_header = not (os.path.exists(mpath) and args.resume
                        and os.path.getsize(mpath) > 0)
    fout = open(mpath, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS_V2)
    if write_header:
        writer.writeheader()
        fout.flush()

    n_done = 0
    t_all = time.time()
    try:
        for room_id in rooms:
            occ = occ_cache[room_id]
            for seed in args.seeds:
                for noise_val in args.noise:
                    mid = f"{room_id}_s{seed}_n{noise_val:.2f}_s16"
                    if mid in done:
                        continue
                    row = run_mission_s16_thrml(
                        occ, room_id, seed, noise_val, mid,
                        telemetry_dir=tdir, verbose=True,
                        progress=args.progress)
                    writer.writerow(row)
                    fout.flush()
                    n_done += 1
                    done.add(mid)
                    if args.smoke and n_done >= args.smoke:
                        print(f"smoke: {n_done} missions done, stopping")
                        return
    finally:
        fout.close()

    print(f"\ndone: {n_done} missions in {time.time()-t_all:.0f}s")
    print(f"manifest: {mpath}")


if __name__ == "__main__":
    main()
