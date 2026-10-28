"""Closed-loop test harness for thermopilot formulations.

Small 3D grid worlds. Compares:
  - ASTAR: host A* (baseline, from slow-drone/planner.py logic, simplified 3D)
  - S1: TSU single-step action selection (one-hot over K actions)
  - S2: TSU receding-horizon (K actions x T steps)

Perception: ground-truth occupancy + noise -> P(occupied) via simple model.
(This isolates PLANNING; perception is already validated separately.)
"""
import numpy as np
import heapq
import sys
sys.path.insert(0, '.')
from ising import one_hot, linear_costs, bilinear, z1_check
from gibbs_ising import gibbs_ising

MOVES_6 = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
MOVES_5 = MOVES_6[:5]


def make_world(shape=(8, 8, 4), obstacle_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    occ = (rng.random(shape) < obstacle_frac).astype(float)
    return occ


def add_wall(occ, x0, x1, y, z0, z1):
    occ[x0:x1, y, z0:z1] = 1.0
    return occ


def noisy_perception(occ, fp=0.05, fn=0.05, seed=1):
    """Simulate P(occupied): flip with fp/fn rates, return probabilistic map.
    
    P values are chosen to be consistent with the log-odds sensor model
    (l_occ=+0.60, l_free=-0.73 from SENSOR_FIX.md):
    - False negative (occupied, reads miss): P=0.325 (= sigmoid(-0.73), 1 miss).
      This is ABOVE the D5 threshold (0.3), so D5 blocks it. A real mapper
      would be cautious about a single miss reading on a voxel.
    - False positive (free, reads hit): P=0.8 (conservative; blocked anyway).
    - True occupied: P=0.9. True free: P=0.05.
    """
    rng = np.random.default_rng(seed)
    p = occ.copy().astype(float)
    flip = rng.random(occ.shape)
    # false positives: free -> high P
    p[(occ == 0) & (flip < fp)] = 0.8
    # false negatives: occupied -> P=0.325 (1 miss in log-odds; D5 blocks)
    p[(occ == 1) & (flip < fn)] = 0.325
    # base: occupied->0.9, free->0.05 with small noise
    base = np.where(occ == 1, 0.9, 0.05)
    noise = rng.normal(0, 0.03, occ.shape)
    p = np.where((p == occ), np.clip(base + noise, 0.01, 0.99), p)
    return np.clip(p, 0.01, 0.99)


def astar_path(p_occ, start, goal, p_block=0.3):
    nx, ny, nz = p_occ.shape
    def in_b(v):
        return 0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz
    def cost(v):
        return float('inf') if p_occ[v] > p_block else 1.0 + 5.0 * p_occ[v]
    if cost(start) == float('inf'):
        return None
    open_h = [(0, start)]
    g = {start: 0.0}
    came = {}
    closed = set()
    while open_h:
        _, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(cur)
        for dx, dy, dz in MOVES_6:
            nb = (cur[0]+dx, cur[1]+dy, cur[2]+dz)
            if not in_b(nb) or nb in closed:
                continue
            c = cost(nb)
            if c == float('inf'):
                continue
            tg = g[cur] + c
            if nb not in g or tg < g[nb]:
                g[nb] = tg
                came[nb] = cur
                h = abs(nb[0]-goal[0]) + abs(nb[1]-goal[1]) + abs(nb[2]-goal[2])
                heapq.heappush(open_h, (tg + h, nb))
    return None


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])


class S1Agent:
    """Single-step TSU action selection."""

    def __init__(self, moves=MOVES_6, w_safe=0.7, w_prog=0.3, P=2.0,
                 n_sweeps=2000, seed=0):
        self.moves = moves
        self.K = len(moves)
        self.w_safe = w_safe
        self.w_prog = w_prog
        self.P = P
        self.n_sweeps = n_sweeps
        self.seed = seed
        self.step_count = 0
        h_oh, J_oh = one_hot(self.K, P)
        self.h_oh = h_oh
        self.J_oh = J_oh

    def act(self, pos, goal, p_occ):
        nx, ny, nz = p_occ.shape
        d_pos = manhattan(pos, goal)
        costs = []
        targets = []
        for (dx, dy, dz) in self.moves:
            t = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            targets.append(t)
            if t == goal:
                costs.append(0.0)  # goal bonus: always take it
                continue
            if not (0 <= t[0] < nx and 0 <= t[1] < ny and 0 <= t[2] < nz):
                costs.append(1.0)  # out of bounds = max cost
                continue
            p = float(p_occ[t])
            if p > 0.3:  # D5-soft: matches A* p_block
                costs.append(1.0)
                continue
            # progress: 0 = stepped toward goal, 0.5 = lateral, 1.0 = away
            prog = (manhattan(t, goal) - d_pos + 1.0) / 2.0
            prog = min(max(prog, 0.0), 1.0)
            c = self.w_safe * p + self.w_prog * prog
            costs.append(min(max(c, 0.0), 1.0))
        costs = np.array(costs)
        h = self.h_oh + linear_costs(costs)
        marg, best_s, best_E, _ = gibbs_ising(
            h, self.J_oh, n_sweeps=self.n_sweeps, burn_in=self.n_sweeps//4,
            seed=self.seed + self.step_count)
        self.step_count += 1
        x_best = ((best_s + 1) / 2).astype(int)
        # fallback: if not one-hot (shouldn't happen), take argmin cost
        if x_best.sum() != 1:
            return targets[int(np.argmin(costs))], {"method": "fallback"}
        return targets[int(np.argmax(x_best))], {"method": "tsu", "costs": costs}


class S2Agent:
    # F4 (2026-10-06): Z1 hardware field limit. The forbidden-move
    # penalty uses the hardware maximum instead of 1e6 (which needed |h|≈500k).
    # Correction (2026-10-28): this figure is UNSOURCED, not "(assumed)" as
    # the original F4 note said — no published Z1 field cap exists. 6.0 is
    # borrowed from the |J|<=6 figure label in arXiv 2608.01615 and kept as
    # a reasonable working number; do not cite it as a datasheet limit.
    _Z1_MAX_H = 6.0
    _FORBID_PENALTY_H = _Z1_MAX_H  # strongest representable suppression
    """Directional trajectory-probing TSU planner: K actions x T steps.

    What this actually is (honest framing per REDTEAM I2): for each
    (timestep t, action i), we probe along direction d_i with a ray-cast
    from the current position. The cost c_{t,i} reflects the worst obstacle
    cost along that ray plus an uncertainty penalty for poorly-observed
    voxels (weak-evidence idea applied to planning). The temporal smoothness
    coupling stitches probe directions into coherent action sequences.

    This is NOT true trajectory optimization (which would need T-th order
    couplings to evaluate actual action sequences). It is directional
    lookahead: "don't move toward things that look blocked within T steps,
    prefer directions that make progress, don't zigzag." For local obstacle
    avoidance this is sufficient; global reasoning comes from the host (S16).
    """

    def __init__(self, moves=None, T=3, w_safe=0.7, w_prog=0.3,
                 w_smooth=0.2, P=2.0, n_sweeps=3000, seed=0,
                 p_block=0.3, **_ignored):
        # **_ignored: absorbs legacy kwargs (e.g. min_obs) for compatibility.
        # Uncertainty now lives in the perception map (P blended toward 0.5
        # for poorly-observed voxels), not in planner parameters.
        from harness import MOVES_6 as _M6
        self.moves = moves or _M6
        self.K = len(self.moves)
        self.T = T
        self.w_safe = w_safe
        self.w_prog = w_prog
        self.w_smooth = w_smooth
        self.p_block = p_block
        self.P = P
        self.n_sweeps = n_sweeps
        self.seed = seed
        self.step_count = 0
        n = self.K * self.T
        self.n = n
        # Build static structure: one-hot per timestep + temporal couplings
        h_oh = np.zeros(n)
        J = np.zeros((n, n))
        for t in range(T):
            idx = [t * self.K + i for i in range(self.K)]
            ho, Jo = one_hot(self.K, P)
            h_oh[idx] += ho
            for ii in range(self.K):
                for jj in range(self.K):
                    J[idx[ii], idx[jj]] += Jo[ii, jj]
        # temporal smoothness: penalize direction change
        # Tmat[i,j] = 0 if same direction, higher if reversal
        Tmat = np.zeros((self.K, self.K))
        for i, mi in enumerate(self.moves):
            for j, mj in enumerate(self.moves):
                dot = mi[0]*mj[0] + mi[1]*mj[1] + mi[2]*mj[2]
                if dot == 1:
                    Tmat[i, j] = 0.0
                elif dot == 0:
                    Tmat[i, j] = 0.5
                else:  # reversal
                    Tmat[i, j] = 1.0
        self.Tmat = Tmat
        for t in range(T - 1):
            idx0 = [t * self.K + i for i in range(self.K)]
            idx1 = [(t + 1) * self.K + j for j in range(self.K)]
            hx, hy, Jxy = bilinear(self.w_smooth * Tmat)
            h_oh[idx0] += hx
            h_oh[idx1] += hy
            for ii in range(self.K):
                for jj in range(self.K):
                    J[idx0[ii], idx1[jj]] += Jxy[ii, jj]
                    J[idx1[jj], idx0[ii]] += Jxy[ii, jj]
        self.h_static = h_oh
        self.J = J

    def _ray_cost(self, pos, direction, steps, goal, p_occ, obs_count=None):
        """Cost of probing along `direction` for `steps` voxels from `pos`.

        Ray-cast: walk the straight-line ray, computing mean P(occupied).
        P values are assumed honest about uncertainty (poorly-observed voxels
        have P blended toward 0.5 by the perception model) — uncertainty lives
        in the MAP, not in the planner.

        Returns cost in [0, 1]:
          - 1.0 if ray leaves the world or the immediate voxel is blocked
          - 0.0 if ray reaches the goal
          - else w_safe*mean_p + w_prog*progress(endpoint)

        The immediate-voxel hard block (s=1) catches definite obstacles.
        Deeper voxels use soft mean costs — a single false positive at
        distance shouldn't veto a direction, but genuinely cluttered
        directions get penalized. The D5-hard guard (in run_mission) provides
        the final safety check on the actually-proposed move.
        """
        dx, dy, dz = direction
        nx, ny, nz = p_occ.shape
        d_pos = manhattan(pos, goal)
        sum_p = 0.0
        n_vox = 0
        endpoint = pos
        for s in range(1, steps + 1):
            v = (pos[0] + s * dx, pos[1] + s * dy, pos[2] + s * dz)
            endpoint = v
            if not (0 <= v[0] < nx and 0 <= v[1] < ny and 0 <= v[2] < nz):
                return 1.0  # ray leaves world: blocked
            if v == goal:
                return 0.0  # ray reaches goal: free
            pv = float(p_occ[v])
            # Hard block only on the immediate next voxel (s=1): this is the
            # voxel we'd actually enter.
            if s == 1 and pv > self.p_block:
                return 1.0
            sum_p += pv
            n_vox += 1
        # Soft obstacle cost: mean P along the ray.
        mean_p = sum_p / max(n_vox, 1)
        # Progress from endpoint (same normalization as before).
        prog = (manhattan(endpoint, goal) - d_pos + steps) / (2.0 * steps)
        prog = min(max(prog, 0.0), 1.0)
        c = self.w_safe * mean_p + self.w_prog * prog
        return min(max(c, 0.0), 1.0)

    def act(self, pos, goal, p_occ, obs_count=None, forbidden_moves=None):
        nx, ny, nz = p_occ.shape
        h = self.h_static.copy()
        # forbidden_moves: set of action indices D5 will deny (e.g. sensor
        # blind-spot verticals into unobserved voxels). Penalize heavily so
        # the sampler plans around the constraint instead of fighting D5.
        # (2026-10-05: S2 kept proposing blind verticals, D5 denied them,
        # loop -> max_steps. Now S2 knows.)
        _fb = set(forbidden_moves) if forbidden_moves else set()
        for t in range(self.T):
            for i, d in enumerate(self.moves):
                idx = t * self.K + i
                c = self._ray_cost(pos, d, t + 1, goal, p_occ, obs_count)
                h[idx] += -c / 2.0  # linear cost lowering
                if i in _fb:
                    # F4 (2026-10-06): bounded Z1-compatible forbidden-move penalty.
                    # Old: c += 1e6 → h=-500004, 83,000× over Z1's |h|≤6.
                    # D5 provides the hard safety guarantee at execution; the S2
                    # penalty is for efficiency (don't waste samples on D5-denied
                    # moves). h=-6.0 gives ~1.6e5:1 Boltzmann suppression at
                    # beta=1 — gibbs_ising has no temperature, so flipping a
                    # spin against the field costs dE=12 and exp(12)≈1.6e5 —
                    # sufficient. (Correction 2026-10-28: this comment
                    # previously said ~245:1; that figure was wrong — 245:1
                    # would need dE≈5.5, i.e. beta≈0.46 or h≈-2.75.)
                    h[idx] = -self._FORBID_PENALTY_H
        marg, best_s, best_E, _ = gibbs_ising(
            h, self.J, n_sweeps=self.n_sweeps, burn_in=self.n_sweeps//4,
            seed=self.seed + self.step_count)
        self.step_count += 1
        x_best = ((best_s + 1) / 2).astype(int).reshape(self.T, self.K)
        # first action of the plan
        row0 = x_best[0]
        if row0.sum() != 1:
            # fallback: greedy one-step on ray costs (no NameError; max_d fixed)
            max_d = float(nx + ny + nz)
            costs = [self._ray_cost(pos, d, 1, goal, p_occ, obs_count)
                     for d in self.moves]
            # F2 (2026-10-06): apply forbidden penalty in fallback too.
            # Old: fallback ignored _fb entirely → 25% forbidden proposals
            # with degenerate sampler. Cost-domain equivalent of h=-6.0
            # (h = -c/2, so c += 12.0 mirrors the main path).
            for i in _fb:
                costs[i] += self._FORBID_PENALTY_H * 2.0
            i = int(np.argmin(costs))
        else:
            i = int(np.argmax(row0))
        dx, dy, dz = self.moves[i]
        # max_abs_h: visibility for the declared |h| cap (see _Z1_MAX_H).
        # The dynamic field accumulates unbounded at the ray-cost lowering
        # above; nothing clips it (clipping to an unsourced cap would
        # silently distort the cost ordering), so the value is surfaced
        # per decision instead. tests/test_field_cap_visibility.py tracks
        # the held-out violation.
        return (pos[0]+dx, pos[1]+dy, pos[2]+dz), {
            "method": "tsu-s2", "max_abs_h": float(np.abs(h).max())}


def noisy_perception_with_counts(occ, fp=0.05, fn=0.05, seed=1,
                                 obs_radius=8, n_scans=5,
                                 l_occ=0.60, l_free=-0.73,
                                 start=None, goal=None):
    """Simulate P(occupied) via log-odds accumulation, mimicking fly.py.

    A real occupancy mapper accumulates log-odds over lidar scans:
      L += l_occ on hit, L += l_free on miss, clamped to [-4, +4].
    P = sigmoid(L).

    This gives a NATURAL uncertainty model (no artificial blending needed):
      - 1 miss: L=-0.73 → P=0.325 → blocked by D5 (cautious) ✓
      - 2 misses: L=-1.46 → P=0.188 → free ✓
      - 1 hit: L=0.6 → P=0.646 → blocked ✓
      - False-negative miss on wall: L=-0.73 → P=0.325 → BLOCKED ✓
      - Unobserved: L=0 → P=0.5 → blocked (unknown != free) ✓

    The sensor values l_occ=+0.60, l_free=-0.73 are the fixed sensor model
    from SENSOR_FIX.md (not OctoMap's +0.85/-0.4).

    Vantage points: if start/goal are given, vantage points are placed at
    start plus interpolated points toward goal (simulating the drone's
    actual observation trajectory). Otherwise, random free-space points.
    A voxel is "observed" from a vantage if within `obs_radius` (Manhattan)
    AND has line-of-sight on true occupancy. Hit/miss sampled with fp/fn.

    Returns (p_occ, obs_count). obs_count is for diagnostics; P already
    encodes uncertainty honestly.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = occ.shape
    L = np.zeros(occ.shape)  # log-odds, start at 0 (P=0.5, unknown)
    obs_count = np.zeros(occ.shape, dtype=int)
    free = np.argwhere(occ < 0.5)
    if len(free) == 0:
        return np.full(occ.shape, 0.5), obs_count
    # vantage points: start + interpolated toward goal (if given), else random
    vps = []
    if start is not None and goal is not None:
        for k in range(n_scans):
            a = k / max(n_scans - 1, 1)
            vp = (int(round(start[0] + (goal[0] - start[0]) * a)),
                  int(round(start[1] + (goal[1] - start[1]) * a)),
                  int(round(start[2] + (goal[2] - start[2]) * a)))
            # clamp to free space (nudge if inside obstacle)
            if occ[vp] > 0.5:
                # find nearest free voxel
                dists = np.abs(free - np.array(vp)).sum(axis=1)
                vp = tuple(free[np.argmin(dists)])
            vps.append(vp)
    else:
        for s in range(n_scans):
            vps.append(tuple(free[rng.integers(len(free))]))
    for vp in vps:
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if abs(ix - vp[0]) + abs(iy - vp[1]) + abs(iz - vp[2]) > obs_radius:
                        continue
                    # line-of-sight on TRUE occupancy
                    steps = max(abs(ix - vp[0]), abs(iy - vp[1]), abs(iz - vp[2]), 1)
                    blocked = False
                    for k in range(1, steps):
                        q = (int(round(vp[0] + (ix - vp[0]) * k / steps)),
                             int(round(vp[1] + (iy - vp[1]) * k / steps)),
                             int(round(vp[2] + (iz - vp[2]) * k / steps)))
                        if occ[q] > 0.5:
                            blocked = True
                            break
                    if blocked:
                        continue
                    # sample hit/miss with fp/fn noise
                    is_occ = occ[ix, iy, iz] > 0.5
                    r = rng.random()
                    if is_occ:
                        hit = r > fn  # false negative: occupied but reads miss
                    else:
                        hit = r < fp  # false positive: free but reads hit
                    L[ix, iy, iz] = np.clip(
                        L[ix, iy, iz] + (l_occ if hit else l_free), -4.0, 4.0)
                    obs_count[ix, iy, iz] += 1
    p_occ = 1.0 / (1.0 + np.exp(-np.clip(L, -30, 30)))
    return np.clip(p_occ, 0.01, 0.99), obs_count


def run_mission(agent_fn, occ, start, goal, p_occ, max_steps=200, p_block=0.3,
                obs_count=None):
    """Execute closed-loop. Returns (success, steps, collisions, path).

    D5-hard: the controller refuses any move into a voxel with P(occ)>p_block.
    If refused, drone stays put (replans next step). Matches fly.py safe_to_enter.

    obs_count: optional int array (same shape as p_occ). Passed through to
    agent_fn(pos, goal, p_occ, obs_count) when the agent accepts it.
    """
    pos = start
    path = [pos]
    collisions = 0
    refusals = 0
    nx, ny, nz = occ.shape
    import inspect
    takes_obs = False
    try:
        takes_obs = 'obs_count' in inspect.signature(agent_fn).parameters
    except (TypeError, ValueError):
        takes_obs = False
    for step in range(max_steps):
        if pos == goal:
            return True, step, collisions, path
        if takes_obs:
            nxt, info = agent_fn(pos, goal, p_occ, obs_count)
        else:
            nxt, info = agent_fn(pos, goal, p_occ)
        # D5-hard guard
        if (0 <= nxt[0] < nx and 0 <= nxt[1] < ny and 0 <= nxt[2] < nz):
            if float(p_occ[nxt]) > p_block and nxt != goal:
                refusals += 1
                path.append(pos)  # stay put
                if refusals > 50:
                    return False, step, collisions, path  # stuck
                continue
        else:
            # out of bounds proposal: treat as refused
            refusals += 1
            path.append(pos)
            if refusals > 50:
                return False, step, collisions, path
            continue
        # collision check against TRUE occupancy
        if occ[nxt] > 0.5:
            collisions += 1
            return False, step, collisions, path
        pos = nxt
        path.append(pos)
    return False, max_steps, collisions, path
