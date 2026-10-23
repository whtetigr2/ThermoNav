#!/usr/bin/env python3
"""Ceilinged-maze generator for the wavefront experiments.

Extracted verbatim (functions `_lattice_solvable` and
`build_ceiling_maze`) from the lab's research script `sim_2m.py`,
sha256 ca64dc5670e8a3cf62a8496455963e8af21733ae03533cf17611560bd0744f34
— the exact generator that produced the mazes behind every number in
`docs/RESULTS.md` and the saved fields in `wavefront/data/`. Only the
two maze-construction functions are vendored here; the research
script's planner/follower stack is not part of this repository.

Mazes: 40 x 40 x 16 worlds, vertical walls z=0..14, roof at z=15,
gaps forced across >= 2 z-levels, solvable by construction (lattice
BFS check before a maze is accepted).
"""
from collections import deque

import numpy as np


def _lattice_solvable(wall):
    start, goal = (0, 5, 0), (9, 5, 0)
    seen = {start}
    dq = deque([start])
    while dq:
        x, y, l = dq.popleft()
        if (x, y, l) == goal:
            return True
        for dx, dy, dl in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (x + dx, y + dy, l + dl)
            if (0 <= n[0] < 10 and 0 <= n[1] < 10 and 0 <= n[2] < 3
                    and not wall[n[2], n[1], n[0]] and n not in seen):
                seen.add(n)
                dq.append(n)
    return False


def build_ceiling_maze(rng):
    """Random ceilinged comb maze. Vertical walls only (solvable by
    construction). Walls z=0..14, roof at z=15. Gaps at random (ly,
    z-level), forced across >=2 z-levels for 3D threading.
    Returns (rocks, start, goal, spec, wall) or None."""
    for _ in range(50):
        wall = np.zeros((3, 10, 10), dtype=bool)  # [ll, ly, lx]
        spec = {"vwalls": [], "hwalls": []}
        # vertical walls (along y) at lx 1..8: 3-4 walls, min separation 2.
        # Separation guarantees maneuvering room between gaps.
        # Comb = solvable by construction (snake through the gaps).
        n_v = int(rng.integers(3, 5))
        candidates = list(range(1, 9))
        rng.shuffle(candidates)
        v_lx = []
        for c in candidates:
            if all(abs(c - x) >= 2 for x in v_lx):
                v_lx.append(c)
            if len(v_lx) == n_v:
                break
        if len(v_lx) < n_v:
            continue
        v_lx.sort()
        for lx in v_lx:
            gw = int(rng.choice([1, 1, 1, 2]))  # gap width in lattice cells
            g0 = int(rng.integers(0, 11 - gw))
            gll = int(rng.integers(0, 3))
            spec["vwalls"].append([int(lx), g0, g0 + gw, gll])
            for ly in range(10):
                for ll in range(3):
                    if g0 <= ly < g0 + gw and ll == gll:
                        continue
                    wall[ll, ly, lx] = True
        # force 3D: gaps across >= 2 z-levels
        zlevels = {g[3] for g in spec["vwalls"]}
        if len(zlevels) < 2:
            continue
        # comb is solvable by construction; verify anyway
        if not _lattice_solvable(wall):
            continue
        # expand to world rocks (walls to z=14, no fly-over under roof at z=15)
        rocks = set()
        for ll in range(3):
            z_lo = ll * 4
            z_hi = (ll + 1) * 4 if ll < 2 else 15  # ll=2 covers z=8..14
            for ly in range(10):
                for lx in range(10):
                    if wall[ll, ly, lx]:
                        for x in range(lx * 4, lx * 4 + 4):
                            for y in range(ly * 4, ly * 4 + 4):
                                for z in range(z_lo, z_hi):
                                    rocks.add((x, y, z))
        # ROOF: full layer at z=15
        for x in range(40):
            for y in range(40):
                rocks.add((x, y, 15))
        start, goal = (2, 20, 2), (37, 20, 2)
        spec["n_rocks"] = len(rocks)
        spec["zlevels"] = sorted(zlevels)
        return rocks, start, goal, spec, wall
    return None
