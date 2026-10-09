"""Environment: occupancy grids for drone simulation.

Two sources:
  build_room(...)  - synthetic 3D room with box obstacles (for pipeline validation)
  load_map(...)    - REAL building floor plans from MovingAI benchmarks,
                     extruded to 3D (for approach validation)

MovingAI maps are 2D grids from real buildings (@=wall, .=free).
We extrude walls to full height and add floor/ceiling to make a 3D voxel grid.
The topology (corridors, rooms, doorways) is real, not synthetic.
"""
import os
import numpy as np

# Where the MovingAI .map files live (repo-relative; override if yours differ)
MAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "gauntlet", "maps")


def build_room(nx=32, ny=32, nz=16, seed=0):
    """Build ground-truth occupancy: 1=occupied, 0=free. Outer walls + interior boxes."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Outer walls (1 voxel thick)
    occ[0, :, :] = 1
    occ[-1, :, :] = 1
    occ[:, 0, :] = 1
    occ[:, -1, :] = 1
    occ[:, :, 0] = 1   # floor
    occ[:, :, -1] = 1  # ceiling

    # Interior box obstacles (x0,x1, y0,y1, z0,z1) — deterministic layout
    boxes = [
        (6, 10, 6, 10, 1, 9),    # pillar A
        (20, 24, 18, 22, 1, 12), # pillar B
        (12, 18, 12, 14, 1, 6),  # low wall
        (24, 27, 6, 9, 1, 15),   # tall cabinet
    ]
    for (x0, x1, y0, y1, z0, z1) in boxes:
        occ[x0:x1, y0:y1, z0:z1] = 1

    return occ


def _parse_movingai(path):
    """Parse a MovingAI .map file. Returns 2D bool array (True=occupied)."""
    with open(path) as f:
        lines = f.readlines()
    # Header: type, height, width, map
    height = width = None
    grid_start = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("height"):
            height = int(line.split()[1])
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line == "map":
            grid_start = i + 1
            break
    if grid_start is None:
        raise ValueError(f"bad .map file: {path}")
    grid = np.zeros((height, width), dtype=bool)
    for r in range(height):
        row = lines[grid_start + r].rstrip("\n")
        for c, ch in enumerate(row[:width]):
            if ch == "@":
                grid[r, c] = True
            # '.' and anything else = free
    return grid


def _find_window(grid_2d, size=32, min_free_frac=0.35, seed=0):
    """Find a size×size window with interesting structure for drone flight.

    Prefers windows with 35-75% free space: enough walls to be challenging,
    enough open space to be navigable. Pure open areas (100% free) are boring;
    solid walls (<35% free) are unflyable.
    """
    rng = np.random.default_rng(seed)
    H, W = grid_2d.shape
    if H < size or W < size:
        raise ValueError(f"map too small: {H}x{W}, need {size}x{size}")
    # Try random windows, score by how close free_frac is to ideal range
    best = None
    best_score = -1
    best_free = 0
    for _ in range(500):
        r = rng.integers(0, H - size + 1)
        c = rng.integers(0, W - size + 1)
        window = grid_2d[r:r + size, c:c + size]
        free_frac = 1.0 - window.mean()
        # Ideal: 0.35-0.75 free. Score = 1 if in range, decays outside.
        if 0.35 <= free_frac <= 0.75:
            score = 1.0
        elif free_frac < 0.35:
            score = free_frac / 0.35  # penalize too-dense
        else:
            score = max(0, 1.0 - (free_frac - 0.75) * 4)  # penalize too-open
        # Bonus for windows with actual wall structure (not just border)
        interior = window[4:-4, 4:-4]
        if interior.mean() > 0.05:  # at least 5% walls in interior
            score += 0.5
        if score > best_score:
            best_score = score
            best_free = free_frac
            best = (r, c)
        if score >= 1.5:
            break  # ideal window with interior walls
    if best_free < min_free_frac:
        raise ValueError(f"no window with >={min_free_frac} free space "
                         f"(best={best_free:.2f})")
    r, c = best
    return grid_2d[r:r + size, c:c + size], (r, c), best_free


def load_map(name, nx=32, ny=32, nz=16, seed=0):
    """Load a REAL building floor plan as a 3D occupancy grid.

    Args:
        name: map file name (e.g. "orz100d.map") in MAP_DIR.
        nx, ny, nz: output grid dimensions.
        seed: for window selection.

    Returns:
        (occ, info) where occ is (nx,ny,nz) uint8 and info is a dict with
        source details.

    The 2D walls are extruded to full height (real walls go floor-to-ceiling).
    Floor (z=0) and ceiling (z=nz-1) are marked occupied.
    Outer border is forced occupied (1 voxel thick) for containment.
    """
    path = os.path.join(MAP_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"map not found: {path}")
    grid_2d = _parse_movingai(path)
    window, (wr, wc), free_frac = _find_window(grid_2d, size=nx, seed=seed)

    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    # Extrude walls to full height (z=1..nz-2; floor/ceiling set below)
    for ix in range(nx):
        for iy in range(ny):
            if window[iy, ix]:  # note: grid_2d is [row, col] = [y, x]
                occ[ix, iy, 1:nz - 1] = 1
    # Floor and ceiling
    occ[:, :, 0] = 1
    occ[:, :, nz - 1] = 1
    # Outer border (containment)
    occ[0, :, :] = 1
    occ[-1, :, :] = 1
    occ[:, 0, :] = 1
    occ[:, -1, :] = 1

    info = {
        "source": "MovingAI",
        "file": name,
        "window_origin": (wr, wc),
        "window_free_frac": float(free_frac),
        "note": "REAL building floor plan (2D walls extruded to 3D).",
    }
    return occ, info


HM3D_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "rooms")


def load_hm3d(name):
    """Load a voxelized HM3D scene as a 3D occupancy grid.

    Args:
        name: scene id, e.g. "00808-y9hTuugGdiq". Loads
            <HM3D_DIR>/<name>_occ.npy.

    Returns:
        (occ, info) where occ is (32,32,16) uint8 (1=occupied)
        and info describes the source.

    The grids are true 3D (not extruded 2D): furniture, multi-story
    structure, doorways all captured at 0.25m voxels. See
    hm3d_scenes/HM3D_VOXELIZED.md for generation details.

    HM3D is licensed for academic/non-commercial research use.
    """
    path = os.path.join(HM3D_DIR, f"{name}_occ.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"HM3D grid not found: {path}")
    occ = np.load(path).astype(np.uint8)
    info = {
        "source": "HM3D",
        "scene": name,
        "note": "REAL 3D scan (Matterport), voxelized at 0.25m. "
                "Academic/non-commercial use only.",
    }
    return occ, info


def free_poses(occ, n_poses=10, z=8, seed=1):
    """Waypoints through free space at height z.

    For the synthetic room: hand-picked waypoints.
    For other maps: random free voxels sampled at height z.
    """
    nx, ny, nz = occ.shape
    # hand-picked free-space waypoints for the default synthetic room
    pts = [(4, 4), (4, 27), (14, 27), (14, 20), (27, 20),
           (27, 27), (27, 12), (16, 4), (10, 16), (22, 14)]
    poses = []
    for (x, y) in pts[:n_poses]:
        if 0 <= x < nx and 0 <= y < ny and occ[x, y, z] == 0:
            poses.append((float(x) + 0.5, float(y) + 0.5, float(z) + 0.5))
    if len(poses) >= 2:
        return poses
    # Fallback: random free voxels (for real maps where hand-picked don't apply)
    rng = np.random.default_rng(seed)
    free_voxels = np.argwhere(occ[:, :, z] == 0)
    if len(free_voxels) < 2:
        raise RuntimeError("not enough free space at height z")
    idx = rng.choice(len(free_voxels), size=min(n_poses, len(free_voxels)),
                     replace=False)
    return [(float(v[0]) + 0.5, float(v[1]) + 0.5, float(z) + 0.5)
            for v in free_voxels[idx]]
