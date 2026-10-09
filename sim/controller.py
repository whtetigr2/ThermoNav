"""Dead-simple waypoint follower for sim validation.

Plain English: the drone has a current position and a target waypoint.
Each step, it moves toward the waypoint at a fixed speed. No PID, no
dynamics — just straight-line movement. Good enough to validate whether
the perception+planning loop works.
"""
import numpy as np

# Drone speed: voxels per sim step (1 Hz loop => 1 voxel/second)
SPEED = 1.0


def move_toward(current_pose, target_pose, speed=SPEED):
    """Move from current_pose toward target_pose by at most `speed`.

    Args:
        current_pose: (x, y, z) tuple.
        target_pose: (x, y, z) tuple.
        speed: max distance to move this step.

    Returns:
        New (x, y, z) pose.
    """
    cur = np.array(current_pose, dtype=float)
    tgt = np.array(target_pose, dtype=float)
    delta = tgt - cur
    dist = np.linalg.norm(delta)
    if dist < 1e-9:
        return tuple(cur)  # already there
    if dist <= speed:
        return tuple(tgt)  # close enough: snap to target
    step = delta / dist * speed
    return tuple(cur + step)


def distance(a, b):
    """Euclidean distance between two poses."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))
