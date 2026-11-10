"""Which frames of a recorded joint trajectory a replay sends, one action each: `s<k>` every k-th
frame, `rest` the frames where an arm comes to rest and the first and last frame of every gripper
change that crosses the grip threshold. The last frame is always sent and frame 0, the start, never.
`s1` keeps the frames aligned 1:1 with the trajectory's.
"""

import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.demos.demo import Array, Demo
from roboscript.embodiments import EmbodimentProfile

STRATEGY = r"s[1-9][0-9]*|rest"
MOVING_RAD = 1e-4
"""An arm whose joint commands change by more than this between frames moves."""


def replay_frames(
    strategy: str, commands: ArrayLike, profile: EmbodimentProfile, grip_threshold: float = 0.5
) -> list[int]:
    """The frames of `commands` (N, action_dim) that `strategy` sends, in order."""
    v = np.asarray(commands, dtype=float)
    n = len(v)
    if not re.fullmatch(STRATEGY, strategy):
        raise ValueError(f"replay strategy {strategy!r}: use s<k> (every k-th frame) or rest")
    if strategy == "rest":
        grippers = v[:, [arm.gripper_index for arm in profile.arms.values()]]
        frames = _rests(v, profile) + [f for run in _changes(grippers, grip_threshold) for f in run]
    else:
        frames = list(range(int(strategy[1:]), n, int(strategy[1:])))
    return sorted({f for f in [*frames, n - 1] if 0 < f < n})


def commands(demo: Demo, profile: EmbodimentProfile) -> Array:
    """The action vectors (N, action_dim) of a demo's joint and gripper commands."""
    v = np.zeros((demo.meta.frames, profile.action_dim))
    for arm, spec in profile.arms.items():
        track = demo.arms[arm]
        if track.joints is None:
            raise ValueError(f"{demo.meta.task}: the {arm} arm has no joint commands to replay")
        v[:, slice(*spec.action)] = track.joints
        v[:, spec.gripper_index] = track.gripper
    return v


def _changes(grippers: NDArray[np.float64], threshold: float) -> list[tuple[int, int]]:
    """Per gripper column, the frames each run of changing commands starts from and ends at,
    for the runs that cross `threshold`."""
    runs = []
    for g in grippers.T:
        changing = np.r_[np.abs(np.diff(g)) > 1e-4, False]
        t = 0
        while t < len(g):
            if not changing[t]:
                t += 1
                continue
            first = t
            while t < len(g) and changing[t]:
                t += 1
            after = min(t, len(g) - 1)
            if (g[first] >= threshold) != (g[after] >= threshold):
                runs.append((first, after))
    return sorted(runs)


def _rests(v: NDArray[np.float64], profile: EmbodimentProfile) -> list[int]:
    """Frames where an arm's joint commands stop changing."""
    out = []
    for arm in profile.arms.values():
        moving = np.abs(np.diff(v[:, slice(*arm.action)], axis=0)).max(axis=1) > MOVING_RAD
        moving = np.r_[moving, False]
        out += [t for t in range(1, len(v)) if moving[t - 1] and not moving[t]]
    return out
