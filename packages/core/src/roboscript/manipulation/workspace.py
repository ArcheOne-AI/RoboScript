"""The probed workspace (docs/concepts/embodied-core.md, pre-checks): per arm of an embodiment,
the grid cells its TCP reaches with at least one of a few typical orientations (top-down at four
yaws, horizontal toward the front, left, right and the diagonals, and tilted 45 deg), found once
with the planner's IK (`probe`) and stored as a JSON file. The executor refuses a target whose TCP
lies farther than a margin from every reachable cell: a quick `unreachable` with the distance,
before any planning; the planner still decides everything else.
"""

import functools
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, PositiveFloat

from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.motion import MAX_GOALS, IKRequest, Planner
from roboscript.contracts.robot import Arm
from roboscript.embodiments import EmbodimentProfile
from roboscript.spatial.rotation import quat_from_matrix

_ORIGIN: Final = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))


class Workspace(BaseModel):
    """A grid of `shape` cells of `cell_m` from `low` (world), and per arm the flat indices (C
    order) of the cells whose center its TCP reaches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embodiment: str
    low: Vec3
    cell_m: PositiveFloat
    shape: tuple[int, int, int]
    reachable: dict[Arm, list[int]]

    def distance(self, arm: Arm, point: ArrayLike) -> float:
        """How far `point` is from the nearest reachable cell (0 inside one)."""
        index = np.unravel_index(np.asarray(self.reachable[arm], dtype=int), self.shape)
        centers = np.asarray(self.low) + (np.stack(index, axis=1) + 0.5) * self.cell_m
        if not len(centers):
            return math.inf
        outside = np.maximum(np.abs(centers - np.asarray(point)) - self.cell_m / 2, 0.0)
        return float(np.linalg.norm(outside, axis=1).min())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json() + "\n", encoding="utf-8")


@functools.lru_cache(maxsize=4)
def load_workspace(path: Path) -> Workspace:
    """A stored workspace (cached: executors are built per episode)."""
    return Workspace.model_validate_json(path.read_text(encoding="utf-8"))


def orientations() -> list[Pose]:
    """The probe's TCP orientations: the gripper's x (approach) down at yaws of 0, 45, 90 and 135
    deg, horizontal toward +y and at 45 and 90 deg either side, and 45 deg down toward +y, -x
    and +x."""
    found = []
    down = np.array([0.0, 0.0, -1.0])
    for yaw in (0, 45, 90, 135):
        side = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
        found.append(_frame(down, side))
    up = np.array([0.0, 0.0, 1.0])
    for heading in (90, 45, 135, 0, 180):
        a = math.radians(heading)
        found.append(_frame(np.array([math.cos(a), math.sin(a), 0.0]), up))
    for heading in (90, 180, 0):
        a = math.radians(heading)
        tilted = np.array([math.cos(a), math.sin(a), -1.0]) / math.sqrt(2)
        found.append(_frame(tilted, up))
    return found


async def probe(
    planner: Planner,
    profile: EmbodimentProfile,
    low: Vec3,
    high: Vec3,
    cell_m: float,
    seed: dict[Arm, list[float]],
) -> Workspace:
    """Probe every arm's reach with IK (from `seed` joints) at the centers of the cells of the
    box `low`..`high`: a cell is reachable if some probe orientation is."""
    shape = tuple(math.ceil(round((h - lo) / cell_m, 9)) for lo, h in zip(low, high, strict=True))
    assert len(shape) == 3
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
    centers = np.asarray(low) + (grid.reshape(-1, 3) + 0.5) * cell_m
    reachable: dict[Arm, list[int]] = {}
    for arm, spec in profile.arms.items():
        base = spec.planner.base if spec.planner is not None else _ORIGIN
        reached = np.zeros(len(centers), dtype=bool)
        for rotation in orientations():
            todo = np.flatnonzero(~reached)
            for chunk in _chunks(todo):
                goals = [
                    profile.ee_pose(arm, Pose(p=tuple(centers[i].tolist()), q=rotation.q))  # type: ignore[arg-type]
                    for i in chunk
                ]
                solved = await planner.ik(
                    IKRequest(arm=arm, base=base, goals=goals, seed=seed[arm])
                )
                reached[list(chunk)] |= np.array([s.success for s in solved])
        reachable[arm] = np.flatnonzero(reached).tolist()
    return Workspace(
        embodiment=profile.name,
        low=low,
        cell_m=cell_m,
        shape=shape,
        reachable=reachable,  # type: ignore[arg-type]
    )


REACH_POINTS: Final = 16
"""Points along the line to a target that `reach_gap` probes (the start excluded)."""


async def reach_gap(
    planner: Planner,
    profile: EmbodimentProfile,
    arm: Arm,
    base: Pose,
    seed: Sequence[float],
    start: ArrayLike,
    goal: ArrayLike,
) -> NDArray[np.float64]:
    """How far the TCP point `goal` (world) lies beyond `arm`'s reach toward it: `goal` less
    the farthest of `REACH_POINTS` points along the line from `start` (its TCP now) that the
    TCP reaches at some probe orientation (`orientations`), IK from `seed`; zeros when `goal`
    itself is reached so. What an `unreachable` receipt says (`Executor.reach_note`)."""
    a, b = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    points = [a + t * (b - a) for t in np.arange(1, REACH_POINTS + 1) / REACH_POINTS]
    rotations = orientations()
    goals = [
        profile.ee_pose(arm, Pose(p=tuple(p.tolist()), q=r.q))  # type: ignore[arg-type]
        for p in points
        for r in rotations
    ]
    reached = np.zeros(len(goals), dtype=bool)
    for chunk in _chunks(np.arange(len(goals))):
        request = IKRequest(arm=arm, base=base, goals=[goals[i] for i in chunk], seed=list(seed))
        reached[chunk] = [s.success for s in await planner.ik(request)]
    per_point = reached.reshape(len(points), len(rotations)).any(axis=1)
    farthest = int(np.flatnonzero(per_point)[-1]) if per_point.any() else -1
    nearest = a if farthest < 0 else points[farthest]
    return b - nearest


def _chunks(values: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    return [values[i : i + MAX_GOALS] for i in range(0, len(values), MAX_GOALS)]


def _frame(x: NDArray[np.float64], hint: NDArray[np.float64]) -> Pose:
    """An orientation with x along `x` and z as near `hint` as it can be."""
    x = x / np.linalg.norm(x)
    z = hint - (hint @ x) * x
    z = z / np.linalg.norm(z)
    q = quat_from_matrix(np.column_stack([x, np.cross(z, x), z]))
    return Pose(p=(0.0, 0.0, 0.0), q=tuple(q.tolist()))  # type: ignore[arg-type]
