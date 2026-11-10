"""Synthetic demonstrations, built the way a planned arm moves: every planned leg starts and ends at
rest with a bell-shaped speed, and a gripper changes only while its arm stands. Motions are made one
physics step at a time, one action per call, and recorded as a collector saves frames: at each
action's start (repeating the frame before), after its first step, every `frame_steps` steps and at
its end."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import Arm
from roboscript.demos import ArmTrack, Demo, DemoMeta, Provenance, arm_track
from roboscript.spatial.pose import pose_compose
from roboscript.spatial.rotation import quat_from_axis_angle, quat_multiply

OFFSET = (0.12, 0.0, 0.0)
DOWN = quat_from_axis_angle([0.0, 1.0, 0.0], np.pi / 2)  # the gripper pointing straight down


class Motion:
    """One arm's TCP and gripper command: `states[0]` before any step, then one per step."""

    def __init__(self, xyz: ArrayLike = (0.0, 0.0, 0.9), quat: ArrayLike = DOWN) -> None:
        self.xyz = np.asarray(xyz, dtype=float)
        self.quat = np.asarray(quat, dtype=float)
        self.grip = 1.0
        self.states = [self._state()]
        self.actions: list[tuple[int, int]] = []  # first and last state of each action

    def _state(self) -> np.ndarray:
        return np.r_[self.xyz, self.quat, self.grip]

    def _act(self, steps: int, step: Callable[[int], None]) -> "Motion":
        first = len(self.states)
        for i in range(steps):
            step(i)
            self.states.append(self._state())
        self.actions.append((first, len(self.states) - 1))
        return self

    def rest(self, steps: int) -> "Motion":
        return self._act(steps, lambda i: None)

    def move(
        self,
        d: ArrayLike = (0.0, 0.0, 0.0),
        steps: int = 200,
        turn_deg: float = 0.0,
        axis: ArrayLike = (0.0, 0.0, 1.0),
        floor: float = 0.0,
    ) -> "Motion":
        """Move by `d` and turn about the world `axis`, with a bell-shaped speed; `floor` keeps the
        ends from reaching rest, as when the next leg starts before the arm settles."""
        start, q0 = self.xyz.copy(), self.quat.copy()
        # Like a planned trajectory, the arm moves a little from the first step on.
        w = np.sin(np.linspace(0, np.pi, steps + 2)[1:-1]) ** 2 + floor
        progress = np.cumsum(w) / w.sum()

        def step(i: int) -> None:
            self.xyz = start + progress[i] * np.asarray(d, dtype=float)
            turn = quat_from_axis_angle(axis, np.radians(turn_deg * progress[i]))
            self.quat = quat_multiply(turn, q0)

        return self._act(steps, step)

    def follow(self, positions: ArrayLike) -> "Motion":
        """One action through `positions` (N, 3), one per step."""
        positions = np.asarray(positions, dtype=float)

        def step(i: int) -> None:
            self.xyz = positions[i]

        return self._act(len(positions), step)

    def path(self, *points: ArrayLike, steps: int = 200, dip: float = 0.35) -> "Motion":
        """Legs through `points` back to back as one action, the speed dipping to `dip` of its
        peak between them (it starts and stops abruptly at the ends)."""
        track, start = [], self.xyz
        for end in points:
            w = np.sin(np.linspace(0, np.pi, steps)) ** 2 + dip / (1 - dip)
            d = np.asarray(end, dtype=float) - start
            track += [start + f * d for f in np.cumsum(w) / w.sum()]
            start = np.asarray(end, dtype=float)
        return self.follow(track)

    def gripper(
        self, to: float, steps: int = 200, push: ArrayLike = (0.0, 0.0, 0.0), settle: int = 0
    ) -> "Motion":
        """Ramp the command to `to` over `steps`; the fingers shove the arm by `push` meanwhile,
        and on at that rate for `settle` more steps of the same action."""
        g0, start, rate = self.grip, self.xyz.copy(), np.asarray(push, dtype=float) / steps

        def step(i: int) -> None:
            self.grip = g0 + (to - g0) * min(i + 1, steps) / steps
            self.xyz = start + rate * (i + 1)

        return self._act(steps + settle, step)

    def frames(self, frame_steps: int) -> list[int]:
        """The states a collector would save."""
        out: list[int] = []
        for first, last in self.actions:
            out += [first - 1, *range(first, last + 1, frame_steps), last]
        return out

    def track(self, frame_steps: int = 1, frames: list[int] | None = None) -> ArmTrack:
        """Recorded every `frame_steps` steps, or at `frames` (holding the last state beyond)."""
        idx = np.minimum(
            self.frames(frame_steps) if frames is None else frames, len(self.states) - 1
        )
        states = np.array(self.states)[idx]
        ee = pose_compose(states[:, :7], [-OFFSET[0], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        return arm_track(ee, states[:, 7], OFFSET)


def pick_and_place(grasp_dz: float = -0.08, approach: bool = True) -> Motion:
    """Above the object, down `grasp_dz` to it (both in one move without `approach`), grasp,
    lift, over the target, set down, release, retreat; the arm rests between moves."""
    m = Motion().rest(30)
    if approach:
        m.move((0.2, 0.0, -0.05)).rest(30).move((0.0, 0.0, grasp_dz), 120).rest(30)
    else:
        m.move((0.2, 0.0, -0.05 + grasp_dz)).rest(30)
    m.gripper(0.0).rest(30)
    m.move((0.0, 0.0, 0.1)).rest(30)
    m.move((0.15, 0.1, 0.0)).rest(30)
    m.move((0.0, 0.0, -0.08), 120).rest(30)
    m.gripper(1.0).rest(30)
    return m.move((0.0, 0.0, 0.1)).rest(30)


def demo_of(
    motions: dict[Arm, Motion],
    frame_steps: int = 1,
    episode: int = 0,
    task: str = "synthetic",
    success_frame: int | None = None,
) -> Demo:
    """All arms recorded at the frames of the one with the most steps."""
    clock = max(motions.values(), key=lambda m: len(m.states)).frames(frame_steps)
    arms = {arm: m.track(frames=clock) for arm, m in motions.items()}
    meta = DemoMeta(
        task=TaskName(task),
        seed=episode,
        instructions=["do the task"],
        frames=len(clock),
        frame_steps=frame_steps,
        ee_to_tcp=dict.fromkeys(arms, OFFSET),
        success_frame=success_frame,
        provenance=Provenance(
            source="synthetic", path=Path("/synthetic.hdf5"), sha256="0" * 64, episode=episode
        ),
    )
    return Demo(meta, arms)


@pytest.fixture
def motion() -> type[Motion]:
    return Motion


@pytest.fixture
def make_demo() -> Callable[..., Demo]:
    return demo_of


@pytest.fixture
def pick_place() -> Callable[..., Motion]:
    return pick_and_place
