"""What an adapter writes: a `SimBackend`, the one object that touches the official benchmark.

The env planes (`env.py`) are the same for every benchmark; a backend answers a few questions
about one open episode of the official simulator, on the one thread the planes call it from. It
is the only code of an adapter that imports the official package, and only inside functions, so
the adapter's agent side (descriptor, config, skills) imports without it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from roboscript.contracts.env import ConditionTruth, EndRule, Progress
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import ArmName, ArmState, RobotState
from roboscript.embodiments import EmbodimentProfile

NOT_CARRIED: Final = (
    "env.agent.execute carries joint positions only, so this adapter runs a step only when it is "
    "a joint-position step it can honour"
)
"""The sentence every `NotImplementedError` of an unsupported action names."""

HOLD_ATOL: Final = 1e-3
"""Joints this close (rad) to where the robot is are a hold: the one joint-position step a
delta-controlled robot can honour with no decision about the wire (a zero delta)."""


def not_carried(what: str) -> NotImplementedError:
    """The refusal for an action `what` the wire cannot carry yet (a step of that kind)."""
    return NotImplementedError(f"{what}: {NOT_CARRIED}")


def is_hold(qpos: Sequence[float], current: Sequence[float]) -> bool:
    """Whether `qpos` asks the joints to stay where `current` has them (any gripper command)."""
    wanted, now = np.asarray(qpos, dtype=float), np.asarray(current, dtype=float)
    return bool(np.allclose(wanted, now, atol=HOLD_ATOL))


@dataclass(frozen=True)
class Frame:
    """One camera's capture: intrinsics, OpenCV camera-to-world, RGB and depth in meters."""

    K: NDArray[np.float64]
    cam2world: NDArray[np.float64]
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]


@dataclass(frozen=True)
class Opened:
    """What the official simulator says about the episode it just reset: the instruction and the
    official step limit (None where the reset options leave it off). `unsolvable`: why the reset
    does not count, as the benchmark's own rule skips it (a stability check, say); None: it
    counts."""

    instruction: str
    step_limit: int | None
    unsolvable: str | None = None


@dataclass(frozen=True)
class Verdict:
    """The official judgement now. `success_step`: where a success latch first fired (None where
    the benchmark has no latch); `score`: the terminal score where it judges by one; `ended`: why
    the benchmark ended the episode before its step limit without success (a failure it checks
    for fired: no action counts after it), None while it runs."""

    success: bool
    success_step: int | None
    steps_used: int
    step_limit: int | None
    score: float | None = None
    ended: str | None = None


@dataclass(frozen=True)
class ArmReading:
    """One arm as the simulator reports it: the ee pose (world), the joints measured and
    commanded (a controller with no joint targets reports the measured ones as commands), the
    measured gripper opening (normalized, 1 open) and the gripper command."""

    ee: Pose
    joints: list[float]
    joints_cmd: list[float]
    gripper: float
    gripper_cmd: float


def build_robot_state(
    profile: EmbodimentProfile,
    step: int,
    step_limit: int | None,
    arms: dict[ArmName, ArmReading],
    qpos_cmd: list[float],
) -> RobotState:
    """The profile's robot state from arm readings; the TCP follows from the ee and the
    profile's `ee_to_tcp`."""
    return RobotState(
        step=step,
        step_limit=step_limit,
        arms={
            arm: ArmState(
                ee=reading.ee,
                tcp=profile.tcp_pose(arm, reading.ee),
                joints=reading.joints,
                joints_cmd=reading.joints_cmd,
                gripper=reading.gripper,
                gripper_cmd=reading.gripper_cmd,
            )
            for arm, reading in arms.items()
        },
        qpos_cmd=qpos_cmd,
    )


class SimBackend(Protocol):
    """The official simulator, for the planes. Every method runs on the one simulation thread.

    `open` resets one task by an index (a seed, an initial-state index or an instance, as the
    benchmark's `reset` says) with the benchmark's reset options (`options`, the model that
    declares them and their defaults; the planes validate `EpisodeSpec.options` against it) and
    raises `ProtocolError("invalid_request", ...)` for an index it does not have. `step` runs one
    counted action from a joint-position step and raises `NotImplementedError` (`not_carried`)
    for one it cannot carry. `frames` renders the named cameras. `object_poses` is ground truth:
    the control plane's, never an agent's.
    """

    options: type[BaseModel]

    def start(self) -> str | None:
        """Import the official package and warm up; returns its version or commit."""
        ...

    def open(self, task: TaskName, index: int, options: Any) -> Opened: ...

    def close(self) -> None: ...

    def robot(self) -> RobotState: ...

    def frames(self, cameras: list[str]) -> dict[str, Frame]: ...

    def step(self, qpos: list[float]) -> None: ...

    def verdict(self) -> Verdict: ...

    def object_poses(self) -> dict[str, Pose]: ...


@runtime_checkable
class TruthGeometry(Protocol):
    """What a backend may add to its ground truth (`gt_snapshot`, the control plane's): each
    object's convex hulls in the world (one vertex list per shape), which a truth proxy is measured
    from (`roboscript.demos.gt_scene`), and the table top's height."""

    def object_hulls(self) -> dict[str, list[list[tuple[float, float, float]]]]: ...

    def table_top_z(self) -> float | None: ...


@runtime_checkable
class EndRules(Protocol):
    """What a backend may add to its ground truth: the failures the benchmark ends the open
    episode for that it can state as an object moving or rising past a limit (`EndRule`)."""

    def end_rules(self) -> list[EndRule]: ...


@runtime_checkable
class StageProgress(Protocol):
    """What a backend may add to its ground truth: how far the benchmark's own success checks
    have got in the open episode (`Progress`), where the benchmark tracks stages."""

    def progress(self) -> Progress | None: ...


@runtime_checkable
class SuccessConditions(Protocol):
    """What a backend may add to its ground truth after an episode (`diagnose`): the benchmark's
    success checks evaluated now, with how far their numeric arguments are from flipping them
    (`ConditionTruth`), where the benchmark's checks can be read. Called once the episode's
    result is judged: evaluating a check may touch the benchmark's own bookkeeping."""

    def conditions(self) -> list[ConditionTruth]: ...
