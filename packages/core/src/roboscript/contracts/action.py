"""Joint-space action batches for `env.agent.execute` (docs/concepts/embodied-core.md),
and the kinds of action an embodiment declares it takes (`ActionSpace`).

`execute` carries joint positions only; the other kinds are declared, so that a skill needing
one is refused on an embodiment without it (`skills.check_interface`).
"""

from typing import Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.ids import BatchId
from roboscript.contracts.robot import ArmName, RobotState

ActionKind = Literal["joint_position", "ee_delta", "base"]
"""What one action commands: joint positions (the embodiment's joint vector), end-effector
deltas, or a mobile base's motion."""
ACTION_KINDS: Final[tuple[ActionKind, ...]] = ("joint_position", "ee_delta", "base")


class EeDeltaControl(BaseModel):
    """End-effector delta control: an action moves the ee of each of `arms` by a translation (m)
    and a rotation (axis-angle, rad) expressed in `frame` (`world`; `base`, the robot's root; or
    `ee`, the ee's own frame), at most `max_translation_m` and `max_rotation_rad` (None: no
    bound), with a gripper command per arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arms: list[ArmName] = Field(min_length=1)
    frame: Literal["world", "base", "ee"]
    max_translation_m: PositiveFloat | None = None
    max_rotation_rad: PositiveFloat | None = None


class BaseControl(BaseModel):
    """Mobile-base control: an action commands the base's planar velocity in its own frame,
    forward, sideways (only when `holonomic`) and turning, at most `max_speed_m_s` and
    `max_turn_rad_s` (None: no bound)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    holonomic: bool
    max_speed_m_s: PositiveFloat | None = None
    max_turn_rad_s: PositiveFloat | None = None


class ActionSpace(BaseModel):
    """The kinds of action an embodiment takes, at least one: `joint_position`, its joint vector
    (`EmbodimentProfile.action_dim`, laid out by its arms and joint groups); `ee_delta`; `base`.
    The default is joint positions only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    joint_position: bool = True
    ee_delta: EeDeltaControl | None = None
    base: BaseControl | None = None

    @model_validator(mode="after")
    def _some(self) -> Self:
        if not self.kinds():
            raise ValueError("an embodiment takes at least one kind of action")
        return self

    def kinds(self) -> frozenset[ActionKind]:
        flags = (self.joint_position, self.ee_delta is not None, self.base is not None)
        return frozenset(kind for kind, on in zip(ACTION_KINDS, flags, strict=True) if on)


class ActionStep(BaseModel):
    """One counted `take_action(qpos)`. With `until_gripper_stable` it is repeated, at most
    `max_repeats` more times, until the measured openings of the grippers it commands below
    open settle (change by less than 1e-4 in a call); open fingers hold nothing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qpos: list[FiniteFloat]
    until_gripper_stable: bool = False
    max_repeats: NonNegativeInt = 3


class ActionBatch(BaseModel):
    """Steps run in order; executing a `batch_id` again returns the first result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: BatchId
    steps: list[ActionStep] = Field(min_length=1)
    stop_on_success: bool = True


class StepState(BaseModel):
    """The robot after one executed action; `index` is its step in the batch."""

    model_config = ConfigDict(frozen=True)

    index: NonNegativeInt
    robot: RobotState


Stopped = Literal["completed", "success", "step_limit", "ended", "error"]
"""Why a batch ended: all steps ran, the success latch, the step limit, the benchmark ended the
episode before its limit without success (a failure it checks for fired:
no action counts after it), a simulator error."""


class ExecuteResult(BaseModel):
    """One state per executed action; `executed` counts actions, repeats included."""

    model_config = ConfigDict(frozen=True)

    batch_id: BatchId
    states: list[StepState]
    executed: NonNegativeInt
    stopped: Stopped
    error: str | None = None
