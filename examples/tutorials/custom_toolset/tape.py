"""A toolset of your own: `tape`, a tape measure on the robot's own arms.

A toolset is a class with a `name`, an optional pydantic `Config` and `@tool` methods. It reads
the episode only through `ctx` (here the env's agent plane: proprioception, nothing a real robot
lacks), so it is tier D. The first paragraph of a docstring is what the model reads.
"""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.errors import ToolError
from roboscript.contracts.robot import ArmName, RobotState
from roboscript.tools import ToolContext, Toolset, tool


class TapeConfig(BaseModel):
    """The toolset's section of a run's config. Unknown keys are refused (conformance checks it)."""

    model_config = ConfigDict(extra="forbid")

    unit: Literal["m", "cm"] = "m"


class Gap(BaseModel):
    distance: float
    unit: Literal["m", "cm"]


class Moved(BaseModel):
    label: str
    distance: dict[ArmName, float]
    unit: Literal["m", "cm"]


class Tape(Toolset):
    name = "tape"
    Config = TapeConfig
    config: TapeConfig  # set before the first call

    def _scale(self) -> float:
        return 100.0 if self.config.unit == "cm" else 1.0

    @tool(kind="perception", tier="D")
    async def arm_gap(
        self,
        ctx: ToolContext,
        first: Annotated[ArmName, Field(description="An arm, by the name the robot gives it.")],
        second: Annotated[ArmName, Field(description="The other arm.")],
    ) -> Gap:
        """How far apart the two arms' TCPs (fingertip midpoints) are now."""
        robot = await ctx.env.robot(ctx.env_episode)
        a, b = _tcp(robot, first), _tcp(robot, second)
        return Gap(distance=math.dist(a, b) * self._scale(), unit=self.config.unit)

    @tool(kind="pure", tier="D")
    async def mark(self, ctx: ToolContext, label: Annotated[str, Field(max_length=40)]) -> Moved:
        """Remember where every TCP is now under `label`."""
        robot = await ctx.env.robot(ctx.env_episode)
        marks = ctx.state.setdefault("marks", {})  # this toolset's dict for this episode only
        marks[label] = {arm: _tcp(robot, arm) for arm in robot.arms}
        return Moved(label=label, distance=dict.fromkeys(robot.arms, 0.0), unit=self.config.unit)

    @tool(kind="perception", tier="D")
    async def moved_since(
        self, ctx: ToolContext, label: Annotated[str, Field(max_length=40)]
    ) -> Moved:
        """How far each arm's TCP has moved since `mark(label)`."""
        marks = ctx.state.get("marks", {})
        if label not in marks:
            raise ToolError(
                f"no mark named {label!r}; marks: {sorted(marks)}"
            )  # the model reads it
        robot = await ctx.env.robot(ctx.env_episode)
        distance = {
            arm: math.dist(_tcp(robot, arm), before) * self._scale()
            for arm, before in marks[label].items()
        }
        return Moved(label=label, distance=distance, unit=self.config.unit)


def _tcp(robot: RobotState, arm: str) -> tuple[float, float, float]:
    if arm not in robot.arms:
        raise ToolError(f"no arm named {arm!r}; this robot has {sorted(robot.arms)}")
    return robot.arms[arm].tcp.p
