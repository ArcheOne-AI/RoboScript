"""The manipulation toolset over the Hub, MockEnv and MockPlanner: `check_reach` moving nothing,
hard failures under S2 and P3, and the action lock."""

import math
from collections.abc import Callable
from typing import Any

import anyio
import numpy as np
import pytest
from pydantic import BaseModel

from roboscript.contracts.hub import CallOutcome
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.motion import PlanRequest, PlanResult
from roboscript.contracts.receipt import (
    ActionReceipt,
)
from roboscript.contracts.tools import TextPart, ToolCall
from roboscript.envs.mock import TABLE_Z
from roboscript.manipulation import Reach
from roboscript.motion import MockPlanner
from roboscript.tools import TABLE, LocalProvider, ToolContext, Toolset, ToolsetRuntime, tool
from roboscript.toolsets.manipulation import (
    Manipulation,
)

Model = Callable[..., ToolCall]
DOWN = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # the mock gripper points down
HOME = {"left": (-0.25, -0.2, 0.85), "right": (0.25, -0.2, 0.85)}  # MockEnv's start TCPs
CONFIG = {"planner": {"mock": True}, "table_top_z": TABLE_Z}


def ee(x: float, y: float, z: float, q: tuple[float, ...] = DOWN) -> list[float]:
    """The mock ee pose that puts the TCP (0.1 m below it, gripper down) at (x, y, z)."""
    return [x, y, z + 0.1, *q]


def why(stage: str = "reach") -> dict[str, str]:
    return {"stage": stage, "scene": "a test scene", "progress": "testing", "plan": "move"}


def manipulation(toolset: Manipulation | None = None, **config: Any) -> LocalProvider:
    """The toolset with the Mock benchmark's skillset (its bindings, no skills of its own)."""
    return LocalProvider(toolset or Manipulation(), CONFIG | config, ["mock"])


def receipt_of(outcome: CallOutcome) -> ActionReceipt:
    assert not outcome.is_error, outcome.content
    return ActionReceipt.model_validate(outcome.structured)


def text_of(outcome: CallOutcome) -> str:
    return "\n".join(part.text for part in outcome.content if isinstance(part, TextPart))


class Episode:
    """A Hub episode on MockEnv: `act` calls an action as a model does, `tcp` measures."""

    def __init__(self, rig: Any, hub: Any, model: Model, ep: EpisodeId, env_ep: EpisodeId) -> None:
        self.rig, self.hub, self.model, self.ep, self.env_ep = rig, hub, model, ep, env_ep

    async def call(self, tool: str, **arguments: Any) -> CallOutcome:
        if tool != "check_reach":
            arguments = {"rationale": why(), **arguments}
        return await self.hub.call(self.ep, self.model(tool, **arguments))

    async def act(self, tool: str, **arguments: Any) -> ActionReceipt:
        return receipt_of(await self.call(tool, **arguments))

    async def tcp(self, arm: str) -> np.ndarray:
        return np.array((await self.rig.env.robot(self.env_ep)).arms[arm].tcp.p)

    async def step(self) -> int:
        return int((await self.rig.env.robot(self.env_ep)).step)


async def episode(rig: Any, hub: Any, model: Model, stage: str = "s1") -> Episode:
    ep, env_ep = await rig.episode(hub, stage=stage)
    return Episode(rig, hub, model, ep, env_ep)


class Gated(MockPlanner):
    """MockPlanner whose free plans wait until `gate` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.gate = anyio.Event(), anyio.Event()

    async def plan(self, request: PlanRequest) -> PlanResult:
        self.entered.set()
        await self.gate.wait()
        return await super().plan(request)


class Planned(Manipulation):
    """The toolset on a given planner."""

    def __init__(self, planner: MockPlanner, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.given = planner

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.planner = self.given


class Done(BaseModel):
    ok: bool = True


class Tables(Toolset):
    """Stands in for perception: sets the table estimate in the episode's store."""

    name = "tables"

    @tool(kind="pure", tier="D")
    async def set_table(self, ctx: ToolContext, z: float) -> Done:
        """Set the table estimate."""
        ctx.frames.set_fact(TABLE, {"z": z, "extent": [-0.6, 0.6, -0.35, 0.35]})
        return Done()


async def test_check_reach_plans_and_moves_nothing(rig: Any, model: Model) -> None:
    async with rig.hub(manipulation()) as hub:
        robot = await episode(rig, hub, model, stage="dev")  # a development tool
        near = await robot.call(
            "check_reach", left={"action": "move_to_pose", "pose": ee(0, 0, 0.9)}
        )
        found = Reach.model_validate(near.structured)
        assert (found.reachable, found.reason, found.steps) == (True, "ok", 1)
        assert found.targets["left"].p == pytest.approx((0.0, 0.0, 0.9))
        assert text_of(near).startswith("Reachable (ok): 1 steps planned. Nothing moved.")
        far = await robot.call(
            "check_reach", right={"action": "move_to_pose", "pose": ee(0.9, 0, 0.9)}
        )
        missed = Reach.model_validate(far.structured)
        assert (missed.reachable, missed.reason) == (False, "unreachable")
        assert text_of(far).startswith(
            "Not reachable (unreachable): run as it is, it would be a hard failure."
        )
        both = await robot.call(
            "check_reach",
            left={"action": "move_to_pose", "pose": ee(0.23, -0.2, 0.85)},
            right={"action": "open_gripper"},
        )
        apart = Reach.model_validate(both.structured)
        assert (apart.reachable, apart.reason) == (False, "invalid_args")  # the arms' gap
        assert set(apart.targets) == {"left"} and "within 2.0 cm" in apart.note
        nothing = await robot.call("check_reach")
        assert nothing.is_error and "give an action" in text_of(nothing)
        assert await robot.step() == 0  # nothing moved
        assert not far.episode_ended and (await hub.state(robot.ep)).ended is None


@pytest.mark.parametrize(("stage", "ends"), [("s2", True), ("p3", False)])
async def test_a_hard_failure_ends_s2_and_p3_goes_on(
    rig: Any, model: Model, stage: str, ends: bool
) -> None:
    async with rig.hub(manipulation()) as hub:
        robot = await episode(rig, hub, model, stage=stage)
        far = await robot.call("move_to_pose", arm="left", pose=ee(0.9, 0.0, 0.9))
        receipt = receipt_of(far)
        assert (receipt.reason, receipt.steps_used, far.episode_ended) == ("unreachable", 0, ends)
        text = text_of(far)
        assert text.startswith("Receipt: unreachable (hard failure).\nleft TCP commanded")
        status = await hub.state(robot.ep)
        if ends:
            assert status.ended is not None and status.ended.reason == "fail_fast:unreachable@reach"
            assert text.endswith("Hard failure (unreachable): the episode has ended.")
        else:
            assert status.ended is None and "the attempt goes on" in text
            assert (await robot.act("move_by", arm="left", z=0.05)).ok


async def test_one_action_at_a_time(rig: Any, model: Model) -> None:
    gated = Gated()
    async with rig.hub(manipulation(Planned(gated))) as hub:
        robot = await episode(rig, hub, model, stage="dev")  # with check_reach
        done: dict[str, CallOutcome] = {}

        async def first() -> None:
            done["first"] = await robot.call("move_to_pose", arm="left", pose=ee(0, 0, 0.9))

        async with anyio.create_task_group() as group:
            group.start_soon(first)
            await gated.entered.wait()
            second = await robot.call("open_gripper", arm="right")
            assert second.is_error and second.error_code == "action_in_progress"
            meanwhile = await robot.call("check_reach", right={"action": "open_gripper"})
            assert not meanwhile.is_error  # planning is no action
            gated.gate.set()
        assert receipt_of(done["first"]).ok
        assert (await robot.act("open_gripper", arm="right")).reason == "ok"
