"""The executor on MockEnv with MockPlanner: what it sends (steps), what the env then shows, and
the receipts, for free and straight moves, replanning after a failed plan, grasps, simulator
errors and the step limit."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.errors import InfraError
from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.contracts.motion import IKRequest, IKSolution, PlanReason, PlanRequest, PlanResult
from roboscript.contracts.robot import Arm
from roboscript.embodiments import EmbodimentProfile
from roboscript.envs.mock import TASKS, MockConfig, MockEnv
from roboscript.manipulation import Executor, ExecutorConfig, Grip, Move
from roboscript.motion import MockPlanner
from roboscript.spatial.rotation import quat_from_axis_angle, quat_multiply
from roboscript.testing.skills import MOCK_EXECUTOR

DOWN = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # the mock gripper points down
Answer = Callable[[PlanRequest, PlanResult], PlanResult]


class ScriptedPlanner(MockPlanner):
    """MockPlanner whose next plans `answers` rewrite (given the request and MockPlanner's
    straight joint-space plan); it records every plan and IK request."""

    def __init__(self) -> None:
        super().__init__()
        self.answers: list[Answer] = []
        self.plans: list[PlanRequest] = []
        self.iks: list[IKRequest] = []

    async def plan(self, request: PlanRequest) -> PlanResult:
        self.plans.append(request)
        result = await super().plan(request)
        return self.answers.pop(0)(request, result) if self.answers else result

    async def ik(self, request: IKRequest) -> list[IKSolution]:
        self.iks.append(request)
        return await super().ik(request)


def refused(reason: PlanReason = "no_path") -> Answer:
    return lambda request, result: PlanResult(reason=reason, detail="scripted", dt=result.dt)


def reshaped(change: Callable[[np.ndarray], np.ndarray]) -> Answer:
    """A plan along `change(path)` (joints of the mock robot = its ee pose) with its ee path."""

    def answer(request: PlanRequest, result: PlanResult) -> PlanResult:
        path = change(np.asarray(result.path))
        ee = [Pose.from_list(ee_of(q)) for q in path]
        return result.model_copy(update={"path": path.tolist(), "ee_path": ee})

    return answer


def ee_of(q: np.ndarray) -> list[float]:
    """The mock robot's ee pose for joints (x, y, z, roll, pitch, yaw), R = Rz Ry Rx."""
    about = [quat_from_axis_angle(axis, q[3 + i]) for i, axis in enumerate(np.eye(3))]
    return [*q[:3], *quat_multiply(about[2], quat_multiply(about[1], about[0]))]


@dataclass
class Rig:
    env: MockEnv
    episode: EpisodeId
    planner: ScriptedPlanner

    def executor(self, config: ExecutorConfig | None = None, **kwargs: float) -> Executor:
        profile = self.env.profile
        tuning = config or MOCK_EXECUTOR
        return Executor(self.env, self.episode, self.planner, profile, tuning, **kwargs)

    async def tcp(self, arm: Arm) -> np.ndarray:
        return np.array((await self.env.robot(self.episode)).arms[arm].tcp.p)


async def make_rig(tmp_path: Path, task: str = "lift_red", step_limit: int | None = None) -> Rig:
    tasks = dict(TASKS)
    if step_limit is not None:
        tasks[TaskName(task)] = tasks[TaskName(task)].model_copy(update={"step_limit": step_limit})
    env = MockEnv(MockConfig(task=TaskName(task), blob_dir=tmp_path, tasks=tasks))
    opened = await env.open(EpisodeSpec(task=TaskName(task), seed=3, trace_expert=True))
    return Rig(env, opened.episode_id, ScriptedPlanner())


@pytest.fixture
async def mock_rig(tmp_path: Path) -> Rig:
    return await make_rig(tmp_path)


def at(x: float, y: float, z: float) -> Pose:
    """The mock ee pose that puts the TCP (0.1 m below it, gripper down) at (x, y, z)."""
    return Pose(p=(x, y, z + 0.1), q=DOWN)


HOME_TCP = {"left": (-0.25, -0.2, 0.85), "right": (0.25, -0.2, 0.85)}


def bulge(path: np.ndarray) -> np.ndarray:
    """The path bent 5 cm sideways in the middle (a curved plan, as cuRobo's often are)."""
    out = path.copy()
    out[:, 1] += 0.05 * np.sin(np.pi * np.linspace(0.0, 1.0, len(path)))
    return out


def wrap(path: np.ndarray) -> np.ndarray:
    """The same ends, but the yaw joint turns a full turn on the way."""
    out = path.copy()
    out[:, 5] += np.linspace(0.0, 2 * math.pi, len(path))
    return out


async def test_a_free_move_takes_one_step_to_its_target(mock_rig: Rig) -> None:
    receipt = await mock_rig.executor().run(left=[Move(at(-0.2, -0.1, 0.9))])
    assert (receipt.reason, receipt.arm, receipt.steps_used) == ("ok", "left", 1)
    assert receipt.commanded is not None and receipt.commanded.tcp.p == pytest.approx(
        (-0.2, -0.1, 0.9)
    )
    assert receipt.deviation_m == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(await mock_rig.tcp("left"), (-0.2, -0.1, 0.9))
    assert np.allclose(await mock_rig.tcp("right"), HOME_TCP["right"])  # held its drive target
    assert list(receipt.tcp_path) == ["left"] and receipt.steps_left == 39
    assert receipt.note == "left 0 mm from its target"


async def test_a_straight_move_is_ik_along_the_line_not_a_plan(mock_rig: Rig) -> None:
    receipt = await mock_rig.executor().run(left=[Move(at(-0.25, -0.2, 0.75), straight=True)])
    assert receipt.reason == "ok" and receipt.steps_used == 1
    assert not mock_rig.planner.plans
    [ik] = mock_rig.planner.iks
    heights = [round(goal.p[2] - 0.1, 3) for goal in ik.goals]  # 2 cm apart down the line
    assert heights == [0.83, 0.81, 0.79, 0.77, 0.75]
    assert np.allclose(await mock_rig.tcp("left"), (-0.25, -0.2, 0.75))


def away(q: list[float]) -> list[float]:
    return [q[0] + 0.05, *q[1:]]


async def test_a_failed_plan_is_planned_again_up_to_three_times(mock_rig: Rig) -> None:
    mock_rig.planner.answers = [refused()] * 3
    assert (await mock_rig.executor().run(left=[Move(at(-0.2, -0.1, 0.9))])).reason == "ok"
    assert len(mock_rig.planner.plans) == 4
    mock_rig.planner.answers = [refused()] * 4
    receipt = await mock_rig.executor().run(left=[Move(at(-0.1, -0.1, 0.9))])
    assert (receipt.reason, receipt.steps_used, receipt.ok) == ("plan_failed", 0, False)
    assert receipt.hard_failure and "no_path" in receipt.note and "tried 4 times" in receipt.note
    assert len(mock_rig.planner.plans) == 8
    assert (await mock_rig.env.robot(mock_rig.episode)).step == 1  # nothing more executed


async def test_a_grasp_settles_holds_and_lifts(mock_rig: Rig) -> None:
    red = (await mock_rig.env.gt_snapshot(mock_rig.episode)).objects["red"].p
    executor = mock_rig.executor()
    grasp = [
        Move(at(red[0], red[1], red[2] + 0.1)),
        Move(at(*red), straight=True, contact=True),
        Grip(0.0, grasp=True),
    ]
    plan = await executor.plan(left=grasp)
    assert [step.until_gripper_stable for step in plan.steps] == [False, False, True]
    assert plan.steps[-1].max_repeats == 3
    receipt = await executor.run(left=grasp)
    assert receipt.reason == "ok" and receipt.steps_used == 4  # 2 moves, close, 1 settle hold
    assert receipt.gripper is not None and receipt.gripper.holding
    assert receipt.gripper.measured == pytest.approx(0.05 / 0.08)
    assert "holding (opening 0.625)" in receipt.note
    lift = await executor.run(left=[Move(at(red[0], red[1], red[2] + 0.1))])
    assert lift.reason == "ok" and executor.succeeded
    assert "success" not in lift.note  # a receipt shows what a robot measures
    assert (await mock_rig.env.judge(mock_rig.episode)).success


def dip(path: np.ndarray) -> np.ndarray:
    """The path sagging 10 cm in the middle."""
    out = path.copy()
    out[:, 2] -= 0.1 * np.sin(np.pi * np.linspace(0.0, 1.0, len(path)))
    return out


async def test_a_simulator_error_is_an_infrastructure_fault(
    mock_rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        error = "the simulator raised RuntimeError"
        return ExecuteResult(
            batch_id=batch.batch_id, states=[], executed=0, stopped="error", error=error
        )

    monkeypatch.setattr(mock_rig.env, "execute", failing)
    with pytest.raises(InfraError, match="the simulator raised RuntimeError") as caught:
        await mock_rig.executor().run(left=[Move(at(-0.2, -0.1, 0.9))])
    assert caught.value.code == "sim_fault"


async def test_the_step_limit_ends_an_action(tmp_path: Path) -> None:
    rig = await make_rig(tmp_path, step_limit=2)
    rig.planner.answers = [reshaped(bulge)]
    receipt = await rig.executor().run(left=[Move(at(-0.1, -0.1, 0.8))])
    assert (receipt.reason, receipt.steps_used, receipt.steps_left) == ("budget_exhausted", 2, 0)
    assert receipt.note.startswith("the step limit (2) was reached")


def stopped(profile: EmbodimentProfile, low_x: float, high_x: float) -> EmbodimentProfile:
    """The profile with the left arm's x joint stopping at [low_x, high_x] (the mock's limits
    allow more), as a wrist stops before the limits of its planner model."""
    wide = [(-10.0, 10.0)] * 6
    left = profile.arms["left"].model_copy(update={"stops": [(low_x, high_x), *wide[1:]]})
    return profile.model_copy(update={"arms": {**profile.arms, "left": left}})


def turn(q: list[float]) -> list[float]:
    """A whole turn about the gripper's axis: the same pose, as a solver may return it."""
    return [*q[:5], q[5] + 2 * math.pi]


def box(x: float, y: float, z: float, size: float) -> OrientedBox:
    return OrientedBox(center=(x, y, z), size=(size, size, size), q=(1.0, 0.0, 0.0, 0.0))
