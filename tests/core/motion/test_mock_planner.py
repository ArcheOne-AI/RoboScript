"""The mock planner: conformance, agreement with MockEnv's robot, and its own behavior."""

import math
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import BatchId, TaskName
from roboscript.contracts.motion import FKRequest, PlanRequest
from roboscript.contracts.robot import RobotState
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.kernel import entrypoints
from roboscript.motion import MockPlanner
from roboscript.testing.planner import PlannerConformance, PlannerTarget

WORLD = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))  # MockEnv's joints are world poses
HOME = [-0.25, -0.2, 0.95, 0.0, math.pi / 2, 0.0]  # MockEnv's left arm at reset, pointing down
AWAY = [-0.1, -0.05, 0.9, 0.2, 1.2, 0.3]
SAMPLES = {"left": [HOME, AWAY], "right": [[0.25, -0.2, 0.95, 0.0, math.pi / 2, 0.0], AWAY]}


@pytest.fixture
def planner_target() -> PlannerTarget:
    return PlannerTarget(MockPlanner(), WORLD, SAMPLES, lock_drift_deg=1e-6)


class TestMockPlannerConformance(PlannerConformance):
    pass


async def test_fk_is_mockenvs_robot(tmp_path: Path) -> None:
    env = MockEnv(MockConfig(task="lift_red", blob_dir=tmp_path))
    episode = (await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))).episode_id
    planner = MockPlanner(env.profile)
    state = await env.robot(episode)
    await _fk_agrees(planner, state)
    left = state.arms["left"]
    goal = Pose(p=(0.05, 0.1, 0.9), q=left.ee.q)
    plan = await planner.plan(PlanRequest(arm="left", base=WORLD, start=left.joints_cmd, goal=goal))
    qpos = list(state.qpos_cmd)
    qpos[0:6] = (plan.path or [])[-1]  # the left arm's joints in MockEnv's action layout
    await env.execute(episode, ActionBatch(batch_id=BatchId("b"), steps=[ActionStep(qpos=qpos)]))
    moved = await env.robot(episode)
    assert np.allclose(moved.arms["left"].ee.p, goal.p)
    await _fk_agrees(planner, moved)


async def _fk_agrees(planner: MockPlanner, state: RobotState) -> None:
    for arm, spec in state.arms.items():
        fk = await planner.fk(FKRequest(arm=arm, base=WORLD, joints=[spec.joints_cmd]))
        assert np.allclose(fk.ee[0].as_list(), spec.ee.as_list(), atol=1e-12)
        assert np.allclose(fk.tcp[0].as_list(), spec.tcp.as_list(), atol=1e-12)


def test_planners_are_registered_entry_points() -> None:
    try:
        metadata.distribution("roboscript")
    except metadata.PackageNotFoundError:
        pytest.skip("roboscript is not installed here (the simulator env uses PYTHONPATH)")
    assert {"curobo", "mock"} <= set(entrypoints.names("roboscript.planners"))
    assert entrypoints.load("roboscript.planners", "mock") is MockPlanner
