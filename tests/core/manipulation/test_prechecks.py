"""The executor's pre-checks on MockEnv: the arms' distance moving together, and P3's refusal of a
repeated failure (the action API's)."""

import math
from pathlib import Path

import numpy as np

from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.manipulation import (
    ActionApi,
    Executor,
    GraspSpec,
    Move,
    ObjectProxy,
    Part,
)
from roboscript.motion import MockPlanner
from roboscript.testing.skills import MOCK_EXECUTOR

PROFILE = load_embodiment(MOCK_BIMANUAL)
DOWN = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)


def at(x: float, y: float, z: float) -> Pose:
    """The mock ee pose putting the TCP (0.1 m below it, gripper down) at (x, y, z)."""
    return Pose(p=(x, y, z + 0.1), q=DOWN)


async def executor(tmp_path: Path, **config: object) -> Executor:
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path))
    opened = await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))
    return Executor(
        env, opened.episode_id, MockPlanner(), PROFILE, MOCK_EXECUTOR.model_copy(update=config)
    )  # type: ignore[arg-type]


async def test_arms_moving_together_keep_apart(tmp_path: Path) -> None:
    ex = await executor(tmp_path)
    close = await ex.run(left=[Move(at(-0.01, 0.0, 0.9))], right=[Move(at(0.01, 0.0, 0.9))])
    assert close.reason == "invalid_args" and close.steps_used == 0
    assert close.note == (
        "the TCPs would come within 2.0 cm of each other (from 50.0 cm); moving together keeps "
        "them 3 cm apart"
    )
    apart = await ex.run(left=[Move(at(-0.05, 0.0, 0.9))], right=[Move(at(0.05, 0.0, 0.9))])
    assert apart.reason == "ok"
    # One arm at a time is not checked (hand-overs).
    alone = await ex.run(left=[Move(at(0.02, 0.0, 0.9))])
    assert alone.reason == "ok"


async def api(tmp_path: Path, **config: object) -> ActionApi:
    ex = await executor(tmp_path, **config)
    return ActionApi(ex)


async def test_p3_refuses_repeating_the_call_that_just_failed(tmp_path: Path) -> None:
    p3 = await api(tmp_path, reject_repeats=True)
    beyond = await p3.move_to_pose("left", at(0.9, 0.0, 0.9))
    first = await p3.run(beyond)
    assert first.reason == "unreachable"
    again = await p3.run(await p3.move_to_pose("left", at(0.903, 0.0, 0.9)))
    assert again.reason == "rejected_repeat" and again.steps_used == 0
    assert "repeats the one that just failed" in again.note
    still = await p3.run(beyond)
    assert still.reason == "rejected_repeat"  # a rejection keeps the failure to compare with
    reach = await p3.check_reach(beyond)  # planning only: never refused so, forgets nothing
    assert reach.reason == "unreachable"
    assert (await p3.run(beyond)).reason == "rejected_repeat"
    other = await p3.run(await p3.move_to_pose("left", at(0.92, 0.0, 0.9)))
    assert other.reason == "unreachable"  # 2 cm away: a new try
    fine = await p3.run(await p3.move_to_pose("left", at(0.1, 0.0, 0.9)))
    assert fine.reason == "ok"
    assert (await p3.run(beyond)).reason == "unreachable"  # the failure before is forgotten
    await p3.hold(1)  # a hold forgets it too
    assert (await p3.run(beyond)).reason == "unreachable"
    # A call the API refuses (nothing plannable) counts as a failure: its repeat is refused.
    part = Part(frame=np.eye(4), size=np.full(3, 0.05), x_known=False)
    far = ObjectProxy("thing", np.eye(4) + np.diag([0, 0, 0, 0]), "half_pi", {"whole": part})
    far = far.at(np.array([0.0, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0]))
    grasp = GraspSpec(approach="top", offset_m=(0.0, 0.0, 0.29))
    refused = await p3.run(await p3.grasp_object("left", far, grasp))
    assert refused.reason == "unreachable" and "no pre-grasp" in refused.note
    assert (await p3.run(await p3.grasp_object("left", far, grasp))).reason == "rejected_repeat"
    s2 = await api(tmp_path)
    await s2.run(beyond)
    assert (await s2.run(beyond)).reason == "unreachable"  # off outside P3


def state(step: int, opening: float, command: float) -> RobotState:
    tcp = Pose(p=(0.0, 0.0, 0.9), q=DOWN)
    arm = ArmState(
        ee=tcp,
        tcp=tcp,
        joints=[0.0] * 6,
        joints_cmd=[0.0] * 6,
        gripper=opening,
        gripper_cmd=command,
    )
    rest = arm.model_copy(update={"gripper": 1.0, "gripper_cmd": 1.0})
    return RobotState(
        step=step, step_limit=40, arms={"left": arm, "right": rest}, qpos_cmd=[0.0] * 14
    )
