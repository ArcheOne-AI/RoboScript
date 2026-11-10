"""MockEnv: the env conformance suite over RPC (both ports, in-process) and its own behavior."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.ids import BatchId, EpisodeId, TaskName, new_id
from roboscript.envs import env_apps
from roboscript.envs.mock import TABLE_Z, MockConfig, MockEnv
from roboscript.rpc import client
from roboscript.testing.env import EnvConformance, EnvTarget

TOKEN = SecretStr("control-token-for-tests")
UNSOLVABLE = 13


@pytest.fixture
async def env_target(tmp_path: Path) -> AsyncIterator[EnvTarget]:
    config = MockConfig(
        task="lift_red", blob_dir=tmp_path, step_delay_s=0.01, unsolvable_seeds=[UNSOLVABLE]
    )
    env = MockEnv(config)
    control_app, agent_app = env_apps(env, TOKEN)
    env.agent_url = "http://agent"
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(control_app)) as control_http,
        httpx.AsyncClient(transport=httpx.ASGITransport(agent_app)) as agent_http,
    ):

        def connect(interface: type[Any], url: str, token: SecretStr | None) -> Any:
            http = agent_http if url == env.agent_url else control_http
            return client(interface, url, token=token, http=http, timeout=10)

        yield EnvTarget(
            url="http://control",
            token=TOKEN,
            connect=connect,
            profile=env.profile,
            task=TaskName("lift_red"),
            seed=7,
            unsolvable_seed=UNSOLVABLE,
        )


class TestMockEnvConformance(EnvConformance):
    exhaust_step_limit = True
    traces_expert = True


@pytest.fixture
def mock(tmp_path: Path) -> MockEnv:
    return MockEnv(MockConfig(task="lift_red", blob_dir=tmp_path, unsolvable_seeds=[UNSOLVABLE]))


def batch(*qpos: np.ndarray, stable: bool = False) -> ActionBatch:
    steps = [ActionStep(qpos=list(q), until_gripper_stable=stable) for q in qpos]
    return ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)


async def grasp(mock: MockEnv, episode: EpisodeId, box: str) -> np.ndarray:
    """Move the left TCP onto `box`, close (until stable), lift 10 cm; returns the last qpos."""
    target = (await mock.gt_snapshot(episode)).objects[box].p
    qpos = np.array((await mock.robot(episode)).qpos_cmd)
    qpos[:3] = [target[0], target[1], target[2] + 0.1]  # the TCP is 0.1 m below the ee
    closed = qpos.copy()
    closed[6] = 0.0
    lifted = closed.copy()
    lifted[2] += 0.1
    result = await mock.execute(episode, batch(qpos, closed))
    assert result.stopped == "completed"
    await mock.execute(episode, batch(lifted))
    return lifted


async def test_grasp_lift_and_success_latch(mock: MockEnv) -> None:
    opened = await mock.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))
    episode = opened.episode_id
    before = (await mock.robot(episode)).arms["left"].gripper
    lifted = await grasp(mock, episode, "red")
    state = await mock.robot(episode)
    assert before == 1.0 and state.arms["left"].gripper == pytest.approx(0.05 / 0.08)
    assert state.arms["left"].gripper_cmd == 0.0  # the command says closed, the fingers hold a box
    verdict = await mock.judge(episode)
    assert verdict.success and verdict.success_step == 3 and verdict.steps_used == 3
    red = (await mock.gt_snapshot(episode)).objects["red"].p
    assert red[2] == pytest.approx(TABLE_Z + 0.025 + 0.1)
    after = await mock.execute(episode, batch(lifted))  # success latched: nothing more runs
    assert (after.stopped, after.executed) == ("success", 0)
    diagnosis = await mock.diagnose(episode)
    moved = {o.name: o.displacement_m for o in diagnosis.objects}
    assert moved["red"] == pytest.approx(0.1) and moved["blue"] == 0.0
    [box] = [o for o in diagnosis.objects if o.name == "red"]
    assert box.final_center is not None and box.final_center[2] == pytest.approx(red[2])
    assert diagnosis.notes == []  # notes name no object


async def test_scenes_follow_the_seed(mock: MockEnv) -> None:
    def poses(snapshot: Any) -> dict[str, list[float]]:
        return {name: pose.as_list() for name, pose in snapshot.objects.items()}

    first = await mock.open(EpisodeSpec(task=TaskName("lift_red"), seed=1))
    one = poses(await mock.gt_snapshot(first.episode_id))
    again = await mock.open(EpisodeSpec(task=TaskName("lift_red"), seed=1))
    assert poses(await mock.gt_snapshot(again.episode_id)) == one
    other = await mock.open(EpisodeSpec(task=TaskName("lift_red"), seed=2))
    assert poses(await mock.gt_snapshot(other.episode_id)) != one
    paraphrase = await mock.open(
        EpisodeSpec(task=TaskName("lift_red"), seed=1, options={"instructions": "paraphrase"})
    )
    texts = (first.instruction, paraphrase.instruction)
    assert texts == ("lift the red box", "raise the red block")
    free = EpisodeSpec(task=TaskName("lift_red"), seed=1, options={"step_limit": False})
    assert (await mock.open(free)).step_limit is None
    for refused in ({"language": "en"}, {"instructions": "other"}):
        spec = EpisodeSpec(task=TaskName("lift_red"), seed=1, options=refused)
        with pytest.raises(ProtocolError, match="reset options"):
            await mock.open(spec)
