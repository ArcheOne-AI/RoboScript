"""The BEHAVIOR adapter skeleton on a fake official module: its descriptor and embodiment pass the
benchmark and embodiment suites, its env server passes the env suite, and BEHAVIOR's
particulars are checked: its declared capabilities, a terminal q-score with no latch, and a joint
step that fills the torso, arms and grippers and leaves the base still."""

from collections.abc import AsyncIterator
from pathlib import Path

import fakes
import numpy as np
import pytest
from serving import batch, open_episode, served

from roboscript.blobs import LocalBlobStore
from roboscript.testing.benchmarks import BenchmarkConformance, BenchmarkTarget
from roboscript.testing.embodiments import EmbodimentConformance, EmbodimentTarget
from roboscript.testing.env import EnvConformance, EnvTarget
from roboscript_behavior.benchmark import BEHAVIOR
from roboscript_behavior.config import BehaviorConfig
from roboscript_behavior.sim.host import make_env


@pytest.fixture
async def env_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[EnvTarget]:
    fakes.install_behavior(monkeypatch)
    # an env that reloads an instance: the suite reopens resets
    config = BehaviorConfig(task="turning_on_radio", blob_dir=tmp_path, once_per_reset=False)
    env = make_env(config, LocalBlobStore(tmp_path))
    async with served(env, seed=3, reset_by="initial_state") as target:
        yield target


@pytest.fixture
def benchmark_target() -> BenchmarkTarget:
    return BenchmarkTarget("roboscript_behavior.benchmark:BEHAVIOR")


@pytest.fixture
def embodiment_target() -> EmbodimentTarget:
    return EmbodimentTarget(BEHAVIOR.embodiment)


class TestBehaviorEnvConformance(EnvConformance):
    pass


class TestBehaviorBenchmarkConformance(BenchmarkConformance):
    pass


class TestR1ProConformance(EmbodimentConformance):
    pass


def test_the_descriptor_declares_its_capabilities() -> None:
    profile = BEHAVIOR.profile()
    assert list(profile.arms) == ["left", "right"] and list(profile.joint_groups) == ["torso"]
    assert profile.actions.kinds() == {"joint_position", "base"}
    assert profile.actions.base is not None and profile.actions.base.holonomic
    capabilities = BEHAVIOR.capabilities
    assert capabilities.success == {"terminal_score"} and capabilities.reset == {"initial_state"}
    assert capabilities.reruns is False and not capabilities.expert


async def test_the_verdict_is_a_terminal_q_score_with_no_latch(env_target: EnvTarget) -> None:
    episode = await open_episode(env_target)
    start = await episode.control.judge(episode.id)
    assert (start.success, start.success_step, start.score) == (False, None, 0.0)
    rest = await episode.rest()
    left_closed = rest.copy()
    left_closed[7] = 0.0  # the left gripper's normalized command: closed
    half = await episode.agent.execute(episode.id, batch(left_closed, left_closed))
    assert half.stopped == "completed"
    assert (await episode.control.judge(episode.id)).score == 0.5
    # all four goals are met after two more steps; the batch does not stop there (no latch)
    rest_of_it = await episode.agent.execute(episode.id, batch(*[left_closed] * 5))
    assert (rest_of_it.stopped, rest_of_it.executed) == ("completed", 5)
    final = await episode.control.judge(episode.id)
    assert (final.success, final.success_step, final.score) == (True, None, 1.0)
    assert final.steps_used == 7


async def test_a_joint_step_fills_the_torso_arms_and_grippers_and_leaves_the_base_still(
    env_target: EnvTarget,
) -> None:
    episode = await open_episode(env_target)
    qpos = np.array(await episode.rest())
    qpos[0] = 0.4  # the left arm's first joint
    qpos[8] = -0.3  # the right arm's
    qpos[16:20] = [0.1, 0.2, 0.3, 0.4]  # the torso
    result = await episode.agent.execute(episode.id, batch(qpos.tolist()))
    assert result.executed == 1
    action = fakes.FakeBehaviorEnv.instances[-1].actions[-1]
    assert action[0:3].tolist() == [0.0, 0.0, 0.0]  # the base
    assert action[3:7].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert action[7] == pytest.approx(0.4) and action[15] == pytest.approx(-0.3)
    state = result.states[0].robot
    assert state.arms["left"].joints_cmd[0] == pytest.approx(0.4)
    assert state.qpos_cmd[16:20] == pytest.approx([0.1, 0.2, 0.3, 0.4])
