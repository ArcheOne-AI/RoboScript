"""The privileged toolset: the simulator's objects at tier P (S1 only in the test method), and
only on an env that declares ground-truth snapshots."""

from pathlib import Path

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import TaskName
from roboscript.contracts.tools import EnvBinding, ToolCall
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider
from roboscript.toolsets.privileged import Privileged, TrueObjects, mountable


async def test_true_objects_are_s1_s_alone(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs)
    async with Hub(
        [LocalProvider(Privileged(env))], blobs=blobs, phases=TEST_PHASES, env_client=lambda _: env
    ) as hub:
        opened = await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))
        binding = EnvBinding(
            url="inproc://mock", episode_id=opened.episode_id, token=opened.agent_token
        )

        async def episode(stage: str, role: str, tier: str) -> str:
            spec = OpenEpisode(
                stage=stage,
                role=role,
                tier=tier,
                env=binding,
                embodiment=str(MOCK_BIMANUAL),
                structured=True,
            )  # type: ignore[arg-type]
            return (await hub.open_episode(spec)).episode_id

        seen = {}
        for stage, role, tier in (
            ("s1", "actor", "P"),
            ("s1", "actor", "D"),
            ("s1", "reviser", "P"),
        ):
            ep = await episode(stage, role, tier)
            seen[stage, role, tier] = [
                t.name for t in await hub.tools(ep) if t.name == "true_objects"
            ]
        privileged = await episode("s1", "demo_reader", "P")
        call = ToolCall(call_id="c1", name="true_objects", arguments={}, origin="model")
        found = await hub.call(privileged, call)
    assert seen == {
        ("s1", "actor", "P"): ["true_objects"],
        ("s1", "actor", "D"): [],  # tier D: no privileged tool
        ("s1", "reviser", "P"): [],  # no live scene
    }
    assert not found.is_error
    names = {o.name for o in TrueObjects.model_validate(found.structured).objects}
    assert "red" in names


@pytest.mark.parametrize("stage", ["s2", "p3"])
async def test_s2_and_p3_never_see_the_truth(tmp_path: Path, stage: str) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs)
    async with Hub([LocalProvider(Privileged(env))], blobs=blobs, phases=TEST_PHASES) as hub:
        for role in ("actor", "reviser"):
            session = await hub.open_episode(OpenEpisode(stage=stage, role=role))  # type: ignore[arg-type]
            assert not [t for t in await hub.tools(session.episode_id) if t.name == "true_objects"]


def test_it_is_mountable_only_where_the_env_declares_truth_snapshots() -> None:
    assert mountable(EnvCapabilities(truth=True)) == []
    assert mountable(EnvCapabilities()) == ["the env does not declare truth"]
    robot = EnvCapabilities(real_robot=True, success=frozenset({"external"}))
    assert mountable(robot) == ["the env does not declare truth"]
