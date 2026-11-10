"""Mount `tape` in a Hub over MockEnv and call it as a model would (through the Hub, so that
visibility, validation and the event log apply). No brain: a call is made directly."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from custom_toolset.tape import Tape
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import BatchId, TaskName, new_id
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import EnvBinding, ToolCall
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.tools import LocalProvider

# `tape.*` shows every tool of the toolset, in the order the class declares them.
PHASE = PhaseSpec.model_validate(
    {
        "name": "tutorial",
        "kind": "learn",
        "visibility": {"actor": ["tape.*", "hub.finish"]},
        "tiers": {"actor": "D"},
    }
)


async def demo(directory: Path, unit: str = "cm") -> list[dict[str, Any] | str]:
    """Measure the arm gap, mark, move the left arm 5 cm along x, measure again."""
    blobs = LocalBlobStore(directory / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=directory / "blobs"), blobs=blobs)
    # The toolset's config is data: the same mapping a run's config file would give.
    provider = LocalProvider(Tape(), {"unit": unit})
    async with Hub([provider], blobs=blobs, phases=[PHASE], env_client=lambda _: env) as hub:
        opened = await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=0))
        binding = EnvBinding(
            url="inproc://mock", episode_id=opened.episode_id, token=opened.agent_token
        )
        spec = OpenEpisode(
            stage="tutorial",
            role="actor",
            env=binding,
            embodiment=str(MOCK_BIMANUAL),
            structured=True,  # outcomes also carry the result as data
        )
        session = await hub.open_episode(spec)
        seen: list[dict[str, Any] | str] = [", ".join(t.name for t in session.tools)]
        count = 0

        async def call(name: str, **arguments: Any) -> dict[str, Any] | str:
            nonlocal count
            count += 1
            made = ToolCall(call_id=f"c{count}", name=name, arguments=arguments, origin="model")
            outcome = await hub.call(session.episode_id, made)
            return outcome.structured or outcome.error_code or ""

        seen.append(await call("arm_gap", first="left", second="right"))
        await call("mark", label="start")
        robot = await env.robot(opened.episode_id)  # stands in for a motion tool
        qpos = list(robot.qpos_cmd)
        qpos[0] += 0.05  # the left arm's ee x, per the embodiment's action layout
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=[ActionStep(qpos=qpos)])
        await env.execute(opened.episode_id, batch)
        seen.append(await call("moved_since", label="start"))
        seen.append(await call("moved_since", label="nowhere"))  # a ToolError becomes an outcome
        seen.append(await call("arm_gap", first="left", second="tail"))
        await hub.close_episode(session.episode_id)
        await env.close(opened.episode_id)
    return seen


if __name__ == "__main__":
    for line in asyncio.run(demo(Path(tempfile.mkdtemp(prefix="tape-")))):
        print(line)
