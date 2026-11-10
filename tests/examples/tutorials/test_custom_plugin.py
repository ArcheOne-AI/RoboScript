"""The `forbid_arm` guard and the `tool_tally` sink pass their conformance suites, named by
import path (the tutorial is not an installed package), and work in a composed run."""

import json
from pathlib import Path

import pytest
from custom_plugin.plugins import FORBID_ARM, TOOL_TALLY

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import TaskName
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import EnvBinding, ToolCall
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub, load_guard
from roboscript.learning.compose import compose_run, open_sinks
from roboscript.learning.events import EventRouter
from roboscript.learning.runspec import PluginItem
from roboscript.records.sinks import load_sink
from roboscript.testing.guards import GuardConformance
from roboscript.testing.hold import Hold
from roboscript.testing.plugins import SinkConformance
from roboscript.tools import LocalProvider

pytestmark = pytest.mark.unit

# Not installed, so named by import path; an installed package names them by the entry-point
# names its pyproject.toml registers (`forbid_arm`, `tool_tally`), and the rest reads the same.
GUARD = "custom_plugin.plugins:FORBID_ARM"
SINK = "custom_plugin.plugins:TOOL_TALLY"


def test_the_plugins_load_by_import_path() -> None:
    assert load_guard(GUARD) is FORBID_ARM
    assert load_sink(SINK) is TOOL_TALLY


class TestForbidArm(GuardConformance):
    guard_name = GUARD


class TestToolTally(SinkConformance):
    sink_name = SINK


PICKUP = "custom_skillset.pickup:PICKUP_SKILLS"
PHASE = PhaseSpec.model_validate(
    {
        "name": "tutorial",
        "kind": "learn",
        "visibility": {"actor": ["hold.*", "hub.finish"]},
        "tiers": {"actor": "D"},
    }
)
RATIONALE = {"stage": "lift", "scene": "a red box", "progress": "starting", "plan": "pick it up"}


async def test_a_run_composes_the_plugins_and_they_act(tmp_path: Path) -> None:
    items = [
        PluginItem(group="guards", name=GUARD, overrides={"arms": ["right"]}),
        PluginItem(group="event_sinks", name=SINK),
    ]
    blobs = LocalBlobStore(tmp_path / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs)
    run_dir = tmp_path / "run"
    async with compose_run(items) as composition:
        with open_sinks(composition.sinks, run_dir) as sinks:
            events = EventRouter(run_dir / "events.jsonl", sinks=sinks)
            provider = LocalProvider(Hold(), None, ["mock", PICKUP], load_embodiment(MOCK_BIMANUAL))
            async with Hub(
                [provider],
                blobs=blobs,
                phases=[PHASE],
                events=events,
                pipeline=composition.pipeline,
                env_client=lambda _: env,
            ) as hub:
                opened = await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=0))
                binding = EnvBinding(
                    url="inproc://mock", episode_id=opened.episode_id, token=opened.agent_token
                )
                session = await hub.open_episode(
                    OpenEpisode(
                        stage="tutorial", role="actor", env=binding, embodiment=str(MOCK_BIMANUAL)
                    )
                )
                x, y, z = (await env.gt_snapshot(opened.episode_id)).objects["red"].p
                outcomes = []
                for n, arm in enumerate(["right", "left"]):
                    arguments = {"rationale": RATIONALE, "arm": arm, "tcp": [x, y, z], "lift": 0.1}
                    call = ToolCall(
                        call_id=f"c{n}", name="pick_up", arguments=arguments, origin="model"
                    )
                    outcomes.append(await hub.call(session.episode_id, call))
                judged = await env.judge(opened.episode_id)
                await env.close(opened.episode_id)
            events.close()
    assert [o.error_code for o in outcomes] == ["arm_off_limits", None]
    assert judged.success  # the left arm lifted the box
    assert json.loads((run_dir / "tool_tally.json").read_text(encoding="utf-8")) == {"pick_up": 2}


async def test_a_bad_config_is_refused_before_anything_runs() -> None:
    item = PluginItem(group="guards", name=GUARD, overrides={"arm": ["right"]})
    with pytest.raises(ValueError, match="invalid config for guard 'forbid_arm'"):
        async with compose_run([item]):
            pass
