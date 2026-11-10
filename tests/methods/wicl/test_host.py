"""The Agent host with wicl's runtime (`Wicl.runtime`: its P3 prompts, demonstration views
and finish gate, and its `done` toolset mounted): the tests of the core's host rig
(`tests/core/agent/test_host.py`) that depend on what wicl shows and gates."""

import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import anyio
import pytest

from roboscript.agent import call, read_lines
from roboscript.blobs import LocalBlobStore
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.learning.agent_host import AgentHostConfig, demo_views
from roboscript.learning.demo import TRUTH_FILE
from roboscript.memory import DoneSpecParam
from roboscript.perception.criteria import DoneSpec
from roboscript.records import EventWriter
from roboscript.testing.memory import sample_memory
from roboscript.testing.protocols import write_demo_package
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import LocalProvider
from roboscript.toolsets.scene import SceneConfig
from roboscript_methods.wicl.goal import demo_goal
from roboscript_methods.wicl.method import Wicl
from roboscript_methods.wicl.toolsets.done import Done


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_test_host", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load(Path(__file__).resolve().parents[2] / "core" / "agent" / "test_host.py")
KEY, TASK, PROFILE, RATIONALE = _core.KEY, _core.TASK, _core.PROFILE, _core.RATIONALE
REPO = _core.REPO
WICL = Wicl.runtime(REPO / "recipes" / "wicl")
summary = _core.summary


@pytest.fixture
def rig(tmp_path: Path, events: EventWriter) -> Callable[..., Any]:
    """The core's host rig running wicl, with its `done` toolset (`check_done`, for a
    demonstration's end gate) mounted."""
    return _core.start_rig(
        tmp_path, events, WICL, [LocalProvider(Done(), SceneConfig(pipeline=PERCEPTION))]
    )


@pytest.mark.parametrize("level", ["none", "goal", "ops", "full"])
async def test_the_actor_reads_the_demonstration_at_the_hosts_level(
    rig: Callable[..., Any], tmp_path: Path, level: str
) -> None:
    """The host's `demo_context` (loaded by `demo_views` from the package path of its config)
    reaches the Actor's first request, and with full its episode holds the recording's frames;
    never the package's ground-truth layer. The level is recorded in the episode's config."""
    package = tmp_path / "package"
    await write_demo_package(package, LocalBlobStore(tmp_path / "recording"))
    (package / TRUTH_FILE).write_text(
        '{"secret": "the ground truth (0.100, 0.200)"}', encoding="utf-8"
    )
    host = AgentHostConfig.model_validate(
        {
            "manifest": str(tmp_path / "manifest.json"),
            "method": "wicl",
            "recipe": "recipes/wicl",
            "episodes": {"records": str(tmp_path / "host"), "embodiment": str(MOCK_BIMANUAL)},
            "toolsets": [{"name": "sensing"}],
            "demo_context": level,
            **({} if level == "none" else {"demo": str(package)}),
        }
    )
    demos = await demo_views(host, [TASK], LocalBlobStore(tmp_path / "blobs"), PROFILE, WICL)
    # with a package, shown or not, check_done also compares with its final state
    assert level == "none" or demo_goal(demos[TASK].goal).views is not None
    assert level == "none" or demo_goal(demos[TASK].goal).press is False  # a lift pushes nothing
    looked = [[call("get_image", camera="overhead", frame="demo:2")]] if level == "full" else []
    async with rig([*looked, [call("finish", note="done")]], demos=demos) as started:
        with anyio.fail_after(30):
            await started.bench.episode()
    episode = started.records / KEY / "ep000"
    calls = (episode / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "ground truth" not in calls and "(0.100, 0.200" not in calls
    assert ("# The demonstration" in calls) == (level != "none")
    assert ("the goal" in calls) == (level == "goal")
    assert ("left K2 grasp: before" in calls) == (level in ("ops", "full"))
    config = json.loads((episode / "config.json").read_text(encoding="utf-8"))
    assert config["demo_context"]["demo_context"] == level
    if level == "full":
        turns = read_lines(episode / "turns.jsonl")
        [shown] = turns[0]["exchanges"][0]["calls"]
        assert shown["error"] is None and "demo:2" in shown["content"][0]["text"]
    with pytest.raises(ValueError, match="demo names the package demo_context shows"):
        AgentHostConfig.model_validate({**host.model_dump(), "demo": None, "demo_context": "ops"})


async def test_wicl_s_done_gate_holds_a_finish_until_check_done(
    rig: Callable[..., Any],
) -> None:
    """The host runs the method's finish gate (`MethodRuntime.finish.host`): a Memory with done
    criteria opens its episode gated on `check_done`, so a finish with no check after the last
    action is refused (wicl's `DoneGate`)."""
    doc = sample_memory(TASK)
    gap = DoneSpec(metric="gap_z", a="tool", b="table", value=0.0, tol=0.01)
    doc = doc.replace(params={**doc.params, "tool_down": DoneSpecParam(value=gap)})
    script = [[call("finish", note="done")], [call("finish", note="done, surely")]]
    async with rig(script, memory=doc) as started:
        with anyio.fail_after(30):
            await started.bench.episode()
    turns = read_lines(started.records / KEY / "ep000" / "turns.jsonl")
    [refused] = turns[0]["exchanges"][0]["calls"]
    assert refused["error"] == "finish_refused", refused
    assert "check_done" in refused["content"][0]["text"]


_HEX = re.compile(r"(?<!\.)\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,64}\b")  # ids, hashes; not numbers
