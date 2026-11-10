"""The Agent host: P3 episodes that a benchmark's own client drives, here `MockClient` (a client
loop over `MockEnv`). Brains are scripted."""

import contextlib
import functools
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import httpx
import pytest
from pydantic import SecretStr

from roboscript import rpc
from roboscript.agent import ScriptedBrain, call, read_lines
from roboscript.agent.host import (
    AgentHostService,
    DemoView,
    FinishCheck,
    Guards,
    HostConfig,
    OpeningHook,
    _voidable,
)
from roboscript.agent.scripted import Script
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import InfraError, ProtocolError, ToolError
from roboscript.contracts.host import INFRA_ENDS, AgentHost
from roboscript.contracts.hub import EpisodeEnd, EpisodeSeed
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.contracts.versions import HEADER
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs import BridgeConfig, Bridges
from roboscript.envs.mock import MOCK, MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.learning import DemoPackage
from roboscript.learning.agent_host import (
    actor_prompt,
    frozen_memory,
    opening_hook,
)
from roboscript.learning.runtime import MethodRuntime
from roboscript.memory import (
    FileMemoryStore,
    MemoryDoc,
    VersionSource,
    WorkflowStage,
    build_manifest,
)
from roboscript.memory.artifact import MemoryArtifacts
from roboscript.records import EventWriter, read_events
from roboscript.settings import source_sha
from roboscript.testing import mock as mock_toolset
from roboscript.testing.hold import Hold
from roboscript.testing.host import MockClient
from roboscript.testing.memory import sample_memory
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES, TestMethod
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import LocalProvider, demo_alias
from roboscript.tools.skillsets import guides
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.sensing import Sensing
from roboscript.toolsets.worldstate import WorldState

REPO = Path(__file__).resolve().parents[3]
TASK = TaskName("lift_red")
PROFILE = load_embodiment(MOCK_BIMANUAL)


class PlainBriefs:
    """A method's demonstration views, plainly (`MethodRuntime.demo`): above `none` the package's
    instruction, at `full` also its recording's frames as `demo:<n>`."""

    levels = ("goal", "ops", "full")

    def brief(self, package: DemoPackage, level: str) -> list[ContentPart]:
        if level == "none":
            return []
        return [TextPart(text=f"# The demonstration\nIts instruction: {package.instruction}")]

    def seed(self, package: DemoPackage, level: str) -> EpisodeSeed | None:
        if level != "full":
            return None
        return EpisodeSeed(
            frames=[e.model_copy(update={"alias": demo_alias(e.alias)}) for e in package.frames]
        )


RUNTIME = replace(TestMethod.runtime(REPO), demo=PlainBriefs())  # plain prompts and views
KEY = "lift_red.test.seed0"
RATIONALE = {"stage": "wait", "scene": "three boxes", "progress": "at rest", "plan": "stay"}
TOKEN = SecretStr("a-host-token-for-the-tests")


@dataclass
class Rig:
    service: AgentHostService
    bench: MockClient
    records: Path
    blobs: LocalBlobStore


def start_rig(
    tmp_path: Path,
    events: EventWriter,
    runtime: MethodRuntime = RUNTIME,
    extra: Sequence[LocalProvider] = (),
) -> Callable[..., Any]:
    """`async with start_rig(...)(script, finish_check=...) as r:` a started host over sensing,
    hold, the Mock toolset and the world state toolset (and `extra`), running `runtime`, with
    lift_red's frozen Memory (and `demos`' views of the demonstration), and a bench driving it."""

    @contextlib.asynccontextmanager
    async def start(
        script: Script,
        finish_check: FinishCheck | None = None,
        guards: Guards | None = None,
        opening: OpeningHook | None = None,
        demos: dict[TaskName, DemoView] | None = None,
        config: HostConfig | None = None,
        memory: MemoryDoc | None = None,
    ) -> AsyncIterator[Rig]:
        blobs = LocalBlobStore(tmp_path / "blobs")
        store = FileMemoryStore(tmp_path / "memory")
        doc = memory or sample_memory(TASK)
        await store.commit(doc, status="s1_draft", source=VersionSource(kind="s1"))
        await store.freeze(TASK, 0, reason="for the test")
        manifest = await build_manifest(store, [TASK])
        bridges = Bridges()
        providers = [
            LocalProvider(Sensing()),
            LocalProvider(Hold()),
            LocalProvider(Mock(), {"act_seconds": 0.0}),
            LocalProvider(WorldState(), SceneConfig(pipeline=PERCEPTION)),
            *extra,
        ]
        hub = Hub(
            providers, blobs=blobs, phases=TEST_PHASES, events=events, env_client=bridges.client
        )
        config = config or HostConfig(
            records=tmp_path / "host",
            embodiment=str(MOCK_BIMANUAL),
            guards=guards or Guards(),
            bridge=BridgeConfig(observe_timeout_s=5, report_timeout_s=5),
            latch_finish=False,  # the scripts finish once; `test_a_finish_before_the_latch...`
        )
        phase = runtime.host_phase
        assert phase is not None
        guide = guides(MOCK.skillsets)  # the Mock's guide: these toolsets mount no skillset
        prompt, digest = actor_prompt(
            runtime.prompts({}, guide), load_embodiment(MOCK_BIMANUAL), phase
        )
        service = AgentHostService(
            config,
            hub=hub,
            bridges=bridges,
            blobs=blobs,
            events=events,
            load=frozen_memory(MemoryArtifacts(memory=store), runtime.frozen),
            manifest=manifest,
            brain=lambda: ScriptedBrain(script),
            prompt=prompt,
            phase=phase,
            prompt_digest=digest,
            finish_check=finish_check,
            opening=opening if opening is not None else opening_hook(hub, runtime.opening),
            finish=runtime.finish.host,  # the method's finish gate, as `serve_host` gives it
            demos=demos,
        )
        env = MockEnv(MockConfig(task=TASK, blob_dir=tmp_path / "blobs"), blobs)
        async with hub, service:
            with pytest.raises(InfraError, match="still loading its Memory"):
                await service.reset(KEY, TASK, manifest.id)
            assert not service.ready
            await service.start()
            assert service.ready
            bench = MockClient(env, service, KEY, TASK, manifest.id)
            yield Rig(service, bench, tmp_path / "host", blobs)

    return start


@pytest.fixture
def rig(tmp_path: Path, events: EventWriter) -> Callable[..., Any]:
    return start_rig(tmp_path, events)


def summary(rig: Rig, index: int = 0) -> dict[str, Any]:
    return json.loads(
        (rig.records / KEY / f"ep{index:03d}" / "summary.json").read_text(encoding="utf-8")
    )


async def test_an_episode_that_holds_then_finishes(rig: Callable[..., Any]) -> None:
    script = [[call("hold", rationale=RATIONALE, steps=3)], [call("finish", note="done")]]
    async with rig(script) as started:
        with anyio.fail_after(30):
            judge = await started.bench.episode()
    assert (judge.success, judge.steps_used) == (False, 40)  # held still to the limit
    done = summary(started)
    assert done["end"]["reason"] == "finish" and done["official"] == {
        "success": False,
        "steps": 40,
        "step_limit": 40,
        "failed_calls": 0,
    }
    assert (done["infra"], done["guard_tripped"]) == (False, False)
    assert done["provenance"]["source"] == source_sha()  # no launcher named a config
    assert (done["provenance"]["config"], done["provenance"]["config_hash"]) == (None, None)
    assert (done["turns"], done["model_calls"], done["memory"]["version"]) == (2, 2, 0)
    assert done["instruction"] == "raise the red block"  # the client's paraphrased instruction
    episode = started.records / KEY / "ep000"
    turns = read_lines(episode / "turns.jsonl")
    [held] = turns[0]["exchanges"][0]["calls"]
    assert held["name"] == "hold" and "steps_used=3" in held["content"][0]["text"]
    assert turns[1]["observation"][0]["text"].startswith("Observation frame_2 at step 3 of 40")
    config = json.loads((episode / "config.json").read_text(encoding="utf-8"))
    assert config["manifest"] == started.bench.manifest and {"hold", "observe"} <= set(
        config["tools"]
    )
    assert len(config["prompt"]["templates"]) == 64  # the recipe's version, as S1 and S2 record
    calls = read_lines(episode / "llm_calls.jsonl")
    prompt = next(line for line in calls if line["kind"] == "context")["instructions"]
    assert "raise the red block" in prompt and "(p3)" in prompt  # P3's rules
    assert "finish is refused" not in prompt  # no finish guard is installed
    assert (
        (episode / "memory_before.md")
        .read_text(encoding="utf-8")
        .startswith("# TASK MEMORY: lift_red")
    )
    kinds = [event.kind for event in read_events(episode / "events.jsonl")]
    assert {"p3.poll", "p3.report", "env.observe", "episode.close"} <= set(kinds)
    assert (episode / "llm_calls.jsonl").is_file()


@pytest.fixture
def outage(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Mock's `fail` fails as infrastructure while the list's last entry is true (a service
    restarting), else as the tool refusing."""
    down = [True]

    def failing(message: str) -> Exception:
        return InfraError("not_ready", message) if down[-1] else ToolError(message)

    monkeypatch.setattr(mock_toolset, "ToolError", failing)
    return down


async def test_an_episode_ending_after_an_unanswered_infrastructure_error_is_lost(
    rig: Callable[..., Any], outage: list[bool]
) -> None:
    """A call fails as infrastructure and the agent finishes with no call answered after it: the
    episode ends as `infra_error`, which the client's poll and the summary report as lost, so the
    launcher runs it again (only that episode); its run again, the infrastructure back, counts."""
    script = [[call("fail")], [call("finish", note="done")]]
    async with rig(script) as started:
        with anyio.fail_after(30):
            await started.bench.episode()
            lost = summary(started, 0)
            outage.append(False)
            await started.bench.episode()  # the launcher's run again of the lost episode
    assert lost["end"]["reason"] == "infra_error" and lost["infra"] is True
    assert lost["end"]["note"].startswith("it ended (finish) after a call failed")
    assert lost["end"]["note"].endswith("; its note: done")
    assert (lost["voided_end"], lost["guard_tripped"]) == ("finish", False)
    assert lost["voided_by"] == "fail: The tool is unavailable right now (not_ready); try again."
    assert lost["end"]["reason"] in INFRA_ENDS  # the relay's test: lost, not finished
    again = summary(started, 1)
    assert (again["end"]["reason"], again["infra"]) == ("finish", False)
    assert "voided_end" not in again and "voided_by" not in again


async def test_a_success_that_latches_after_the_finish_is_not_lost(
    rig: Callable[..., Any], outage: list[bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call fails as infrastructure, the agent finishes, and the task's success latches while
    the client holds still to the limit: the host has answered the client's poll as lost, but the
    official outcome restores the end it had, so the episode counts as the success it is."""
    script = [[call("fail")], [call("finish", note="done")]]
    async with rig(script) as started:
        env = started.bench.env
        judge = env.judge

        async def latching(episode: Any) -> Any:  # the latch fires during the hold
            judged = await judge(episode)
            if judged.step_limit is not None and judged.steps_used >= judged.step_limit:
                return judged.model_copy(update={"success": True})
            return judged

        monkeypatch.setattr(env, "judge", latching)
        with anyio.fail_after(30):
            assert (await started.bench.episode()).success
    done = summary(started)
    assert (done["end"]["reason"], done["infra"], done["official"]["success"]) == (
        "finish",
        False,
        True,
    )
    assert "voided_end" not in done


def test_a_latched_success_is_never_lost_for_an_infrastructure_error() -> None:
    """Whatever ended it after an unanswered infrastructure error, an episode whose success
    latched (the bridge saw the batch stop there, or the client reported it) is not lost."""

    def episode(ended: str | None, success: bool | None, end: str) -> Any:
        outcome = None if success is None else SimpleNamespace(success=success)
        at = EpisodeEnd(reason=end, at=0.0)
        return SimpleNamespace(bridge=SimpleNamespace(ended=ended), outcome=outcome, end=at)

    assert not _voidable(episode("success", None, "env_ended"))
    assert not _voidable(episode(None, True, "finish"))
    assert _voidable(episode("step_limit", False, "env_ended"))
    assert _voidable(episode(None, None, "finish"))
    assert not _voidable(episode(None, None, "infra_error"))  # lost already
    assert not _voidable(episode(None, None, "cancelled"))
    assert not _voidable(episode(None, None, "config_error:identify"))


async def test_a_hard_failure_does_not_end_a_p3_episode(rig: Callable[..., Any]) -> None:
    """Error recovery: after each hard failure the agent is told to move to a safe pose and try
    again, as often as it happens; the episode ends when the agent finishes or at the limit."""
    lost = call("act", rationale={**RATIONALE, "stage": "grasp"}, outcome="lost_hold")
    empty = call("act", rationale={**RATIONALE, "stage": "grasp"}, outcome="empty_grasp")
    look = call("look")  # a look between identical retries: the retry guard lets them through
    script = [[lost], [empty], [look, lost], [look, lost], [call("finish", note="recovered")]]
    async with rig(script) as started:
        with anyio.fail_after(30):
            await started.bench.episode()
    done = summary(started)
    assert (done["end"]["reason"], done["turns"], done["infra"]) == ("finish", 5, False)
    turns = read_lines(started.records / KEY / "ep000" / "turns.jsonl")
    texts = [turn["exchanges"][0]["calls"][-1]["content"][-1]["text"] for turn in turns[:4]]
    assert all("the attempt goes on. Move to a safe pose" in text for text in texts), texts
    config = json.loads(
        (started.records / KEY / "ep000" / "config.json").read_text(encoding="utf-8")
    )
    assert config["limits"] == {
        "max_model_calls": 160,
        "wall_clock_s": 400.0,
        "max_format_errors": 3,
        "credit_model_time": True,
        "wall_clock_cap_s": 2400.0,
        "max_stalled_calls": 30,
    }


@dataclass(frozen=True)
class Hinted:
    """A method's demonstration end that the host reads only the latch hint of (`OpaqueGoal`)."""

    hint: str


async def test_a_finish_before_the_latch_is_refused_with_the_hint(
    rig: Callable[..., Any], tmp_path: Path
) -> None:
    """P3 (the default): a finish while the benchmark has not ended the episode (its success
    latch has not fired) is refused up to `finish_refusals` times with the demonstration's latch
    hint, then the finish check, then the finish goes through."""
    stages: list[str] = []

    async def check(episode: EpisodeId, stage: WorkflowStage) -> str | None:
        stages.append(stage.name)
        return "near(A, base) fails"

    view = DemoView(brief=[], seed=None, label={}, goal=Hinted(". Keep moving the left arm up"))
    config = HostConfig(
        records=tmp_path / "host",
        embodiment=str(MOCK_BIMANUAL),
        bridge=BridgeConfig(observe_timeout_s=5, report_timeout_s=5),
        finish_refusals=2,
    )
    script = [[call("finish", note=f"done {n}")] for n in range(4)]
    async with rig(script, finish_check=check, demos={TASK: view}, config=config) as started:
        with anyio.fail_after(30):
            await started.bench.episode()
    assert summary(started)["end"]["note"] == "done 3"
    turns = read_lines(started.records / KEY / "ep000" / "turns.jsonl")
    said = [t["exchanges"][0]["calls"][0]["content"][0]["text"] for t in turns[:3]]
    assert all(t["exchanges"][0]["calls"][0]["error"] == "finish_refused" for t in turns[:3])
    assert all("has not latched" in text and "left arm up" in text for text in said[:2])
    assert "near(A, base) fails" in said[2] and stages == ["set_down"]


async def test_the_plane_over_rpc(rig: Callable[..., Any]) -> None:
    script = [[call("hold", rationale=RATIONALE, steps=2)], [call("finish", note="done")]]
    async with rig(script) as started, anyio.create_task_group() as tg:
        app = rpc.make_app(rpc.routes(AgentHost, started.service, token=TOKEN))
        url = await tg.start(functools.partial(rpc.serve, app))
        host = rpc.client(AgentHost, url, token=TOKEN, timeout=10)  # type: ignore[type-abstract]
        stranger = rpc.client(AgentHost, url, token=SecretStr("x" * 20), timeout=10)  # type: ignore[type-abstract]
        with pytest.raises(ProtocolError, match="unauthorized"):
            await stranger.poll(KEY)
        with pytest.raises(ProtocolError, match="not_found"):
            await host.poll("no.such.slot")
        headers = {"Authorization": f"Bearer {TOKEN.get_secret_value()}", HEADER: "agent.host/1.0"}
        async with httpx.AsyncClient() as http:  # a key is a plain file name
            escape = await http.post(
                f"{url}/v1/agent.host/poll", json={"key": "../x"}, headers=headers
            )
        assert (escape.status_code, escape.json()["code"]) == (400, "invalid_request")
        bench = MockClient(started.bench.env, host, KEY, TASK, started.bench.manifest)
        with anyio.fail_after(30):
            judge = await bench.episode()
        await rpc.aclose(host)
        await rpc.aclose(stranger)
        tg.cancel_scope.cancel()
    assert judge.steps_used == 40 and summary(started)["end"]["reason"] == "finish"
