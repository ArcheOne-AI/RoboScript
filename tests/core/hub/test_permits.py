"""The Hub's permit guard (docs/concepts/agent-runtime.md): a model call runs only by consuming a
live permit of its episode; `strict` matches the arguments' canonical hash, `tool_only` the tool
alone; arguments that are not plain JSON never match and never reach a tool."""

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.gateway import Permit, args_hash
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import EpisodeId, ToolName
from roboscript.contracts.tools import ToolCall
from roboscript.hub import DEFAULT_PIPELINE, Hub, Pipeline
from roboscript.hub.permits import NO_PERMIT, PermitLedger, with_permits
from roboscript.hub.steps import NOT_PLAIN, episode_open, resolve_aliases, validate, visibility
from roboscript.records import EventWriter, read_events
from roboscript.tools import LocalProvider, ToolContext, Toolset, tool

RATIONALE = {"stage": "grasp", "scene": "a box", "progress": "start", "plan": "act"}
Model = Callable[..., ToolCall]


class Scaled(BaseModel):
    factor: float


class Scale(Toolset):
    """A tool with a number, which clients may send as 2 or 2.0 (not catalogued: stage dev)."""

    name = "scale"

    def __init__(self) -> None:
        self.calls = 0

    @tool(kind="pure", tier="D")
    async def scale(self, ctx: ToolContext, factor: Annotated[float, Field(gt=0)]) -> Scaled:
        """Scale something by a factor."""
        self.calls += 1
        return Scaled(factor=factor)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def permit(episode: EpisodeId, name: str, arguments: dict[str, Any]) -> Permit:
    return Permit(
        episode_id=episode, tool=ToolName(name), args_hash=args_hash(arguments), call_id="call_1"
    )


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ledger(events: EventWriter, clock: Clock) -> PermitLedger:
    return PermitLedger(events, ttl_s=60, clock=clock)


@pytest.fixture
def scale() -> Scale:
    return Scale()


@pytest.fixture
async def guarded(
    make_hub: Callable[..., Hub], ledger: PermitLedger, scale: Scale
) -> AsyncIterator[Hub]:
    pipeline = with_permits(DEFAULT_PIPELINE, ledger)
    async with make_hub(LocalProvider(scale), pipeline=pipeline) as hub:
        yield hub


async def open_ep(
    hub: Hub, ledger: PermitLedger, stage: str = "s2", mode: str = "strict"
) -> EpisodeId:
    episode = (await hub.open_episode(OpenEpisode(stage=stage))).episode_id  # type: ignore[arg-type]
    ledger.open(episode, mode)  # type: ignore[arg-type]
    return episode


def code(outcome: CallOutcome) -> str | None:
    return outcome.error_code if outcome.is_error else None


def test_the_guard_runs_right_after_visibility(ledger: PermitLedger) -> None:
    pipeline = with_permits(DEFAULT_PIPELINE, ledger)
    assert pipeline.pre == (episode_open, visibility, ledger.guard, validate, resolve_aliases)
    assert pipeline.guards == DEFAULT_PIPELINE.guards
    with pytest.raises(ValueError, match="visibility"):
        with_permits(Pipeline(pre=(), guards=(), around=(), post=()), ledger)


async def test_a_model_call_needs_a_permit_and_uses_it_once(
    guarded: Hub, ledger: PermitLedger, model: Model
) -> None:
    ep = await open_ep(guarded, ledger)
    refused = await guarded.call(ep, model("look"))
    assert (code(refused), refused.content[0].text) == ("no_permit", NO_PERMIT)  # type: ignore[union-attr]
    ledger.add([permit(ep, "look", {})])
    assert code(await guarded.call(ep, model("look"))) is None
    assert code(await guarded.call(ep, model("look"))) == "no_permit"  # a replay
    counters = (await guarded.state(ep)).counters
    assert (counters.info_calls, counters.rejected) == (1, 2)


@pytest.mark.parametrize(
    "bad",
    [float("inf"), "\ud800", [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]],
    ids=["inf", "surrogate", "deep"],
)
async def test_arguments_that_are_not_plain_json_are_refused_and_logged_without_them(
    guarded: Hub,
    ledger: PermitLedger,
    model: Model,
    scale: Scale,
    artifacts_dir: Path,
    bad: Any,
) -> None:
    for mode in ("strict", "tool_only"):
        ep = await open_ep(guarded, ledger, stage="dev", mode=mode)
        ledger.add([permit(ep, "scale", {"factor": 1})])
        outcome = await guarded.call(ep, model("scale", factor=bad))
        expected = (
            ("no_permit", NO_PERMIT) if mode == "strict" else ("invalid_arguments", NOT_PLAIN)
        )
        assert (code(outcome), outcome.content[0].text) == expected  # type: ignore[union-attr]
    assert scale.calls == 0
    [*_, logged] = [
        e.data for e in read_events(artifacts_dir / "events.jsonl") if e.kind == "tool.call"
    ]
    assert logged["name"] == "scale" and logged["arguments"] is None


async def test_tool_only_matches_the_tool_and_records_the_argument_mismatch(
    guarded: Hub, ledger: PermitLedger, model: Model, artifacts_dir: Path
) -> None:
    ep = await open_ep(guarded, ledger, mode="tool_only")
    ledger.add([permit(ep, "act", {"rationale": RATIONALE, "outcome": "ok"})])
    rewritten = model("act", rationale=RATIONALE, outcome="stopped_short")
    assert code(await guarded.call(ep, rewritten)) is None
    assert code(await guarded.call(ep, model("act", rationale=RATIONALE))) == "no_permit"
    events = list(read_events(artifacts_dir / "events.jsonl"))
    [issued] = [e.data for e in events if e.kind == "permit.issue"]
    [consumed] = [e.data for e in events if e.kind == "permit.consume"]
    assert issued == {
        "episode_id": ep,
        "tool": "act",
        "args_hash": args_hash({"rationale": RATIONALE, "outcome": "ok"}),
        "call_id": "call_1",
    }
    assert consumed == {
        "call_id": rewritten.call_id,
        "permit_call_id": "call_1",
        "tool": "act",
        "mode": "tool_only",
        "args_match": False,
    }


async def test_permits_belong_to_their_episode(
    guarded: Hub, ledger: PermitLedger, model: Model
) -> None:
    a, b = await open_ep(guarded, ledger), await open_ep(guarded, ledger)
    ledger.add([permit(a, "look", {})])
    assert code(await guarded.call(b, model("look"))) == "no_permit"
    assert code(await guarded.call(a, model("look"))) is None
    ledger.close(a)
    ledger.add([permit(b, "look", {})])
    with pytest.raises(ProtocolError, match="takes no permits"):
        ledger.add([permit(a, "look", {})])
    with pytest.raises(ProtocolError, match="one episode"):
        ledger.add([permit(b, "look", {}), permit(a, "look", {})])
    unregistered = (await guarded.open_episode(OpenEpisode(stage="s2"))).episode_id
    assert code(await guarded.call(unregistered, model("look"))) == "no_permit"


async def test_system_and_service_calls_need_no_permit(guarded: Hub, ledger: PermitLedger) -> None:
    ep = await open_ep(guarded, ledger)
    assert code(await guarded.system_call(ep, ToolName("look"), {})) is None
    observed = await guarded.service_call(ep, "look", {}, parent="call-0")
    assert observed.frame  # type: ignore[attr-defined]
