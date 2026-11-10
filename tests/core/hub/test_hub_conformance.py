"""Conformance of the in-process Hub on the Mock benchmark (docs/concepts/agent-runtime.md): the
tool list, the invariants and isolation between episodes (by episode id)."""

import importlib.metadata
import inspect
from collections.abc import Callable
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import ENDED_MESSAGE, Budgets, CallOutcome, OpenEpisode, ToolHub
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ToolCall
from roboscript.hub import Hub
from roboscript.kernel import entrypoints
from roboscript.testing.mock import Mock
from roboscript.testing.phases import dev_phases
from roboscript.tools import FrameRef, LocalProvider, ToolContext, Toolset, tool

RATIONALE = {"stage": "grasp", "scene": "a box", "progress": "start", "plan": "act"}
STAGES = ("s1", "s2", "p3", "dev")
Model = Callable[..., ToolCall]


class Seen(BaseModel):
    frame: FrameRef


class Looker(Toolset):
    """Takes a frame: aliases must resolve per episode."""

    name = "looker"

    def __init__(self) -> None:
        self.frames: list[str] = []

    @tool(kind="pure", tier="D")
    async def peek(self, ctx: ToolContext, frame: FrameRef) -> Seen:
        """Look at a frame."""
        self.frames.append(frame)
        return Seen(frame=frame)


async def open_ep(hub: Hub, stage: str = "s2", **options: Any) -> EpisodeId:
    return (await hub.open_episode(OpenEpisode(stage=stage, **options))).episode_id  # type: ignore[arg-type]


def error(outcome: CallOutcome) -> tuple[str | None, str]:
    assert outcome.is_error, outcome
    return outcome.error_code, outcome.content[0].text  # type: ignore[union-attr]


def test_the_hub_implements_the_tool_hub_contract() -> None:
    for name in (name for name in vars(ToolHub) if not name.startswith("_")):
        method = getattr(Hub, name)
        assert inspect.iscoroutinefunction(method), name
        assert inspect.signature(method) == inspect.signature(getattr(ToolHub, name)), name


async def test_c1_each_stage_sees_exactly_its_tools_in_a_fixed_order(hub: Hub) -> None:
    for stage in STAGES:
        session = await hub.open_episode(OpenEpisode(stage=stage))  # type: ignore[arg-type]
        names = [tool.name for tool in session.tools]
        assert names == ["look", "act", "fail", "finish", "scratchpad"], stage
        assert await hub.tools(session.episode_id) == session.tools  # constant for the episode
        assert (await hub.state(session.episode_id)).tools == names
        assert all(tool.output_schema is None for tool in session.tools)  # no structured content
        assert {tool.meta.tier for tool in session.tools} == {"D"}


async def test_c5_a_second_action_or_finish_in_flight_is_refused_not_queued(
    make_hub: Callable[..., Hub], mock: Mock, model: Model
) -> None:
    async with make_hub(act_seconds=0.3) as hub:
        ep = await open_ep(hub)
        first: list[CallOutcome] = []

        async def act() -> None:
            first.append(await hub.call(ep, model("act", rationale=RATIONALE)))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(act)
            with anyio.fail_after(2):
                while not (await hub.state(ep)).acting:  # noqa: ASYNC110 - another task acts
                    await anyio.sleep(0.01)
            second = await hub.call(ep, model("act", rationale=RATIONALE))
            early = await hub.call(ep, model("finish", note="done"))  # would end mid-action
            info = await hub.call(ep, model("look"))  # information calls go on meanwhile
        assert error(second)[0] == error(early)[0] == "action_in_progress"
        assert not info.is_error and not first[0].is_error
        assert mock.acts == 1  # the refused action never ran
        assert (await hub.state(ep)).ended is None  # nor did the refused finish
        assert not (await hub.call(ep, model("act", rationale=RATIONALE))).is_error
        assert mock.acts == 2
        assert (await hub.call(ep, model("finish", note="done"))).episode_ended


async def test_c5_every_call_after_finish_gets_one_fixed_error(hub: Hub, model: Model) -> None:
    ep = await open_ep(hub)
    done = await hub.call(ep, model("finish", note="the box is placed"))
    assert not done.is_error and done.episode_ended
    assert done.content[0].text == "Finished: the episode is over."  # type: ignore[union-attr]
    calls = [
        model("look"),
        model("act", rationale=RATIONALE),
        model("finish", note="again"),
        model("hidden"),
        model("no_such_tool", x=1),
    ]
    outcomes = [await hub.call(ep, call) for call in calls]
    outcomes.append(await hub.system_call(ep, "look", {}))
    assert {error(o) for o in outcomes} == {("episode_ended", ENDED_MESSAGE)}
    assert all(o.episode_ended for o in outcomes)
    ended = (await hub.state(ep)).ended
    assert ended is not None and (ended.reason, ended.note) == ("finish", "the box is placed")


async def test_c5_over_budget_calls_are_refused(hub: Hub, model: Model) -> None:
    ep = await open_ep(hub, budgets=Budgets(max_info_calls=1, max_turns=2))
    assert not (await hub.call(ep, model("look"))).is_error
    assert error(await hub.call(ep, model("look"))) == (
        "budget_exhausted",
        "All 1 information calls are used up; act or finish.",
    )
    assert not (await hub.system_call(ep, "look", {})).is_error  # the runtime is not budgeted
    assert not (await hub.call(ep, model("act", rationale=RATIONALE))).episode_ended
    last = await hub.call(ep, model("act", rationale=RATIONALE))
    assert not last.is_error and last.episode_ended  # the last turn ends the episode
    status = await hub.state(ep)
    assert status.ended is not None and status.ended.reason == "max_turns"
    assert (status.counters.turns, status.counters.info_calls, status.counters.rejected) == (
        2,
        1,
        1,
    )
    assert status.counters.system_calls == 1


async def test_c5_call_takes_model_calls_whatever_their_origin_says(hub: Hub) -> None:
    ep = await open_ep(hub, budgets=Budgets(max_info_calls=1))
    claimed = ToolCall(call_id="c1", name="look", arguments={}, origin="system")
    assert not (await hub.call(ep, claimed)).is_error
    over = await hub.call(ep, claimed.model_copy(update={"call_id": "c2"}))
    assert error(over)[0] == "budget_exhausted"  # budgeted like any model call
    counters = (await hub.state(ep)).counters
    assert (counters.info_calls, counters.system_calls) == (1, 0)


async def test_c5_hidden_tools_are_refused_like_missing_ones(hub: Hub, model: Model) -> None:
    for stage in STAGES:
        ep = await open_ep(hub, stage)
        hidden = await hub.call(ep, model("hidden"))
        missing = await hub.call(ep, model("hidde"))
        assert error(hidden) == ("unknown_tool", "There is no tool named 'hidden'.")
        assert error(missing) == ("unknown_tool", "There is no tool named 'hidde'.")
        assert error(await hub.system_call(ep, "hidden", {}))[0] == "unknown_tool"


async def test_c6_episodes_do_not_share_locks_ends_or_state(
    make_hub: Callable[..., Hub], model: Model
) -> None:
    async with make_hub(act_seconds=0.3) as hub:
        a, b = await open_ep(hub), await open_ep(hub)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(hub.call, a, model("act", rationale=RATIONALE))
            with anyio.fail_after(2):
                while not (await hub.state(a)).acting:  # noqa: ASYNC110 - another task acts
                    await anyio.sleep(0.01)
            in_b = await hub.call(b, model("act", rationale=RATIONALE))  # a's lock is a's
        assert not in_b.is_error
        assert "action 1 of this episode" in in_b.content[0].text  # type: ignore[union-attr]
        await hub.end_episode(a, "success")
        assert error(await hub.call(a, model("look")))[0] == "episode_ended"
        assert not (await hub.call(b, model("look"))).is_error
        await hub.close_episode(a)
        assert error(await hub.call(a, model("look")))[0] == "episode_ended"
        assert error(await hub.call(EpisodeId("ep-000000000000"), model("look"))) == (
            "unknown_episode",
            "Unknown episode.",
        )
        with pytest.raises(ProtocolError, match="closed"):
            await hub.state(a)


async def test_c6_aliases_belong_to_their_episode(blobs: LocalBlobStore, model: Model) -> None:
    looker = Looker()
    async with Hub(
        [LocalProvider(Mock()), LocalProvider(looker)],
        blobs=blobs,
        phases=dev_phases([looker.name]),
    ) as hub:
        a, b = await open_ep(hub, "dev"), await open_ep(hub, "dev")  # looker is not catalogued
        assert "frame_1" in (await hub.call(a, model("look"))).content[0].text  # type: ignore[union-attr]
        assert not (await hub.call(a, model("peek", frame="frame_1"))).is_error
        code, text = error(await hub.call(b, model("peek", frame="frame_1")))
        assert code == "invalid_arguments" and "unknown alias 'frame_1'" in text
        await hub.call(b, model("look"))
        assert not (await hub.call(b, model("peek", frame="frame_1"))).is_error
        assert len(set(looker.frames)) == 2  # the same alias, each episode's own frame


def _installed() -> bool:
    try:
        importlib.metadata.distribution("roboscript")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _installed(), reason="the simulator env runs roboscript from source")
def test_the_mock_toolset_is_registered() -> None:
    assert entrypoints.load("roboscript.toolsets", "mock") is Mock
