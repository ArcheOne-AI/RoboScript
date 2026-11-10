"""Conformance checks for brains (docs/concepts/agent-runtime.md): a brain plays an episode through
the Hub until it ends, runs at most one physical action per turn, stops when the Hub ends the
episode, and leaves a log from which every request it sent can be rebuilt.

A `BrainTarget` gives `session(stage)`, a fresh `AgentSession` whose episode sees the Mock
toolset's `look`, `act` and `fail` and `finish`, and `brain(script)`, the brain under test made
to play `script`: its model's replies in order, each a list of `(tool, arguments)` calls. Use the
checks from pytest by subclassing `BrainConformance` as a `Test*` class with a `brain_target`
fixture, or without pytest via `roboscript.testing.precheck`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from roboscript.agent import AgentSession, Brain, read_lines, rebuild_requests
from roboscript.blobs import BlobStore
from roboscript.contracts.ids import PhaseName

Script = list[list[tuple[str, dict[str, Any]]]]
RATIONALE = {"stage": "grasp", "scene": "a box", "progress": "at it", "plan": "act"}


@dataclass(frozen=True)
class BrainTarget:
    session: Callable[[PhaseName], Awaitable[AgentSession]]
    brain: Callable[[Script], Brain]
    blobs: BlobStore


class BrainConformance:
    """The checks, as `test_*` methods taking a `BrainTarget`."""

    async def test_plays_until_finish(self, brain_target: BrainTarget) -> None:
        session = await brain_target.session("s1")
        script: Script = [
            [("look", {})],
            [("act", {"rationale": RATIONALE})],
            [("finish", {"note": "ok"})],
        ]
        result = await brain_target.brain(script).run(session)
        assert result.end.reason == "finish", result.end
        counters = (await session.hub.state(session.episode_id)).counters
        assert (counters.turns, counters.info_calls) == (2, 1)
        requests = await rebuild_requests(session.recorder.calls, brain_target.blobs)
        assert len(requests) == result.model_calls == 3
        assert all(request["tools"] and request["input"] for request in requests)
        assert len(read_lines(session.recorder.turns)) == result.turns == 2

    async def test_one_physical_action_per_turn(self, brain_target: BrainTarget) -> None:
        session = await brain_target.session("s1")
        act = ("act", {"rationale": RATIONALE})
        result = await brain_target.brain([[act, act], [("finish", {"note": "ok"})]]).run(session)
        assert result.end.reason == "finish", result.end
        assert (await session.hub.state(session.episode_id)).counters.turns == 2
        [first, _] = read_lines(session.recorder.turns)
        statuses = [call["status"] for call in first["exchanges"][0]["calls"]]
        assert statuses == ["executed", "refused"]

    async def test_stops_when_the_hub_ends_the_episode(self, brain_target: BrainTarget) -> None:
        session = await brain_target.session("s2")  # fail-fast
        lost = ("act", {"rationale": RATIONALE, "outcome": "lost_hold"})
        result = await brain_target.brain([[lost], [("look", {})]]).run(session)
        assert result.end.reason == "fail_fast:lost_hold@grasp", result.end
        assert result.model_calls == 1
