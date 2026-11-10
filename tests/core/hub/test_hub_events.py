"""The Hub's event log: every call and its outcome, episode life, one trace per episode, and
nothing secret."""

from collections import Counter
from collections.abc import Callable

from pydantic import BaseModel

from roboscript.contracts.hub import Budgets, OpenEpisode
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import EnvBinding, ToolCall
from roboscript.hub import Hub
from roboscript.records import EventWriter, read_events
from roboscript.tools import ObjectRef

RATIONALE = {"stage": "grasp", "scene": "a box", "progress": "start", "plan": "act"}
SECRET = "agent-plane-secret-token"


async def test_every_call_and_outcome_is_recorded_without_secrets(
    make_hub: Callable[..., Hub], events: EventWriter, model: Callable[..., ToolCall]
) -> None:
    binding = EnvBinding(url="http://127.0.0.1:1", episode_id=EpisodeId("env-1"), token=SECRET)
    async with make_hub(env_client=lambda binding: None) as hub:
        session = await hub.open_episode(
            OpenEpisode(stage="s2", env=binding, budgets=Budgets(max_info_calls=2))
        )
        ep = session.episode_id
        outcomes = [
            await hub.call(ep, model("look")),
            await hub.system_call(ep, "look", {}),
            await hub.call(ep, model("fail")),
            await hub.call(ep, model("hidden")),
            await hub.call(ep, model("look")),  # over the information budget
            await hub.call(ep, model("act", rationale=RATIONALE, outcome="lost_hold")),
            await hub.call(ep, model("look")),  # after the fail-fast end
        ]
        await hub.close_episode(ep)
    log = [e for e in read_events(events.path) if e.episode_id == ep]
    kinds = Counter(e.kind for e in log)
    assert kinds == {
        "episode.open": 1,
        "tool.call": 7,
        "tool.result": 4,  # reached their tool: look, look, fail, act
        "tool.reject": 3,  # hidden, over budget, after the end
        "hub.alias": 2,  # the two frames the looks returned
        "episode.end": 1,
        "episode.close": 1,
    }
    calls = {e.data["call_id"]: e for e in log if e.kind == "tool.call"}
    ends = [e for e in log if e.kind in ("tool.result", "tool.reject")]
    assert sorted(e.data["call_id"] for e in ends) == sorted(calls)  # one outcome per call
    for outcome, end in zip(outcomes, ends, strict=True):
        assert end.data["call_id"] == outcome.call_id
        assert end.data["code"] == outcome.error_code
        assert end.data["content"] == [part.model_dump(mode="json") for part in outcome.content]
    assert calls[outcomes[1].call_id].data["origin"] == "system"
    assert [e.data["step"] for e in ends if e.kind == "tool.reject"] == [
        "visibility",
        "budget",
        "episode_open",
    ]
    act = ends[5]
    assert act.data["result"]["reason"] == "lost_hold" and act.data["result"]["ok"] is False
    end = next(e for e in log if e.kind == "episode.end")
    assert end.data["reason"] == "fail_fast:lost_hold@grasp"
    assert len({e.trace.trace_id for e in log if e.trace}) == 1  # the episode is one trace
    assert all(e.trace for e in log)
    opened = next(e for e in log if e.kind == "episode.open")
    assert opened.data["tools"] == {tool.name: tool.hash for tool in session.tools}
    assert list(opened.data["tools"]) == ["look", "act", "fail", "finish", "scratchpad"]
    assert (opened.data["role"], opened.data["env"]["token"]) == ("actor", "**********")
    assert SECRET not in events.path.read_text(encoding="utf-8")


class Found(BaseModel):
    obj: ObjectRef
