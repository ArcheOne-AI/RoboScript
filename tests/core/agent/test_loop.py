"""The own loop with scripted replies over the Hub, MockEnv, sensing and the Mock toolset: turn
structure, one action per turn, nudges and format errors, limits, stage rules, infrastructure
failures, and the record: every request is rebuilt from the log as it was sent."""

from typing import Any

import anyio
import numpy as np

from roboscript.agent import (
    Limits,
    LoopConfig,
    ScriptedBrain,
    call,
    read_lines,
    rebuild_requests,
    think,
)
from roboscript.agent.recording import digest
from roboscript.contracts.hub import Budgets
from roboscript.contracts.tools import ImagePart, TextPart
from roboscript.records import read_events

RATIONALE = {"stage": "grasp", "scene": "a red box", "progress": "starting", "plan": "act"}
CAMERAS = ["overhead", "wrist_left", "wrist_right"]
TOOLS = [
    *("observe", "get_image", "show_frames", "get_robot_state", "look", "act", "fail"),
    "finish",
    "scratchpad",
]
ACT = {"rationale": RATIONALE}


def recorder(requests: list[dict[str, Any]], replies: list[Any]) -> Any:
    """A script that keeps every request it is sent and answers with `replies` in order."""

    async def reply(request: dict[str, Any]) -> Any:
        requests.append(request)
        return replies[len(requests) - 1]

    return reply


def kinds(items: list[dict[str, Any]]) -> list[str]:
    return [item.get("type", item.get("role", "")) for item in items]


def logged(session: Any) -> list[dict[str, Any]]:
    """The requests as `llm_calls.jsonl` logs them (images as blob URLs), with their context."""
    lines = read_lines(session.recorder.calls)
    contexts = {line["sha256"]: line for line in lines if line["kind"] == "context"}
    return [
        {
            **line["request"],
            "instructions": contexts[line["context"]]["instructions"],
            "tools": contexts[line["context"]]["tools"],
        }
        for line in lines
        if line["kind"] == "request"
    ]


def calls_of(session: Any) -> list[dict[str, Any]]:
    """Every call the model made, in order, as `turns.jsonl` records it."""
    turns = read_lines(session.recorder.turns)
    return [c for turn in turns for exchange in turn["exchanges"] for c in exchange["calls"]]


def events(rig: Any, session: Any, kind: str) -> list[Any]:
    return [
        event
        for event in read_events(rig.events.path)
        if event.episode_id == session.episode_id and event.kind == kind
    ]


def system_observes(rig: Any, session: Any) -> int:
    return sum(e.data["origin"] == "system" for e in events(rig, session, "tool.call"))


async def test_turns_open_with_the_observation_and_close_with_an_action(rig: Any) -> None:
    replies = [[call("get_robot_state")], [call("act", **ACT)], [call("finish", note="done")]]
    session = await rig.session()
    result = await ScriptedBrain(replies).run(session)
    assert (result.end.reason, result.end.note, result.turns, result.model_calls) == (
        "finish",
        "done",
        2,
        3,
    )
    turns = read_lines(session.recorder.turns)
    assert [turn["action"] for turn in turns] == ["act", "finish"]
    assert [len(turn["exchanges"]) for turn in turns] == [2, 1]
    assert turns[0]["rationale"] == RATIONALE
    assert (turns[0]["end"], turns[1]["end"]["reason"]) == (None, "finish")
    labels = [part["label"] for part in turns[0]["observation"] if part["type"] == "image"]
    assert labels == CAMERAS
    act = turns[0]["exchanges"][1]["calls"][0]
    assert (act["name"], act["status"], act["error"]) == ("act", "executed", None)
    lines = [line["kind"] for line in read_lines(session.recorder.calls)]
    assert lines == ["context", "request", "response", "request", "response", "request", "response"]

    requests = logged(session)
    first = requests[0]
    assert [tool["name"] for tool in first["tools"]] == TOOLS
    assert all(tool["strict"] and tool["type"] == "function" for tool in first["tools"])
    assert (first["store"], first["parallel_tool_calls"]) == (False, False)
    [opening] = first["input"]
    images = [part for part in opening["content"] if part["type"] == "input_image"]
    assert opening["role"] == "user" and len(images) == 3
    assert all(part["image_url"].startswith("rs-blob:") for part in images)
    assert kinds(requests[1]["input"]) == ["user", "function_call", "function_call_output"]
    assert kinds(requests[2]["input"]) == [
        "user",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
        "user",
    ]
    assert system_observes(rig, session) == 2
    assert len(events(rig, session, "model.request")) == len(events(rig, session, "model.response"))
    assert len(events(rig, session, "env.observe")) == 2  # both observations went to the store


async def test_one_physical_action_per_turn(rig: Any) -> None:
    act = call("act", **ACT)
    replies = [[act, call("act", **ACT), call("get_robot_state")], [call("finish", note="done")]]
    session = await rig.session()
    result = await ScriptedBrain(replies).run(session)
    assert (result.end.reason, result.turns) == ("finish", 2)
    assert rig.mock.acts == 1
    assert (await rig.hub.state(session.episode_id)).counters.turns == 2  # act, then finish
    made = calls_of(session)[:3]
    assert [(c["status"], c["error"]) for c in made] == [
        ("executed", None),
        ("refused", "one_action_per_turn"),
        ("refused", "one_action_per_turn"),
    ]
    assert act["call_id"] in made[1]["content"][0]["text"]  # names the turn's action


async def test_replies_without_a_valid_tool_call_are_nudged_then_end_the_episode(
    rig: Any,
) -> None:
    broken = {"type": "function_call", "call_id": "c-bad", "name": "act", "arguments": "{nope"}
    replies = ["I will lift it now.", [broken], [call("teleport")]]
    session = await rig.session()
    result = await ScriptedBrain(replies).run(session)
    assert (result.end.reason, result.model_calls) == ("format_errors", 3)
    requests = logged(session)
    nudge = {"role": "user", "content": [{"type": "input_text", "text": LoopConfig().nudge}]}
    assert kinds(requests[1]["input"]) == ["user", "message", "user"]
    assert requests[1]["input"][2] == nudge
    output = requests[2]["input"][-1]
    assert (output["type"], output["call_id"]) == ("function_call_output", "c-bad")
    assert output["output"].startswith("Error (invalid_arguments)")
    exchanges = read_lines(session.recorder.turns)[0]["exchanges"]
    assert exchanges[0]["nudge"] == LoopConfig().nudge
    assert [(c["status"], c["error"]) for c in calls_of(session)] == [
        ("invalid", "invalid_arguments"),
        ("executed", "unknown_tool"),
    ]
    reached = {event.data["call_id"] for event in events(rig, session, "tool.call")}
    assert "c-bad" not in reached  # never sent to the Hub


async def test_every_request_is_rebuilt_from_the_log_as_it_was_sent(rig: Any) -> None:
    key = await rig.blobs.aput_image(np.full((8, 8, 3), 90, np.uint8))
    brief = [TextPart(text="Material for this episode."), ImagePart(blob=key, label="key frame")]
    requests: list[dict[str, Any]] = []
    replies = [
        [think("look closer"), call("get_image", camera="overhead", frame="frame_1")],
        [call("act", **ACT)],
        [call("finish", note="done")],
    ]
    session = await rig.session(brief=brief)
    await ScriptedBrain(recorder(requests, replies)).run(session)
    assert await rebuild_requests(session.recorder.calls, rig.blobs) == requests
    lines = read_lines(session.recorder.calls)
    assert [line["kind"] for line in lines].count("context") == 1  # instructions and tools once
    assert kinds(requests[0]["input"][0]["content"]) == ["input_text", "input_text", "input_image"]
    assert all(url.startswith("data:image/png;base64,") for url in image_urls(requests[1]))
    assert len(image_urls(requests[1])) == 5  # the key frame, the observation's 3, get_image's 1
    text = session.recorder.calls.read_text(encoding="utf-8")
    assert "base64" not in text and "data:image" not in text  # images are blob references
    sent = [event.data["sha256"] for event in events(rig, session, "model.request")]
    assert sent == [digest(request) for request in logged(session)]


async def test_the_loop_limits(rig: Any) -> None:
    replies = [[call("act", **ACT)], [call("look")]]
    session = await rig.session(limits=Limits(max_model_calls=1))
    result = await ScriptedBrain(replies).run(session)
    assert (result.end.reason, result.model_calls, result.turns) == ("max_model_calls", 1, 2)
    assert system_observes(rig, session) == 1  # checked before the next turn's observation

    async def slow(request: dict[str, Any]) -> Any:
        await anyio.sleep(5)
        return [call("look")]

    async def timing_out(request: dict[str, Any]) -> Any:
        raise TimeoutError("read timed out")

    brief = [TextPart(text="Go.")]
    for limit, model, reason in (
        ({"budgets": Budgets(wall_clock_s=0.5)}, slow, "wall_clock"),  # the Hub's clock
        ({"limits": Limits(wall_clock_s=0.3)}, slow, "wall_clock"),  # the loop's own
        ({}, timing_out, "gateway_error"),  # a model's own timeout is not the wall clock
    ):
        session = await rig.session(brief=brief, **limit)
        config = LoopConfig(observe=False, retry_waits_s=[])
        result = await ScriptedBrain(model, config).run(session)
        assert (result.end.reason, result.model_calls) == (reason, 1)
        lines = read_lines(session.recorder.calls)
        [response] = [line for line in lines if line["kind"] == "response"]
        assert response["response"] is None and response["error"].startswith(reason)


async def test_infrastructure_failures_end_the_episode(rig: Any) -> None:
    rig.env.broken = {"observe"}
    session = await rig.session()
    result = await ScriptedBrain([[call("look")]]).run(session)
    assert (result.end.reason, result.model_calls) == ("infra_error", 0)
    [turn] = read_lines(session.recorder.turns)
    assert turn["observation"][0]["text"].startswith("Error (infra_error)")

    rig.env.broken = {"robot"}
    state = call("get_robot_state")
    replies = [[state], [call("look")], [state], [state], [call("finish", note="?")]]
    session = await rig.session()
    result = await ScriptedBrain(replies).run(session)
    assert (result.end.reason, result.model_calls) == ("infra_error", 4)
    assert [c["error"] for c in calls_of(session)] == [
        "infra_error",
        None,  # a working call in between starts the count again
        "infra_error",
        "infra_error",
    ]


async def test_stage_rules_and_hub_budgets_end_the_loop(rig: Any) -> None:
    lost = [[call("act", **ACT, outcome="lost_hold")], [call("finish", note="?")]]
    session = await rig.session(stage="s2")
    result = await ScriptedBrain(lost).run(session)
    assert (result.end.reason, result.model_calls) == ("fail_fast:lost_hold@grasp", 1)
    [turn] = read_lines(session.recorder.turns)
    assert (turn["action"], turn["end"]["reason"]) == ("act", "fail_fast:lost_hold@grasp")

    acting = [[call("act", **ACT)]] * 2
    result = await ScriptedBrain(acting).run(await rig.session(budgets=Budgets(max_turns=1)))
    assert (result.end.reason, result.turns) == ("max_turns", 1)

    result = await ScriptedBrain([[call("get_robot_state")]]).run(await rig.session())
    assert (result.end.reason, result.model_calls) == ("script_exhausted", 2)


def image_urls(request: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "input_image":
                urls.append(value["image_url"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(request["input"])
    return urls
