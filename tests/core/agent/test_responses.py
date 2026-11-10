"""ResponsesBrain through a real gateway in front of a scripted fake upstream: the upstream receives
exactly the requests the log records (store off, strict tools, no parallel calls, images as data
URLs, encrypted reasoning replayed only within its turn, the episode as prompt-cache key); a
failed model call ends the episode with a classified reason. Then the brain conformance suite on
both brains of the own loop."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from roboscript.agent import (
    ResponsesBrain,
    ResponsesConfig,
    ScriptedBrain,
    call,
    read_lines,
    rebuild_requests,
)
from roboscript.records import read_events
from roboscript.testing.brain import BrainConformance, BrainTarget, Script
from roboscript.testing.gateway import USAGE, Events, GatewayRig, Status

RATIONALE = {"stage": "grasp", "scene": "a red box", "progress": "starting", "plan": "act"}
REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA-1"}


def completed(*items: dict[str, Any]) -> Events:
    """An upstream answer whose only event is the completed response with `items`."""
    response = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "m-up",
        "output": list(items),
        "usage": USAGE,
    }
    return Events([{"type": "response.completed", "sequence_number": 0, "response": response}])


def function_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


async def test_what_the_upstream_receives_is_what_the_log_records(rig: Any, tmp_path: Path) -> None:
    async with GatewayRig(tmp_path / "gateway", ["ep1"]) as gateway:
        upstream = gateway.upstream["ep1"]
        upstream.reply(
            completed(REASONING, function_call("get_robot_state", {}, "c1")),
            completed(function_call("act", {"rationale": RATIONALE}, "c2")),
            completed(function_call("finish", {"note": "done"}, "c3")),
        )
        session = await rig.session(models=gateway.openai)
        result = await ResponsesBrain(ResponsesConfig(model="m")).run(session)
        log = list(read_events(Path(gateway.config["log_file"])))
    assert (result.end.reason, result.turns, result.model_calls) == ("finish", 2, 3)
    assert result.usage.input_tokens == 3 * USAGE["input_tokens"]
    seen = [request.body for request in upstream.seen]
    sent = await rebuild_requests(session.recorder.calls, rig.blobs)
    assert [{**request, "model": "m-up", "stream": True} for request in sent] == seen
    first = seen[0]
    assert (first["store"], first["parallel_tool_calls"]) == (False, False)
    assert first["include"] == ["reasoning.encrypted_content"]
    assert first["reasoning"] == {"effort": "low", "summary": "auto"}
    assert (first["tool_choice"], first["prompt_cache_key"]) == ("required", session.episode_id)
    assert all(tool["strict"] for tool in first["tools"])
    urls = [p["image_url"] for p in first["input"][0]["content"] if p["type"] == "input_image"]
    assert len(urls) == 3 and all(url.startswith("data:image/png;base64,") for url in urls)
    assert REASONING in seen[1]["input"]  # replayed in its turn
    assert not any(item.get("type") == "reasoning" for item in seen[2]["input"])
    calls = [item for item in seen[2]["input"] if item.get("type") == "function_call"]
    assert [c["call_id"] for c in calls] == ["c1", "c2"] and all("id" not in c for c in calls)
    responses = [line for line in read_lines(session.recorder.calls) if line["kind"] == "response"]
    assert [line["response"]["deployment"] for line in responses] == ["ep1"] * 3
    assert {(event.kind, event.episode_id) for event in log} == {
        ("gateway.request", session.episode_id)  # the gateway's own kind: usage counted once
    }


@pytest.mark.parametrize(
    ("reply", "model", "reason"),
    [
        (
            Status(400, {"error": {"message": "unsafe", "code": "invalid_prompt"}}),
            "m",
            "model_error:policy:invalid_prompt",
        ),
        (
            Status(400, {"error": {"message": "no", "code": "unsupported_parameter"}}),
            "m",
            "model_error:config:unsupported_parameter",
        ),
        (Status(503), "m", "gateway_error"),
        (None, "no-such-model", "model_error:config:model_not_found"),
    ],
)
async def test_a_failed_model_call_ends_the_episode_with_a_classified_reason(
    rig: Any, tmp_path: Path, reply: Status | None, model: str, reason: str
) -> None:
    async with GatewayRig(tmp_path / "gateway", ["ep1"]) as gateway:
        if reply is not None:
            gateway.upstream["ep1"].reply(reply)
        session = await rig.session(models=gateway.openai)
        config = ResponsesConfig(model=model, retry_waits_s=[])
        result = await ResponsesBrain(config).run(session)
    assert (result.end.reason, result.model_calls) == (reason, 1)
    [line] = [line for line in read_lines(session.recorder.calls) if line["kind"] == "response"]
    assert line["response"] is None and line["error"].startswith(reason)


"""An upstream's mid-stream failure, which the gateway answers with a 503 once its only deployment
has failed on the request."""


class TestScriptedBrainConformance(BrainConformance):
    @pytest.fixture
    def brain_target(self, rig: Any) -> BrainTarget:
        def brain(script: Script) -> ScriptedBrain:
            return ScriptedBrain([[call(name, **args) for name, args in reply] for reply in script])

        return BrainTarget(lambda stage: rig.session(stage=stage), brain, rig.blobs)


class TestResponsesBrainConformance(BrainConformance):
    @pytest.fixture
    async def brain_target(self, rig: Any, tmp_path: Path) -> AsyncIterator[BrainTarget]:
        async with GatewayRig(tmp_path / "gateway", ["ep1"]) as gateway:

            def brain(script: Script) -> ResponsesBrain:
                for n, reply in enumerate(script):
                    items = [
                        function_call(name, args, f"c{n}-{i}")
                        for i, (name, args) in enumerate(reply)
                    ]
                    gateway.upstream["ep1"].reply(completed(*items))
                return ResponsesBrain(ResponsesConfig(model="m"))

            yield BrainTarget(
                session=lambda stage: rig.session(stage=stage, models=gateway.openai),
                brain=brain,
                blobs=rig.blobs,
            )
