"""The gateway over HTTP, in front of fake upstreams on local sockets (`GatewayRig`): routing,
failover, aggregation, authentication and the request log."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import openai
import pytest

from roboscript.testing.gateway import (
    USAGE,
    Drop,
    Events,
    GatewayRig,
    Reply,
    Status,
    Text,
)

HI: dict[str, Any] = {"model": "m", "input": [{"role": "user", "content": "hi"}]}
MESSAGES = [{"role": "user", "content": "hi"}]
CHAT: dict[str, Any] = {"model": "m", "messages": MESSAGES}
SUMMARY = [{"type": "summary_text", "text": "look first"}]
REASONING = {"type": "reasoning", "id": "rs_1", "summary": SUMMARY}
TURNS = [  # a conversation whose reasoning only the organization that wrote it can read
    {"role": "user", "content": "hi"},
    {**REASONING, "encrypted_content": "gAAAA-bound-to-its-organization"},
    {"type": "function_call", "call_id": "c1", "name": "observe", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c1", "output": "a red cube"},
]
STRIPPED = [TURNS[0], REASONING, *TURNS[2:]]  # the summary stays
"""What a busy relay answers (HTTP 529)."""


async def test_responses_are_streamed_upstream_and_aggregated_when_asked(tmp_path: Path) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig, rig.openai("s1") as client:
        answer = await client.responses.create(model="m", input="hi")
        assert answer.output_text == "ok"
        rig.upstream["ep1"].reply(Text("streamed"))
        stream = await client.responses.create(model="m", input="hi", stream=True)
        events = [event async for event in stream]
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert events[-1].response.output_text == "streamed"
    seen = rig.upstream["ep1"].seen
    # only `model` and `stream` change: every Responses call is streamed upstream
    assert [s.body for s in seen] == [{"model": "m-up", "input": "hi", "stream": True}] * 2
    assert seen[0].headers["authorization"] == f"Bearer {rig.keys['ep1']}"
    assert rig.upstream["ep2"].seen == []
    lines = rig.log()
    assert [(x["session"], x["model"], x["deployment"], x["stream"]) for x in lines] == [
        ("s1", "m", "ep1", False),
        ("s1", "m", "ep1", True),
    ]
    for line in lines:
        assert (line["protocol"], line["status"], line["error"], line["usage"]) == (
            "responses",
            200,
            None,
            USAGE,
        )
        assert line["tier"] == "primary"
        assert 0 <= line["first_byte_s"] <= line["total_s"]
        assert [a["outcome"] for a in line["attempts"]] == [200]


async def test_429_fails_over_and_cools_down_for_retry_after(tmp_path: Path) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2"], retry_after_cap_s=20) as rig:
        ep1, ep2 = rig.upstream["ep1"], rig.upstream["ep2"]
        ep1.reply(Status(429, headers={"retry-after": "7"}))
        response = await rig.post("/v1/responses", HI)
        assert response.status_code == 200 and response.headers["x-rs-deployment"] == "ep2"
        rig.clock.advance(6.5)  # ep1 still cools down, though it is its turn
        assert (await rig.post("/v1/responses", HI)).headers["x-rs-deployment"] == "ep2"
        rig.clock.advance(0.5)
        assert (await rig.post("/v1/responses", HI)).headers["x-rs-deployment"] == "ep1"
        ep2.reply(Status(429, headers={"retry-after": "3600"}))  # capped at 20 s
        assert (await rig.post("/v1/responses", HI)).headers["x-rs-deployment"] == "ep1"
        rig.clock.advance(19.5)  # ep2 still cools down
        assert (await rig.post("/v1/responses", HI)).headers["x-rs-deployment"] == "ep1"
        rig.clock.advance(0.5)
        assert (await rig.post("/v1/responses", HI)).headers["x-rs-deployment"] == "ep2"
    outcomes = [[a["outcome"] for a in x["attempts"]] for x in rig.log()]
    assert outcomes == [[429, 200], [200], [200], [429, 200], [200], [200]]


async def test_5xx_and_connection_errors_fail_over(tmp_path: Path) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2", "ep3"], unreachable=["ep2"]) as rig:
        rig.upstream["ep1"].reply(Status(502, {"error": {"message": "bad gateway upstream"}}))
        response = await rig.post("/v1/chat/completions", CHAT)
        assert response.status_code == 200 and response.headers["x-rs-deployment"] == "ep3"
    [line] = rig.log()
    ep1, ep2, ep3 = line["attempts"]
    assert (ep1["outcome"], ep2["outcome"], ep3["outcome"]) == (502, "ConnectError", 200)
    assert "bad gateway upstream" in ep1["detail"]


async def test_other_4xx_pass_through_without_failover_or_cooldown(tmp_path: Path) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig:
        first = await rig.post("/v1/responses", HI, session="s1")
        assert first.headers["x-rs-deployment"] == "ep1"
        message = f"Invalid 'input' sent with key {rig.keys['ep1']}"
        body = {"error": {"message": message, "type": "invalid_request_error", "code": "x"}}
        rig.upstream["ep1"].reply(Status(400, body))
        response = await rig.post("/v1/responses", {**HI, "stream": True}, session="s1")
        assert response.status_code == 400 and response.headers["x-rs-deployment"] == "ep1"
        error = response.json()["error"]
        assert (error["code"], error["type"]) == ("x", "invalid_request_error")
        assert error["message"] == "Invalid 'input' sent with key **********"  # never the key
        # no cooldown: the session's deployment is still ready, so it stays there
        again = await rig.post("/v1/responses", HI, session="s1")
        assert again.headers["x-rs-deployment"] == "ep1"
    assert rig.upstream["ep2"].seen == []
    lines = rig.log()
    assert [(x["status"], x["error"]) for x in lines] == [
        (200, None),
        (400, "upstream_status"),
        (200, None),
    ]


async def test_when_no_deployment_answers_the_client_gets_503(tmp_path: Path) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2"], cooldown_s=30) as rig:
        rig.upstream["ep1"].reply(Status(500))
        rig.upstream["ep2"].reply(Status(503, headers={"retry-after": "12"}))
        response = await rig.post("/v1/responses", HI)
        assert response.status_code == 503 and response.headers["retry-after"] == "12"
        error = response.json()["error"]
        assert error["code"] == "no_deployment_available"
        assert "(ep1: 500; ep2: 503)" in error["message"]
        # both cool down: the one whose cooldown ends first is tried all the same
        response = await rig.post("/v1/responses", HI)
        assert response.status_code == 200 and response.headers["x-rs-deployment"] == "ep2"
    assert len(rig.upstream["ep1"].seen) == 1


async def test_a_forced_move_strips_encrypted_reasoning(tmp_path: Path) -> None:
    body = {"model": "m", "input": TURNS, "include": ["reasoning.encrypted_content"]}
    async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig:
        first = await rig.post("/v1/responses", body, session="s1")
        assert first.headers["x-rs-deployment"] == "ep1"
        assert "x-rs-affinity-break" not in first.headers
        rig.upstream["ep1"].reply(Status(500))
        moved = await rig.post("/v1/responses", body, session="s1")
        assert moved.headers["x-rs-deployment"] == "ep2"
        assert moved.headers["x-rs-affinity-break"] == "1"
        rig.clock.advance(60)  # ep1 is back, but s1 now lives on ep2
        after = await rig.post("/v1/responses", body, session="s1")
        assert after.headers["x-rs-deployment"] == "ep2"
        assert "x-rs-affinity-break" not in after.headers
    assert [s.body["input"] for s in rig.upstream["ep1"].seen] == [TURNS, TURNS]
    stripped, kept = (s.body for s in rig.upstream["ep2"].seen)
    assert stripped["input"] == STRIPPED
    assert stripped["include"] == ["reasoning.encrypted_content"]
    assert kept["input"] == TURNS
    assert [x["affinity_break"] for x in rig.log()] == [False, True, False]


async def test_the_token_is_required_and_sessions_come_from_key_or_headers(
    tmp_path: Path,
) -> None:
    async with GatewayRig(tmp_path, ["ep1"]) as rig:
        for authorization in (None, "Bearer wrong", "Bearer wrong:s1", "Basic x"):
            headers = {} if authorization is None else {"authorization": authorization}
            response = await rig.http.post("/v1/responses", json=HI, headers=headers)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_api_key"
        assert (await rig.http.get("/v1/models")).status_code == 401
        assert rig.upstream["ep1"].seen == []
        await rig.post("/v1/responses", HI, session="via-key")
        await rig.post("/v1/responses", HI, headers={"x-rs-session-id": "via-header"})
        await rig.post("/v1/responses", HI, headers={"session-id": "via-codex-header"})
        await rig.post("/v1/responses", HI, session="key", headers={"x-rs-session-id": "no"})
        await rig.post("/v1/responses", HI, headers={"x-rs-session-id": "ours", "session-id": "no"})
        await rig.post("/v1/responses", HI)
        response = await rig.post("/v1/responses", HI, session="../../etc")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_session_id"
    lines = rig.log()
    assert [(x["status"], x["session"]) for x in lines] == [
        *[(401, None)] * 4,
        (200, "via-key"),
        (200, "via-header"),
        (200, "via-codex-header"),
        (200, "key"),
        (200, "ours"),
        (200, None),
        (400, None),
    ]


async def test_the_request_log_holds_no_secrets(tmp_path: Path) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig, rig.openai("s1") as client:
        key = rig.keys["ep1"]
        rig.upstream["ep1"].reply(
            Status(429, {"error": {"message": f"key {key} is over its quota"}}),
            # the key sits across the 300-character cut of the logged detail
            Status(500, {"error": {"message": f"{'x' * 264} key {key}"}}),
        )
        await client.responses.create(model="m", input="hi")
        rig.clock.advance(60)
        rig.upstream["ep2"].reply(Status(500))
        with pytest.raises(openai.InternalServerError):  # pinned to ep2, then ep1: both fail
            await client.responses.create(model="m", input="hi")
        stream = await client.responses.create(model="m", input="hi", stream=True)
        [_ async for _ in stream]
        await rig.http.post("/v1/responses", json=HI, headers={"authorization": "Bearer guess-1"})
    text = rig.log_text()
    for secret in (rig.token.get_secret_value(), key[:10], *rig.keys.values(), "guess-1"):
        assert secret not in text
    assert "127.0.0.1" not in text
    first, second, *_ = rig.log()
    assert "key ********** is over its quota" in first["attempts"][0]["detail"]
    assert second["attempts"][1]["detail"].endswith("key **********")


async def eventually(condition: Callable[[], bool], timeout_s: float = 5) -> None:
    """Wait until `condition()` holds."""
    with anyio.fail_after(timeout_s):
        while not condition():  # noqa: ASYNC110 - the state belongs to other tasks and servers
            await anyio.sleep(0.02)


async def logged(rig: GatewayRig, lines: int) -> list[dict[str, Any]]:
    """The request log once it has `lines` lines (a stream's line comes after the client left)."""
    await eventually(lambda: len(rig.log()) >= lines)
    return rig.log()


STARTED: dict[str, Any] = {"type": "response.created", "response": {}}


STREAM_FAILED: dict[str, Any] = {
    "type": "response.failed",
    "response": {"status": "failed", "error": {"code": "stream_error", "message": "interrupted"}},
}


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (Events([STARTED, STREAM_FAILED]), "response_failed:stream_error"),
        (Events([STARTED, {"type": "error", "code": "server_error"}]), "stream_error"),
        (Drop(after=1), "RemoteProtocolError"),
        (Events([STARTED]), "no_final_event"),
    ],
    ids=["failed-response", "error-event", "broken-off", "ended-early"],
)
async def test_a_stream_that_fails_before_the_client_got_anything_fails_over(
    tmp_path: Path, reply: Reply, outcome: str
) -> None:
    """Only the preamble came (held back): the whole request goes to the next deployment."""
    async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig, rig.openai("s1") as client:
        rig.upstream["ep1"].reply(reply)
        stream = await client.responses.create(model="m", input="hi", stream=True)
        events = [event async for event in stream]
    assert [e.type for e in events].count("response.created") == 1  # ep1's was never sent
    assert events[-1].type == "response.completed" and events[-1].response.output_text == "ok"
    [line] = await logged(rig, 1)
    assert [a["outcome"] for a in line["attempts"]] == [outcome, 200]
    assert (line["deployment"], line["status"], line["error"]) == ("ep2", 200, None)


async def test_a_stream_that_breaks_after_the_client_got_some_of_it_ends_with_an_error(
    tmp_path: Path,
) -> None:
    async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig:
        rig.upstream["ep1"].reply(Drop(after=3))
        response = await rig.post("/v1/responses", {**HI, "stream": True})
        assert response.status_code == 200 and "upstream_disconnected" in response.text
    assert rig.upstream["ep2"].seen == []  # bytes had gone out: no retry
    [line] = await logged(rig, 1)
    assert (line["deployment"], line["error"]) == ("ep1", "upstream_disconnected")
