"""Permits in the gateway (docs/concepts/agent-runtime.md): once a model answer in a session a Hub
registered is complete, its tool calls are permitted at the Hub's intake, in one batch, before the
client can see them; failed and broken-off answers leave none; the session needs its key."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import openai
import pytest
from pydantic import SecretStr

from roboscript.contracts.errors import InfraError
from roboscript.contracts.gateway import (
    Permit,
    PermitIntake,
    SessionRegistration,
    args_hash,
    description_hash,
    session_key_hash,
)
from roboscript.contracts.ids import SessionId
from roboscript.rpc import make_app, routes
from roboscript.testing.gateway import (
    USAGE,
    Events,
    GatewayRig,
    ToolCall,
    responses_events,
    serving,
)

ACT = "Perform one physical action; it takes a while."
LOOK = "Look at the scene: a new frame with its camera image."
PARAMETERS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
KEY = SecretStr("session-key-0123456789")
MESSAGES = [{"role": "user", "content": "go"}]


def function(name: str, description: str) -> dict[str, Any]:
    return {"type": "function", "name": name, "description": description, "parameters": PARAMETERS}


TOOLS = [function("mcp__rs__act", ACT), function("mcp__rs__look", LOOK)]


def answer(
    *calls: tuple[str, str], text: str | None = None, status: str = "completed"
) -> dict[str, Any]:
    """A Responses answer: an optional message, then function calls (name, arguments)."""
    output: list[dict[str, Any]] = []
    if text is not None:
        content = [{"type": "output_text", "text": text, "annotations": []}]
        output.append(
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content,
            }
        )
    for n, (name, arguments) in enumerate(calls, 1):
        output.append(
            {
                "id": f"fc_{n}",
                "type": "function_call",
                "status": "completed",
                "call_id": f"call_{n}",
                "name": name,
                "arguments": arguments,
            }
        )
    payload = {"id": "resp_1", "object": "response", "created_at": 0, "status": status}
    return payload | {"model": "m-up", "output": output, "usage": USAGE}


@dataclass
class Intake:
    """A Hub's permit intake: keeps what it is sent; `hold` delays the acknowledgement and
    `fail` refuses it."""

    token: SecretStr = field(default_factory=lambda: SecretStr("permit-token-0123456789"))
    permits: list[Permit] = field(default_factory=list)
    batches: int = 0
    hold: asyncio.Event | None = None
    fail: bool = False

    async def issue(self, permits: list[Permit]) -> None:
        if self.hold is not None:
            await self.hold.wait()
        if self.fail:
            raise InfraError("busy", "the Hub is busy (a detail the brain must not see)")
        self.permits += permits
        self.batches += 1


@dataclass
class Setup:
    rig: GatewayRig
    intake: Intake
    url: str

    async def register(self, session: str = "ep-1", **changes: Any) -> None:
        registration = {
            "session_id": SessionId(session),
            "permit_url": self.url,
            "permit_token": self.intake.token,
            "expires_at": time.time() + 600,
            "session_key_hash": session_key_hash(KEY),
            "tools": {description_hash(ACT): "act", description_hash(LOOK): "look"},
        }
        await self.rig.admin().register(SessionRegistration(**{**registration, **changes}))

    def client(self, session: str = "ep-1", key: SecretStr | None = KEY) -> openai.AsyncOpenAI:
        return self.rig.openai(session, key=key)

    def reply(self, *replies: Any) -> None:
        self.rig.upstream["ep1"].reply(*replies)

    async def lines(self, count: int) -> list[dict[str, Any]]:
        """The request log once it has `count` lines (a stream's is written as it closes)."""
        with anyio.fail_after(5):
            while len(self.rig.log()) < count:  # noqa: ASYNC110
                await anyio.sleep(0.01)
        return self.rig.log()


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[Setup]:
    async with GatewayRig(tmp_path, ["ep1"]) as rig:
        intake = Intake()
        app = make_app(routes(PermitIntake, intake, token=intake.token))
        async with serving(app, tmp_path) as url:
            ready = Setup(rig, intake, url)
            await ready.register()
            yield ready


def seen(intake: Intake) -> list[tuple[str, str, str | None]]:
    return [(p.tool, p.args_hash, p.call_id) for p in intake.permits]


async def test_an_aggregated_answer_waits_for_its_permits(setup: Setup) -> None:
    setup.reply(ToolCall("mcp__rs__act", '{"speed": 1.0}'))
    setup.intake.hold = asyncio.Event()
    async with setup.client() as client, anyio.create_task_group() as tasks:
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(await client.responses.create(model="m", input="go", tools=TOOLS))  # type: ignore[arg-type]

        tasks.start_soon(ask)
        await anyio.sleep(0.3)
        assert not answers  # the answer waits for the Hub's acknowledgement
        setup.intake.hold.set()
    assert answers[0].output[0].name == "mcp__rs__act"
    assert seen(setup.intake) == [("act", args_hash({"speed": 1}), "call_1")]  # 1.0 as 1
    assert setup.rig.log()[-1]["permits"] == {"issued": 1, "skipped": 0}


async def test_a_failed_or_broken_off_answer_gets_no_permits(setup: Setup) -> None:
    calls = (("mcp__rs__look", "{}"), ("mcp__rs__act", "{}"))
    failed = responses_events(answer(*calls, status="failed"))
    failed[-1] = {**failed[-1], "type": "response.failed"}
    failed[-1]["response"] = {
        **failed[-1]["response"],
        "error": {"code": "invalid_prompt", "message": "boom"},
    }
    broken = responses_events(answer(*calls))[:-1]  # no final event at all
    setup.reply(Events(failed), Events(failed), Events(broken))
    async with setup.client() as client:
        with pytest.raises(openai.APIStatusError):
            await client.responses.create(model="m", input="go", tools=TOOLS)  # type: ignore[arg-type]
        stream = await client.responses.create(model="m", input="go", tools=TOOLS, stream=True)  # type: ignore[arg-type]
        assert [event.type async for event in stream][-1] == "response.failed"
        stream = await client.responses.create(model="m", input="go", tools=TOOLS, stream=True)  # type: ignore[arg-type]
        kinds = [event.type async for event in stream]
    assert not any("function_call" in kind for kind in kinds)  # its held calls were dropped
    assert setup.intake.permits == []


async def test_a_registered_session_needs_its_key(setup: Setup) -> None:
    setup.reply(ToolCall("mcp__rs__act", "{}"))
    for key in (None, SecretStr("wrong-key-0123456789")):
        async with setup.client(key=key) as client:
            with pytest.raises(openai.PermissionDeniedError, match="key"):
                await client.responses.create(model="m", input="go", tools=TOOLS)  # type: ignore[arg-type]
    headers = {**setup.rig.auth("ep-1"), "x-rs-session-key": KEY.get_secret_value()}
    body = {"model": "m", "input": "go", "tools": TOOLS}
    assert (
        await setup.rig.http.post("/v1/responses", json=body, headers=headers)
    ).status_code == 200
    assert setup.rig.upstream["ep1"].seen[-1].body["input"] == "go"  # only the keyed one left
    assert len(setup.rig.upstream["ep1"].seen) == 1
    assert [line["error"] for line in setup.rig.log()] == ["session_key_required"] * 2 + [None]
