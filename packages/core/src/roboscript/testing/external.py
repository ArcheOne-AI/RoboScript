"""Conformance of the Hub's surface for external brains (docs/concepts/agent-runtime.md): the
tool list, the invariants, isolation and permits, over MCP against a served Hub whose gateway
answers with scripted tool calls. Needs the `mcp` extra.

`ExternalBrainConformance` holds the checks as `test_*` methods taking a `ServedHub`; subclass
it as a `Test*` class with a `served_hub` fixture, or run it with `roboscript.testing.precheck`.
`ServedHub` serves a Hub over the Mock toolset with permits on, behind a real gateway in front of
`FakeUpstream`. Its `ScriptedBrain` does what a framework does, with a scripted model: it turns
the MCP tool list into function tools named like Codex's (`mcp__rs__<name>`), lets the model
propose each call through the gateway in the episode's session, with its key (which permits
it), then makes the call over MCP.
"""

import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import anyio
import httpx
import openai
from pydantic import SecretStr
from typing_extensions import Self

from roboscript import rpc
from roboscript.blobs import open_store
from roboscript.contracts.gateway import PermitMode
from roboscript.contracts.hub import (
    ENDED_MESSAGE,
    Budgets,
    EpisodeSession,
    HubControl,
    OpenEpisode,
)
from roboscript.contracts.ids import ToolName
from roboscript.hub.service import HubServerConfig, PermitsConfig, ToolsetMount
from roboscript.hub.tokens import Tokens, control_token, service_key
from roboscript.integrations.hub_server import run
from roboscript.integrations.mcp import Era, connect
from roboscript.integrations.mcp.serve import ToolsetServer, ToolsetServerConfig
from roboscript.integrations.mcp.wire import CODE, ENDED
from roboscript.testing.gateway import GatewayRig, ToolCall, running, serving
from roboscript.testing.mock import Mock

PREFIX = "mcp__rs__"  # how Codex names the tools of our server `rs` for its model
# The tools an Actor sees in a Mock episode: the Mock benchmark's, then the Hub's own.
MOCK_TOOLS = ["look", "act", "fail", "finish", "scratchpad"]
RATIONALE = {"stage": "grasp", "scene": "a box", "progress": "start", "plan": "act"}


@dataclass(frozen=True)
class Reply:
    """A `tools/call` answer as the checks read it."""

    is_error: bool
    code: str | None
    text: str
    images: int
    ended: bool


class ServedHub:
    """`async with ServedHub(root) as hub:` a Hub served by `roboscript.integrations.hub_server`
    over the Mock toolset (its actions take `act_seconds`; on a remote toolset server with
    `remote`), permits on in `mode`, MCP spoken in `era`, behind a `GatewayRig`; its tokens run
    on a clock that `advance` moves. Files go under `root`."""

    def __init__(
        self,
        root: Path,
        *,
        era: Era = "2026-07-28",
        mode: PermitMode = "strict",
        act_seconds: float = 0.3,
        remote: bool = False,
    ) -> None:
        self.root = root
        self.era = era
        self.mode = mode
        self.act_seconds = act_seconds
        self.remote = remote
        self.secret = SecretStr(secrets.token_urlsafe(32))
        self.toolset_url: str | None = None  # the remote toolset server's endpoint
        self._shift = 0.0
        self.tokens = Tokens.from_secret(self.secret, clock=lambda: time.time() + self._shift)
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        stack = self._stack
        try:
            self.gateway = await stack.enter_async_context(
                GatewayRig(self.root / "gateway", ["ep1"])
            )
            mount = await self._mock(stack)
            config = HubServerConfig(
                method="roboscript.testing.phases:TestMethod",
                toolsets=[mount],
                blobs=str(self.root / "blobs"),
                events=self.root / "hub.jsonl",
                permits=PermitsConfig(gateway=self.gateway.url, mode=self.mode),
            )
            serve_hub = run(
                config,
                ready_file=self.root / "hub.ready.json",
                secret=self.secret,
                admin_token=self.gateway.admin_token,
                tokens=self.tokens,
            )
            ready = await stack.enter_async_context(
                running(serve_hub, self.root / "hub.ready.json")
            )
            self.planes = ready.urls
            http = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
            self.control: HubControl = rpc.client(
                HubControl,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
                self.planes["hub.control"],
                token=control_token(self.secret),
                http=http,
                timeout=60,
            )
        except BaseException:
            await stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()

    @property
    def events(self) -> Path:
        return self.root / "hub.jsonl"

    def advance(self, seconds: float) -> None:
        """Move the tokens' clock."""
        self._shift += seconds

    async def open(self, stage: str = "s2", **options: Any) -> EpisodeSession:
        return await self.control.open_episode(OpenEpisode(stage=stage, **options))  # type: ignore[arg-type]

    @asynccontextmanager
    async def mcp(self, token: SecretStr) -> AsyncIterator[Any]:
        """An MCP client of the Hub with `token` as bearer."""
        async with connect(self.planes["mcp"], token=token, era=self.era) as client:
            yield client

    @asynccontextmanager
    async def brain(self, session: EpisodeSession) -> AsyncIterator["ScriptedBrain"]:
        """The scripted brain of `session`, connected with its episode token."""
        assert session.access is not None
        async with self.mcp(session.access.token) as client:
            yield ScriptedBrain(self, session, client, await client.list_tools())

    def offered(self) -> list[str]:
        """The tool names the last model request through the gateway offered."""
        return [tool["name"] for tool in self.gateway.upstream["ep1"].seen[-1].body["tools"]]

    async def _mock(self, stack: AsyncExitStack) -> ToolsetMount:
        settings = {"act_seconds": self.act_seconds}
        if not self.remote:
            return ToolsetMount(name="mock", config=settings)
        config = ToolsetServerConfig(blobs=str(self.root / "blobs"), toolset=settings)
        http = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
        blobs = open_store(config.shared_blobs())
        tokens = Tokens({"service": service_key(self.secret)}, clock=self.tokens.clock)
        server = ToolsetServer(Mock(), config, tokens=tokens, blobs=blobs, http=http)
        await stack.enter_async_context(server)
        url = await stack.enter_async_context(serving(server.app(), self.root))
        self.toolset_url = f"{url}/mcp"
        return ToolsetMount(name="mock", url=self.toolset_url)


class ScriptedBrain:
    """A brain with a scripted model: each call is first proposed by a model answer through the
    gateway, under the Codex-style name, then made over MCP with the episode token."""

    def __init__(self, hub: ServedHub, session: EpisodeSession, client: Any, listing: Any) -> None:
        self.hub = hub
        self.session = session
        self.client = client
        self.listing = listing
        self.functions = [
            {
                "type": "function",
                "name": PREFIX + tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": True,
            }
            for tool in listing.tools
        ]

    async def propose(self, name: str, arguments: dict[str, Any] | None = None) -> None:
        """A model answer proposing `name(arguments)` passes the gateway for this episode."""
        upstream = self.hub.gateway.upstream["ep1"]
        upstream.reply(ToolCall(PREFIX + name, json.dumps(arguments or {})))
        gateway, session = self.hub.gateway, self.session
        async with gateway.openai(session.episode_id, key=session.session_key) as model:
            tools: Any = self.functions
            answer = await model.responses.create(model="m", input="go on", tools=tools)
        assert answer.output[0].name == PREFIX + name  # type: ignore[union-attr]

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        propose: bool = True,
        proposed: dict[str, Any] | None = None,
    ) -> Reply:
        """Call `name` over MCP, first proposed by the model (with `proposed` if the client
        sends other arguments than the model gave) unless `propose` is false."""
        if propose:
            await self.propose(name, arguments if proposed is None else proposed)
        return reply(await self.client.call_tool(name, arguments or {}))


def reply(result: Any) -> Reply:
    meta = result.meta or {}
    texts = [part.text for part in result.content if part.type == "text"]
    images = sum(1 for part in result.content if part.type == "image")
    return Reply(result.is_error, meta.get(CODE), " ".join(texts), images, bool(meta.get(ENDED)))


class ExternalBrainConformance:
    """The checks, as `test_*` methods taking a `ServedHub` (Mock toolset, permits on)."""

    stages: ClassVar[tuple[str, ...]] = ("s1", "s2", "p3", "dev")

    async def test_c1_tools_are_the_episode_s_in_fixed_order(self, served_hub: ServedHub) -> None:
        for stage in self.stages:
            session = await served_hub.open(stage)
            async with served_hub.brain(session) as brain:
                names = [tool.name for tool in brain.listing.tools]
                assert names == [tool.name for tool in session.tools] == MOCK_TOOLS, stage
                assert all(tool.output_schema is None for tool in brain.listing.tools)
                assert not (await brain.call("look")).is_error
                assert [tool.name for tool in (await brain.client.list_tools()).tools] == names
                assert served_hub.offered() == [PREFIX + name for name in names]  # no `hidden`
        structured = await served_hub.open("s2", structured=True)
        async with served_hub.brain(structured) as brain:
            assert all(tool.output_schema is not None for tool in brain.listing.tools)

    async def test_c5_one_action_at_a_time(self, served_hub: ServedHub) -> None:
        session = await served_hub.open()
        async with served_hub.brain(session) as brain, anyio.create_task_group() as tasks:
            await brain.propose("act", {"rationale": RATIONALE})
            first: list[Reply] = []

            async def act() -> None:
                first.append(await brain.call("act", {"rationale": RATIONALE}, propose=False))

            tasks.start_soon(act)
            with anyio.fail_after(10):
                while not (await served_hub.control.state(session.episode_id)).acting:  # noqa: ASYNC110
                    await anyio.sleep(0.01)
            second = await brain.call("act", {"rationale": RATIONALE})
        assert second.code == "action_in_progress" and not first[0].is_error
        assert (await served_hub.control.state(session.episode_id)).counters.turns == 1

    async def test_c5_calls_after_finish_get_the_fixed_error(self, served_hub: ServedHub) -> None:
        session = await served_hub.open()
        async with served_hub.brain(session) as brain:
            done = await brain.call("finish", {"note": "the box is placed"})
            assert not done.is_error and done.ended
            for name, arguments in [("look", {}), ("act", {"rationale": RATIONALE})]:
                after = await brain.call(name, arguments)
                assert (after.code, after.text, after.ended) == (
                    "episode_ended",
                    ENDED_MESSAGE,
                    True,
                )
            again = await brain.call("finish", {"note": "again"}, propose=False)
            assert again.code == "episode_ended"

    async def test_c5_over_budget_and_hidden_calls_are_refused(self, served_hub: ServedHub) -> None:
        session = await served_hub.open(budgets=Budgets(max_info_calls=1))
        async with served_hub.brain(session) as brain:
            assert not (await brain.call("look")).is_error
            assert (await brain.call("look")).code == "budget_exhausted"
            hidden = await brain.call("hidden", propose=False)  # no model can propose it
            assert (hidden.code, hidden.text) == (
                "unknown_tool",
                "There is no tool named 'hidden'.",
            )

    async def test_c6_a_token_opens_only_its_own_episode(self, served_hub: ServedHub) -> None:
        a, b = await served_hub.open(), await served_hub.open()
        assert a.access is not None and b.access is not None
        async with served_hub.brain(a) as brain:
            assert not (await brain.call("act", {"rationale": RATIONALE})).is_error
        turns = {
            s.episode_id: (await served_hub.control.state(s.episode_id)).counters.turns
            for s in (a, b)
        }
        assert turns == {a.episode_id: 1, b.episode_id: 0}
        forged = SecretStr(a.access.token.get_secret_value()[:-2] + "xx")
        await _unauthorized(served_hub, forged)
        await served_hub.control.close_episode(a.episode_id)
        await _unauthorized(served_hub, a.access.token)  # revoked with its episode
        async with served_hub.brain(b):  # B's still works
            pass
        served_hub.advance(b.access.expires_at - time.time() + 1)
        await _unauthorized(served_hub, b.access.token)  # expired

    async def test_c10_only_model_proposed_calls_run(self, served_hub: ServedHub) -> None:
        session = await served_hub.open()
        episode = session.episode_id
        async with served_hub.brain(session) as brain:
            unproposed = await brain.call("act", {"rationale": RATIONALE}, propose=False)
            assert unproposed.code == "no_permit"
            assert (await served_hub.control.state(episode)).counters.turns == 0  # it never ran
            await brain.propose("look")
            assert not (await brain.call("look", propose=False)).is_error
            replayed = await brain.call("look", propose=False)  # a permit is used once
            assert replayed.code == "no_permit"
            proposed = {"rationale": RATIONALE, "outcome": "ok"}
            rewritten = await brain.call(
                "act", {"rationale": RATIONALE, "outcome": "stopped_short"}, proposed=proposed
            )
            if served_hub.mode == "strict":
                assert rewritten.code == "no_permit"
            else:
                assert not rewritten.is_error and "stopped_short" in rewritten.text
        looked = await served_hub.control.system_call(episode, ToolName("look"), {})
        assert not looked.is_error  # the runtime's channel needs no permit
        assert (await served_hub.control.state(episode)).counters.system_calls == 1
        async with served_hub.gateway.openai(episode) as keyless:  # the session needs its key
            try:
                await keyless.responses.create(model="m", input="go")
            except openai.PermissionDeniedError as error:
                assert "key" in error.message
            else:
                raise AssertionError("a request without the session key went through")


async def _unauthorized(hub: ServedHub, token: SecretStr) -> None:
    """The MCP endpoint answers `token` with 401 and the Bearer challenge."""
    listing = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {
        "Authorization": f"Bearer {token.get_secret_value()}",
        "Accept": "application/json, text/event-stream",
    }
    async with httpx.AsyncClient(trust_env=False) as http:
        response = await http.post(hub.planes["mcp"], json=listing, headers=headers)
    assert response.status_code == 401, (response.status_code, response.text)
    assert response.headers["www-authenticate"].startswith("Bearer")
