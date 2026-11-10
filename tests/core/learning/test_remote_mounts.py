"""Remote toolsets in a process that drives its Hub itself (`roboscript.learning.mounts`): the
learning runner mounts a toolset served by its own server exactly as the same toolset in
process (tool specs in order, results), and the information boundary holds for remote tools as
for local ones. The Agent host's half is in `test_agent_host.py`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from roboscript.agent import ScriptedBrain
from roboscript.agent.scripted import call
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.events import EventKind
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import ToolName
from roboscript.contracts.tools import Tier, ToolCall
from roboscript.hub import Hub
from roboscript.hub.tokens import Tokens, service_key
from roboscript.integrations.mcp import RemoteProvider
from roboscript.integrations.mcp.remote import serve_remote
from roboscript.integrations.mcp.serve import ToolsetServer, ToolsetServerConfig
from roboscript.learning import Evaluate
from roboscript.records import EventWriter, read_events
from roboscript.testing.gateway import serving
from roboscript.testing.methods import PrivilegeProbe, relayed_phases
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES, extended_phases
from roboscript.testing.protocols import LIFT_RED, NO_OBSERVE, RATIONALE, lift_memory
from roboscript.tools import LocalProvider, Toolset

SECRET = SecretStr("m" * 40)
FAST = {"act_seconds": 0.0}


@asynccontextmanager
async def served(
    toolset: Toolset, blobs: Path, config: dict[str, Any] | None = None, key: SecretStr = SECRET
) -> AsyncIterator[str]:
    """`toolset` on its own MCP server for the block (the run's blob store, the service key of
    `key`): its endpoint."""
    settings = ToolsetServerConfig(blobs=str(blobs), toolset=config or {})
    tokens = Tokens({"service": service_key(key)})
    async with httpx.AsyncClient(trust_env=False) as http:
        server = ToolsetServer(
            toolset, settings, tokens=tokens, blobs=LocalBlobStore(blobs), http=http
        )
        async with server, serving(server.app(), blobs.parent) as url:
            yield f"{url}/mcp"


@pytest.fixture
def secret(monkeypatch: pytest.MonkeyPatch) -> SecretStr:
    """The run's `RS_HUB_SECRET`, as a deployment sets it."""
    monkeypatch.setenv("RS_HUB_SECRET", SECRET.get_secret_value())
    return SECRET


def mock_actor() -> ScriptedBrain:
    """An Actor that calls each of the Mock toolset's tools once, then finishes."""
    replies = [
        [call("look")],
        [call("act", rationale=RATIONALE)],
        [call("fail")],
        [call("finish", note="done")],
    ]
    return ScriptedBrain(replies, NO_OBSERVE)


async def evaluated(setup: Any, mock: Any) -> tuple[list[Any], list[Any]]:
    """One `evaluate` try with the Mock toolset mounted as `mock`: the runner's mounted tools
    (name, descriptor) in order, and what each Mock call came back as."""
    await setup.start(lift_memory(0.06))
    await setup.store.freeze(LIFT_RED, 0, reason="for the test")
    async with setup.runner({"actor": lambda _episode: mock_actor()}, extra=[mock]) as runner:
        mounted = [(name, tool.descriptor) for name, tool in runner.hub.mounted.items()]
        await runner.run(Evaluate(), LIFT_RED, {"seeds": [20000]})
    results = [
        {key: event.data.get(key) for key in ("name", "is_error", "code", "content")}
        for event in read_events(
            setup.run_dir / LIFT_RED / "eval" / "v000" / "20000" / "try1" / "events.jsonl"
        )
        if event.kind == EventKind.TOOL_RESULT and event.data["name"] in ("look", "act", "fail")
    ]
    return mounted, results


async def test_the_runner_mounts_a_served_toolset_as_the_same_one_in_process(
    setup: Any, secret: SecretStr
) -> None:
    local = await evaluated(setup, LocalProvider(Mock(), FAST))
    setup.run_dir = setup.run_dir.with_name("run-remote")
    mock = Mock()
    async with served(mock, setup.blobs.root, FAST) as url:
        remote = await evaluated(setup, RemoteProvider(Mock, url))
    assert mock.acts == 1  # the server acted
    assert remote[0] == local[0]  # names, descriptions, schemas, order
    assert [r["name"] for r in local[1]] == ["look", "act", "fail"]
    assert remote[1] == local[1]


def text(outcome: Any) -> str:
    return "\n".join(part.text for part in outcome.content if getattr(part, "text", None))


async def test_remote_tools_keep_the_information_boundary(tmp_path: Path) -> None:
    """A tier-P tool served remotely is shown and called as a local one would be: it runs for an
    S1-like learn phase at tier P (a remote tool's call back through the Hub included); an S2-like
    D-only learn phase and an eval phase never show it, refuse it as a model's call and refuse a
    remote tool's call back for it, the eval phase even for an episode recorded at tier P. (The
    refusal of a call back reaches the remote tool as a failure of the Hub, not as the
    `PermissionError` a local tool gets: the model reads that the tool failed.)"""
    learn = {"actor": ["privilege_probe.peek", "privilege_probe.relay"]}
    phases = relayed_phases(extended_phases(TEST_PHASES, learn, after="mock", only=["s1", "s2"]))
    tokens = Tokens.from_secret(SECRET)
    blobs = LocalBlobStore(tmp_path / "blobs")
    log = tmp_path / "hub.jsonl"
    async with served(PrivilegeProbe(), blobs.root) as url:
        remote = RemoteProvider(PrivilegeProbe, url)
        with EventWriter(log, "hub") as events:
            hub = Hub(
                [LocalProvider(Mock(), FAST), remote], blobs=blobs, phases=phases, events=events
            )
            async with hub, serve_remote(hub, [remote], tokens=tokens):

                async def run(stage: str, name: str, tier: Tier = "D") -> tuple[Any, Any]:
                    session = await hub.open_episode(OpenEpisode(stage=stage, tier=tier))
                    request = ToolCall(
                        call_id="c1", name=ToolName(name), arguments={}, origin="model"
                    )
                    return session, await hub.call(session.episode_id, request)

                session, relayed = await run("s1", "relay", tier="P")
                assert {"peek", "relay"} <= {tool.name for tool in session.tools}
                assert not relayed.is_error and "peeked=true" in text(relayed)
                for stage in ("s2", "p3"):
                    session, peeked = await run(stage, "peek")
                    assert "peek" not in [tool.name for tool in session.tools], stage
                    assert (peeked.is_error, peeked.error_code) == (True, "unknown_tool"), stage
                    _, relayed = await run(stage, "relay")
                    assert relayed.is_error and "peeked" not in text(relayed), stage
                session, _ = await run("p3", "relay")
                hub.store.get(session.episode_id).tier = "P"  # had anything opened one at tier P
                again = ToolCall(call_id="c2", name=ToolName("relay"), arguments={}, origin="model")
                relayed = await hub.call(session.episode_id, again)
                assert relayed.is_error and "peeked" not in text(relayed)
    peeks = [
        event.data
        for event in read_events(log)
        if event.kind == EventKind.TOOL_RESULT and event.data["name"] == "peek"
    ]
    # only S1's call back ran: the Hub refused the others before any tool (S2's, P3's, and the
    # first of the episode later set to tier P), and that episode's second never reached it (its
    # service token is tier D's)
    assert [(peek["origin"], peek["is_error"]) for peek in peeks] == [("service", False)]
