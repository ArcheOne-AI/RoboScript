"""Remote toolsets (docs/concepts/agent-runtime.md): a toolset on its own MCP server, mounted in
the Hub by `RemoteProvider`, behaves like a local one; the server takes only service tokens of
the context's episode; an unreachable server is an infrastructure error."""

import functools
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import httpx
import numpy as np
import pytest
from pydantic import BaseModel, SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.camera import CameraFrame, CameraModel
from roboscript.contracts.errors import ToolError
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import EpisodeId, FrameId, ToolName, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState
from roboscript.contracts.tools import ToolCall
from roboscript.hub import Hub
from roboscript.hub.service import HubService, serve_hub
from roboscript.hub.tokens import Tokens, control_token, service_key
from roboscript.integrations.mcp import RemoteProvider, connect, hub_app
from roboscript.integrations.mcp.serve import ToolsetServer, ToolsetServerConfig
from roboscript.integrations.mcp.wire import CONTEXT, FRAMES, PROBLEM
from roboscript.records import EventWriter, read_events
from roboscript.testing.gateway import running, serving
from roboscript.testing.mock import Mock
from roboscript.testing.phases import dev_phases
from roboscript.tools import (
    TABLE,
    FrameRef,
    LocalProvider,
    ToolContext,
    Toolset,
    tool,
)

SECRET = SecretStr("r" * 40)
EYE = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class Relayed(BaseModel):
    frame: FrameRef
    calls: int
    episode: str
    stage: str


class Snapped(BaseModel):
    frame: FrameRef


class Recalled(BaseModel):
    step: int
    table: dict[str, Any] | None


class Marked(BaseModel):
    objects: list[str]


CUBE = np.array([[x, y, z] for x in (0, 0.05) for y in (0, 0.05) for z in (0.7, 0.75)])


class Relay(Toolset):
    """Served remotely: calls another toolset through the Hub, keeps per-episode state, takes
    frames and sets facts."""

    name = "relay"

    def __init__(self) -> None:
        self.closed: list[EpisodeId] = []
        self.hold: anyio.Event | None = None  # set: `mark` waits for it after its edit
        self.held = anyio.Event()

    @tool(kind="perception", tier="D")
    async def relay(self, ctx: ToolContext) -> Relayed:
        """Look through the Hub and count this episode's relays."""
        looked = await ctx.call("look")
        ctx.state["calls"] = ctx.state.get("calls", 0) + 1
        return Relayed(
            frame=looked.frame,  # type: ignore[attr-defined]
            calls=ctx.state["calls"],
            episode=ctx.episode_id,
            stage=ctx.context.stage,
        )

    @tool(kind="perception", tier="D")
    async def snap(self, ctx: ToolContext, step: int) -> Snapped:
        """Take a frame and estimate the table."""
        rgb = await ctx.blobs.aput_image(np.zeros((2, 4, 3), np.uint8))
        K = ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
        camera = CameraModel(name="head", width=4, height=2, K=K, cam2world=EYE)
        robot = RobotState(step=step, step_limit=10, arms={}, qpos_cmd=[])
        frame = FrameId(new_id("frame"))
        cameras = {"head": CameraFrame(camera=camera, rgb=rgb)}
        await ctx.frames.put(Observation(frame_id=frame, step=step, cameras=cameras, robot=robot))
        ctx.frames.set_fact(TABLE, {"z": 0.75, "extent": [-0.5, 0.5, -0.3, 0.3]})
        return Snapped(frame=frame)

    @tool(kind="perception", tier="D")
    async def mark(self, ctx: ToolContext, role: str) -> Marked:
        """Register a cube under a role in the episode's scene registry."""
        async with ctx.frames.edit_registry() as registry:
            registry.register(role, CUBE, frame=FrameId("frame-a"), step=0)
            names = [o.name for o in registry.objects.values()]
        if self.hold is not None:
            self.held.set()
            await self.hold.wait()
        return Marked(objects=names)

    @tool(kind="pure", tier="D")
    async def recall(self, ctx: ToolContext, frame: FrameRef) -> Recalled:
        """Read a frame and the table back."""
        observation = await ctx.frames.get(frame)
        return Recalled(step=observation.step, table=ctx.frames.fact(TABLE))

    @tool(kind="pure", tier="D")
    async def broken(self, ctx: ToolContext, how: Literal["refuse", "crash"]) -> Relayed:
        """Fail in one of two ways."""
        if how == "refuse":
            raise ToolError("refused on purpose")
        raise RuntimeError("crashed")

    async def close_episode(self, episode_id: EpisodeId) -> None:
        self.closed.append(episode_id)


@dataclass
class Remote:
    service: HubService
    relay: Relay
    server: ToolsetServer
    url: str

    async def open(self) -> EpisodeId:
        return (await self.service.open_episode(OpenEpisode(stage="dev"))).episode_id  # relay: dev


async def call(
    service: HubService, episode: EpisodeId, name: str, **arguments: object
) -> CallOutcome:
    request = ToolCall(
        call_id=new_id("call"), name=ToolName(name), arguments=arguments, origin="model"
    )
    return await service.call(episode, request)


def text(outcome: CallOutcome) -> str:
    return outcome.content[0].text  # type: ignore[union-attr]


@pytest.fixture
async def remote(
    tmp_path: Path, blobs: LocalBlobStore, events: EventWriter
) -> AsyncIterator[Remote]:
    relay = Relay()
    config = ToolsetServerConfig(blobs=str(blobs.root))
    server_tokens = Tokens({"service": service_key(SECRET)})  # all a toolset server holds
    async with httpx.AsyncClient(trust_env=False) as http:
        server = ToolsetServer(relay, config, tokens=server_tokens, blobs=blobs, http=http)
        async with server, serving(server.app(), tmp_path) as url:
            provider = RemoteProvider(Relay, f"{url}/mcp")
            mock = LocalProvider(Mock(), {"act_seconds": 0.01})
            service = HubService(
                Hub(
                    [mock, provider], blobs=blobs, phases=dev_phases([provider.name]), events=events
                ),
                Tokens.from_secret(SECRET),
            )
            provider.remote_call = service.remote_call
            ready = tmp_path / "hub.ready.json"
            serving_hub = functools.partial(
                serve_hub,
                service,
                host="127.0.0.1",
                control_port=0,
                control_token=control_token(SECRET),
                mcp=functools.partial(hub_app, host="127.0.0.1"),
                ready_file=ready,
            )
            async with service, running(serving_hub(), ready):
                yield Remote(service, relay, server, f"{url}/mcp")


async def test_a_remote_toolset_runs_like_a_local_one(remote: Remote, artifacts_dir: Path) -> None:
    a, b = await remote.open(), await remote.open()
    first = await call(remote.service, a, "relay")
    assert text(first) == f'relay: frame="frame_1", calls=1, episode="{a}", stage="dev"'
    assert "calls=2" in text(await call(remote.service, a, "relay"))
    assert "calls=1" in text(await call(remote.service, b, "relay"))  # state is per episode
    events = list(read_events(artifacts_dir / "events.jsonl"))
    relays = [
        e.data["call_id"] for e in events if e.kind == "tool.call" and e.data["name"] == "relay"
    ]
    parents = [
        e.data["parent"]
        for e in events
        if e.kind == "tool.call" and e.data["origin"] == "service" and e.data["name"] == "look"
    ]
    assert parents == relays  # each relay's look ran in the Hub on its behalf


async def test_the_server_takes_only_service_tokens_of_the_context_s_episode(
    remote: Remote,
) -> None:
    remote.service.mcp_url = "http://127.0.0.1:1/mcp"  # so that episodes get episode tokens
    session = await remote.service.open_episode(OpenEpisode(stage="dev"))
    other = await remote.open()
    assert session.access is not None
    call_look = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "relay"}}
    headers = {
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": "2026-07-28",
        "Authorization": f"Bearer {session.access.token.get_secret_value()}",
    }
    async with httpx.AsyncClient(trust_env=False) as http:
        response = await http.post(remote.url, json=call_look, headers=headers)
    assert response.status_code == 401  # an episode token opens no toolset server
    context = {"episode_ref": other, "stage": "dev", "tier": "D"}
    frames = {"frames": [], "facts": {}}
    with remote.service.remote_call(session.episode_id, "c1") as own:
        async with connect(remote.url, token=own.token) as client:
            crossed = await client.call_tool("relay", {}, meta={CONTEXT: context, FRAMES: frames})
            staged = {**context, "episode_ref": session.episode_id, "stage": "s1"}
            restaged = await client.call_tool("relay", {}, meta={CONTEXT: staged, FRAMES: frames})
            bare = await client.call_tool("relay", {})
    assert crossed.is_error and crossed.meta[PROBLEM]["code"] == "unauthorized"
    assert restaged.is_error and restaged.meta[PROBLEM]["code"] == "unauthorized"
    assert bare.is_error and bare.meta[PROBLEM]["code"] == "invalid_request"


async def test_an_unreachable_toolset_server_is_an_infrastructure_error(
    blobs: LocalBlobStore,
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    provider = RemoteProvider(Relay, f"http://127.0.0.1:{port}/mcp")
    service = HubService(
        Hub([provider], blobs=blobs, phases=dev_phases([provider.name])), Tokens.from_secret(SECRET)
    )
    provider.remote_call = service.remote_call
    async with service:
        episode = (await service.open_episode(OpenEpisode(stage="dev"))).episode_id
        outcome = await call(service, episode, "relay")
        await service.close_episode(episode)  # a server that cannot hear it keeps nothing open
    assert outcome.error_code == "infra_error"
