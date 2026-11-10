"""The Hub's MCP endpoint in detail (docs/concepts/agent-runtime.md): images arrive as images,
lists carry private cache hints, and service tokens work only inside a running remote call."""

import base64
import functools
import io
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.hub import EpisodeSession, OpenEpisode
from roboscript.hub import Hub
from roboscript.hub.service import HubService, serve_hub
from roboscript.hub.steps import NOT_PLAIN
from roboscript.hub.tokens import Tokens, control_token
from roboscript.integrations.mcp import connect, hub_app
from roboscript.integrations.mcp.wire import PARENT, PROBLEM
from roboscript.records import EventWriter, read_events
from roboscript.testing.external import MOCK_TOOLS
from roboscript.testing.gateway import running
from roboscript.testing.mock import Mock, mock_image
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider

SECRET = SecretStr("m" * 40)


@dataclass
class McpHub:
    """A served Hub over the Mock toolset, permits off, with its MCP endpoint."""

    service: HubService
    url: str

    async def open(self, **options: object) -> EpisodeSession:
        return await self.service.open_episode(OpenEpisode(stage="s2", **options))  # type: ignore[arg-type]


@pytest.fixture
async def mcp_hub(
    tmp_path: Path, blobs: LocalBlobStore, events: EventWriter
) -> AsyncIterator[McpHub]:
    hub = Hub(
        [LocalProvider(Mock(), {"act_seconds": 0.05})],
        blobs=blobs,
        phases=TEST_PHASES,
        events=events,
    )
    service = HubService(hub, Tokens.from_secret(SECRET))
    ready = tmp_path / "hub.ready.json"
    serving = functools.partial(
        serve_hub,
        service,
        host="127.0.0.1",
        control_port=0,
        control_token=control_token(SECRET),
        mcp=functools.partial(hub_app, host="127.0.0.1"),
        ready_file=ready,
    )
    async with service, running(serving(), ready) as found:
        yield McpHub(service, found.urls["mcp"])


def token_of(session: EpisodeSession) -> SecretStr:
    assert session.access is not None
    return session.access.token


async def test_images_arrive_as_images_and_lists_are_private(mcp_hub: McpHub) -> None:
    session = await mcp_hub.open()
    async with connect(mcp_hub.url, token=token_of(session)) as client:
        listing = await client.list_tools()
        result = await client.call_tool("look", {})
    assert [tool.name for tool in listing.tools] == MOCK_TOOLS
    assert listing.cache_scope == "private" and 3_500_000 < listing.ttl_ms <= 3_600_000
    text, image = result.content
    assert text.type == "text" and text.text == 'look: frame="frame_1", image=<image 1>'
    assert image.type == "image" and image.mime_type == "image/png"
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(image.data))))
    np.testing.assert_array_equal(decoded, mock_image())
    assert result.structured_content is None and result.is_error is False


async def test_service_tokens_work_only_inside_a_running_remote_call(
    mcp_hub: McpHub, artifacts_dir: Path
) -> None:
    a, b = await mcp_hub.open(), await mcp_hub.open()
    service = mcp_hub.service
    with service.remote_call(a.episode_id, "call-9") as remote:
        async with connect(mcp_hub.url, token=remote.token) as client:
            inside = await client.call_tool("look", {}, meta={PARENT: remote.handle})
            hidden = await client.call_tool("hidden", {}, meta={PARENT: remote.handle})
            unhandled = await client.call_tool("look", {})
            guessed = await client.call_tool("look", {}, meta={PARENT: "rc-000000000000"})
            deep = await client.call_tool(
                "look",
                {"x": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]},
                meta={PARENT: remote.handle},
            )
        with service.remote_call(b.episode_id, "call-1") as other:  # B's handle, A's token
            async with connect(mcp_hub.url, token=remote.token) as client:
                crossed = await client.call_tool("look", {}, meta={PARENT: other.handle})
    async with connect(mcp_hub.url, token=remote.token) as client:
        after = await client.call_tool("look", {}, meta={PARENT: remote.handle})
    assert inside.structured_content["frame"].startswith("frame-")  # an id, not an alias
    assert hidden.structured_content == {"ok": True}  # any mounted tool, visible or not
    for refused in (unhandled, guessed, crossed, after):
        assert refused.is_error and refused.meta[PROBLEM]["code"] == "unauthorized"
    assert deep.is_error and deep.meta[PROBLEM]["detail"] == NOT_PLAIN
    status = await service.state(a.episode_id)
    assert status.counters.info_calls == status.counters.system_calls == 0  # no budget used
    calls = [
        e.data
        for e in read_events(artifacts_dir / "events.jsonl")
        if e.kind == "tool.call" and e.data["origin"] == "service"
    ]
    assert [(c["name"], c["parent"]) for c in calls] == [("look", "call-9"), ("hidden", "call-9")]
