"""A served Hub without its MCP endpoint (docs/concepts/agent-runtime.md): `HubControl` over RPC,
the tokens an episode gets and loses, and the permit intake."""

import functools
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from roboscript import rpc
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import InfraError, ProtocolError
from roboscript.contracts.gateway import (
    Permit,
    PermitIntake,
    SessionRegistration,
    args_hash,
)
from roboscript.contracts.hub import HubControl, OpenEpisode
from roboscript.contracts.ids import EpisodeId, SessionId, ToolName
from roboscript.contracts.tools import ToolCall
from roboscript.hub import DEFAULT_PIPELINE, Hub
from roboscript.hub.permits import PermitLedger, with_permits
from roboscript.hub.service import HubService, serve_hub
from roboscript.hub.tokens import Tokens, control_token
from roboscript.records import EventWriter
from roboscript.testing.gateway import running
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider, ToolContext, Toolset, tool

SECRET = SecretStr("h" * 40)
CONFIGS = Path(__file__).parents[3] / "configs"
OBSERVE = ToolCall(call_id="c1", name=ToolName("look"), arguments={}, origin="model")


@dataclass
class Admin:
    """A gateway's admin plane in memory."""

    registered: dict[SessionId, SessionRegistration] = field(default_factory=dict)
    unregistered: list[SessionId] = field(default_factory=list)
    down: bool = False

    async def register(self, registration: SessionRegistration) -> None:
        if self.down:
            raise InfraError("not_ready", "the gateway is down")
        self.registered[registration.session_id] = registration

    async def unregister(self, session_id: SessionId) -> None:
        if self.down:
            raise InfraError("not_ready", "the gateway is down")
        self.unregistered.append(session_id)


@dataclass
class Served:
    service: HubService
    admin: Admin
    planes: dict[str, str]
    http: httpx.AsyncClient

    def control(self, token: SecretStr) -> HubControl:
        return rpc.client(
            HubControl, self.planes["hub.control"], token=token, http=self.http, timeout=30
        )  # type: ignore[type-abstract]

    def intake(self, token: SecretStr) -> PermitIntake:
        return rpc.client(
            PermitIntake, self.planes["hub.permits"], token=token, http=self.http, timeout=30
        )  # type: ignore[type-abstract]


def service_over(
    blobs: LocalBlobStore, events: EventWriter, admin: Admin, *providers: LocalProvider
) -> HubService:
    ledger = PermitLedger(events)
    mock = LocalProvider(Mock(), {"act_seconds": 0.01})
    hub = Hub(
        [mock, *providers],
        blobs=blobs,
        phases=TEST_PHASES,
        events=events,
        pipeline=with_permits(DEFAULT_PIPELINE, ledger),
    )
    tokens = Tokens.from_secret(SECRET)
    return HubService(hub, tokens, ledger=ledger, gateway=admin, token_ttl_s=600)


@pytest.fixture
async def served(
    tmp_path: Path, blobs: LocalBlobStore, events: EventWriter
) -> AsyncIterator[Served]:
    admin = Admin()
    service = service_over(blobs, events, admin)
    serving_hub = functools.partial(
        serve_hub,
        service,
        host="127.0.0.1",
        control_port=0,
        control_token=control_token(SECRET),
        ready_file=tmp_path / "hub.ready.json",
    )
    async with (
        service,
        httpx.AsyncClient(trust_env=False) as http,
        running(serving_hub(), tmp_path / "hub.ready.json") as ready,
    ):
        yield Served(service, admin, ready.urls, http)


def permit(episode: EpisodeId, arguments: dict[str, Any] | None = None) -> Permit:
    return Permit(
        episode_id=episode,
        tool=ToolName("look"),
        args_hash=args_hash(arguments or {}),
        call_id="call_1",
    )


async def test_tokens_name_their_episode_as_opened_and_die_with_it(served: Served) -> None:
    served.service.mcp_url = "http://127.0.0.1:1/mcp"  # tokens need an endpoint to be for
    session = await served.service.open_episode(OpenEpisode(stage="s2"))
    access = session.access
    assert access is not None and access.mcp_url == "http://127.0.0.1:1/mcp"
    claims = served.service.authenticate(access.token.get_secret_value())
    assert (claims.sub, claims.typ, claims.stage, claims.role) == (
        session.episode_id,
        "episode",
        "s2",
        "actor",
    )
    assert claims.tools == [t.name for t in session.tools]
    assert claims.catalog == served.service.catalog
    with served.service.remote_call(session.episode_id, "c7") as remote:
        assert served.service.in_flight(session.episode_id, remote.handle) == "c7"
        assert served.service.authenticate(remote.token.get_secret_value()).typ == "service"
        service_token = remote.token
    assert served.service.in_flight(session.episode_id, remote.handle) is None  # done
    with served.service.remote_call(session.episode_id, "c8") as again:
        assert again.token is service_token and again.handle != remote.handle
    await served.service.close_episode(session.episode_id)
    for token in (access.token, service_token):
        with pytest.raises(ProtocolError, match="revoked"):
            served.service.authenticate(token.get_secret_value())
    assert served.admin.unregistered == [session.episode_id]


async def test_the_intake_takes_permits_with_their_episode_s_token(served: Served) -> None:
    a = await served.service.open_episode(OpenEpisode(stage="s2"))
    b = await served.service.open_episode(OpenEpisode(stage="s2"))
    token_a = served.admin.registered[SessionId(a.episode_id)].permit_token
    await served.intake(token_a).issue([permit(a.episode_id)])
    assert not (await served.service.call(a.episode_id, OBSERVE)).is_error
    for permits in ([permit(b.episode_id)], [permit(EpisodeId("ep-unknown"))], []):
        with pytest.raises(ProtocolError) as caught:
            await served.intake(token_a).issue(permits)
        assert caught.value.code == "unauthorized"


class Done(BaseModel):
    ok: bool = True


class Twins(Toolset):
    """Two tools a model cannot tell apart by their descriptions."""

    name = "twins"

    @tool(kind="pure", tier="D")
    async def left(self, ctx: ToolContext) -> Done:
        """Do it."""
        return Done()

    @tool(kind="pure", tier="D")
    async def right(self, ctx: ToolContext) -> Done:
        """Do it."""
        return Done()


def test_permits_need_tools_told_apart_by_description(
    blobs: LocalBlobStore, events: EventWriter
) -> None:
    with pytest.raises(ValueError, match=r"twins\.left and twins\.right share a description"):
        service_over(blobs, events, Admin(), LocalProvider(Twins()))
