"""A served Hub (docs/concepts/agent-runtime.md): the Hub plus what reaches it from other
processes.

`HubService` wraps an in-process `Hub` and implements `HubControl`, the runner's plane. Opening an
episode with permits on opens its permit ledger, makes the key its gateway session's requests must
carry (`EpisodeSession.session_key`) and registers the session with the gateway, which from then
on posts permits for the tool calls its models propose to the service's `PermitIntake`; with an
MCP endpoint, it also issues the brain's episode token (`EpisodeSession.access`). Remote toolsets
get a service token per episode, valid for calls back through the Hub only while one of the
Hub's calls to them is running (`remote_call`). Closing the episode revokes its tokens, drops its
unused permits and unregisters the session. `serve_hub` puts the control plane and the permit
intake on one port and the MCP endpoint (built by `roboscript.integrations.mcp`) on another;
`python -m roboscript.integrations.hub_server` composes and serves one from
`configs/hub-server.yaml`.
"""

import functools
import hmac
import logging
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from anyio.abc import TaskStatus
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveFloat,
    SecretStr,
    field_validator,
    model_validator,
)
from starlette.applications import Starlette
from starlette.types import ASGIApp
from typing_extensions import Self

from roboscript.contracts.errors import ProtocolError, RSError
from roboscript.contracts.gateway import (
    GatewayAdmin,
    Permit,
    PermitIntake,
    PermitMode,
    SessionRegistration,
    description_hash,
    session_key_hash,
)
from roboscript.contracts.hub import (
    CallOutcome,
    EpisodeAccess,
    EpisodeSession,
    EpisodeStatus,
    EpisodeSummary,
    HubControl,
    OpenEpisode,
)
from roboscript.contracts.ids import EpisodeId, SessionId, ToolName, new_id
from roboscript.contracts.tools import ToolCall, ToolDescriptor
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.hub.config import HubConfig
from roboscript.hub.hub import Hub
from roboscript.hub.lock import catalog_lock, lock_hash
from roboscript.hub.permits import PermitLedger
from roboscript.hub.tokens import TokenClaims, Tokens, TokenType
from roboscript.rpc import make_app, routes, serve, write_ready
from roboscript.settings import EnvSettings

log = logging.getLogger(__name__)


class PermitsConfig(BaseModel):
    """Only model-proposed calls run: `gateway` is the model gateway's base URL, whose admin plane
    (bearer `RS_GATEWAY_ADMIN_TOKEN`) takes the episodes' registrations within `timeout_s`;
    `mode` is how calls match permits unless an episode says otherwise (`OpenEpisode.permits`);
    a permit not used within `ttl_s` of its arrival lapses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gateway: str
    mode: PermitMode = "strict"
    timeout_s: PositiveFloat = 10.0
    ttl_s: PositiveFloat = 600.0


class ToolsetMount(BaseModel):
    """A toolset to mount by its name in the `roboscript.toolsets` entry-point group: in this
    process with `config`, or served by a remote toolset server whose MCP endpoint is `url`
    (`python -m roboscript.integrations.mcp.serve`, which holds the toolset's config); with the
    skillsets named in `skillsets` (`tools.skillsets`; a remote toolset's server names the same:
    the Hub reads their tools' specs here)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    url: str | None = None
    skillsets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _config_is_local(self) -> Self:
        if self.url is not None and self.config:
            raise ValueError(f"toolset {self.name}: a remote toolset's config is its server's")
        return self


class GuardItem(BaseModel):
    """A guard to add to the built-in pipeline, by its name in `roboscript.guards`, with
    `overrides` of its config: the shape of a run's guard plugin (`learning.runspec.PluginItem`
    with group `guards`), so a run's guard items fit here unchanged. Guards run in the Hub's own
    process: no `url`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group: Literal["guards"] = "guards"
    name: str
    overrides: dict[str, JsonValue] = {}
    url: str | None = None

    @field_validator("url")
    @classmethod
    def _in_process(cls, url: str | None) -> str | None:
        if url is not None:
            raise ValueError("guards run in the Hub's own process: no url")
        return url


def embodiment_profile(ref: str | None) -> EmbodimentProfile | None:
    """The profile a served Hub or toolset server mounts its toolsets for (`embodiment` in their
    configs): a name registered in `roboscript.embodiments` or a profile's YAML file, as a run's
    `embodiment` (`load_embodiment`); None: none, so no fit check (`tools.skillsets.mounted`)."""
    if ref is None:
        return None
    try:
        return load_embodiment(ref)
    except (LookupError, OSError, ImportError, yaml.YAMLError) as error:
        raise ValueError(f"embodiment {ref!r}: {type(error).__name__}: {error}") from None


class HubServerConfig(BaseModel):
    """`configs/hub-server.yaml`: where a served Hub listens (port 0 picks a free one; no
    `mcp_port`, no MCP endpoint), the method whose phases its episodes have (`roboscript.methods`
    name or `package.module:Class`), what it mounts, the robot it mounts it for (`embodiment`:
    each toolset's tools that do not fit it are left out, a named skillset that does not fit it
    refuses the mount, as a run's toolsets are mounted, `learning.mounts`; None: no fit check),
    the guards it adds to the built-in pipeline (composed as a run composes its guard plugins),
    the blob store it shares with its toolsets (a directory or URL; unset, `RS_BLOB_DIR`), its
    event log, how long its tokens live, and permits (None: off)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    control_port: int = Field(default=0, ge=0, le=65535)
    mcp_port: int | None = Field(default=0, ge=0, le=65535)
    method: str
    toolsets: list[ToolsetMount] = Field(min_length=1)
    embodiment: str | None = None
    guards: list[GuardItem] = Field(default_factory=list)
    blobs: str | None = None
    events: Path = Path("logs/hub.jsonl")
    hub: HubConfig = HubConfig()
    token_ttl_s: PositiveFloat = 3600.0
    permits: PermitsConfig | None = None
    catalog_lock: Path | None = None

    @field_validator("embodiment")
    @classmethod
    def _loads(cls, ref: str | None) -> str | None:
        embodiment_profile(ref)
        return ref

    def profile(self) -> EmbodimentProfile | None:
        """The profile `embodiment` names (None: none)."""
        return embodiment_profile(self.embodiment)

    @field_validator("guards")
    @classmethod
    def _once_each(cls, guards: list[GuardItem]) -> list[GuardItem]:
        names = [guard.name for guard in guards]
        if twice := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"guards named twice: {twice}")
        return guards

    def shared_blobs(self) -> str:
        """`blobs`, else `RS_BLOB_DIR` (from the environment or `.env`)."""
        blobs = self.blobs or EnvSettings().blob_dir
        if blobs is None:
            raise ValueError("set blobs in the hub-server config, or RS_BLOB_DIR (a stack sets it)")
        return str(blobs)


@dataclass(frozen=True)
class RemoteCall:
    """What a call to a remote toolset carries: the episode's service token, the Hub's MCP
    endpoint to call other toolsets back at, and `handle`, the call's id there."""

    token: SecretStr
    hub_url: str | None
    handle: str


@dataclass
class _Served:
    """What the service keeps per episode it opened: its session as opened, its tokens, its
    permit token and session key (with permits on), and its remote calls running now (handle
    -> the Hub's call id)."""

    session: EpisodeSession
    adopted: bool = False  # opened on the Hub by the process driving it, not here
    tokens: list[TokenClaims] = field(default_factory=list)
    permit_token: SecretStr | None = None
    session_key: SecretStr | None = None
    service: tuple[SecretStr, TokenClaims] | None = None
    remote: dict[str, str] = field(default_factory=dict)


class HubService:
    """`HubControl` over `hub`, plus the permit intake and what the MCP endpoint and remote
    toolsets need.

    With permits on, `ledger` must be the one whose guard is in the Hub's pipeline and
    `gateway` the gateway's admin plane. `mcp_url` and `permit_url` are set by `serve_hub`
    once their listeners are up: episodes get tokens only with an MCP endpoint, and permits
    need the intake. `async with` starts and stops the Hub.

    `adopt`: a process that drives the Hub itself (the Agent host, the learning runner) opens
    and closes episodes on the Hub, not here; with `adopt` on, such an episode gets its service
    token at its first remote call, for remote toolsets only (`roboscript.integrations.mcp.
    remote.serve_remote`), and is forgotten once the Hub has closed it and that token expired.
    """

    def __init__(
        self,
        hub: Hub,
        tokens: Tokens,
        *,
        ledger: PermitLedger | None = None,
        gateway: GatewayAdmin | None = None,
        permit_mode: PermitMode = "strict",
        token_ttl_s: float = 3600.0,
        adopt: bool = False,
    ) -> None:
        if (ledger is None) != (gateway is None):
            raise ValueError("permits need both the ledger in the Hub's pipeline and the gateway")
        if ledger is not None:
            if ledger.guard not in hub.pipeline.pre:
                raise ValueError("the Hub's pipeline lacks the ledger's guard: use with_permits")
            _check_descriptions(hub)
        self.hub = hub
        self.tokens = tokens
        self.ledger = ledger
        self.gateway = gateway
        self.permit_mode = permit_mode
        self.token_ttl_s = token_ttl_s
        self.adopt = adopt
        self.catalog = lock_hash(catalog_lock(hub.mounted, hub.phases.specs))
        self.mcp_url: str | None = None
        self.permit_url: str | None = None
        self._served: dict[EpisodeId, _Served] = {}

    async def __aenter__(self) -> Self:
        await self.hub.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.hub.__aexit__(*exc_info)

    # HubControl

    async def open_episode(self, spec: OpenEpisode) -> EpisodeSession:
        """Open the episode in the Hub, then register it for permits and issue its episode
        token; if either fails, the episode is closed again."""
        if spec.permits is not None and self.ledger is None:
            raise ProtocolError("invalid_request", "permits are off in this Hub")
        wall_clock = spec.budgets.wall_clock_s
        tokened = self.ledger is not None or self.mcp_url is not None
        if tokened and wall_clock is not None and wall_clock > self.token_ttl_s:
            detail = f"a {wall_clock:g} s wall clock outlives the {self.token_ttl_s:g} s tokens"
            raise ProtocolError("invalid_request", detail)
        session = await self.hub.open_episode(spec)
        served = self._served[session.episode_id] = _Served(session)
        try:
            access = None if self.mcp_url is None else self._access(served)
            if self.ledger is not None:
                await self._register(served, spec.permits or self.permit_mode)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._forget(session.episode_id)
            raise
        return session.model_copy(update={"access": access, "session_key": served.session_key})

    async def call(self, episode_id: EpisodeId, call: ToolCall) -> CallOutcome:
        return await self.hub.call(episode_id, call)

    async def system_call(
        self, episode_id: EpisodeId, name: ToolName, arguments: dict[str, Any]
    ) -> CallOutcome:
        return await self.hub.system_call(episode_id, name, arguments)

    async def tools(self, episode_id: EpisodeId) -> list[ToolDescriptor]:
        return await self.hub.tools(episode_id)

    async def end_episode(
        self, episode_id: EpisodeId, reason: str, note: str = ""
    ) -> EpisodeStatus:
        return await self.hub.end_episode(episode_id, reason, note)

    async def credit_clock(self, episode_id: EpisodeId, seconds: float, reason: str) -> None:
        await self.hub.credit_clock(episode_id, seconds, reason)

    async def close_episode(self, episode_id: EpisodeId) -> EpisodeSummary:
        """Close the episode in the Hub (its remote toolsets free its state while its service
        token still works), then revoke its tokens, drop its unused permits and unregister its
        session. The gateway being unreachable does not keep an episode open: its registration
        then lapses at its expiry, with a warning here."""
        summary = await self.hub.close_episode(episode_id)
        await self._forget(episode_id, closed=True)
        return summary

    async def state(self, episode_id: EpisodeId) -> EpisodeStatus:
        return await self.hub.state(episode_id)

    # PermitIntake

    async def issue(self, permits: list[Permit]) -> None:
        if self.ledger is None:
            raise ProtocolError("not_found", "this Hub takes no permits")
        self.ledger.add(permits)

    def allow_permits(self, bearer: SecretStr | None, method: str, args: dict[str, Any]) -> bool:
        """The intake's `allow`: the bearer must be the token of the permits' episode."""
        permits: list[Permit] = args["permits"]
        served = self._served.get(permits[0].episode_id) if permits else None
        return _same(bearer, None if served is None else served.permit_token)

    # For the MCP endpoint and remote toolsets

    def authenticate(self, token: str) -> TokenClaims:
        """The claims of a token for the MCP endpoint, naming an episode this Hub has open
        (ended ones included) exactly as it was opened, from this catalog; else
        `ProtocolError("unauthorized")`."""
        claims = self.tokens.verify(token, ("episode", "service"))
        served = self._served.get(claims.sub) or self._adopted(claims.sub)
        if served is None or self.hub.store.find(claims.sub) is None:
            raise ProtocolError("unauthorized", "the token's episode is not open in this Hub")
        session = served.session
        opened = (session.stage, session.tier, session.role, [t.name for t in session.tools])
        if (claims.stage, claims.tier, claims.role, claims.tools) != opened:
            raise ProtocolError("unauthorized", "the token does not match its episode")
        if claims.catalog != self.catalog:
            raise ProtocolError("unauthorized", "the token is for another catalog")
        return claims

    @contextmanager
    def remote_call(self, episode_id: EpisodeId, call_id: str) -> Iterator[RemoteCall]:
        """For the time of one call to a remote toolset: the episode's service token (issued
        once, renewed near expiry) and a fresh handle under which that toolset may call others
        back through the Hub (`in_flight`)."""
        served = self._served.get(episode_id) or self._adopted(episode_id)
        if served is None:
            raise ProtocolError("not_found", f"episode {episode_id} was not opened here")
        if served.service is None or served.service[1].exp - self.tokens.clock() < 60:
            served.service = self._issue("service", served)
        handle = new_id("rc")
        served.remote[handle] = call_id
        try:
            yield RemoteCall(served.service[0], self.mcp_url, handle)
        finally:
            del served.remote[handle]

    def in_flight(self, episode_id: EpisodeId, handle: str) -> str | None:
        """The Hub's call id of the episode's remote call running under `handle`, if any."""
        served = self._served.get(episode_id)
        return None if served is None else served.remote.get(handle)

    # Internals

    def _adopted(self, episode_id: EpisodeId) -> _Served | None:
        """With `adopt` on, an episode open in the Hub as the process driving it opened it, now
        known here (and adopted ones the Hub has closed, whose token has expired, forgotten)."""
        if not self.adopt:
            return None
        now = self.tokens.clock()
        gone = [
            closed
            for closed, served in self._served.items()
            if served.adopted
            and self.hub.store.find(closed) is None
            and (served.service is None or served.service[1].exp <= now)
        ]
        for closed in gone:
            for claims in self._served.pop(closed).tokens:
                self.tokens.revoke(claims)
        state = self.hub.store.find(episode_id)
        if state is None:
            return None
        session = EpisodeSession(
            episode_id=state.id,
            stage=state.stage,
            tier=state.tier,
            role=state.role,
            tools=[self.hub.mounted[name].descriptor for name in state.tools],
            system_tools=list(state.system_tools),
            budgets=state.budgets,
            rules=state.rules,
        )
        served = self._served[episode_id] = _Served(session, adopted=True)
        return served

    def _issue(self, typ: TokenType, served: _Served) -> tuple[SecretStr, TokenClaims]:
        session = served.session
        issued = self.tokens.issue(
            typ,
            episode=session.episode_id,
            stage=session.stage,
            tier=session.tier,
            role=session.role,
            tools=[tool.name for tool in session.tools],
            catalog=self.catalog,
            ttl_s=self.token_ttl_s,
        )
        served.tokens.append(issued[1])
        return issued

    def _access(self, served: _Served) -> EpisodeAccess:
        assert self.mcp_url is not None
        token, claims = self._issue("episode", served)
        return EpisodeAccess(mcp_url=self.mcp_url, token=token, expires_at=claims.exp)

    async def _register(self, served: _Served, mode: PermitMode) -> None:
        assert self.ledger is not None and self.gateway is not None
        if self.permit_url is None:
            raise RuntimeError("permits need the permit intake served first (serve_hub)")
        session = served.session
        self.ledger.open(session.episode_id, mode)
        served.permit_token = SecretStr(secrets.token_urlsafe(32))
        served.session_key = SecretStr(secrets.token_urlsafe(32))
        registration = SessionRegistration(
            session_id=SessionId(session.episode_id),
            permit_url=self.permit_url,
            permit_token=served.permit_token,
            expires_at=self.tokens.clock() + self.token_ttl_s,
            session_key_hash=session_key_hash(served.session_key),
            tools={description_hash(tool.description): tool.name for tool in session.tools},
        )
        await self.gateway.register(registration)

    async def _forget(self, episode_id: EpisodeId, closed: bool = False) -> None:
        """Revoke an episode's tokens, drop its permits and registration (and, unless the
        caller closed it already, close it)."""
        if not closed:
            await self.hub.close_episode(episode_id)
        served = self._served.pop(episode_id, None)
        if served is None:
            return
        for claims in served.tokens:
            self.tokens.revoke(claims)
        if self.ledger is None or self.gateway is None:
            return
        self.ledger.close(episode_id)
        if served.permit_token is not None:
            try:
                await self.gateway.unregister(SessionId(episode_id))
            except RSError as error:
                log.warning("could not unregister session %s at the gateway: %s", episode_id, error)


def _same(given: SecretStr | None, expected: SecretStr | None) -> bool:
    return (
        given is not None
        and expected is not None
        and hmac.compare_digest(
            given.get_secret_value().encode(), expected.get_secret_value().encode()
        )
    )


def _check_descriptions(hub: Hub) -> None:
    """Permits map called names to tools by description, so no two tools may share one."""
    seen: dict[str, str] = {}
    for tool in hub.mounted.values():
        other = seen.setdefault(description_hash(tool.descriptor.description), tool.qualified)
        if other != tool.qualified:
            raise ValueError(
                f"{other} and {tool.qualified} share a description; permits tell tools apart "
                "by it, so give each its own"
            )


def control_app(service: HubService, token: SecretStr) -> Starlette:
    """The control port: `hub.control` (bearer `token`) and `hub.permits` (per-episode tokens)."""
    control = routes(HubControl, service, token=token)
    intake = routes(PermitIntake, service, token=None, allow=service.allow_permits)
    return make_app(control, intake)


async def serve_hub(
    service: HubService,
    *,
    host: str,
    control_port: int,
    control_token: SecretStr,
    mcp: Callable[[HubService], ASGIApp] | None = None,
    mcp_port: int = 0,
    ready_file: Path | None = None,
    task_status: TaskStatus[dict[str, str]] = anyio.TASK_STATUS_IGNORED,
) -> None:
    """Serve `service` until SIGINT or SIGTERM: the control port, and the MCP endpoint that
    `mcp(service)` makes (its path `/mcp`) on its own port. Once both listen, the planes' URLs
    go to `task_status` and `ready_file`."""
    async with anyio.create_task_group() as tg:
        control = control_app(service, control_token)
        url = await tg.start(functools.partial(serve, control, host=host, port=control_port))
        service.permit_url = url
        planes = {"hub.control": url, "hub.permits": url}
        if mcp is not None:
            base = await tg.start(functools.partial(serve, mcp(service), host=host, port=mcp_port))
            service.mcp_url = planes["mcp"] = f"{base}/mcp"
        log.info("Hub planes: %s", planes)
        if ready_file is not None:
            write_ready(ready_file, url, planes)
        task_status.started(planes)
