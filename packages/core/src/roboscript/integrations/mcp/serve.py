"""A remote toolset server (docs/concepts/agent-runtime.md): a toolset registered in the
`roboscript.toolsets` entry-point group, served over MCP for a Hub to mount (`RemoteProvider`).

    python -m roboscript.integrations.mcp.serve --toolset <name> --config <name or path>
        [--ready-file PATH]

It speaks MCP at `/mcp` (statelessly) and takes only the Hub's service tokens, verified with
`RS_HUB_SERVICE_KEY` (the one key it holds; it opens nothing else). Each call's `_meta` carries
its `CallContext`, which must name the token's episode, stage and tier; the handle of the Hub's
call; the Hub's MCP endpoint; and the episode's frames and scene facts. The tool sees a frame
store made from them; what it puts there (frames, facts) goes back with the answer for the Hub to
keep. The answer is the tool's result model as structured content; a toolset's exception travels
as a problem (`isError`) and is raised again in the Hub. A tool calling another toolset
(`ToolContext.call`) calls back through the Hub's endpoint with the same token and handle.
Per-episode toolset state lives here until the Hub closes the episode (`POST /episode/close`
with its service token) or `state_idle_s` passes without a call.
"""

import argparse
import functools
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import anyio
import httpx
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_core import to_jsonable_python
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from typing_extensions import Self

import roboscript
from roboscript import rpc
from roboscript.blobs import BlobStore, open_store
from roboscript.contracts.env import EnvAgent
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import CallContext, EnvBinding
from roboscript.embodiments import EmbodimentProfile
from roboscript.gateway import openai_client
from roboscript.hub.service import embodiment_profile
from roboscript.hub.tokens import TokenClaims, Tokens, TokenType, hub_service_key
from roboscript.integrations.mcp.client import connect
from roboscript.integrations.mcp.wire import (
    CONTEXT,
    FRAMES,
    HUB,
    PARENT,
    TRACEPARENT,
    WRITES,
    BearerAuth,
    bearer_of,
    claims_of,
    error_result,
    from_structured,
    mcp_app,
    structured,
    traced,
)
from roboscript.kernel import entrypoints
from roboscript.records import EventWriter
from roboscript.settings import EnvSettings, load_config_ref
from roboscript.tools import (
    FrameStore,
    LocalProvider,
    ToolContext,
    Toolset,
    ToolsetRuntime,
    ToolSpec,
)
from roboscript.tools.skillsets import GROUP as SKILLSET_GROUP
from roboscript.tools.skillsets import load_skillset

TOOLSETS: Final = "roboscript.toolsets"
_SERVICE: Final[tuple[TokenType, ...]] = ("service",)  # the only tokens a toolset server takes


class ToolsetServerConfig(BaseModel):
    """`configs/toolset-server.yaml`: where the server listens (port 0 picks a free one), the
    blob store it shares with the Hub (a directory or URL; unset, `RS_BLOB_DIR`), its event log,
    the timeout of each request to an episode's env, how long an episode's state outlives its
    last call, the toolset's own config (`toolset`), the skillsets mounted on it (`skillsets`,
    `tools.skillsets`) and the robot it mounts for (`embodiment`: its tools that do not fit it
    left out; None: no fit check). A Hub mounting this server names the same skillsets and
    embodiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=0, ge=0, le=65535)
    blobs: str | None = None
    events: Path | None = None
    env_timeout_s: PositiveFloat = 300.0
    state_idle_s: PositiveFloat = 3600.0
    toolset: dict[str, Any] = Field(default_factory=dict)
    skillsets: list[str] = Field(default_factory=list)
    embodiment: str | None = None

    @field_validator("embodiment")
    @classmethod
    def _loads(cls, ref: str | None) -> str | None:
        embodiment_profile(ref)
        return ref

    def profile(self) -> EmbodimentProfile | None:
        """The profile `embodiment` names (None: none)."""
        return embodiment_profile(self.embodiment)

    def shared_blobs(self) -> str:
        """`blobs`, else `RS_BLOB_DIR` (from the environment or `.env`)."""
        blobs = self.blobs or EnvSettings().blob_dir
        if blobs is None:
            raise ValueError("set blobs in the toolset-server config, or RS_BLOB_DIR")
        return str(blobs)


class ToolsetServer:
    """`toolset` behind MCP; `async with` starts and stops it. `tokens` needs only the service
    key; `models` is the model-client factory the toolset gets (the gateway's by default);
    `clock` (monotonic seconds) times idle state."""

    def __init__(
        self,
        toolset: Toolset,
        config: ToolsetServerConfig,
        *,
        tokens: Tokens,
        blobs: BlobStore,
        http: httpx.AsyncClient,
        events: EventWriter | None = None,
        models: Callable[..., Any] = openai_client,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = toolset.name
        self.config = config
        self.tokens = tokens
        self.blobs = blobs
        self.http = http
        self.events = events
        self.provider = LocalProvider(toolset, config.toolset, config.skillsets, config.profile())
        self.models = models
        self._clock = clock
        self._states: dict[EpisodeId, tuple[dict[str, Any], float]] = {}
        self._callees = {spec.name: spec for spec in self.provider.specs()}  # its own first

    async def __aenter__(self) -> Self:
        runtime = ToolsetRuntime(
            config=self.provider.config,
            blobs=self.blobs,
            env_client=self._env_client,
            models=self.models,
            call=lambda ctx, name, args: ctx.service_call(ctx, name, args),
            events=self.events,
        )
        await self.provider.start(runtime)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.provider.stop()

    def app(self) -> Starlette:
        """The MCP endpoint `/mcp`, `POST /episode/close`, `/healthz` and `/readyz`."""
        server: Server[Any] = Server(
            f"rs-{self.name}",
            version=roboscript.__version__,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )
        close = Route("/episode/close", BearerAuth(self._close, self._verify), methods=["POST"])
        return mcp_app(server, self._verify, self.config.host, [close])

    def _verify(self, token: str) -> TokenClaims:
        return self.tokens.verify(token, _SERVICE)

    async def _close(self, scope: Any, receive: Any, send: Any) -> None:
        """The Hub closed the token's episode: drop its state, tell the toolset, and refuse the
        token from now on (the Hub revokes it too, but this server does not hear of that)."""
        claims: TokenClaims = Request(scope).state.claims
        self._states.pop(claims.sub, None)
        self.tokens.revoke(claims)
        await self.provider.close_episode(claims.sub)
        await Response(status_code=204)(scope, receive, send)

    async def _list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        tools = [
            types.Tool(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
                output_schema=spec.output_schema,
            )
            for spec in self.provider.specs()
        ]
        return types.ListToolsResult(tools=tools)

    async def _call_tool(
        self, ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        claims, meta = claims_of(ctx), dict(params.meta or {})
        try:
            context = CallContext.model_validate(meta[CONTEXT])
            seed = EpisodeSeed.model_validate(meta[FRAMES])
        except (KeyError, ValidationError):
            detail = f"a toolset call carries {CONTEXT!r} and {FRAMES!r} in _meta"
            return error_result(ProtocolError("invalid_request", detail))
        if (context.episode_ref, context.stage, context.tier) != (
            claims.sub,
            claims.stage,
            claims.tier,
        ):
            return error_result(ProtocolError("unauthorized", "the token is for another episode"))
        if params.name not in {spec.name for spec in self.provider.specs()}:
            return error_result(
                ProtocolError("not_found", f"{self.name} has no tool {params.name!r}")
            )
        writes: list[dict[str, Any]] = []
        hub = meta.get(HUB)
        tool_ctx = ToolContext(
            context=context,
            toolset=self.name,
            call_id=str(meta.get(PARENT, "")),
            env_client=None if context.env is None else self._env_client(context.env),
            blobs=self.blobs,
            state=self._state(context.episode_ref),
            frames=FrameStore(
                self.blobs,
                alias=str,  # the Hub gives a frame put here its alias when it keeps it
                emit=lambda kind, data: writes.append({"kind": kind, "data": data}),
                seed=seed,
            ),
            service_call=functools.partial(
                self._call_back, bearer_of(ctx), hub if isinstance(hub, str) else None
            ),
            event_sink=self._event,
        )
        with traced(meta):
            try:
                result = await self.provider.call(params.name, params.arguments or {}, tool_ctx)
            except Exception as error:
                return error_result(error)
        return structured(result, {WRITES: writes})

    def _state(self, episode_id: EpisodeId) -> dict[str, Any]:
        now, idle = self._clock(), self.config.state_idle_s
        self._states = {e: entry for e, entry in self._states.items() if now - entry[1] < idle}
        state = self._states.get(episode_id, ({}, now))[0]
        self._states[episode_id] = (state, now)
        return state

    async def _call_back(
        self,
        token: SecretStr,
        hub: str | None,
        parent: ToolContext,
        name: str,
        args: Mapping[str, Any],
    ) -> BaseModel:
        """`ToolContext.call` here: another toolset's tool, run by the Hub for the episode."""
        if hub is None:
            raise LookupError("the Hub sent no endpoint to call other toolsets through")
        spec = self._callee(name)
        meta: dict[str, Any] = {PARENT: parent.call_id}
        if parent.context.trace is not None:
            meta[TRACEPARENT] = parent.context.trace.to_traceparent()
        async with connect(hub, token=token, read_timeout_s=None) as client:
            result = await client.call_tool(
                name, to_jsonable_python(args), meta=cast(types.RequestParamsMeta, meta)
            )
        return from_structured(spec.result, result)

    def _callee(self, name: str) -> ToolSpec:
        """The spec of a tool of this server's toolset, or of any registered toolset or
        skillset, for its result model."""
        if name not in self._callees:
            for toolset in entrypoints.names(TOOLSETS):
                for spec in entrypoints.load(TOOLSETS, toolset).specs():
                    self._callees.setdefault(spec.name, spec)
            for skillset in entrypoints.names(SKILLSET_GROUP):
                for ext in load_skillset(skillset).extensions:
                    for spec in ext.specs():
                        self._callees.setdefault(spec.name, spec)
        if name not in self._callees:
            raise LookupError(f"no registered toolset has a tool {name!r}")
        return self._callees[name]

    def _env_client(self, binding: EnvBinding) -> EnvAgent:
        return rpc.client(
            EnvAgent,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            binding.url,
            token=binding.token,
            http=self.http,
            timeout=self.config.env_timeout_s,
        )

    def _event(self, ctx: ToolContext, kind: str, data: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.emit(f"{ctx.toolset}.{kind}", data, episode_id=ctx.episode_id)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m roboscript.integrations.mcp.serve",
        description="Serve a registered toolset over MCP for a Hub to mount.",
    )
    parser.add_argument("--toolset", required=True, help="a name in roboscript.toolsets")
    parser.add_argument("--config", required=True, help="YAML path or name under configs/")
    parser.add_argument("--ready-file", type=Path, help="written with the URL once listening")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_ref(args.config, ToolsetServerConfig)
    anyio.run(_serve, args.toolset, config, args.ready_file)


async def _serve(name: str, config: ToolsetServerConfig, ready_file: Path | None) -> None:
    tokens = Tokens({"service": hub_service_key()})
    blobs = open_store(config.shared_blobs())
    events = None if config.events is None else EventWriter(config.events, f"toolset.{name}")
    toolset = entrypoints.load(TOOLSETS, name)()
    try:
        async with (
            httpx.AsyncClient(trust_env=False) as http,
            ToolsetServer(
                toolset, config, tokens=tokens, blobs=blobs, http=http, events=events
            ) as server,
        ):
            await rpc.serve(server.app(), host=config.host, port=config.port, ready_file=ready_file)
    finally:
        await blobs.aclose()
        if events is not None:
            events.close()


if __name__ == "__main__":
    main()
