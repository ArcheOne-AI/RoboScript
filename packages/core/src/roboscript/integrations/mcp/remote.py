"""Mount a remote toolset server in the Hub (docs/concepts/agent-runtime.md): `RemoteProvider` is
a `ToolProvider` like `LocalProvider` whose calls go over MCP to
`python -m roboscript.integrations.mcp.serve`.

Its specs come from the toolset's class and the skillsets mounted on it (`tools.skillsets`: the
server names the same ones), so the Hub validates, aliases and renders exactly as for
a local toolset, while the server holds the toolset's config and resources. Each call is one MCP
connection with the episode's service token as bearer, carrying in `_meta` the `CallContext`
(tokens revealed: the server reaches the env with them), the handle under which the server may
call other toolsets back through the Hub while this call runs, the Hub's MCP endpoint, the trace,
and the episode's frames and facts; what the tool put into the frame store comes back and is
kept here (its facts all or none: if it changed the scene registry and the registry here changed
meanwhile, the call fails with a tool error that asks the model to call again;
`FrameStore.merge`), and the answer is the result model as structured content, or the toolset's
error, raised here again. A server that cannot be reached is an `InfraError`. When the Hub
closes an episode the server served, the server frees its state for it.

A process that drives its Hub itself (the Agent host, the learning runner) mounts remote
toolsets with `serve_remote`: it checks each server as the run starts (`check_server`: it
answers, and lists exactly the tools the run mounts from it), then serves the Hub's MCP endpoint
on loopback for their calls back and gives them service tokens of the episodes it opens
(`HubService(adopt=True)`).
"""

import functools
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from typing import Any, Final, cast

import anyio
import httpx2
import mcp_types as types
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, SecretStr

from roboscript.contracts.errors import InfraError
from roboscript.contracts.events import EventKind
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import REVEAL, EpisodeId, new_id
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.contracts.trace import get_trace
from roboscript.embodiments import EmbodimentProfile
from roboscript.hub import Hub
from roboscript.hub.service import HubService, RemoteCall
from roboscript.hub.tokens import Tokens, hub_secret
from roboscript.integrations.mcp.client import connect
from roboscript.integrations.mcp.hub import hub_app
from roboscript.integrations.mcp.wire import (
    CONTEXT,
    FRAMES,
    HUB,
    PARENT,
    TRACEPARENT,
    WRITES,
    from_structured,
)
from roboscript.kernel import NoConfig
from roboscript.rpc import serve
from roboscript.tools import ToolContext, ToolProvider, Toolset, ToolsetRuntime, ToolSpec
from roboscript.tools.skillsets import mounted

CHECK_TTL_S: Final = 60.0
"""How long the token `check_server` lists a server's tools with lives (revoked right after)."""
CHECK_TIMEOUT_S: Final = 30.0
"""How long `check_server` waits for a server to connect and list its tools: a hung server stops
a run's start in this time, not after the client's own (minutes-long) timeouts."""

RemoteCalls = Callable[[EpisodeId, str], AbstractContextManager[RemoteCall]]
"""(episode, the Hub's call id) -> what one remote call carries (`HubService.remote_call`)."""


class RemoteProvider:
    """`toolset` served at `url` (its MCP endpoint). `remote_call`, the source of service tokens
    and handles, is set by whoever serves the Hub (`HubService.remote_call`) before calls come."""

    def __init__(
        self,
        toolset: type[Toolset],
        url: str,
        skillsets: Sequence[str] = (),
        profile: EmbodimentProfile | None = None,
    ) -> None:
        self.name = toolset.name
        self.url = url
        self.config: BaseModel = NoConfig()
        self.remote_call: RemoteCalls | None = None
        self.skillsets = tuple(skillsets)
        found = mounted(toolset, skillsets, profile)
        self.unmounted = found.unmounted
        self._specs = {spec.name: spec for spec in found.specs}
        self._served: set[EpisodeId] = set()

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    async def start(self, runtime: ToolsetRuntime) -> None:
        """Nothing to start: the server holds the toolset."""

    async def stop(self) -> None:
        """Nothing to stop: each call has its own connection."""

    async def call(
        self, name: str, args: Mapping[str, Any] | BaseModel, ctx: ToolContext
    ) -> BaseModel:
        spec = self._specs[name]
        params = args if isinstance(args, spec.params) else spec.params.model_validate(args)
        arguments = params.model_dump(mode="json", exclude_computed_fields=True)
        base = ctx.frames.snapshot()
        with self._remote(ctx.episode_id, ctx.call_id) as remote:
            meta: dict[str, Any] = {
                CONTEXT: ctx.context.model_dump(mode="json", context=REVEAL),
                FRAMES: base.model_dump(mode="json"),
                PARENT: remote.handle,
            }
            if remote.hub_url is not None:
                meta[HUB] = remote.hub_url
            if (trace := get_trace()) is not None:
                meta[TRACEPARENT] = trace.to_traceparent()
            self._served.add(ctx.episode_id)
            with self._reachable():
                async with connect(self.url, token=remote.token, read_timeout_s=None) as client:
                    answer = await client.call_tool(
                        name, arguments, meta=cast(types.RequestParamsMeta, meta)
                    )
        result = from_structured(spec.result, answer)
        await _keep(ctx, (answer.meta or {}).get(WRITES) or [], base)
        return result

    async def close_episode(self, episode_id: EpisodeId) -> None:
        """Tell the server, if it served the episode, to free its state for it."""
        if episode_id not in self._served:
            return
        self._served.discard(episode_id)
        with self._remote(episode_id, "close") as remote, self._reachable():
            async with httpx2.AsyncClient(trust_env=False, timeout=30.0) as http:
                close = self.url.removesuffix("/mcp") + "/episode/close"
                headers = {"Authorization": f"Bearer {remote.token.get_secret_value()}"}
                (await http.post(close, headers=headers)).raise_for_status()

    def _remote(self, episode_id: EpisodeId, call_id: str) -> AbstractContextManager[RemoteCall]:
        if self.remote_call is None:
            raise RuntimeError(f"remote toolset {self.name}: serve the Hub before calling it")
        return self.remote_call(episode_id, call_id)

    @contextmanager
    def _reachable(self) -> Iterator[None]:
        """A failed connection or protocol exchange with the server becomes `InfraError`."""
        try:
            yield
        except Exception as error:
            if not _unreachable(error):
                raise
            detail = f"toolset server {self.name} at {self.url}: {_leaves_text(error)}"
            raise InfraError("not_ready", detail) from error


async def check_server(provider: RemoteProvider, tokens: Tokens) -> None:
    """A run's startup check of the server `provider` mounts: it answers at `provider.url` and
    lists the tools the run mounts from it, with the same descriptions and schemas, in the same
    order; else a `ValueError` saying what to fix. It lists them with a service token of no
    episode (it opens nothing: a call needs a context naming the token's episode)."""
    token, claims = tokens.issue(
        "service",
        episode=EpisodeId(new_id("check")),
        stage="check",
        tier="D",
        role="check",
        tools=[],
        catalog="",
        ttl_s=CHECK_TTL_S,
    )
    where = f"toolset server {provider.name} at {provider.url}"
    auth = _Bearer(token)
    try:
        with anyio.fail_after(CHECK_TIMEOUT_S):
            async with connect(provider.url, auth=auth) as client:
                listed = (await client.list_tools()).tools
    except TimeoutError as error:
        raise ValueError(
            f"{where}: did not list its tools within {CHECK_TIMEOUT_S:g} s: check that it is "
            "the toolset server and not hung, or fix the url"
        ) from error
    except Exception as error:
        if not _unreachable(error):
            raise
        raise ValueError(f"{where}: {_fix(provider, error, auth.refused)}") from error
    finally:
        tokens.revoke(claims)
    expected = {s.name: (s.description, s.input_schema, s.output_schema) for s in provider.specs()}
    served = {t.name: (t.description, t.input_schema, t.output_schema) for t in listed}
    if served == expected and list(served) == list(expected):
        return
    problems = []
    if missing := [name for name in expected if name not in served]:
        problems.append(f"it lacks {missing}")
    if extra := [name for name in served if name not in expected]:
        problems.append(f"it adds {extra}")
    if changed := [n for n in expected if n in served and served[n] != expected[n]]:
        problems.append(f"{changed} have another description or schema")
    if not problems:
        problems.append(f"it lists them as {list(served)}, not {list(expected)}")
    skillsets = list(provider.skillsets) or "none"
    raise ValueError(
        f"{where} does not serve the tools this run mounts as {provider.name} (skillsets "
        f"{skillsets}): {'; '.join(problems)}. Serve the same toolset with the same skillsets, "
        "from the same code, or fix the url"
    )


@asynccontextmanager
async def serve_remote(
    hub: Hub, providers: Sequence[ToolProvider], *, tokens: Tokens | None = None
) -> AsyncIterator[None]:
    """Mount the remote toolsets among `providers` (those of `hub`, started) for the block: each
    server checked (`check_server`), the Hub's MCP endpoint served on loopback for their calls
    back, their service tokens issued with `tokens` (default: from `RS_HUB_SECRET`, whose
    `service_key` the servers hold) for the episodes the caller opens on `hub`. Nothing happens
    without a remote toolset."""
    remote = [provider for provider in providers if isinstance(provider, RemoteProvider)]
    if not remote:
        yield
        return
    tokens = Tokens.from_secret(hub_secret()) if tokens is None else tokens
    for provider in remote:
        await check_server(provider, tokens)
    service = HubService(hub, tokens, adopt=True)
    host = "127.0.0.1"
    failure: Exception | None = None
    async with anyio.create_task_group() as group:
        endpoint = functools.partial(serve, hub_app(service, host=host), host=host)
        service.mcp_url = f"{await group.start(endpoint)}/mcp"
        for provider in remote:
            provider.remote_call = service.remote_call
        try:
            yield
        except Exception as error:  # raised below, outside the group, which would wrap it
            failure = error
        finally:
            for provider in remote:
                provider.remote_call = None
            group.cancel_scope.cancel()
    if failure is not None:
        raise failure


class _Bearer(httpx2.Auth):
    """The token as bearer, noting whether the server refused it (the MCP client reports a 401
    only as a generic error)."""

    def __init__(self, token: SecretStr) -> None:
        self.token = token
        self.refused = False

    def auth_flow(
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token.get_secret_value()}"
        response = yield request
        self.refused = self.refused or response.status_code == 401


def _fix(provider: RemoteProvider, error: BaseException, refused: bool) -> str:
    """What to fix when a server cannot be reached or refuses the run's token."""
    if refused:
        return (
            "it refuses this run's service token: give it RS_HUB_SERVICE_KEY = "
            "roboscript.hub.tokens.service_key(RS_HUB_SECRET) of this run's RS_HUB_SECRET"
        )
    return (
        f"cannot be reached ({_leaves_text(error)}): start it (python -m "
        f"roboscript.integrations.mcp.serve --toolset {provider.name} --config ...) or fix "
        "the url"
    )


async def _keep(ctx: ToolContext, writes: list[dict[str, Any]], base: EpisodeSeed) -> None:
    """Keep in the episode's frame store what the remote tool put into its copy, made from
    `base`: its frames, then its facts (the last of each), all or none (`FrameStore.merge`)."""
    facts: dict[str, dict[str, Any]] = {}
    for write in writes:
        if write.get("kind") == EventKind.ENV_OBSERVE:
            entry = FrameEntry.model_validate(write["data"])
            data = await ctx.blobs.aget_bytes(entry.observation)
            await ctx.frames.put(Observation.model_validate_json(data))
        elif write.get("kind") == EventKind.SCENE_FACT:
            facts[write["data"]["name"]] = write["data"]["value"]
    await ctx.frames.merge(facts, base)


def _leaves(error: BaseException) -> Iterator[BaseException]:
    """The errors an exception group holds, however nested (the MCP client's task groups)."""
    nested = getattr(error, "exceptions", None)
    if nested is None:
        yield error
        return
    for inner in nested:
        yield from _leaves(inner)


def _unreachable(error: BaseException) -> bool:
    """Whether the connection or the protocol failed, rather than code on this side."""
    return all(isinstance(leaf, (httpx2.HTTPError, MCPError, OSError)) for leaf in _leaves(error))


def _leaves_text(error: BaseException) -> str:
    return "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in _leaves(error))
