"""The Hub's MCP endpoint (docs/concepts/agent-runtime.md): `/mcp`, MCP over Streamable HTTP for
both client generations (the 2026-07-28 per-request envelope, and the initialize handshake served
statelessly, without `Mcp-Session-Id`).

The bearer token decides everything; arguments never name an episode. `tools/list` shows the
token's episode its visible tools in the Hub's order, constant for the episode (`cacheScope:
private`, fresh until the token expires). An episode token's `tools/call`s are the model's and
pass the whole pipeline, permits included; every one reaches the Hub, unknown and hidden names
too, so its fixed errors and its event log apply. Results are text then images (`isError` for
errors, `_meta` saying whether the episode is over); structured content only for episodes that
opted in. A service token's calls are a remote toolset calling another toolset outside the
pipeline, accepted only while the Hub's call to that toolset runs (`_meta` names its handle) and
answered with the result as structured content. The runtime's system calls are not here: they
go through the control plane.
"""

import base64
import functools
from typing import Any

import mcp_types as types
from mcp.server import Server, ServerRequestContext
from starlette.applications import Starlette

import roboscript
from roboscript.blobs import BlobStore
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import CallOutcome
from roboscript.contracts.ids import ToolName, new_id
from roboscript.contracts.tools import TextPart, ToolCall, ToolDescriptor
from roboscript.hub.service import HubService
from roboscript.hub.steps import NOT_PLAIN, plain_json
from roboscript.hub.tokens import TokenClaims
from roboscript.integrations.mcp.wire import (
    CODE,
    ENDED,
    PARENT,
    claims_of,
    error_result,
    mcp_app,
    structured,
    traced,
)

SERVER_NAME = "rs"  # short: clients prefix tool names with it (`mcp__rs__observe`)


def hub_app(service: HubService, *, host: str = "127.0.0.1") -> Starlette:
    """The endpoint as an app (routes `/mcp`, `/healthz`, `/readyz`) serving `service`."""
    server: Server[Any] = Server(
        SERVER_NAME,
        version=roboscript.__version__,
        on_list_tools=functools.partial(_list_tools, service),
        on_call_tool=functools.partial(_call_tool, service),
    )
    return mcp_app(server, service.authenticate, host)


def _listing(descriptor: ToolDescriptor) -> types.Tool:
    return types.Tool(
        name=descriptor.name,
        description=descriptor.description,
        input_schema=descriptor.input_schema,
        output_schema=descriptor.output_schema,
    )


async def _list_tools(
    service: HubService,
    ctx: ServerRequestContext[Any, Any],
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    claims = claims_of(ctx)
    tools = [_listing(d) for d in await service.tools(claims.sub)]
    fresh_ms = max(0, int((claims.exp - service.tokens.clock()) * 1000))
    return types.ListToolsResult(tools=tools, ttl_ms=fresh_ms, cache_scope="private")


async def _call_tool(
    service: HubService, ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
) -> types.CallToolResult:
    claims = claims_of(ctx)
    name, arguments, meta = ToolName(params.name), params.arguments or {}, params.meta or {}
    with traced(meta):
        if claims.typ == "service":
            return await _service_call(service, claims, name, arguments, meta.get(PARENT))
        call = ToolCall(call_id=new_id("mcp"), name=name, arguments=arguments, origin="model")
        outcome = await service.call(claims.sub, call)
    return await _result(outcome, service.hub.blobs)


async def _service_call(
    service: HubService, claims: TokenClaims, name: str, arguments: dict[str, Any], handle: Any
) -> types.CallToolResult:
    """A remote toolset's call of another tool, on behalf of the Hub's call it is serving."""
    parent = service.in_flight(claims.sub, handle) if isinstance(handle, str) else None
    if parent is None:
        detail = "a service call must come from within a running call of this episode"
        return error_result(ProtocolError("unauthorized", detail))
    if not plain_json(arguments):
        return error_result(ProtocolError("invalid_request", NOT_PLAIN))
    try:
        return structured(await service.hub.service_call(claims.sub, name, arguments, parent))
    except Exception as error:
        return error_result(error)


async def _result(outcome: CallOutcome, blobs: BlobStore) -> types.CallToolResult:
    """An outcome as MCP content: text and base64 images, in order."""
    content: list[types.ContentBlock] = []
    for part in outcome.content:
        if isinstance(part, TextPart):
            content.append(types.TextContent(text=part.text))
        else:
            data = base64.b64encode(await blobs.aget_bytes(part.blob)).decode("ascii")
            content.append(types.ImageContent(data=data, mime_type=part.blob.media_type))
    meta: dict[str, Any] = {ENDED: outcome.episode_ended}
    if outcome.error_code is not None:
        meta[CODE] = outcome.error_code
    return types.CallToolResult(
        content=content,
        structured_content=outcome.structured,
        is_error=outcome.is_error,
        meta=meta,  # type: ignore[call-arg]  # the SDK's models take field names too
    )
