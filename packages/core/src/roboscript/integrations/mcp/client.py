"""MCP clients of our servers over Streamable HTTP: the Hub's MCP endpoint (tests, conformance,
adapters) and remote toolset servers (`RemoteProvider`)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr

Era = Literal["2026-07-28", "legacy"]
"""The protocol generation: the 2026-07-28 per-request envelope, or the initialize handshake."""


@asynccontextmanager
async def connect(
    url: str,
    *,
    token: SecretStr | None = None,
    auth: httpx2.Auth | None = None,
    era: Era = "2026-07-28",
    read_timeout_s: float | None = 300.0,
) -> AsyncIterator[Client]:
    """A connected client of the MCP endpoint at `url`, with `token` as bearer (or `auth`
    signing each request). `read_timeout_s` bounds the wait for each piece of an answer; None
    leaves a call's deadline to the caller. Proxy variables are ignored: our servers are local.
    The client caches nothing."""
    headers = {} if token is None else {"Authorization": f"Bearer {token.get_secret_value()}"}
    timeout = httpx2.Timeout(30.0, read=read_timeout_s)
    async with httpx2.AsyncClient(
        headers=headers, auth=auth, timeout=timeout, trust_env=False
    ) as http:
        transport = streamable_http_client(url, http_client=http)
        async with Client(
            transport, mode="legacy" if era == "legacy" else era, cache=None
        ) as client:
            yield client
