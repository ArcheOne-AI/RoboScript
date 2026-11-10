"""Compose and serve a Hub (docs/concepts/agent-runtime.md):

    python -m roboscript.integrations.hub_server --config <name or path> [--ready-file PATH]

From `configs/hub-server.yaml` (`HubServerConfig`): the toolsets, in this process or on remote
toolset servers (`RemoteProvider`), mounted for its `embodiment` as a run's are
(`learning.mounts.provider`: tools that do not fit the robot are left out); the guards added to
the built-in pipeline, composed as a run composes its guard plugins
(`roboscript.learning.compose.compose_run`); the blob store and event log; tokens with keys
derived from `RS_HUB_SECRET`; permits, registered on the gateway's admin plane with
`RS_GATEWAY_ADMIN_TOKEN`; the catalog lock to hold the tools to. The control plane and the permit
intake share one port; the MCP endpoint (unless `mcp_port` is null) has its own; the ready file
lists each plane's URL. It lives here, not in `roboscript.hub`, because it mounts the MCP
pieces, which only this package may import.
"""

import argparse
import functools
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
from anyio.abc import TaskStatus
from pydantic import SecretStr

from roboscript import rpc
from roboscript.blobs import BlobStore, open_store
from roboscript.contracts.gateway import GatewayAdmin
from roboscript.gateway import openai_client
from roboscript.hub import Hub, Pipeline
from roboscript.hub.lock import catalog_lock, check_lock
from roboscript.hub.permits import PermitLedger, with_permits
from roboscript.hub.service import HubServerConfig, HubService, serve_hub
from roboscript.hub.tokens import Tokens, control_token, hub_secret
from roboscript.integrations.mcp import RemoteProvider, hub_app
from roboscript.integrations.mcp.remote import check_server
from roboscript.learning import mounts
from roboscript.learning.compose import compose_run
from roboscript.learning.method import load_method
from roboscript.learning.runspec import PluginItem
from roboscript.records import EventWriter
from roboscript.settings import EnvSettings, load_config_ref
from roboscript.tools import ToolProvider

_ADMIN_TOKEN_LEN = 16


@asynccontextmanager
async def compose(
    config: HubServerConfig,
    *,
    tokens: Tokens,
    blobs: BlobStore,
    events: EventWriter | None,
    gateway: GatewayAdmin | None,
    models: Callable[..., Any] = openai_client,
) -> AsyncIterator[HubService]:
    """The service the config describes, started for the block: its toolsets mounted, its
    guards added to the built-in pipeline exactly as a run's guard plugins are (every guard and
    its config checked here, before the Hub starts), the permit guard right after `visibility`
    when permits are on, remote toolsets given their service tokens. Each remote toolset's
    server is checked before the Hub starts, as a run checks its own (`check_server`: it answers
    and lists exactly the tools this Hub mounts from it, so the two name the same skillsets and
    embodiment); a mismatch refuses the start with a `ValueError` saying what to fix."""
    items = [PluginItem.model_validate(item.model_dump()) for item in config.guards]
    providers = mount(config)
    async with compose_run(items) as composed:
        service = _service(
            config, providers, composed.pipeline, tokens, blobs, events, gateway, models
        )
        for provider in providers:
            if isinstance(provider, RemoteProvider):
                await check_server(provider, tokens)
        async with service:
            yield service


def _service(
    config: HubServerConfig,
    providers: list[ToolProvider],
    pipeline: Pipeline,
    tokens: Tokens,
    blobs: BlobStore,
    events: EventWriter | None,
    gateway: GatewayAdmin | None,
    models: Callable[..., Any],
) -> HubService:
    permits = config.permits
    ledger = None if permits is None else PermitLedger(events, ttl_s=permits.ttl_s)
    if ledger is not None:
        pipeline = with_permits(pipeline, ledger)
    hub = Hub(
        providers,
        blobs=blobs,
        phases=load_method(config.method).phases,
        config=config.hub,
        events=events,
        pipeline=pipeline,
        models=models,
    )
    if config.catalog_lock is not None:
        check_lock(config.catalog_lock, catalog_lock(hub.mounted, hub.phases.specs))
    service = HubService(
        hub,
        tokens,
        ledger=ledger,
        gateway=gateway,
        permit_mode="strict" if permits is None else permits.mode,
        token_ttl_s=config.token_ttl_s,
    )
    for provider in providers:
        if isinstance(provider, RemoteProvider):
            provider.remote_call = service.remote_call
    return service


def mount(config: HubServerConfig) -> list[ToolProvider]:
    """The providers of `config.toolsets`, in order, each with its skillsets, for the config's
    embodiment (`learning.mounts.provider`, as a run mounts its toolsets; no embodiment: no fit
    check)."""
    profile = config.profile()
    return [
        mounts.provider(item.name, item.config, item.skillsets, item.url, profile)
        for item in config.toolsets
    ]


async def run(
    config: HubServerConfig,
    *,
    ready_file: Path | None = None,
    secret: SecretStr | None = None,
    admin_token: SecretStr | None = None,
    tokens: Tokens | None = None,
    task_status: TaskStatus[dict[str, str]] = anyio.TASK_STATUS_IGNORED,
) -> None:
    """Serve the Hub `config` describes until SIGINT or SIGTERM (or cancellation). `secret`
    defaults to `RS_HUB_SECRET`, `admin_token` to `RS_GATEWAY_ADMIN_TOKEN` (needed with permits);
    `tokens` replaces the token issuer (tests move its clock). The planes' URLs go to
    `task_status` and `ready_file` once they listen."""
    secret = hub_secret() if secret is None else secret
    tokens = Tokens.from_secret(secret) if tokens is None else tokens
    async with AsyncExitStack() as stack:
        blobs = open_store(config.shared_blobs())
        stack.push_async_callback(blobs.aclose)
        events = stack.enter_context(EventWriter(config.events, "hub"))
        http = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
        gateway = None
        if config.permits is not None:
            gateway = rpc.client(
                GatewayAdmin,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
                config.permits.gateway,
                token=_admin_token(admin_token),
                http=http,
                timeout=config.permits.timeout_s,
            )
        service = await stack.enter_async_context(
            compose(config, tokens=tokens, blobs=blobs, events=events, gateway=gateway)
        )
        mcp = None if config.mcp_port is None else functools.partial(hub_app, host=config.host)
        await serve_hub(
            service,
            host=config.host,
            control_port=config.control_port,
            control_token=control_token(secret),
            mcp=mcp,
            mcp_port=config.mcp_port or 0,
            ready_file=ready_file,
            task_status=task_status,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m roboscript.integrations.hub_server", description="Serve a Tool Hub."
    )
    parser.add_argument("--config", default="hub-server", help="YAML path or name under configs/")
    parser.add_argument("--ready-file", type=Path, help="written with the planes' URLs")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config_ref(args.config, HubServerConfig)
    anyio.run(functools.partial(run, config, ready_file=args.ready_file))


def _admin_token(given: SecretStr | None) -> SecretStr:
    token = EnvSettings().gateway_admin_token if given is None else given
    if token is None or len(token.get_secret_value()) < _ADMIN_TOKEN_LEN:
        raise ValueError(
            f"permits need RS_GATEWAY_ADMIN_TOKEN in .env ({_ADMIN_TOKEN_LEN} or more characters, "
            "the same the gateway reads)"
        )
    return token


if __name__ == "__main__":
    main()
