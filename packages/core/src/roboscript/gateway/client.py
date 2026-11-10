"""Model clients that go through the gateway (docs/concepts/agent-runtime.md)."""

from pathlib import Path

import openai
from pydantic import SecretStr

from roboscript.gateway.config import GatewayConfig
from roboscript.records import replay
from roboscript.records.replay.http import replay_client, tape_client
from roboscript.settings import EnvSettings, load_config


def openai_client(
    session_id: str,
    *,
    key: SecretStr | None = None,
    base_url: str | None = None,
    token: SecretStr | None = None,
    timeout: float = 900.0,
) -> openai.AsyncOpenAI:
    """An OpenAI client whose calls belong to `session_id` (API key `<token>:<session id>`, or
    `<token>:<session id>:<key>` with the session key a Hub with permits on hands out,
    `EpisodeSession.session_key`).

    `base_url` defaults to `gateway_url()`, `token` to `RS_GATEWAY_TOKEN` from `.env`. Retries
    are off because the gateway fails over; `timeout` covers a whole call, failovers included.
    """
    tape = replay.active()
    if tape is not None and tape.replays("model:"):  # answered from the tape: no gateway
        return replay_client(session_id, tape, timeout)  # type: ignore[no-any-return]
    if token is None:
        token = EnvSettings().gateway_token
        if token is None:
            raise ValueError("RS_GATEWAY_TOKEN is not set: add it to .env (see .env.example)")
    session = session_id if key is None else f"{session_id}:{key.get_secret_value()}"
    http = openai.DefaultAsyncHttpxClient(trust_env=False)  # the gateway is local
    return openai.AsyncOpenAI(
        api_key=f"{token.get_secret_value()}:{session}",
        base_url=gateway_url() if base_url is None else base_url,
        max_retries=0,
        timeout=timeout,
        http_client=http if tape is None else tape_client(http, tape),  # recorded
    )


def gateway_url(root: Path | None = None) -> str:
    """`RS_GATEWAY_URL` when set (a machine that calls another's gateway), else
    `http://<listen>/v1` from `configs/gateway.yaml` (under `root`, the repository's by default), a
    wildcard host replaced by loopback."""
    if (named := EnvSettings().gateway_url) is not None:
        return named.rstrip("/")
    host, port = load_config("gateway", GatewayConfig, root=root).address
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://[{host}]:{port}/v1" if ":" in host else f"http://{host}:{port}/v1"
