"""roboscript.gateway.config: what an invalid YAML config says is wrong, and that errors in
`.env.api` name variables but no values."""

import re
from pathlib import Path
from typing import Any

import pytest

from roboscript.gateway.config import GatewayConfig, read_upstreams

A = {"url_env": "A_URL", "key_env": "A_KEY", "protocols": ["responses"], "stream_only": True}
B = {"url_env": "B_URL", "key_env": "B_KEY", "protocols": ["responses", "chat"]}
BASE: dict[str, Any] = {
    "schema": "rs-gateway/v1",
    "listen": "127.0.0.1:18790",
    "endpoints": {"a": A, "b": B},
    "models": {
        "m": {
            "deployments": [
                {"endpoint": "a", "upstream": "m-a"},
                {"endpoint": "b", "upstream": "m"},
            ]
        }
    },
    "aliases": {"brain": "m"},
}
UNCAPPABLE = {"endpoint": "a", "upstream": "x", "max_inflight": 0}  # a deployment with no slot
FLOORED = {"endpoint": "a", "upstream": "x", "max_inflight": 8, "min_inflight": 9}


def config(**changes: Any) -> GatewayConfig:
    return GatewayConfig.model_validate({**BASE, **changes})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"models": {"m": {"deployments": [{"endpoint": "c", "upstream": "x"}]}}},
            "names unknown endpoints ['c']",
        ),
        (
            {"models": {"m": {"deployments": [{"endpoint": "a", "upstream": "x"}] * 2}}},
            "lists an endpoint twice",
        ),
        ({"models": {"m": {"deployments": []}}}, "at least 1 item"),
        ({"aliases": {"brain": "n"}}, "alias 'brain' names unknown model 'n'"),
        ({"aliases": {"m": "m"}}, "alias 'm' is also a model"),
        ({"aliases": {"a brain": "m"}}, "names must be letters"),
        (
            {"endpoints": {"a": {**A, "protocols": ["responses", "chat"]}, "b": B}},
            "a stream_only endpoint speaks only responses",
        ),
        ({"endpoints": {"a": {**B, "protocols": ["chat", "chat"]}, "b": B}}, "name one twice"),
        ({"endpoints": {"a": {**B, "url_env": "A-URL"}, "b": B}}, "String should match pattern"),
        ({"listen": "18790"}, "listen must look like 127.0.0.1:18790"),
        ({"listen": "localhost:99999"}, "listen must look like 127.0.0.1:18790"),
        ({"schema": "rs-gateway/v2"}, "'rs-gateway/v1'"),
        ({"cooldown_s": 0}, "greater than 0"),
        ({"max_pins": 0}, "greater than 0"),
        ({"queue_timeout_s": -1}, "greater than or equal to 0"),
        ({"pin_wait_s": -1}, "greater than or equal to 0"),
        ({"stats_window_s": 0}, "greater than 0"),
        ({"models": {"m": {"deployments": [UNCAPPABLE]}}}, "greater than 0"),
        (
            {"models": {"m": {"deployments": [FLOORED]}}},
            "a: min_inflight 9 is above max_inflight 8",
        ),
        ({"aimd_recover_s": 0}, "greater than 0"),
        (
            {
                "models": {
                    "m": {
                        "deployments": [{"endpoint": "a", "upstream": "x"}],
                        "default_reasoning_effort": "Low",
                    }
                }
            },
            "String should match pattern",
        ),
        (
            {"models": {"m": {"deployments": [{"endpoint": "a", "upstream": "x", "tier": "b"}]}}},
            "'primary' or 'fallback'",
        ),
        ({"surprise": 1}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_configs_say_what_is_wrong(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        config(**changes)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("A_URL=https://a/v1\nA_KEY=sk-secret-1\nB_URL=https://b/v1\n", "set B_KEY"),
        ("A_URL=https://a/v1\nA_KEY=sk-secret-1\nB_URL=https://b/v1\nB_KEY=\n", "set B_KEY"),
        ("A_URL=https://a/v1\nA_KEY=sk-secret-1\nsk-secret-2\n", "set B_KEY, B_URL"),
        (
            "A_URL=relay-sk-secret-3/v1\nA_KEY=k\nB_URL=https://b/v1\nB_KEY=k\n",
            "A_URL must be an http(s) URL",
        ),
    ],
)
def test_env_api_errors_name_variables_but_no_values(
    tmp_path: Path, text: str, message: str
) -> None:
    env_api = tmp_path / ".env.api"
    env_api.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        read_upstreams(config(), env_api)
    assert "sk-secret" not in str(error.value)
