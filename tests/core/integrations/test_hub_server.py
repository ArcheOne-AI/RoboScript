"""The served Hub as processes (docs/concepts/agent-runtime.md): `python -m
roboscript.integrations.hub_server` with the Mock toolset on `python -m
roboscript.integrations.mcp.serve` (holding only the service key), permits on through a real
gateway; a brain reaches it over MCP, the runner over HubControl, and both processes stop on
SIGTERM."""

import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import anyio
import httpx
import pytest
import yaml
from pydantic import SecretStr

from roboscript import rpc
from roboscript.contracts.hub import HubControl, OpenEpisode
from roboscript.hub.tokens import control_token, service_key
from roboscript.integrations.mcp import connect
from roboscript.integrations.mcp.serve import ToolsetServerConfig
from roboscript.rpc import Ready, read_ready
from roboscript.settings import load_config_file
from roboscript.testing.external import MOCK_TOOLS, PREFIX, RATIONALE
from roboscript.testing.gateway import GatewayRig, ToolCall

SECRET = SecretStr("p" * 40)
CONFIGS = Path(__file__).parents[3] / "configs"


@contextmanager
def process(args: list[str], log: Path, env: dict[str, str]) -> Iterator[subprocess.Popen[bytes]]:
    """A service process, stopped with SIGTERM when the block ends (it must exit cleanly)."""
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", *args], stdout=out, stderr=subprocess.STDOUT, env=env
        )
        try:
            yield proc
        finally:
            proc.send_signal(signal.SIGTERM)
            code = proc.wait(timeout=20)
    assert code == 0, tail(log)


async def ready(path: Path, proc: subprocess.Popen[bytes], log: Path) -> Ready:
    with anyio.fail_after(60):
        while (found := read_ready(path)) is None:
            assert proc.poll() is None, tail(log)
            await anyio.sleep(0.05)
    return found


def tail(log: Path) -> str:
    return log.read_text(encoding="utf-8")[-3000:]


async def test_a_served_hub_with_a_remote_toolset(tmp_path: Path, artifacts_dir: Path) -> None:
    async with GatewayRig(tmp_path / "gateway", ["ep1"]) as rig:
        env = {
            **os.environ,
            "RS_HUB_SERVICE_KEY": service_key(SECRET).get_secret_value(),  # the toolset server's
        }
        hub_env = {
            **os.environ,
            "RS_HUB_SECRET": SECRET.get_secret_value(),
            "RS_GATEWAY_ADMIN_TOKEN": rig.admin_token.get_secret_value(),
        }
        blobs = str(tmp_path / "blobs")
        toolset_config = tmp_path / "toolset-server.yaml"
        toolset_config.write_text(
            yaml.safe_dump({"blobs": blobs, "toolset": {"act_seconds": 0.05}}), encoding="utf-8"
        )
        serve_args = ["roboscript.integrations.mcp.serve", "--toolset", "mock"]
        serve_args += ["--config", str(toolset_config), "--ready-file", str(tmp_path / "t.json")]
        toolset_log = artifacts_dir / "toolset.log"
        with process(serve_args, toolset_log, env) as toolset:
            toolset_url = (await ready(tmp_path / "t.json", toolset, toolset_log)).url
            hub_config = tmp_path / "hub-server.yaml"
            hub_config.write_text(
                yaml.safe_dump(
                    {
                        "method": "roboscript.testing.phases:TestMethod",
                        "toolsets": [{"name": "mock", "url": f"{toolset_url}/mcp"}],
                        "blobs": blobs,
                        "events": str(artifacts_dir / "hub.jsonl"),
                        "permits": {"gateway": rig.url},
                    }
                ),
                encoding="utf-8",
            )
            hub_args = ["roboscript.integrations.hub_server", "--config", str(hub_config)]
            hub_args += ["--ready-file", str(tmp_path / "h.json")]
            hub_log = artifacts_dir / "hub.log"
            with process(hub_args, hub_log, hub_env) as hub:
                planes = (await ready(tmp_path / "h.json", hub, hub_log)).urls
                async with httpx.AsyncClient(trust_env=False) as http:
                    control = rpc.client(
                        HubControl,  # type: ignore[type-abstract]
                        planes["hub.control"],
                        token=control_token(SECRET),
                        http=http,
                        timeout=60,
                    )
                    session = await control.open_episode(OpenEpisode(stage="s2"))
                    assert session.access is not None and session.access.mcp_url == planes["mcp"]
                    async with connect(planes["mcp"], token=session.access.token) as brain:
                        listing = await brain.list_tools()
                        assert [tool.name for tool in listing.tools] == MOCK_TOOLS
                        act = next(t for t in listing.tools if t.name == "act")
                        offered = [
                            {
                                "type": "function",
                                "name": PREFIX + "act",
                                "description": act.description,
                                "parameters": act.input_schema,
                            }
                        ]
                        rig.upstream["ep1"].reply(
                            ToolCall(PREFIX + "act", json.dumps({"rationale": RATIONALE}))
                        )
                        async with rig.openai(session.episode_id, session.session_key) as model:
                            await model.responses.create(model="m", input="act", tools=offered)
                        result = await brain.call_tool("act", {"rationale": RATIONALE})
                    assert not result.is_error, result
                    assert "action 1 of this episode" in result.content[0].text
                    await control.close_episode(session.episode_id)


def test_the_toolset_server_example_is_valid() -> None:
    config = load_config_file(CONFIGS / "toolset-server.example.yaml", ToolsetServerConfig)
    assert config.toolset == {"act_seconds": 2.0}
    with pytest.raises(ValueError):
        ToolsetServerConfig(blobs="b", port=70000)
