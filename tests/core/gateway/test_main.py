"""`python -m roboscript.gateway` as a service of the stack supervisor, before fake upstreams."""

import sys
from importlib import metadata
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from roboscript.gateway import openai_client
from roboscript.records import read_events
from roboscript.stack import Interpreters, ServiceSpec, StackConfig, up
from roboscript.testing.gateway import FakeUpstream, serving


def _installed() -> bool:
    try:
        metadata.distribution("roboscript")
    except metadata.PackageNotFoundError:
        return False
    return True


# The routing policy is an entry point, which only an installed distribution registers.
pytestmark = pytest.mark.skipif(not _installed(), reason="roboscript is not installed")


async def test_the_gateway_serves_as_a_stack_service(tmp_path: Path) -> None:
    upstream = FakeUpstream({"ep1": False})
    log_file = tmp_path / "gateway.jsonl"
    config = {
        "schema": "rs-gateway/v1",
        "listen": "127.0.0.1:0",
        "endpoints": {
            "ep1": {"url_env": "EP1_URL", "key_env": "EP1_KEY", "protocols": ["responses"]}
        },
        "models": {"m": {"deployments": [{"endpoint": "ep1", "upstream": "m-up"}]}},
        "state_file": str(tmp_path / "state.json"),
        "log_file": str(log_file),
    }
    config_path = tmp_path / "gateway.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    token = "tok-main-test"
    async with serving(upstream.app, tmp_path) as url:
        env_api = tmp_path / "upstreams.env"
        env_api.write_text(f"EP1_URL={url}/ep1/v1\nEP1_KEY=sk-main-test-key\n", encoding="utf-8")
        gateway = ServiceSpec(
            name="gateway",
            module="roboscript.gateway",
            args=["--config", str(config_path)],
            env={"RS_ENV_API_FILE": str(env_api), "RS_GATEWAY_TOKEN": token},
            ready_timeout_s=60,
        )
        python = Path(sys.executable)
        stack = StackConfig(
            interpreters=Interpreters(agent=python, sim=python),
            run_dir=tmp_path / "run",
            services=[gateway],
        )
        async with up(stack) as running:
            base_url = f"{running.endpoints['gateway']}/v1"
            client = openai_client("s1", base_url=base_url, token=SecretStr(token))
            async with client:
                answer = await client.responses.create(model="m", input="hi")
    assert answer.output_text == "ok"
    assert upstream["ep1"].seen[0].body["model"] == "m-up"
    [event] = read_events(log_file)
    assert (event.service, event.episode_id, event.data["deployment"]) == ("gateway", "s1", "ep1")
    service_log = (tmp_path / "run" / "logs" / "gateway.log").read_text(encoding="utf-8")
    assert "1 endpoints, models ['m']" in service_log
    assert "sk-main-test-key" not in service_log and token not in service_log
