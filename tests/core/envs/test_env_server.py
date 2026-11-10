"""`serve_env` for real: MockEnv as a service process under the supervisor, checked over sockets."""

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import SecretStr

import roboscript
from roboscript.contracts.ids import TaskName
from roboscript.envs import control_token
from roboscript.envs.mock import MOCK
from roboscript.rpc import client
from roboscript.stack import Interpreters, ServiceSpec, StackConfig, up
from roboscript.testing import precheck
from roboscript.testing.env import EnvConformance, EnvTarget

TOKEN = "control-token-for-the-server-test"


class MockConformance(EnvConformance):
    traces_expert = True  # MockEnv's scripted expert


def env_file(directory: Path, **values: str) -> Path:
    path = directory / "test.env"
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    path.chmod(0o600)
    return path


async def test_a_served_mock_env_passes_conformance(artifacts_dir: Path) -> None:
    config = artifacts_dir / "mock.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "task": "stack_red_on_blue",
                "blob_dir": str(artifacts_dir / "blobs"),
                "step_delay_s": 0.02,
            }
        ),
        encoding="utf-8",
    )
    service = ServiceSpec(
        name="mock",
        module="roboscript.envs.mock",
        args=["--config", str(config)],
        env={"RS_ENV_FILE": str(env_file(artifacts_dir, RS_ENV_TOKEN=TOKEN))},
    )
    python = Path(sys.executable)
    stack = StackConfig(
        interpreters=Interpreters(agent=python, sim=python),
        sim_pythonpath=[Path(roboscript.__file__).parents[1]],
        run_dir=artifacts_dir,
        services=[service],
    )
    async with up(stack) as running:
        urls = running.urls["mock"]
        assert urls["env.control"] == urls["env.public"] == running.endpoints["mock"]
        assert urls["env.agent"] != urls["env.control"]

        def connect(interface: type[Any], url: str, token: SecretStr | None) -> Any:
            return client(interface, url, token=token, timeout=10)

        target = EnvTarget(
            url=urls["env.control"],
            token=SecretStr(TOKEN),
            connect=connect,
            profile=MOCK.profile(),
            task=TaskName("stack_red_on_blue"),
            seed=11,
        )
        results = await precheck(MockConformance(), target)
    assert results and all(failure is None for failure in results.values()), results
    assert TOKEN not in (artifacts_dir / "logs" / "mock.log").read_text(encoding="utf-8")


def test_the_control_token_comes_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RS_ENV_TOKEN", raising=False)
    monkeypatch.setenv("RS_ENV_FILE", str(env_file(tmp_path, RS_ENV_TOKEN=TOKEN)))
    assert control_token().get_secret_value() == TOKEN
    for weak in ("change-me", ""):
        monkeypatch.setenv("RS_ENV_FILE", str(env_file(tmp_path, RS_ENV_TOKEN=weak)))
        with pytest.raises(ValueError, match=r"set RS_ENV_TOKEN in \.env"):
            control_token()
