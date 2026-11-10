"""roboscript.settings: repository root, `.env` and machine-local YAML configs."""

import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from roboscript.settings import (
    EnvSettings,
    config_hash,
    load_config,
    path_variable,
)

_VARIABLES = (
    "RS_ROOT",
    "RS_ENV_FILE",
    "SIM_ROOT",
    "RS_RUNS_ROOT",
    "RS_GATEWAY_TOKEN",
    "RS_GATEWAY_ADMIN_TOKEN",
    "RS_HUB_SECRET",
    "RS_HUB_SERVICE_KEY",
    "RS_ENV_TOKEN",
    "RS_HOST_TOKEN",
    "RS_BLOB_DIR",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may see this machine's variables, nor the repository's real `.env`."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)


def write_env(path: Path, **values: str) -> Path:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return path


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str
    cooldown_s: int = 30
    token: SecretStr | None = None


def test_env_settings_read_the_repository_dotenv_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RS_ROOT", str(tmp_path))
    write_env(
        tmp_path / ".env",
        SIM_ROOT="/opt/sim",
        RS_GATEWAY_TOKEN="gw-token",
        RS_HUB_SECRET="hub-secret",
        SOME_OTHER_TOOL="ignored",
    )
    write_env(tmp_path / ".env.api", RS_RUNS_ROOT="/from/api", RS_GW_EP1_KEY="sk-1")
    settings = EnvSettings()
    assert path_variable("SIM_ROOT") == Path("/opt/sim")
    assert settings.runs_root is None  # .env.api is never read here
    assert settings.gateway_token is not None and settings.hub_secret is not None
    assert settings.gateway_token.get_secret_value() == "gw-token"
    assert settings.hub_secret.get_secret_value() == "hub-secret"
    assert "gw-token" not in repr(settings) and "hub-secret" not in repr(settings)


def test_load_config_reads_overrides_and_validates(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    (configs / "stacks").mkdir(parents=True)
    (configs / "gateway.yaml").write_text(
        "listen: 127.0.0.1:18790\ncooldown_s: 45\n", encoding="utf-8"
    )
    (configs / "stacks" / "dev-local.yaml").write_text("listen: 127.0.0.1:1\n", encoding="utf-8")
    config = load_config("gateway", GatewayConfig, root=tmp_path, overrides=["cooldown_s=60"])
    assert config == GatewayConfig(listen="127.0.0.1:18790", cooldown_s=60)
    assert load_config("stacks/dev-local", GatewayConfig, root=tmp_path).listen == "127.0.0.1:1"


def test_config_hash_is_stable_and_never_sees_secrets() -> None:
    first = GatewayConfig(listen="x", cooldown_s=1, token=SecretStr("secret-one"))
    second = GatewayConfig.model_validate({"token": "secret-two", "cooldown_s": 1, "listen": "x"})
    assert re.fullmatch(r"[0-9a-f]{64}", config_hash(first))
    assert config_hash(first) == config_hash(second)
    assert config_hash(first) != config_hash(GatewayConfig(listen="x", cooldown_s=2))
