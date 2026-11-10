"""Gateway tests see neither this machine's gateway variables nor its `.env` / `.env.api`."""

import pytest

_VARIABLES = (
    "RS_ROOT",
    "RS_ENV_FILE",
    "RS_ENV_API_FILE",
    "RS_GATEWAY_TOKEN",
    "RS_GATEWAY_ADMIN_TOKEN",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
