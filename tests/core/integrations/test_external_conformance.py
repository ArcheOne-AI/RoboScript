"""The external-brain conformance suite against a served Hub over MCP (the
current MCP generation, strict permits), the Mock toolset on a remote toolset server."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from roboscript.testing.external import ExternalBrainConformance, ServedHub

SETUPS = [("2026-07-28", "strict", True)]


@pytest.fixture(params=SETUPS, ids=lambda s: f"{s[0]}-{s[1]}{'-remote' if s[2] else ''}")
async def served_hub(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[ServedHub]:
    era, mode, remote = request.param
    async with ServedHub(tmp_path, era=era, mode=mode, remote=remote) as hub:
        yield hub


class TestServedHub(ExternalBrainConformance):
    pass
