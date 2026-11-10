"""`run.json`: what a run is made of, written once as the runner starts."""

import json

import pytest

from roboscript.hub.lock import catalog_lock, lock_hash
from roboscript.tools import LocalProvider
from roboscript.toolsets.manipulation import Manipulation

from .conftest import Setup


async def test_run_json_names_the_code_method_skillsets_and_catalog(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RS_SOURCE_SHA", "abc1234")
    manipulation = LocalProvider(Manipulation(), {"planner": {"mock": True}}, ["mock"])
    async with setup.runner({}, extra=[manipulation]) as runner:
        catalog = lock_hash(catalog_lock(runner.hub.mounted, runner.hub.phases.specs))
    run = json.loads((setup.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["source"] == "abc1234"
    assert run["method"]["name"] == "test"
    assert run["method"]["tiers"]["s1"]["actor"] == "P"
    assert run["method"]["tiers"]["p3"] == {"actor": "D", "demo_reader": "D", "reviser": "D"}
    assert run["skillsets"]["manipulation"] == {"mock": "1"}
    assert run["skillsets"]["sensing"] == {}
    assert run["catalog"] == catalog
