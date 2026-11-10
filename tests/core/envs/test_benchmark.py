"""The benchmark descriptor and its loader (import path or entry-point name)."""

from pathlib import Path

import pytest

from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.ids import TaskName
from roboscript.envs import (
    MissingCapabilities,
    check_capabilities,
    load_benchmark,
    offered_by,
)
from roboscript.envs.mock import MOCK, MockConfig, MockEnv
from roboscript.kernel import entrypoints


def test_load_by_import_path_or_name() -> None:
    assert load_benchmark("roboscript.envs.mock:MOCK") is MOCK
    with pytest.raises(TypeError, match="is a dict, not a Benchmark"):
        load_benchmark("roboscript.envs.mock:TASKS")
    with pytest.raises(LookupError, match="nothing named 'nope'"):
        load_benchmark("nope")
    if "mock" not in entrypoints.names("roboscript.benchmarks"):
        pytest.skip("roboscript is not installed here (the simulator env uses PYTHONPATH)")
    assert load_benchmark("mock") is MOCK


async def test_declared_capabilities(tmp_path: Path) -> None:
    """The descriptor and the server's `discover()` declare the same faces; the mock is no
    demonstration source, so a method that needs one is stopped with the reason."""
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path))
    found = (await env.discover()).capabilities
    assert found.model_copy(update={"sensing": None}) == MOCK.capabilities
    assert found.sensing is not None and found.sensing.cameras == list(env.profile.cameras)
    assert await offered_by(MOCK) == MOCK.capabilities
    check_capabilities(EnvCapabilities(success={"latch"}, expert=True), found, who="m", env="mock")
    needs = EnvCapabilities(reset={"seed"}, demos=True)
    with pytest.raises(MissingCapabilities, match="m needs what mock does not offer") as raised:
        check_capabilities(needs, MOCK.capabilities, who="m", env="mock")
    assert raised.value.missing == ["demos"]
