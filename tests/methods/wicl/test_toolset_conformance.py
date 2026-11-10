"""The toolset conformance suite (`roboscript.testing.toolsets`) on the toolsets wicl
registers in `roboscript.toolsets`, loaded through the entry point as the core's tests load the
built-ins."""

import pytest

from roboscript.testing import registered
from roboscript.testing.toolsets import ToolsetConformance, ToolsetTarget

# P4's `rsi` toolset is tested with P4 (tests/methods/rsi)
BUILTIN = [n for n in registered("roboscript.toolsets", "roboscript-methods") if n != "rsi"]

pytestmark = pytest.mark.skipif(
    not BUILTIN,
    reason="roboscript-methods is not installed here (the simulator env uses PYTHONPATH)",
)


@pytest.fixture(params=BUILTIN)
def toolset_target(request: pytest.FixtureRequest) -> ToolsetTarget:
    return ToolsetTarget.registered(request.param)


class TestWiclToolsets(ToolsetConformance):
    pass


def test_wicl_registers_its_toolsets() -> None:
    assert BUILTIN == ["demo", "done"]
