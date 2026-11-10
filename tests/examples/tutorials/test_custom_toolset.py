"""The `tape` toolset passes the toolset conformance suite, with and without pytest, and the demo
shows what the README says."""

from pathlib import Path

import pytest
from custom_toolset.demo import demo
from custom_toolset.tape import Tape

from roboscript.testing import precheck
from roboscript.testing.toolsets import ToolsetConformance, ToolsetTarget

pytestmark = pytest.mark.unit


@pytest.fixture
def toolset_target() -> ToolsetTarget:
    # Not registered in an entry-point group here, so the class is named directly. A package
    # would register it and use `ToolsetTarget.registered("tape")`.
    return ToolsetTarget("tape", Tape, {"unit": "cm"})


class TestTape(ToolsetConformance):
    pass


async def test_the_suite_runs_without_pytest() -> None:
    results = await precheck(ToolsetConformance(), ToolsetTarget("tape", Tape))
    assert results and all(why is None for why in results.values()), results


async def test_the_demo_measures_in_the_configured_unit(tmp_path: Path) -> None:
    listed, gap, moved, missing_mark, missing_arm = await demo(tmp_path, unit="cm")
    assert listed == "arm_gap, mark, moved_since, finish"
    assert gap["unit"] == "cm" and gap["distance"] == pytest.approx(50.0, abs=1e-6)
    assert moved["distance"]["left"] == pytest.approx(5.0, abs=1e-6)
    assert moved["distance"]["right"] == pytest.approx(0.0, abs=1e-6)
    assert (missing_mark, missing_arm) == ("tool_error", "tool_error")
