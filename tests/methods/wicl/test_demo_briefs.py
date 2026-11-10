"""wicl's briefs of a demonstration package (`roboscript_methods.wicl.roles`): a Reviser reads the
ground truth and an Actor never does, and the demo-context levels are wicl's, which the core
checks a host's against. Built from the synthetic recording of
`tests/core/learning/test_demo_package.py`."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from roboscript.contracts.tools import TextPart
from roboscript.learning import load_demo_truth, package_images
from roboscript.learning.runtime import check_level
from roboscript_methods.wicl.roles import (
    Briefs,
    demo_context_brief,
    demo_seed,
    truth_brief,
)


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_test_demo_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load(Path(__file__).resolve().parents[2] / "core" / "learning" / "test_demo_package.py")
lift_package, truth_package = _core.lift_package, _core.truth_package


async def test_a_reviser_reads_the_ground_truth_and_an_actor_never_does(tmp_path: Path) -> None:
    package = await truth_package(tmp_path)
    layer = load_demo_truth(tmp_path / "p")
    assert layer is not None
    [text] = truth_brief(layer)
    assert isinstance(text, TextPart)
    assert "(and categories): box (box), laptop (laptop)." in text.text
    moments = [line for line in text.text.splitlines() if line.startswith("At ")]
    assert moments == [
        "At the start:",
        "At left K1 (approach):",
        "At left K2 (grasp):",
        "At left K3 (lift):",
        "At the success:",
    ]
    assert "frame_" not in text.text  # a Reviser holds only its tries' frames
    assert "- box: (0.060, 0.000, 0.000), q (1.000, 0.000, 0.000, 0.000)" in text.text
    assert "never write these names or numbers into the Memory" in text.text
    # What an S2 or P3 Actor is shown, at every level: no ground truth, no coordinates or
    # tolerances; the images are the package's drawn views only.
    drawn = {blob.id for blob in package_images(package)}
    for level in ("none", "goal", "ops", "full"):
        parts = demo_context_brief(package, level)  # type: ignore[arg-type]
        shown = "\n".join(part.text for part in parts if isinstance(part, TextPart))
        assert "box" not in shown and "laptop" not in shown and "ground truth" not in shown
        assert "0.060" not in shown and "within" not in shown and " deg" not in shown
        assert {p.blob.id for p in parts if not isinstance(p, TextPart)} <= drawn
        seed = demo_seed(package, level)  # type: ignore[arg-type]
        assert seed is None or {e.observation.id for e in seed.frames} == {
            e.observation.id for e in package.frames
        }  # the recording's observations, which hold no ground truth


def test_the_levels_are_wicl_s_and_the_core_checks_a_host_s_against_them() -> None:
    """wicl declares its demo-context levels (`roles.DemoContext`, `Briefs.levels`); the core
    accepts exactly those for a host's `demo_context` (`learning.runtime.check_level`)."""
    assert Briefs().levels == ("none", "goal", "ops", "full")
    for level in Briefs().levels:
        check_level(Briefs(), level)
    with pytest.raises(ValueError, match=r"'sketch' .* \(none, goal, ops, full\)$"):
        check_level(Briefs(), "sketch")
