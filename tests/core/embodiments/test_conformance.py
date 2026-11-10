"""The embodiment conformance suite (`roboscript.testing.embodiments`) on every profile the core
registers in `roboscript.embodiments`, loaded through the entry point as a third-party package's
would be, and on profiles that break it."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.testing import precheck, registered
from roboscript.testing.embodiments import EmbodimentConformance, EmbodimentTarget

BUILTIN = registered("roboscript.embodiments")

pytestmark = pytest.mark.skipif(
    not BUILTIN, reason="roboscript is not installed here (the simulator env uses PYTHONPATH)"
)


@pytest.fixture(params=BUILTIN)
def embodiment_target(request: pytest.FixtureRequest) -> EmbodimentTarget:
    return EmbodimentTarget(request.param)


class TestBuiltinEmbodiments(EmbodimentConformance):
    pass


def test_the_core_registers_the_mock_robot() -> None:
    assert "mock_bimanual" in BUILTIN


def _profile(tmp_path: Path, change: Any) -> EmbodimentTarget:
    data = yaml.safe_load(MOCK_BIMANUAL.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return EmbodimentTarget(path)


@pytest.mark.parametrize(
    ("check", "change"),
    [
        (
            "test_arms_start_inside_their_stops",
            lambda d: d["arms"]["left"].update(stops=[[-1, 1]] * 6),  # home x is -0.25..., z 0.95
        ),
        (
            "test_pose_conventions_are_frames",
            lambda d: d["poses"].update(tcp_closing_axis=[0.0, 2.0, 0.0]),
        ),
        (
            "test_planner_assets_are_relative",
            lambda d: d["arms"]["left"].update(planner={"urdf": "/abs/arm.urdf", "base": _BASE}),
        ),
    ],
)
async def test_the_suite_catches_a_broken_profile(tmp_path: Path, check: str, change: Any) -> None:
    results = await precheck(EmbodimentConformance(), _profile(tmp_path, change))
    assert results[check] is not None and results[check].startswith("Assert"), results


_BASE = {"p": [0.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0, 0.0]}
