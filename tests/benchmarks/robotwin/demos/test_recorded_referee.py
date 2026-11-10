"""roboscript.demos.referee on recorded RoboTwin attempts."""

import json
from pathlib import Path

from roboscript.demos import (
    KeypointConfig,
    RefereeConfig,
)
from roboscript.demos.referee import symmetric

CONFIG = KeypointConfig()
STRICT = RefereeConfig(xy_m=0.02, z_m=0.01)


RECORDED = json.loads(
    (Path(__file__).parent / "data" / "recorded_referee.json").read_text(encoding="utf-8")
)
"""A recorded fixture: demonstrations' keypoints and ground-truth poses at them, successful
attempts' executed TCP samples and grippers, and cup hulls."""


def test_the_plain_cups_are_symmetric_and_the_mugs_are_not() -> None:
    """Cup hulls (a convex decomposition: inner walls, tapered): model 0 is a plain cup, model 6 a
    mug with a handle."""
    assert symmetric(RECORDED["cup0_hull"])
    assert not symmetric(RECORDED["cup6_hull"])
