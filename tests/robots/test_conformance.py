"""The robot conformance suite (`roboscript_robots.testing`) on every robot the package registers
in `roboscript.robots`, loaded through the entry point as a third-party package's would be."""

from pathlib import Path

import pytest

from roboscript.testing import registered
from roboscript_robots.base import GROUP
from roboscript_robots.testing import RobotConformance, RobotTarget

BUILTIN = registered(GROUP, "roboscript-robots")

pytestmark = pytest.mark.skipif(
    not BUILTIN, reason="roboscript-robots is not installed here (no entry points)"
)


@pytest.fixture(params=BUILTIN)
def robot_target(request: pytest.FixtureRequest, tmp_path: Path) -> RobotTarget:
    return RobotTarget(request.param, tmp_path)


class TestBuiltinRobots(RobotConformance):
    pass


def test_the_package_registers_a_fake_robot_for_each_configuration() -> None:
    assert BUILTIN == ["fake_dual_arm", "fake_mobile_manipulator", "fake_single_arm"]
