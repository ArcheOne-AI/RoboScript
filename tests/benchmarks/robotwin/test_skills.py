"""RoboTwin's skillset `robotwin` (docs/how-to/add-a-skillset.md): it mounts on the core toolsets it
extends, alike in process and served, and only where it is named."""

import pytest

from roboscript.embodiments import load_embodiment
from roboscript.testing.skills import PluginTarget, SkillSetPluginConformance
from roboscript.tools.skillsets import load_skillset, mounted
from roboscript.toolsets.manipulation import Manipulation
from roboscript.toolsets.perception import Perception
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.skills import ROBOTWIN_SKILLS

pytestmark = pytest.mark.unit

ADDED = {
    "manipulation": ["rotate_about"],
    "perception": ["estimate_hinge", "measure_turn"],
}


class TestRoboTwinSkillSet(SkillSetPluginConformance):
    @pytest.fixture
    def plugin_target(self) -> PluginTarget:
        hosts = {"manipulation": Manipulation, "perception": Perception}
        return PluginTarget("robotwin", load_embodiment(ROBOTWIN.embodiment), hosts)


def test_it_is_registered_and_named_by_the_benchmark() -> None:
    assert ROBOTWIN.skillsets == ["robotwin"]
    assert load_skillset("robotwin") is ROBOTWIN_SKILLS
    assert load_skillset("roboscript_robotwin.skills:ROBOTWIN_SKILLS") is ROBOTWIN_SKILLS


@pytest.mark.parametrize("host", [Manipulation, Perception])
def test_its_skills_exist_only_where_it_is_mounted(host: type) -> None:
    without = [spec.name for spec in mounted(host, ["mock"]).specs]  # the Mock's bindings only
    with_it = [spec.name for spec in mounted(host, ["robotwin"]).specs]
    assert not set(ADDED[host.name]) & set(without)
    assert with_it == without + ADDED[host.name]


"""The P3 host's prompt digest (no tool list), as each P3 record's `templates` holds it."""


SIM_SIDE = (
    "roboscript_robotwin.sim",
    "roboscript_robotwin.eval",
)
"""RoboTwin's simulator side and its official evaluation (the ground-truth scene reader is the
core's `roboscript.demos.gt_scene`, in `FORBIDDEN`)."""


def test_it_keeps_to_the_information_boundary() -> None:
    from roboscript.testing.boundary import check_boundary

    assert check_boundary("robotwin", forbidden=SIM_SIDE) == []
