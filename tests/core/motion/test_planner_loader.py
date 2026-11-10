"""The planner service loads its planner by registered name or import path."""

import pytest

from roboscript.kernel import entrypoints
from roboscript.motion.config import CuroboConfig
from roboscript.motion.curobo import CuroboPlanner
from roboscript.motion.service import CUROBO, load_planner


def test_the_default_is_curobo_by_import_path() -> None:
    assert load_planner(CUROBO) is CuroboPlanner
    assert CuroboPlanner.Config is CuroboConfig


@pytest.mark.skipif(
    "curobo" not in entrypoints.names("roboscript.planners"),
    reason="roboscript is not installed here (PYTHONPATH only)",
)
def test_by_registered_name() -> None:
    assert load_planner("curobo") is CuroboPlanner
    with pytest.raises(TypeError, match="no Config model"):
        load_planner("mock")  # in process only
