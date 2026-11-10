"""RoboTwin's binding data with none of its own skills (`roboscript_robotwin.skills.
bindings_only`): the P1 skill builder's starting point on RoboTwin."""

import pytest

from roboscript_robotwin.bindings import BINDINGS
from roboscript_robotwin.skills.bindings_only import BINDINGS_ONLY, GUIDE

pytestmark = pytest.mark.unit


def test_bindings_only_has_robotwins_bindings_and_no_skill() -> None:
    manifest = BINDINGS_ONLY.manifest
    assert manifest.skills == []
    assert manifest.bindings == BINDINGS
    assert manifest.envs == ["robotwin"] and manifest.layer == "benchmark"
    assert manifest.guide == GUIDE.read_text(encoding="utf-8")
