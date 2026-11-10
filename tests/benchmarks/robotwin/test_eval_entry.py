"""RoboTwin's evaluation is registered like any benchmark's (`roboscript.evaluations`) and
composes a graph that runs."""

from pathlib import Path

from roboscript.contracts.jobs import EVALUATIONS
from roboscript.kernel import entrypoints
from roboscript.testing.jobs import check_evaluation
from roboscript_robotwin.eval.blocks import P3Settings
from roboscript_robotwin.eval.jobs import P3Jobs


def test_robotwin_registers_its_evaluation(tmp_path: Path) -> None:
    assert "robotwin" in entrypoints.names(EVALUATIONS)
    factory = entrypoints.load(EVALUATIONS, "robotwin")
    assert factory is P3Jobs and factory.settings is P3Settings
    assert check_evaluation("robotwin", ["red", "blue"], tmp_path) == []
