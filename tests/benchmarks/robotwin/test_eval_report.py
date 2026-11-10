"""RoboTwin P3's part of a pipeline's status and summary (`P3Report`) conforms
(`roboscript.testing.report`), its spare lane included."""

from pathlib import Path

from roboscript.orchestrate.jobs import RunKey
from roboscript.testing.report import check_report_part
from roboscript_robotwin.eval.blocks import P3Settings
from roboscript_robotwin.eval.report import P3Report


def test_robotwin_reports_its_evaluation(tmp_path: Path) -> None:
    part = P3Report(
        P3Settings(blocks=[0, 1], randomized="spare", spare_parallel=True),
        units=lambda key: key.variant != "first",
        gated=lambda run_dir, task: False,
        zeroed=lambda run_dir, task: False,
        stopped=lambda row, rows: False,
    )
    keys = [RunKey("red", "first"), RunKey("red", "default")]
    assert check_report_part(part, keys, tmp_path) == []
    assert list(part.readings) == ["p3"]
    assert part.flags([]) == []  # every official block counts alike
