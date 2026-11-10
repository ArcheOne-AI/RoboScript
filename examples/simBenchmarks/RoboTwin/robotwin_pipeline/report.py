"""The pipeline's status and summary, read from the jobs' records (never from a job's report of
itself): `status.json` and `status.md` (a compact table) while it runs, `summary.json` and
`summary.md` once every job has ended (or on demand: `run_pipeline.py --summary <dir>`).

The core's `roboscript.orchestrate.report.Report` of two parts, in order: wicl's
(`roboscript_methods.wicl.report.LearnReport`: a task's S1, a variant's S2) and RoboTwin
P3's (`roboscript_robotwin.eval.report.P3Report`: a variant's blocks, its spare lane and the
rate over the variants). The pipeline gives them how a variant's S2 ended (`run.s2_exit`),
whether its P3 does not count (`run.p3_gated`: `p3.unconverged: stop`, or its S2 stopped with no
first-try success, which P3 counts 0), and which jobs are variants; a variant whose task stopped
at S1 (`not_passed`) counts 0 in P3 too.
"""

import time
from pathlib import Path
from typing import Any, Final

from roboscript.contracts.ids import TaskName
from roboscript.orchestrate.report import Report
from roboscript_methods.wicl.ops import S1
from roboscript_methods.wicl.report import LearnReport
from roboscript_robotwin.eval.report import P3Labels, P3Report
from robotwin_pipeline.config import PipelineConfig
from robotwin_pipeline.run import p3_gated, run_keys, s2_exit

LABELS: Final = P3Labels(
    zeroed="stopped at S1 or S2 counted 0",
    not_counted="not counted (p3.unconverged stop)",
    cell="P3 (S2 not converged, not counted)",
)
"""How P3's part names the variants an S2 that did not converge sets apart."""


def report(config: PipelineConfig) -> Report:
    """The pipeline's report under `config`."""

    def exit(run_dir: Path, task: TaskName) -> dict[str, Any] | None:
        return s2_exit(config, run_dir, task)

    def zeroed(run_dir: Path, task: TaskName) -> bool:
        return (s2_exit(config, run_dir, task) or {}).get("s2_exit") == "none"

    def gated(run_dir: Path, task: TaskName) -> bool:
        return p3_gated(config, run_dir, task)

    learn = LearnReport(exit)
    p3 = P3Report(
        config.p3,
        units=lambda key: key.variant != S1,
        gated=gated,
        zeroed=zeroed,
        stopped=learn.stopped,
        labels=LABELS,
    )
    return Report((learn, p3))


def status(
    directory: Path, config: PipelineConfig, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Every job's row, the counts by status and what the scheduler adds (GPUs, alerts)."""
    return report(config).status(directory, run_keys(config), time.time(), extra)


def summary(directory: Path, config: PipelineConfig) -> dict[str, Any]:
    """The status with each job's details and failure taxonomy, the tokens and the flags."""
    return report(config).summary(directory, run_keys(config), time.time())


def write_status(
    directory: Path, config: PipelineConfig, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return report(config).write_status(directory, run_keys(config), time.time(), extra)


def write_summary(directory: Path, config: PipelineConfig) -> dict[str, Any]:
    return report(config).write_summary(directory, run_keys(config), time.time())
