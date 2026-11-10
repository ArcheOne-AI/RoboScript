"""Pytest plugin that records every test session under `<rootdir>/test-results/`.

Enable it with `-p roboscript.testing.pytest_plugin` (the root `pyproject.toml` does).
Each session writes `test-results/<YYYYmmdd-HHMMSS>-<sha>/`:

    summary.json    totals and failures; read this first
    junit.xml
    pytest.log      Python logging records of the whole session
    tmp/            pytest basetemp
    cases/<test>/   the `artifacts_dir` of each test; kept only for failed tests, which
                    also get stdout.txt (captured output) and error.txt (the failure)

`test-results/latest` points to the newest run. Finished runs beyond the newest
`rs_keep_runs` (ini option, default 10) are deleted; `--keep-passed` keeps the case
directories of passed tests. Options the user passes (`--basetemp`, `--junitxml`,
`--log-file`) take precedence.

`RS_SOURCE_SHA` names the source revision; the default is `git rev-parse --short HEAD`,
plus `-dirty` when the tree has changes, or `nogit`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import roboscript

_TOTALS = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
# Options that only print information: no test session worth recording.
_INFO_OPTIONS = ("help", "markers", "showfixtures", "show_fixtures_per_test")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("roboscript").addoption(
        "--keep-passed",
        action="store_true",
        help="keep the test-results case directories of passed tests",
    )
    parser.addini("rs_keep_runs", "number of test-results runs to keep", type="int", default=10)


# tryfirst: tmpdir, junitxml and logging read their options in their own pytest_configure.
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if any(getattr(config.option, name) for name in _INFO_OPTIONS):
        return
    keep_runs = config.getini("rs_keep_runs")
    if keep_runs < 1:
        raise pytest.UsageError(f"rs_keep_runs must be at least 1, got {keep_runs}")
    now = dt.datetime.now().astimezone()
    sha = _source_sha(config.rootpath)
    # test-results may be a symlink (to a faster disk): pytest's tmp factory refuses a basetemp
    # whose resolved path differs from the path it was given.
    results_root = (config.rootpath / "test-results").resolve()
    run_dir = _new_run_dir(results_root, f"{now:%Y%m%d-%H%M%S}-{sha}")
    if config.option.basetemp is None:
        config.option.basetemp = str(run_dir / "tmp")
    if config.option.xmlpath is None:
        config.option.xmlpath = str(run_dir / "junit.xml")
    if config.option.log_file is None and not config.getini("log_file"):
        config.option.log_file = str(run_dir / "pytest.log")
    recorder = _Recorder(config, run_dir, sha, now, keep_runs, config.option.keep_passed)
    config.stash[_RECORDER] = recorder
    config.pluginmanager.register(recorder, "roboscript-results")


@pytest.fixture
def artifacts_dir(request: pytest.FixtureRequest) -> Path:
    """This test's directory under `test-results/<run>/cases/`; kept if the test fails."""
    path = request.config.stash[_RECORDER].case_dir(request.node.nodeid)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(eq=False)  # pytest keeps registered plugins in sets: hash by identity
class _Recorder:
    config: pytest.Config
    run_dir: Path
    sha: str
    started_at: dt.datetime
    keep_runs: int
    keep_passed: bool
    start: float = field(default_factory=time.monotonic)
    totals: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_TOTALS, 0))
    failures: dict[str, dict[str, str]] = field(default_factory=dict)

    def case_dir(self, nodeid: str) -> Path:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid).strip("_")[:100]
        return self.run_dir / "cases" / f"{name}-{hashlib.sha1(nodeid.encode()).hexdigest()[:8]}"

    def pytest_report_header(self) -> str:
        return f"test results: {self.run_dir}"

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.totals["error"] += 1
            self._record_failure(report)
        elif report.skipped:
            self.totals["skipped"] += 1

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # The category the terminal summary uses: "failed", "error", "xfailed", "" ...
        status = self.config.hook.pytest_report_teststatus(report=report, config=self.config)
        if status[0] in self.totals:
            self.totals[status[0]] += 1
        if report.failed:
            self._record_failure(report)
        elif report.when == "teardown" and report.nodeid not in self.failures:
            case = self.case_dir(report.nodeid)
            if not self.keep_passed and case.exists():
                shutil.rmtree(case)

    def _record_failure(self, report: pytest.TestReport | pytest.CollectReport) -> None:
        case = self.case_dir(report.nodeid)
        case.mkdir(parents=True, exist_ok=True)
        # A report carries the captured sections of all earlier phases, so overwrite.
        sections = "".join(f"----- {title} -----\n{text}\n" for title, text in report.sections)
        (case / "stdout.txt").write_text(sections, encoding="utf-8")
        with (case / "error.txt").open("a", encoding="utf-8") as f:
            f.write(f"----- {report.when} -----\n{report.longreprtext}\n")
        self.failures.setdefault(
            report.nodeid,
            {
                "nodeid": report.nodeid,
                "reason": _first_line(report),
                "dir": case.relative_to(self.run_dir).as_posix(),
            },
        )

    def pytest_sessionfinish(self, exitstatus: int) -> None:
        summary = {
            "run_id": self.run_dir.name,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_s": round(time.monotonic() - self.start, 3),
            "exit_status": int(exitstatus),
            "sha": self.sha,
            "python": platform.python_version(),
            "roboscript": roboscript.__version__,
            "totals": self.totals,
            "failures": list(self.failures.values()),
        }
        text = json.dumps(summary, indent=2) + "\n"
        (self.run_dir / "summary.json").write_text(text, encoding="utf-8")
        root = self.run_dir.parent
        link = root / f".latest-{os.getpid()}"
        link.symlink_to(self.run_dir.name, target_is_directory=True)
        os.replace(link, root / "latest")
        # Only finished runs: a session running elsewhere has no summary.json yet.
        finished = sorted(p.parent for p in root.glob("????????-??????-*/summary.json"))
        for old in finished[: max(0, len(finished) - self.keep_runs)]:
            if old != self.run_dir:
                shutil.rmtree(old, ignore_errors=True)


_RECORDER = pytest.StashKey[_Recorder]()


def _source_sha(cwd: Path) -> str:
    sha = os.environ.get("RS_SOURCE_SHA")
    if sha:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", sha):
            raise pytest.UsageError(f"RS_SOURCE_SHA must be a plain revision name, got {sha!r}")
        return sha
    try:
        head = _git(cwd, "rev-parse", "--short", "HEAD")
        dirty = _git(cwd, "--no-optional-locks", "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return f"{head}-dirty" if dirty else head


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _new_run_dir(root: Path, name: str) -> Path:
    run_dir, n = root / name, 1
    while True:
        try:
            run_dir.mkdir(parents=True)
            return run_dir
        except FileExistsError:  # another session started in the same second
            n += 1
            run_dir = root / f"{name}-{n}"


def _first_line(report: pytest.TestReport | pytest.CollectReport) -> str:
    crash = getattr(report.longrepr, "reprcrash", None)
    text = crash.message if crash is not None else str(report.longrepr)
    return text.strip().split("\n", 1)[0]
