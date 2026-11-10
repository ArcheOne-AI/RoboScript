"""Tests of the RoboTwin example's pipeline (`examples/simBenchmarks/RoboTwin/robotwin_pipeline`).
It is not a package: this file puts the example directory on `sys.path` (as `PYTHONPATH` does
for its scripts), with this directory and the RoboTwin tests' helpers (`official_fakes`,
`fake_motion`) and, for the simulator env, `roboscript_robotwin`'s sources. The processes the
tests start (pipeline workers run `pipeline_fakes.py` as a script) get the helpers through
`PYTHONPATH`."""

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HELPERS = HERE.parent
PATHS = (
    REPO / "packages" / "benchmarks" / "robotwin" / "src",
    HELPERS,
    REPO / "examples" / "simBenchmarks" / "RoboTwin",
    HERE,
)
for path in PATHS:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _helpers_on_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(HELPERS), inherited) if p))
