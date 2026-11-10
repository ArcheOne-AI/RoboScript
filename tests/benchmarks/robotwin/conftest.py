"""Tests of the RoboTwin integration. `roboscript_robotwin` is installed in the agent venv (a
workspace member) but not in the simulator env, which gets `packages/benchmarks/robotwin/src` on
PYTHONPATH (README); this file adds it for either, and this directory for the tests'
helper modules (`robotwin_fakes`, `task_whitelist`, `snapshot_harness`) and the RoboTwin example
directory (where `scripts/run_s2.py` and `robotwin_pipeline` are, not
packages)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (
    REPO / "examples" / "simBenchmarks" / "RoboTwin",
    REPO / "packages" / "benchmarks" / "robotwin" / "src",
    HERE,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
