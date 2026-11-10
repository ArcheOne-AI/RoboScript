"""Tests of the BEHAVIOR entry skeleton. `roboscript_behavior` is installed in the agent venv (a
workspace member); this file adds its sources for any other interpreter, and this directory for
the fake official module (`fakes`). No test imports the real OmniGibson."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[2] / "packages" / "benchmarks" / "behavior" / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
