"""Tests of the real-robot template (`examples/realRobots/_template`). It is not a package: this
file puts it on `sys.path`, as `PYTHONPATH` does for its scripts. No test needs hardware: the
robots package's fake backend stands in for it."""

import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[3] / "examples" / "realRobots" / "_template"
if str(TEMPLATE) not in sys.path:
    sys.path.insert(0, str(TEMPLATE))
