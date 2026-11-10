"""Tests of the tutorials (`examples/tutorials/`). They are not packages of the repository: this
file puts their directory on `sys.path`, as running them from there (or `PYTHONPATH`) would."""

import sys
from pathlib import Path

TUTORIALS = Path(__file__).resolve().parents[3] / "examples" / "tutorials"
if str(TUTORIALS) not in sys.path:
    sys.path.insert(0, str(TUTORIALS))
