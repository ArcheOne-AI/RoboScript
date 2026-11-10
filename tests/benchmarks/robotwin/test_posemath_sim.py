"""The pose-math ports against RoboTwin's own functions on the real simulator (markers `sim` and
`gpu`): every traced `choose_grasp_pose`, `get_grasp_pose` and `get_place_pose` call of two
tasks' expert pre-runs (a side grasp, grasps chosen among contact points and placements with
placement points), the port's result within 1e-6 of the official code run in float64 and within
1e-4 of it as run (float32 SAPIEN poses).

    ROBOTWIN_ROOT=... python -m pytest -m sim tests/benchmarks/robotwin/test_posemath_sim.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from roboscript_robotwin.config import robotwin_root

pytestmark = [pytest.mark.sim, pytest.mark.gpu]
TASKS = ("place_empty_cup", "handover_block")


def test_the_ports_give_the_official_poses(artifacts_dir: Path) -> None:
    root = robotwin_root()
    if root is None:
        pytest.skip("set ROBOTWIN_ROOT (environment or .env)")
    out = artifacts_dir / "rows.jsonl"
    env = {**os.environ, "ROBOTWIN_ROOT": str(root)}
    for task in TASKS:
        command = [sys.executable, "-m", "roboscript_robotwin.sim.posemath_check", "--task", task]
        subprocess.run(
            [*command, "--seeds", "1", "--out", str(out)], env=env, check=True, timeout=900
        )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {row["task"] for row in rows} == set(TASKS)
    assert {row["fn"] for row in rows} >= {"choose_grasp_pose", "get_place_pose"}
    for row in rows:
        assert row["unused_plans"] == 0 and row["targets_diff"] <= 1e-4 and row["tilt"], row
        if row["diff64"] is None:
            assert row["diff64_same_none"] and row["diff_same_none"], row
        else:
            assert row["diff64"] <= 1e-6 and row["diff"] <= 1e-4, row
