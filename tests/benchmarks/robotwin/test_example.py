"""The RoboTwin example without the simulator: descriptor, profile, configs, and the two
boundaries (the agent side imports no simulator code; the shim reads only whitelisted fields)."""

import ast
import collections
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from task_whitelist import task_uses, to_local

from roboscript.envs import load_benchmark
from roboscript_robotwin.benchmark import HERE, P3_TASK_CONFIGS, ROBOTWIN, TASK_CONFIGS
from roboscript_robotwin.config import RoboTwinConfig
from roboscript_robotwin.sim import shim, simulator

SRC = HERE.parent
EXAMPLE = HERE.parents[4] / "examples" / "simBenchmarks" / "RoboTwin"


def test_the_descriptor() -> None:
    assert load_benchmark("roboscript_robotwin.benchmark:ROBOTWIN") is ROBOTWIN
    assert (ROBOTWIN.name, ROBOTWIN.interpreter, ROBOTWIN.env_server) == (
        "robotwin",
        "sim",
        "roboscript_robotwin.sim.host",
    )
    assert ROBOTWIN.pythonpath == [SRC]
    assert len(ROBOTWIN.tasks) == 50 and ROBOTWIN.tasks["beat_block_hammer"].step_limit == 400
    categories = collections.Counter(info.category for info in ROBOTWIN.tasks.values())
    assert categories == {"C1": 4, "C2": 11, "C3": 10, "C4": 10, "C5": 15}
    assert all(400 <= info.step_limit <= 1700 for info in ROBOTWIN.tasks.values())
    RoboTwinConfig.model_validate(
        ROBOTWIN.env_config | {"blob_dir": "/tmp/blobs", "task": "beat_block_hammer"}
    )
    with pytest.raises(ValueError, match="RoboTwin has no task 'nope'"):
        RoboTwinConfig(task="nope", blob_dir=Path("/tmp/blobs"))
    assert RoboTwinConfig(task="beat_block_hammer").blob_dir is None  # the stack's RS_BLOB_DIR


def test_the_aloha_agilex_profile() -> None:
    profile = ROBOTWIN.profile()
    assert profile.name == "aloha_agilex" and profile.action_dim == 14
    left, right = profile.arms["left"], profile.arms["right"]
    assert (left.action, left.gripper_index, right.action, right.gripper_index) == (
        (0, 6),
        6,
        (7, 13),
        13,
    )
    assert left.ee_to_tcp == right.ee_to_tcp == (0.12, 0.0, 0.0)
    assert left.gripper.scale == (-0.01, 0.045)
    assert left.gripper.opening(0.0) == pytest.approx(left.gripper.empty, abs=1e-4)
    assert left.gripper.holding_threshold == right.gripper.holding_threshold == 0.188
    head = profile.cameras["head_camera"]
    assert (head.type, head.width, head.height) == ("Large_D435", 640, 480)
    assert sorted(profile.cameras) == ["head_camera", "left_camera", "right_camera"]


def test_the_task_configs() -> None:
    clean, randomized = (
        yaml.safe_load(TASK_CONFIGS[mode].read_text(encoding="utf-8"))
        for mode in ("clean", "randomized")
    )
    for config in (clean, randomized):
        assert config["camera"]["head_camera_type"] == "Large_D435"
        assert config["camera"]["wrist_camera_type"] == "D435"
        data = config["data_type"]
        assert data["rgb"] and data["depth"] and data["qpos"] and data["endpose"]
        assert not any(data[k] for k in ("mesh_segmentation", "actor_segmentation", "pointcloud"))
        assert (
            not config["collect_data"]
            and not config["eval_video_log"]
            and config["render_freq"] == 0
        )
    assert clean["domain_randomization"] != randomized["domain_randomization"]
    assert {k: v for k, v in clean.items() if k != "domain_randomization"} == {
        k: v for k, v in randomized.items() if k != "domain_randomization"
    }
    for mode, path in P3_TASK_CONFIGS.items():  # P3: the same with the official video on
        p3 = yaml.safe_load(path.read_text(encoding="utf-8"))
        s2 = yaml.safe_load(TASK_CONFIGS[mode].read_text(encoding="utf-8"))
        assert p3 == {**s2, "eval_video_log": True} and path.stem == f"rs_agentic_{mode}_p3"


def test_the_agent_side_imports_no_simulator_code() -> None:
    demos = [
        "roboscript_robotwin.frames",
        "roboscript_robotwin.demos.records",
        "roboscript_robotwin.demos.validate",
    ]
    if importlib.util.find_spec("h5py"):  # the expert data's reader (the simulator env has it)
        demos += ["roboscript_robotwin.demos.expert", "roboscript_robotwin.demos.recorded"]
        demos += ["roboscript_robotwin.demos.review"]
    code = (
        "import sys, roboscript_robotwin, roboscript_robotwin.benchmark\n"
        "import roboscript_robotwin.config, roboscript_robotwin.demos.replay\n"
        "import roboscript_robotwin.eval.install, roboscript_robotwin.eval.launch\n"
        "import roboscript_robotwin.eval.policy\n"
        f"import {', '.join(demos)}\n"
        "bad = ('roboscript_robotwin.sim', 'sapien', 'envs', 'torch', 'curobo', 'mplib')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in bad or m.startswith(bad[0])))"
    )
    assert _imported(code) == "[]"


def _imported(code: str) -> str:
    path = os.pathsep.join([str(SRC), *sys.path])
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": path},
    )
    return done.stdout.strip()


def test_the_shim_touches_only_whitelisted_attributes() -> None:
    """Every use of the task object in shim.py is an attribute chain listed in `ALLOWED` (all of
    them used), inside a function that takes it as its parameter `task`, or passes it on to a
    function of shim.py itself; nothing rebinds `task`."""
    used, stray = task_uses(Path(shim.__file__), "task", to_local)
    assert not stray, f"shim.py uses the task object outside attribute chains on lines {stray}"
    assert used == shim.ALLOWED


def test_the_agent_plane_reaches_the_task_object_only_through_the_shim() -> None:
    """In simulator.py, `robot`, `observe` and `execute` hand the task object (`self._task`, or a
    local name bound to it) only to `shim.*` functions."""
    tree = ast.parse(Path(simulator.__file__).read_text(encoding="utf-8"))
    names = {"robot", "observe", "execute"}
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {method.name for method in methods} == names
    for method in methods:
        parents = {child: node for node in ast.walk(method) for child in ast.iter_child_nodes(node)}
        aliases, bindings = set(), set()
        for assign in (node for node in ast.walk(method) if isinstance(node, ast.Assign)):
            target, value = assign.targets[0], assign.value
            both_tuples = isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
            pairs = zip(target.elts, value.elts, strict=True) if both_tuples else [(target, value)]  # type: ignore[attr-defined]
            for name, bound in pairs:
                if _self_task(bound):
                    aliases.add(name.id)  # type: ignore[attr-defined]
                    bindings.add(bound)
        for node in ast.walk(method):
            alias = (
                isinstance(node, ast.Name) and node.id in aliases and isinstance(node.ctx, ast.Load)
            )
            if alias or (_self_task(node) and node not in bindings):
                call = parents.get(node)
                to_shim = (
                    isinstance(call, ast.Call)
                    and node in call.args
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "shim"
                )
                assert to_shim, f"simulator.{method.name}, line {node.lineno}"


def _self_task(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_task"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def test_the_pipeline_imports_no_simulator_code() -> None:
    code = (
        "import sys, robotwin_pipeline, robotwin_pipeline.worker\n"
        "bad = ('roboscript_robotwin.sim', 'sapien', 'envs', 'torch', 'curobo', 'mplib')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in bad or m.startswith(bad[0])))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert done.stdout.strip() == "[]"
