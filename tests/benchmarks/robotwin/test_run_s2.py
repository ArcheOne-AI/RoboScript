"""`scripts/run_s2.py` without the simulator: the stack it starts for a practice and the toolsets
it mounts (manipulation with the stack's planner; the scene toolsets with the stack's planner and
vision service)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

from roboscript.contracts.ids import TaskName
from roboscript.learning import LearningConfig
from roboscript.settings import repo_root
from roboscript_robotwin.benchmark import HERE, ROBOTWIN
from roboscript_robotwin.config import RoboTwinConfig

EXAMPLE = HERE.parents[4] / "examples" / "simBenchmarks" / "RoboTwin"

SCRIPT = EXAMPLE / "scripts" / "run_s2.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_s2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_stack_of_a_practice(tmp_path: Path) -> None:
    script = load_script()
    config = LearningConfig(run_dir=tmp_path, method="wicl", recipe=Path("recipes/wicl"))
    tasks = [TaskName("beat_block_hammer"), TaskName("open_laptop")]
    stack = script.stack_config(config, tasks, Path(sys.executable), [4, 5])
    assert [s.name for s in stack.services] == [
        "env-beat_block_hammer",
        "env-open_laptop",
        "motion",
    ]
    assert [s.gpu for s in stack.services] == [4, 5, 4]
    env = stack.services[0]
    assert (env.module, env.interpreter) == (ROBOTWIN.env_server, "sim")
    written = yaml.safe_load(Path(env.args[1]).read_text(encoding="utf-8"))
    assert RoboTwinConfig.model_validate(written).task == "beat_block_hammer"
    assert stack.run_dir == tmp_path and ROBOTWIN.pythonpath[0] in stack.sim_pythonpath
    assert RoboTwinConfig.model_validate(written).blob_dir == stack.shared_blob_dir()
    motion = stack.services[-1]
    assert (motion.module, motion.interpreter, motion.args) == (
        "roboscript.motion",
        "sim",
        ["--config", "motion"],
    )


def test_robotwin_stacks_mount_every_skillset_tool(tmp_path: Path) -> None:
    """S1's and S2's toolsets mount RoboTwin's skillset, so its tools, which wicl's phases
    name by category (`manipulation.+`, `perception.+`), are all there: none drops out of a
    RoboTwin run."""
    from roboscript.hub.visibility import Phases, absent, mount
    from roboscript.tools.skillsets import extension_tools
    from roboscript_methods.wicl.method import Wicl

    script = load_script()
    (tmp_path / "configs").mkdir()
    example = repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "manipulation.example.yaml"
    (tmp_path / "configs" / "manipulation.yaml").write_text(
        example.read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = script.manipulation_config("http://127.0.0.1:41995", root=tmp_path)
    providers = script.toolsets(config, {"motion": "http://127.0.0.1:1"}, demo=True)
    mounted = mount(providers)
    added = extension_tools("manipulation") | extension_tools("perception")
    assert {"rotate_about", "estimate_hinge", "measure_turn"} <= added
    assert "follow_path" in mounted  # the core manipulation toolset's own
    assert added <= set(mounted)
    assert all(not names for names in absent(providers).values())
    phases = Phases(Wicl.phases, mounted, absent(providers))
    assert {"rotate_about", "follow_path"} <= set(phases.tools("p3", "actor"))
