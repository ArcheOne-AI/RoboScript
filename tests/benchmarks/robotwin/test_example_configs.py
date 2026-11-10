"""The RoboTwin example's configs against the models that read them, and the benchmark's entry
point."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from roboscript.agent import ResponsesBrain
from roboscript.demos import DemosConfig
from roboscript.envs import load_benchmark
from roboscript.kernel import entrypoints
from roboscript.learning import brain_maker
from roboscript.learning.agent_host import AgentHostConfig, toolset_provider
from roboscript.perception.config import PerceptionConfig
from roboscript.settings import load_config_file
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import (
    RoboTwinConfig,
    with_perception_binding,
    with_perception_place,
)

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "examples/simBenchmarks/RoboTwin/configs"
EXAMPLE = CONFIGS / "demos.example.yaml"


def test_robotwin_loads_by_name_with_its_env_config_model() -> None:
    if "robotwin" not in entrypoints.names("roboscript.benchmarks"):
        pytest.skip("roboscript-robotwin is not installed here")
    robotwin = load_benchmark("robotwin")
    assert robotwin.config_model().__name__ == "RoboTwinConfig"
    assert robotwin.eval_seeds is not None and robotwin.eval_seeds.block(1) == (200000, 100000)
    assert robotwin.capabilities.demos


def test_the_example_config_is_valid_and_holds_the_defaults() -> None:
    config = DemosConfig.model_validate(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))
    assert set(config.keypoints.angle_deg_by_category) == {"C1", "C2", "C3", "C4", "C5"}
    defaults = config.keypoints.model_copy(update={"angle_deg_by_category": {}})
    # the defaults, and the settling only the benchmark's config sets
    assert config.disturbance.settle_steps == 400
    expected = DemosConfig(disturbance=config.disturbance)
    assert config.model_copy(update={"keypoints": defaults}) == expected
    with pytest.raises(ValidationError, match="min_move"):
        DemosConfig.model_validate(
            {"keypoints": {"min_move": 0.01}, "disturbance": {"settle_steps": 400}}
        )
    with pytest.raises(ValidationError, match="disturbance"):
        DemosConfig.model_validate({})


def test_the_example_config() -> None:
    config = load_config_file(
        CONFIGS / "agent_host.example.yaml",
        AgentHostConfig,
    )
    assert config.episodes.guards.wall_clock_s_per_step == 10.0 and config.port == 0
    assert config.blob_dir is None  # the stack's RS_BLOB_DIR
    names = ["sensing", "manipulation", "perception", "geometry", "worldstate", "done"]
    assert [toolset.name for toolset in config.toolsets] == names
    assert all(toolset_provider(spec).name == spec.name for spec in config.toolsets)
    # started by hand, the host perceives with RoboTwin's perception binding as the launcher's do
    configs = {spec.name: spec.config for spec in config.toolsets}
    bound = PerceptionConfig.model_validate
    place, binding = bound(with_perception_place({})), bound(with_perception_binding({}))
    assert bound(configs["manipulation"]["table_perception"]) == place
    for name in names[2:]:
        assert bound(configs[name]["pipeline"]) == binding
    assert config.recipe == Path("recipes/wicl")
    assert isinstance(brain_maker(config.brain)(), ResponsesBrain)


def test_the_env_config_example() -> None:
    example = CONFIGS / "robotwin_env.example.yaml"
    config = RoboTwinConfig.model_validate(yaml.safe_load(example.read_text(encoding="utf-8")))
    assert config.mode == "clean" and config.task == "beat_block_hammer"
    assert config.blob_dir is None  # RS_BLOB_DIR, the stack's


def _dispatch_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "dispatch_script", CONFIGS.parent / "scripts" / "dispatch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dispatch_example_checks(capsys: pytest.CaptureFixture[str]) -> None:
    """scripts/dispatch.py on configs/dispatch.example.yaml: the pipeline config every worker
    would run is valid, and tasks without a cost get their official step limit."""
    dispatch = _dispatch_script()
    config_file = CONFIGS / "dispatch.example.yaml"
    sets = ["pipeline=examples/simBenchmarks/RoboTwin/configs/pipeline.example.yaml"]
    config = dispatch.checked(config_file, sets, None)
    assert config.costs == {t: float(ROBOTWIN.tasks[t].step_limit) for t in config.tasks}
    assert config.unit.forced == {"gpu_cap_gb": 47.0, "gpu_margin_gb": 1.5}
    assert dispatch.main(["check", str(config_file), *(f"--set={s}" for s in sets)]) == 0
    assert capsys.readouterr().out.startswith("worker-a (1 GPUs, cost ")
