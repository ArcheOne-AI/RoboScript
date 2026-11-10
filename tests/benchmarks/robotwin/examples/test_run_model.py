"""A run's model (`PipelineConfig.model`): every model call of a run is the run's model by default.
Set, it reaches every brain of `learning.brains`, P3's Agent host brain and the scene toolsets'
VLM of every stage."""

from typing import Final

from robotwin_pipeline import compose_config
from robotwin_pipeline.run import run_models
from robotwin_pipeline.schedule import check_actor_brain

from roboscript.learning import LearningConfig
from roboscript.settings import load_config_file, repo_root
from roboscript_robotwin.eval.launch import P3Config, scene_names

EXAMPLE: Final = str(
    repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "pipeline.example.yaml"
)
P3: Final = repo_root() / "examples/simBenchmarks/RoboTwin/configs/p3.example.yaml"
SOL: Final = "gpt-6.1-sol"


def _launcher() -> P3Config:
    return load_config_file(P3, P3Config)


def _scene_models(launcher: P3Config) -> set[str | None]:
    scene = scene_names(spec.name for spec in launcher.toolsets())
    assert scene  # the example mounts scene toolsets, or the check below says nothing
    return {
        (spec.get("config") or {}).get("vlm", {}).get("model")
        for spec in launcher.host["toolsets"]
        if spec["name"] in scene
    }


def test_the_run_model_reaches_every_brain_p3_and_the_vlm() -> None:
    before = compose_config(EXAMPLE)
    config = compose_config(EXAMPLE, [f"model={SOL}"])
    for role, brain in config.learning["brains"].items():
        was = before.learning["brains"][role]
        assert brain["config"] == {**was["config"], "model": SOL}, role  # effort etc. kept
    assert config.vlm_model == SOL
    launcher = run_models(config, _launcher())
    assert launcher.host["brain"]["config"]["model"] == SOL
    assert _scene_models(launcher) == {SOL}
    learning = LearningConfig.model_validate({**config.learning, "run_dir": config.root})
    from roboscript.learning.config import BrainSpec

    check_actor_brain(config, learning, BrainSpec.model_validate(launcher.host["brain"]))
