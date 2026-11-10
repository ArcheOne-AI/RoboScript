"""The layered config of the RoboTwin wicl run (docs/concepts/operations.md): P3's Actor brain
must be the S2 one, no tasks named runs every task, and the brains name their models' context
windows."""

from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from robotwin_pipeline import compose_config

from roboscript.settings import repo_root

HERE: Final = Path(__file__).resolve().parent
EXAMPLE: Final = str(
    repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "pipeline.example.yaml"
)
ADDED: Final = ("binding", "protocol_recipe", "seeds")
"""Keys the layered config adds: where it came from, the composed protocol recipe, the seeds."""


def _before(name: str) -> dict[str, Any]:
    values = yaml.safe_load((HERE / "pre_split" / name).read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _same(composed: Any, before: Any, where: str = "") -> None:
    """Equal key by key, naming the first key that differs."""
    if isinstance(before, dict):
        assert isinstance(composed, dict), where
        assert sorted(composed) == sorted(before), f"{where}: keys {sorted(composed)}"
        for key in before:
            _same(composed[key], before[key], f"{where}.{key}")
    else:
        assert composed == before, f"{where}: {composed!r} != {before!r}"


def test_p3s_actor_brain_must_be_s2s() -> None:
    """P3's launcher config holds the Actor's brain outside the layers: the example's equals the
    recipe's Actor, another is refused unless `p3.actor_brain_differs`."""
    from robotwin_pipeline.schedule import check_actor_brain

    from roboscript.learning import LearningConfig
    from roboscript.learning.config import BrainSpec

    config = compose_config(EXAMPLE)
    learning = LearningConfig.model_validate({**config.learning, "run_dir": config.root})
    p3 = repo_root() / "examples/simBenchmarks/RoboTwin/configs/p3.example.yaml"
    host = yaml.safe_load(p3.read_text(encoding="utf-8"))["host"]
    check_actor_brain(config, learning, BrainSpec.model_validate(host["brain"]))
    low = BrainSpec(config={"model": "gpt-5.6-luna", "reasoning_effort": "low"})
    with pytest.raises(ValueError, match=r"differs from learning\.brains\.actor"):
        check_actor_brain(config, learning, low)
    allowed = compose_config(EXAMPLE, ["p3.actor_brain_differs=true"])
    check_actor_brain(allowed, learning, low)


def test_no_tasks_runs_every_task() -> None:
    """The binding names no tasks: the run is every task of the benchmark."""
    from roboscript_robotwin.benchmark import ROBOTWIN

    tasks = compose_config(EXAMPLE).tasks
    assert len(tasks) == len(ROBOTWIN.tasks) == 50 and set(tasks) == set(ROBOTWIN.tasks)


def test_the_models_context_windows_are_explicit() -> None:
    """Every brain of the binding, P3's host brain and the example Agent host's name their
    models' context windows themselves, not through the budget's defaults."""
    windows = {
        "brain-default": 272000,
        "reviser": 272000,
        "gpt-5.6-luna": 272000,
        "gpt-6.1-sol": 1000000,
    }
    config = compose_config(EXAMPLE)
    for role, brain in config.learning["brains"].items():
        assert brain["config"]["context"]["windows"] == windows, role
    configs = repo_root() / "examples/simBenchmarks/RoboTwin/configs"
    for name in ("p3.example.yaml", "agent_host.example.yaml"):
        values = yaml.safe_load((configs / name).read_text(encoding="utf-8"))
        host = values.get("host", values)
        assert host["brain"]["config"]["context"]["windows"] == windows, name
