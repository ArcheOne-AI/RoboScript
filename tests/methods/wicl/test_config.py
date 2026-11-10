"""The learning configs: `configs/learning.example.yaml`, the recipe's `protocols.yaml` against
each protocol's Config, the protocol and brain entry points."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from roboscript.agent import ResponsesBrain, ScriptedBrain
from roboscript.learning import (
    BrainSpec,
    Evaluate,
    LearningConfig,
    brain_factory,
    load_protocol,
    load_protocol_recipe,
)
from roboscript.settings import load_config_file
from roboscript_methods.wicl.imitate import Imitate
from roboscript_methods.wicl.practice import Practice, PracticeConfig

REPO = Path(__file__).resolve().parents[3]


def test_the_example_config_validates() -> None:
    config = load_config_file(REPO / "configs" / "learning.example.yaml", LearningConfig)
    assert config.run_dir.is_absolute() and config.recipe.name == "wicl"
    assert set(config.brains) == {"actor", "reviser", "demo_reader"}
    assert len({spec.config["model"] for spec in config.brains.values()}) == 1
    assert {spec.name for spec in config.brains.values()} == {"responses"}
    with pytest.raises(ValidationError):
        LearningConfig.model_validate({"run_dir": "x", "max_infra_reruns": -1})


def test_the_recipe_parameters_fit_the_protocols(tmp_path: Path) -> None:
    recipe = load_protocol_recipe(REPO / "recipes" / "wicl")
    assert recipe.roles["actor"].budgets.max_turns == 45
    assert recipe.roles["reviser"].limits.max_model_calls == 40
    assert set(recipe.protocols) == {"imitate", "practice", "evaluate"}
    for protocol in (Imitate(), Practice(), Evaluate()):
        extra = {"demo": str(tmp_path)} if protocol.name == "imitate" else {}
        protocol.Config.model_validate({**recipe.protocols[protocol.name], **extra})
    practice = PracticeConfig.model_validate(recipe.protocols["practice"])
    assert (practice.first_seed, practice.max_seeds) == (10000, 60)
    assert (practice.stop_passes, practice.stop_window) == (3, 5)
    raw = yaml.safe_load((REPO / "recipes" / "wicl" / "protocols.yaml").read_text(encoding="utf-8"))
    assert set(raw) == {"roles", "protocols"}


def test_protocols_and_brains_come_from_their_entry_points() -> None:
    assert {load_protocol(name).name for name in ("imitate", "practice", "evaluate")} == {
        "imitate",
        "practice",
        "evaluate",
    }
    assert isinstance(brain_factory(BrainSpec())(None), ResponsesBrain)
    spec = BrainSpec(name="scripted", config={"replies": [[{"name": "finish"}]]})
    assert isinstance(brain_factory(spec)(None), ScriptedBrain)
    with pytest.raises(LookupError, match="nothing named 'nope'"):
        brain_factory(BrainSpec(name="nope"))
    with pytest.raises(LookupError, match=r"roboscript\.protocols"):
        load_protocol("nope")
