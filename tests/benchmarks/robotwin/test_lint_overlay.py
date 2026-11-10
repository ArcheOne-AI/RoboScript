"""RoboTwin's lint vocabulary lives in its own overlays: no overlay is the generic rules, and the
run configs name the overlays."""

from pathlib import Path

import yaml

from roboscript.memory import load_recipe
from roboscript.settings import repo_root
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_robotwin.config import (
    LINT_OVERLAYS,
)

ROOT = repo_root()
BEFORE = Path(__file__).parent / "data" / "pre_split_lint"
RUN = ROOT / "examples" / "simBenchmarks" / "RoboTwin" / "run"


def _before(name: str) -> dict:
    return yaml.safe_load((BEFORE / name).read_text("utf-8"))


def test_no_overlay_is_the_generic_rules() -> None:
    memory = load_recipe(ROOT / "recipes" / "wicl").lint
    rsi = load_rsi_recipe(ROOT / "recipes" / "rsi").lint
    text = repr((memory.model_dump(), rsi.model_dump())).lower()
    for word in ("robotwin", "sapien", "mplib", "check_success", "play_once", "functional_point"):
        assert word not in text
    assert memory.object_words == [] and rsi.global_only == {}
    assert memory.rules["seed"].patterns and rsi.everywhere["seed"].patterns


def test_the_run_configs_name_the_overlays() -> None:
    pipeline = yaml.safe_load((RUN / "wicl" / "pipeline.yaml").read_text("utf-8"))
    assert pipeline["learning"]["lint_overlay"] == LINT_OVERLAYS["memory"]
    p4 = yaml.safe_load((RUN / "rsi" / "p4.yaml").read_text("utf-8"))
    assert p4["lint_overlay"] == LINT_OVERLAYS["rsi"]
