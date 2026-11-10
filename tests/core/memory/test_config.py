"""configs/memory.example.yaml against MemoryConfig: a misspelt key names no phase of the method."""

import shutil
from pathlib import Path

import pytest

from roboscript.memory import (
    load_recipe,
)

REPO = Path(__file__).resolve().parents[3]


def test_a_misspelt_key_names_no_phase_of_the_method(tmp_path: Path) -> None:
    """Every key of `memory.yaml` but `schedule` is a phase's rules, so a misspelt key whose
    value passes as rules (`schedul: {}`) loads as a phase; the runner's start check
    (`check_phases`) names it and the method's phases. The recipe as shipped passes."""
    phases = ["s1", "s2", "p3", "dev"]
    load_recipe(REPO / "recipes" / "wicl").check_phases(phases)
    shutil.copytree(REPO / "recipes" / "wicl", tmp_path, dirs_exist_ok=True)
    main = tmp_path / "memory.yaml"
    main.write_text(main.read_text(encoding="utf-8") + "schedul: {}\n", encoding="utf-8")
    recipe = load_recipe(tmp_path)
    with pytest.raises(ValueError, match=r"keys \['schedul'\] that name no phase of the method"):
        recipe.check_phases(phases)
