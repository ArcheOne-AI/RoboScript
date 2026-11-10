"""Fixtures shared by the core tests: the wicl recipe's Memory settings and a sample Memory."""

from pathlib import Path

import pytest

from roboscript.memory import MemoryDoc, MemoryRecipe, load_recipe
from roboscript.testing.memory import sample_memory

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def recipe() -> MemoryRecipe:
    return load_recipe(
        REPO / "recipes" / "wicl", REPO / "tests/core/memory/fixtures/lint_overlay.yaml"
    )


@pytest.fixture
def memory() -> MemoryDoc:
    return sample_memory()
