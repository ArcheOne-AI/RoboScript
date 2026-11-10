"""Memory settings: the recipe's method content, and the component config.

A recipe directory (e.g. `recipes/wicl/`) holds `memory.yaml` (the edit rules of each phase
that edits Memory, under the phase's name, e.g. wicl's `s1` and `s2`, and the method's
`schedule` section) and `memory_lint.yaml` (the linter's vocabularies and limits);
`load_recipe` reads both. `MemoryConfig` (`configs/memory.example.yaml`) names the store and the
recipe. Relative paths are relative to the repository root.
"""

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, model_validator

from roboscript.contracts.ids import PhaseName
from roboscript.memory.calls import ToolSig
from roboscript.memory.edits import EditRules
from roboscript.memory.linter import LintConfig
from roboscript.settings import load_config_file, read_mapping, repo_root

RECIPE_FILE: Final = "memory.yaml"
LINT_FILE: Final = "memory_lint.yaml"
SCHEDULE: Final = "schedule"


class MemoryRecipe(BaseModel):
    """The edit rules of each phase that edits Memory, by the method's phase name (`stages`; a
    method may let one phase edit freely and hold another to costs and revision limits), the
    linter's configuration, and `schedule`, the method's own schedule section as written, which
    the core carries and the method reads and validates (e.g.
    `roboscript_methods.wicl.memory.schedule`; empty: the method's default)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stages: dict[PhaseName, EditRules] = {}
    schedule: dict[str, Any] = {}
    lint: LintConfig = LintConfig()

    def with_tools(self, tools: Mapping[str, ToolSig]) -> "MemoryRecipe":
        """The recipe with the tools a call template may call (`tools.tool_signatures`): the
        linter checks the calls Memories write against them."""
        return self.model_copy(update={"lint": self.lint.model_copy(update={"tools": dict(tools)})})

    def rules(self, stage: PhaseName) -> EditRules:
        """The rules an `Editor` of phase `stage` enforces; a phase the recipe gives none is an
        error."""
        found = self.stages.get(stage)
        if found is None:
            raise ValueError(
                f"the recipe's {RECIPE_FILE} gives no edit rules for phase {stage!r} "
                f"(it gives {sorted(self.stages)})"
            )
        return found

    def check_phases(self, phases: Collection[PhaseName]) -> None:
        """Raise `ValueError` for a key of `memory.yaml` that names none of the method's `phases`
        (every key but `schedule` is a phase's name, so a misspelt `schedule` lands here)."""
        if unknown := sorted(set(self.stages) - set(phases)):
            raise ValueError(
                f"the recipe's {RECIPE_FILE} has keys {unknown} that name no phase of the method "
                f"({sorted(phases)}); its keys are phases' names and {SCHEDULE!r}"
            )


class _RecipeFile(BaseModel):
    """`memory.yaml`: `schedule`, and every other key a phase's edit rules."""

    model_config = ConfigDict(extra="forbid")

    stages: dict[PhaseName, EditRules] = {}
    schedule: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _by_phase(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stages = {key: value for key, value in data.items() if key != SCHEDULE}
        return {"stages": stages, **({SCHEDULE: data[SCHEDULE]} if SCHEDULE in data else {})}


def overlaid(base: Any, overlay: Any) -> Any:
    """Lint config values with a benchmark's over them: mappings key by key, a list's words and
    patterns the overlay's and then the base's, anything else the overlay's."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        kept = {k: overlaid(v, overlay[k]) if k in overlay else v for k, v in base.items()}
        return {**kept, **{k: v for k, v in overlay.items() if k not in base}}
    if isinstance(base, list) and isinstance(overlay, list):
        return [*overlay, *base]
    return overlay


def load_recipe(directory: Path, lint_overlay: Path | None = None) -> MemoryRecipe:
    """The recipe in `directory`: `memory.yaml` and `memory_lint.yaml`, the latter under a
    benchmark's vocabulary `lint_overlay` when given (`overlaid`; relative: to the repository
    root). Without one, the generic rules alone."""
    directory = directory if directory.is_absolute() else repo_root() / directory
    main = load_config_file(directory / RECIPE_FILE, _RecipeFile)
    values = read_mapping(directory / LINT_FILE)
    if lint_overlay is not None:
        path = lint_overlay if lint_overlay.is_absolute() else repo_root() / lint_overlay
        values = overlaid(values, read_mapping(path))
    lint = LintConfig.model_validate(values)
    return MemoryRecipe(stages=main.stages, schedule=main.schedule, lint=lint)


class StoreSpec(BaseModel):
    """A store in `roboscript.memory_stores` and its options (the file store's: `root`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "file"
    options: dict[str, Any] = {}


class MemoryConfig(BaseModel):
    """`configs/memory.yaml`: where Memory versions go, and the recipe directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store: StoreSpec = StoreSpec()
    recipe: Path
