"""P4's recipe (`recipes/rsi/`): the RSI document's rules (`rsi.yaml`: edit budgets by round,
field caps, token budgets of what an Actor reads, the dedup threshold, GLOBAL's evidence rule, the
retire rules), its lint vocabulary (`lint.yaml`) and the reflection's rules (`reflect.yaml`: what
the episode digest names, the reflection tools' caps, the review passes, each role's model-call
limit). `load_rsi_recipe` reads and validates them; the files are the only source of the values
(the models have no defaults). The roles' prompts are `prompts/<role>.md` beside them.
"""

import math
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from roboscript.agent.context import ContextBudget
from roboscript.memory.config import overlaid
from roboscript.memory.linter import VocabularyRule
from roboscript.settings import load_config_file, read_mapping, repo_root

RULES_FILE: Final = "rsi.yaml"
LINT_FILE: Final = "lint.yaml"
REFLECT_FILE: Final = "reflect.yaml"
PROMPTS_DIR: Final = "prompts"
CHARS_PER_TOKEN: Final[float] = ContextBudget.model_fields["chars_per_token"].default
"""The agent loop's estimate of prompt text (`agent.context.ContextBudget`)."""


class FieldCaps(BaseModel):
    """The longest `when`, `do` and `check` (characters) and the most evidence refs of an entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    when: PositiveInt
    do: PositiveInt
    check: PositiveInt
    evidence: PositiveInt


class RsiRules(BaseModel):
    """`global_edits` / `task_edits`: the edits a round accepts in GLOBAL and in each task's part,
    by round (index 0: round 1; later rounds keep the last value). An evaluated Actor reads GLOBAL
    and its task's entries as patches to apply (`doc.RsiDoc.context`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    global_edits: tuple[PositiveInt, ...] = Field(min_length=1)
    task_edits: tuple[PositiveInt, ...] = Field(min_length=1)
    caps: FieldCaps
    global_tokens: PositiveInt
    task_tokens: PositiveInt
    dedup_threshold: float = Field(gt=0, le=1)
    global_min_tasks: PositiveInt
    retire_min_counts: PositiveInt
    retire_idle_rounds: PositiveInt

    def budget(self, global_part: bool, round: int) -> int:
        """The edits round `round` (from 1) accepts in GLOBAL or in one task's part."""
        edits = self.global_edits if global_part else self.task_edits
        return edits[min(round, len(edits)) - 1]

    def tokens(self, text: str) -> int:
        """`text`'s estimated tokens, as the agent loop's context budget estimates prompt text."""
        return math.ceil(len(text) / CHARS_PER_TOKEN)


class RsiLintConfig(BaseModel):
    """Vocabulary rules by name: `everywhere` for every part, `global_only` for GLOBAL."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    everywhere: dict[str, VocabularyRule]
    global_only: dict[str, VocabularyRule] = {}


class RoleLimits(BaseModel):
    """The model calls one episode of each reflection role may make."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reflector: PositiveInt
    curator: PositiveInt
    reviewer: PositiveInt


class ReflectRules(BaseModel):
    """The reflection's rules (`reflect.yaml`). The digest names keyframes around the calls of
    `grasp_tools` and `place_tools` (at most `digest_actions` such calls, the first and the last
    halves), reads the done check (`done_tool`) as passed when its result starts with `done_pass`,
    groups refused `finish` calls by cause (`finish_refusals`: a cause's name -> the start of the
    refusal's text, after its error code; none matching: `other`), and lists at most
    `digest_failures` groups of failed calls with at most `digest_turns` turns each.
    `frames_per_call`: images one `rsi_frames` call returns; `page_chars`: the longest page of
    turns `rsi_episode` returns; `lessons`: the candidate GLOBAL lessons one reflector may stage
    (0: none); `arm_tool`: the tool whose calls choose an arm (the digest lists each with its
    answer and the turn's rationale); `arm_named`: a regular expression (any case) that finds an
    arm the instruction names; `passes`: review passes (the first review, then one revision of what
    it rejected); `consolidate`: whether a round ends with the consolidation step
    (`reflect._consolidate`); `limits`: each role's model calls per episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grasp_tools: tuple[str, ...]
    place_tools: tuple[str, ...]
    done_tool: str
    done_pass: str
    finish_refusals: dict[str, str]
    digest_actions: PositiveInt
    digest_failures: PositiveInt
    digest_turns: PositiveInt
    frames_per_call: PositiveInt
    page_chars: PositiveInt
    lessons: NonNegativeInt
    arm_tool: str
    arm_named: str
    passes: PositiveInt = Field(le=2)
    consolidate: bool
    limits: RoleLimits


class RsiRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: RsiRules
    lint: RsiLintConfig
    reflect: ReflectRules
    prompts: Path


def load_rsi_recipe(directory: Path, lint_overlay: Path | None = None) -> RsiRecipe:
    """`rsi.yaml`, `lint.yaml` and `reflect.yaml` in `directory` (relative: to the repository
    root), and its `prompts/` directory; `lint.yaml` under a benchmark's vocabulary `lint_overlay`
    when given (`roboscript.memory.config.overlaid`), else the generic rules alone."""
    directory = directory if directory.is_absolute() else repo_root() / directory
    prompts = directory / PROMPTS_DIR
    if not prompts.is_dir():
        raise ValueError(f"{directory} has no {PROMPTS_DIR}/ (the reflection roles' prompts)")
    lint = read_mapping(directory / LINT_FILE)
    if lint_overlay is not None:
        path = lint_overlay if lint_overlay.is_absolute() else repo_root() / lint_overlay
        lint = overlaid(lint, read_mapping(path))
    return RsiRecipe(
        rules=load_config_file(directory / RULES_FILE, RsiRules),
        lint=RsiLintConfig.model_validate(lint),
        reflect=load_config_file(directory / REFLECT_FILE, ReflectRules),
        prompts=prompts,
    )
