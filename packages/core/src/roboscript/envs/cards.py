"""Success condition cards (docs/concepts/embodied-core.md): per task, a short text
derived from the benchmark's official success check that says what success requires at the end
(which objects, which relations, orientation, containment or order, qualitative tolerances). A
card is part of the task's definition, as its instruction is: every role reads it right under the
instruction (`task_text`). It is data, never a scene's: no per-seed numbers, no poses, nothing of
the simulator's state (`check_card`).

A benchmark keeps its cards in one directory (`Benchmark.cards`), one file per task,
`<task>.md`: a YAML front matter (`CardMeta`) between `---` lines, then the card's text. A task
without a file has no card, and its prompts and records hold none.

    ---
    task: <task>
    version: 1
    source: <benchmark> <commit> <task> run_reward
    source_sha256: <sha256 of the success check's source text>
    ---
    Each item hangs from the rack on a hook: it rests on the hook with nothing holding it, not
    merely touching the rack.
"""

import hashlib
import re
from collections.abc import Collection
from pathlib import Path
from typing import Annotated, Final

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, StringConstraints, ValidationError

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import TaskName

SUFFIX: Final = ".md"
MAX_CHARS: Final = 1200
"""A card's text at most: a few sentences, not a manual."""
MAX_COUNT: Final = 12
"""The largest integer a card may write (a count or a position in an order)."""
LEAD: Final = (
    "Success conditions (the benchmark's own success check, the same on every scene; what must "
    "hold when the task ends):"
)
"""The line that introduces a card under the instruction."""

_FENCE: Final = "---"
_NUMBER: Final = re.compile(r"(?:(?<![\w-])-)?\d+(?:[.,]\d+)*")  # "1-2" is no negative
_UNIT: Final = re.compile(
    r"\d\s*(?:°|(?:mm|cm|m|deg|degrees?|rad|radians?|%|s|ms|seconds?|kg|g|n)\b)", re.IGNORECASE
)
_TUPLE: Final = re.compile(r"[(\[{][^)\]}]*\d\s*,\s*-?\d[^)\]}]*[)\]}]")
_STATE: Final = re.compile(
    r"/World\w*|\bprims?\b|\bseeds?\b|\bquaternions?\b|\bxyz\b|\benv_\w*|\w+\.(?:usd|py)\b",
    re.IGNORECASE,
)
_CODE: Final = re.compile(r"```|==|!=|<=|>=|&&|\|\||\bdef |\bimport |\blambda\b|\bself\.|\b\w+\(")


class CardMeta(BaseModel):
    """A card's front matter: its `task` (the file's name), its `version` (raised whenever its
    text changes), where it was derived from (`source`: the benchmark's revision and success
    check) and that source's sha256 (`source_sha256`), so a card can be told stale when the check
    changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    version: PositiveInt
    source: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class TaskCard(CardMeta):
    """A task's success condition card: its front matter and `text`."""

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    @property
    def sha256(self) -> str:
        """Of the whole card, front matter and text (its records name it by this)."""
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()

    def record(self) -> dict[str, object]:
        """What an episode's records say of the card it showed (`task.card`, `config.json`)."""
        return {
            "task": self.task,
            "version": self.version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "sha256": self.sha256,
            "text": self.text,
        }


def check_card(text: str) -> list[str]:
    """What keeps `text` from being a card: too long; a number that is not a small integer (a
    decimal, a negative one, one over `MAX_COUNT`), a number with a unit or a tuple of numbers
    (a position, a pose, a tolerance or a scene's value); a word of the simulator's state (a prim
    path, a seed, a quaternion, a source file); code (a fence, an operator, a call). Empty: it
    is one. Words carry tolerances ("fully inside", "upright", "not merely touching")."""
    problems: list[str] = []
    if len(text) > MAX_CHARS:
        problems.append(f"{len(text)} characters, more than {MAX_CHARS}")
    numbers = [n for n in _NUMBER.findall(text) if not n.isdigit() or int(n) > MAX_COUNT]
    if numbers:
        problems.append(f"numbers {sorted(set(numbers))}: only counts and ranks, 0 to {MAX_COUNT}")
    if found := _UNIT.findall(text):
        problems.append(f"numbers with units {sorted(set(found))}: write tolerances in words")
    if found := _TUPLE.findall(text):
        problems.append(f"tuples of numbers {sorted(set(found))}: write no positions or poses")
    if found := _STATE.findall(text):
        problems.append(f"simulator state {sorted(set(found))}: say what holds, not the state")
    if found := _CODE.findall(text):
        problems.append(f"code {sorted(set(found))}: a card is plain text")
    return problems


def read_card(path: Path) -> TaskCard:
    """The card in `path` (`<task>.md`), checked (`check_card`); ValueError naming the file
    otherwise."""
    content = path.read_text(encoding="utf-8")
    head, sep, body = content.removeprefix(_FENCE + "\n").partition("\n" + _FENCE + "\n")
    if not content.startswith(_FENCE + "\n") or not sep:
        raise ValueError(f"{path}: a card starts with front matter between {_FENCE!r} lines")
    try:
        meta = yaml.safe_load(head)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(meta, dict) or "text" in meta:
        raise ValueError(
            f"{path}: its front matter is not a mapping of task, version, source and "
            "source_sha256 (the text goes below it)"
        )
    try:
        card = TaskCard.model_validate({**meta, "text": body})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    if card.task != path.name.removesuffix(SUFFIX):
        raise ValueError(f"{path}: its card is of {card.task!r}, not the file's task")
    if problems := check_card(card.text):
        raise ValueError(f"{path}: not a success condition card: {'; '.join(problems)}")
    return card


def load_cards(
    directory: Path, tasks: Collection[TaskName] | None = None
) -> dict[TaskName, TaskCard]:
    """The cards in `directory` by task (`<task>.md`; a task without a file has none); ValueError
    for a missing directory, a file of no task in `tasks` (None: any name, as the Agent host
    reads its benchmark's directory, which has tasks its manifest has not), or a card that does
    not check."""
    if not directory.is_dir():
        raise ValueError(f"{directory}: no such directory of success condition cards")
    cards: dict[TaskName, TaskCard] = {}
    for path in sorted(directory.glob(f"*{SUFFIX}")):
        if tasks is not None and path.name.removesuffix(SUFFIX) not in tasks:
            raise ValueError(f"{path}: the benchmark has no task {path.stem!r}")
        card = read_card(path)
        cards[card.task] = card
    return cards


def task_text(instruction: str, card: TaskCard | None) -> str:
    """The task as a role reads it: the instruction, then the card under `LEAD` (none: the
    instruction as it is)."""
    if card is None:
        return instruction
    return f"{instruction}\n\n{LEAD}\n{card.text}"


def shown(card: TaskCard | None, prompt: str) -> TaskCard | None:
    """`card` if `prompt` shows it (`task_text`'s block is in it), else None: a record names a
    card only where the model read it."""
    return card if card is not None and f"{LEAD}\n{card.text}" in prompt else None
