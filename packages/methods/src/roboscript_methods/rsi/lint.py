"""The RSI document's lint (docs/concepts/learning.md), on what an Actor reads of an
entry (`when`, `do`, `check`), never on its evidence, which names tasks and episodes by design.

Errors (an edit leaving a new one is refused):
- `global_task`: a GLOBAL entry names a task of the run (`LintContext.tasks`, the benchmark's
  task list);
- `global_object`: it names an object instance: a name the run lists (`LintContext.objects`), or
  an id-like word, a name with a number appended (`obj_3`, `block_2`), as the scene registry
  writes instances;
- `global_number`: it holds a digit;
- `do_number`: a `do` holds a number, in any part (numeric parameters are the Task Memory's);
- the recipe's vocabulary rules (`recipes/rsi/lint.yaml`, `RsiLintConfig`): `everywhere` in every
  part (seeds, coordinates, the evaluator's internals), `global_only` in GLOBAL.

Warning (flagged, never refused): `memory_conflict`, an entry naming what the Task Memory sets,
a parameter or a workflow stage of the Memory of its task (of any task, for GLOBAL): the RSI
document holds no parameter, and departs from a step only as its correction (`corrects`), which
the reviewer accepts only on evidence that the step failed and the correction worked.
Those names come from the `MemoryDoc` or, read from records, from the Actor's view of it
(`memory_before.md`,
`memory_names`).
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, ConfigDict

from roboscript.contracts.ids import TaskName
from roboscript.memory import MemoryDoc
from roboscript.memory.linter import VocabularyRule
from roboscript_methods.rsi.doc import RsiDoc, RsiEntry
from roboscript_methods.rsi.recipe import RsiLintConfig

WARNINGS: Final[frozenset[str]] = frozenset({"memory_conflict"})

INSTANCE: Final = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d+\b")
DIGIT: Final = re.compile(r"\d")
NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")
_STAGE: Final = re.compile(r"^### \d+\. (?P<name>\S+)\s*$")
_PARAM: Final = re.compile(r"^- (?P<name>[A-Za-z_][A-Za-z0-9_]*) = ")
_PARAMS: Final = "## PARAMETERS"


@dataclass(frozen=True)
class MemoryNames:
    """What a Task Memory sets, by name: its parameters and its workflow stages."""

    params: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()

    @classmethod
    def of(cls, doc: MemoryDoc) -> "MemoryNames":
        return cls(params=tuple(doc.params), stages=tuple(s.name for s in doc.workflow))


def memory_names(text: str) -> MemoryNames:
    """The names a Task Memory sets, read from the Actor's view of it (`roboscript.memory.view`
    with `detail=False`, recorded as `memory_before.md`): `### N. <stage>` headings, and the
    `- <name> = ...` lines of its PARAMETERS section. The view does not carry the parameters'
    values back into a `MemoryDoc`; the lint needs only the names."""
    params: list[str] = []
    stages: list[str] = []
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line.split("  [")[0].strip()
        elif match := _STAGE.match(line):
            stages.append(match["name"])
        elif section == _PARAMS and (match := _PARAM.match(line)):
            params.append(match["name"])
    return MemoryNames(params=tuple(params), stages=tuple(stages))


class LintItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: str
    rule: str
    match: str
    message: str

    @property
    def error(self) -> bool:
        return self.rule not in WARNINGS


@dataclass(frozen=True)
class LintContext:
    """What the lint knows of the run: the recipe's vocabulary, its tasks (from the benchmark),
    the object instance names it lists, and the Task Memory of each task (what an entry must not
    set: the document or its names)."""

    vocabulary: RsiLintConfig
    tasks: Collection[str]
    objects: Collection[str] = ()
    memories: Mapping[TaskName, MemoryDoc | MemoryNames] = field(default_factory=dict)


def memory_stages(context: LintContext, task: TaskName) -> tuple[str, ...]:
    """The workflow steps of `task`'s Task Memory, as the run knows them (none: unknown)."""
    memory = context.memories.get(task)
    if memory is None:
        return ()
    return (memory if isinstance(memory, MemoryNames) else MemoryNames.of(memory)).stages


def lint(doc: RsiDoc, context: LintContext) -> list[LintItem]:
    """Every finding, entry by entry in document order; retired entries are not checked."""
    items: list[LintItem] = []
    for task, entry in doc.entries():
        if not entry.readable:
            continue
        items += lint_entry(entry, task, context)
    return items


def lint_entry(entry: RsiEntry, task: TaskName | None, context: LintContext) -> list[LintItem]:
    """`entry`'s findings as an entry of GLOBAL (`task` None) or of `task`'s part."""
    items = _vocabulary(entry, context.vocabulary.everywhere)
    if number := NUMBER.search(entry.do):
        items.append(
            _item(entry, "do_number", number.group(), "a behavior holds no numeric parameter")
        )
    if task is None:
        items += _global(entry, context)
        items += _vocabulary(entry, context.vocabulary.global_only)
    memories = context.memories.values() if task is None else _one(context.memories, task)
    return items + _conflicts(entry, memories)


def _item(entry: RsiEntry, rule: str, match: str, message: str) -> LintItem:
    return LintItem(entry=entry.id, rule=rule, match=match, message=message)


def _vocabulary(entry: RsiEntry, rules: Mapping[str, VocabularyRule]) -> list[LintItem]:
    items = []
    for name in sorted(rules):
        regex = rules[name].regex()
        if regex is not None and (found := regex.search(entry.text)):
            items.append(_item(entry, name, found.group(), rules[name].message))
    return items


def _one(
    memories: Mapping[TaskName, MemoryDoc | MemoryNames], task: TaskName
) -> list[MemoryDoc | MemoryNames]:
    return [memories[task]] if task in memories else []


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9_]+", " ", text.lower()) + " "


def _names(text: str, names: Collection[str]) -> list[str]:
    """`names` found in `text` as words, also with their underscores read as spaces."""
    words = _words(text)
    found = []
    for name in names:
        lowered = name.lower()
        if f" {lowered} " in words or f" {lowered.replace('_', ' ')} " in words:
            found.append(name)
    return sorted(found)


def _global(entry: RsiEntry, context: LintContext) -> list[LintItem]:
    text = entry.text
    items = [
        _item(entry, "global_task", name, "GLOBAL holds experience across tasks: name no task")
        for name in _names(text, context.tasks)
    ]
    instances = sorted({*_names(text, context.objects), *INSTANCE.findall(text.lower())})
    items += [
        _item(entry, "global_object", name, "GLOBAL names kinds of things, not object instances")
        for name in instances
    ]
    if digit := DIGIT.search(text):
        items.append(_item(entry, "global_number", digit.group(), "GLOBAL holds no numbers at all"))
    return items


def _conflicts(entry: RsiEntry, memories: Collection[MemoryDoc | MemoryNames]) -> list[LintItem]:
    known = [m if isinstance(m, MemoryNames) else MemoryNames.of(m) for m in memories]
    names = {name for m in known for name in (*m.params, *m.stages)}
    return [
        _item(
            entry,
            "memory_conflict",
            name,
            "the Task Memory sets this (a parameter or workflow stage) and wins over it",
        )
        for name in _names(entry.text, names)
    ]
