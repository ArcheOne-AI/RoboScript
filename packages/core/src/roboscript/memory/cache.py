"""Per-seed caches, their diffs and merges (docs/concepts/learning.md).

`open_cache(M, ...)` is an `Editor` over a copy of M with an empty SCRATCH and the seed's budget
B_k; only the Reviser edits it, and tries 2 and 3 read it without SCRATCH. `diff(a, b)` lists what
changed from `a` to `b`: parameters field by field, text by section, workflow stage and lesson (the
header, provenance and SCRATCH are not content). `merge(M, C, alpha)` folds a cache that made its
seed succeed into M: numbers interpolate, `v_M + alpha (v_C - v_M)` (to 1e-6, within bounds);
categories, text, the workflow and lessons take the cache's, and so does a spec whose categorical
or other non-numeric fields (a pose grasp's axes and orientation hint, `params.others`) changed,
whole (its numbers were tuned for them); SCRATCH is dropped; every parameter
change is recorded with M's, the cache's and the merged value. A cache equal to M is a jitter: no
new version.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from roboscript.memory.doc import STAGE_FIELDS, Lesson, MemoryDoc, content_hash
from roboscript.memory.edits import Editor, EditRules, EditValidator
from roboscript.memory.linter import LintConfig
from roboscript.memory.params import (
    Change,
    Param,
    bounds,
    categories,
    definition,
    jsonable,
    numbers,
    others,
    shape,
    with_field,
)


class ParamChange(BaseModel):
    """A parameter's change from one document to another; `field` names a spec's field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    field: str | None = None
    how: Literal["added", "removed", "redefined", "changed"]
    numeric: bool = False
    before: JsonValue = None
    after: JsonValue = None


class TextChange(BaseModel):
    """A change of text: `where` is a section, `workflow` (the order), `workflow.<stage>`,
    `workflow.<stage>.<field>` or `lessons.<id>`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    where: str
    how: Literal["added", "removed", "changed", "reordered"]
    before: JsonValue = None
    after: JsonValue = None


class MemoryDiff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    params: list[ParamChange] = []
    text: list[TextChange] = []

    @property
    def empty(self) -> bool:
        return not self.params and not self.text


class ParamMerge(BaseModel):
    """What a merge did to a parameter (field): M's value, the cache's and the result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    field: str | None = None
    how: Literal["interpolated", "replaced", "added", "removed", "redefined"]
    before: JsonValue = None
    cache: JsonValue = None
    after: JsonValue = None


class MergeRecord(BaseModel):
    """A merge's provenance: the seed and try whose cache it merged, the learning rate, the hashes
    of M and of the cache, whether it was a jitter, and every change."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int | None = None
    attempt: int | None = None
    alpha: float
    base: str
    cache: str
    jitter: bool
    params: list[ParamMerge] = []
    text: list[TextChange] = []


@dataclass(frozen=True)
class Merge:
    """The merged document (None for a jitter: M stays) and its record."""

    doc: MemoryDoc | None
    record: MergeRecord


def open_cache(
    memory: MemoryDoc,
    *,
    budget: int,
    seed: int,
    rules: EditRules,
    lint: LintConfig,
    validators: Sequence[EditValidator] = (),
) -> Editor:
    """The cache of seed `seed`: an `Editor` over a copy of `memory` with an empty SCRATCH and the
    seed's budget B_k, checking edits with the method's `validators`."""
    doc = memory.replace(version=None, created=None, source=f"cache of {memory.label}", scratch=[])
    return Editor(
        doc,
        rules=rules,
        lint=lint,
        budget=budget,
        seed=seed,
        source=f"seed {seed}",
        validators=validators,
    )


def diff(a: MemoryDoc, b: MemoryDoc) -> MemoryDiff:
    names = dict.fromkeys([*a.params, *b.params])
    params = [
        c for name in names for c in _param_diff(name, a.params.get(name), b.params.get(name))
    ]
    return MemoryDiff(params=params, text=_text_diff(a, b))


def render_diff(changes: MemoryDiff) -> str:
    """One line per change, for a model: parameters with their values, text by location."""
    lines = []
    for change in changes.params:
        where = change.param if change.field is None else f"{change.param}.{change.field}"
        if change.how == "changed":
            lines.append(f"- params.{where}: {_json(change.before)} -> {_json(change.after)}")
        else:
            lines.append(f"- params.{where}: {change.how}")
    lines += [f"- {change.where}: {change.how}" for change in changes.text]
    return "\n".join(lines) if lines else "(no changes)"


def merge(
    memory: MemoryDoc,
    cache: MemoryDoc,
    alpha: float,
    *,
    seed: int | None = None,
    attempt: int | None = None,
    base: str | None = None,
) -> Merge:
    """Merge `cache`, a copy of `memory` that made seed `seed` succeed on try `attempt`, with
    learning rate `alpha` in (0, 1]; `base`: `memory`'s hash in its store, if it is a version."""
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha {alpha} is not in (0, 1]")
    changes = diff(memory, cache)
    record = MergeRecord(
        seed=seed,
        attempt=attempt,
        alpha=alpha,
        base=base or content_hash(memory),
        cache=content_hash(cache),
        jitter=changes.empty,
    )
    if changes.empty:
        return Merge(None, record)
    params: dict[str, Param] = {}
    merged: list[ParamMerge] = []
    for name, theirs in cache.params.items():
        ours = memory.params.get(name)
        if ours is None:
            params[name] = theirs
            shown = definition(theirs)
            merged.append(ParamMerge(param=name, how="added", cache=shown, after=shown))
        elif shape(ours) != shape(theirs):
            history = [*ours.history, *theirs.history]
            params[name] = theirs.model_copy(update={"history": history})
            before, shown = definition(ours), definition(theirs)
            merged.append(
                ParamMerge(param=name, how="redefined", before=before, cache=shown, after=shown)
            )
        else:
            params[name], entries = _merge_param(name, ours, theirs, alpha, seed)
            merged += entries
    for name, ours in memory.params.items():
        if name not in cache.params:
            merged.append(ParamMerge(param=name, how="removed", before=definition(ours)))
    earlier = {lesson.id: lesson for lesson in memory.lessons}
    lessons = [_merge_lesson(earlier.get(lesson.id), lesson) for lesson in cache.lessons]
    origin = " ".join(
        part
        for part in (
            f"merge into {memory.label}",
            f"from seed {seed}" if seed is not None else "",
            f"try {attempt}" if attempt is not None else "",
        )
        if part
    )
    doc = cache.replace(
        version=None,
        stage="S2",
        created=None,
        source=origin,
        scratch=None,
        params=params,
        lessons=lessons,
    )
    return Merge(doc, record.model_copy(update={"params": merged, "text": changes.text}))


def _param_diff(name: str, a: Param | None, b: Param | None) -> list[ParamChange]:
    if a is None:
        return [ParamChange(param=name, how="added", after=definition(b))]  # type: ignore[arg-type]
    if b is None:
        return [ParamChange(param=name, how="removed", before=definition(a))]
    if shape(a) != shape(b):
        before, after = definition(a), definition(b)
        return [ParamChange(param=name, how="redefined", before=before, after=after)]
    changes = []
    for numeric, fields_a, fields_b in (
        (True, numbers(a), numbers(b)),
        (False, categories(a), categories(b)),
        (False, others(a), others(b)),
    ):
        for field, before in fields_a.items():
            if before != (after := fields_b[field]):
                change = ParamChange(
                    param=name,
                    field=field,
                    how="changed",
                    numeric=numeric,
                    before=jsonable(before),
                    after=jsonable(after),
                )
                changes.append(change)
    return changes


def _text_diff(a: MemoryDoc, b: MemoryDoc) -> list[TextChange]:
    changes = [
        TextChange(
            where=section, how="changed", before=getattr(a, section), after=getattr(b, section)
        )
        for section in ("statement", "objects", "notes")
        if getattr(a, section) != getattr(b, section)
    ]
    ours = {stage.name: stage for stage in a.workflow}
    theirs = {stage.name: stage for stage in b.workflow}
    changes += [
        TextChange(where=f"workflow.{name}", how="removed", before=stage.model_dump())
        for name, stage in ours.items()
        if name not in theirs
    ]
    for name, stage in theirs.items():
        if name not in ours:
            changes.append(
                TextChange(where=f"workflow.{name}", how="added", after=stage.model_dump())
            )
            continue
        for field in STAGE_FIELDS:
            if (before := getattr(ours[name], field)) != (after := getattr(stage, field)):
                where = f"workflow.{name}.{field}"
                changes.append(TextChange(where=where, how="changed", before=before, after=after))
    order_a: list[JsonValue] = [name for name in ours if name in theirs]
    order_b: list[JsonValue] = [name for name in theirs if name in ours]
    if order_a != order_b:
        changes.append(TextChange(where="workflow", how="reordered", before=order_a, after=order_b))
    earlier = {lesson.id: lesson for lesson in a.lessons}
    later = {lesson.id: lesson for lesson in b.lessons}
    for lesson_id, lesson in earlier.items():
        if lesson_id not in later:
            changes.append(
                TextChange(where=f"lessons.{lesson_id}", how="removed", before=_rule(lesson))
            )
    for lesson_id, lesson in later.items():
        before = earlier.get(lesson_id)
        if before is None:
            changes.append(
                TextChange(where=f"lessons.{lesson_id}", how="added", after=_rule(lesson))
            )
        elif _rule(before) != _rule(lesson):
            where = f"lessons.{lesson_id}"
            changes.append(
                TextChange(where=where, how="changed", before=_rule(before), after=_rule(lesson))
            )
    return changes


def _rule(lesson: Lesson) -> dict[str, JsonValue]:
    return {"when": lesson.when, "do": lesson.do}


def _json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False)


def _merge_param(
    name: str, ours: Param, theirs: Param, alpha: float, seed: int | None
) -> tuple[Param, list[ParamMerge]]:
    """Numbers interpolate. A category the cache changed is taken, and so is a whole spec whose
    categorical fields or other non-numeric fields (a pose grasp's axes, `params.others`) changed:
    its numbers were tuned for them."""
    if categories(ours) != categories(theirs) or others(ours) != others(theirs):
        before, cached = jsonable(ours.value), jsonable(theirs.value)
        change = Change(how="merge", before=before, after=cached, seed=seed)
        merged = ParamMerge(param=name, how="replaced", before=before, cache=cached, after=cached)
        return with_field(ours, None, theirs.value, change), [merged]
    param, entries = ours, []
    for field, before in numbers(ours).items():
        if before == (cached := numbers(theirs)[field]):
            continue
        after = _interpolate(before, cached, alpha, bounds(ours, field))
        shown = (jsonable(before), jsonable(cached), jsonable(after))
        change = Change(
            how="merge", field=field, before=shown[0], after=shown[2], seed=seed, alpha=alpha
        )
        param = with_field(param, field, after, change)
        entries.append(
            ParamMerge(
                param=name,
                field=field,
                how="interpolated",
                before=shown[0],
                cache=shown[1],
                after=shown[2],
            )
        )
    return param, entries


def _interpolate(before: Any, cached: Any, alpha: float, limits: tuple[float, float]) -> Any:
    if alpha == 1:
        return cached

    def one(ours: float, theirs: float) -> float:
        return min(max(round(ours + alpha * (theirs - ours), 6), limits[0]), limits[1])

    if isinstance(before, tuple):
        return tuple(one(ours, theirs) for ours, theirs in zip(before, cached, strict=True))
    return one(before, cached)


def _merge_lesson(ours: Lesson | None, theirs: Lesson) -> Lesson:
    """A new lesson keeps the cache's seed; a changed one its first seed; both count a merge."""
    if ours is None:
        return theirs.model_copy(update={"merges": theirs.merges + 1})
    if _rule(ours) == _rule(theirs):
        return ours
    return theirs.model_copy(update={"seed": ours.seed, "merges": ours.merges + 1})
