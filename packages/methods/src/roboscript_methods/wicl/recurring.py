"""Edits a practice's Reviser makes again and again on seeds it then skips
(docs/concepts/learning.md, `PracticeConfig.recurring`).

A seed whose three tries all fail is skipped and its cache discarded, so an edit the Reviser
re-derives on seed after seed never reaches M. `recurring` finds them: over the skipped seeds since
M's version, each parameter's change in the seed's last cache against M, grouped by sameness (the
same fields changed, numbers in the same direction, other values equal). Recurrence alone does not
make a group a merge candidate: the Reviser repeats its priors, and an edit made in five seeds can
be one that makes its call fail more often (each seed's first try failed, and the edit answers that
one scene). A group merges only on the receipts: it recurs in at least `seeds` skipped seeds, no
skipped seed changed the same fields another way, and the calls that passed the parameter by name
(`grasp={spec}`) went through with the group's value more often than with M's beyond noise (their
Wilson 95% intervals do not overlap), each side over at least `calls` calls of the window's tries.
A lower bound above M's rate alone would merge an edit on too few calls.
Parameters no call passes by name (numbers, done criteria) and text never merge here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue

from roboscript.learning.ledger import wilson
from roboscript.memory import MemoryDoc, ParamChange, diff
from roboscript.memory.params import definition, jsonable
from roboscript_methods.wicl.roles import spec_calls

SAME_VALUE: Final = 0.02
"""Values of a non-numeric field (axes, a quaternion) this close are the same."""
SAME_DIRECTION: Final = 0.9
"""Number changes whose directions' cosine is at least this go the same way."""
_THROUGH: Final = frozenset({"ok", "clamped"})


class Recurrence(BaseModel):
    """A parameter change made in several skipped seeds: the fields it changed, the seeds that made
    it and those that changed one of its fields another way (`against`); the receipts of the calls
    passing the parameter with its value (`with_ok` of `with_n`) and with M's (`base_ok` of
    `base_n`); whether it merges and why; `value`: the parameter as the latest seed left it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    fields: list[str]
    seeds: list[int]
    against: list[int]
    with_ok: int
    with_n: int
    base_ok: int
    base_n: int
    merge: bool
    why: str
    value: JsonValue


@dataclass(frozen=True)
class _Seen:
    seed: int
    changes: tuple[ParamChange, ...]
    doc: MemoryDoc


def _changes(base: MemoryDoc, doc: MemoryDoc, name: str) -> tuple[ParamChange, ...]:
    return tuple(c for c in diff(base, doc).params if c.param == name)


def _vector(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(x, int | float) for x in value)


def _same(a: ParamChange, b: ParamChange) -> bool:
    """Two changes of one field go the same way: numbers in the same direction, other values
    equal (vectors within `SAME_VALUE`)."""
    if a.field != b.field or a.how != "changed" or b.how != "changed":
        return False
    if a.numeric:
        da = np.ravel(np.subtract(a.after, a.before))  # type: ignore[arg-type]
        db = np.ravel(np.subtract(b.after, b.before))  # type: ignore[arg-type]
        norm = float(np.linalg.norm(da) * np.linalg.norm(db))
        return norm > 0 and float(da @ db) / norm >= SAME_DIRECTION
    if _vector(a.after) and _vector(b.after) and len(a.after) == len(b.after):  # type: ignore[arg-type]
        return bool(np.allclose(a.after, b.after, atol=SAME_VALUE))  # type: ignore[arg-type]
    return a.after == b.after  # categories, text


def _matches(a: Sequence[ParamChange], b: Sequence[ParamChange]) -> bool:
    """The same fields changed the same way."""
    by = {c.field: c for c in b}
    return len(a) == len(b) and all(c.field in by and _same(c, by[c.field]) for c in a)


def _conflicts(a: Sequence[ParamChange], b: Sequence[ParamChange]) -> bool:
    """A field both change, another way."""
    by = {c.field: c for c in b}
    return any(c.field in by and not _same(c, by[c.field]) for c in a)


def recurring(
    base: MemoryDoc,
    skipped: Sequence[tuple[int, MemoryDoc]],
    tries: Sequence[tuple[MemoryDoc, Path]],
    *,
    seeds: int = 3,
    calls: int = 3,
) -> list[Recurrence]:
    """The changes to `base` (M) that the last caches of `skipped` (seed, cache) share, in two
    seeds or more, most frequent first, with the receipts of `tries` (the Memory a try read and its
    directory: the window's tries, whatever their seed's outcome) and whether each merges."""
    names = dict.fromkeys(c.param for _, doc in skipped for c in diff(base, doc).params)
    found = []
    for name in names:
        if name not in base.params:
            continue  # added by a seed: not a change of M's
        seen = [_Seen(s, ch, doc) for s, doc in skipped if (ch := _changes(base, doc, name))]
        groups: list[list[_Seen]] = []
        for one in seen:
            group = next((g for g in groups if _matches(g[0].changes, one.changes)), None)
            if group is None:
                groups.append([one])
            else:
                group.append(one)
        reference = jsonable(definition(base.params[name]))
        for group in (g for g in groups if len(g) > 1):
            changes = group[0].changes
            against = sorted(one.seed for one in seen if _conflicts(changes, one.changes))
            counts = {"with": [0, 0], "base": [0, 0]}
            for doc, directory in tries:
                param = doc.params.get(name)
                if param is None:
                    continue
                if jsonable(definition(param)) == reference:
                    side = "base"
                elif _matches(changes, _changes(base, doc, name)):
                    side = "with"
                else:
                    continue
                for spec, reason in spec_calls(directory):
                    if spec == name:
                        counts[side][0] += reason in _THROUGH
                        counts[side][1] += 1
            (with_ok, with_n), (base_ok, base_n) = counts["with"], counts["base"]
            merge, why = _verdict(
                len(group), against, with_ok, with_n, base_ok, base_n, seeds, calls
            )
            found.append(
                Recurrence(
                    param=name,
                    fields=sorted(str(c.field) for c in changes),
                    seeds=sorted(one.seed for one in group),
                    against=against,
                    with_ok=with_ok,
                    with_n=with_n,
                    base_ok=base_ok,
                    base_n=base_n,
                    merge=merge,
                    why=why,
                    value=jsonable(definition(group[-1].doc.params[name])),
                )
            )
    return sorted(found, key=lambda r: (-len(r.seeds), r.param))


def _verdict(
    n: int,
    against: Sequence[int],
    with_ok: int,
    with_n: int,
    base_ok: int,
    base_n: int,
    seeds: int,
    calls: int,
) -> tuple[bool, str]:
    rates = f"its calls went through {with_ok} of {with_n}, M's {base_ok} of {base_n}"
    if n < seeds:
        return False, f"made in {n} skipped seeds, fewer than {seeds}"
    if against:
        return False, f"seeds {', '.join(map(str, against))} changed it another way"
    if with_n < calls or base_n < calls:
        return False, f"{rates}: fewer than {calls} calls on a side"
    lower, upper = wilson(with_ok, with_n)[0], wilson(base_ok, base_n)[1]
    if lower <= upper:  # their 95% intervals overlap
        return False, f"{rates}: not better beyond noise ({lower:.2f} <= {upper:.2f})"
    return True, f"{rates} ({lower:.2f} > {upper:.2f})"
