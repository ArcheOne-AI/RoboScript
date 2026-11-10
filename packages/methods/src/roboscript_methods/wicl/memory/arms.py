"""`unbind_arms`: rewrite a Memory that binds the left or the right arm (the `fixed_arm` lint,
docs/concepts/learning.md) into one that names arms by rule, deterministically: no model, no
budget, and it touches exactly the spans the lint flags.

Which arm does which stage is planned in each episode (`plan_arms`), so a side a Memory took
from the demonstration becomes a role: the side the workflow binds first is the stage's arm, the
other side the other arm (a hand-over keeps its two arms apart). A stage's arm line that binds a
side becomes `rule(<stage>)`; elsewhere
`arm=left` and `holding(left)` become
`arm=<arm>` and `holding(<arm>)` (`<other arm>` for the other side), and `the left gripper` becomes
`the chosen gripper` (`the other gripper`). A category parameter whose value is a side is removed
and its `{name}` references become the arm's token. Spec `arm` fields stay: a spec is written for
one arm and the tools mirror it for the other. Text naming both arms as alternatives, and
INSTRUCTION, are left as they are. A second pass changes nothing.
"""

import re
from dataclasses import dataclass
from typing import Final

from roboscript.memory.doc import MemoryDoc, content_hash
from roboscript.memory.linter import SIDES, arm_bindings, arm_category, prose
from roboscript.memory.store import Version

RULE: Final = "the arm plan_arms gives for {stage}"
"""A stage's arm line: `plan_arms` plans every stage's arm in each episode (the instruction's
arm where it names one); short, so that a rewritten Memory stays within the word limit."""


def rule(stage: str) -> str:
    """The arm line of `stage`."""
    return RULE.format(stage=stage)


TOKEN: Final = {0: "<arm>", 1: "<other arm>"}
WORD: Final = {0: "chosen", 1: "other"}
_SIDE: Final = re.compile(r"left|right", re.IGNORECASE)
_REF: Final = re.compile(r"\{([a-z][a-z0-9_]*)\}")


@dataclass(frozen=True)
class ArmRewrite:
    """The rewritten document and what changed: `(where, before, after)` per text field, and
    `(params.<name>, value, "removed")` per category removed."""

    doc: MemoryDoc
    changes: list[tuple[str, str, str]]

    def record(self) -> dict[str, object]:
        """What the rewrite did, for a run's records."""
        rows = [{"where": w, "before": b, "after": a} for w, b, a in self.changes]
        return {"rewrite": "unbind_arms", "hash": content_hash(self.doc), "changes": rows}


def unbind_arms(doc: MemoryDoc) -> ArmRewrite:
    """`doc` with every arm binding rewritten by role (the module's docstring); `doc` itself
    where there is nothing to change."""
    removed = {name: param.value for name, param in doc.params.items() if arm_category(param)}
    fields = list(prose(doc))
    spans = {where: arm_bindings(where, text) for where, text in fields}
    refs = {where: _refs(text, removed) for where, text in fields}
    if not removed and not any(spans.values()):
        return ArmRewrite(doc, [])
    order = sorted(fields, key=lambda field: not field[0].startswith("workflow."))
    found = [
        side
        for where, text in order
        for side in sorted(
            [(a, _side(text[a:b])) for a, b in spans[where]]
            + [(a, removed[name]) for a, _, name in refs[where]]
        )
    ]
    sides = list(dict.fromkeys([side for _, side in found] + list(SIDES)))
    role = {side: sides.index(side) for side in SIDES}
    texts = {}
    changes = []
    for where, text in fields:
        after = _rewrite(where, text, spans[where], refs[where], removed, role)
        if after != text:
            texts[where] = after
            changes.append((where, text, after))
    changes += [(f"params.{name}", value, "removed") for name, value in removed.items()]
    return ArmRewrite(_with_texts(doc, texts, removed), changes)


def unbound(version: Version) -> tuple[Version, ArmRewrite]:
    """A stored version as a reader that cannot commit sees it (a frozen Memory in `evaluate`
    and P3): its document unbound, and its hash the rewritten document's content hash, so the
    records name what the Actor read; the version itself where nothing binds an arm."""
    rewrite = unbind_arms(version.doc)
    if not rewrite.changes:
        return version, rewrite
    meta = version.meta.model_copy(update={"hash": content_hash(rewrite.doc)})
    return Version(rewrite.doc, meta), rewrite


def _refs(text: str, removed: dict[str, str]) -> list[tuple[int, int, str]]:
    return [(*m.span(), m.group(1)) for m in _REF.finditer(text) if m.group(1) in removed]


def _rewrite(
    where: str,
    text: str,
    spans: list[tuple[int, int]],
    refs: list[tuple[int, int, str]],
    removed: dict[str, str],
    role: dict[str, int],
) -> str:
    section, _, rest = where.partition(".")
    if section == "workflow" and rest.endswith(".arm") and (spans or refs):
        return rule(rest[: -len(".arm")])
    edits = [(a, b, _replace(text[a:b], role)) for a, b in spans]
    edits += [(a, b, TOKEN[role[removed[name]]]) for a, b, name in refs]
    for a, b, new in sorted(edits, reverse=True):
        text = text[:a] + new + text[b:]
    return text


def _side(span: str) -> str:
    side = _SIDE.search(span)
    assert side is not None
    return side.group(0).lower()


def _replace(span: str, role: dict[str, int]) -> str:
    """A bound side by its role: a token in a call or check (`arm=`, `(`), a word in prose."""
    side = _SIDE.search(span)
    assert side is not None
    n = role[side.group(0).lower()]
    new = TOKEN[n] if span.startswith("(") or span.lower().startswith("arm") else WORD[n]
    return span[: side.start()] + new + span[side.end() :]


def _with_texts(doc: MemoryDoc, texts: dict[str, str], removed: dict[str, str]) -> MemoryDoc:
    update: dict[str, object] = {}
    for section in ("objects", "notes"):
        lines = getattr(doc, section).splitlines()
        if any(where.startswith(f"{section}:") for where in texts):
            update[section] = "\n".join(
                texts.get(f"{section}:{n}", line) for n, line in enumerate(lines, 1)
            )
    update["workflow"] = [
        stage.model_copy(
            update={
                field: texts[f"workflow.{stage.name}.{field}"]
                for field in ("goal", "arm", "call", "done")
                if f"workflow.{stage.name}.{field}" in texts
            }
        )
        for stage in doc.workflow
    ]
    update["lessons"] = [
        lesson.model_copy(
            update={
                field: texts[f"lessons.{lesson.id}.{field}"]
                for field in ("when", "do")
                if f"lessons.{lesson.id}.{field}" in texts
            }
        )
        for lesson in doc.lessons
    ]
    update["params"] = {name: p for name, p in doc.params.items() if name not in removed}
    return doc.replace(**update)
