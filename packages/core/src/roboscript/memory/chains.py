"""The holds a Memory's workflow makes (`arm_chains`, docs/concepts/embodied-core.md): which arm
does which stage is decided once per episode over these, by `plan_arms`, never written in the
Memory.

A hold opens where a stage grasps a role (`grasp_object(obj=<role>, grasp={spec})`, also inside
`together`) and closes where a placement lets it go (`place_object(..., is_open=true)`, the
default) or a gripper opens (`open_gripper`). A grasp of a role another hold has open is a
hand-over's receiving hold: the other arm makes it, and the next opening gripper closes the
giving hold. The holds a role's placement and held path belong to are the role's newest open one
(a path with no role: the stage's single open hold). Holds open at the same time need different
arms. Each stage belongs to the holds it touches; a stage that touches none (a retreat, a look,
a preshape) belongs to the hold of the stage before it, or before any hold to the first one. Pure
functions over the workflow's call templates (`calls.parse_calls`)."""

import re
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, ConfigDict

from roboscript.memory.calls import Call, parse_calls
from roboscript.memory.doc import MemoryDoc

__all__ = ["Chains", "Hold", "arm_chains"]

_GRASP: Final = "grasp_object"
_PLACE: Final = "place_object"
_PATH: Final = "follow_path"
_OPEN: Final = "open_gripper"
_REF: Final = re.compile(r"^\{?\s*([a-z][a-z0-9_]*)\s*\}?$")
_FALSE: Final = re.compile(r"^(?:false|0|no)$", re.IGNORECASE)


class Hold(BaseModel):
    """One hold: the role it grasps, its grasp, held path and placement specs by parameter name
    (where the workflow names them), the stages it spans, the stage that opens it and the one
    that closes it (None: still held at the end), and the hold it receives the role from (a
    hand-over: the other arm)."""

    model_config = ConfigDict(frozen=True)

    role: str
    grasp: str | None = None
    path: str | None = None
    place: str | None = None
    stages: list[str] = []
    opens: str
    closes: str | None = None
    receives_from: int | None = None


class Chains(BaseModel):
    """The workflow's holds in the order they open, which stage belongs to which holds, and the
    pairs of holds open at the same time."""

    model_config = ConfigDict(frozen=True)

    holds: list[Hold] = []
    stages: dict[str, list[int]] = {}
    together: list[tuple[int, int]] = []

    def record(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def _ref(text: str | None) -> str | None:
    if text is None:
        return None
    match = _REF.match(text.strip())
    return match.group(1) if match else None


def _second(call: Call, name: str) -> str | None:
    """`call`'s `name` argument: by keyword, else the second positional one (the tools take the
    arm first: `grasp_object(<arm>, piece, ...)`)."""
    given = call.keywords.get(name)
    if given is None and len(call.values) > 1:
        given = call.values[1]
    if given is None:
        return None
    text = given.strip().strip("'\"")
    return text or None


def _role(call: Call) -> str | None:
    """The role a grasp or placement acts on (`obj`); another call's only by keyword."""
    if call.name in (_GRASP, _PLACE):
        return _second(call, "obj")
    obj = call.keywords.get("obj")
    return obj.strip().strip("'\"") or None if obj is not None else None


@dataclass
class _Draft:
    role: str
    grasp: str | None
    opens: str
    receives_from: int | None
    path: str | None = None
    place: str | None = None
    closes: str | None = None
    stages: list[str] = field(default_factory=list)


def arm_chains(doc: MemoryDoc | None) -> Chains:
    """`doc`'s holds (the module's docstring); none for a workflow that grasps nothing."""
    if doc is None:
        return Chains()
    holds: list[_Draft] = []
    open_: list[int] = []
    stages: dict[str, list[int]] = {}
    together: set[tuple[int, int]] = set()
    previous: list[int] = []
    pending: list[str] = []  # stages before the first hold

    def newest(role: str | None) -> int | None:
        if role is None:
            return open_[0] if len(open_) == 1 else None
        return next((i for i in reversed(open_) if holds[i].role == role), None)

    def close(index: int, stage: str) -> None:
        holds[index].closes = stage
        if index in open_:
            open_.remove(index)

    for stage in doc.workflow:
        touched: list[int] = []
        for call in parse_calls(stage.call):
            role = _role(call)
            if call.name == _GRASP and role is not None:
                giver = newest(role)
                if giver is not None and holds[giver].opens == stage.name:
                    giver = None  # two grasps of one role at once (a co-grasp), not a hand-over
                together.update((other, len(holds)) for other in open_)
                grasp = _ref(call.keywords.get("grasp"))
                holds.append(_Draft(role, grasp, stage.name, giver))
                open_.append(len(holds) - 1)
                touched.append(len(holds) - 1)
            elif call.name == _PLACE and role is not None:
                index = newest(role)
                if index is None:
                    continue
                holds[index].place = holds[index].place or _ref(call.keywords.get("place"))
                touched.append(index)
                if not _FALSE.match((call.keywords.get("is_open") or "true").strip()):
                    close(index, stage.name)
            elif call.name == _PATH:
                index = newest(role)
                if index is not None:
                    holds[index].path = holds[index].path or _ref(_second(call, "path"))
                    touched.append(index)
            elif call.name == _OPEN:
                givers = [i for i in open_ if any(holds[j].receives_from == i for j in open_)]
                index = givers[0] if givers else newest(None)
                if index is not None:
                    touched.append(index)
                    close(index, stage.name)
        touched = list(dict.fromkeys(touched)) or previous
        if not touched:
            pending.append(stage.name)
            continue
        for index in touched:
            if stage.name not in holds[index].stages:
                holds[index].stages.append(stage.name)
            stages.setdefault(stage.name, []).append(index)
        previous = touched
    if holds:
        for name in pending:
            holds[0].stages.insert(pending.index(name), name)
            stages[name] = [0]
    order = [stage.name for stage in doc.workflow]
    return Chains(
        holds=[Hold.model_validate(vars(h)) for h in holds],
        stages={name: stages[name] for name in order if name in stages},
        together=sorted(together),
    )
