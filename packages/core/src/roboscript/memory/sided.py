"""Sides a Task Memory took from the demonstration's arm, made relative to the acting arm
(docs/concepts/learning.md, `perception.sides`). A done criterion that holds role `a` on the
robot's left or right of role `b` (an `offset_x` criterion away from zero, a done line's
`beside(a, b, left|right, ...)`) took that side from the demonstration's arm only where
`side_rule` says so; elsewhere the side is the task's or `b`'s own. The rule the linter's
`side_bound` uses:

- not where TASK names a side of the robot: the side is the task's;
- not where `a` is placed into or onto `b` (a criterion or done line of the same pair says
  `inside`, `on` or `overlap`): the offset comes from `b`'s geometry (a drawer, a box);
- bound where `a` and `b` are held at the same time by different arms (`a`'s side of `b`
  follows `a`'s arm: a scanner aimed at the box the other arm holds);
- not where the workflow places both, one after the other (a row, a stack: an order the
  task makes);
- bound where `b` is an anchor the workflow never moves (grasps, places or nudges): `a` ends on
  the side its arm brought it to;
- not otherwise (nothing says where the side came from).

The arm is the one `a` was grasped with in the demonstration (`demo_arms`: the fitted grasp
spec's `arm`, or its source's, `demo right K3`). Bound, the side is written relative to it: an
`offset_x` criterion with that `arm`, a `beside` side as `arm_side` or `other_side`."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from roboscript.contracts.robot import Arm
from roboscript.memory.calls import Call, parse_calls
from roboscript.memory.doc import MemoryDoc
from roboscript.memory.params import DoneSpecParam, GraspSpecParam, Param
from roboscript.perception.sides import task_sides

GRASP_CALLS: Final = frozenset({"grasp_object", "co_grasp"})
PLACE_CALLS: Final = frozenset({"place_object"})
MOVE_CALLS: Final = GRASP_CALLS | PLACE_CALLS | {"nudge"}
CONTAIN_METRICS: Final = frozenset({"inside", "overlap"})
_FITTED_ARM: Final = re.compile(r"\bdemo (?P<arm>left|right) K\d+")
_BESIDE: Final = re.compile(
    r"(?<![\w.])(?P<head>beside\s*\(\s*(?P<a>[^,()]+?)\s*,\s*(?P<b>[^,()]+?)\s*,\s*)"
    r"(?P<side>left|right)(?P<tail>\s*[,)])"
)
_CONTAINS: Final = re.compile(
    r"(?<![\w.])(?:on|inside)\s*\(\s*(?P<a>[^,()]+?)\s*,\s*(?P<b>[^,()]+?)\s*[,)]"
)


@dataclass(frozen=True)
class SideRule:
    """Whether a side of `b` that `a` ends on came from the demonstration's arm (`bind`), the arm
    (None: unknown) and why."""

    bind: bool
    arm: Arm | None
    why: str


@dataclass(frozen=True)
class Sided:
    """One robot-frame side a done criterion holds: `where` (`params.<name>` or
    `workflow.<stage>.done`), its text, the pair and `side_rule`'s verdict."""

    where: str
    match: str
    a: str
    b: str
    rule: SideRule


def _calls(doc: MemoryDoc) -> Iterator[Call]:
    for stage in doc.workflow:
        yield from parse_calls(stage.call)


def _obj(call: Call) -> str | None:
    role = call.keywords.get("obj")
    return None if role is None else role.strip()


def fitted_arm(param: Param | None) -> Arm | None:
    """The demonstration arm a grasp spec was fitted at: its `arm`, else its source's."""
    if not isinstance(param, GraspSpecParam):
        return None
    if param.value.arm is not None:
        return param.value.arm
    match = _FITTED_ARM.search(param.source)
    return None if match is None else "left" if match["arm"] == "left" else "right"


def _grasp_arm(doc: MemoryDoc, call: Call) -> Arm | None:
    name = call.keywords.get("grasp")
    return None if name is None else fitted_arm(doc.params.get(name.strip().strip("{}")))


def demo_arms(doc: MemoryDoc) -> dict[str, Arm]:
    """Role -> the arm whose grasp spec takes it last in the workflow: the arm that acted on it
    in the demonstration."""
    found: dict[str, Arm] = {}
    for call in _calls(doc):
        role = _obj(call)
        arm = _grasp_arm(doc, call) if call.name in GRASP_CALLS else None
        if role is not None and arm is not None:
            found[role] = arm
    return found


def _held_together(doc: MemoryDoc, a: str, b: str) -> bool:
    """Whether the workflow holds `a` and `b` at once in different arms: each grasp holds its
    role in its spec's arm until a placement that opens the gripper (`is_open` not false) or an
    `open_gripper` lets go of what the last-grasped arm holds."""
    held: dict[str, Arm] = {}
    last: Arm | None = None
    for call in _calls(doc):
        role = _obj(call)
        if call.name in GRASP_CALLS and role is not None:
            if (arm := _grasp_arm(doc, call)) is not None:
                held[role], last = arm, arm
        elif call.name in PLACE_CALLS and role is not None:
            if call.keywords.get("is_open", "true").strip().lower() != "false":
                held.pop(role, None)
        elif call.name == "open_gripper" and last is not None:
            held = {r: arm for r, arm in held.items() if arm != last}
        if a in held and b in held and held[a] != held[b]:
            return True
    return False


def _contained(doc: MemoryDoc, a: str, b: str) -> bool:
    pair = {a, b}
    for param in doc.params.values():
        if (
            isinstance(param, DoneSpecParam)
            and param.value.metric in CONTAIN_METRICS
            and {param.value.a, param.value.b} == pair
        ):
            return True
    texts = [stage.done for stage in doc.workflow]
    return any(
        {m["a"].strip(), m["b"].strip()} == pair for text in texts for m in _CONTAINS.finditer(text)
    )


def side_rule(doc: MemoryDoc, a: str, b: str) -> SideRule:
    """Whether `a`'s robot-frame side of `b` in a done criterion came from the demonstration's
    arm (the module's docstring)."""
    arm = demo_arms(doc).get(a)
    if task_sides(doc.statement):
        return SideRule(False, arm, "TASK names the side")
    if _contained(doc, a, b):
        why = f"{a} and {b} end one in or on the other: the offset is their geometry"
        return SideRule(False, arm, why)
    if _held_together(doc, a, b):
        return SideRule(True, arm, f"{a} and {b} are held at once by different arms")
    moved = {_obj(c) for c in _calls(doc) if c.name in MOVE_CALLS}
    placed = {_obj(c) for c in _calls(doc) if c.name in PLACE_CALLS}
    if a in placed and b in placed:
        return SideRule(False, arm, f"the workflow places {a} and {b} in turn: the task's order")
    if b not in moved:
        return SideRule(True, arm, f"{b} is an anchor the workflow never moves")
    return SideRule(False, arm, f"{b} is moved and not held with {a}: no arm gave the side")


def sided(doc: MemoryDoc) -> list[Sided]:
    """Every robot-frame side the done criteria hold, each with `side_rule`'s verdict: `offset_x`
    criteria away from zero by more than their band and without `arm`, and done lines'
    `beside(a, b, left|right, ...)`."""
    found = []
    for name, param in doc.params.items():
        if not isinstance(param, DoneSpecParam):
            continue
        spec = param.value
        if spec.metric == "offset_x" and spec.arm is None and abs(spec.value) > spec.tol:
            rule = side_rule(doc, spec.a, spec.b)
            found.append(Sided(f"params.{name}", name, spec.a, spec.b, rule))
    for stage in doc.workflow:
        for m in _BESIDE.finditer(stage.done):
            a, b = m["a"].strip(), m["b"].strip()
            rule = side_rule(doc, a, b)
            found.append(Sided(f"workflow.{stage.name}.done", m[0], a, b, rule))
    return found
