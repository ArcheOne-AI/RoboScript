"""The demonstration's end as targets, as the done gate and the P3 host carry them
(docs/concepts/learning.md): which of them carry over to a Memory's roles (`end_for`), the roles
an end holds (`carried_ends`), and the file a reading is kept in (`load_end_targets`). The gate
itself is the method's (wicl's `finish.DoneGate`, its `MethodRuntime.finish`)."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from roboscript.memory import (
    MemoryDoc,
    PlaceSpecParam,
)
from roboscript.memory.calls import parse_calls
from roboscript.memory.objects import object_roles
from roboscript.perception.criteria import EndTarget
from roboscript.perception.sides import task_sides

PLACE_CALLS: Final = frozenset({"place_object"})
"""The workflow calls that place a role by a PlaceSpec (`obj=` the role, `place=` the spec)."""


def robot_frame_roles(doc: MemoryDoc, positional: bool = False) -> set[str]:
    """The roles a workflow stage places by a PlaceSpec at a robot-frame constant (the last
    placement of the role in workflow order decides). The role is the call's `obj=`; with
    `positional`, a call that names it without the keyword (`place_object(arm, <role>, ...)`)
    counts too: its first positional argument that is a role of OBJECTS."""
    roles = set(object_roles(doc)) if positional else set()
    last: dict[str, bool] = {}
    for stage in doc.workflow:
        for call in parse_calls(stage.call):
            if call.name not in PLACE_CALLS:
                continue
            role = call.keywords.get("obj", "").strip()
            if not role:
                role = next((v.strip() for v in call.values if v.strip() in roles), "")
            name = call.keywords.get("place", "").strip().strip("{}")
            param = doc.params.get(name)
            if role and isinstance(param, PlaceSpecParam):
                last[role] = param.value.target.kind == "robot_frame"
    return {role for role, robot in last.items() if robot}


def end_for(
    doc: MemoryDoc, targets: Sequence[EndTarget], positional: bool = False
) -> list[EndTarget]:
    """The demonstration's end targets (read on its seed) that carry over to another seed under
    `doc`: on roles of its OBJECTS only (a role S2 renamed or dropped is not this Memory's; peers
    likewise), without yaw targets (another scene starts each object at its own yaw, and a
    placement that keeps it is not wrong), and base targets only for roles the workflow places at
    a robot-frame constant (`robot_frame_roles`: the Memory already holds such a constant). One
    demonstration cannot tell an end fixed in the robot's frame from one that only happens to
    lie there, so an object held up where a carry left it gets no base target of its own: S1's
    review flags such a carry written as a displacement (`carried_ends`), and a robot-frame
    PlaceSpec is the form that binds it. Where the Memory's TASK names a side of the robot
    (`perception.sides.task_sides`), the sides are the task's: no target is mirrored, nor follows
    an arm. `positional`: `robot_frame_roles`' (off: a placement that names its role without
    `obj=` gives it no base target)."""
    roles = set(object_roles(doc))
    robot = robot_frame_roles(doc, positional)
    fixed = task_sides(doc.statement)
    kept = []
    for t in targets:
        if t.a not in roles or (t.b is not None and t.b not in roles) or t.kind == "yaw":
            continue
        if t.kind == "base" and t.a not in robot:
            continue
        update: dict[str, object] = {"peers": [p for p in t.peers if p in roles]}
        if fixed:  # the task names its sides: the demonstration's end holds as it lies
            update |= {"mirror": False, "arm": None}
        kept.append(t.model_copy(update=update))
    return kept


def carried_ends(targets: Sequence[EndTarget]) -> list[str]:
    """The roles the demonstration ends holding where an arm carried them, over no anchor: a
    base target (made only for an unanchored role) that knows its holding arm and that the arm
    carried it there (wicl's `perception.endstate.annotate_holds`). For the linter's
    `relative_carry` (`memory.LintConfig.carried_ends`); read on the targets before `end_for`
    drops any."""
    return sorted({t.a for t in targets if t.kind == "base" and t.carried and t.arm is not None})


def load_end_targets(path: Path) -> list[EndTarget]:
    """End targets from a file of `read_end_targets`' report (`{"targets": [...]}`)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EndTarget.model_validate(t) for t in data.get("targets", [])]
