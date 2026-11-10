"""The OBJECTS section's lines (docs/concepts/learning.md): one abstract role per line,

    - <role>: <what it does in the task>; <how to recognise it in general terms>
      x toward: <the end or side of it its frame's x points to, in words>
      arm side: <toward its x | away from its x | either>
      order: <rank> among <group> by <key>

The optional indented `x toward:` line under a role (at most `MAX_TOWARD` characters, no digits)
says, in words that hold for any object of its kind (the spout, the handle, the narrow end), which
way its frame's x points: an S1 Demo Reader writes it from the demonstration's frames with axis
overlays, and the one-time identification uses it to settle a frame whose shape leaves x open
(`object_towards`). The optional `arm side:` line after it (`ARM_SIDES`) says which arm acts on
it by where its x points, as the demonstration had it: `toward its x`, the arm on the side of the
robot its x points to (a vessel carried head first toward its own side), `away from its x`, the
other one, `either`, no matter; `plan_arms` then gives that arm on any scene, mirrored with the
object, instead of the nearer one (a relation of the object's own axis to the robot, no
position). The optional `order:` line after it binds the role among look-alike objects
(the same model: nothing in the instruction or their look tells them apart) by where they lie,
as the demonstration's first frame had them: the roles sharing a `<group>` word take the
objects the identification found for them sorted by `<key>` (`ORDER_KEYS`: `distance to the
robot`, nearest first, or `the robot's left to right`), the role of rank 1 the first
(`by_order`). A plural role (the instruction names several look-alike objects together: "the
rest", "the others") writes `rest` for its rank: it takes every look-alike object of its kind
that no other role takes, as many as the identification finds, sorted by `<key>`
(`rest_order`); the group's ranked roles are sorted among all of them, so rank 1 is the first of
the whole group, and the rest role takes those left.

`<role>` is lowercase letters, digits and `_`, starting with a letter, at most 40 characters, and
each role appears once; a description is at most `MAX_DESCRIPTION` characters, and there are at
most `MAX_ROLES` roles, the limits of the one-time identification (`identify`). Which object
plays a role in an episode (its colour, which one of several) is never written here: each
episode's instruction says it, and the one-time identification reads the roles with the live
instruction. A line of another form, or past a limit, is a lint error (`objects`), never skipped,
so an edit that makes one is refused, and `object_roles` refuses a Memory that has one.
"""

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Final, NamedTuple

import numpy as np

from roboscript.contracts.vision import MAX_QUERIES
from roboscript.memory.calls import parse_calls
from roboscript.memory.doc import MemoryDoc

ROLE_NAME: Final = re.compile(r"[a-z][a-z0-9_]{0,39}")
MAX_ROLES: Final = MAX_QUERIES
"""The identification looks for at most this many objects (GroundingDINO's queries per call)."""
MAX_DESCRIPTION: Final = 200
"""The identification takes descriptions of at most this many characters."""
MAX_TOWARD: Final = 60
_LINE: Final = re.compile(r"^-\s+(?P<role>[^:]*?)\s*:\s*(?P<description>\S.*?)\s*$")
_TOWARD: Final = re.compile(r"^\s+x toward\s*:\s*(?P<text>\S.*?)\s*$")
ARM_SIDES: Final = ("toward its x", "away from its x", "either")
_ARM_SIDE: Final = re.compile(r"^\s+arm side\s*:\s*(?P<text>\S.*?)\s*$")
ORDER_KEYS: Final = ("distance to the robot", "the robot's left to right")
_ORDER: Final = re.compile(r"^\s+order\s*:\s*(?P<text>\S.*?)\s*$")
_ORDER_TEXT: Final = re.compile(
    r"^(?P<rank>[1-9]|rest)\s+among\s+(?P<group>[a-z][a-z_]{0,39})\s+by\s+(?P<key>.+)$"
)


REST: Final = "rest"
"""The rank of a plural role: every look-alike object the other roles leave."""


class Order(NamedTuple):
    """A role's `order:` line: its rank among its group's look-alikes, sorted by `key` (None:
    `rest`, a plural role)."""

    rank: int | None
    group: str
    key: str


class ObjectLine(NamedTuple):
    """A role line: its number in the section, the role, its description, its `x toward:`
    words, its `order:` and its `arm side:` (None: none given)."""

    number: int
    role: str
    description: str
    toward: str | None = None
    order: Order | None = None
    arm_side: str | None = None


def parse_objects(text: str) -> tuple[list[ObjectLine], list[tuple[int, str]]]:
    """The role lines of an OBJECTS section, and the others as (line number, what is wrong)."""
    lines: list[ObjectLine] = []
    problems: list[tuple[int, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        order = _ORDER.match(line)
        if order is not None:
            parsed, why = _order(order["text"], lines[-1] if lines else None)
            if why is not None:
                problems.append((number, why))
            else:
                lines[-1] = lines[-1]._replace(order=parsed)
            continue
        side = _ARM_SIDE.match(line)
        if side is not None:
            why = _side_problem(side["text"], lines[-1] if lines else None)
            if why is not None:
                problems.append((number, why))
            else:
                lines[-1] = lines[-1]._replace(arm_side=side["text"])
            continue
        toward = _TOWARD.match(line)
        if toward is not None:
            why = _toward_problem(toward["text"], lines[-1] if lines else None, number)
            if why is not None:
                problems.append((number, why))
            else:
                lines[-1] = lines[-1]._replace(toward=toward["text"])
            continue
        match = _LINE.match(line.strip())
        if match is None:
            problems.append((number, 'not a role line "- <role>: <what it does>; <what it is>"'))
            continue
        role, description = match["role"], match["description"]
        if ROLE_NAME.fullmatch(role) is None:
            why = f"the role {role!r} is not lowercase letters, digits and _ (at most 40)"
            problems.append((number, why))
        elif role in seen:
            problems.append((number, f"the role {role!r} is given twice"))
        elif len(description) > MAX_DESCRIPTION:
            why = f"the description has {len(description)} characters, more than {MAX_DESCRIPTION}"
            problems.append((number, why))
        elif len(lines) == MAX_ROLES:
            why = f"more than {MAX_ROLES} roles: the identification looks for {MAX_ROLES} at most"
            problems.append((number, why))
        else:
            seen.add(role)
            lines.append(ObjectLine(number, role, description))
    return lines, problems


def parse_order(text: str) -> Order:
    """An `order:` line's text as an `Order`; ValueError saying the form."""
    found, why = _order(text, ObjectLine(0, "role", "description"))
    if found is None:
        raise ValueError(why)
    return found


def _order(text: str, role: ObjectLine | None) -> tuple[Order | None, str | None]:
    if role is None:
        return None, "an order: line goes under its role's line"
    if role.order is not None:
        return None, f"the role {role.role!r} has two order: lines"
    match = _ORDER_TEXT.match(text)
    if match is None or match["key"] not in ORDER_KEYS:
        keys = " or ".join(ORDER_KEYS)
        form = '"<rank> among <group> by <key>", the rank 1-9 or rest,'
        return None, f"order: is {form} the key {keys}"
    rank = None if match["rank"] == REST else int(match["rank"])
    return Order(rank, match["group"], match["key"]), None


def _toward_problem(text: str, role: ObjectLine | None, number: int) -> str | None:
    if role is not None and role.toward is not None:
        return f"the role {role.role!r} has two x toward: lines"
    if role is None or role.number != number - 1 or role.order is not None:
        return "an x toward: line goes right under its role's line"
    if len(text) > MAX_TOWARD:
        return f"x toward: has {len(text)} characters, more than {MAX_TOWARD}"
    if any(c.isdigit() for c in text):
        return "x toward: names a part or side in words, without numbers"
    return None


def _side_problem(text: str, role: ObjectLine | None) -> str | None:
    if role is None:
        return "an arm side: line goes under its role's line"
    if role.arm_side is not None:
        return f"the role {role.role!r} has two arm side: lines"
    if role.order is not None:
        return "an arm side: line goes before the role's order: line"
    if text not in ARM_SIDES:
        return "arm side: is " + ", ".join(repr(a) for a in ARM_SIDES)
    return None


SIDE_MIN: Final = 0.34
"""An object's x points to one side of the robot when its horizontal direction's share along the
robot's lateral axis is at least this (sin 20 deg); nearer straight ahead or back it points to
neither."""


def pointed_arm(x_axis: Sequence[float], lateral: Sequence[float]) -> str | None:
    """The arm on the side of the robot an object's frame x (world) points to: the robot's left
    is its base frame's +y (`lateral`, world), its left arm's side; None when x points within 20
    deg of straight ahead or back, or up."""
    x = np.asarray(x_axis, dtype=float)[:2]
    left = np.asarray(lateral, dtype=float)[:2]
    norms = float(np.linalg.norm(x)) * float(np.linalg.norm(left))
    if norms < 1e-9:
        return None
    share = float(x @ left) / norms
    if abs(share) < SIDE_MIN:
        return None
    return "left" if share > 0 else "right"


def directed_arm(rule: str, x_axis: Sequence[float], lateral: Sequence[float]) -> str | None:
    """The arm an `arm side:` rule gives for an object whose frame x (world) points as `x_axis`
    (None: `either`, or x points to neither side)."""
    pointed = pointed_arm(x_axis, lateral)
    if rule == "either" or pointed is None:
        return None
    if rule == "toward its x":
        return pointed
    return "right" if pointed == "left" else "left"


def side_rule(arm: str, x_axis: Sequence[float], lateral: Sequence[float]) -> str | None:
    """The `arm side:` rule that gives `arm` for an object whose frame x points as `x_axis`
    (None: x points to neither side)."""
    pointed = pointed_arm(x_axis, lateral)
    if pointed is None:
        return None
    return ARM_SIDES[0] if pointed == arm else ARM_SIDES[1]


def object_arm_sides(memory: MemoryDoc) -> dict[str, str]:
    """The roles' `arm side:` lines, role -> one of `ARM_SIDES` (roles without one left out)."""
    lines, _ = parse_objects(memory.objects)
    return {line.role: line.arm_side for line in lines if line.arm_side is not None}


def object_towards(memory: MemoryDoc) -> dict[str, str]:
    """The roles' `x toward:` words, role -> words (roles without one left out)."""
    lines, _ = parse_objects(memory.objects)
    return {line.role: line.toward for line in lines if line.toward is not None}


def object_orders(memory: MemoryDoc) -> dict[str, Order]:
    """The roles' `order:` lines, role -> order (roles without one left out)."""
    lines, _ = parse_objects(memory.objects)
    return {line.role: line.order for line in lines if line.order is not None}


LOOKALIKE_M: Final = 0.01
"""Objects whose measured extents (sorted) differ by at most this each look alike."""


def ranks(
    starts: Mapping[str, Sequence[float]],
    sizes: Mapping[str, Sequence[float]],
    obj: str,
    lateral: Sequence[float],
    base: Sequence[float],
) -> dict[str, tuple[int, int]] | None:
    """`obj`'s rank among the objects that look like it (`LOOKALIKE_M`) by each `ORDER_KEYS`
    key, as (rank, how many); None when nothing else looks like it."""
    mine = np.sort(np.asarray(sizes[obj], dtype=float))
    alike = [
        o for o in starts if np.all(np.abs(np.sort(np.asarray(sizes[o])) - mine) <= LOOKALIKE_M)
    ]
    if len(alike) < 2:
        return None
    left = np.asarray(lateral, dtype=float)[:2]
    keyed = {
        ORDER_KEYS[0]: lambda o: float(np.linalg.norm(np.subtract(starts[o], base)[:2])),
        ORDER_KEYS[1]: lambda o: -float(np.asarray(starts[o], dtype=float)[:2] @ left),
    }
    return {key: (sorted(alike, key=f).index(obj) + 1, len(alike)) for key, f in keyed.items()}


def by_order(
    orders: Mapping[str, Order],
    bound: Mapping[str, str],
    starts: Mapping[str, Sequence[float]],
    lateral: Sequence[float],
    base: Sequence[float],
    extra: Mapping[str, Collection[str]] | None = None,
) -> dict[str, str]:
    """Each ordered role's object once its group's are sorted (`order:`): `bound` the objects
    the identification gave the roles (role -> object; roles it left unbound take none),
    `starts` their centers, `lateral` the robot's left direction and `base` where it stands
    (world); `extra`, by group, more objects its ranked roles are sorted among (a rest role's;
    the ranked roles take the first). Roles with no order keep theirs, and so does a rest role
    (`rest_order` gives it its objects)."""
    out = dict(bound)
    groups: dict[tuple[str, str], list[str]] = {}
    for role, order in orders.items():
        if role in bound and order.rank is not None:
            groups.setdefault((order.group, order.key), []).append(role)
    left = np.asarray(lateral, dtype=float)[:2]
    for (group, key), roles in groups.items():
        if len({bound[r] for r in roles}) < len(roles):
            continue  # one object for two roles: the identification's problem to report
        objects = sorted({bound[r] for r in roles} | set((extra or {}).get(group, ())))
        if key == ORDER_KEYS[0]:
            score = {o: float(np.linalg.norm(np.subtract(starts[o], base)[:2])) for o in objects}
        else:
            score = {o: -float(np.asarray(starts[o], dtype=float)[:2] @ left) for o in objects}
        ranked = sorted(objects, key=score.__getitem__)  # as `ranks` orders them
        by_rank = sorted(roles, key=lambda r: orders[r].rank)
        for role, obj in zip(by_rank, ranked[: len(roles)], strict=True):
            out[role] = obj
    return out


def rest_order(
    order: Order,
    objects: Sequence[str],
    starts: Mapping[str, Sequence[float]],
    lateral: Sequence[float],
    base: Sequence[float],
) -> list[str]:
    """A rest role's objects sorted by its `order:` line's key (`by_order`'s arguments)."""
    if order.key == ORDER_KEYS[0]:
        far = {o: float(np.linalg.norm(np.subtract(starts[o], base)[:2])) for o in objects}
        return sorted(objects, key=far.__getitem__)
    left = np.asarray(lateral, dtype=float)[:2]
    return sorted(objects, key=lambda o: -float(np.asarray(starts[o], dtype=float)[:2] @ left))


def order_text(order: Order) -> str:
    """An `order:` line's text, as written."""
    rank = REST if order.rank is None else order.rank
    return f"{rank} among {order.group} by {order.key}"


def object_roles(memory: MemoryDoc) -> dict[str, str]:
    """The roles of a Memory's OBJECTS section, role -> description, in order; ValueError naming
    the first line that is not a role line."""
    lines, problems = parse_objects(memory.objects)
    if problems:
        number, why = problems[0]
        raise ValueError(f"OBJECTS line {number}: {why}")
    return {line.role: line.description for line in lines}


CARRY_CALLS: Final = frozenset({"grasp_object", "place_object"})
"""The workflow calls whose `obj=` is something an arm carries."""


def carried_roles(memory: MemoryDoc) -> set[str]:
    """The roles a workflow stage grasps or places (`obj=` of `CARRY_CALLS`)."""
    found = set()
    for stage in memory.workflow:
        for call in parse_calls(stage.call):
            if call.name in CARRY_CALLS and (role := call.keywords.get("obj", "").strip()):
                found.add(role)
    return found
