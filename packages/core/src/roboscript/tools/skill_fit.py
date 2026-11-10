"""The fit contract of a skillset's written skills (docs/concepts/learning.md, "Written
skills"): how a skill's parameters are fitted on one segment of a demonstration, so that S1 can
store them in the Memory (a `skill_spec` parameter) and S2 and P3 run the skill with them.

A skillset gives each of its written skills a `Fitter` (`tools.skillsets.SkillSetPlugin.fitters`):
`fit(segment, recording, scene, base, conventions) -> dict | None`, the skill's parameters that
reproduce `segment` (relative to the objects involved), or None where the skill does not apply.
It is the P1 builder's `fit.py` (`roboscript_methods.skill_builder`, `builder/README.md`, "A
written skill"), which the builder runs on the recording's ground truth. At S1 it runs on what a
real robot has (tier D): the recorded proprioception (`DemoTracks`) and the objects as this
episode perceived them on the demonstration's own scene, moved by the recorded grasps (wicl's
`toolsets.skill_fit`). So a fitter reads only these duck-typed inputs:

- `segment`: a `FitSegment`;
- `recording`: `.demo` (a `demos.Demo`: arm tracks and `meta.ee_to_tcp`), `.objects` (object
  id -> its pose per frame, `[x, y, z, qw, qx, qy, qz]` rows) and `.poses(frame)` (object id ->
  `Pose` at that frame);
- `scene`: `.boxes` and `.poses` (keyed by the objects it has) and `.at(poses, links)`, the scene
  with the objects at `poses`, whose `.proxy(id)` is the object's `ObjectProxy` there;
- `base`: the robot base's pose in the world; `conventions`: the embodiment's `PoseConventions`.

A fitter that reaches for anything else (the recording's commands, ground-truth parts) raises,
and S1 reports that it cannot be fitted from perception.

A fit must be relative (the rule for written skills): its
positions are relative to a perceived object, a part or the gripper, never in the world's or the
robot's frame, so that they carry over to another scene. `absolute_numbers` is the rule.

With the fit goes the skill's `act` (`WrittenFit`), which S1 rehearses on the demonstration
(building its motions, moving nothing) to measure how closely the fitted call reproduces the
segment, so that the Agent chooses among the skills that fit by that, not by their names.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm

SegmentKind = Literal["grasp", "place", "carry", "touch", "release", "move", "home"]
"""What a segment ends with (the P1 builder's `demo_coverage.segments` kinds)."""


@dataclass(frozen=True)
class FitSegment:
    """One arm's motion over frames [`start`, `end`] of the demonstration and the event that ends
    it: a `grasp` or `touch` of `obj` (the TCP then at `tcp`), the `place` of `obj` that rests
    from frame `rest` on or beside `target`, its `release` to the other arm, its `carry` to where
    it is at `rest`, a `move` with no event, or `home`; `handover`: a grasp of what the other arm
    holds; `closed`: the gripper stayed closed through it (a press)."""

    arm: Arm
    kind: SegmentKind
    start: int
    end: int
    tcp: Pose
    obj: str | None = None
    rest: int | None = None
    target: str | None = None
    handover: bool = False
    closed: bool = False

    @property
    def name(self) -> str:
        what = f" {self.obj}" if self.obj else ""
        what += " from the other arm" if self.handover and self.kind == "grasp" else ""
        onto = f" at {self.target}" if self.target else ""
        return f"{self.arm} {self.kind}{what}{onto} [{self.start}, {self.end}]"


Fitter: TypeAlias = Callable[[FitSegment, Any, Any, Pose, Any], "dict[str, Any] | None"]
"""`fit(segment, recording, scene, base, conventions)`: the skill's parameters, or None."""


Actor: TypeAlias = Callable[..., Awaitable[Any]]
"""A written skill's `act(api, arm, obj, target, params) -> list[Built]`."""


@dataclass(frozen=True)
class WrittenFit:
    """A written skill's fit and, where the skillset gives it, its `act` (what S1 rehearses to
    measure the fit's residual; None: the fit is listed without one)."""

    fit: Fitter
    act: Actor | None = None


RELATIVE_RULE: Final = (
    "A fit's params hold no absolute position: every position in them is relative to a perceived "
    "object, a part or the gripper, never in the world's or the robot's frame. Checked by fitting "
    "again on the demonstration moved relative to the robot and the world (its objects and the "
    "gripper's path, together): every number in params must stay the same."
)
"""The rule, word for word as the P1 gate states it (`demo_coverage.fitting.RELATIVE_RULE`)."""
SHIFTS_M: Final[tuple[tuple[float, float, float], ...]] = ((0.04, -0.03, 0.0),)
"""The shift (m, world) the P1 gate moves the demonstration by: the objects' poses, each arm's
ee and TCP path, the cameras and the segment's TCP move; the joints and the robot base do not."""
TOLERANCE: Final = 0.01
"""How far a number may move under the shift and still count as the same (the P1 gate's)."""


def absolute_numbers(
    fitted: Any, shifted: Sequence[Any], tolerance: float = TOLERANCE
) -> list[str]:
    """`RELATIVE_RULE` on a fit's output: where it breaks it.

    `fitted` is a skill's fit on the demonstration as recorded; `shifted` the refit on the
    demonstration moved by each of `SHIFTS_M` (None where the refit returned None or raised).
    Returns the path of each number that moved by more than `tolerance` (e.g. `start[1]`,
    `waypoints[2][3]`) and of each place whose structure differs (a key, a length, a type, a
    refit that is None); empty: the fit is relative. It reads plain data only (what the fits
    returned, as JSON-like values), never the skill, as the P1 gate's check (`AbsoluteParams`)
    does."""
    found: dict[str, None] = {}

    def walk(a: Any, b: Any, where: str) -> None:
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            if a != b:
                found[where or "(the fit)"] = None
        elif isinstance(a, int | float) and isinstance(b, int | float):
            if abs(float(a) - float(b)) > tolerance:
                found[where or "(the fit)"] = None
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                found[where or "(the fit)"] = None
            for key in (key for key in a if key in b):  # in the fit's order
                walk(a[key], b[key], f"{where}.{key}" if where else str(key))
        elif isinstance(a, list | tuple) and isinstance(b, list | tuple):
            if len(a) != len(b):
                found[where or "(the fit)"] = None
            for k, (x, y) in enumerate(zip(a, b, strict=False)):
                walk(x, y, f"{where}[{k}]")
        elif a != b:
            found[where or "(the fit)"] = None

    for other in shifted:
        walk(fitted, other, "")
    return list(found)
