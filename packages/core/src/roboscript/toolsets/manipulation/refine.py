"""Coarse and fine grasps and placements (docs/concepts/embodied-core.md): what the manipulation
toolset does between an action's coarse move to its pre-pose and the fine rest, and after a release.

At the pre-pose the objects an action depends on are measured again (`SceneMeasure.refine`: the
grasped object; the held one and a placement's target object), with the arm's wrist camera close
and the head view. A measurement is taken only when it is `ok`, at least `min_confidence` sure and
moves the object (or the part measured) at most `max_shift_m` (`cut_max_shift_m` for one an image
border cut) and turns it at most `max_turn_deg` (a re-measured frame already continues the old one
up to the footprint's symmetric turns, so a larger turn is a misfit, not a correction); it is then
stored the scene toolsets' deterministic way (`remeasure` in the same frame and cameras). An object
not seen where it was but found nearby (`relocated`: knocked or fallen) is taken whatever its shift
and turn when sure. Otherwise the coarse estimate stays, and the receipt says so. After a release
the placed object is measured again (`remeasure`), and after a grasp that missed (`empty_grasp`,
`lost_hold`, a deviated or short stop) the object it aimed at, which the fingers may have knocked;
the receipt notes where one was found. Each decision is a `manipulation.refine` event with the
correction in mm and deg (a hinge's, `manipulation.hinge`; a look, `manipulation.look`).

`ToolMeasure`, the default, calls the scene toolsets' `refine`, `remeasure`, `look_pose`,
and `estimate_hinge` tools in the same Hub; without them mounted nothing is refined (the note says
so). Before the first grasp or placement of an object whose registration was cut by an image
border, the arm first looks at it from `look_pose` and it is measured again there; a hinge turn
refines the part first and turns about the given hinge (`estimate_hinge` is only noted): the
given line is the joint's first guess, and the action API follows the part's joint from the
gripper's achieved positions (`ActionApi`, `manipulation.articulation`).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm, RobotState
from roboscript.manipulation.specs import HingeSpec
from roboscript.perception.registry import SceneRegistry, object_geometry
from roboscript.spatial.rotation import quat_angle_deg
from roboscript.tools import ToolContext
from roboscript.toolsets.scene import hinge_spec


class RefineConfig(BaseModel):
    """Whether grasps and placements go coarse, then fine (`enabled`), and which measurements
    at the pre-pose are taken: at least `min_confidence` sure, moving the object at most
    `max_shift_m` (`cut_max_shift_m`, when set, for an object whose registration an image border
    cut) and turning it at most `max_turn_deg`. With `look_refused` an action refused on the
    coarse geometry of such an object looks at it first and is built again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_shift_m: float = Field(default=0.05, gt=0.0)
    cut_max_shift_m: float | None = Field(default=0.1, gt=0.0)
    """The shift allowed for an object whose registration a border cut: the head view saw part of
    it, so the wrist camera near it may move it further; None: `max_shift_m`."""
    look_refused: bool = True
    """An action refused on the coarse geometry of such an object first looks at it
    (`look_pose`) and is built again; False: only an action that builds looks."""
    max_turn_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    doubt_shift_m: float | None = Field(default=0.02, gt=0.0)
    """A grasp does not close on an object whose measurement at the pre-pose was not taken as
    unsure (`low_confidence` or `inconsistent`, below `doubt_confidence`) yet moved it farther
    than this: the approach or a neighbour has likely knocked it, so the arm stops at the pre-pose
    and says so; None: it closes."""
    doubt_confidence: float = Field(default=0.05, ge=0.0, le=1.0)


class Measured(Protocol):
    """What a measurement reports (`toolsets.scene.MeasuredReport`): the object, the frame and
    cameras it was measured in, its pose, how far it moved, how sure it is, why it is (not)
    usable and whether the registry now has it."""

    obj: str
    frame: str
    cameras: list[str]
    pose: Pose
    shift_m: float
    confidence: float
    reason: str
    stored: bool


class Hinge(Protocol):
    """A hinge estimate (`perception.measure.HingeEstimate`)."""

    part: str
    side: str
    at: str
    confidence: float


class SceneMeasure(Protocol):
    """Measuring objects again (the scene toolsets'): `refine` reports, near `arm`'s wrist
    camera, without storing; `keep` stores a report deterministically (the same frame and
    cameras); `remeasure` measures and stores when sure; `look_pose` is where `arm`'s wrist
    camera sees an object whole; `hinge` estimates a part's hinge. None: nothing to be had."""

    async def refine(
        self, ctx: ToolContext, target: str, arm: Arm, part: str | None = None
    ) -> Measured | None: ...

    async def keep(self, ctx: ToolContext, measured: Measured) -> None: ...

    async def remeasure(self, ctx: ToolContext, obj: str) -> Measured | None: ...

    async def look_pose(self, ctx: ToolContext, obj: str, arm: Arm) -> Pose | None: ...

    async def hinge(self, ctx: ToolContext, obj: str, part: str) -> Hinge | None: ...


class ToolMeasure:
    """Through the scene toolsets' `refine`, `remeasure`, `look_pose` and `estimate_hinge`
    tools in the same Hub."""

    async def refine(
        self, ctx: ToolContext, target: str, arm: Arm, part: str | None = None
    ) -> Measured | None:
        return await _call(ctx, "refine", obj=target, arm=arm, part=part, store=False)  # type: ignore[no-any-return]

    async def keep(self, ctx: ToolContext, measured: Measured) -> None:
        args = {"frame": measured.frame, "cameras": measured.cameras, "store": True}
        await _call(ctx, "remeasure", obj=measured.obj, **args)

    async def remeasure(self, ctx: ToolContext, obj: str) -> Measured | None:
        return await _call(ctx, "remeasure", obj=obj)  # type: ignore[no-any-return]

    async def look_pose(self, ctx: ToolContext, obj: str, arm: Arm) -> Pose | None:
        found = await _call(ctx, "look_pose", obj=obj, arm=arm)
        return None if found is None else found.pose  # type: ignore[no-any-return]

    async def hinge(self, ctx: ToolContext, obj: str, part: str) -> Hinge | None:
        return await _call(ctx, "estimate_hinge", obj=obj, part=part)  # type: ignore[no-any-return]


async def _call(ctx: ToolContext, tool: str, **args: object) -> Any:
    try:
        return await ctx.call(tool, **args)
    except (LookupError, ToolError, ValidationError):
        return None


async def refine(
    ctx: ToolContext,
    measure: SceneMeasure,
    config: RefineConfig,
    targets: Mapping[Arm, Sequence[str]],
    part: str | None = None,
    doubts: dict[str, str] | None = None,
) -> list[str]:
    """Measure each arm's objects at its pre-pose and keep what may be kept (see the module;
    `part`: judged by that part of each); the notes for the receipt. `doubts` receives, per
    object, why a grasp should not close on it (`RefineConfig.doubt_shift_m`)."""
    robot = await ctx.env.robot(ctx.env_episode)
    notes: list[str] = []
    for arm, objects in targets.items():
        for obj in dict.fromkeys(objects):
            measured = await measure.refine(ctx, obj, arm, part)
            registry = await ctx.frames.registry()
            note, taken = _judge(ctx, registry, obj, part, arm, measured, robot, config)
            if taken:
                assert measured is not None
                await measure.keep(ctx, measured)
            elif doubts is not None and (doubt := _doubt(registry, obj, measured, config)):
                doubts[obj] = doubt
            notes.append(note)
    return notes


def _doubt(
    registry: SceneRegistry, obj: str, measured: Measured | None, config: RefineConfig
) -> str | None:
    """Why a grasp should not close on `obj` after a measurement that was not taken, if it
    should not (`RefineConfig.doubt_shift_m`)."""
    if measured is None or config.doubt_shift_m is None:
        return None
    if measured.reason not in ("low_confidence", "inconsistent"):
        return None
    if measured.confidence >= config.doubt_confidence or measured.shift_m <= config.doubt_shift_m:
        return None
    return (
        f"{registry.objects[obj].name} was seen {measured.shift_m * 100:.1f} cm from where it is "
        f"registered, but not surely (confidence {measured.confidence:.2f}): it may have been "
        "knocked, so the gripper did not close on the old estimate; observe it and register it "
        "again, then grasp"
    )


def _judge(
    ctx: ToolContext,
    registry: SceneRegistry,
    obj: str,
    part: str | None,
    arm: Arm,
    measured: Measured | None,
    robot: RobotState,
    config: RefineConfig,
) -> tuple[str, bool]:
    """Whether a measurement may be kept, and the receipt's note on it; logged."""
    name = registry.objects[obj].name + ("" if part is None else f" {part}")
    if measured is None:
        event = {"arm": arm, "object": obj, "part": part, "taken": False}
        ctx.emit("refine", {**event, "reason": "unavailable"})
        return f"{name} not refined (no measurement): kept the coarse estimate", False
    geometry = object_geometry(registry.objects[obj], robot)
    before = geometry.pose if part is None else geometry.parts[part].pose
    shift = float(np.linalg.norm(np.subtract(measured.pose.p, before[:3])))
    turn = float(quat_angle_deg(measured.pose.q, before[3:]))
    sure = measured.confidence >= config.min_confidence
    if measured.reason == "relocated":  # gone from where it was, found nearby: no bound on it
        event = {"arm": arm, "object": obj, "part": part, "taken": sure, "reason": "relocated"}
        ctx.emit("refine", {**event, "shift_mm": round(shift * 1000, 1),
                            "confidence": measured.confidence})  # fmt: skip
        if not sure:
            return f"{name} not where it was registered; a look-alike nearby not sure", False
        return f"{name} not where it was registered: found {shift * 1000:.0f} mm away", True
    limit = config.max_shift_m
    if registry.objects[obj].partially_visible and config.cut_max_shift_m is not None:
        limit = config.cut_max_shift_m
    taken = measured.reason == "ok" and sure and shift <= limit and turn <= config.max_turn_deg
    event = {"arm": arm, "object": obj, "part": part, "taken": taken, "reason": measured.reason}
    event |= {"shift_mm": round(shift * 1000, 1), "turn_deg": round(turn, 1)}
    ctx.emit("refine", {**event, "confidence": measured.confidence})
    correction = f"{shift * 1000:.0f} mm, {turn:.0f} deg, confidence {measured.confidence:.2f}"
    if not taken:
        kept = "kept the coarse estimate"
        return f"{name} refinement not taken ({measured.reason}, {correction}): {kept}", False
    return f"{name} refined by {correction}", True


async def remeasure(ctx: ToolContext, measure: SceneMeasure, obj: str) -> str | None:
    """Measure an object again (released, or missed by a grasp); the measurement stores itself
    when it is sure. A note for the receipt when it was found away from where it was."""
    measured = await measure.remeasure(ctx, obj)
    reason = "unavailable" if measured is None else measured.reason
    ctx.emit("remeasure", {"object": obj, "reason": reason})
    if measured is None or reason != "relocated":
        return None
    registry = await ctx.frames.registry()
    name = registry.objects[obj].name
    if not measured.stored:
        return f"{name} is no longer where it was registered (a look-alike nearby, not sure)"
    away = f"{measured.shift_m * 1000:.0f} mm"
    return f"{name} was knocked away: found {away} from where it was, registered there"


async def hinge(
    ctx: ToolContext, measure: SceneMeasure, obj: str, given: HingeSpec
) -> tuple[HingeSpec, str]:
    """The hinge a turn takes: always the given one (its side, height and offset, as a fit on the
    demonstration or the caller wrote it, with its angle's sign about that line); vision's
    estimate, when there is one, is only noted (logged with whether it agrees): an estimate on
    the far side of the grip (on a lid's front, the hinge at its back) would turn the arc the
    other way."""
    found = await measure.hinge(ctx, obj, given.part)
    event = {"object": obj, "part": given.part, "taken": False}
    if found is None:
        ctx.emit("hinge", {**event, "confidence": None})
        return given, f"{given.part}'s hinge not estimated: turned about the given one"
    spec = hinge_spec(found)
    agrees = (spec.side, spec.at) == (given.side, given.at)
    ctx.emit(
        "hinge",
        {**event, "confidence": found.confidence, "side": found.side, "at": found.at,
         "agrees": agrees},
    )  # fmt: skip
    if agrees:
        return given, f"{given.part}'s hinge on its {given.side} {given.at}, as estimated"
    return given, (
        f"{given.part}'s hinge turned about as given ({given.side} {given.at}); vision estimated "
        f"its {spec.side} {spec.at} (confidence {found.confidence:.2f}), not taken"
    )
