"""GraspSpec and PlaceSpec (docs/concepts/embodied-core.md): where and how to grasp and to
place, as the agent writes them, in place of asset annotations. Every number is relative (to an
object or a part, a landmark, the robot base or an action's own distance); no spec holds an object's
absolute position. `synthesis` turns them into contact and placement points and targets,
`fit` recovers them from observed poses, and Memory stores them as parameters.

A grasp is always relative to an object or a part. A placement may also target a constant point in
the robot base frame (`PlaceTarget` kind `robot_frame`): the base origin plus the offset in the
base axes, a task-level reference such as a hand-over point or the first block of a stack, which
carries nothing of one seed's objects.

A held path (`PathSpec`) is the demonstration's own motion of what a gripper held, fitted and
followed whole (`manipulation.paths`); it names no arm.

Grasp and place specs name the arm they were written for. The other arm uses a grasp's mirror
image, and a placement's when it says `mirror` (`synthesis.mirror`, `synthesis.for_arm`):
reflected across the robot's sagittal plane, through the part or target frame's axis nearest the
robot's lateral axis.
"""

import math
from typing import Annotated, Final, Literal

import numpy as np
from pydantic import ConfigDict, Field, FiniteFloat, StringConstraints, model_validator

from roboscript.contracts.refs import ROLE, Ref
from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ToolModel

NAME_PATTERN: Final = r"^[a-z][a-z0-9_]{0,39}$"
Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
"""Parameter, workflow-stage, object-role and part names, in snake_case."""

Landmark = Literal["table_center", "front_edge_mid", "robot_midline"]
Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Purpose = Literal["pick", "pull", "press"]
OTHER_ARM: Final = "other_arm"
"""`x_toward` naming no part: toward the other gripper's TCP (a hand-over's receiving grasp)."""
FAR_END: Final = "far_end"
"""`x_toward` naming no part: toward the end of the object's footprint farthest from the
footprint's own center (a pan's handle the part classifier merged with its rim, however it lies in
the footprint's box; `ObjectProxy.aim`)."""
LOW_END: Final = "low_end"
HIGH_END: Final = "high_end"
"""`x_toward` naming no part: toward the end of the object's long side that stands lower (or
higher) than the other (`ObjectProxy.end_heights`): a front told by height alone, which every
measurement of the object finds again the same way, where a part a VLM names may come out at
either end (the same part can be the low end of one object and the high end of another)."""
GEOMETRIC_ENDS: Final = (FAR_END, LOW_END, HIGH_END)
"""The `x_toward` values that name a place on the object's geometry, not a registered part."""
END_BAND: Final = 0.25
"""An end of an object's long side: its outer quarter of the length."""
END_HEIGHT_MIN_M: Final = 0.015
"""Two ends tell a front by height only where their tops lie this far apart at least."""
TILT_LIMIT_DEG: Final = 75.0
FRACTION_LIMIT: Final = 1.0
APEX_BAND_M: Final = 0.001
RAY_RADIUS_M: Final = 0.003
CATCH_M: Final = 0.03
"""A surface this far from an apex grasp's approach line along its closing axis still ends up
between the fingers (open, they stand about 4.5 cm either side of the TCP): where the line meets
none (a thinner or wider rim on the next model), the grasp moves along that axis onto the nearest
such surface."""
FREE_TURN_STEP_DEG: Final = 15.0
"""A placement's free turn is searched in steps of this (deg)."""
POSE_AXES_COS: Final = 0.5
"""A pose grasp's closing axis lies at least 60 deg off its approach (it is made square to it)."""
REGION_LIMIT_M: Final = 0.15
"""A placement region's radius at most (a larger one only adds options to search)."""


class SpecError(ValueError):
    """A spec that cannot be made on the scene as it is (a part the object lacks, an aim with
    nothing to aim at): the caller's spec, not a fault of the code. The tools return it to the
    Agent as a refusal; checks outside the tools (a dry run) report it against the spec."""


_Anchor = Annotated[
    str,
    StringConstraints(pattern=r"^(top_center|inside_bottom|center|edge:(front|back|left|right))$"),
]


class _Frozen(ToolModel):
    model_config = ConfigDict(frozen=True)


Fraction = Annotated[FiniteFloat, Field(ge=-FRACTION_LIMIT, le=FRACTION_LIMIT)]


class GraspSpec(_Frozen):
    """Where and how to grasp: a part of the object, the approach (from above, from the side or
    both), an anchor on the part, an offset from it in the part frame (z up, x along the long
    axis toward `x_toward`): `offset_m` in meters plus `offset_frac`, fractions of the part's
    extents along its axes, so that one spec lands on the same place of a larger or smaller
    part (a rocker's lower half, a handle's middle), the closing axis, a yaw about the
    approach, a tilt of the approach about the closing axis (positive tilts it toward the
    closing axis crossed with the approach's reverse, e.g. +y for a top grasp closing along x)
    and a lean of the approach and the closing axis together about the axis across both
    (positive leans the approach toward the closing axis's reverse, e.g. -x for that grasp).
    The `apex` anchor is the part's most prominent point toward the approaching gripper (the
    middle of its surface points within `APEX_BAND_M` of the foremost: a rocker's pressed-out
    edge, a button's top), found on each part anew; from it the offset across the approach is
    taken as usual, and the one along it from the surface the approach then meets (its points
    within `RAY_RADIUS_M` of the approach line), so that fingers go as deep into any part.
    `purpose` says what the hold is for, and so which receipt checks count as a hard failure: `pick`
    (an empty grasp or a lost hold), `pull` (an empty grasp; a pull may slip off at its end) or
    `press` (none: closed fingers push, so the closing axis is only a preference). The `pose`
    anchor writes the grasp as the TCP's pose in the whole object's frame (its box, x toward
    `x_toward` where that names something, else either way the box allows), as the
    demonstration's grasp was: `offset_m` the TCP's position from the box's center,
    `approach_axis` the direction the gripper moves in onto the object (the TCP's pointing
    axis) and `closing_axis` the one its fingers close along, both in that frame; no part,
    fraction, closing extent, yaw, tilt or lean applies, so a part re-registered on the next
    scene cannot move it. `arm` is the
    arm it was written for; the other arm uses its mirror image (`synthesis.mirror`); null
    serves either arm as written. `reference_q`: the grasp's orientation in the robot base frame
    where it was fitted, used only to order candidates: where the object's
    frame leaves several (its turns), those within `api.FACING_DEG` of it are tried first; it
    never moves a grasp, removes a candidate or overrides plannability."""

    part: Name = "whole"
    approach: Literal["top", "side", "auto"] = "auto"
    anchor: Literal["top_center", "side_center", "center", "bottom_center", "apex", "pose"] = (
        "center"
    )
    x_toward: Annotated[
        Name | None,
        Field(
            description="The part the frame's x points to, or other_arm: toward the other "
            "gripper, or far_end: toward its footprint's far end; null: both ways."
        ),
    ] = None
    offset_m: Vec3 = (0.0, 0.0, 0.0)
    offset_frac: tuple[Fraction, Fraction, Fraction] = (0.0, 0.0, 0.0)
    closing: Literal["minor", "major"] = "minor"
    yaw_deg: FiniteFloat = 0.0
    tilt_deg: Annotated[FiniteFloat, Field(ge=-TILT_LIMIT_DEG, le=TILT_LIMIT_DEG)] = 0.0
    lean_deg: Annotated[FiniteFloat, Field(ge=-TILT_LIMIT_DEG, le=TILT_LIMIT_DEG)] = 0.0
    purpose: Purpose = "pick"
    reference_q: Annotated[
        tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None,
        Field(
            description="The grasp's orientation (w, x, y, z) in the robot base frame where it "
            "was fitted: candidates near it are tried first; null: none."
        ),
    ] = None
    arm: Annotated[
        Arm | None, Field(description="The arm it was written for; null: either arm as written.")
    ] = None
    approach_axis: Annotated[
        Vec3 | None,
        Field(
            description="With anchor pose: the direction the gripper moves in onto the object "
            "(the TCP's pointing axis), in the whole object's frame; null otherwise."
        ),
    ] = None
    closing_axis: Annotated[
        Vec3 | None,
        Field(
            description="With anchor pose: the axis the fingers close along, in the whole "
            "object's frame; null otherwise."
        ),
    ] = None

    @model_validator(mode="after")
    def _pose_form(self) -> "GraspSpec":
        axes = (self.approach_axis, self.closing_axis)
        if self.anchor != "pose":
            if any(axis is not None for axis in axes):
                raise ValueError("approach_axis and closing_axis belong to the pose anchor")
            return self
        if self.approach_axis is None or self.closing_axis is None:
            raise ValueError("the pose anchor needs approach_axis and closing_axis")
        if self.part != "whole":
            raise ValueError("the pose anchor is in the whole object's frame: part must be whole")
        shaped = {
            "offset_frac": self.offset_frac != (0.0, 0.0, 0.0),
            "yaw_deg": self.yaw_deg != 0.0,
            "tilt_deg": self.tilt_deg != 0.0,
            "lean_deg": self.lean_deg != 0.0,
        }
        if extra := sorted(name for name, given in shaped.items() if given):
            raise ValueError(f"the pose anchor takes no {', '.join(extra)}: its axes say all")
        a, c = (np.asarray(axis, dtype=float) for axis in axes)
        if min(np.linalg.norm(a), np.linalg.norm(c)) < 1e-6:
            raise ValueError("approach_axis and closing_axis must be directions, not zero")
        cos = abs(float(a @ c)) / float(np.linalg.norm(a) * np.linalg.norm(c))
        if cos > POSE_AXES_COS:
            raise ValueError("closing_axis must lie across approach_axis (at least 60 deg off)")
        return self


class HingeSpec(_Frozen):
    """Where a part turns: a line on one side of the part's upright box (`front`: the face
    toward the robot, `back`: away from it, `left` and `right` as the robot sees them; the face
    whose outward normal is nearest that direction), horizontal along that side at its bottom,
    middle or top, or upright through its middle, moved by `offset_m` (meters: across the side
    (up x its outward normal), out of it, up). A turn about it is right-handed about its
    direction: up x the side's outward normal for a horizontal line, so positive angles lift the
    part's far side; up for an upright one, so positive angles turn counterclockwise seen from
    above."""

    part: Annotated[Name, Field(description="The part that turns; whole: the object's box.")] = (
        "whole"
    )
    side: Literal["front", "back", "left", "right"]
    at: Literal["bottom", "middle", "top", "upright"] = "bottom"
    offset_m: Vec3 = (0.0, 0.0, 0.0)


MAX_PATH_POSES: Final = 32
"""A held path keeps at most this many poses (the demonstration's keypoints from a close)."""
PathPose = tuple[
    FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat
]


class PathSpec(_Frozen):
    """A held path (`manipulation.paths`): what a gripper holding an object or a part did with it,
    as the TCP's poses (x, y, z in meters, then qw, qx, qy, qz) in the body frame of what it
    holds, the first where the gripper closed. The body frame is the whole object's (its box, as
    a pose grasp's: x toward its front where its geometry decides one, `x_toward` where only
    height tells it), also where the arm holds
    one of its articulated parts (the part moves; the body it turns on does not). What the path
    reproduces is the motion: `follow_path` moves what the arms hold by the same rigid motion
    relative to that frame, each gripper from where it holds (so a grasp a few millimeters off
    is carried along), and a door's or a lid's arc needs no hinge. `arm` names the arm it was
    fitted at: the other arm follows its mirror image (reflected through the body frame across
    the vertical plane nearest the robot's sagittal one, as a grasp is); both arms holding one
    object follow it as written; one naming no arm is followed as written, else, when that is
    out of reach from the start, at its mirror image."""

    poses: Annotated[
        list[PathPose],
        Field(
            min_length=2,
            max_length=MAX_PATH_POSES,
            description="The TCP's poses (x, y, z, qw, qx, qy, qz) in the held object's body "
            "frame, the first at the close.",
        ),
    ]
    x_toward: Annotated[
        Name | None,
        Field(
            description="The end of the held object its body frame's x points to, where only "
            "height tells it (low_end or high_end); null: the object's front as its part or "
            "footprint decides it (as fitted)."
        ),
    ] = None
    arm: Annotated[
        Arm | None,
        Field(
            description="The arm it was fitted at (the other arm follows its mirror image); "
            "null: either arm as written (its mirror image when as written is out of reach)."
        ),
    ] = None

    @model_validator(mode="after")
    def _unit_orientations(self) -> "PathSpec":
        for pose in self.poses:
            norm = math.sqrt(sum(v * v for v in pose[3:]))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"a path pose's orientation {list(pose[3:])} is no unit quaternion"
                )
        return self


class PlaceTarget(_Frozen):
    """What to place at: an anchor on an object (or one of its parts), a table landmark, or the
    robot base frame (a constant point: the base origin, the offset in the base axes)."""

    kind: Literal["object", "landmark", "robot_frame"]
    obj: Annotated[
        Annotated[Name, Ref(ROLE)] | None,
        Field(description="An object role from OBJECTS (or the object's alias)."),
    ] = None
    part: Annotated[Name | None, Field(description="A part of the object; null: whole.")] = None
    anchor: _Anchor | None = None
    landmark: Landmark | None = None

    @model_validator(mode="after")
    def _fields_fit_the_kind(self) -> "PlaceTarget":
        given = {name for name in ("obj", "part", "anchor", "landmark") if getattr(self, name)}
        allowed, needed = {
            "object": ({"obj", "part", "anchor"}, {"obj", "anchor"}),
            "landmark": ({"landmark"}, {"landmark"}),
            "robot_frame": (set(), set()),
        }[self.kind]
        if extra := sorted(given - allowed):
            raise ValueError(f"a {self.kind} target takes no {extra}")
        if missing := sorted(needed - given):
            raise ValueError(f"a {self.kind} target needs {missing}")
        return self


class PlaceSpec(_Frozen):
    """Where and how to place: the target, the axes the offset is in (the target's or the
    world's, anchored at the target), the offset, the held object's orientation (`keep` its
    current one, `upright` keeping its yaw, `yaw` = `yaw_deg` from the frame's x, `relative` =
    `rpy_deg` (roll, pitch, yaw about the frame's fixed x, y, z) from the frame, for tilted
    placements) and its placement point (`bottom`, `top`, `center`, `part:<name>`: a part of
    that name or kind, at `placement_point_at` fractions of its extent from its pivot, its root end
    along a known x and its bottom, else at its bottom face center). The
    placement turns the object about its placement point's z only with `constrain:
    align`; else it keeps the object's yaw, so `yaw` and `upright` need `align`. `region_m` > 0
    makes the target a horizontal disk of that radius (drops, piles): its center first, then
    rings, the first plannable point wins. `arm` is the arm it was written for; with `mirror`
    the other arm places at its mirror image, else as written, or at the mirror image when no
    option as written is within its reach (a tilted spec onto an object with no `arm` too: it
    was still measured at one arm's wrist). `x_toward` names the held object's part its frame's
    x points to (`ObjectProxy.facing`: a handle), what `relative` and the placement's object axis
    are written in; null: its frame as measured. `approach` directs the
    placement: the direction (target frame) the held object moves in onto it, the last `approach_m`
    (onto a pin, into a slot), and `free_turn_deg` lets it turn up to that far either way about
    the approach (else about the vertical) through the placement point, nearest first, the
    first option within reach winning (an object hung by a loop or set into a hollow)."""

    target: PlaceTarget
    frame: Literal["target", "world_axes"] = "target"
    offset_m: Vec3 = (0.0, 0.0, 0.0)
    orient: Annotated[
        str,
        StringConstraints(pattern=r"^(keep|upright|yaw|relative)$"),
    ] = "keep"
    yaw_deg: FiniteFloat = 0.0
    rpy_deg: Vec3 = (0.0, 0.0, 0.0)
    placement_point: Annotated[
        str, StringConstraints(pattern=r"^(bottom|top|center|part:[a-z][a-z0-9_]*)$")
    ] = "bottom"
    placement_point_at: tuple[Fraction, Fraction, Fraction] | None = None
    x_toward: Annotated[
        Name | None,
        Field(
            description="The held object's part its frame's x points to (or far_end: its "
            "footprint's far end; low_end or high_end: the end of its long side whose top is "
            "lower or higher, a front told by height alone), for the orientation; null: its "
            "frame as measured."
        ),
    ] = None
    approach: Annotated[
        Vec3 | None,
        Field(
            description="The direction (target frame) the held object moves in onto the "
            "placement; null: as the call's approach_axis says."
        ),
    ] = None
    free_turn_deg: Annotated[
        FiniteFloat,
        Field(
            ge=0.0,
            le=180.0,
            description="How far (deg) the placement may turn either way about its approach "
            "to be within reach; 0: as written.",
        ),
    ] = 0.0
    region_m: Annotated[FiniteFloat, Field(ge=0.0, le=REGION_LIMIT_M)] = 0.0
    receiver: Annotated[
        GraspSpec | None,
        Field(
            description="The grasp the other arm takes the object with where it is placed: only "
            "options where that arm reaches it clear of this gripper count; null: none."
        ),
    ] = None
    arm: Annotated[
        Arm | None,
        Field(
            description="The arm it was written for (the demonstration's arm where it was "
            "fitted); null: either arm as written (a tilted placement onto an object at its "
            "mirror image when as written is out of reach)."
        ),
    ] = None
    mirror: Annotated[
        bool,
        Field(
            description="The other arm places at the mirror image (true) or as written "
            "(false; the mirror image only when as written is out of its reach)."
        ),
    ] = False

    @model_validator(mode="after")
    def _approach_has_a_direction(self) -> "PlaceSpec":
        if self.approach is not None and sum(v * v for v in self.approach) < 1e-12:
            raise ValueError("approach must be a direction, not zero")
        return self
