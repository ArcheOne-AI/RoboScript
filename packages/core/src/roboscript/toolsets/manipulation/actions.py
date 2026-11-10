"""One arm's action as tool parameters (docs/concepts/embodied-core.md): the action API's
calls without their arm, for `together` and `check_reach`, and the parameter types the single-arm
tools share with them (the action API builds their targets). Grasps, placements and hinge turns name
a registered object and take the specs the Memory holds (GraspSpec, PlaceSpec, HingeSpec;
`follow_path` a PathSpec).

The arm-axis rule (`move_axis` arm moves by z only) is part of validation, so a call breaking it
is refused as invalid arguments before anything is counted.
"""

from typing import Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    Field,
    FiniteFloat,
    PositiveInt,
    StringConstraints,
    ValidationInfo,
)

from roboscript.contracts.geometry import Quat, Vec3, unit_quaternion
from roboscript.contracts.robot import Arm
from roboscript.manipulation.api import ARM_AXIS_TAKES_Z, PAST_PLACE_M
from roboscript.manipulation.specs import GraspSpec, HingeSpec, PathSpec, PlaceSpec
from roboscript.tools import ObjectRef, ToolModel


def _unit_orientation(pose: tuple[float, ...]) -> tuple[float, ...]:
    w, x, y, z = pose[3:]
    return (*pose[:3], *unit_quaternion((w, x, y, z)))


def _arm_axis_takes_z(axis: str, info: ValidationInfo) -> str:
    """x and y come before `move_axis` in every model that has it, so they are validated."""
    if axis == "arm" and (info.data.get("x") or info.data.get("y")):
        raise ValueError(ARM_AXIS_TAKES_Z)
    return axis


ArmName = Annotated[Arm, Field(description="The arm: left or right.")]
EePose = Annotated[
    tuple[
        FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat
    ],
    AfterValidator(_unit_orientation),
    Field(
        description=(
            "The ee pose in the world frame (the wrist, not the TCP): x, y, z in meters, then its "
            "orientation qw, qx, qy, qz, a unit quaternion."
        )
    ),
]
ShiftX = Annotated[FiniteFloat, Field(description="Meters along world x (move_axis world).")]
ShiftY = Annotated[FiniteFloat, Field(description="Meters along world y (move_axis world).")]
_NUMBER_BY_NAME: Final = (
    " Or the name of the Memory's number parameter ({name}): the call then moves by its value, "
    "mirrored across the robot's middle when this scene's arms are the mirror image of the ones "
    "the Memory's specs were written for."
)
ShiftZ = Annotated[
    FiniteFloat,
    Field(
        description=(
            "Meters along world z; with move_axis arm, meters back along the gripper's pointing "
            "axis (positive backs away from where it points)."
        )
    ),
]
EndQuat = Annotated[
    Quat | None,
    Field(
        description=(
            "The ee orientation to end with, qw, qx, qy, qz: absolute, not a rotation relative to "
            "the current one. Null keeps the current orientation."
        )
    ),
]
MoveAxis = Annotated[
    Literal["world", "arm"],
    AfterValidator(_arm_axis_takes_z),
    Field(description="world: shift by x, y, z; arm: by z only, along the gripper."),
]
Position = Annotated[
    float, Field(ge=0.0, le=1.0, description="The gripper command: 1 fully open, 0 closed.")
]
Distance = Annotated[FiniteFloat, Field(ge=0.0, le=0.3)]
GraspApproachM = Annotated[
    Distance,
    Field(description="Meters back along the approach from the grasp point where it starts."),
]
GraspOffsetM = Annotated[
    Distance,
    Field(description="Meters back along the approach from the grasp point where it closes."),
]
PlaceApproachM = Annotated[
    Distance,
    Field(description="Meters back from the placement where the last straight move starts."),
]
StopBackM = Annotated[
    FiniteFloat,
    Field(
        ge=-PAST_PLACE_M,
        le=0.3,
        description=(
            "Meters back from the placement where the arm stops and opens; negative: past it, "
            f"down to -{PAST_PLACE_M}, pushing the object on along the approach."
        ),
    ),
]
Constrain = Annotated[
    Literal["free", "align", "auto"],
    Field(
        description=(
            "How the held object turns into the placement: free turns its placement point's z "
            "onto the target's; align also turns it about that axis until object_axis lies along "
            "align_axis; auto aligns by the gripper's own pose (the default)."
        )
    ),
]
AlignAxis = Annotated[
    list[Vec3] | None,
    Field(
        min_length=1,
        description="align: world directions object_axis may align with (the nearest); null: "
        "the target's x, which a stored placement is written against (leave it null unless "
        "the workflow's call names it).",
    ),
]
ObjectAxis = Annotated[
    Vec3, Field(description="align: the axis to align, in the held object's frame or the world.")
]
ObjectAxisType = Annotated[
    Literal["object", "world"], Field(description="The frame object_axis is given in.")
]
ApproachAxis = Annotated[
    Literal["grasp", "target_z"] | Vec3,
    Field(
        description=(
            "The direction of the last move: grasp, from the gripper toward the object; target_z, "
            "along the target pose's z (down onto an upright placement); or a vector in the "
            "target pose's frame."
        )
    ),
]
TheObject = Annotated[ObjectRef, Field(description="A registered object.")]
HeldObject = Annotated[ObjectRef, Field(description="The registered object the arm holds.")]
Angle = Annotated[
    FiniteFloat,
    Field(ge=-180.0, le=180.0, description="Degrees about the hinge, right-handed about it."),
]
Steps = Annotated[
    PositiveInt | None,
    Field(le=36, description="Straight moves along the arc; null: one per 10 degrees."),
]


SPEC_NAME_PATTERN: Final = r"^\{?[a-z][a-z0-9_]{0,39}\}?$"
SpecName = Annotated[str, StringConstraints(pattern=SPEC_NAME_PATTERN)]
"""A Memory spec parameter's name, as the workflow writes it (`{name}`, braces optional)."""
_BY_NAME = (
    "or the name of the Memory's {kind} parameter ({{name}}): the call then runs it with all its "
    "stored fields (axes and orientation hints included), mirrored when another arm acts"
)
GraspArg = Annotated[
    GraspSpec | SpecName,
    Field(
        description="Where and how to grasp: a grasp spec, " + _BY_NAME.format(kind="grasp_spec")
    ),
]
PlaceArg = Annotated[
    PlaceSpec | SpecName,
    Field(
        description="Where and how to place: a place spec, " + _BY_NAME.format(kind="place_spec")
    ),
]
HingeArg = Annotated[
    HingeSpec | SpecName,
    Field(description="The hinge: a hinge spec, " + _BY_NAME.format(kind="hinge_spec")),
]

PathArg = Annotated[
    PathSpec | SpecName,
    Field(
        description="The held path: a path spec, or the name of the Memory's path_spec parameter "
        "({name}), as fitted on the demonstration"
    ),
]
PathArms = Annotated[
    Literal["left", "right", "both"],
    Field(description="The arm holding what moves, or both: two arms holding one object."),
]


def spec_name(value: object) -> str | None:
    """The parameter a spec argument names (braces stripped); None for a spec written out."""
    return value.strip("{}") if isinstance(value, str) else None


ShiftXArg = Annotated[
    FiniteFloat | SpecName,
    Field(description="Meters along world x (move_axis world)." + _NUMBER_BY_NAME),
]
ShiftYArg = Annotated[
    FiniteFloat | SpecName,
    Field(description="Meters along world y (move_axis world)." + _NUMBER_BY_NAME),
]
ShiftZArg = Annotated[
    FiniteFloat | SpecName,
    Field(
        description=(
            "Meters along world z; with move_axis arm, meters back along the gripper's pointing "
            "axis (positive backs away from where it points)." + _NUMBER_BY_NAME
        )
    ),
]


class MoveToPose(ToolModel):
    """move_to_pose: move the ee to a pose along a planned path."""

    action: Literal["move_to_pose"]
    pose: EePose


class MoveBy(ToolModel):
    """move_by: move the ee from where it is now along a straight line."""

    action: Literal["move_by"]
    x: ShiftX = 0.0
    y: ShiftY = 0.0
    z: ShiftZ = 0.0
    quat: EndQuat = None
    move_axis: MoveAxis = "world"


class OpenGripper(ToolModel):
    """open_gripper: set the gripper to pos."""

    action: Literal["open_gripper"]
    pos: Position = 1.0


class CloseGripper(ToolModel):
    """close_gripper: set the gripper to pos; closing on nothing is no failure."""

    action: Literal["close_gripper"]
    pos: Position = 0.0


class ReturnHome(ToolModel):
    """return_home: move the ee back to its pose at the start of the episode."""

    action: Literal["return_home"]


class GraspObject(ToolModel):
    """grasp_object: grasp a registered object where the grasp spec says, then close."""

    action: Literal["grasp_object"]
    obj: TheObject
    grasp: GraspArg
    approach_m: GraspApproachM = 0.1
    grasp_offset_m: GraspOffsetM = 0.0
    gripper_pos: Position = 0.0


class PlaceObject(ToolModel):
    """place_object: carry the object the arm holds to where the place spec says, then open."""

    action: Literal["place_object"]
    obj: HeldObject
    place: PlaceArg
    approach_m: PlaceApproachM = 0.1
    stop_back_m: StopBackM = 0.02
    is_open: Annotated[bool, Field(description="Open the gripper at the end.")] = True
    constrain: Constrain = "auto"
    align_axis: AlignAxis = None
    object_axis: ObjectAxis = (1.0, 0.0, 0.0)
    object_axis_type: ObjectAxisType = "object"
    approach_axis: ApproachAxis = "grasp"


class RotateAbout(ToolModel):
    """rotate_about: turn the gripper, and what it holds, about a hinge of a registered object."""

    action: Literal["rotate_about"]
    obj: TheObject
    hinge: HingeArg
    angle_deg: Angle
    steps: Steps = None


ArmAction = Annotated[
    MoveToPose
    | MoveBy
    | OpenGripper
    | CloseGripper
    | ReturnHome
    | GraspObject
    | PlaceObject
    | RotateAbout,
    Field(discriminator="action"),
]
"""One arm's part of `together`, or an action `check_reach` plans."""


class StageArm(ToolModel):
    """An arm the Agent gives a stage in `plan_arms`."""

    stage: Annotated[str, Field(min_length=1, description="A workflow stage's name.")]
    arm: ArmName


class ArmMove(ToolModel):
    """One arm's action in `together`."""

    arm: ArmName
    action: ArmAction
