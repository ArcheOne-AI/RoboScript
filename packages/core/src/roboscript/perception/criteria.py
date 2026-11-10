"""Done criteria as data (docs/concepts/perception.md, docs/concepts/learning.md):
`DoneSpec`, the metric vocabulary and its units, bounds and perception floors. A done criterion is a
Task Memory parameter type (`memory.params.DoneSpecParam`), so the Memory engine types, renders and
bounds it from here; measuring one is a method's (e.g. wicl's `perception.done`).

The metrics: `distance_xy`, `offset_x`, `offset_y`,
`gap_z`, `overlap`, `inside`, `yaw_diff` (binary: role `a` relative to role `b`, or `table` for
`gap_z`) and `held`, `surface_change`, `press` (unary).

Also the data the done check is configured and fed with, which the core passes around: a
demonstration's end as targets (`EndTarget`, read e.g. by wicl's `perception.endstate`) and the
check's settings (`DoneCheckConfig`, the scene toolsets' `done` config).
"""

from typing import Annotated, Final, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ToolModel

Metric = Literal[
    "distance_xy",
    "offset_x",
    "offset_y",
    "gap_z",
    "overlap",
    "inside",
    "yaw_diff",
    "held",
    "surface_change",
    "press",
]
METRICS: Final[tuple[str, ...]] = get_args(Metric)
UNARY: Final = frozenset({"held", "surface_change", "press"})
PARTED: Final = frozenset({"surface_change", "press"})
"""The metrics that may measure a part of `a`."""
TABLE: Final = "table"
"""The `b` of a `gap_z` criterion measured against the table."""

UNITS: Final[dict[str, tuple[str, tuple[float, float], float]]] = {
    "distance_xy": ("m", (0.0, 1.0), 0.01),
    "offset_x": ("m", (-1.0, 1.0), 0.01),
    "offset_y": ("m", (-1.0, 1.0), 0.01),
    "gap_z": ("m", (-0.5, 0.5), 0.01),
    "overlap": ("fraction", (0.0, 1.0), 0.1),
    "inside": ("flag", (0.0, 1.0), 0.5),
    "yaw_diff": ("deg", (0.0, 90.0), 10.0),
    "held": ("flag", (0.0, 1.0), 0.5),
    "surface_change": ("m", (0.0, 0.1), 0.003),
    "press": ("m", (0.0, 0.5), 0.01),
}
"""Per metric: unit, the bounds of its value, how far one revision may move value or tolerance."""

TOL_FLOOR_M: Final = 0.01
"""The narrowest band (m) a done tolerance on where an object is can have: perception's error on
an object's position. A band narrower is noise to the
check, not a stricter one."""
TOL_FLOOR_DEG: Final = 5.0
"""The narrowest band (deg) a yaw tolerance can have, by the same argument (box fits and part
axes are off by a few degrees; `endstate.YAW_TOL_DEG` keeps a degree more)."""
FLOORS: Final[dict[str, float]] = {
    **dict.fromkeys(("distance_xy", "offset_x", "offset_y", "gap_z", "press"), TOL_FLOOR_M),
    "yaw_diff": TOL_FLOOR_DEG,
}
"""The metrics whose tolerance is held to perception's error (`floored`); the others are flags,
shares or surface depth changes, whose noise is their own."""

_Role = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[a-z0-9][a-z0-9_]*$")]


class DoneSpec(ToolModel):
    """One done criterion: `metric` of role `a` (its part `part` for `surface_change` and
    `press`), relative to role `b` (or `table` for `gap_z`) for a binary metric; it holds when
    the measured value is within `tol` of `value`. `arm` (`offset_x` only): the value is where
    `a` ends with that arm acting on it, a side the demonstration's arm gave it; with the other
    arm acting it mirrors (`for_arm`). None: the side is the robot's, fixed (a task that names
    it)."""

    model_config = ConfigDict(frozen=True)

    metric: Metric
    a: _Role
    b: _Role | None = None
    part: _Role | None = None
    value: FiniteFloat
    tol: PositiveFloat
    arm: Arm | None = None

    @model_validator(mode="after")
    def _form(self) -> "DoneSpec":
        unary = self.metric in UNARY
        if unary != (self.b is None):
            raise ValueError(f"{self.metric} {'takes no b' if unary else 'needs b'}")
        if self.a == TABLE:
            raise ValueError("a is an object role: the table is only gap_z's b")
        if self.b == TABLE and self.metric != "gap_z":
            raise ValueError("only gap_z is measured against the table")
        if self.b == self.a:
            raise ValueError("a and b must be different roles")
        if self.arm is not None and self.metric not in LATERAL:
            raise ValueError("only offset_x, a side of the robot, is relative to an arm")
        if self.part is not None and self.metric not in PARTED:
            raise ValueError("only surface_change and press measure a part")
        low, high = UNITS[self.metric][1]
        if not low <= self.value <= high:
            raise ValueError(f"{self.metric} value {self.value:g} is outside [{low:g}, {high:g}]")
        if self.tol > high - low:
            raise ValueError(
                f"{self.metric} tol {self.tol:g} is wider than its span {high - low:g}"
            )
        return self

    def label(self) -> str:
        """The criterion without its numbers: `metric a[.part] [b]`."""
        a = self.a if self.part is None else f"{self.a}.{self.part}"
        return " ".join(x for x in (self.metric, a, self.b) if x is not None)


LATERAL: Final = frozenset({"offset_x"})
"""The metrics along the robot's lateral axis (x, right), which an arm's side mirrors."""


def for_arm(spec: DoneSpec, arm: Arm | None) -> DoneSpec:
    """`spec` as it holds with `arm` acting on its `a`: written for its own `arm` as it is, for
    the other arm mirrored across the robot's sagittal plane (the lateral value negated); one
    with no `arm`, or with the acting arm not known, as written."""
    if spec.arm is None or arm is None or arm == spec.arm or spec.metric not in LATERAL:
        return spec
    return spec.model_copy(update={"value": -spec.value, "arm": arm})


def floored(spec: DoneSpec) -> DoneSpec:
    """`spec` with its tolerance at least perception's error (`FLOORS`): a band narrower than the
    check can resolve is read at the floor."""
    floor = FLOORS.get(spec.metric)
    if floor is None or spec.tol >= floor:
        return spec
    return spec.model_copy(update={"tol": floor})


EndKind = Literal["holds", "lift", "yaw", "place", "overlap", "pair", "base", "act"]
_EndRole = Annotated[str, Field(min_length=1, max_length=40)]


class EndTarget(BaseModel):
    """One condition of the demonstration's end on role `a` (see the module): `value` (arms,
    m above the table, world yaw deg, or footprint share), `offset` (place and pair: a's center
    minus `b`'s, xy m, world axes; base: a's center in the robot's base frame, xy m), `tol`,
    `period` (yaw: the turn it looks the same under, deg), `peers` (place: the roles
    interchangeable with `a` on `b`) and `mirror`: its mirror image across the robot's sagittal
    plane holds as well; base: `arm`, the arm that holds `a` at the demonstration's end (None:
    none, or not known), and `carried`, that arm moved it there (`annotate_holds`); `motion`:
    the world xy displacement that carried `a` into the demonstration's success latch (pair:
    a's less b's; None: not moving there), its last motion (`annotate_motion`); act: `offset`,
    a's acting end's center minus b's (xy, world axes), `value`, its bottom above b's top (m)."""

    model_config = ConfigDict(frozen=True)

    kind: EndKind
    a: _EndRole
    b: _EndRole | None = None
    value: FiniteFloat = 0.0
    offset: tuple[FiniteFloat, FiniteFloat] | None = None
    tol: PositiveFloat
    period: PositiveFloat = 360.0
    peers: list[str] = []
    mirror: bool = False
    arm: Arm | None = None
    carried: bool = False
    motion: tuple[FiniteFloat, FiniteFloat] | None = None

    def label(self) -> str:
        return " ".join(x for x in ("end", self.kind, self.a, self.b) if x is not None)


class DoneCheckConfig(BaseModel):
    """How `check_done` holds the Memory's criteria to what they can tell apart (the scene
    toolsets' `done` config). `coverage`: an object the episode took (an arm held it) needs a
    criterion or an end target that says where it ends (`POSITIONAL` among the criteria; an end
    target's place, pair, base or lift), else the check fails for it: a done check blind to an
    object that was moved cannot tell a failed placement from a success. `tol_cap`: a horizontal
    criterion's band (`distance_xy`, `offset_x`, `offset_y`) is at most `cap_extent` of the
    smallest horizontal extent of its objects, and a `gap_z` over the table at most `cap_lift` of
    its value, never below `cap_floor_m`: a band wider than the object itself accepts a placement
    beside the target. `end_pairs`: `read_end_targets` also reads the demonstration's end as
    pair targets (where moved objects end relative to each other) and base targets (where an
    unanchored one ends in the robot's base frame), which carry over to other seeds in their
    mirror image too, and `check_done` measures pair and base targets it is given; `end_targets`:
    `check_done` measures the demonstration's end targets it is given at all (off: it ignores
    them); `release_fallback`: at the demonstration's
    end, an object its arm let go of that the last keyframe does not measure again with
    confidence stays where the replayed release left it, instead of being left out (and with it
    every target on it: `toolsets.worldstate.demo_end_reason`); `relative_height`: a lift
    criterion (a `gap_z` over the table of an object in the air) and a lift end target also hold
    the object's height over the robot's base to the demonstration's (`lift_excess`), when the
    done fact names the demonstration's frame (`reference`); `latch_margin_m`: when the
    demonstration's last keyframe is its success frame (the done fact's `latched`: the recording
    stops the moment the official check latches, so a lift there is at the threshold, not past
    it), a lift criterion and a lift end target are one-sided minimums: the object at least this
    much above the demonstration's height (perception's height noise), overshoot passing, never a
    band that reaches below the latch value, and a place, pair or base end target of a role an
    arm was still carrying into the latch (`EndTarget.motion`, `endstate.annotate_motion`) is
    one-sided along that motion by as much (0: a plain band).
    `stuck_after`: after this many failing calls in a row that made no progress (no call with
    fewer failing lines than the best since the arms last took or let go of something, or a role
    left undecided that often), `check_done` says plainly that the corrections are not closing
    the gap and names what else to try, the demonstration's motion into its latch first;
    `give_up_after`: after this many calls without progress (a role left undecided does not count:
    it is a perception gap, not a sign the task cannot still latch), it stops holding `finish`
    (`DoneReport.gave_up`), so an episode that cannot be confirmed done may stop instead of
    checking for hours; 0: never. `give_up_identical`: after this many failing calls in a row with
    the same failing lines (since the arms last took or let go of something), it stops holding
    `finish` too: the Actor's corrections leave the verdict as it was; 0: never."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: bool = True
    tol_cap: bool = True
    cap_extent: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    cap_lift: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    cap_floor_m: PositiveFloat = 0.01
    end_pairs: bool = True
    end_targets: bool = True
    release_fallback: bool = True
    relative_height: bool = True
    latch_margin_m: Annotated[float, Field(ge=0.0, le=0.05)] = 0.01
    stuck_after: NonNegativeInt = 3
    give_up_after: NonNegativeInt = 30
    give_up_identical: NonNegativeInt = 15

    @model_validator(mode="after")
    def _give_up_after_stuck(self) -> Self:
        for name in ("give_up_after", "give_up_identical"):
            cap = getattr(self, name)
            if cap and cap < self.stuck_after:
                raise ValueError(
                    f"{name} {cap} must be 0 or at least stuck_after {self.stuck_after}: the "
                    "Actor is told what else to try before the check gives up"
                )
        return self
