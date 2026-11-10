"""The parameter table (`params.json`, docs/concepts/embodied-core.md and
docs/concepts/learning.md).

Memory text only names parameters, as `{name}`; their values live here. A parameter is a number, a
category, a GraspSpec / PlaceSpec / HingeSpec / PathSpec (`roboscript.manipulation.specs`) or a
done criterion (`DoneSpec`, `perception.criteria`; wicl's `check_done` measures every one;
its `value` and `tol` are numeric, `metric`, `a`, `b` and `part` categorical) or a written
skill's parameters (`skill_spec`: the skill's name and its parameter object as fitted on the
demonstration, `tools.skill_fit`; taken whole, like a held path), and carries
its value, unit, bounds, trust radius, frame, source and history. A spec's fields are either
numeric (`offset_m`, `yaw_deg`, a GraspSpec's `offset_frac`, `tilt_deg` and `lean_deg`, a
PlaceSpec's `rpy_deg` and `region_m`; a HingeSpec's `offset_m` only: interpolated on merge) or
categorical (replaced; a PlaceSpec's `mirror` and `approach` too; a hinge's `part`, `side` and
`at`; its angle is a number of its own; a PathSpec's `poses`, a held path as fitted, whole);
to keep that split, PlaceSpec's `orient` `yaw_deg:<v>` is `orient: yaw` plus the numeric
`yaw_deg`. Every offset is relative (to an object, a part, a landmark, the TCP or, for a
placement, the robot base); there is no world frame to put an absolute position in.

The `*Def` models are what `define_param` takes (`kind` required); the stored `*Param` models add
the harness's `source` and `history`.
"""

import json
import math
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    PositiveFloat,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from pydantic_core import to_jsonable_python

from roboscript.contracts.tools import ToolModel
from roboscript.manipulation.specs import GraspSpec, HingeSpec, Name, PathSpec, PlaceSpec
from roboscript.perception.criteria import UNITS, DoneSpec

Label = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_:.+-]{0,59}$")]
"""A category value."""

Frame = Literal["object", "part", "tcp", "landmark", "robot_frame"]

FITTED_SOURCE: Final = "demo"
"""How the source of a spec fitted on the demonstration begins: `demo left K2`."""


def _ordered(bounds: tuple[float, float]) -> tuple[float, float]:
    if bounds[0] > bounds[1]:
        raise ValueError(f"bounds {list(bounds)} must be [low, high]")
    return bounds


Bounds = Annotated[tuple[FiniteFloat, FiniteFloat], AfterValidator(_ordered)]

GRASP_NUMBERS: Final = ("offset_m", "offset_frac", "yaw_deg", "tilt_deg", "lean_deg")
PLACE_NUMBERS: Final = ("offset_m", "yaw_deg", "rpy_deg", "region_m")
GRASP_CATEGORIES: Final = ("part", "approach", "anchor", "x_toward", "closing", "purpose", "arm")
PLACE_CATEGORIES: Final = (
    "target",
    "frame",
    "orient",
    "placement_point",
    "arm",
    "mirror",
    "approach",
)
GRASP_OTHERS: Final = ("reference_q", "approach_axis", "closing_axis")
PLACE_OTHERS: Final = ("placement_point_at", "x_toward", "free_turn_deg", "receiver")
"""A spec's fields that are neither numbers nor categories (a pose grasp's axes, its orientation
hint, a placement's search and receiver): a merge takes them whole, with the spec, like a category;
no `set_number` or `set_category` targets them."""
HINGE_NUMBERS: Final = ("offset_m",)
HINGE_CATEGORIES: Final = ("part", "side", "at")
PATH_CATEGORIES: Final = ("poses", "x_toward")
DONE_NUMBERS: Final = ("value", "tol")
DONE_CATEGORIES: Final = ("metric", "a", "b", "part")


def _json_object(text: str) -> str:
    try:
        found = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not JSON: {error}") from None
    if not isinstance(found, dict):
        raise ValueError("the skill's parameters are a JSON object")
    return text


class _Frozen(ToolModel):
    model_config = ConfigDict(frozen=True)


class SpecTrust(_Frozen):
    """How far one revision may move a spec's numbers: the offset's length of change (m), a
    grasp's fractional offset's (of the part's extents), yaw and a grasp's tilt and lean (deg), a
    placement's roll-pitch-yaw's length of change (deg) and region radius (m)."""

    offset_m: PositiveFloat = 0.02
    offset_frac: PositiveFloat = 0.15
    yaw_deg: PositiveFloat = 15.0
    tilt_deg: PositiveFloat = 15.0
    lean_deg: PositiveFloat = 15.0
    rpy_deg: PositiveFloat = 15.0
    region_m: PositiveFloat = 0.02


class SpecBounds(_Frozen):
    """Hard limits of a spec's numbers: each offset component (m; from the robot base origin
    for a robot-frame target), each component of a grasp's fractional offset, yaw, a grasp's tilt
    and lean, and each roll-pitch-yaw component of a placement (deg), a placement region's radius
    (m)."""

    offset_m: Bounds = (-0.3, 0.3)
    offset_frac: Bounds = (-1.0, 1.0)
    robot_frame_m: Bounds = (-1.5, 1.5)
    yaw_deg: Bounds = (-180.0, 180.0)
    tilt_deg: Bounds = (-75.0, 75.0)
    lean_deg: Bounds = (-75.0, 75.0)
    rpy_deg: Bounds = (-180.0, 180.0)
    region_m: Bounds = (0.0, 0.15)


class NumberDef(_Frozen):
    """A number: value within bounds, unit (m, deg, ...), trust radius, and the frame it is
    relative to (null for scalars such as distances)."""

    kind: Literal["number"]
    value: FiniteFloat
    unit: Annotated[str, StringConstraints(max_length=12)]
    bounds: Bounds
    trust: PositiveFloat
    frame: Frame | None = None

    @model_validator(mode="after")
    def _in_bounds(self) -> "NumberDef":
        _check_bounds("value", self.value, self.bounds)
        return self


class CategoryDef(_Frozen):
    """A category: its value and, if given, the values it may take."""

    kind: Literal["category"]
    value: Label
    choices: list[Label] | None = None

    @model_validator(mode="after")
    def _in_choices(self) -> "CategoryDef":
        if self.choices is not None:
            if not self.choices or len(set(self.choices)) != len(self.choices):
                raise ValueError("choices must be distinct values, at least one")
            if self.value not in self.choices:
                raise ValueError(f"{self.value!r} is not one of the choices {self.choices}")
        return self


class GraspSpecDef(_Frozen):
    kind: Literal["grasp_spec"]
    value: GraspSpec
    trust: SpecTrust = SpecTrust()
    bounds: SpecBounds = SpecBounds()

    @model_validator(mode="after")
    def _in_bounds(self) -> "GraspSpecDef":
        _check_spec_bounds(self.value, self.bounds)
        return self


class PlaceSpecDef(_Frozen):
    kind: Literal["place_spec"]
    value: PlaceSpec
    trust: SpecTrust = SpecTrust()
    bounds: SpecBounds = SpecBounds()

    @model_validator(mode="after")
    def _in_bounds(self) -> "PlaceSpecDef":
        _check_spec_bounds(self.value, self.bounds)
        return self


class HingeSpecDef(_Frozen):
    kind: Literal["hinge_spec"]
    value: HingeSpec
    trust: SpecTrust = SpecTrust()
    bounds: SpecBounds = SpecBounds()

    @model_validator(mode="after")
    def _in_bounds(self) -> "HingeSpecDef":
        _check_spec_bounds(self.value, self.bounds)
        return self


class PathSpecDef(_Frozen):
    """A held path, as fitted on the demonstration: no number of it is edited or interpolated (a
    merge takes it whole)."""

    kind: Literal["path_spec"]
    value: PathSpec


class SkillSpecDef(_Frozen):
    """A written skill's parameters as fitted on the demonstration (`tools.skill_fit`): the
    skill's name and its parameter object, as JSON text (a tool schema takes no free-keyed
    object), which `run_skill` takes by the parameter's name; no field of it is edited or
    interpolated (a merge takes it whole). Its values are the skill's fit's, relative to the
    objects the skill acts on."""

    kind: Literal["skill_spec"]
    skill: Name
    value: Annotated[
        str,
        AfterValidator(_json_object),
        Field(description="The skill's parameter object as JSON text, as fit_skill gave it."),
    ]


class DoneSpecDef(_Frozen):
    """A done criterion; its trust radius and bounds are its metric's (`perception.done.UNITS`;
    a tolerance within the value's span)."""

    kind: Literal["done_spec"]
    value: DoneSpec


ParamDef = Annotated[
    NumberDef
    | CategoryDef
    | GraspSpecDef
    | PlaceSpecDef
    | HingeSpecDef
    | PathSpecDef
    | DoneSpecDef,
    Field(discriminator="kind"),
]
SkillParamDef = Annotated[
    NumberDef
    | CategoryDef
    | GraspSpecDef
    | PlaceSpecDef
    | HingeSpecDef
    | PathSpecDef
    | DoneSpecDef
    | SkillSpecDef,
    Field(discriminator="kind"),
]
"""`ParamDef` with `skill_spec`: what `define_param` takes on a stack whose skillsets fit written
skills (`toolsets.memory.SkillMemoryTools`); elsewhere a model is not shown the kind."""


class Change(BaseModel):
    """One change of a parameter in the Memory's history: its definition, an edit, a merge (with
    the learning rate) or an edited version (committed outside a phase); `field` names a spec's
    field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    how: Literal["define", "edit", "merge", "edited"]
    field: str | None = None
    before: JsonValue = None
    after: JsonValue = None
    seed: int | None = None
    alpha: float | None = None


class NumberParam(NumberDef):
    kind: Literal["number"] = "number"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class CategoryParam(CategoryDef):
    kind: Literal["category"] = "category"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class GraspSpecParam(GraspSpecDef):
    kind: Literal["grasp_spec"] = "grasp_spec"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class PlaceSpecParam(PlaceSpecDef):
    kind: Literal["place_spec"] = "place_spec"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class HingeSpecParam(HingeSpecDef):
    kind: Literal["hinge_spec"] = "hinge_spec"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class PathSpecParam(PathSpecDef):
    kind: Literal["path_spec"] = "path_spec"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class SkillSpecParam(SkillSpecDef):
    kind: Literal["skill_spec"] = "skill_spec"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


class DoneSpecParam(DoneSpecDef):
    kind: Literal["done_spec"] = "done_spec"
    source: str = ""
    history: list[Change] = Field(default_factory=list)


Param = Annotated[
    NumberParam
    | CategoryParam
    | GraspSpecParam
    | PlaceSpecParam
    | HingeSpecParam
    | PathSpecParam
    | DoneSpecParam
    | SkillSpecParam,
    Field(discriminator="kind"),
]
PARAMS: Final = TypeAdapter(dict[Name, Param])
_STORED: Final[dict[str, type[BaseModel]]] = {
    "number": NumberParam,
    "category": CategoryParam,
    "grasp_spec": GraspSpecParam,
    "place_spec": PlaceSpecParam,
    "hinge_spec": HingeSpecParam,
    "path_spec": PathSpecParam,
    "done_spec": DoneSpecParam,
    "skill_spec": SkillSpecParam,
}


def stored(definition: BaseModel, source: str, history: list[Change]) -> Param:
    """A definition as a stored parameter."""
    kind = definition.kind  # type: ignore[attr-defined]
    return _STORED[kind].model_validate(  # type: ignore[return-value]
        {**dict(definition), "source": source, "history": history}
    )


def numbers(param: Param) -> dict[str | None, Any]:
    """A parameter's numeric fields by name (None: the number itself)."""
    if isinstance(param, NumberParam):
        return {None: param.value}
    if isinstance(param, CategoryParam | PathSpecParam | SkillSpecParam):
        return {}
    fields = {
        GraspSpecParam: GRASP_NUMBERS,
        PlaceSpecParam: PLACE_NUMBERS,
        DoneSpecParam: DONE_NUMBERS,
    }.get(type(param), HINGE_NUMBERS)
    return {field: getattr(param.value, field) for field in fields}


def categories(param: Param) -> dict[str | None, Any]:
    """A parameter's categorical fields by name (None: the category itself)."""
    if isinstance(param, CategoryParam | SkillSpecParam):  # a skill's parameters: whole
        return {None: param.value}
    if isinstance(param, GraspSpecParam):
        return {field: getattr(param.value, field) for field in GRASP_CATEGORIES}
    if isinstance(param, PlaceSpecParam):
        return {field: getattr(param.value, field) for field in PLACE_CATEGORIES}
    if isinstance(param, HingeSpecParam):
        return {field: getattr(param.value, field) for field in HINGE_CATEGORIES}
    if isinstance(param, DoneSpecParam):
        return {field: getattr(param.value, field) for field in DONE_CATEGORIES}
    if isinstance(param, PathSpecParam):
        return {field: getattr(param.value, field) for field in PATH_CATEGORIES}
    return {}


def others(param: Param) -> dict[str, Any]:
    """A spec's fields that are neither numeric nor categorical (`GRASP_OTHERS`, `PLACE_OTHERS`)."""
    fields = {GraspSpecParam: GRASP_OTHERS, PlaceSpecParam: PLACE_OTHERS}.get(type(param), ())
    return {field: getattr(param.value, field) for field in fields}


def with_field(param: Param, field: str | None, value: Any, change: Change | None = None) -> Param:
    """`param` with one field replaced (None: the value of a number or category), validated
    (ValueError), and `change` appended to its history."""
    data = dict(param)
    if field is None:
        data["value"] = value
    else:
        spec = param.value
        assert isinstance(spec, GraspSpec | PlaceSpec | HingeSpec | PathSpec | DoneSpec), (
            f"{param.kind}: no {field}"
        )
        data["value"] = type(spec).model_validate({**dict(spec), field: value})
    if change is not None:
        data["history"] = [*param.history, change]
    return type(param).model_validate(data)


def shape(param: Param) -> dict[str, Any]:
    """What defines a parameter besides its value and provenance: kind, unit, bounds, trust,
    frame, choices. Parameters of one shape are the same parameter, merged field by field."""
    return param.model_dump(mode="json", exclude={"value", "source", "history"})


def definition(param: Param) -> dict[str, Any]:
    """A parameter as JSON, without its provenance."""
    return param.model_dump(mode="json", exclude={"source", "history"})


def trust(param: Param, field: str | None) -> float:
    if isinstance(param, NumberParam):
        return param.trust
    if isinstance(param, DoneSpecParam):
        return UNITS[param.value.metric][2]
    return float(getattr(param.trust, field or ""))  # type: ignore[union-attr]


def bounds(param: Param, field: str | None) -> tuple[float, float]:
    """A numeric field's limits; a robot-frame placement's offset has the robot frame's."""
    if isinstance(param, NumberParam):
        return param.bounds
    if isinstance(param, DoneSpecParam):
        low, high = UNITS[param.value.metric][1]
        return (low, high) if field == "value" else (0.0, high - low)
    if isinstance(param, PlaceSpecParam) and field == "offset_m":
        robot = param.value.target.kind == "robot_frame"
        return param.bounds.robot_frame_m if robot else param.bounds.offset_m
    return getattr(param.bounds, field or "")  # type: ignore[union-attr, no-any-return]


def distance(a: float | tuple[float, ...], b: float | tuple[float, ...]) -> float:
    """How far apart two values of one numeric field are (Euclidean for vectors)."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return math.dist(a, b)
    return abs(float(a) - float(b))  # type: ignore[arg-type]


def jsonable(value: Any) -> JsonValue:
    return to_jsonable_python(value)  # type: ignore[no-any-return]


def _check_bounds(name: str, value: float, limits: tuple[float, float]) -> None:
    if not limits[0] <= value <= limits[1]:
        raise ValueError(f"{name} {value:g} is outside its bounds [{limits[0]:g}, {limits[1]:g}]")


def _check_spec_bounds(spec: GraspSpec | PlaceSpec | HingeSpec, limits: SpecBounds) -> None:
    robot = isinstance(spec, PlaceSpec) and spec.target.kind == "robot_frame"
    for component in spec.offset_m:
        _check_bounds(
            "offset_m component", component, limits.robot_frame_m if robot else limits.offset_m
        )
    if isinstance(spec, HingeSpec):
        return
    _check_bounds("yaw_deg", spec.yaw_deg, limits.yaw_deg)
    if isinstance(spec, GraspSpec):
        for component in spec.offset_frac:
            _check_bounds("offset_frac component", component, limits.offset_frac)
        _check_bounds("tilt_deg", spec.tilt_deg, limits.tilt_deg)
        _check_bounds("lean_deg", spec.lean_deg, limits.lean_deg)
    else:
        for component in spec.rpy_deg:
            _check_bounds("rpy_deg component", component, limits.rpy_deg)
        _check_bounds("region_m", spec.region_m, limits.region_m)


def fitted_value(param: Param, field: str | None) -> Any:
    """A numeric or categorical target's value as the demonstration last fitted the parameter
    (where it was fitted, `FITTED_SOURCE`; None otherwise): its value before the first change
    of that field since the latest definition, or its value now if none changed it."""
    if not param.source.startswith(f"{FITTED_SOURCE} "):
        return None
    now = {**numbers(param), **categories(param)}
    if field not in now:
        return None
    defined = max((i for i, c in enumerate(param.history) if c.how == "define"), default=-1)
    for change in param.history[defined + 1 :]:
        before = change.before
        if change.field is None and field is not None and isinstance(before, dict):
            before = before.get(field)  # the whole spec replaced (a merge)
        elif change.field != field:
            continue
        return tuple(before) if isinstance(before, list) else before
    return now[field]


def demo_fit(param: Param) -> GraspSpec | None:
    """The grasp as the demonstration last fitted it (its latest definition, where the spec was
    fitted: `FITTED_SOURCE`), or None."""
    if not param.source.startswith(f"{FITTED_SOURCE} "):
        return None
    for change in reversed(param.history):
        if change.how == "define" and isinstance(change.after, dict):
            return GraspSpec.model_validate(change.after)
    return None
