"""Typed Memory edits (docs/concepts/learning.md): the only way to
change a cache, and how the S1 roles write and revise Memory.

An edit is one operation, `EditOp` (tagged by `op`). `apply` makes it on a document or raises
`Rejected` saying why not: an unknown parameter, stage or lesson, a value out of bounds or not
among the choices, a document that would not validate, an edit that changes nothing. An `Editor`
holds a document under edit and adds what depends on the session: the budget charged by
`EditCosts` (a cache's B_k, see `roboscript.memory.cache.open_cache`; a draft has none), the
limits of a revision (one Reviser run: at most `max_numbers` numeric targets changed, each within
its trust radius of its value when the revision started; parameters defined in the revision are
exempt) and the linter, which must pass after every edit. Each edit is made whole or not at all,
and every attempt is recorded.
"""

import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from roboscript.contracts.tools import ToolModel
from roboscript.manipulation.specs import GraspSpec, HingeSpec, Name, PlaceSpec, PlaceTarget, Vec3
from roboscript.memory.calls import parse_calls, template_refs
from roboscript.memory.doc import (
    STAGE_FIELDS,
    Lesson,
    LessonId,
    Line,
    MemoryDoc,
    Phrase,
    Text,
    WorkflowStage,
    place_call_args,
)
from roboscript.memory.linter import FITTED_SOURCE, Finding, LintConfig, errors, prose
from roboscript.memory.linter import lint as lint_doc
from roboscript.memory.params import (
    Change,
    DoneSpecParam,
    ParamDef,
    PlaceSpecParam,
    SkillParamDef,
    bounds,
    categories,
    demo_fit,
    distance,
    fitted_value,
    jsonable,
    numbers,
    stored,
    trust,
    with_field,
)
from roboscript.spatial.rotation import (
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    rpy_to_matrix,
)

EPSILON: Final = 1e-9  # float slack on trust radii
_REPEAT: Final = 1e-6  # a number this near one already tried is that value again
_FIT_ROUNDING: Final = 1e-3  # a number this near a fitted one is it, as the Memory shows it

Why = Annotated[Phrase, Field(description="Why: the failure or the evidence this edit answers.")]


FITTED: Final = r"^(left|right) K[1-9][0-9]*$"
"""Where `demo_relation` fitted a spec: its arm and number (wicl's `relation.NAME`)."""


class _Op(ToolModel):
    model_config = ConfigDict(frozen=True)


class SetNumber(_Op):
    """Set a number parameter, or a spec's numeric field (offset_m as [x, y, z] in m, yaw_deg, a
    grasp's offset_frac as [x, y, z] fractions of the part's extents, tilt_deg and lean_deg, a
    placement's rpy_deg as [roll, pitch, yaw] in deg and region_m, a done criterion's value and
    tol), within its bounds; when revising, within its trust radius of its value at the
    revision's start."""

    op: Literal["set_number"]
    param: Name
    field: Annotated[
        Literal[
            "offset_m",
            "offset_frac",
            "yaw_deg",
            "tilt_deg",
            "lean_deg",
            "rpy_deg",
            "region_m",
            "value",
            "tol",
        ]
        | None,
        Field(description="A spec's numeric field; null for a number parameter."),
    ] = None
    value: FiniteFloat | Vec3
    why: Why


class SetCategory(_Op):
    """Set a category parameter, or a spec's categorical field (grasp: part, approach, anchor,
    x_toward, closing, purpose, arm; place: target, frame, orient, placement_point, arm, mirror,
    approach: the direction [x, y, z] in the placement's frame the held object moves in onto it,
    or null for the call's approach_axis; done criterion: metric, a, b, part)."""

    op: Literal["set_category"]
    param: Name
    field: Annotated[
        Literal[
            "part",
            "approach",
            "anchor",
            "x_toward",
            "closing",
            "purpose",
            "arm",
            "target",
            "frame",
            "orient",
            "placement_point",
            "mirror",
            "metric",
            "a",
            "b",
        ]
        | None,
        Field(description="A spec's categorical field; null for a category parameter."),
    ] = None
    value: Annotated[
        Annotated[str, StringConstraints(max_length=100)] | bool | PlaceTarget | Vec3 | None,
        Field(
            description="The new value: an object for a place target, true or false for "
            "mirror, [x, y, z] for a placement's approach, null only for x_toward, arm, a "
            "placement's approach and a done criterion's b and part."
        ),
    ]
    why: Why


class RotateSpec(_Op):
    """Turn a grasp in its `pose` form (whose axes say its orientation: it takes no yaw, tilt,
    lean or closing) rigidly by `deg` about its approach axis (the fingers' closing axis turns
    about the pointing axis) or about its closing axis (the pointing axis tilts about it); its
    orientation hint (`reference_q`) turns with it. A numeric edit: when revising, within the
    spec's yaw trust radius."""

    op: Literal["rotate_spec"]
    param: Name
    about: Literal["approach", "closing"]
    deg: Annotated[FiniteFloat, Field(ge=-180.0, le=180.0)]
    why: Why


class AddLesson(_Op):
    """Add a lesson: when it applies and what to do. Near the limit, merge or replace lessons
    rather than add."""

    op: Literal["add_lesson"]
    when: Phrase
    do: Phrase
    why: Why


class EditLesson(_Op):
    """Rewrite a lesson's condition, action or both (null keeps one)."""

    op: Literal["edit_lesson"]
    lesson: LessonId
    when: Phrase | None = None
    do: Phrase | None = None
    why: Why

    @model_validator(mode="after")
    def _changes_something(self) -> "EditLesson":
        if self.when is None and self.do is None:
            raise ValueError("give the new `when`, the new `do`, or both")
        return self


class RemoveLesson(_Op):
    """Remove a lesson."""

    op: Literal["remove_lesson"]
    lesson: LessonId
    why: Why


class EditStage(_Op):
    """Rewrite a workflow stage's goal, arm rule or call template (a local edit)."""

    op: Literal["edit_stage"]
    stage: Name
    field: Literal["goal", "arm", "call"]
    text: Line
    why: Why


class AddStage(_Op):
    """Add a workflow stage (a structure change)."""

    op: Literal["add_stage"]
    stage: WorkflowStage
    before: Annotated[
        Name | None, Field(description="The stage to insert it before; null: at the end.")
    ] = None
    why: Why


class RemoveStage(_Op):
    """Remove a workflow stage (a structure change)."""

    op: Literal["remove_stage"]
    stage: Name
    why: Why


class MoveStage(_Op):
    """Move a workflow stage (a structure change)."""

    op: Literal["move_stage"]
    stage: Name
    before: Annotated[
        Name | None, Field(description="The stage to move it before; null: to the end.")
    ] = None
    why: Why


class SetDone(_Op):
    """Replace a workflow stage's completion predicates (a structure change)."""

    op: Literal["set_done"]
    stage: Name
    done: Line
    why: Why


class DefineParam(_Op):
    """Define a new parameter; define it before any text names it as {name}."""

    op: Literal["define_param"]
    name: Name
    param: ParamDef
    fitted: Annotated[
        str | None,
        Field(
            pattern=FITTED,
            description="A spec fitted on the demonstration (by the Demo Reader): where it was "
            "fitted (e.g. left K2), the spec as fitted there; null otherwise.",
        ),
    ] = None
    why: Why


class DefineSkillParam(DefineParam):
    """Define a new parameter; define it before any text names it as {name}."""

    param: SkillParamDef


class FitSpec(_Op):
    """Define a spec as the demonstration fits it where a stage happens there (S1's Reviser):
    the grasp where that arm closes (its pose form), the hinge a held part turns about there, or
    the placement it releases at (its first consistent form). The way to add a spec a stage lacks
    (a second arm's grasp, the handle a stage pulls): a spec is never written by hand. Naming a
    spec of the same kind that exists refits it: its values become the fit's (one edit), its
    history kept."""

    op: Literal["fit_spec"]
    name: Name
    at: Annotated[
        str,
        Field(
            pattern=FITTED,
            description="Where the stage happens in the demonstration: the arm and its stop, "
            "e.g. left K3.",
        ),
    ]
    why: Why


class RemoveParam(_Op):
    """Remove a parameter that no text names any more."""

    op: Literal["remove_param"]
    name: Name
    why: Why


class RenameParam(_Op):
    """Rename a parameter, keeping its value, limits, source and history; every `{name}` in the
    text becomes `{new_name}`."""

    op: Literal["rename_param"]
    name: Name
    new_name: Name
    why: Why


class SetText(_Op):
    """Replace the text of TASK, OBJECTS or NOTES (INSTRUCTION, the demonstration's instruction,
    is never edited)."""

    op: Literal["set_text"]
    section: Literal["statement", "objects", "notes"]
    text: Text
    why: Why


class Scratch(_Op):
    """Note a hypothesis or an observation in SCRATCH (caches only; never merged)."""

    op: Literal["scratch"]
    text: Phrase


EditOp = Annotated[
    SetNumber
    | SetCategory
    | RotateSpec
    | AddLesson
    | EditLesson
    | RemoveLesson
    | EditStage
    | AddStage
    | RemoveStage
    | MoveStage
    | SetDone
    | DefineParam
    | FitSpec
    | RemoveParam
    | RenameParam
    | SetText
    | Scratch,
    Field(discriminator="op"),
]
SkillEditOp = Annotated[
    SetNumber
    | SetCategory
    | RotateSpec
    | AddLesson
    | EditLesson
    | RemoveLesson
    | EditStage
    | AddStage
    | RemoveStage
    | MoveStage
    | SetDone
    | DefineSkillParam
    | FitSpec
    | RemoveParam
    | RenameParam
    | SetText
    | Scratch,
    Field(discriminator="op"),
]
"""`EditOp` whose `define_param` also takes a written skill's `skill_spec` (`DefineSkillParam`):
the `memory_edit` schema of a stack whose skillsets fit written skills
(`toolsets.memory.SkillMemoryTools`). Every member is an `EditOp` member or a subclass of one."""
EDIT_OP: Final[TypeAdapter[SkillEditOp]] = TypeAdapter(SkillEditOp)
"""Parses either form (a stored record, a test's JSON): the wider one."""
RecordedOp = SkillEditOp | EditOp
"""An edit as recorded: an op the narrower schema made, or one the wider (or `EDIT_OP`) did."""
_PARAM_DEF: Final[TypeAdapter[ParamDef]] = TypeAdapter(ParamDef)


class EditCosts(BaseModel):
    """Budget units an accepted edit uses, by operation (defaults:
    numeric 0, categorical 1, lesson or local workflow edit 1, structure change 2, SCRATCH 0;
    parameter definitions and section rewrites count as structure changes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_number: NonNegativeInt = 0
    set_category: NonNegativeInt = 1
    rotate_spec: NonNegativeInt = 0
    add_lesson: NonNegativeInt = 1
    edit_lesson: NonNegativeInt = 1
    remove_lesson: NonNegativeInt = 1
    edit_stage: NonNegativeInt = 1
    add_stage: NonNegativeInt = 2
    remove_stage: NonNegativeInt = 2
    move_stage: NonNegativeInt = 2
    set_done: NonNegativeInt = 2
    define_param: NonNegativeInt = 2
    fit_spec: NonNegativeInt = 2
    remove_param: NonNegativeInt = 2
    rename_param: NonNegativeInt = 1
    set_text: NonNegativeInt = 2
    scratch: NonNegativeInt = 0

    def of(self, op: "EditOp") -> int:
        return int(getattr(self, op.op))


class EditRules(BaseModel):
    """What an editor enforces besides its budget: the costs; the limits of a revision, at most
    `max_numbers` numeric targets (a number, or a spec's numeric field) changed, each within its
    trust radius (`trust_radius`), None and false lifting them; with `clamp_to_trust`, a numeric
    edit that goes beyond its radius is moved the radius toward its value instead of rejected
    (its record says so); and what
    may change (`scope`): anything, the done criteria alone (`done`: done_spec parameters
    defined, set or removed; S1's criteria step, which runs after the
    passes and so must leave what they ran as it is), or what makes a Memory executable
    (`cleanup`: text, stages, lessons, number and category parameters whose values the Memory
    already states, renames; never a spec defined, edited or removed: the demonstration's fits stay
    as fitted). `keep_fits` (S1's Reviser): a spec fitted on the demonstration is never removed,
    a spec is defined only as the demonstration fits it (`fitted`; the Reviser's `fit_spec`), and a
    stage's call swaps its own fitted spec only for one fitted at the same keypoint (its arm and
    number): the fit is corrected by editing it, never replaced by a hand-written spec or one of
    another stage's keypoint; a spec another stage uses too (one arm's grasp passed to the other
    arm's stage) is not that stage's own, so it may be swapped for the stage's own fit.
    `strict_done` (the Revisers': the Agent's exit condition stays strict,
    only the official check decides success): a done criterion is only ever tightened, never
    loosened. Its band (value within tol) may shrink inside itself; an edit that lets the band
    reach anywhere it did not, changes what it measures or removes it is refused. Adding a
    criterion adds a condition and stays allowed. A gate that is too strict costs no success
    (the official check ends a try the moment it latches); one that is too loose costs a false
    finish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    costs: EditCosts = EditCosts()
    max_numbers: PositiveInt | None = 3
    trust_radius: bool = True
    clamp_to_trust: bool = True
    scope: Literal["all", "done", "cleanup"] = "all"
    keep_fits: bool = False
    strict_done: bool = False


FIT_TURNS: Final = 2.0
"""A pose grasp turns at most this many yaw trust radii from its demonstration fit in all: small
rotations add up over revisions, and a grasp turned far off its fit can fail what depends on it."""
DEAD_TURN_DEG: Final = 0.5
"""A placement's orientation edit that tilts its up by less than this only turns it about it."""


AXES: Final = "axes"
"""The target a pose grasp's orientation is (its approach and closing axes, `rotate_spec`): the
revert and repeat guards compare it as a rotation."""
MIN_TURN_DEG: Final = 0.5
"""A pose grasp's rotation smaller than this is rounding (its axes are shown to 4 decimals)."""
SAME_TURN_DEG: Final = 3.0
"""Two grasp orientations this near are one (rounding, a few degrees of noise)."""


def grasp_params(doc: MemoryDoc) -> set[str]:
    """The parameters the workflow's grasp calls pass (`grasp_object`, `co_grasp`): their specs
    and numbers."""
    found: set[str] = set()
    for stage in doc.workflow:
        for call in parse_calls(stage.call):
            if call.name not in _GRASP_CALLS:
                continue
            text = " ".join([*call.keywords.values(), *call.values])
            found |= {
                name
                for name in template_refs(text, doc.params)
                if doc.params[name].kind in ("grasp_spec", "number")
            }
    return found


_GRASP_CALLS: Final = ("grasp_object", "co_grasp")


def _dead_lever(doc: MemoryDoc, op: EditOp) -> str | None:
    """Why an edit changes nothing, if it does not: a pose grasp turned half a turn about its
    approach (its candidates already hold it both ways, a parallel gripper holding alike rolled:
    the turn only reorders them); a placement's orientation edit under the `constrain` its
    workflow's `place_object` call passes (auto, the default, or free), where the
    placement turns the object about its up by its own rule, reading only which way the target's
    up points, so a `yaw_deg` (an upright yaw) or an `rpy_deg` that leaves the placed up where it
    was is a dead lever."""
    if isinstance(op, RotateSpec):
        if op.about != "approach" or abs(abs(op.deg) - 180.0) > SAME_TURN_DEG:
            return None
        return (
            f"{op.param} turned half a turn about its approach holds the object the same way: its "
            "grasp candidates already include that roll (a parallel gripper holds alike rolled), "
            "so the turn only reorders them and changes nothing. Where the held object came out "
            "half a turn round, name its front (the placement's and the grasp's x_toward), or "
            "change what the evidence names"
        )
    if not isinstance(op, SetNumber) or op.field not in ("rpy_deg", "yaw_deg"):
        return None
    param = doc.params.get(op.param)
    if not isinstance(param, PlaceSpecParam):
        return None
    spec = param.value
    constrain = place_call_args(doc, op.param)["constrain"]
    if constrain == "align":
        return None
    if op.field == "yaw_deg":
        if spec.orient != "yaw":
            return None
    else:
        if spec.orient != "relative" or isinstance(op.value, float | int):
            return None
        up = rpy_to_matrix(spec.rpy_deg)[:, 2], rpy_to_matrix(op.value)[:, 2]
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(np.dot(*up))))))
        if tilt >= DEAD_TURN_DEG:
            return None
    return (
        f"{op.param}.{op.field} only turns the placed object about its up, and the workflow's "
        f"place_object call with {op.param} passes constrain {constrain}, under which the "
        "placement turns it about its up by its own rule: this edit changes nothing. Pass "
        "constrain align in that call (edit the stage) for the turn to count, or change what the "
        "evidence names"
    )


class Rejected(Exception):
    """Why an edit cannot be made, with the linter's findings when they are the reason."""

    def __init__(self, reason: str, findings: Sequence[Finding] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.findings = list(findings)


class EditRecord(BaseModel):
    """One edit attempt (the `memory.edit` event): accepted or not and why, its cost (charged
    only when accepted), the budget after it, and the linter's findings: the errors behind a
    rejection, or the warnings about the edited Memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: PositiveInt
    revision: PositiveInt
    seed: int | None
    op: RecordedOp
    accepted: bool
    cost: NonNegativeInt
    used: NonNegativeInt
    budget: NonNegativeInt | None
    reason: str = ""
    findings: list[Finding] = []
    clamped: str = ""
    """An accepted edit moved short of the value asked for, to its trust radius (`op` holds the
    value applied): what was asked and how far it went."""

    @property
    def left(self) -> int | None:
        return None if self.budget is None else self.budget - self.used


def apply(
    doc: MemoryDoc,
    op: EditOp,
    *,
    seed: int | None = None,
    source: str = "",
    lesson_floor: int = 0,
    refit: bool = False,
) -> MemoryDoc:
    """`doc` with the edit `op` made; `seed` and `source` mark what it adds (lessons, parameters,
    history), and a new lesson is numbered above `lesson_floor` and every lesson in `doc`; with
    `refit`, a fitted definition replaces a spec of its kind of that name (`FitSpec`). Raises
    `Rejected`. Pure: budgets, revision limits and the linter are the Editor's."""
    try:
        return _apply(doc, op, seed, source, lesson_floor, refit)
    except ValidationError as error:
        raise Rejected(_problems(error)) from None


@dataclass(frozen=True)
class EditorState:
    """An `Editor`'s state (`Editor.snapshot`)."""

    doc: MemoryDoc
    used: int
    revision: int
    records: tuple[EditRecord, ...]
    start: tuple[tuple[tuple[str, str | None], Any], ...]
    defined: frozenset[str]
    lessons: int


def _error_key(finding: Finding) -> tuple[str, str, str]:
    return finding.rule, finding.where, finding.match


class EditValidator:
    """A method's checks on edits (docs/concepts/learning.md: the engine keeps typed edits, the
    budget and this interface; what a method forbids beyond them is its validators'). The editor
    runs each stage's checks in its validators' order, at a fixed place among its own: `rules`
    with the rule checks, before the revision limits; `pending` after the revision limits, the
    locked sections and the dead-lever check, before the budget; `applied` on the document the
    edit makes, after `effect`, before the linter. Each returns why the edit is refused, or None.
    A validator holds its own state (such as what the tries under review saw)."""

    def rules(self, editor: "Editor", op: "EditOp") -> str | None:
        return None

    def pending(self, editor: "Editor", op: "EditOp") -> str | None:
        return None

    def applied(self, editor: "Editor", doc: MemoryDoc, op: "EditOp") -> str | None:
        return None


class Editor:
    """A Memory document under edit: a cache (a copy of M with SCRATCH and the seed's budget, see
    `cache.open_cache`) or a draft the S1 roles edit directly (no budget). It must start
    lint-clean and stays so, unless it is opened to `repair` a stored Memory that fails the
    linter: then no edit may add an error beyond those it opened with, so edits only fix, and
    `errors_left` says what remains. `revise()` starts the next revision; `records` holds every
    attempt. `locked`: sections no edit may rewrite, each with why (S1's Reviser: OBJECTS as it
    was when an attempt passed on the seed); `effect`: why an edit (the document before and
    after it) cannot change behaviour, refusing it (None: it can); `matched`: by workflow stage,
    the demonstration's keypoint the expert review matched the stage's release to (`left K8`),
    where a spec replacing the stage's own fit may also be fitted (`keep_fits`); `failed`: by
    numeric or category target (parameter, field), the values attempts on this seed already
    failed with, which an edit may not set again (S1's Reviser); `validators`: the
    method's own checks (`EditValidator`), run in order at their stages. `object_fits`: by
    keypoint, the demonstration's placement relative to an object there, which `fit_spec` refits
    a placement targeting an object with where `fits` has it in another frame (a refit keeps the
    target kind; without one it is refused). `first`: the parameters the expert review names at
    its first divergence (None: no review); setting another parameter back to the
    demonstration's fit passes the revert and repeat guards, since the attempts that failed with
    that value failed first elsewhere (`_back_to_fit`)."""

    def __init__(
        self,
        doc: MemoryDoc,
        *,
        rules: EditRules,
        lint: LintConfig,
        budget: int | None = None,
        seed: int | None = None,
        source: str = "",
        repair: bool = False,
        reverts: Mapping[tuple[str, str | None], Any] | None = None,
        evidence: Collection[str] = (),
        fits: Mapping[str, GraspSpec | PlaceSpec | HingeSpec | str] | None = None,
        object_fits: Mapping[str, PlaceSpec] | None = None,
        locked: Mapping[str, str] | None = None,
        effect: Callable[[MemoryDoc, MemoryDoc, "EditOp"], str | None] | None = None,
        matched: Mapping[str, str] | None = None,
        failed: Mapping[tuple[str, str | None], Collection[Any]] | None = None,
        validators: Sequence["EditValidator"] = (),
        first: Collection[str] | None = None,
    ) -> None:
        opened = errors(lint_doc(doc, lint))
        if opened and not repair:
            listed = "; ".join(f"{f.where}: {f.rule} {f.match!r}" for f in opened[:5])
            raise ValueError(
                f"{doc.task} {doc.label} fails the linter ({listed}): fix it in an edited version "
                "before editing"
            )
        if budget is not None and budget < 0:
            raise ValueError(f"budget {budget} is negative")
        self.base = self.doc = doc
        self.rules = rules
        self.lint = lint
        self._open_errors = {_error_key(f) for f in opened}
        self.budget = budget
        self.seed = seed
        self.source = source
        self.used = 0
        self.revision = 1
        self.records: list[EditRecord] = []
        self._start: dict[tuple[str, str | None], Any] = {}  # numeric targets changed: start value
        self._defined: set[str] = set()  # parameters defined in this revision
        self._lessons = _highest_lesson(doc)  # lesson numbers are never reused
        self._reverts = dict(reverts or {})
        self._evidence = frozenset(evidence)
        self._fits = fits
        self._object_fits = dict(object_fits or {})
        self._locked = dict(locked or {})
        self._effect = effect
        self._matched = dict(matched or {})
        self._failed = {key: list(values) for key, values in (failed or {}).items()}
        self.validators = tuple(validators)
        self._first = None if first is None else frozenset(first)

    @property
    def left(self) -> int | None:
        return None if self.budget is None else self.budget - self.used

    @property
    def locked(self) -> frozenset[str]:
        """The sections no edit may rewrite (`locked`)."""
        return frozenset(self._locked)

    @property
    def refits(self) -> bool:
        """Whether a spec can be refit at its demonstration keypoint here (`fit_spec`)."""
        return self._fits is not None

    @property
    def numbers_left(self) -> int | None:
        """How many more numeric targets this revision may change."""
        limit = self.rules.max_numbers
        return None if limit is None else limit - len(self._start)

    @property
    def rejected(self) -> list[EditRecord]:
        return [record for record in self.records if not record.accepted]

    def revise(self) -> None:
        """Start the next revision (the next Reviser run on this document)."""
        self.revision += 1
        self._start.clear()
        self._defined.clear()

    def errors_left(self) -> list[Finding]:
        """The linter's errors on the document now (none unless opened to `repair`)."""
        return errors(lint_doc(self.doc, self.lint))

    def snapshot(self) -> EditorState:
        """What the editor holds now, to `restore` if a run's edits are voided."""
        return EditorState(
            doc=self.doc,
            used=self.used,
            revision=self.revision,
            records=tuple(self.records),
            start=tuple(self._start.items()),
            defined=frozenset(self._defined),
            lessons=self._lessons,
        )

    def restore(self, state: EditorState) -> None:
        """Go back to a `snapshot`: its document, budget, revision limits and records."""
        self.doc, self.used, self.revision = state.doc, state.used, state.revision
        self.records = list(state.records)
        self._start = dict(state.start)
        self._defined = set(state.defined)
        self._lessons = state.lessons

    def edit(self, op: EditOp | Mapping[str, Any]) -> EditRecord:
        """Make one edit (an `EditOp`, or its JSON form, validated here) or record why not."""
        edit: EditOp = EDIT_OP.validate_python(op) if isinstance(op, Mapping) else op
        cost = self.rules.costs.of(edit)
        try:
            if isinstance(edit, FitSpec):  # the definition the demonstration's fit makes
                return self._commit(edit, self._fitted(edit), cost, refit=True)
            edit, clamped = self._clamp(edit)
            return self._commit(edit, edit, cost, clamped)
        except Rejected as rejected:
            return self._record(edit, cost, rejected.reason, rejected.findings)

    def _fitted(self, op: FitSpec) -> "DefineParam":
        """`op` as the definition of the spec the demonstration fits at its keypoint (`fits`:
        each keypoint's spec, or why it has none)."""
        if self._fits is None:
            raise Rejected("fit_spec needs the demonstration's fits: S1's Reviser only")
        found = self._fits.get(op.at)
        if found is None:
            known = ", ".join(sorted(k for k, v in self._fits.items() if not isinstance(v, str)))
            raise Rejected(f"the demonstration has no {op.at}; fits at: {known}")
        if isinstance(found, str):
            raise Rejected(f"no spec fits at {op.at}: {found}")
        was = self.doc.params.get(op.name)
        relative = isinstance(was, PlaceSpecParam) and was.value.target.kind == "object"
        if was is None and isinstance(found, PlaceSpec) and op.at in self._object_fits:
            # a new placement where the demonstration places at an object: relative to it
            found = self._object_fits[op.at]
        elif relative and isinstance(found, PlaceSpec) and found.target.kind != "object":
            # a refit keeps an object-relative placement relative to an object (one in the
            # robot's frame would not move with the object on another scene)
            kept = self._object_fits.get(op.at)
            assert isinstance(was, PlaceSpecParam)
            if kept is None:
                raise Rejected(
                    f"{op.name} is placed relative to {was.value.target.obj}, and the "
                    f"demonstration fits no placement relative to an object at {op.at} (only in "
                    f"the {found.target.kind.replace('_', ' ')}): a refit keeps a placement's "
                    "target kind; refit it where it is placed at an object, or edit its fields"
                )
            # its target named by the spec's role, as the Memory names it
            role = kept.target.model_copy(update={"obj": was.value.target.obj})
            found = kept.model_copy(update={"target": role})
        kind = {GraspSpec: "grasp_spec", PlaceSpec: "place_spec", HingeSpec: "hinge_spec"}
        definition = _PARAM_DEF.validate_python({"kind": kind[type(found)], "value": found})
        return DefineParam(
            op="define_param", name=op.name, param=definition, fitted=op.at, why=op.why
        )

    def _clamp(self, op: EditOp) -> tuple[EditOp, str]:
        """`op` moved to its trust radius when it asks for more (`clamp_to_trust`), along the
        straight line from the target's value when the revision started (a rotation: its angle
        cut to the radius), and what was asked; `op` unchanged and "" otherwise (a value out of
        its bounds too: rejected as asked). A clamped edit
        then meets every other check as an edit of the value it applies would."""
        if not (self.rules.trust_radius and self.rules.clamp_to_trust):
            return op, ""
        if isinstance(op, RotateSpec):
            param = self.doc.params.get(op.param)
            if param is None or op.param in self._defined:
                return op, ""
            left = trust(param, "yaw_deg") - self._turned(op.param)
            if abs(op.deg) <= left + EPSILON or left < MIN_TURN_DEG:
                return op, ""  # with nothing left, the check refuses it as asked
            applied = math.copysign(round(left, 3), op.deg)
            note = (
                f"{op.param} asked to turn {op.deg:g} deg; turned {applied:g}, what its trust "
                "radius leaves this revision"
            )
            return op.model_copy(update={"deg": applied}), note
        if not isinstance(op, SetNumber) or op.param in self._defined:
            return op, ""
        param = self.doc.params.get(op.param)
        if param is None or op.field not in numbers(param):
            return op, ""  # the edit's own checks say what is wrong
        low, high = bounds(param, op.field)
        asked = op.value if isinstance(op.value, tuple) else (op.value,)
        if not all(low <= v <= high for v in asked):
            return op, ""  # a value out of bounds is rejected as asked, not moved toward
        start = self._start.get((op.param, op.field), numbers(param)[op.field])
        radius = trust(param, op.field)
        away = distance(start, op.value)
        if away <= radius + EPSILON:
            return op, ""
        share = radius / away
        if isinstance(op.value, tuple):
            applied: Any = tuple(s + (v - s) * share for s, v in zip(start, op.value, strict=True))
        else:
            applied = float(start) + (float(op.value) - float(start)) * share
        note = (
            f"{_target(op.param, op.field)} asked for {show(op.value)}, {away:.4g} from "
            f"{show(start)} (its value when this revision started); moved {radius:g} toward it, "
            f"to {show(applied)}, its trust radius"
        )
        return op.model_copy(update={"value": applied}), note

    def _commit(
        self, edit: EditOp, op: EditOp, cost: int, clamped: str = "", refit: bool = False
    ) -> EditRecord:
        """Make `op` (what `edit` does: itself, or a fit's definition), recording `edit`
        (`clamped`: how `_clamp` cut it; `refit`: a fit's definition may replace a spec of its
        kind, `FitSpec`); `Rejected` when it cannot be made."""
        if self.rules.scope == "cleanup" and (why := _cleanup_refusal(self.doc, op)):
            raise Rejected(why)
        if self.rules.scope == "done" and not _done_edit(self.doc, op):
            raise Rejected(
                "only the done criteria may change here: define_param of kind done_spec, "
                "set_number, set_category or remove_param on one"
            )
        if self.rules.keep_fits and (why := _fit_refusal(self.doc, op, self._matched)):
            raise Rejected(why)
        for validator in self.validators:
            if why := validator.rules(self, op):
                raise Rejected(why)
        if not self._back_to_fit(op):
            if why := self._reverted(op):
                raise Rejected(why)
            if why := self._repeated(op):
                raise Rejected(why)
        if isinstance(op, SetText) and op.section in self._locked:
            raise Rejected(self._locked[op.section])
        if why := _dead_lever(self.doc, op):
            raise Rejected(why)
        for validator in self.validators:
            if why := validator.pending(self, op):
                raise Rejected(why)
        if (left := self.left) is not None and cost > left:
            raise Rejected(f"{edit.op} costs {cost}; {left} of {self.budget} units are left")
        doc = apply(
            self.doc,
            op,
            seed=self.seed,
            source=self.source,
            lesson_floor=self._lessons,
            refit=refit,
        )
        if self._effect is not None and (why := self._effect(self.doc, doc, op)):
            raise Rejected(why)
        for validator in self.validators:
            if why := validator.applied(self, doc, op):
                raise Rejected(why)
        started = self._check_revision(op)
        findings = lint_doc(doc, self.lint)
        if problems := [f for f in errors(findings) if _error_key(f) not in self._open_errors]:
            raise Rejected("the Memory would fail the linter", problems)
        self.doc = doc
        self.used += cost
        self._lessons = max(self._lessons, _highest_lesson(doc))
        if isinstance(op, SetNumber) and started is not None:
            key = (op.param, op.field)
            if distance(started, op.value) == 0:  # back where it began: not a change
                self._start.pop(key, None)
            else:
                self._start.setdefault(key, started)
        if isinstance(op, RotateSpec) and op.param not in self._defined:
            self._start.setdefault((op.param, AXES), started)  # counted once accepted
        if isinstance(op, DefineParam):
            self._defined.add(op.name)
        return self._record(edit, cost, findings=findings, clamped=clamped)

    def _back_to_fit(self, op: EditOp) -> bool:
        """Whether `op` sets a parameter the review's first divergence does not name (`first`)
        back to the demonstration's fit of it: the value or orientation it was fitted with."""
        if self._first is None or not isinstance(op, SetNumber | SetCategory | RotateSpec):
            return False
        param = self.doc.params.get(op.param)
        if param is None or op.param in self._first:
            return False
        if isinstance(op, RotateSpec):
            fit, turned = grasp_axes_of(demo_fit(param)), _rotated_axes(self.doc, op)
            return fit is not None and turned is not None and axes_deg(turned, fit) <= SAME_TURN_DEG
        fit = fitted_value(param, op.field)
        if fit is None:
            return False
        if isinstance(op, SetNumber):
            return distance(op.value, fit) <= _FIT_ROUNDING
        return jsonable(op.value) == jsonable(fit)

    def _reverted(self, op: EditOp) -> str | None:
        """Why an edit is refused as a revert (S1's Reviser, `reverts`): it sets a target the
        previous revision changed back to (a number: at least halfway back to) its value before
        that change, while this revision's evidence (`evidence`: the parameters the privileged
        review names) does not concern its parameter: new evidence is what justifies undoing a
        change, a repeat of the same reasoning is not."""
        if isinstance(op, RotateSpec):
            before = self._reverts.get((op.param, AXES))
            if op.param in self._evidence or before is None or op.param not in self.doc.params:
                return None
            turned = _rotated_axes(self.doc, op)
            if turned is None or axes_deg(turned, before) > SAME_TURN_DEG:
                return None
            return (
                f"{op.param}'s orientation was turned by the previous revision; turning it back "
                "needs new evidence about it (the expert comparison naming this parameter): keep "
                "it, or change what the evidence names"
            )
        if not isinstance(op, SetNumber | SetCategory) or op.param in self._evidence:
            return None
        key = (op.param, op.field)
        if key not in self._reverts or op.param not in self.doc.params:
            return None
        before, param = self._reverts[key], self.doc.params[op.param]
        if isinstance(op, SetNumber):
            now = numbers(param).get(op.field)
            if now is None:
                return None
            back = distance(op.value, before) <= distance(now, before) / 2 + EPSILON
            back = back or _reverses(before, now, op.value)
        else:
            back = jsonable(op.value) == jsonable(before)
        if not back:
            return None
        return (
            f"{_target(*key)} was changed from {show(before)} by the previous revision; "
            "setting it back, or turning that change around past "
            "where it started, needs new evidence about it (the expert comparison naming this "
            "parameter): keep it, change it further in the direction the evidence shows, or "
            "change what the evidence names"
        )

    def _repeated(self, op: EditOp) -> str | None:
        """Why an edit is refused as a repeat (`failed`): it sets a target to a value attempts on
        this seed already failed with (and none passed with), however it got changed since."""
        if isinstance(op, RotateSpec):
            tried = self._failed.get((op.param, AXES), [])
            if op.param not in self.doc.params or not tried:
                return None
            turned = _rotated_axes(self.doc, op)
            if turned is None or all(axes_deg(turned, value) > SAME_TURN_DEG for value in tried):
                return None
            return (
                f"{op.param} turned so is an orientation already tried on this seed (within "
                f"{SAME_TURN_DEG:g} deg), and the attempt failed with it: turning it back repeats "
                "a failure; turn it elsewhere, or change what the expert comparison names"
            )
        if not isinstance(op, SetNumber | SetCategory):
            return None
        tried = self._failed.get((op.param, op.field), [])
        if isinstance(op, SetNumber):
            again = any(distance(op.value, value) <= _REPEAT for value in tried)
        else:
            again = any(jsonable(op.value) == jsonable(value) for value in tried)
        if not again:
            return None
        return (
            f"{_target(op.param, op.field)} = {show(op.value)} was already tried on this seed and "
            "the attempt failed with it: setting it back repeats a failure; change it to a value "
            "not yet tried, or change what the expert comparison names"
        )

    def _check_revision(self, op: EditOp) -> Any:
        """Check a numeric edit against the revision's limits; its target's value when the
        revision started (None for other edits and parameters defined in this revision)."""
        if isinstance(op, RotateSpec) and op.param not in self._defined:
            return self._check_rotation(op)
        if not isinstance(op, SetNumber) or op.param in self._defined:
            return None
        key = (op.param, op.field)
        param = self.doc.params[op.param]
        start = self._start.get(key, numbers(param)[op.field])
        limit = self.rules.max_numbers
        if limit is not None and key not in self._start and len(self._start) >= limit:
            changed = ", ".join(_target(*k) for k in self._start)
            raise Rejected(f"this revision has changed {limit} numeric targets already ({changed})")
        if self.rules.trust_radius:
            radius = trust(param, op.field)
            if (away := distance(start, op.value)) > radius + EPSILON:
                raise Rejected(
                    f"{_target(*key)} may move at most {radius:g} from {show(start)}, its value "
                    f"when this revision started; {show(op.value)} is {away:.4g} away"
                )
        return start

    def _check_rotation(self, op: "RotateSpec") -> tuple[Vec3, Vec3] | None:
        """A rotation counts as one numeric target of its spec (`<param>.axes`); the spec may
        turn at most its yaw trust radius in one revision, all its rotations together
        (clamped asks add up, and can turn a closing axis off the part it closes on). Its
        orientation when the revision started."""
        key = (op.param, AXES)
        param = self.doc.params[op.param]
        limit = self.rules.max_numbers
        if limit is not None and key not in self._start and len(self._start) >= limit:
            changed = ", ".join(_target(*k) for k in self._start)
            raise Rejected(f"this revision has changed {limit} numeric targets already ({changed})")
        start = self._start.get(key)
        start = start if isinstance(start, tuple) else grasp_axes(param)
        if self.rules.trust_radius:
            radius = trust(param, "yaw_deg")
            turned = _rotated_axes(self.doc, op)
            away = abs(op.deg) if turned is None or start is None else axes_deg(turned, start)
            if away > radius + MIN_TURN_DEG:
                raise Rejected(
                    f"{op.param} may turn at most {radius:g} deg in one revision, all its "
                    f"rotations together; this would turn it {away:.0f} deg from its orientation "
                    "when the revision started"
                )
        fit = grasp_axes_of(demo_fit(param))
        turned = _rotated_axes(self.doc, op)
        now = grasp_axes(param)
        if fit is not None and turned is not None and now is not None:
            limit = FIT_TURNS * trust(param, "yaw_deg")
            total, was = axes_deg(turned, fit), axes_deg(now, fit)
            if total > limit + MIN_TURN_DEG and total > was + MIN_TURN_DEG:
                raise Rejected(
                    f"{op.param} would turn {total:.0f} deg from its orientation as the "
                    f"demonstration fitted it (it is {was:.0f} deg from it now); a pose grasp "
                    f"turns at most {limit:g} deg from its fit in all, and turning back toward "
                    "it is always allowed. A grasp that needs more is not the demonstration's "
                    "grasp: refit it (fit_spec at the keypoint it was fitted at), or change the "
                    "approach, depth or arm"
                )
        return start

    def _turned(self, name: str) -> float:
        """How far (deg) a pose grasp has turned in this revision."""
        start = self._start.get((name, AXES))
        now = grasp_axes(self.doc.params.get(name))
        if not isinstance(start, tuple) or now is None:
            return 0.0
        return axes_deg(now, start)

    def _record(
        self,
        op: EditOp,
        cost: int,
        reason: str | None = None,
        findings: Sequence[Finding] = (),
        clamped: str = "",
    ) -> EditRecord:
        """Record an attempt: rejected with `reason`, or accepted with the linter's `findings`
        on the edited Memory (only warnings: errors reject)."""
        record = EditRecord(
            seq=len(self.records) + 1,
            revision=self.revision,
            seed=self.seed,
            op=op,
            accepted=reason is None,
            cost=cost,
            used=self.used,
            budget=self.budget,
            reason=reason or "",
            findings=list(findings),
            clamped=clamped,
        )
        self.records.append(record)
        return record


REVERSE_COS: Final = -0.5
"""An edit reverses the previous revision's change to a number when it moves the value at more
than 120 degrees from that change's direction (a scalar: the other way)."""


def _reverses(before: Any, now: Any, value: Any) -> bool:
    """Whether setting a number from `now` to `value` turns around the change the previous
    revision made from `before` to `now` (`REVERSE_COS`): a flip of sign past where it started
    is a revert too (a number can otherwise swing from side to side over revisions, each step
    going back more than halfway from where the last began)."""
    last = np.atleast_1d(np.asarray(now, dtype=float) - np.asarray(before, dtype=float))
    step = np.atleast_1d(np.asarray(value, dtype=float) - np.asarray(now, dtype=float))
    size = float(np.linalg.norm(last) * np.linalg.norm(step))
    return size > EPSILON and float(last @ step) / size <= REVERSE_COS


def _done_edit(doc: MemoryDoc, op: EditOp) -> bool:
    """Whether `op` touches only the done criteria (`EditRules.scope` done)."""
    if isinstance(op, DefineParam):
        return op.param.kind == "done_spec"
    if isinstance(op, SetNumber | SetCategory | RemoveParam):
        name = op.name if isinstance(op, RemoveParam) else op.param
        return isinstance(doc.params.get(name), DoneSpecParam)
    return False


def _apply(
    doc: MemoryDoc, op: EditOp, seed: int | None, source: str, lesson_floor: int, refit: bool
) -> MemoryDoc:
    if isinstance(op, SetNumber | SetCategory):
        return _set(doc, op, seed)
    if isinstance(op, RotateSpec):
        return _rotate(doc, op, seed)
    if isinstance(op, AddLesson):
        number = max(lesson_floor, _highest_lesson(doc)) + 1
        lesson = Lesson(id=f"L{number}", when=op.when, do=op.do, seed=seed)
        return doc.replace(lessons=[*doc.lessons, lesson])
    if isinstance(op, EditLesson):
        index = _lesson(doc, op.lesson)
        lesson = doc.lessons[index]
        rule = {"when": op.when or lesson.when, "do": op.do or lesson.do}
        if (rule["when"], rule["do"]) == (lesson.when, lesson.do):
            raise Rejected(f"lesson {op.lesson} says that already")
        lessons = [*doc.lessons[:index], lesson.model_copy(update=rule), *doc.lessons[index + 1 :]]
        return doc.replace(lessons=lessons)
    if isinstance(op, RemoveLesson):
        index = _lesson(doc, op.lesson)
        return doc.replace(lessons=[*doc.lessons[:index], *doc.lessons[index + 1 :]])
    if isinstance(op, EditStage | SetDone):
        field, text = ("done", op.done) if isinstance(op, SetDone) else (op.field, op.text)
        index = _stage(doc, op.stage)
        stage = doc.workflow[index]
        if getattr(stage, field) == text:
            raise Rejected(f"the {field} of stage {op.stage} says that already")
        edited = WorkflowStage.model_validate({**dict(stage), field: text})
        return doc.replace(workflow=[*doc.workflow[:index], edited, *doc.workflow[index + 1 :]])
    if isinstance(op, AddStage):
        if any(stage.name == op.stage.name for stage in doc.workflow):
            raise Rejected(f"there is a stage {op.stage.name} already")
        at = len(doc.workflow) if op.before is None else _stage(doc, op.before)
        return doc.replace(workflow=[*doc.workflow[:at], op.stage, *doc.workflow[at:]])
    if isinstance(op, RemoveStage):
        index = _stage(doc, op.stage)
        return doc.replace(workflow=[*doc.workflow[:index], *doc.workflow[index + 1 :]])
    if isinstance(op, MoveStage):
        moved = doc.workflow[_stage(doc, op.stage)]
        if op.before is not None:
            _stage(doc, op.before)
            if op.before == op.stage:
                raise Rejected("a stage cannot move before itself")
        rest = [stage for stage in doc.workflow if stage.name != op.stage]
        at = len(rest) if op.before is None else [s.name for s in rest].index(op.before)
        order = [*rest[:at], moved, *rest[at:]]
        if order == doc.workflow:
            raise Rejected(f"stage {op.stage} is there already")
        return doc.replace(workflow=order)
    if isinstance(op, DefineParam):
        old = doc.params.get(op.name)
        history: list[Change] = []
        if old is not None and not (refit and op.fitted is not None):
            raise Rejected(f"parameter {op.name} exists: set its values, or remove it first")
        if old is not None:
            if old.kind != op.param.kind:
                raise Rejected(
                    f"{op.name} is a {old.kind}, and the fit at {op.fitted} a {op.param.kind}: "
                    "fit it under another name"
                )
            if jsonable(old.value) == jsonable(op.param.value):
                raise Rejected(f"{op.name} is already as the demonstration fits it at {op.fitted}")
            history = list(old.history)
        before = None if old is None else jsonable(old.value)
        change = Change(how="define", before=before, after=jsonable(op.param.value), seed=seed)
        origin = source if op.fitted is None else f"{FITTED_SOURCE} {op.fitted}"
        param = stored(op.param, origin, [*history, change])
        return doc.replace(params={**doc.params, op.name: param})
    if isinstance(op, RemoveParam):
        _param(doc, op.name)
        return doc.replace(params={k: v for k, v in doc.params.items() if k != op.name})
    if isinstance(op, RenameParam):
        _param(doc, op.name)
        if op.new_name in doc.params:
            raise Rejected(f"parameter {op.new_name} exists already")
        if op.new_name == op.name:
            raise Rejected(f"{op.name} is named that already")
        return _renamed(doc, op.name, op.new_name)
    if isinstance(op, SetText):
        new_doc = doc.replace(**{op.section: op.text})
        if getattr(new_doc, op.section) == getattr(doc, op.section):
            title = "TASK" if op.section == "statement" else op.section.upper()
            raise Rejected(f"{title} says that already")
        return new_doc
    assert isinstance(op, Scratch)
    if doc.scratch is None:
        raise Rejected("SCRATCH exists only in a cache")
    return doc.replace(scratch=[*doc.scratch, op.text])


def _renamed(doc: MemoryDoc, old: str, new: str) -> MemoryDoc:
    """`doc` with parameter `old` named `new` (in place in the table) and its `{old}` and
    `{old.field}` references in every text rewritten."""
    pattern = re.compile(rf"\{{{re.escape(old)}(?=[.}}])")

    def text(value: str) -> str:
        return pattern.sub("{" + new, value)

    stages = [
        stage.model_copy(update={f: text(getattr(stage, f)) for f in STAGE_FIELDS})
        for stage in doc.workflow
    ]
    lessons = [
        lesson.model_copy(update={"when": text(lesson.when), "do": text(lesson.do)})
        for lesson in doc.lessons
    ]
    params = {(new if k == old else k): v for k, v in doc.params.items()}
    return doc.replace(
        workflow=stages,
        lessons=lessons,
        params=params,
        objects=text(doc.objects),
        notes=text(doc.notes),
    )


_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(cm|mm|m|deg)?(?!\w|\.\d)")


def _stated(doc: MemoryDoc) -> set[float]:
    """The numbers the Memory's text states as literals (a length in cm or mm also in meters):
    the values a cleanup may give a parameter."""
    found: set[float] = set()
    for _, text in prose(doc):
        for match in _NUMBER.finditer(text):
            value = float(match[1])
            found.add(round(value / {"cm": 100, "mm": 1000}.get(match[2] or "", 1), 9))
    return found


def _cleanup_refusal(doc: MemoryDoc, op: EditOp) -> str | None:
    """Why `op` is not a cleanup's (`EditRules.scope` cleanup), or None."""
    if isinstance(op, DefineParam):
        if op.fitted is not None or op.param.kind in _SPECS:
            return (
                "a spec is the demonstration's fit: a cleanup renames it (rename_param), never "
                "defines one"
            )
        if op.param.kind == "number" and round(op.param.value, 9) not in _stated(doc):
            return (
                f"{op.param.value:g} is not a number the Memory's text states: a cleanup defines a "
                "parameter for a value a call or stage writes as a literal, never an invented one "
                "or a copy of another parameter's value (rename that parameter instead)"
            )
    if isinstance(op, SetNumber | SetCategory | RotateSpec):
        held = doc.params.get(op.param)
        if held is not None and held.kind.endswith("_spec"):
            return f"{op.param} is a spec: a cleanup leaves its fitted values as they are"
    if isinstance(op, RemoveParam):
        param = doc.params.get(op.name)
        if param is not None and param.source.startswith(f"{FITTED_SOURCE} "):
            return f"{op.name} is the demonstration's fit: a cleanup keeps it"
    return None


def _set(doc: MemoryDoc, op: SetNumber | SetCategory, seed: int | None) -> MemoryDoc:
    param = _param(doc, op.param)
    numeric = isinstance(op, SetNumber)
    fields = numbers(param) if numeric else categories(param)
    if op.field not in fields:
        kind = "numeric" if numeric else "categorical"
        other = "set_category" if numeric else "set_number"
        named = ", ".join("null" if f is None else f for f in fields) or f"none (use {other})"
        raise Rejected(
            f"{op.param} ({param.kind}) has no {kind} field {op.field or 'null'}; its {kind} "
            f"fields: {named}"
        )
    spec = param.value
    if isinstance(spec, GraspSpec) and spec.anchor == "pose" and op.field in POSE_NO_OPS:
        raise Rejected(
            f"{op.param} is a grasp in its pose form: its axes say its orientation, so "
            f"{op.field} does nothing to it; turn it with rotate_spec (about its approach or "
            "closing axis) or move it with offset_m"
        )
    old = fields[op.field]
    if op.field == "arm" and (fitted := _FITTED_ARM.match(param.source)) and op.value != fitted[1]:
        raise Rejected(
            f"{op.param} was fitted at the {fitted[1]} arm's keypoint ({param.source}): its arm "
            "says whose wrist its orientation was measured at, not which arm acts (the workflow's "
            "arm line and plan_arms decide that); the other arm already uses its mirror image"
        )
    if numeric and isinstance(old, tuple) != isinstance(op.value, tuple):
        shape = "[x, y, z]" if isinstance(old, tuple) else "a number"
        raise Rejected(f"{_target(op.param, op.field)} takes {shape}")
    if old == op.value:
        raise Rejected(f"{_target(op.param, op.field)} is {show(old)} already")
    change = Change(
        how="edit", field=op.field, before=jsonable(old), after=jsonable(op.value), seed=seed
    )
    try:
        new = with_field(param, op.field, op.value, change)
    except ValidationError as error:
        raise Rejected(f"{_target(op.param, op.field)}: {_problems(error)}") from None
    return doc.replace(params={**doc.params, op.param: new})


_FITTED_ARM: Final = re.compile(rf"{FITTED_SOURCE} (left|right) K[1-9][0-9]*$")
"""A spec fitted at a demonstration keypoint (`demo <arm> K<n>`): its arm is that keypoint's."""
POSE_NO_OPS: Final = ("approach", "closing")
"""A pose grasp's fields its synthesis ignores (its axes say both): editing one changes nothing."""
_SPECS: Final = ("grasp_spec", "place_spec", "hinge_spec", "path_spec", "skill_spec")


def _rotate(doc: MemoryDoc, op: RotateSpec, seed: int | None) -> MemoryDoc:
    """`op.param`'s pose grasp turned rigidly about its approach or closing axis (see
    `RotateSpec`); its `reference_q` (the contact frame in the robot base: y against the
    approach, z the closing axis) turned by the same body rotation."""
    param = _param(doc, op.param)
    spec = param.value
    if not isinstance(spec, GraspSpec) or spec.anchor != "pose":
        raise Rejected(
            f"{op.param} is not a grasp in its pose form: rotate_spec turns a pose grasp's axes; "
            "turn other grasps with yaw_deg, tilt_deg or lean_deg"
        )
    if op.deg == 0:
        raise Rejected("a turn of 0 deg changes nothing")
    assert spec.approach_axis is not None and spec.closing_axis is not None  # the validator's
    a, c = np.asarray(spec.approach_axis, float), np.asarray(spec.closing_axis, float)
    angle = np.radians(op.deg)
    if op.about == "approach":  # the closing axis about the pointing axis (-y of the contact)
        update = {"closing_axis": _rodrigues(c, a, angle)}
        body = quat_from_axis_angle([0.0, -1.0, 0.0], angle)
    else:  # the pointing axis about the closing axis (z of the contact)
        update = {"approach_axis": _rodrigues(a, c, angle)}
        body = quat_from_axis_angle([0.0, 0.0, 1.0], angle)
    if spec.reference_q is not None:
        turned = quat_normalize(quat_multiply(spec.reference_q, body))
        update["reference_q"] = tuple(round(float(v), 4) for v in turned)
    before = {k: jsonable(getattr(spec, k)) for k in update}
    try:
        value = GraspSpec.model_validate({**dict(spec), **update})
        change = Change(how="edit", field="axes", before=before, after=jsonable(update), seed=seed)
        new = type(param).model_validate(
            {**dict(param), "value": value, "history": [*param.history, change]}
        )
    except ValidationError as error:
        raise Rejected(f"{op.param}: {_problems(error)}") from None
    return doc.replace(params={**doc.params, op.param: new})


def grasp_axes_of(spec: GraspSpec | None) -> tuple[Vec3, Vec3] | None:
    """A pose grasp spec's approach and closing axes; None for another spec or none."""
    if spec is None or spec.anchor != "pose":
        return None
    if spec.approach_axis is None or spec.closing_axis is None:
        return None
    return (spec.approach_axis, spec.closing_axis)


def grasp_axes(param: Any) -> tuple[Vec3, Vec3] | None:
    """A pose grasp's orientation target (`AXES`): its approach and closing axes; None for
    another parameter."""
    spec = getattr(param, "value", None)
    return grasp_axes_of(spec) if isinstance(spec, GraspSpec) else None


def axes_deg(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """The angle (deg) of the rotation between two grasp orientations given as (approach,
    closing) axes: the frames they span, so a quaternion's sign or a rounding never counts."""

    def frame(axes: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(axes[0], dtype=float)
        x = x / np.linalg.norm(x)
        y = np.asarray(axes[1], dtype=float)
        y = y - x * (x @ y)
        y = y / np.linalg.norm(y)
        return np.column_stack([x, y, np.cross(x, y)])

    cos = (float(np.trace(frame(a).T @ frame(b))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _rotated_axes(doc: MemoryDoc, op: RotateSpec) -> tuple[Vec3, Vec3] | None:
    """The orientation `op` would give its grasp (None where it cannot be made)."""
    try:
        turned = _rotate(doc, op, None)
    except Rejected:
        return None
    return grasp_axes(turned.params[op.param])


def _rodrigues(v: Any, axis: Any, angle: float) -> tuple[float, float, float]:
    """`v` turned by `angle` (rad) about `axis`, right-handed, rounded as the Memory shows it."""
    k = axis / np.linalg.norm(axis)
    out = v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * (k @ v) * (1 - np.cos(angle))
    return (round(float(out[0]), 4), round(float(out[1]), 4), round(float(out[2]), 4))


def _fitted(doc: MemoryDoc, name: str) -> str | None:
    """Where a spec parameter was fitted on the demonstration (`left K2`), or None."""
    param = doc.params.get(name)
    if param is None or param.kind not in _SPECS:
        return None
    source = param.source
    return (
        source.removeprefix(f"{FITTED_SOURCE} ") if source.startswith(f"{FITTED_SOURCE} ") else None
    )


def _fit_refusal(doc: MemoryDoc, op: EditOp, matched: Mapping[str, str]) -> str | None:
    """Why `op` would lose or displace a fit (`EditRules.keep_fits`), or None. A spec replacing a
    stage's fit is fitted where that fit was, or where the expert review matched the stage's
    release (`matched`: the arm that releases there in the demonstration, which a fit made for
    another arm's stage is not)."""
    if isinstance(op, RemoveParam) and (where := _fitted(doc, op.name)) is not None:
        return (
            f"{op.name} is the demonstration's fit at {where}: it stays; correct it where the "
            "comparison says (set_number, set_category, or rotate_spec for a pose grasp)"
        )
    if isinstance(op, DefineParam) and op.fitted is not None and op.name in doc.params:
        where = _fitted(doc, op.name)
        users = [s.name for s in doc.workflow if op.name in template_refs(s.call, doc.params)]
        allowed = {where, *(matched[u] for u in users if u in matched)} - {None}
        if allowed and op.fitted not in allowed:
            return (
                f"{op.name} was fitted at {where}: a refit is made where it was fitted, or where "
                f"the review matched its stage's release ({', '.join(sorted(allowed))}), not at "
                f"{op.fitted}"
            )
    if isinstance(op, DefineParam) and op.param.kind in _SPECS and op.fitted is None:
        return (
            "a spec is defined only as the demonstration fits it at the keypoint where its stage "
            "happens, never written by hand: fit_spec(name, at) defines it there; or "
            "correct the existing fit"
        )
    if isinstance(op, EditStage) and op.field == "call":
        index = _stage(doc, op.stage)
        before = set(template_refs(doc.workflow[index].call, doc.params))
        after = set(template_refs(op.text, doc.params))
        others = {
            name
            for stage in doc.workflow
            if stage.name != op.stage
            for name in template_refs(stage.call, doc.params)
        }
        for gone in sorted(before - after):
            where = _fitted(doc, gone)
            if where is None or gone in others:  # another stage's fit, never this stage's own
                continue
            kind = doc.params[gone].kind
            release = matched.get(op.stage)
            allowed = {where} if release is None else {where, release}
            for added in sorted(after - before):
                if doc.params[added].kind == kind and _fitted(doc, added) not in allowed:
                    at = where if release in (None, where) else f"{where} or at {release}"
                    since = (
                        ""
                        if release in (None, where)
                        else f" (the review matched this stage's release to {release}: the "
                        f"{release.split()[0]} arm releases there in the demonstration)"
                    )
                    return (
                        f"stage {op.stage} used {gone}, fitted at {where}: a {kind} replacing it "
                        f"must be fitted at {at}{since}, and {added} is "
                        f"{_fitted(doc, added) or 'not fitted'}; fit one there (fit_spec) or "
                        f"correct {gone} instead"
                    )
    return None


def _highest_lesson(doc: MemoryDoc) -> int:
    return max((int(lesson.id[1:]) for lesson in doc.lessons), default=0)


def _param(doc: MemoryDoc, name: str) -> Any:
    if name not in doc.params:
        raise Rejected(f"no parameter {name}; parameters: {', '.join(doc.params) or 'none'}")
    return doc.params[name]


def _stage(doc: MemoryDoc, name: str) -> int:
    names = [stage.name for stage in doc.workflow]
    if name not in names:
        raise Rejected(f"no workflow stage {name}; stages: {', '.join(names) or 'none'}")
    return names.index(name)


def _lesson(doc: MemoryDoc, lesson_id: str) -> int:
    ids = [lesson.id for lesson in doc.lessons]
    if lesson_id not in ids:
        raise Rejected(f"no lesson {lesson_id}; lessons: {', '.join(ids) or 'none'}")
    return ids.index(lesson_id)


def _problems(error: ValidationError) -> str:
    problems = error.errors(include_url=False, include_context=False, include_input=False)
    return "; ".join(p["msg"].removeprefix("Value error, ") for p in problems)


def targets(doc: MemoryDoc) -> dict[tuple[str, str | None], Any]:
    """Every numeric and categorical target of `doc`'s parameters, with its value, and each pose
    grasp's orientation (`AXES`)."""
    found = {
        (name, field): value
        for name, param in doc.params.items()
        for fields in (numbers, categories)
        for field, value in fields(param).items()
    }
    for name, param in doc.params.items():  # a pose grasp's orientation, compared as a rotation
        if (axes := grasp_axes(param)) is not None:
            found[(name, AXES)] = axes
    return found


def changed_targets(before: MemoryDoc, after: MemoryDoc) -> dict[tuple[str, str | None], Any]:
    """The numeric and categorical targets (`numbers`, `categories`) of parameters both
    documents have, of one kind, that differ in `after`, with their values in `before`: what a
    revision changed (`Editor`'s `reverts`)."""
    out: dict[tuple[str, str | None], Any] = {}
    for name, old in before.params.items():
        new = after.params.get(name)
        if new is None or new.kind != old.kind:
            continue
        for fields in (numbers, categories):
            was, now = fields(old), fields(new)
            for field, value in was.items():
                if field in now and jsonable(now[field]) != jsonable(value):
                    out[(name, field)] = value
        was_axes, now_axes = grasp_axes(old), grasp_axes(new)
        if (
            was_axes is not None
            and now_axes is not None
            and axes_deg(was_axes, now_axes) > SAME_TURN_DEG
        ):
            out[(name, AXES)] = was_axes
    return out


def _target(param: str, field: str | None) -> str:
    return param if field is None else f"{param}.{field}"


def show(value: Any) -> str:
    """A value as an edit's message shows it."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        return "[" + ", ".join(f"{v:g}" for v in value) + "]"
    if isinstance(value, BaseModel):
        return value.model_dump_json(exclude_none=True)
    return repr(value)
