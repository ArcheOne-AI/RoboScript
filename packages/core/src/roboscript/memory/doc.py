"""The Task Memory document (docs/concepts/learning.md): `memory.md` plus `params.json`.

```
# TASK MEMORY: <task>   (<version>, <stage>, <created>, <source>)
## INSTRUCTION  [stable]   the demonstration's instruction: metadata for the linter and S1, never
                           in a model's view (each episode's live instruction says the task)
## TASK         [stable]   the task in one or two sentences over the OBJECTS roles: no arm, no
                           look, a side only where every instruction of the task names it
## OBJECTS      [stable]   abstract roles, how to find them, parts and symmetry, what varies
## WORKFLOW     [stable]   per stage: goal, arm rule, call template, completion predicates
## LESSONS      [dynamic]  condition and action, with the source seed and the merge count
## NOTES                   strategies, not numbers
## SCRATCH                 caches only; never merged
```

Text names parameters as `{name}`; their values are in `params.json`. Documents are immutable;
edits make new ones (`roboscript.memory.edits`). `parse_memory` inverts `render_memory` exactly,
accepts hand-edited files (blank lines, section order) and names the line at fault.
"""

import hashlib
import json
import re
from collections.abc import Collection
from pathlib import Path
from typing import Annotated, Any, Final, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.tools import ToolModel
from roboscript.manipulation.specs import GraspSpec, Name, PlaceSpec, PlaceTarget
from roboscript.memory.params import (
    FITTED_SOURCE,
    PARAMS,
    CategoryParam,
    DoneSpecParam,
    GraspSpecParam,
    HingeSpecParam,
    NumberParam,
    Param,
    PathSpecParam,
    PlaceSpecParam,
    SkillSpecParam,
)
from roboscript.perception.criteria import UNITS
from roboscript.records import write_atomic


def _line(text: str) -> str:
    return " ".join(text.split())


def _phrase(text: str) -> str:
    if not (text := _line(text)):
        raise ValueError("must not be empty")
    return text


def _text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            raise ValueError(
                f"line {number} is a heading: headings belong to the Memory; use lists"
            )
    return "\n".join(lines)


Line = Annotated[str, StringConstraints(max_length=500), AfterValidator(_line)]
"""One line of text: whitespace, newlines included, collapses to single spaces."""
Phrase = Annotated[str, StringConstraints(max_length=500), AfterValidator(_phrase)]
"""A non-empty line."""
Text = Annotated[str, StringConstraints(max_length=6000), AfterValidator(_text)]
"""Lines of text without headings or surrounding blank lines."""
LessonId = Annotated[str, StringConstraints(pattern=r"^L[1-9][0-9]{0,3}$")]
TaskLabel = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")]
Section = Literal["instruction", "statement", "objects", "workflow", "lessons", "notes", "scratch"]
SECTIONS: Final[tuple[Section, ...]] = get_args(Section)
STAGE_FIELDS: Final = ("goal", "arm", "call", "done")
MEMORY_FILE: Final = "memory.md"
PARAMS_FILE: Final = "params.json"
PARAMS_SCHEMA: Final = "rs.memory.params/1"


class WorkflowStage(ToolModel):
    """A workflow stage: its goal; which arm, by a rule the demonstration supports that also
    holds on other scenes (a fixed arm only when the instruction names one); the call template
    (tool names, parameters as {name}); and the completion predicates (such as
    `holding(<arm>) and lifted(<object role>, {<param>})`)."""

    model_config = ConfigDict(frozen=True)

    name: Name
    goal: Line = ""
    arm: Line = ""
    call: Line = ""
    done: Line = ""


class Lesson(BaseModel):
    """A lesson: when it applies and what to do; `seed` is the seed whose cache added it,
    `merges` how many merges added or changed it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: LessonId
    when: Phrase
    do: Phrase
    seed: int | None = None
    merges: NonNegativeInt = 0


class MemoryDoc(BaseModel):
    """A Memory version, a draft or a cache. `version` is None until a store commits it;
    `scratch` is None except in a cache."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskLabel
    version: NonNegativeInt | None = None
    stage: Literal["S1", "S2"] = "S1"
    created: Annotated[str, StringConstraints(pattern=r"^[0-9][0-9T:.+Z-]*$")] | None = None
    source: Line = ""
    instruction: Text = ""
    """The demonstration's instruction (stored, never in a view: `view`)."""
    statement: Text = ""
    """The TASK section: what the task does over the OBJECTS roles, valid for any episode."""
    objects: Text = ""
    workflow: list[WorkflowStage] = []
    lessons: list[Lesson] = []
    notes: Text = ""
    scratch: list[Phrase] | None = None
    params: dict[Name, Param] = {}

    @model_validator(mode="after")
    def _names_are_unique(self) -> "MemoryDoc":
        for kind, names in (
            ("workflow stages", [stage.name for stage in self.workflow]),
            ("lessons", [lesson.id for lesson in self.lessons]),
        ):
            if twice := sorted({name for name in names if names.count(name) > 1}):
                raise ValueError(f"{kind} named twice: {twice}")
        return self

    def replace(self, **update: Any) -> "MemoryDoc":
        """A copy with fields replaced, validated (ValueError)."""
        return MemoryDoc.model_validate({**dict(self), **update})

    @property
    def label(self) -> str:
        return "draft" if self.version is None else f"v{self.version:03d}"


def validation_message(error: ValidationError) -> str:
    """What a validation error says, location first, for a person or a model."""
    problems = error.errors(include_url=False, include_context=False, include_input=False)
    return "; ".join(
        ".".join(map(str, p["loc"]))
        + ": " * bool(p["loc"])
        + p["msg"].removeprefix("Value error, ")
        for p in problems
    )


def content_hash(doc: MemoryDoc) -> str:
    """sha256 of the document without SCRATCH (what an Actor reads), as canonical JSON; fields
    at their defaults are left out, so a field added later with a default does not change it."""
    data = doc.model_dump(mode="json", exclude={"scratch"}, exclude_defaults=True)
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def written_hash(doc: MemoryDoc, params: str | bytes) -> str:
    """A version's hash with defaults included and the parameters as `params` (a `params.json`) has
    them, which still proves a version unchanged after a field was added to the document."""
    data = doc.model_dump(mode="json", exclude={"scratch"})
    data["params"] = json.loads(params)["params"]
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


# memory.md

_TITLES: Final[dict[Section, str]] = {
    section: "TASK" if section == "statement" else section.upper() for section in SECTIONS
}
_ZONES: Final[dict[Section, str]] = {
    "instruction": "stable",
    "statement": "stable",
    "objects": "stable",
    "workflow": "stable",
    "lessons": "dynamic",
}
_HEADER = re.compile(r"^# TASK MEMORY: (?P<task>\S+)(?:\s+\((?P<meta>.*)\))?\s*$")
_SECTION = re.compile(r"^## (?P<title>\S+)(?:\s+\[(?:stable|dynamic)\])?\s*$")
_STAGE = re.compile(r"^### (?P<number>\d+)\. (?P<name>\S+)\s*$")
_FIELD = re.compile(r"^- (?P<field>goal|arm|call|done):(?P<text>.*)$")
_LESSON = re.compile(r"^- (?P<id>L\d+)(?: \((?P<provenance>.*)\))?\s*$")
_LESSON_FIELD = re.compile(r"^\s+(?P<field>when|do):(?P<text>.*)$")
_ITEM = re.compile(r"^- (?P<text>.*)$")
_TEXT: Final = TypeAdapter(Text)
_PHRASE: Final = TypeAdapter(Phrase)


def render_memory(doc: MemoryDoc) -> str:
    """`memory.md`: provenance included, and SCRATCH in a cache."""
    shown = [s for s in SECTIONS if s != "scratch" or doc.scratch is not None]
    return "\n\n".join([_header(doc), *(render_section(doc, s) for s in shown)]) + "\n"


def render_section(doc: MemoryDoc, section: Section, *, provenance: bool = True) -> str:
    """One section; `provenance` adds each lesson's source seed and merge count."""
    zone = _ZONES.get(section)
    title = f"## {_TITLES[section]}" + (f"  [{zone}]" if zone else "")
    if section == "workflow":
        body = "\n\n".join(_stage(i, stage) for i, stage in enumerate(doc.workflow, 1))
    elif section == "lessons":
        body = "\n".join(_lesson(lesson, provenance) for lesson in doc.lessons)
    elif section == "scratch":
        body = "\n".join(f"- {note}" for note in doc.scratch or ())
    else:
        body = getattr(doc, section)
    return f"{title}\n{body}" if body else title


def parse_memory(text: str, params: dict[str, Param] | None = None) -> MemoryDoc:
    """The document `text` holds, with `params`; ValueError naming the file's line at fault."""
    lines = text.splitlines()
    first = next((n for n, line in enumerate(lines, 1) if line.strip()), 1)
    header = _HEADER.match(lines[first - 1]) if first <= len(lines) else None
    if header is None:
        raise ValueError(f"line {first}: expected '# TASK MEMORY: <task>   (<version>, ...)'")
    data: dict[str, Any] = {"task": header["task"]}
    if header["meta"] is not None:
        data |= _parse_meta(first, header["meta"])
    _valid(first, MemoryDoc, data)
    blocks: dict[Section, list[tuple[int, str]]] = {}
    current: list[tuple[int, str]] | None = None
    for number, line in enumerate(lines[first:], first + 1):
        if match := _SECTION.match(line):
            section = next((s for s, title in _TITLES.items() if title == match["title"]), None)
            if section is None or section in blocks:
                what = "an unknown" if section is None else "a repeated"
                raise ValueError(f"line {number}: {what} section {match['title']!r}")
            current = blocks[section] = [(number, line)]
        elif current is not None:
            current.append((number, line))
        elif line.strip():
            raise ValueError(f"line {number}: text before the first section")
    for section, block in blocks.items():
        data[section] = _PARSERS[section](block[0][0], block[1:])
    return MemoryDoc.model_validate({**data, "params": params or {}})


def _header(doc: MemoryDoc) -> str:
    meta = [doc.label, doc.stage, doc.created or "-", *([doc.source] if doc.source else [])]
    return f"# TASK MEMORY: {doc.task}   ({', '.join(meta)})"


def _parse_meta(number: int, meta: str) -> dict[str, Any]:
    parts = meta.split(", ", 3)
    if len(parts) < 3 or not re.fullmatch(r"draft|v\d+", parts[0]):
        raise ValueError(
            f"line {number}: expected '(<v001|draft>, <stage>, <created>[, <source>])', "
            f"got ({meta})"
        )
    label, stage, created, *source = parts
    return {
        "version": None if label == "draft" else int(label[1:]),
        "stage": stage,
        "created": None if created == "-" else created,
        "source": source[0] if source else "",
    }


def _stage(number: int, stage: WorkflowStage) -> str:
    fields = (f"- {field}: {getattr(stage, field)}".rstrip() for field in STAGE_FIELDS)
    return "\n".join([f"### {number}. {stage.name}", *fields])


def _lesson(lesson: Lesson, provenance: bool) -> str:
    shown = ([f"seed {lesson.seed}"] if lesson.seed is not None else []) + (
        [f"{lesson.merges} merge{'s' * (lesson.merges != 1)}"] if lesson.merges else []
    )
    head = f"- {lesson.id}" + (f" ({', '.join(shown)})" if provenance and shown else "")
    return f"{head}\n  when: {lesson.when}\n  do: {lesson.do}"


def _valid(number: int, kind: Any, data: Any) -> Any:
    """`data` validated as `kind` (a model or an adapter), or ValueError naming the line."""
    try:
        return (
            kind.validate_python(data)
            if isinstance(kind, TypeAdapter)
            else kind.model_validate(data)
        )
    except ValidationError as error:
        raise ValueError(f"line {number}: {validation_message(error)}") from None


def _parse_text(number: int, lines: list[tuple[int, str]]) -> str:
    for at, line in lines:
        if line.lstrip().startswith("#"):
            raise ValueError(f"line {at}: a heading inside a section; use lists")
    return _valid(number, _TEXT, "\n".join(line for _, line in lines))  # type: ignore[no-any-return]


def _parse_workflow(number: int, lines: list[tuple[int, str]]) -> list[WorkflowStage]:
    stages: list[tuple[int, dict[str, str]]] = []
    for at, line in lines:
        if match := _STAGE.match(line):
            if any(stage["name"] == match["name"] for _, stage in stages):
                raise ValueError(f"line {at}: a second stage named {match['name']!r}")
            stages.append((at, {"name": match["name"]}))
        elif (match := _FIELD.match(line)) and stages and match["field"] not in stages[-1][1]:
            stages[-1][1][match["field"]] = match[
                "text"
            ].strip()  # a line at its length limit parses back
        elif line.strip():
            raise ValueError(f"line {at}: expected '### <n>. <stage>' or '- goal|arm|call|done:'")
    return [_valid(at, WorkflowStage, stage) for at, stage in stages]


def _parse_lessons(number: int, lines: list[tuple[int, str]]) -> list[Lesson]:
    lessons: list[tuple[int, dict[str, Any]]] = []
    for at, line in lines:
        if match := _LESSON.match(line):
            if any(lesson["id"] == match["id"] for _, lesson in lessons):
                raise ValueError(f"line {at}: a second lesson {match['id']}")
            lessons.append((at, {"id": match["id"], **_parse_provenance(at, match["provenance"])}))
        elif (
            (match := _LESSON_FIELD.match(line))
            and lessons
            and match["field"] not in lessons[-1][1]
        ):
            lessons[-1][1][match["field"]] = match[
                "text"
            ].strip()  # a line at its length limit parses back
        elif line.strip():
            raise ValueError(f"line {at}: expected '- L<n> (...)' or an indented 'when:' / 'do:'")
    return [_valid(at, Lesson, lesson) for at, lesson in lessons]


def _parse_provenance(number: int, text: str | None) -> dict[str, int]:
    found: dict[str, int] = {}
    for part in text.split(", ") if text else ():
        if match := re.fullmatch(r"seed (-?\d+)", part):
            found["seed"] = int(match[1])
        elif match := re.fullmatch(r"(\d+) merges?", part):
            found["merges"] = int(match[1])
        else:
            raise ValueError(f"line {number}: expected 'seed <n>' or '<n> merges', got {part!r}")
    return found


def _parse_scratch(number: int, lines: list[tuple[int, str]]) -> list[str]:
    notes = []
    for at, line in lines:
        if match := _ITEM.match(line):
            notes.append(_valid(at, _PHRASE, match["text"]))
        elif line.strip():
            raise ValueError(f"line {at}: expected '- <note>'")
    return notes


_PARSERS: Final = {
    "instruction": _parse_text,
    "statement": _parse_text,
    "objects": _parse_text,
    "workflow": _parse_workflow,
    "lessons": _parse_lessons,
    "notes": _parse_text,
    "scratch": _parse_scratch,
}


# params.json and files


def dump_params(doc: MemoryDoc) -> str:
    """`params.json`: the parameters in definition order."""
    params = {name: param.model_dump(mode="json") for name, param in doc.params.items()}
    return (
        json.dumps({"schema": PARAMS_SCHEMA, "params": params}, indent=2, ensure_ascii=False) + "\n"
    )


def load_params(text: str | bytes) -> dict[str, Param]:
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"schema", "params"}:
        raise ValueError("expected an object with the keys 'schema' and 'params'")
    if data["schema"] != PARAMS_SCHEMA:
        raise ValueError(f"schema {data['schema']!r} is not {PARAMS_SCHEMA!r}")
    return PARAMS.validate_python(data["params"])


def write_doc(directory: Path, doc: MemoryDoc) -> None:
    """Write `memory.md` and `params.json` into `directory`, each file atomically."""
    write_atomic(directory / MEMORY_FILE, render_memory(doc).encode("utf-8"))
    write_atomic(directory / PARAMS_FILE, dump_params(doc).encode("utf-8"))


def read_doc(directory: Path) -> MemoryDoc:
    try:
        params = load_params((directory / PARAMS_FILE).read_bytes())
        return parse_memory((directory / MEMORY_FILE).read_text(encoding="utf-8"), params)
    except ValueError as error:
        raise ValueError(f"{directory}: {error}") from None


# What models read


_READING: Final[tuple[Section, ...]] = ("statement", "objects", "workflow", "lessons", "notes")


def view(
    doc: MemoryDoc, section: Section | Literal["params"] | None = None, *, detail: bool = True
) -> str:
    """The document for a model: all of it, or one section. With `detail` (the Demo Reader and
    the Revisers, through `memory_read`): the header's provenance, lesson provenance, parameter
    limits and sources, SCRATCH in a cache, and the parameters first, so that a cut keeps them.
    The Actor reads `view(doc, detail=False)` in reading order, the parameters after the workflow
    that names them, and none of the above, so a cache left unchanged reads exactly like M.
    Neither ever shows INSTRUCTION, the demonstration's instruction: a model reads its own
    episode's instruction, and the Memory's TASK says the task in terms of its roles (an empty
    TASK is left out)."""
    if section == "instruction":
        raise ValueError("INSTRUCTION is the demonstration's instruction: no model reads it")
    if section == "params":
        return render_params(doc, detail=detail) + "\n"
    if section is not None:
        return render_section(doc, section, provenance=detail) + "\n"
    params = render_params(doc, detail=detail)
    if not detail:
        reading = [s for s in _READING if s != "statement" or doc.statement]
        cut = reading.index("workflow") + 1
        parts = [render_section(doc, s, provenance=False) for s in reading[:cut]]
        rest = [render_section(doc, s, provenance=False) for s in reading[cut:]]
        return "\n\n".join([f"# TASK MEMORY: {doc.task}", *parts, params, *rest]) + "\n"
    shown = [
        s
        for s in SECTIONS
        if s != "instruction"
        and (s != "scratch" or doc.scratch is not None)
        and (s != "statement" or doc.statement)
    ]
    parts = [render_section(doc, s) for s in shown]
    return "\n\n".join([_header(doc), params, *parts]) + "\n"


def render_params(doc: MemoryDoc, *, detail: bool = True) -> str:
    lines = ["## PARAMETERS"]
    for name, param in doc.params.items():
        limits = _limits(param) if detail else ""
        lines.append(f"- {name} = {_value(param)}" + (f"  ({limits})" if limits else ""))
    return "\n".join(lines)


def _as_shown(a: Any, b: Any) -> bool:
    """Whether two field values read the same as `number` shows them (a value typed back from the
    listing equals its fit): a spec restored to its fitted values reads "as fitted"."""
    numeric = isinstance(a, float | int) and isinstance(b, float | int)
    if numeric and not isinstance(a, bool) and not isinstance(b, bool):
        return number(float(a)) == number(float(b))
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return len(a) == len(b) and all(_as_shown(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def number(value: float) -> str:
    """A number to 0.1 mm or 0.0001 deg, without trailing zeros."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def vector(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(map(number, values)) + "]"


def target_text(target: PlaceTarget) -> str:
    if target.kind == "object":
        where = target.obj if target.part is None else f"{target.obj}.{target.part}"
        return f"{where} {target.anchor}"
    return f"landmark {target.landmark}" if target.kind == "landmark" else "robot_frame"


def _value(param: Param) -> str:
    if isinstance(param, NumberParam):
        return f"{number(param.value)} {param.unit}".rstrip()
    if isinstance(param, CategoryParam):
        return param.value
    if isinstance(param, GraspSpecParam):
        return grasp_text(param.value)
    if isinstance(param, HingeSpecParam):
        hinge = param.value
        fields = [f"part={hinge.part}", f"side={hinge.side}", f"at={hinge.at}"]
        fields += [f"offset_m={vector(hinge.offset_m)}"]
        return f"hinge_spec({', '.join(fields)})"
    if isinstance(param, PathSpecParam):
        first, last = param.value.poses[0], param.value.poses[-1]
        arm = "" if param.value.arm is None else f", arm={param.value.arm}"
        return (
            f"path_spec({len(param.value.poses)} TCP poses in the held object's frame, "
            f"{vector(first[:3])} to {vector(last[:3])}{arm})"
        )
    if isinstance(param, SkillSpecParam):
        shown = json.dumps(json.loads(param.value), separators=(",", ":"))
        shown = shown if len(shown) <= _SKILL_CHARS else shown[: _SKILL_CHARS - 3] + "..."
        return f"skill_spec({param.skill}, params {shown})"
    if isinstance(param, DoneSpecParam):
        done = param.value
        unit = UNITS[done.metric][0]
        return f"done_spec({done.label()} = {number(done.value)} ± {number(done.tol)} {unit})"
    return place_text(param.value)


# A spec as its parameter line shows it: the fields always shown, then every other field that is
# not at its default, so that nothing the spec holds is lost on the way to a model that reads it
# (an Actor transcribing it into a call, a Reviser editing it). `parse_spec` reads it back.
_SKILL_CHARS: Final = 160
"""How much of a skill's parameter object a Memory line shows (the rest is in `params.json`)."""
_GRASP_SHOWN: Final = ("part", "approach", "anchor", "offset_m", "closing", "yaw_deg")
_GRASP_ORDER: Final = (
    "part",
    "approach",
    "anchor",
    "approach_axis",
    "closing_axis",
    "x_toward",
    "offset_m",
    "offset_frac",
    "closing",
    "yaw_deg",
    "tilt_deg",
    "lean_deg",
    "purpose",
    "reference_q",
    "arm",
)
_POSE_UNSHOWN: Final = ("closing", "yaw_deg")
"""A pose grasp's axes say its orientation: these show only where they are not at their
defaults (`closing` does not apply to it, `yaw_deg` is refused)."""
_PLACE_SHOWN: Final = ("target", "frame", "offset_m", "orient", "placement_point")
_PLACE_ORDER: Final = (
    "target",
    "frame",
    "offset_m",
    "orient",
    "yaw_deg",
    "rpy_deg",
    "placement_point",
    "placement_point_at",
    "x_toward",
    "approach",
    "free_turn_deg",
    "region_m",
    "receiver",
    "arm",
    "mirror",
)


def grasp_text(grasp: GraspSpec) -> str:
    """`grasp_spec(...)`: every field the grasp holds (see `_GRASP_SHOWN`)."""
    shown = set(_GRASP_SHOWN) - (set(_POSE_UNSHOWN) if grasp.anchor == "pose" else set())
    return f"grasp_spec({', '.join(_fields(grasp, _GRASP_ORDER, shown))})"


def place_text(place: PlaceSpec) -> str:
    """`place_spec(...)`: every field the placement holds, its receiving grasp nested."""
    return f"place_spec({', '.join(_fields(place, _PLACE_ORDER, set(_PLACE_SHOWN)))})"


def _fields(spec: GraspSpec | PlaceSpec, order: tuple[str, ...], shown: set[str]) -> list[str]:
    known = type(spec).model_fields
    assert set(order) == set(known), f"{type(spec).__name__}: fields {sorted(known)} vs {order}"
    out = []
    for name in order:
        value = getattr(spec, name)
        if name in shown or value != known[name].default:
            out.append(f"{name}={_field_text(value)}")
    return out


def _field_text(value: Any) -> str:
    if isinstance(value, PlaceTarget):
        return target_text(value)
    if isinstance(value, GraspSpec):
        return grasp_text(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float | int):
        return number(float(value))
    if isinstance(value, tuple | list):  # a vector, or a list of them (a path's poses)
        if all(isinstance(v, float | int) and not isinstance(v, bool) for v in value):
            return vector(tuple(value))
        return "[" + ", ".join(map(_field_text, value)) + "]"
    return "null" if value is None else str(value)


_SPEC_TEXT = re.compile(r"^\s*(?P<kind>grasp_spec|place_spec)\((?P<body>.*)\)\s*$", re.DOTALL)
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def parse_spec(text: str) -> GraspSpec | PlaceSpec:
    """The spec a `grasp_spec(...)` or `place_spec(...)` line shows (`grasp_text`,
    `place_text`), validated: what a model reading the line has to write the spec with.
    ValueError when the text is no such spec or the spec it gives is invalid (a pose grasp
    without its axes)."""
    match = _SPEC_TEXT.match(text)
    if match is None:
        raise ValueError(f"not a grasp_spec(...) or place_spec(...): {text[:60]!r}")
    data: dict[str, Any] = {}
    for item in _split(match["body"]):
        name, sep, raw = item.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep or not name:
            raise ValueError(f"expected <field>=<value>, got {item.strip()!r}")
        if name in data:
            raise ValueError(f"{name} given twice")
        data[name] = _target(raw) if name == "target" else _raw_value(raw)
    kind = GraspSpec if match["kind"] == "grasp_spec" else PlaceSpec
    try:
        return kind.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{match['kind']}: {validation_message(error)}") from None


def as_read(param: GraspSpecParam | PlaceSpecParam) -> GraspSpec | PlaceSpec:
    """The spec a model reading `param`'s line can write (`parse_spec` of `_value`): what an
    Actor sends when it transcribes it faithfully. ValueError when the line gives no valid spec."""
    return parse_spec(_value(param))


def lost_fields(spec: GraspSpec | PlaceSpec, read: GraspSpec | PlaceSpec) -> list[str]:
    """The fields of `spec` its line does not carry (`read` = `as_read`), as shown, `name=value`."""
    return [f"{name}={mine}" for name, mine, _ in field_diff(spec, read)]


def field_diff(
    a: BaseModel, b: BaseModel, skip: Collection[str] = ()
) -> list[tuple[str, str, str]]:
    """The fields two specs of one kind differ in, as the Memory shows them: (field, a's, b's)."""
    return [
        (name, _field_text(getattr(a, name)), _field_text(getattr(b, name)))
        for name in type(a).model_fields
        if name not in skip and not _same(getattr(a, name), getattr(b, name))
    ]


def _split(body: str) -> list[str]:
    """`body` split at its top-level commas (none inside brackets or parentheses)."""
    items, depth, start = [], 0, 0
    for at, char in enumerate(body):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(body[start:at])
            start = at + 1
    items.append(body[start:])
    return [item for item in items if item.strip()]


def _raw_value(raw: str) -> Any:
    if raw.startswith("grasp_spec("):
        return parse_spec(raw)
    if raw.startswith("[") and raw.endswith("]"):
        return [_raw_value(item.strip()) for item in _split(raw[1:-1])]
    if raw in ("true", "false"):
        return raw == "true"
    if raw == "null":
        return None
    return float(raw) if _NUMBER.match(raw) else raw


def _target(raw: str) -> PlaceTarget:
    if raw == "robot_frame":
        return PlaceTarget(kind="robot_frame")
    head, _, tail = raw.partition(" ")
    if head == "landmark":
        return PlaceTarget.model_validate({"kind": "landmark", "landmark": tail})
    obj, _, part = head.partition(".")
    return PlaceTarget.model_validate(
        {"kind": "object", "obj": obj, "part": part or None, "anchor": tail}
    )


def _limits(param: Param) -> str:
    if isinstance(param, NumberParam):
        low, high = param.bounds
        text = f"bounds [{number(low)}, {number(high)}], trust ±{number(param.trust)}"
        text += f", frame {param.frame}" if param.frame else ""
    elif isinstance(param, CategoryParam):
        text = f"choices: {', '.join(param.choices)}" if param.choices else ""
    elif isinstance(param, DoneSpecParam):
        unit, (low, high), trust = UNITS[param.value.metric]
        text = f"value within [{number(low)}, {number(high)}] {unit}, trust ±{number(trust)} {unit}"
    elif isinstance(param, PathSpecParam):
        text = "as fitted on the demonstration, followed whole (follow_path)"
    elif isinstance(param, SkillSpecParam):
        text = f"as fitted on the demonstration, run whole (run_skill with skill {param.skill})"
    elif isinstance(param, HingeSpecParam):
        trust, bounds = param.trust, param.bounds
        text = (
            f"trust ±{number(trust.offset_m)} m offset; bounds {vector(bounds.offset_m)} m per "
            "offset component"
        )
    else:
        trust, bounds = param.trust, param.bounds
        robot = isinstance(param, PlaceSpecParam) and param.value.target.kind == "robot_frame"
        text = (
            f"trust ±{number(trust.offset_m)} m offset, ±{number(trust.yaw_deg)} deg yaw; "
            f"bounds {vector(bounds.robot_frame_m if robot else bounds.offset_m)} m per offset "
            f"component, {vector(bounds.yaw_deg)} deg yaw"
        )
        if isinstance(param, GraspSpecParam):
            text += (
                f"; ±{number(trust.offset_frac)} of the part's extents offset, within "
                f"{vector(bounds.offset_frac)} per component"
            )
            text += f"; tilt ±{number(trust.tilt_deg)} deg within {vector(bounds.tilt_deg)} deg"
            text += f"; lean ±{number(trust.lean_deg)} deg within {vector(bounds.lean_deg)} deg"
        else:
            text += f"; region ±{number(trust.region_m)} m within {vector(bounds.region_m)} m"
            text += f"; rpy ±{number(trust.rpy_deg)} deg within {vector(bounds.rpy_deg)} deg"
    source = f"from {param.source}" if param.source else ""
    return "; ".join(part for part in (text, source, _drift(param)) if part)


def _drift(param: Param) -> str:
    """How a spec fitted on the demonstration differs from its fit (the value it was defined
    with): the fields changed since, at their fitted values; empty when none has."""
    if not param.source.startswith(f"{FITTED_SOURCE} ") or not param.history:
        return ""
    first = param.history[0]
    spec = param.value
    if first.how != "define" or not isinstance(first.after, dict):
        return ""
    fit = type(spec).model_validate(first.after)
    # every field, the pose grasp's axes and reference included
    changed = [f for f in type(spec).model_fields if not _same(getattr(fit, f), getattr(spec, f))]
    if not changed:
        return "as fitted"
    shown = [f"{field}={_field_text(getattr(fit, field))}" for field in changed]
    return "changed since fitted; as fitted: " + ", ".join(shown)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, BaseModel) or isinstance(b, BaseModel):
        return _field_text(a) == _field_text(b)
    return a is b is None or (a is not None and b is not None and _as_shown(a, b))


PLACE_CALL_ARG: Final = re.compile(r"\b(approach_m|stop_back_m|constrain)\s*[:=]\s*([^,()]+)")
"""A `place_object` argument a workflow call line passes (`approach_m:0.07`, `constrain: align`,
`stop_back_m:{place_stop_back}`)."""
PLACE_DEFAULTS: Final[dict[str, float | str]] = {
    "approach_m": 0.1,
    "stop_back_m": 0.02,
    "constrain": "auto",
}
"""`place_object`'s defaults for the arguments read off a call."""
_CONSTRAINS: Final = ("auto", "align", "free")


def place_call_args(doc: MemoryDoc, name: str) -> dict[str, float | str]:
    """The `place_object` arguments the workflow passes with the placement spec `name` (its first
    stage whose call names `{name}`): `approach_m`, `stop_back_m` and `constrain`, a `{param}` read
    from the Memory, else the call's defaults."""
    found = dict(PLACE_DEFAULTS)
    stage = next(
        (s for s in doc.workflow if "place_object" in s.call and f"{{{name}}}" in s.call), None
    )
    if stage is None:
        return found
    for key, raw in PLACE_CALL_ARG.findall(stage.call):
        text = raw.strip().strip("'\"")
        if (ref := re.fullmatch(r"\{(\w+)\}", text)) is not None:
            param = doc.params.get(ref[1])
            if not isinstance(param, NumberParam | CategoryParam):
                continue
            text = str(param.value)
        if key == "constrain":
            if text in _CONSTRAINS:
                found[key] = text
            continue
        try:
            found[key] = float(text)
        except ValueError:
            continue
    return found
