"""The Memory linter (docs/concepts/learning.md): pure and deterministic, run on every write.

Vocabulary rules come from the recipe (e.g. `recipes/wicl/memory_lint.yaml`): each has a
message and words or regular expressions (any case) that must not appear in Memory text,
parameter names and labels, such as asset annotations, simulator names, seed numbers, evaluator
words, episode facts and absolute positions. Words match between alphanumeric boundaries, so `_`
separates them as a space does (`seed_10003_yaw` holds `seed`); patterns use the same boundaries,
written `(?<![^\\W_])` and `(?![^\\W_])`. A rule may apply only under one way of writing fixed
target regions, and its `role_words` and `role_patterns` apply to the OBJECTS role descriptions
alone (abstract roles: which object plays a role comes from each episode's instruction). Built in:
`objects` (an OBJECTS line that is no role line, or a role given twice), `words` (the text's
length), `lessons` (how many), `unknown_param` (`{name}` or `{name.field}` naming nothing),
`fixed_target` (robot-frame constants while fixed target regions must be landmarks), `fixed_arm`
(a workflow, OBJECTS, NOTES or lesson line that binds the left or the right arm, or a category
parameter whose value is one: which arm acts depends on where the object lies and on what each
episode's instruction names, so a stage's arm line is `the arm plan_arms gives for <stage>`
(`plan_arms` plans every stage's arm in each episode), never a side, not even with the
demonstration's instruction as the reason; only text naming both arms as alternatives is exempt
(INSTRUCTION, the demonstration's instruction, is metadata no model reads and is not linted);
wicl's `unbind_arms` rewrites Memories that bind one), `done_role`
(a done criterion relating something other than an OBJECTS role or the table) and the warning
`unused_param` (done criteria are exempt: `check_done` uses them, not text). A PlaceSpec may
always target the robot base frame (a hand-over point, the first block of a stack); a GraspSpec
has no field that could: a grasp is relative to an object or a part. Errors block a write;
warnings do not. SCRATCH is never linted: it is never merged.

Warnings (advice an edit's result carries; never errors, since an editor refuses a Memory that
fails the linter, and a frozen Memory must stay editable): a vocabulary rule with `level:
warning` (privileged signals and bare keypoint ids), `appearance_name` (a role name or a
`query=` naming a colour or material word the demonstration's INSTRUCTION also holds, from the
recipe's `appearance_words`: roles are functional, the live instruction says which object plays
them), `foreign_object` (OBJECTS or WORKFLOW naming an object kind from the recipe's
`object_words` that the demonstration's INSTRUCTION does not name: an object of another
episode, whose instruction named other objects, left in text that every episode reads; a
stage naming one kind while OBJECTS names another is a Memory no scene fits), `target_role` (a
PlaceSpec target naming no OBJECTS role), `spec_drift` (a spec fitted on
the demonstration whose numbers or categories moved from the fit, which is the first `define` in
its history, by more than the config's bounds), `handwritten_spec` (a spec with no fit) and
`done_untestable` (a done line that holds no complete predicate call, or a workflow none of
whose done lines states a result on an object), `absolute_height` (a length parameter with no
frame of `LintConfig.absolute_height_m` or more: a world height, such as a table's or a lift's
end, which another scene's table, lower or higher, moves; a Memory binds heights relative to
the table, an object or the robot base, and a lift's or gap's own size is far smaller),
`undirected_carry` (a role the workflow ends holding at a robot-frame placement with no `arm
side:` line: the arm that carries it then follows distance, and when the object's direction
decides the end, as a one-ended object does, the nearer arm carries it the wrong way round; `arm
side: either` says it does not matter), `relative_carry` (a role the demonstration ends holding
where an arm carried it, over no anchor (`LintConfig.carried_ends`, from the demonstration's
end targets and keypoints), that no placement places (displacements alone carry it): a carry
written as a displacement from its start lands wherever the start puts it, and its relative
done criteria pass anywhere; where the end is the robot's, a PlaceSpec with a
robot-frame target is the allowed form; a role with an articulated part is left out, a part
moves on its body).

`side_bound` (a side of the robot in a done criterion that TASK does not name: an `offset_x`
criterion without `arm`, a done line's `beside` toward left or right; such a side came from the
demonstration's arm and is written relative to the acting arm, `perception.sides`), the warnings
`side_order` (a role ordered by the robot's left to right while TASK names no side) and
`no_statement` (an empty TASK).

`instance_name` (a role description naming a kind of object, from `object_words`, that one of
the task's known instructions (`LintConfig.instructions`: the demonstration's and the ones S2
has seen) names and another does not: the instruction varies the objects, so the description
names one episode's instance, and the identification would search for it in every other;
`instance_level` ranks it, a warning unless the recipe makes it an error), the warning
`alias_role` (a role whose description calls it an alias of, or the same object as,
another role: one object is one role, else the identification registers it twice or not at all;
merge them and use the one name everywhere; a warning, since an error would refuse to open a
Memory that has one) and `spec_arm_rule` (text that has the Actor take the acting arm
from a spec, "the fitted arm encoded in the spec", or not ask `plan_arms`: a spec's arm is whose
wrist it was measured at, not which arm acts; and a spec fitted at one arm's keypoint whose arm
an edit set to null, which the other arm then uses as written instead of mirrored).
"""

import math
import re
from collections.abc import Iterator, Mapping
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from roboscript.manipulation.specs import GEOMETRIC_ENDS, OTHER_ARM, Name
from roboscript.memory.calls import (
    ToolSig,
    demo_quantities,
    literal_numbers,
    parse_calls,
    part_names,
    spec_mismatches,
    template_refs,
    unbraced,
    unknown_calls,
)
from roboscript.memory.doc import STAGE_FIELDS, MemoryDoc
from roboscript.memory.objects import parse_objects
from roboscript.memory.params import (
    FITTED_SOURCE,
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
from roboscript.memory.sided import sided
from roboscript.perception.criteria import TABLE
from roboscript.perception.predicates import PREDICATES
from roboscript.perception.sides import ARM_SIDE, OTHER_SIDE, task_sides

BUILTIN: Final = (
    "objects",
    "words",
    "lessons",
    "unknown_param",
    "fixed_target",
    "fixed_arm",
    "unused_param",
    "unfitted_spec",
    "done_role",
    "appearance_name",
    "foreign_object",
    "target_role",
    "spec_drift",
    "handwritten_spec",
    "done_untestable",
    "unknown_tool",
    "wrong_param_kind",
    "unparameterised",
    "unbraced_param",
    "undefined_frame",
    "unknown_part",
    "unregistered_part",
    "absolute_height",
    "undirected_carry",
    "relative_carry",
    "instance_name",
    "alias_role",
    "spec_arm_rule",
    "side_bound",
    "side_order",
    "no_statement",
)
START: Final = r"(?<![^\W_])"
END: Final = r"(?![^\W_])"
_REF: Final = re.compile(r"\{([a-z][a-z0-9_]*)(?:\.([a-z][a-z0-9_]*))?\}")
_MATCH_CHARS: Final = 80
SIDES: Final = ("left", "right")
"""The arm names of the two-arm convention (`contracts.robot.Arm`), which `fixed_arm` looks for."""
_SIDE: Final = r"(?:left|right)"
_ARM_WORD: Final = re.compile(START + _SIDE + END, re.IGNORECASE)
"""In a stage's arm line, any left or right binds the arm."""
_ARM_BOUND: Final = re.compile(
    "|".join(
        (
            START  # the left arm (not a side, nor the fingers: where, not who)
            + _SIDE
            + r"[- ](?:arms?|hands?|grippers?)"
            + END
            + r"(?![- ]side)(?!\s+(?:fingers?|jaws?|tips?|links?|joints?)\b)",
            START + r"arm\s*[=:]\s*" + _SIDE + END,  # arm=right
            r"\(\s*" + _SIDE + r"\s*[,)]",  # holding(left), near(right, ...)
        )
    ),
    re.IGNORECASE,
)
"""Elsewhere, a left or right that binds an arm (not one that says where something lies)."""
_EITHER: Final = re.compile(
    START + _SIDE + r"(?:[- ](?:arms?|hands?|grippers?))?\s*(?:or|and|/|,|vs\.?)\s*" + _SIDE + END,
    re.IGNORECASE,
)
"""Text that states the rule names both arms as alternatives."""
_SPEC_ARM_RULE: Final = re.compile(
    r"(?:do\s+not|don't|never|\bnot|without|instead\s+of|rather\s+than)\s+"
    r"(?:call(?:ing)?\s+|ask(?:ing)?\s+|us(?:e|ing)\s+)?(?:plan_arms|choose_arm)"
    r"|\b(?:do\s+not|don't|never|not)\b[^.;:,\n]{0,80}?\b(?:by|via|from|through|with|using)\s+"
    r"(?:call(?:ing)?\s+|ask(?:ing)?\s+)?(?:plan_arms|choose_arm)"
    r"|(?:do\s+not|don't|never)\s+substitute\s+the\s+near(?:er|est)\s+arm"
    r"|(?:fitted|spec's|demonstration-derived|demo-derived)\s+arm\b"
    r"|\barm\s+(?:encoded|stored|recorded|written|saved)\s+(?:in|on|with)\s+(?:the\s+)?"
    r"(?:[\w-]+\s+){0,3}spec\b"
    r"|\barm\s+of\s+the\s+(?:[\w-]+\s+){0,3}spec\b",
    re.IGNORECASE,
)
"""Text that has the Actor take the acting arm from a spec, or not ask `plan_arms`: a spec's
`arm` is whose wrist it was measured at, not which arm acts."""
_SPEC_ARM: Final = (
    "a spec's arm says whose wrist it was measured at (the other arm uses its mirror image), "
    "not which arm acts: write the stage's arm line `the arm plan_arms gives for <stage>`, "
    "which the Actor asks plan_arms for"
)
_FIXED_ARM: Final = (
    "which arm acts depends on each scene and instruction: plan_arms plans it in each episode, "
    "so write the stage's arm line `the arm plan_arms gives for <stage>` and arm=<arm> (<arm of "
    "<role>> for another role's hold) in calls and checks, never a fixed left or right arm, not "
    "even the one the demonstration's instruction named"
)

FixedTargets = Literal["landmark", "robot_frame"]


class VocabularyRule(BaseModel):
    """Text that must not appear: words or phrases, and regular expressions; `role_words` and
    `role_patterns` only in OBJECTS role descriptions; with `fixed_targets`, only while fixed
    target regions are written that way. A rule of `level` warning only advises."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message: str
    words: list[str] = []
    patterns: list[str] = []
    role_words: list[str] = []
    role_patterns: list[str] = []
    fixed_targets: FixedTargets | None = None
    level: Literal["error", "warning"] = "error"

    @field_validator("patterns", "role_patterns")
    @classmethod
    def _compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"pattern {pattern!r}: {error}") from None
        return patterns

    @model_validator(mode="after")
    def _not_empty(self) -> "VocabularyRule":
        if not (self.words or self.patterns or self.role_words or self.role_patterns):
            raise ValueError("a rule needs words or patterns")
        return self

    def regex(self) -> re.Pattern[str] | None:
        return _regex(self.words, self.patterns)

    def role_regex(self) -> re.Pattern[str] | None:
        """What must not appear in a role description, beyond `regex`."""
        return _regex(self.role_words, self.role_patterns)


class LintConfig(BaseModel):
    """The recipe's vocabulary rules by name, the text's word limit, the lesson limit, and how
    fixed target regions are written besides a PlaceSpec's robot-frame target: relative to a
    `landmark` (default) or as `robot_frame` numbers. `appearance_words`: the
    words (colours, materials, textures) that name how an object looks, not what it is for;
    `object_words`: kinds of object (`foreign_object`); `spec_drift_*`: how far a spec fitted on
    the demonstration may move from its fit before a warning (position in meters, a grasp's
    fractional offset, angles in degrees). `tools`: the tools a template may call with their
    parameters, set by whoever mounts the toolsets (empty: the call check is off);
    `executable_level`: how the checks that a Memory can be carried out as written
    (`unknown_tool`, `unparameterised`, `undefined_frame`) rank; `absolute_height_m`: a length
    parameter with no frame this long or longer is a world height (`absolute_height`; None:
    off); `instructions`: the task's instructions known so far, the demonstration's included
    (`instance_name`; fewer than two: off), set by whoever runs the editor; `instance_level`: how
    `instance_name` ranks; `carried_ends`: the roles the demonstration ends holding where an
    arm carried them, over no anchor (`relative_carry`; empty: off), set by whoever reads its
    end targets (`tools.done.carried_ends`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: dict[Name, VocabularyRule] = {}
    max_words: PositiveInt = 1200
    max_lessons: PositiveInt = 12
    fixed_targets: FixedTargets = "landmark"
    appearance_words: list[str] = []
    object_words: list[str] = []
    spec_drift_m: PositiveFloat = 0.05
    spec_drift_frac: PositiveFloat = 0.5
    spec_drift_deg: PositiveFloat = 30.0
    tools: dict[Name, ToolSig] = {}
    executable_level: Literal["error", "warning"] = "error"
    absolute_height_m: PositiveFloat | None = None
    instructions: list[str] = []
    instance_level: Literal["error", "warning"] = "warning"
    carried_ends: list[str] = []

    @field_validator("rules")
    @classmethod
    def _not_builtin(cls, rules: dict[str, VocabularyRule]) -> dict[str, VocabularyRule]:
        if taken := sorted(set(rules) & set(BUILTIN)):
            raise ValueError(f"rule names {taken} are built in")
        return rules


class Finding(BaseModel):
    """What broke which rule where: `where` is a section line (`objects:3`), a field
    (`workflow.grasp.call`, `lessons.L2.do`, `params.grasp_a.part`) or `memory`. Errors block a
    write; warnings are advice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    where: str
    match: str = ""
    message: str
    level: Literal["error", "warning"] = "error"


def lint(doc: MemoryDoc, config: LintConfig) -> list[Finding]:
    """Every finding in `doc`, errors and warnings, in document order."""
    return _lint(doc, config)


def _lint(doc: MemoryDoc, config: LintConfig) -> list[Finding]:
    findings: list[Finding] = []
    texts = list(prose(doc))
    active = {
        name: rule
        for name, rule in config.rules.items()
        if rule.fixed_targets in (None, config.fixed_targets)
    }
    roles, problems = parse_objects(doc.objects)
    for number, why in problems:
        where = f"objects:{number}"
        line = doc.objects.splitlines()[number - 1]
        findings.append(Finding(rule="objects", where=where, match=_cut(line), message=why))
    checks = [(where, text, "regex") for where, text in [*texts, *_labels(doc)]]
    checks += [(f"objects:{role.number}", role.description, "role_regex") for role in roles]
    checks += [
        (f"objects:{role.number + 1}", role.toward, "role_regex") for role in roles if role.toward
    ]
    for where, text, kind in checks:
        for name, rule in active.items():
            regex = getattr(rule, kind)()
            if regex is None:
                continue
            for match in dict.fromkeys(m.group(0) for m in regex.finditer(text)):
                finding = Finding(
                    rule=name,
                    where=where,
                    match=_cut(match),
                    message=rule.message,
                    level=rule.level,
                )
                findings.append(finding)
    findings += _fixed_arms(texts)
    findings += _spec_arm_rules(doc, texts)
    names = {role.role for role in roles}
    findings += _appearance(doc, config, texts, roles)
    findings += _foreign(doc, config, texts)
    findings += _instances(config, roles)
    findings += _aliases(roles)
    findings += _undirected(doc, roles)
    findings += _relative_carries(doc, config, roles)
    findings += _done_lines(doc)
    findings += _sides(doc, roles)
    findings += _executable(doc, config, texts, roles)
    used: set[str] = set()
    for where, text in texts:
        used.update(unbraced(text, doc.params))
        for name, field in dict.fromkeys(_REF.findall(text)):
            used.add(name)
            if problem := _unknown(doc.params.get(name), field):
                ref = f"{{{name}.{field}}}" if field else f"{{{name}}}"
                findings.append(
                    Finding(rule="unknown_param", where=where, match=ref, message=problem)
                )
    words = sum(len(text.split()) for _, text in texts)
    if words > config.max_words:
        message = f"the text has more than {config.max_words} words: shorten or merge, not append"
        findings.append(Finding(rule="words", where="memory", match=str(words), message=message))
    if len(doc.lessons) > config.max_lessons:
        message = f"more than {config.max_lessons} lessons: merge or replace, not add"
        count = str(len(doc.lessons))
        findings.append(Finding(rule="lessons", where="lessons", match=count, message=message))
    for name, param in doc.params.items():
        where = f"params.{name}"
        if arm_category(param):
            findings.append(
                Finding(rule="fixed_arm", where=where, match=param.value, message=_FIXED_ARM)
            )
        if config.fixed_targets == "landmark" and _robot_frame(param):
            message = (
                "fixed target regions are written relative to a landmark (table_center, "
                "front_edge_mid, robot_midline) or as a PlaceSpec's robot_frame target, not as "
                "robot-frame numbers"
            )
            findings.append(
                Finding(rule="fixed_target", where=where, match="robot_frame", message=message)
            )
        findings += _spec_findings(name, param, names, config)
        if (height := _world_height(param, config)) is not None:
            message = (
                f"{height:g} m reads as a world height, which another scene's table moves: bind "
                "a height relative to the table, an object or the robot base (a lift by, a gap "
                "over, a robot-frame place target), never a world z"
            )
            findings.append(_warn("absolute_height", where, f"{height:g} m", message))
        if isinstance(param, DoneSpecParam):
            for other in (param.value.a, param.value.b):
                if other is not None and other not in names | {TABLE}:
                    message = "a done criterion relates OBJECTS roles (gap_z: or the table)"
                    findings.append(
                        Finding(rule="done_role", where=where, match=other, message=message)
                    )
        elif name not in used:
            message = "no text names this parameter: name it where it is used, or remove it"
            findings.append(
                Finding(
                    rule="unused_param", where=where, match=name, message=message, level="warning"
                )
            )
    return findings


def unfitted(doc: MemoryDoc) -> list[Finding]:
    """The specs (grasp, place, hinge, path) not fitted on the demonstration (`define_param` with
    `fitted`): errors in S1's draft, whose specs must reproduce the demonstration."""
    message = (
        "a spec in S1's draft is fitted on the demonstration: define it from demo_relation's "
        "result, with `fitted` naming where"
    )
    return [
        Finding(rule="unfitted_spec", where=f"params.{name}", match=name, message=message)
        for name, param in doc.params.items()
        if isinstance(
            param, GraspSpecParam | PlaceSpecParam | HingeSpecParam | PathSpecParam | SkillSpecParam
        )
        and not param.source.startswith(f"{FITTED_SOURCE} ")
    ]


_OTHER_ARM: Final = re.compile(r"\bother arm\b|<other arm>", re.IGNORECASE)
"""A stage's arm line that has it act with the other arm than the one before it."""


def spec_arms(doc: MemoryDoc) -> list[Finding]:
    """The stages that pass a spec fitted at another arm's keypoint (errors in S1's finish
    checks, never in `lint`, so stored Memories stay valid): a fitted spec passed by stages
    whose arm lines put them on different arms (one written as the other arm, one not: a
    hand-over's receiving grasp reusing the giver's), or twice in one call (`together` giving it
    to both arms).
    Each arm's grasp is fitted at its own keypoint (the Reviser's `fit_spec`)."""
    found: list[Finding] = []
    users: dict[str, list[tuple[str, bool]]] = {}
    for stage in doc.workflow:
        for name in template_refs(stage.call, doc.params):
            param = doc.params[name]
            if not param.kind.endswith("_spec") or not param.source.startswith(f"{FITTED_SOURCE} "):
                continue
            where = param.source.removeprefix(f"{FITTED_SOURCE} ")
            ref = re.compile(rf"\{{{re.escape(name)}[.}}]")
            passed = max(
                (
                    len(ref.findall(" ".join([*call.keywords.values(), *call.values])))
                    for call in parse_calls(stage.call)
                ),
                default=0,
            )
            if passed > 1:
                message = (
                    f"{name} is fitted at {where}, one arm's keypoint, and this call passes it "
                    "to more than one arm: fit each arm's own at its keypoint (fit_spec)"
                )
                found.append(
                    Finding(
                        rule="spec_arm",
                        where=f"workflow.{stage.name}.call",
                        match=name,
                        message=message,
                    )
                )
            users.setdefault(name, []).append((stage.name, bool(_OTHER_ARM.search(stage.arm))))
    for name, used in users.items():
        first = used[0][1]
        where = doc.params[name].source.removeprefix(f"{FITTED_SOURCE} ")
        for stage, other in used[1:]:
            if other == first:
                continue
            message = (
                f"{name} is fitted at {where}, the keypoint of stage {used[0][0]}'s arm; this "
                "stage acts with the other arm: fit that arm's own spec at the keypoint where "
                "this stage happens (fit_spec) and pass it here"
            )
            found.append(
                Finding(
                    rule="spec_arm", where=f"workflow.{stage}.call", match=name, message=message
                )
            )
    return found


_ARM_RULE: Final = re.compile(
    r"plan_arms|choose_arm|other arm|holding|holds|both arms|instruction names|<arm>|<other arm>"
    r"|<arm of ",
    re.IGNORECASE,
)
"""An arm line that states a rule `plan_arms` resolves (in S1, to the demonstration's arm)."""
_ACTS: Final = re.compile(r"\barm\s*=")


def draft_advice(doc: MemoryDoc) -> list[Finding]:
    """What S1's draft should still change (the Demo Reader's finish is refused, a bounded
    number of times, while any is left; never an error elsewhere: stored Memories stay valid):
    `grasp_form`, a grasp fitted on the demonstration and picked up that is not in its pose
    form (the expert's TCP in the whole object's frame, which no part box re-registered on the
    next scene moves; a moving part's grasp, a drawer's handle, may keep its part form), and
    `stage_arm`, a stage that acts with an arm whose arm line states no rule `plan_arms`
    resolves (`plan_arms`' arm for the stage, the other arm, the arm holding a role; not one that
    takes the arm from a spec or skips `plan_arms`, `spec_arm_rule`), so that in S1 the arm is the
    demonstration's; and `statement`, an empty TASK."""
    found: list[Finding] = []
    if not doc.statement.strip():
        message = (
            "write TASK (set_text, section statement): what the task does over the OBJECTS "
            "roles, in one or two sentences that hold for every episode, with no arm and no look"
        )
        found.append(Finding(rule="statement", where="statement", match="", message=message))
    for name, param in doc.params.items():
        if not isinstance(param, GraspSpecParam) or not param.source.startswith(FITTED_SOURCE):
            continue
        if param.value.anchor != "pose" and param.value.purpose == "pick":
            message = (
                "define this grasp in its pose form (demo_relation's first, at the keypoint where "
                "the gripper closes), unless it grasps a moving part"
            )
            found.append(
                Finding(rule="grasp_form", where=f"params.{name}", match=name, message=message)
            )
    for stage in doc.workflow:
        if _ACTS.search(stage.call) and (
            not _ARM_RULE.search(stage.arm) or _SPEC_ARM_RULE.search(stage.arm)
        ):
            message = (
                "write the stage's arm line as the arm plan_arms gives for <stage> (the arm "
                "holding <role> where a stage continues a hold)"
            )
            where = f"workflow.{stage.name}.arm"
            found.append(
                Finding(
                    rule="stage_arm",
                    where=where,
                    match=_cut(stage.arm or "(none)"),
                    message=message,
                )
            )
    return found


def errors(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.level == "error"]


def _unknown(param: Param | None, field: str) -> str:
    """Why `{name}` or `{name.field}` names nothing, or ""."""
    if param is None:
        return "names no parameter: define it first (define_param) or fix the name"
    if field and (
        isinstance(param, NumberParam | CategoryParam)
        or field not in type(param.value).model_fields
    ):
        return f"a {param.kind} has no field {field}"
    return ""


def arm_bindings(where: str, text: str) -> list[tuple[int, int]]:
    """The spans of `text` (at `where`, as `prose` names it) that bind the left or the right
    arm: in a stage's arm line any side, elsewhere a side that binds an arm (`_ARM_BOUND`); none
    in INSTRUCTION (the demonstration's own instruction), nor where the text names both arms as
    alternatives (`_EITHER`'s span alone: other bindings on the line still count)."""
    if where.startswith("instruction:"):
        return []
    section, _, rest = where.partition(".")
    field = rest.rpartition(".")[2]
    regex = _ARM_WORD if section == "workflow" and field == "arm" else _ARM_BOUND
    either = [m.span() for m in _EITHER.finditer(text)]
    return [
        m.span()
        for m in regex.finditer(text)
        if not any(start < m.end() and m.start() < end for start, end in either)
    ]


def arm_category(param: Param) -> bool:
    """A category whose value is an arm (`left`, `right`): a fixed arm, as a parameter."""
    return isinstance(param, CategoryParam) and param.value in SIDES


def _fixed_arms(texts: list[tuple[str, str]]) -> list[Finding]:
    """`fixed_arm`: prose that binds the left or the right arm (`arm_bindings`)."""
    found: list[Finding] = []
    for where, text in texts:
        spans = arm_bindings(where, text)
        for match in dict.fromkeys(text[start:end] for start, end in spans):
            found.append(
                Finding(rule="fixed_arm", where=where, match=_cut(match), message=_FIXED_ARM)
            )
    return found


def spec_arm_text(doc: MemoryDoc) -> list[Finding]:
    """The workflow and lesson text that takes the acting arm from a spec or skips `plan_arms`
    (`spec_arm_rule`'s text findings, as errors): the S1 Reviser's finish is refused while any
    is left, never a stored Memory's validity (in `lint` they are warnings)."""
    texts = [(w, t) for w, t in prose(doc) if w.startswith(("workflow.", "lessons."))]
    return [
        f.model_copy(update={"level": "error"})
        for f in _spec_arm_rules(doc, texts)
        if not f.where.startswith("params.")
    ]


def _spec_arm_rules(doc: MemoryDoc, texts: list[tuple[str, str]]) -> list[Finding]:
    """`spec_arm_rule`, a warning: text that takes the acting arm from a spec or tells the Actor
    not to ask `plan_arms` (`_SPEC_ARM_RULE`); a spec fitted at one arm's keypoint whose arm is
    null is named, with the arm to set back."""
    found: list[Finding] = []
    for where, text in texts:
        if where.startswith("instruction:"):
            continue
        for match in dict.fromkeys(m.group(0) for m in _SPEC_ARM_RULE.finditer(text)):
            found.append(_warn("spec_arm_rule", where, match, _SPEC_ARM))
    for name, param in doc.params.items():
        if not isinstance(param, GraspSpecParam | PlaceSpecParam) or param.value.arm is not None:
            continue
        unset = any(c.field == "arm" and c.after is None for c in param.history)
        if unset and (fitted := _FITTED_AT.match(param.source)):
            message = (
                f"{name} was fitted at the {fitted[1]} arm's keypoint ({param.source}) but its "
                f"arm is null, so the other arm uses it as written, not as its mirror image: set "
                f"its arm back to {fitted[1]} (set_category); which arm acts stays the "
                "plan_arms' plan"
            )
            found.append(_warn("spec_arm_rule", f"params.{name}", name, message))
    return found


_FITTED_AT: Final = re.compile(rf"{FITTED_SOURCE} (left|right) K[1-9][0-9]*$")
"""A spec fitted at a demonstration keypoint (`demo <arm> K<n>`)."""


def _robot_frame(param: Param) -> bool:
    return isinstance(param, NumberParam) and param.frame == "robot_frame"


def prose(doc: MemoryDoc) -> Iterator[tuple[str, str]]:
    """The text written for models, by location: it may name parameters and counts as words."""
    for section in ("statement", "objects", "notes"):
        for number, line in enumerate(getattr(doc, section).splitlines(), 1):
            yield f"{section}:{number}", line
    for stage in doc.workflow:
        for field in STAGE_FIELDS:
            yield f"workflow.{stage.name}.{field}", getattr(stage, field)
    for lesson in doc.lessons:
        yield f"lessons.{lesson.id}.when", lesson.when
        yield f"lessons.{lesson.id}.do", lesson.do


def _labels(doc: MemoryDoc) -> Iterator[tuple[str, str]]:
    """Names and category labels: checked against the vocabularies only."""
    for stage in doc.workflow:
        yield f"workflow.{stage.name}", stage.name
    for name, param in doc.params.items():
        yield f"params.{name}", name
        if isinstance(param, CategoryParam):
            yield f"params.{name}.value", param.value
            yield f"params.{name}.choices", " ".join(param.choices or ())
        elif isinstance(param, GraspSpecParam):
            yield f"params.{name}.part", param.value.part
            yield f"params.{name}.x_toward", param.value.x_toward or ""
        elif isinstance(param, HingeSpecParam):
            yield f"params.{name}.part", param.value.part
        elif isinstance(param, DoneSpecParam):
            yield f"params.{name}.part", param.value.part or ""
        elif isinstance(param, PlaceSpecParam):
            target = param.value.target
            yield f"params.{name}.target", " ".join(filter(None, (target.obj, target.part)))
            yield f"params.{name}.orient", param.value.orient
            yield f"params.{name}.placement_point", param.value.placement_point


_METERS: Final = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _world_height(param: Param, config: LintConfig) -> float | None:
    """A length parameter with no frame at `absolute_height_m` or more, in meters, else None."""
    limit = config.absolute_height_m
    if limit is None or not isinstance(param, NumberParam) or param.frame is not None:
        return None
    scale = _METERS.get(param.unit.strip().lower())
    if scale is None:
        return None
    meters = abs(param.value) * scale
    return meters if meters >= limit else None


def _warn(rule: str, where: str, match: str, message: str) -> Finding:
    return Finding(rule=rule, where=where, match=_cut(match), message=message, level="warning")


_QUERY: Final = re.compile(r"""\bquery\s*=\s*(?:"([^"]*)"|'([^']*)'|([^,)]*))""")
_WORDS: Final = re.compile(r"[a-z]+")


def _appearance(
    doc: MemoryDoc,
    config: LintConfig,
    texts: list[tuple[str, str]],
    roles: list[Any],
) -> list[Finding]:
    """`appearance_name`: a role name, or a `query=` in the workflow or a lesson, that holds a
    word of `appearance_words` which the demonstration's INSTRUCTION also holds. What a role is
    for does not change with the demonstration's instance; the live instruction says which
    object plays it."""
    vocabulary = set(_WORDS.findall(doc.instruction.lower())) & {
        word.lower() for word in config.appearance_words
    }
    if not vocabulary:
        return []
    message = (
        "a role or a query named by how the demonstration's object looks: name it by what it is "
        "for (the live instruction says which object plays it), and write the query as the "
        "role's kind"
    )
    found: list[Finding] = []
    for role in roles:
        if vocabulary & set(role.role.lower().split("_")):
            found.append(_warn("appearance_name", f"objects:{role.number}", role.role, message))
    for where, text in texts:
        if where.startswith("statement:"):
            for word in sorted(vocabulary & set(_WORDS.findall(text.lower()))):
                found.append(_warn("appearance_name", where, word, _LOOK))
        if where.startswith(("workflow.", "lessons.")):
            for match in _QUERY.finditer(text):
                query = next(group for group in match.groups() if group is not None)
                if vocabulary & set(_WORDS.findall(query.lower())):
                    found.append(_warn("appearance_name", where, query.strip(), message))
    return found


def _foreign(doc: MemoryDoc, config: LintConfig, texts: list[tuple[str, str]]) -> list[Finding]:
    """`foreign_object`: OBJECTS or WORKFLOW naming a kind of object (`object_words`, a plural
    `s` or `es` allowed) that the demonstration's INSTRUCTION does not hold, not even inside a
    longer word."""
    kinds = {word.lower() for word in config.object_words}
    if not kinds:
        return []
    instruction = doc.instruction.lower()
    named = {k for k in kinds if k in instruction}  # inside a longer word too: kitchenpot
    message = (
        "names an object kind the demonstration's instruction does not: an object of another "
        "episode's instruction. Describe the role by what it is for (the live instruction says "
        "which object plays it), or name the demonstration's own object"
    )
    found: list[Finding] = []
    for where, text in texts:
        if not where.startswith(("statement:", "objects:", "workflow.")):
            continue  # NOTES and lessons may speak of kinds in general
        words = {_singular(w, kinds) for w in _WORDS.findall(text.lower())}
        for word in sorted((words & kinds) - named):
            found.append(_warn("foreign_object", where, word, message))
    return found


def _instances(config: LintConfig, roles: list[Any]) -> list[Finding]:
    """`instance_name`: role descriptions naming a kind of object some known instruction names
    and another does not."""
    kinds = {word.lower() for word in config.object_words}
    texts = [text.lower() for text in dict.fromkeys(config.instructions)]
    if len(texts) < 2 or not kinds:
        return []
    named = [{k for k in kinds if re.search(START + k + r"(?:e?s)?" + END, t)} for t in texts]
    varied = set.union(*named) - set.intersection(*named)
    message = (
        "names a kind of object that the task's instructions vary (one names it, another does "
        "not): the description names one episode's instance, and the identification would look "
        "for it in every episode. Describe the role by what it does and its general shape; the "
        "live instruction names the object"
    )
    found: list[Finding] = []
    for role in roles:
        words = {_singular(w, kinds) for w in _WORDS.findall(role.description.lower())}
        for word in sorted(words & varied):
            finding = Finding(
                rule="instance_name",
                where=f"objects:{role.number}",
                match=word,
                message=message,
                level=config.instance_level,
            )
            found.append(finding)
    return found


_ALIAS: Final = re.compile(r"\b(?:alias|aliases|same object|same as|duplicate)\b", re.IGNORECASE)


def _aliases(roles: list[Any]) -> list[Finding]:
    """`alias_role`: a role described as an alias of (the same object as) another role."""
    names = {role.role for role in roles}
    found: list[Finding] = []
    for role in roles:
        if _ALIAS.search(role.description) is None:
            continue
        others = [n for n in names - {role.role} if re.search(START + n + END, role.description)]
        for other in sorted(others):
            message = (
                f"calls {role.role} the same object as {other}: one object is one role. Merge "
                f"them: keep one name in OBJECTS, the workflow and the specs, and remove the other"
            )
            found.append(_warn("alias_role", f"objects:{role.number}", other, message))
    return found


def _undirected(doc: MemoryDoc, roles: list[Any]) -> list[Finding]:
    """`undirected_carry`: a role whose last placement in the workflow is held at a robot-frame
    target, with no `arm side:` line."""
    lines = {role.role: role for role in roles}
    last: dict[str, bool] = {}
    for stage in doc.workflow:
        for call in parse_calls(stage.call):
            if call.name != "place_object":
                continue
            role = call.keywords.get("obj", "").strip()
            name = call.keywords.get("place", "").strip().strip("{}")
            param = doc.params.get(name)
            held = call.keywords.get("is_open", "").strip().lower() in ("false", "0")
            robot = isinstance(param, PlaceSpecParam) and param.value.target.kind == "robot_frame"
            if role in lines:
                last[role] = held and robot
    message = (
        "is carried and held at a robot-frame target, and no arm side: line says which arm "
        "carries it: plan_arms then takes the nearer arm, and where the object's direction "
        "decides its end (a head that must point to one side) the other arm carries it the wrong "
        "way round. Under its role in OBJECTS (after its x toward: line), write arm side: toward "
        "its x, away from its x (the demonstration's arm relative to where its x pointed: "
        "plan_arms' side on the demonstration's scene), or either"
    )
    return [
        _warn("undirected_carry", f"objects:{lines[role].number}", role, message)
        for role, held in last.items()
        if held and lines[role].arm_side is None
    ]


def _relative_carries(doc: MemoryDoc, config: LintConfig, roles: list[Any]) -> list[Finding]:
    """`relative_carry`: a role of `LintConfig.carried_ends` with no articulated part that no
    placement in the workflow places (displacements alone carry it). One placed relative to
    another object is left out: its end follows that object, which a base target cannot tell."""
    lines = {role.role: role for role in roles}
    placed: set[str] = set()
    jointed: set[str] = set()
    for stage in doc.workflow:
        for call in parse_calls(stage.call):
            role = call.keywords.get("obj", "").strip() or (call.values[0] if call.values else "")
            if call.keywords.get("articulated", "").strip().lower() in ("true", "1"):
                jointed.add(role)
            if call.name != "place_object":
                continue
            role = call.keywords.get("obj", "").strip()
            if not role and len(call.values) > 1:
                role = call.values[1]  # place_object(arm, obj, ...)
            placed.add(role)
    message = (
        "ends, in the demonstration, held up where its arm carried it, over nothing that places "
        "it, and the workflow carries it by displacements alone: a carry written as a "
        "displacement from where it started lands wherever this scene's start leaves it, and "
        "relative done criteria (its spacing from another object, its gap over the table) pass "
        "it anywhere. If the demonstration's end is the same in the robot's frame "
        "whatever the start (check it against the tries), write the carry as a place_object with "
        "a PlaceSpec whose target is robot-frame: robot-frame constants are allowed in a "
        "PlaceSpec only, never in a GraspSpec"
    )
    return [
        _warn("relative_carry", f"objects:{lines[role].number}", role, message)
        for role in config.carried_ends
        if role in lines and role not in jointed and role not in placed
    ]


def _singular(word: str, kinds: set[str]) -> str:
    """`word` as a kind of `kinds` when it is one's plural, else as it is."""
    for end in ("es", "s"):
        if word.endswith(end) and word[: -len(end)] in kinds:
            return word[: -len(end)]
    return word


_FITTED_NUMBERS: Final = {
    "grasp_spec": ("offset_m", "offset_frac", "yaw_deg", "tilt_deg", "lean_deg"),
    "place_spec": ("offset_m", "yaw_deg", "rpy_deg"),
    "hinge_spec": ("offset_m",),
    "path_spec": (),
    "skill_spec": (),
}
_FITTED_CATEGORIES: Final = {
    "grasp_spec": ("part", "approach", "anchor", "x_toward", "closing"),
    "place_spec": ("frame", "orient", "placement_point", "mirror"),
    "hinge_spec": ("part", "side", "at"),
    "path_spec": (),
    "skill_spec": (),
}
_SPECS: Final = (GraspSpecParam, PlaceSpecParam, HingeSpecParam, PathSpecParam, SkillSpecParam)


def _spec_findings(name: str, param: Param, roles: set[str], config: LintConfig) -> list[Finding]:
    """`target_role`, `handwritten_spec` and `spec_drift` for one parameter."""
    where = f"params.{name}"
    found: list[Finding] = []
    if isinstance(param, PlaceSpecParam):
        target = param.value.target
        if target.kind == "object" and target.obj not in roles:
            message = "a PlaceSpec's target is an OBJECTS role: rename it or add the role"
            found.append(_warn("target_role", f"{where}.target", str(target.obj), message))
    if not isinstance(param, _SPECS):
        return found
    if not param.source.startswith(f"{FITTED_SOURCE} "):
        message = (
            "a spec with no fit on the demonstration: fit it from demo_relation (define_param "
            "with `fitted`) and let tries move it from there"
        )
        return [*found, _warn("handwritten_spec", where, name, message)]
    fit = next((change.after for change in param.history if change.how == "define"), None)
    if isinstance(fit, dict) and (moved := _drift(param, fit, config)):
        message = (
            "the spec moved far from its fit on the demonstration: a hand-tuned value is one "
            "scene's; refit it, or keep the fit and change the stage"
        )
        found.append(_warn("spec_drift", where, "; ".join(moved), message))
    return found


def _drift(param: Any, fit: dict[str, Any], config: LintConfig) -> list[str]:
    """How a fitted spec's numbers and categories differ from its fit, past the config's bounds."""
    now = param.value.model_dump(mode="json")
    moved: list[str] = []
    # An offset is relative to a frame: where the target or the axes changed, it says nothing.
    reframed = param.kind == "place_spec" and (
        any(fit.get("target", {}).get(k, v) != v for k, v in now["target"].items())
        or fit.get("frame", now["frame"]) != now["frame"]
    )
    for key in _FITTED_NUMBERS[param.kind]:
        if key not in fit or key not in now or (reframed and key == "offset_m"):
            continue
        before, after = fit[key], now[key]
        if key == "offset_m":
            shift = math.dist(before, after)
            if shift > config.spec_drift_m:
                moved.append(f"offset {shift * 100:.1f} cm")
        elif key == "offset_frac":
            shift = max(abs(a - b) for a, b in zip(before, after, strict=True))
            if shift > config.spec_drift_frac:
                moved.append(f"offset_frac {shift:.2f}")
        else:
            pairs = list(zip(before, after, strict=True)) if key == "rpy_deg" else [(before, after)]
            turn = max(abs((a - b + 180.0) % 360.0 - 180.0) for a, b in pairs)
            if turn > config.spec_drift_deg:
                moved.append(f"{key.removesuffix('_deg')} {turn:.0f} deg")
    for key in _FITTED_CATEGORIES[param.kind]:
        if key in fit and key in now and fit[key] != now[key]:
            moved.append(f"{key} {fit[key]} -> {now[key]}")
    if param.kind == "place_spec" and isinstance(fit.get("target"), dict):
        for key in ("kind", "part", "anchor", "landmark"):
            if key in fit["target"] and fit["target"][key] != now["target"][key]:
                moved.append(f"target {key} {fit['target'][key]} -> {now['target'][key]}")
    return moved


_CALL: Final = re.compile(r"(?<![\w.])([a-z_]+)\s*\(([^()]*)\)")
_ARGS: Final = {  # the arguments a predicate takes at least, thresholds included
    "holding": 1,
    "gripper_open": 2,
    "lifted": 2,
    "on": 4,
    "inside": 2,
    "beside": 5,
    "near": 3,
    "pressed": 2,
    "moved": 2,
    "yaw_aligned": 3,
}
_THRESHOLDS: Final = {  # the positions of a predicate's numeric arguments
    "gripper_open": (1,),
    "lifted": (1,),
    "on": (2, 3),
    "beside": (3, 4),
    "near": (2,),
    "pressed": (1,),
    "moved": (1,),
}
_NUMERIC_KEYS: Final = frozenset(
    {"fraction", "height", "tol", "min_overlap", "dmin", "dmax", "distance"}
)
_NUMERIC: Final = re.compile(r"^(?:[<>]=?\s*)?-?\d|^\{|^null$|^none$", re.IGNORECASE)
_RESULT: Final = frozenset({"lifted", "on", "inside", "beside", "pressed", "moved", "yaw_aligned"})
"""Predicates that state a result on an object (the others say where the arm and gripper are)."""
_PRESS: Final = re.compile(START + r"(?:press(?:ed|ing)?|push(?:ed|ing)?)" + END)


def _not_a_number(name: str, index: int, arg: str) -> bool:
    """Whether `arg` of predicate `name` stands where a threshold goes and is no number (a
    `{parameter}` counts): a keyword argument named like one, or the positional one at its place."""
    key, sep, value = arg.partition("=")
    if sep and not value.startswith("="):
        return key.strip() in _NUMERIC_KEYS and not _NUMERIC.match(value.strip())
    return index in _THRESHOLDS.get(name, ()) and not _NUMERIC.match(arg.strip())


def _done_lines(doc: MemoryDoc) -> list[Finding]:
    """`done_untestable`: a stage's done line is checked by `check`, so it holds predicate calls
    with their arguments; and some stage's states a result on an object."""
    found: list[Finding] = []
    for stage in doc.workflow:
        where = f"workflow.{stage.name}.done"
        known = [
            (m[1], [a for a in m[2].split(",") if a.strip()])
            for m in _CALL.finditer(stage.done)
            if m[1] in _ARGS
        ]
        why = ""
        if not stage.done:  # not written yet: stages are added one at a time
            continue
        if not known:
            why = "no predicate call: write holding, lifted, on, beside, ... with thresholds"
        elif short := [n for n, args in known if len(args) < _ARGS[n]]:
            why = f"{short[0]} is short of arguments: it takes at least {_ARGS[short[0]]}"
        elif words := [
            (n, a) for n, args in known for i, a in enumerate(args) if _not_a_number(n, i, a)
        ]:
            why = f"{words[0][0]}'s argument {words[0][1].strip()!r} is no number: a threshold"
        elif _PRESS.search(stage.done) and not any(n == "pressed" for n, _ in known):
            why = "names a press but measures none: use pressed(obj, distance)"
        if why:
            found.append(_warn("done_untestable", where, stage.done, why))
    stated = (m[1] for stage in doc.workflow for m in _CALL.finditer(stage.done))
    if any(stage.done for stage in doc.workflow) and not _RESULT & set(stated):
        message = (
            "no stage's done states a result on an object: add on, inside, beside, pressed, ..."
        )
        last = doc.workflow[-1]
        found.append(_warn("done_untestable", f"workflow.{last.name}.done", last.done, message))
    return found


_LOOK: Final = (
    "TASK names how the demonstration's object looks: say the task over the roles, as it holds "
    "for every episode (the live instruction says which object plays each role)"
)
_SIDE_ORDER: Final = "the robot's left to right"


def _sides(doc: MemoryDoc, roles: list[Any]) -> list[Finding]:
    """`side_bound` (an error): a side of the robot (left, right) in a done criterion that came
    from the arm that acted in the demonstration (`memory.sided.side_rule`: TASK does not name
    it, `a` does not go into or onto `b`, and `a` is held with `b` in the other arm or `b` is an
    anchor the workflow never moves), so that a scene where the other arm acts ends on the other
    side: an `offset_x` criterion away from zero by more than its band, without `arm`, or a done
    line's `beside(a, b, left|right, ...)`. `side_order` (a warning): a role ordered by the
    robot's left to right while TASK names no side: the order only tells look-alike objects
    apart, never which arm takes which. `no_statement` (a warning): an empty TASK."""
    found: list[Finding] = []
    if not doc.statement.strip():
        message = (
            "TASK is empty: say what the task does over the OBJECTS roles in one or two "
            "sentences, with no arm and no look (set_text section statement)"
        )
        found.append(_warn("no_statement", "statement", "", message))
    if task_sides(doc.statement):
        return found
    for item in sided(doc):
        if not item.rule.bind:
            continue
        arm = "" if item.rule.arm is None else f" ({item.rule.arm} in the demonstration)"
        if item.where.startswith("params."):
            message = (
                f"{item.match} holds {item.a} on one side of {item.b} in the robot's frame (x), "
                f"a side the demonstration's arm gave it ({item.rule.why}). Set the criterion's "
                f"arm to the arm that acts on {item.a}{arm}: its value then holds for that arm "
                "and mirrors for the other"
            )
        else:
            message = (
                f"beside holds {item.a} on a side of the robot the demonstration's arm gave it "
                f"({item.rule.why}); write {ARM_SIDE} (the side of the arm acting on "
                f"{item.a}{arm}) or {OTHER_SIDE} instead"
            )
        found.append(
            Finding(rule="side_bound", where=item.where, match=_cut(item.match), message=message)
        )
    for role in roles:
        if role.order is not None and role.order.key == _SIDE_ORDER:
            message = (
                f"{role.role} is ordered by the robot's left to right while TASK names no side: "
                "the order tells look-alike objects apart only; which arm takes which is "
                "decided on each scene"
            )
            found.append(_warn("side_order", f"objects:{role.number}", role.role, message))
    return found


_FRAME_SLACK_M: Final = 0.02
"""A horizontal offset this small in a target's frame is the fit's noise, whatever the frame."""
_OPEN_X: Final = re.compile(
    START + r"(?:any|either|equivalent|all sides|no distinguished|no named|no specific)" + END,
    re.IGNORECASE,
)
_NO_FRONT: Final = re.compile(
    r"no (?:intrinsic|distinguished|named)[^;,.]{0,30}(?:direction|side|front|frame)", re.IGNORECASE
)
_NO_PART: Final = frozenset({"", "whole", "null", "none", OTHER_ARM, *GEOMETRIC_ENDS})
_DEMO_NUMBER: Final = (
    "a quantity the demonstration showed is a parameter: define it (define_param) and write "
    "{name} here"
)
_LITERAL: Final = "a length or angle in a template is a parameter: define it and write {name}"
_UNKNOWN_PART: Final = (
    "a part of an OBJECTS role: describe it in the role (what it is, where), or register it "
    "with a template's register_part or locate_part"
)
_OPEN_FRAME: Final = (
    "the placement is in the target's frame, but the role leaves its x open: give the role an "
    "`x toward:` that fixes it, or write the placement in world_axes"
)


def _executable(
    doc: MemoryDoc, config: LintConfig, texts: list[tuple[str, str]], roles: list[Any]
) -> list[Finding]:
    """Whether the Memory can be carried out as written (docs/concepts/learning.md):
    `unknown_tool` (a call to a tool the Actor lacks, or with parameters it does not take; needs
    `config.tools`), `unparameterised` (a quantity the demonstration showed, or a length or angle,
    in prose or as a literal instead of a parameter), `undefined_frame` (a placement in a
    target's frame whose x the role leaves open), ranked by `config.executable_level`; and the
    advice `unbraced_param` (a template naming a parameter without its braces) and
    `unknown_part` (a part nothing registers or describes)."""
    level = config.executable_level
    found: list[Finding] = []
    templates = [(w, t) for w, t in texts if w.startswith("workflow.") and w.endswith(".call")]
    goals = [(w, t) for w, t in texts if w.startswith("workflow.") and w.endswith(".goal")]
    lessons = [(w, t) for w, t in texts if w.startswith("lessons.")]
    for where, text in [*templates, *lessons]:
        if config.tools:
            for match, why in unknown_calls(text, config.tools, PREDICATES):
                found.append(
                    Finding(rule="unknown_tool", where=where, match=match, message=why, level=level)
                )
            kinds = {name: param.kind for name, param in doc.params.items()}
            for match, why in spec_mismatches(text, config.tools, kinds, PREDICATES):
                found.append(
                    Finding(
                        rule="wrong_param_kind", where=where, match=match, message=why, level=level
                    )
                )
    for where, text in templates:
        for name in unbraced(text, doc.params):
            message = f"write the parameter with its braces: {{{name}}}"
            found.append(_warn("unbraced_param", where, name, message))
    for where, text in [*templates, *goals, *lessons]:
        named = unbraced(text, doc.params) or _REF.findall(text)
        for match in [] if named else demo_quantities(text):
            found.append(
                Finding(
                    rule="unparameterised",
                    where=where,
                    match=_cut(match),
                    message=_DEMO_NUMBER,
                    level=level,
                )
            )
        if not where.startswith("lessons."):
            for match in literal_numbers(text, calls=where.endswith(".call")):
                found.append(
                    Finding(
                        rule="unparameterised",
                        where=where,
                        match=_cut(match),
                        message=_LITERAL,
                        level=level,
                    )
                )
    found += _frames(doc, roles, level)
    found += _parts(doc, roles, templates, config.tools)
    if config.tools:
        found += _unregistered(doc, templates, config.tools, level)
    return found


def _open_x(role: Any) -> bool:
    """Whether a role's description leaves its frame's x open (any side will do)."""
    if role.toward is not None:
        return _OPEN_X.search(role.toward) is not None
    return _NO_FRONT.search(role.description) is not None


def _frames(doc: MemoryDoc, roles: list[Any], level: str) -> list[Finding]:
    """`undefined_frame`: a placement whose horizontal offset or turn is in the target's frame
    (`frame: target`) of a role whose x is open."""
    open_roles = {role.role for role in roles if _open_x(role)}
    found: list[Finding] = []
    for name, param in doc.params.items():
        if not isinstance(param, PlaceSpecParam):
            continue
        spec = param.value
        if (
            spec.frame != "target"
            or spec.target.kind != "object"
            or spec.target.obj not in open_roles
        ):
            continue
        turned = spec.orient == "yaw" and abs(spec.yaw_deg) > 1.0
        turned = turned or (spec.orient == "relative" and abs(spec.rpy_deg[2]) > 1.0)
        if turned or max(abs(spec.offset_m[0]), abs(spec.offset_m[1])) >= _FRAME_SLACK_M:
            found.append(
                Finding(
                    rule="undefined_frame",
                    where=f"params.{name}",
                    match=str(spec.target.obj),
                    message=_OPEN_FRAME,
                    level=level,
                )  # type: ignore[arg-type]
            )
    return found


def _unregistered(
    doc: MemoryDoc, templates: list[tuple[str, str]], tools: Mapping[str, ToolSig], level: str
) -> list[Finding]:
    """`unregistered_part`: a spec a template uses is of a part (not the whole object) that no
    template registers: an Actor registers a part itself (the episode's opening identifies whole
    objects only), so the grasp or placement would find none."""
    registered = {p for _, t in templates for p in part_names(parse_calls(t), tools)}
    used = {name for _, text in templates for name in template_refs(text, doc.params)}
    found: list[Finding] = []
    for name in sorted(used):
        param = doc.params[name]
        if isinstance(param, GraspSpecParam | HingeSpecParam):
            part = param.value.part
        elif isinstance(param, PlaceSpecParam):
            part = param.value.target.part or ""
        else:
            continue
        if part not in _NO_PART and part not in registered:
            message = (
                f"{name} is of the part {part}, which no call registers: register it before "
                f"it is used (register_part or locate_part with part={part})"
            )
            found.append(
                Finding(
                    rule="unregistered_part",
                    where=f"params.{name}.part",
                    match=part,
                    message=message,
                    level=level,  # type: ignore[arg-type]
                )
            )
    return found


def _parts(
    doc: MemoryDoc,
    roles: list[Any],
    templates: list[tuple[str, str]],
    tools: Mapping[str, ToolSig] | None = None,
) -> list[Finding]:
    """`unknown_part`: a part a spec or a call names that no template registers (`part=` of a
    call) and no role description or `x toward:` mentions."""
    known = {part for _, text in templates for part in part_names(parse_calls(text), tools)}
    words = set(
        _WORDS.findall(" ".join(f"{r.description} {r.toward or ''}" for r in roles).lower())
    )
    used: list[tuple[str, str]] = []
    for name, param in doc.params.items():
        value = param.value
        if isinstance(param, GraspSpecParam):
            used += [
                (f"params.{name}.part", value.part),
                (f"params.{name}.x_toward", value.x_toward or ""),
            ]
        elif isinstance(param, HingeSpecParam):
            used.append((f"params.{name}.part", value.part))
        elif isinstance(param, PlaceSpecParam):
            used += [
                (f"params.{name}.target", value.target.part or ""),
                (f"params.{name}.x_toward", value.x_toward or ""),
            ]
            if value.placement_point.startswith("part:"):
                used.append(
                    (f"params.{name}.placement_point", value.placement_point.removeprefix("part:"))
                )
    for where, text in templates:
        for call in parse_calls(text):
            used += [
                (where, call.keywords[key].strip("\"'"))
                for key in ("part", "x_toward")
                if key in call.keywords
            ]
    found: list[Finding] = []
    for where, part in used:
        part = part.strip()
        if part in _NO_PART or part in known or part.startswith(("{", "<")):
            continue
        if all(token in words for token in part.lower().split("_")):
            continue
        found.append(_warn("unknown_part", where, part, _UNKNOWN_PART))
    return found


def _regex(words: list[str], patterns: list[str]) -> re.Pattern[str] | None:
    if not words and not patterns:
        return None
    alternatives = [START + re.escape(word) + END for word in words]
    alternatives += [f"(?:{pattern})" for pattern in patterns]
    return re.compile("|".join(alternatives), re.IGNORECASE)


def _cut(text: str) -> str:
    return text if len(text) <= _MATCH_CHARS else text[:_MATCH_CHARS] + "..."
