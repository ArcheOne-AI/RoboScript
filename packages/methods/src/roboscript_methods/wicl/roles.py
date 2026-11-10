"""The roles' prompts and briefs (docs/concepts/learning.md): each role's system prompt from the
recipe's templates (`<recipe>/prompts/`), and the brief, the first user message it reads: the
demonstration package for the Demo Reader and the hinted Actor, the material of a failure for the
Reviser. The Actor reads the Memory without provenance (`view(detail=False)`); the Demo Reader
and the Reviser read it with provenance, parameter limits and SCRATCH.
"""

import copy
import hashlib
import itertools
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast, get_args

import numpy as np
import yaml
from pydantic import ValidationError

from roboscript.agent import placeholders, read_lines, render_prompt
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.events import EventKind
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.receipt import ActionReceipt
from roboscript.contracts.tools import ContentPart, ImagePart, TextPart
from roboscript.demos import Keypoint, frame_alias
from roboscript.embodiments import EmbodimentProfile
from roboscript.envs.cards import TaskCard, task_text
from roboscript.learning.context import TaskContext
from roboscript.learning.demo import DemoPackage, DemoTruth, goal_frames
from roboscript.learning.episode import (
    EVENTS,
    EXPLAINED,
    OBJECTS,
    PHYSICS,
    REFEREE,
    EpisodeOutcome,
)
from roboscript.learning.opening import load
from roboscript.manipulation.specs import GraspSpec
from roboscript.memory import (
    Editor,
    EditRecord,
    Finding,
    MemoryDiff,
    MemoryDoc,
    ToolSig,
    render_diff,
    view,
)
from roboscript.perception.criteria import UNITS, DoneCheckConfig, DoneSpec, floored
from roboscript.records import read_events
from roboscript.skills import render_guides
from roboscript.spatial.rotation import quat_angle_deg
from roboscript.tools import demo_alias
from roboscript.toolsets.memory import CLAMPED, NO_TRUST
from roboscript_methods.wicl.diagnosis import MOVED_M, Explained, Fate, Named, fell
from roboscript_methods.wicl.extension import RELATIONS, UNCONFIRMED
from roboscript_methods.wicl.ledger import failure_labels
from roboscript_methods.wicl.memory.faces import (
    BOX,
    FACE_TILT_DEG,
    Grasped,
    face_fit,
    face_lines,
    find_box_faces,
)
from roboscript_methods.wicl.perception.done import capped, verdict

DemoContext = Literal["none", "goal", "ops", "full"]
"""How much of the demonstration an S2 or P3 Actor reads (`demo_context_brief`, `demo_seed`): none
(the clean one-shot, the core's `agent.host.NO_DEMO`), its goal's views, its operations, or those
and its recording as a frame source."""
LEVELS: Final = get_args(DemoContext)

TEMPLATES: Final = ("actor.md", "demo_reader.md", "reviser.md", "criteria.md", "cleanup.md")
GUIDES: Final = "conventions.md"
"""The name the mounted skillsets' guides enter the prompts' digest under. It is part of
what the digest hashes, so it stays fixed."""
CONVENTIONS: Final = "conventions"
"""The placeholder the guides fill in a template."""
STAGE_RULES: Final = "stage_rules.yaml"
LIBRARY: Final = "library.md"
"""What a P3 Actor in library mode reads where its Task Memory would go (`Prompts.library_actor`):
outside `TEMPLATES`, so the digest of every bound prompt does not depend on it."""
_CALL_ARGS: Final = 200  # characters of a call's arguments in a log
_CALL_OUTPUT: Final = 400  # of its output
_TEXT: Final = 300  # of the model's text and an observation's
_ENDING: Final = 2000  # of the arguments and the output of the call that ended a try
_CALL_ID: Final = re.compile(r"\((call_[A-Za-z0-9_-]+)\)")
"""A call id in an end's note (`fail_fast` names the call that ended the try)."""
_REJECTED: Final = 20  # rejected edits shown, the latest
_FRAME: Final = re.compile(r"(?<![\w:.])frame_[0-9]+(?![\w])")
_CUES: Final = {
    "close": "the gripper closes",
    "open": "the gripper opens",
    "stop": "the arm comes to rest",
    "valley": "the arm slows down",
    "corner": "the path turns",
}


class Prompts:
    """A recipe's role templates, with their digest (the prompt version in `config.json`), and
    the guides of the skillsets the stack mounts (`guides`, `SkillSet.guide`), which every role
    reads where its template has `{{conventions}}`. The guides are part of the digest, joined a
    blank line apart under `GUIDES`. ValueError when a template has `{{conventions}}` and there
    is no guide: the roles would read nothing where the robot's conventions belong. `card`: the
    task's success condition card, which every role reads under the instruction (`with_card`;
    `envs.cards.task_text`); not part of the digest, which is the templates' version: a record
    names the card it showed (`task.card`, `config.json`)."""

    card: TaskCard | None = None

    def __init__(
        self,
        directory: Path,
        tools: Mapping[str, ToolSig] | None = None,
        guides: Sequence[str] = (),
    ) -> None:
        self.directory = directory
        self.tool_list = tool_listing(tools or {})
        self.guides = tuple(guides)
        self._templates = {name: (directory / name).read_text("utf-8") for name in TEMPLATES}
        needing = [n for n, text in self._templates.items() if CONVENTIONS in placeholders(text)]
        if needing and not self.guides:
            raise ValueError(
                f"{', '.join(needing)} in {directory} read {{{{{CONVENTIONS}}}}} but no mounted "
                "skillset has a guide (SkillSet.guide): mount the env's skillsets, or give the "
                "runner the guides of its env's (ProtocolRunner(guides=...))"
            )
        rules = (directory / STAGE_RULES).read_text("utf-8")
        self.stage_rules: dict[str, str] = yaml.safe_load(rules)
        parts = [*self._templates.items(), (STAGE_RULES, rules), ("tools", self.tool_list)]
        if self.guides:
            parts.append((GUIDES, "\n\n".join(self.guides)))
        digest = hashlib.sha256()
        for name, text in sorted(parts):
            digest.update(f"{name}\0{text}\0".encode())
        self.digest = digest.hexdigest()

    def with_card(self, card: TaskCard | None) -> "Prompts":
        """These prompts for a task whose card is `card` (None: these, unchanged)."""
        if card is None:
            return self
        prompts = copy.copy(self)
        prompts.card = card
        return prompts

    def task(self, instruction: str) -> str:
        """The `{{instruction}}` value: the instruction with the card under it, if any."""
        return task_text(instruction, self.card)

    def conventions(self, profile: EmbodimentProfile) -> str:
        """The mounted skillsets' guides for `profile`'s robot (`skills.render_guides`: its world
        axes and TCP offset)."""
        return render_guides(self.guides, profile)

    def actor(
        self, *, instruction: str, memory: MemoryDoc, rules: str, profile: EmbodimentProfile
    ) -> str:
        """`rules`: the stage's key in `stage_rules.yaml` (`s1_hinted`, `s1`, `s2`, `p3`)."""
        values = {
            "robot": profile.description,
            "instruction": self.task(instruction),
            "memory": view(memory, detail=False).strip(),
            "stage_rules": self.stage_rules[rules].strip(),
            "conventions": self.conventions(profile),
        }
        return render_prompt(self._templates["actor.md"], values)

    def library_actor(self, *, instruction: str, rules: str, profile: EmbodimentProfile) -> str:
        """The Actor's prompt in P3's library mode: the Actor's template with the recipe's
        `library.md` where the Memory goes (read only here, so the prompts' digest and every
        bound prompt do not depend on it)."""
        values = {
            "robot": profile.description,
            "instruction": instruction,
            "memory": self._library_template()[0].strip(),
            "stage_rules": self.stage_rules[rules].strip(),
            "conventions": self.conventions(profile),
        }
        return render_prompt(self._templates["actor.md"], values)

    @property
    def library_digest(self) -> str:
        return self._library_template()[1]

    def _library_template(self) -> tuple[str, str]:
        text = (self.directory / LIBRARY).read_text("utf-8")
        return text, hashlib.sha256(text.encode()).hexdigest()

    def demo_reader(
        self, *, instruction: str, memory: MemoryDoc, profile: EmbodimentProfile
    ) -> str:
        values = {
            "instruction": self.task(instruction),
            "memory": view(memory).strip(),
            "tools": self.tool_list,
            "conventions": self.conventions(profile),
        }
        return render_prompt(self._templates["demo_reader.md"], values)

    def reviser(self, *, instruction: str, editor: Editor, profile: EmbodimentProfile) -> str:
        values = {
            "instruction": self.task(instruction),
            "memory": view(editor.doc).strip(),
            "budget": budget_text(editor),
            "tools": self.tool_list,
            "conventions": self.conventions(profile),
        }
        return render_prompt(self._templates["reviser.md"], values)

    def criteria(self, *, instruction: str, editor: Editor, profile: EmbodimentProfile) -> str:
        """The Reviser that writes the done criteria at S1's exit."""
        values = {
            "instruction": self.task(instruction),
            "memory": view(editor.doc).strip(),
            "budget": budget_text(editor),
            "conventions": self.conventions(profile),
        }
        return render_prompt(self._templates["criteria.md"], values)

    def cleanup(self, *, instruction: str, editor: Editor, profile: EmbodimentProfile) -> str:
        """The Reviser that cleans a Memory up before S1 runs again from it (`cleanup`)."""
        values = {
            "instruction": self.task(instruction),
            "memory": view(editor.doc).strip(),
            "budget": budget_text(editor),
            "tools": self.tool_list,
            "conventions": self.conventions(profile),
        }
        return render_prompt(self._templates["cleanup.md"], values)


def role_prompts(ctx: TaskContext) -> Prompts:
    """The task's prompts as wicl's roles read them (its runtime's, `Wicl.runtime`), with
    the task's success condition card (`TaskContext.card`)."""
    if not isinstance(ctx.prompts, Prompts):
        raise TypeError(f"wicl's protocols run with its prompts, not {type(ctx.prompts)}")
    return ctx.prompts.with_card(ctx.card)


def tool_listing(tools: Mapping[str, ToolSig]) -> str:
    """The tools a Memory's call templates may name, generated from the mounted tools'
    signatures: one line each (name, keywords, purpose); empty without tools."""
    if not tools:
        return "(no tool list)"
    lines = [
        "A stage's `call` names only these tools, with only these keywords (required ones plain, "
        "optional ones in brackets; the Actor adds each action's `rationale` itself). The linter "
        "refuses a call to any other name or keyword. Whole objects are not located by a call: "
        "every episode identifies the roles of OBJECTS in the live scene at its start, from each "
        "role's line and the episode's instruction, so a call to `locate`, `identify` or "
        "`register` for an object has no place in a template; what the object is and how to "
        "recognise it in general words (its kind, shape, size relative to the others, what it "
        "does in the task) belongs in its OBJECTS line. Parts are registered by `register_part` "
        "where the list has it.",
        *(sig.line(name) for name, sig in sorted(tools.items())),
    ]
    return "\n".join(lines)


def budget_text(editor: Editor) -> str:
    if editor.budget is None:
        text = "No budget applies: edit the Memory as needed; the linter checks every write."
        return text if editor.rules.trust_radius else f"{text}\n{NO_TRUST}"
    costs = ", ".join(f"{op} {cost}" for op, cost in editor.rules.costs)
    lines = [
        f"{editor.left} of {editor.budget} budget units are left for this seed. Each accepted "
        f"edit costs: {costs}. Rejected edits cost nothing."
    ]
    if editor.numbers_left is not None:
        within = ", each within its trust radius" if editor.rules.trust_radius else ""
        lines.append(
            f"This revision may change {editor.numbers_left} more numeric targets{within}."
        )
        if editor.rules.trust_radius and editor.rules.clamp_to_trust:
            lines.append(CLAMPED)
    return "\n".join(lines)


def demo_brief(
    package: DemoPackage,
    end: Mapping[str, BlobRef] | None = None,
    contacts: Mapping[str, Keypoint] | None = None,
) -> list[ContentPart]:
    """The demonstration as a model reads it: the instruction, each arm's keypoints in order
    (phase, cue, gripper and the frame it is seen in; never coordinates or tolerances), each
    arm's contact past its last keypoint (`demos.relation.contacts`: where `demo_relation` fits
    the held object's placement), the frames loaded, the keyframes, and its final state by
    camera (`end`, `goal.end_views`)."""
    lines = [
        "# The demonstration",
        "An expert did this task once on the scene you start from.",
        f"Instruction: {package.instruction}",
    ]
    aliases = {entry.alias for entry in package.frames}
    for arm, keypoints in package.keypoints.arms.items():
        lines.append(f"{arm.capitalize()} arm keypoints, in the order the expert passed them:")
        for kp in keypoints:
            seen = f", {frame_alias(kp.frame)}" if frame_alias(kp.frame) in aliases else ""
            lines.append(f"- {_keypoint(kp)}{seen}")
        if (contact := (contacts or {}).get(arm)) is not None:
            lines.append(
                f"- {contact.name}: the contact the held object makes at the success frame "
                f"({frame_alias(contact.frame)}), no gripper event marks it (a held tool's "
                f"strike, an object pushed home): demo_relation at {arm} {contact.name} with "
                "the object it meets gives the held object's placement there"
            )
    if package.frames:
        shown = ", ".join(entry.alias for entry in package.frames)
        lines.append(
            f"The recording's frames {shown} are loaded, numbered from its start (frame_1): "
            "look at them with the sensing and perception tools like at your own."
        )
    parts: list[ContentPart] = [TextPart(text="\n".join(lines))]
    for keyframe in package.keyframes:
        parts.append(TextPart(text=f"Keyframe: {keyframe.caption}"))
        parts += [ImagePart(blob=blob, label=keyframe.caption) for blob in keyframe.images]
    return parts + end_brief(package, end or {})


def end_brief(package: DemoPackage, end: Mapping[str, BlobRef]) -> list[ContentPart]:
    """The demonstration's final state (its last frame, the success frame its recording stops
    at) in each camera: what the scene looks like when the task is done."""
    if not end:
        return []
    alias = frame_alias(package.keypoints.success_frame or 0)
    text = (
        f"The demonstration's final state ({alias}, after its last operation: what done looks "
        "like), camera by camera:"
    )
    return [
        TextPart(text=text),
        *(ImagePart(blob=blob, label=f"final state, {name}") for name, blob in end.items()),
    ]


def demo_context_brief(package: DemoPackage, level: DemoContext) -> list[ContentPart]:
    """What an S2 or P3 Actor reads of the demonstration at `level`, in its brief: the goal's
    views (`goal`); each arm's keypoints (phase, cue, gripper; never coordinates or tolerances)
    and each operation's before and after views (`ops`); with `full`, the recording's frames
    loaded as `demo:<n>` (`demo_seed`) and named in the keypoints."""
    if level == "none":
        return []
    lines = [
        "# The demonstration",
        "An expert did this task once, on another scene of it: its objects may be placed "
        "differently from yours.",
        f"Its instruction: {package.instruction}",
    ]
    parts: list[ContentPart] = []
    if level == "goal":
        goal = goal_frames(package)
        if goal is None:
            raise ValueError(f"the package of {package.task} has no goal: rebuild it")
        lines.append("Below is how its scene looked when the task was done.")
        parts.append(TextPart(text="\n".join(lines)))
        parts += _keyframe(goal.caption, goal.images)
        return parts
    if not package.ops:
        raise ValueError(f"the package of {package.task} has no operations: rebuild it")
    aliases = {entry.alias: demo_alias(entry.alias) for entry in package.frames}
    for arm, keypoints in package.keypoints.arms.items():
        lines.append(f"{arm.capitalize()} arm keypoints, in the order the expert passed them:")
        for kp in keypoints:
            seen = aliases.get(frame_alias(kp.frame)) if level == "full" else None
            lines.append(f"- {_keypoint(kp)}" + (f", {seen}" if seen else ""))
    if level == "full":
        shown = ", ".join(aliases.values())
        lines.append(
            f"Its recording's frames {shown} are loaded, numbered from its start (demo:1): look "
            "at them with the sensing and perception tools like at your own frames; they are "
            "not your scene."
        )
    lines.append("Below, each operation just before it and after it.")
    parts.append(TextPart(text="\n".join(lines)))
    for op in package.ops:
        for keyframe in (op.before, op.after):
            parts += _keyframe(keyframe.caption, keyframe.images)
    return parts


def demo_seed(package: DemoPackage, level: DemoContext) -> EpisodeSeed | None:
    """The frames an S2 or P3 Actor's episode starts with at `level`: the recording's, aliased
    `demo:<n>` (only `full`)."""
    if level != "full":
        return None
    frames = [e.model_copy(update={"alias": demo_alias(e.alias)}) for e in package.frames]
    return EpisodeSeed(frames=frames)


class Briefs:
    """wicl's demonstration views (`MethodRuntime.demo`): what an S2 or P3 Actor is shown
    (docs/concepts/learning.md) is nothing (a clean one-shot, the default), its goal, its
    operations as before/after pairs with the keypoint table, or those and its recording as a
    frame source; never its ground-truth layer. Its levels are `DemoContext`'s, which the core
    checks a host's `demo_context` against (`learning.runtime.check_level`)."""

    levels: Collection[str] = LEVELS

    def brief(self, package: DemoPackage, level: str) -> list[ContentPart]:
        return demo_context_brief(package, cast(DemoContext, level))

    def seed(self, package: DemoPackage, level: str) -> EpisodeSeed | None:
        return demo_seed(package, cast(DemoContext, level))


def _keyframe(caption: str, images: Sequence[Any]) -> list[ContentPart]:
    # a keyframe's images are its head view and then its wrist view (`Keyframe`)
    views = ("head view", "wrist view") if len(images) == 2 else ("view",) * len(images)
    shown = [
        ImagePart(blob=blob, label=f"{caption}, {view}")
        for blob, view in zip(images, views, strict=True)
    ]
    return [TextPart(text=caption), *shown]


def truth_brief(truth: DemoTruth) -> list[ContentPart]:
    """A demonstration's ground-truth layer as a role reads it: guidance, not Memory content. Its
    moments are named by keypoint, never by frame: a Reviser holds only its tries' frames."""
    lines = [
        "# The demonstration's ground truth (the simulator's)",
        "For understanding the demonstration only: never write these names or numbers into the "
        "Memory, which must hold only what the robot perceives.",
        "Objects, by the simulator's names (and categories): "
        + ", ".join(f"{name} ({kind})" if kind else name for name, kind in truth.objects.items())
        + ".",
        "Poses are x, y, z in meters and the orientation quaternion w, x, y, z, world frame.",
    ]
    for frame in truth.frames:
        moments = (f"the {what}" if what in ("start", "success") else what for what in frame.what)
        lines.append(f"At {' and '.join(moments)}:")
        for name, pose in frame.poses.items():
            lines.append(f"- {name}: {_xyz(pose.p)}, q {_xyz(pose.q)}")
        lines += [f"- {name} joints: {_xyz(q)}" for name, q in frame.joints.items()]
    return [TextPart(text="\n".join(lines))]


Key = tuple[str, str, str | None, str | None]
"""A relation's row: metric, a, b, part."""


def row_key(value: Mapping[str, Any] | DoneSpec) -> Key:
    if isinstance(value, DoneSpec):
        return (value.metric, value.a, value.b, value.part)
    return (value["metric"], value["a"], value.get("b"), value.get("part"))


def held_at_start(
    criteria: Sequence[DoneSpec], scenes: Sequence[tuple[str, Mapping[str, Any]]]
) -> list[str]:
    """The scenes (a name, the relations read in it with their `start`) where every criterion
    already holds at the start: criteria that would pass before the task is done."""
    found = []
    for name, relations in scenes:
        starts = relations.get("start") or []
        if not criteria or not starts:
            continue
        at = {
            row_key(value): start["measured"]
            for value, start in zip(relations["values"], starts, strict=True)
        }
        if all(verdict(floored(spec), at.get(row_key(spec))) for spec in criteria):
            found.append(name)
    return found


def relations_brief(
    ends: Sequence[tuple[str, Mapping[str, Any]]], target: Mapping[str, Any] | None = None
) -> list[ContentPart]:
    """The criteria Reviser's material (with the demonstration's final state, `end_brief`): each
    relation perception measured at the tries' ends (`relations.json`: the official outcome, the
    values, and each at the try's start), the successes' range and start values and the failed
    values. With `target` (`relations_demo.json`: the same metrics at the demonstration's final
    state, its last keyframe) a row's criterion is the demonstration's end value; the tries'
    successes are where their success check first held (anywhere the check accepts: the object
    just inside, still held), so they only widen the tolerance (half their spread plus the
    metric's perception margin, `UNITS`), never set the value. Rows the demonstration changed
    from its start come first; then rows it ended where it started but a success did not (the
    end state requires them: released, withdrawn); rows that held throughout follow apart (they
    hold before the task is done too). Without a target (or where it is undecided) the
    suggestion is the successes' midpoint, rows the successes changed first. Within a group, the
    most rejecting and least spread first. Rows no try and not the demonstration measured are
    left out (a press on an object nothing pressed). With no ends (no try's end was read) the
    demonstration alone defines the rows and the tolerance is the metric's margin."""
    rows: dict[Key, tuple[list[Any], list[Any], list[Any]]] = {}
    for _, end in ends:
        starts = end.get("start") or [None] * len(end["values"])
        for value, begun in zip(end["values"], starts, strict=True):
            key = row_key(value)
            won, lost, first = rows.setdefault(key, ([], [], []))
            (won if end["success"] else lost).append(value["measured"])
            if end["success"]:
                first.append(None if begun is None else begun["measured"])
    aimed: dict[Key, tuple[float | None, float | None]] = {}
    if target is not None:
        starts = target.get("start") or [None] * len(target["values"])
        for value, begun in zip(target["values"], starts, strict=True):
            aimed[row_key(value)] = (
                value["measured"],
                None if begun is None else begun["measured"],
            )
    for key in aimed:  # the demonstration alone may define the rows (no try read its end)
        rows.setdefault(key, ([], [], []))
    successes = sum(bool(end["success"]) for _, end in ends)
    lines = ["# The ends of the S1 tries, as perception measured them"]
    if ends:
        lines.append(
            f"{successes} tries succeeded and {len(ends) - successes} failed: "
            + ", ".join(f"{n} {'succeeded' if end['success'] else 'failed'}" for n, end in ends)
            + "."
        )
    else:
        lines.append("No try's end was read: the demonstration alone defines the rows.")
    if target is not None:
        lines.append(
            "The demonstration's last frame (its goal keyframe) is the state done looks like: "
            "each row's criterion is its value there; the successes, where there are any, only "
            "show how far a scene varies (the tolerance), never the value."
        )
    lines.append(
        "Each row: metric a[.part] [b] (unit): "
        + ("the demonstration's start and end values; " if target is not None else "")
        + "the successes' start and end values; the failed ends' values; the suggested "
        "criterion (value ± tol) and how many failed ends it rejects (an undecided measurement "
        "is rejected). Offsets are a's centre minus b's along the robot's x and y; gap_z is a's "
        "bottom above b's top (above the table for b table); held is 1 while an arm holds it; "
        "press is how far, seen from above, from a's centre (its part's) an empty gripper last "
        "pushed on a and was stopped by it (undecided until something pressed it)."
    )
    groups: dict[str, list[tuple[int, float, str]]] = {
        "changed": [], "required": [], "unchanged": [], "fallback": [],
    }  # fmt: skip
    undecided = []
    for key, (won, lost, first) in rows.items():
        metric, a, b, part = key
        unit, (low, high), margin = UNITS[metric]
        name = " ".join(x for x in (metric, a if part is None else f"{a}.{part}", b) if x)
        shown = ", ".join("undecided" if v is None else f"{v:.3f}" for v in lost) or "none"
        measured = [v for v in won if v is not None]
        demo_end, demo_start = aimed.get(key, (None, None))
        if not measured and demo_end is None and all(v is None for v in lost):
            continue  # nothing ever measured it: no information
        if (not won or len(measured) < len(won)) and demo_end is None:
            undecided.append(f"- {name}: undecided in {len(won) - len(measured)} successes")
            continue
        top, bottom = (max(measured), min(measured)) if measured else (demo_end, demo_end)
        assert top is not None and bottom is not None
        known = [(s, e) for s, e in zip(first, won, strict=True) if s is not None and e is not None]
        began = "start " + ", ".join(f"{s:.3f}" for s, _ in known) + " -> " if known else ""
        tries = (
            f"successes {began}end {bottom:.3f}..{top:.3f}"
            if measured
            else "successes undecided"
            if won
            else "no success read"
        )
        if demo_end is not None:
            value = min(max(demo_end, low), high)
            tol = min((top - bottom) / 2 + margin, high - low)
            demo = f"demonstration {'?' if demo_start is None else f'{demo_start:.3f}'} -> "
            demo += f"{demo_end:.3f}; "
        else:
            value = min(max((top + bottom) / 2, low), high)
            tol = min((top - bottom) / 2 + margin, high - low)
            demo = "demonstration undecided; " if target is not None else ""
        rejects = sum(v is None or abs(v - value) > tol for v in lost)
        line = (
            f"- {name} ({unit}): {demo}{tries}; failed {shown}; "
            f"suggests {value:.3f} ± {tol:.3f}, rejects {rejects} of {len(lost)}"
        )
        entry = (-rejects, top - bottom, line)
        if demo_end is not None:
            moved = demo_start is None or abs(demo_end - demo_start) > margin
            apart = any(abs(e - demo_end) > margin for e in measured)
            group = "changed" if moved else "required" if apart else "unchanged"
        elif target is not None:
            group = "fallback"
        else:
            still = bool(known) and all(abs(e - s) <= margin for s, e in known)
            group = "unchanged" if still else "changed"
        groups[group].append(entry)
    who = "demonstration" if target is not None else "successes"
    lines += [f"Relations the {who} changed from their start:"]
    lines += [line for *_, line in sorted(groups["changed"])] or ["- none"]
    if groups["required"]:
        lines += [
            "Relations the demonstration ended where it started but a success did not (the end "
            "state requires them):",
            *(line for *_, line in sorted(groups["required"])),
        ]
    if groups["unchanged"]:
        lines += [
            f"Relations every {'end' if target is not None else 'success'} left where it "
            "started (they hold before the task is done as well, so they do not show it done):",
            *(line for *_, line in sorted(groups["unchanged"])),
        ]
    if groups["fallback"]:
        lines += [
            "Relations perception could not measure at the demonstration's end (from the "
            "successes only; use them only where nothing above shows the same):",
            *(line for *_, line in sorted(groups["fallback"])),
        ]
    if undecided:
        lines += ["Rows perception could not measure at every success (not usable):", *undecided]
    return [TextPart(text="\n".join(lines))]


def _keypoint(kp: Keypoint) -> str:
    """A keypoint's name, phase, cue and gripper; its tolerances stay with the referee (the
    orientation one follows the task's category, which follows what its success check weighs)."""
    return f"{kp.name} {kp.phase}: {_CUES[kp.cue]}, gripper {kp.gripper} after it"


@dataclass(frozen=True)
class TryLog:
    """What the Reviser reads about one try: which Memory it read (`label`), how it ended, the
    identification it started from (`identified`: its text as the Actor was shown it), its turns
    (`turns.jsonl`), the registered objects as it ended (`objects`), with its frame aliases
    renamed (`aliases`: to the `log:<try>:<turn>` ones its episode is seeded with), and, when
    the Reviser may see them, the ground-truth diagnosis and (S1) the keypoint report and the
    comparison with the expert (`expert`); the grasp call that ended it, if one did (`ending`); in
    S2, the done criteria it ran under and the relations perception read at its end
    (`relations.json`), which its text weighs against each other."""

    attempt: str
    label: str
    outcome: EpisodeOutcome
    turns: list[dict[str, Any]]
    aliases: Mapping[str, str] = field(default_factory=dict)
    diagnosis: Explained | None = None
    keypoints: Sequence[str] = ()
    identified: Sequence[str] = ()
    objects: Sequence[str] = ()
    criteria: Sequence[DoneSpec] = ()
    relations: Mapping[str, Any] | None = None
    unmoved: Sequence[str] = ()
    repeats: Sequence[str] = ()
    unchanged: Sequence[str] = ()
    expert: Sequence[str] = ()
    ending: "EndingGrasp | None" = None
    closes: Sequence[str] = ()
    """Ground truth (S1): grasps whose close turned the object they closed on (`close_turns`)."""
    grasped: Mapping[str, Grasped] = field(default_factory=dict)
    """The objects it saw as perception fitted them (`grasped_objects`)."""


def reviser_brief(
    tries: Sequence[TryLog],
    *,
    editor: Editor,
    diff: MemoryDiff | None,
    rejected: Sequence[EditRecord],
    images: int,
    misses: Sequence[str] = (),
    history: Sequence[str] = (),
) -> list[ContentPart]:
    """The Reviser's material: the tries of this seed (or attempt), the Memory's changes so far
    (`diff`; None when the Reviser edits the Memory itself), the budget, the task's rejected
    edits, and the latest try's last observed images; `misses` (`criteria_misses`): earlier
    successes the done criteria would have refused; `history` (S1, `imitate.revision_ledger`):
    the seed's earlier attempts and the revisions after them. The tries' frames are loaded under
    the aliases their logs name."""
    lines = []
    if history:
        lines += [
            "# Earlier tries on this seed and the revisions after them (oldest first)",
            "Each revision took effect from the next try on. Before changing a value back or "
            "making a change again, read this: a value a later try passed with, or a change "
            "already made and followed by the same failure, is not the cause unless the latest "
            "try shows new evidence; name that evidence in the edit's why.",
            *history,
            "",
        ]
    lines += ["# What happened"]
    for log in tries:
        lines += ["", *try_text(log)]
    faces = find_box_faces(editor)  # None: the rule is off, nothing to tell
    grasped = {} if faces is None else faces.grasped
    if off := face_lines(editor.doc, grasped, refit=editor.refits):
        lines += [
            "",
            "# Grasps off a box's faces (a rule they break)",
            "A grasp on a box closes along an axis within "
            f"{FACE_TILT_DEG:g} deg of one of its face normals, its point near the middle along "
            "that axis: fingers closing otherwise meet an edge or one side first and turn the box "
            "as they close. When a box turned in the grasp, the fix is the grasp's axes, never a "
            "compensation elsewhere. These Memory grasps break it now:",
            *off,
        ]
    if faces is not None and faces.required:
        lines += [
            "",
            "# Required first: the grasp that tipped a box",
            "This try's ground truth shows a close that turned a box while its grasp sat off the "
            "box's faces, where the demonstration's grasp did not. Until it is fixed, every other "
            "edit to a grasp (its spec, or a number its grasp call takes) is refused. Send first:",
            *(f"- {fix}" for fix in faces.required.values()),
        ]
    if misses and editor.rules.strict_done:
        lines += [
            "",
            "# Earlier successes the done criteria would have refused",
            "The official check confirmed these tries although a criterion failed at their ends. "
            "That cost nothing: the official check ends a try the moment it latches, so a strict "
            "criterion never refuses a success. Done criteria are only tightened, never loosened: "
            "leave these as they are.",
            *misses,
        ]
    elif misses:
        lines += [
            "",
            "# Earlier successes the done criteria would have refused",
            "The official check confirmed these tries, yet at their ends a criterion failed, so "
            "a finish there would have been refused: loosen it (set_number value or tol) so that "
            "it holds at these ends too, unless a failed end above needs it as it is.",
            *misses,
        ]
    if diff is not None:
        lines += ["", "# Your changes so far (the cache against the Memory)", render_diff(diff)]
    lines += ["", "# Budget", budget_text(editor)]
    if rejected:
        latest_rejected = rejected[-_REJECTED:]
        more = "" if len(rejected) <= _REJECTED else f", the last {_REJECTED} of {len(rejected)}"
        lines += ["", f"# Edits rejected earlier on this task (do not repeat them{more})"]
        lines += [f"- {_op(record)}: {record.reason}" for record in latest_rejected]
    latest = tries[-1]
    lines += [
        "",
        "The frames of the tries under # What happened are loaded as log:<try>:<turn> (log:2:3 "
        "is try 2's frame of turn 3, log:2:3.2 its second), the aliases the logs name; look at "
        "them with get_image and the perception tools. check and relation on a try's frame read "
        "the objects as that try left them, under the names its log uses."
        + (
            f" Only the frames of {', '.join(log.attempt for log in tries)} are loaded: the "
            "earlier tries listed above have none here."
            if history
            else ""
        ),
    ]
    parts: list[ContentPart] = [TextPart(text="\n".join(lines))]
    shown = last_images(latest.turns)[:images]
    if shown:
        parts.append(TextPart(text=f"The last observation of {latest.attempt}:"))
        parts += shown
    return parts


def finding_lines(findings: Sequence[Finding]) -> list[str]:
    """One line per linter finding, for a model."""
    return [
        f"- {f.level} {f.rule} at {f.where}"
        + (f" ({f.match!r})" if f.match else "")
        + f": {f.message}"
        for f in findings
    ]


def lint_brief(findings: Sequence[Finding]) -> list[ContentPart]:
    """What an S1 Reviser reads when the Memory it opens fails the linter: the errors, which it
    fixes first (it may not finish while any is left)."""
    if not findings:
        return []
    lines = [
        "# The linter's errors on this Memory: fix them before anything else",
        *finding_lines(findings),
    ]
    return [TextPart(text="\n".join(lines))]


def review_brief(findings: Sequence[Finding]) -> list[ContentPart]:
    """The linter's review notes an S1 Reviser reads after an attempt (`relative_carry`):
    warnings on how the Memory binds the demonstration's end, which it weighs, not must fix."""
    if not findings:
        return []
    lines = ["# Review notes on this Memory (warnings: weigh them)", *finding_lines(findings)]
    return [TextPart(text="\n".join(lines))]


def cleanup_brief(findings: Sequence[Finding], *, editor: Editor) -> list[ContentPart]:
    """The lint-fix Reviser's material (`imitate`'s `cleanup`): the linter's findings on the
    Memory, errors and warnings, under the sections its prompt names."""
    lines = ["# The linter's findings on this Memory", *finding_lines(findings)]
    if not findings:
        lines.append("None.")
    lines += ["", "# Budget", budget_text(editor)]
    return [TextPart(text="\n".join(lines))]


def try_log(
    name: str,
    label: str,
    outcome: EpisodeOutcome,
    *,
    diagnosis: bool,
    aliases: Mapping[str, str] | None = None,
    keypoints: Sequence[str] = (),
    criteria: Sequence[DoneSpec] = (),
    expert: Sequence[str] = (),
    earlier: Sequence[tuple[str, Path]] = (),
    edits: Mapping[str, str] | None = None,
) -> TryLog:
    """A try's log from its directory; with `diagnosis`, the ground truth under `referee/` too,
    in the try's own terms (the Reviser's mode (b), and S1); `criteria`: the done criteria it ran
    under, with the relations read at its end; `expert`: S1's privileged review against the
    expert (`learning.expert_diff.render`), which only S1 passes; `earlier`: the seed's earlier
    tries (a name and its directory, oldest first), whose stored specs that failed to plan at the
    same pose this one did are named (`repeat_lines`); `edits`: the Memory specs the revision
    before this try edited (spec -> what it did), whose failed calls that ran as in the try before
    are named (`unchanged_lines`)."""
    turns = outcome.directory / "turns.jsonl"
    found = outcome.directory / REFEREE / EXPLAINED
    identified = load(outcome.directory)
    objects = outcome.directory / OBJECTS
    return TryLog(
        attempt=name,
        label=label,
        outcome=outcome,
        turns=read_lines(turns) if turns.is_file() else [],
        aliases=dict(aliases or {}),
        diagnosis=Explained.model_validate_json(found.read_bytes())
        if diagnosis and found.is_file()
        else None,
        keypoints=keypoints,
        identified=[]
        if identified is None
        else [p.text for p in identified.content if isinstance(p, TextPart)],
        objects=json.loads(objects.read_bytes()) if objects.is_file() else [],
        criteria=criteria,
        relations=read_relations(outcome.directory),
        unmoved=unmoved_lines(outcome.directory),
        repeats=repeat_lines(outcome.directory, earlier),
        unchanged=unchanged_lines(outcome.directory, earlier[-1] if earlier else None, edits or {}),
        expert=expert,
        ending=ending_grasp(outcome.directory, _CALL_ID.findall(outcome.end.note or "")),
        closes=close_turns(outcome.directory) if diagnosis else [],
        grasped=grasped_objects(outcome.directory),
    )


Vec3Float = tuple[float, float, float]


@dataclass(frozen=True)
class EndingGrasp:
    """The grasp call a try ended on with a hard failure (`ending_grasp`): the tool and its
    receipt, the Memory spec it passed (`spec`: its name, None for one written out in the call;
    `fields`: the spec as expanded), the object it named and that object's line in the registry
    at the end (`registered`: center and size, None when not found), and whether the call found
    it moved (`relocated`)."""

    tool: str
    receipt: ActionReceipt
    obj: str
    spec: str | None
    fields: Mapping[str, Any] | None
    registered: tuple[str, str, Vec3Float, Vec3Float] | None
    relocated: bool = False
    """The call measured the object again and found it moved (`remeasure`, relocated): the
    registry's center is where it was found after, not where the grasp aimed."""


_REGISTERED: Final = re.compile(
    r"^(\S+) (\S+): center \(([-\d.]+), ([-\d.]+), ([-\d.]+)\), "
    r"([\d.]+) x ([\d.]+) x ([\d.]+) m"
)
_SPEC_EXPANDED: Final = "manipulation.spec_expanded"
_REMEASURE: Final = "manipulation.remeasure"


def ending_grasp(directory: Path, calls: Collection[str]) -> EndingGrasp | None:
    """The grasp call among `calls` (the ids the try's end names) whose receipt is a hard failure,
    from the try's events: its receipt (`tool.result`), the spec it passed by name as the tool
    expanded it, and the object's registered line at the end (`objects.json`)."""
    path = directory / EVENTS
    if not calls or not path.is_file():
        return None
    expanded: dict[str, Any] = {}
    arguments: dict[str, Any] = {}
    relocated = False
    for event in read_events(path):
        data = event.data
        if event.kind == _SPEC_EXPANDED and data.get("kind") == "grasp_spec":
            expanded[str(data["name"])] = data["spec"]
        elif event.kind == EventKind.TOOL_CALL and data.get("call_id") in calls:
            arguments, relocated = data.get("arguments") or {}, False
        elif event.kind == _REMEASURE and data.get("reason") == "relocated":
            relocated = True
        elif event.kind == EventKind.TOOL_RESULT and data.get("call_id") in calls:
            result = data.get("result")
            if not isinstance(result, Mapping) or result.get("grasp") is None:
                continue
            receipt = ActionReceipt.model_validate(result)
            if not receipt.hard_failure:
                continue
            given = arguments.get("grasp")
            name = given.strip("{}") if isinstance(given, str) else None
            obj = str(arguments.get("obj") or "")
            return EndingGrasp(
                tool=str(data.get("name")),
                receipt=receipt,
                obj=obj,
                spec=name,
                fields=expanded.get(name)
                if name
                else given
                if isinstance(given, Mapping)
                else None,
                registered=_registered(directory, obj),
                relocated=relocated,
            )
    return None


def _registered(directory: Path, obj: str) -> tuple[str, str, Vec3Float, Vec3Float] | None:
    """An object's alias, role, center and size as `objects.json` lists it (by alias or role)."""
    path = directory / OBJECTS
    if not obj or not path.is_file():
        return None
    for line in json.loads(path.read_bytes()):
        found = _REGISTERED.match(str(line))
        if found and obj in (found[1], found[2]):
            v = [float(x) for x in found.groups()[2:]]
            return found[1], found[2], (v[0], v[1], v[2]), (v[3], v[4], v[5])
    return None


def grasped_objects(directory: Path) -> dict[str, Grasped]:
    """The objects a try saw, as perception last fitted each (`perception.step` fit events),
    by alias and by role (`objects.json`): what the Memory editor checks box grasps against
    (`memory.faces`). Empty when the try has no events."""
    path = directory / EVENTS
    if not path.is_file():
        return {}
    fits: dict[str, Grasped] = {}
    aliases: dict[str, str] = {}
    for event in read_events(path):
        data = event.data
        if event.kind == EventKind.PERCEPTION_STEP and data.get("step") == "fit":
            size = data.get("size")
            if isinstance(size, list) and len(size) == 3 and data.get("shape"):
                fits[str(data["object"])] = Grasped(
                    str(data["shape"]), (float(size[0]), float(size[1]), float(size[2]))
                )
        elif event.kind == EventKind.HUB_ALIAS and data.get("kind") == "obj":
            aliases[str(data["alias"])] = str(data["value"])
    found: dict[str, Grasped] = {}
    objects = directory / OBJECTS
    for line in json.loads(objects.read_bytes()) if objects.is_file() else []:
        matched = _REGISTERED.match(str(line))
        if matched is None or (fit := fits.get(aliases.get(matched[1], ""))) is None:
            continue
        found[matched[1]] = found[matched[2]] = fit
    return found


def close_turns(directory: Path) -> list[str]:
    """`tipped_closes` as the Reviser reads them, one line each."""
    return [
        f"- the {c.arm} grasp_object with {c.spec or 'a spec written out in the call'} on "
        f"{c.obj or 'an object'}: the object turned {c.turned_deg:.0f} deg as the gripper closed"
        for c in tipped_closes(directory)
    ]


@dataclass(frozen=True)
class TippedClose:
    """A grasp whose close turned what it closed on (`tipped_closes`): the arm, the Memory spec
    it passed by name (None: written out in the call), the object as the call named it, and how
    far (deg) the object turned."""

    arm: str
    spec: str | None
    obj: str
    turned_deg: float


def tipped_closes(directory: Path) -> list[TippedClose]:
    """The grasps of a try whose close turned what they closed on by `_TILTED_DEG` or more (the
    arm, the spec, the object as the call named it, how far it turned), from the
    ground-truth poses recorded around each executed batch (`physics.npz`'s `truth.*`, S1 only):
    the object nearest the closing TCP just after the grasp, against its pose just before it.
    Each executed grasp_object whose receipt closed an arm is matched, in order, to that arm's
    gripper closes in the turns whose action was grasp_object. Empty without ground truth."""
    found: list[TippedClose] = []
    physics, events = directory / PHYSICS, directory / EVENTS
    if not physics.is_file() or not events.is_file():
        return []
    with np.load(physics) as loaded:
        recorded = dict(loaded)
    if "truth.sample" not in recorded:
        return []
    samples = np.asarray(recorded["truth.sample"], dtype=int)
    truth = {
        key: np.asarray(value, dtype=float)
        for key, value in recorded.items()
        if key.startswith("truth.") and key != "truth.sample"
    }
    logged = directory / "turns.jsonl"
    actions = (
        {int(t["turn"]): t.get("action") for t in read_lines(logged)} if logged.is_file() else {}
    )
    turns = np.asarray(recorded.get("turn", np.zeros(0)), dtype=int)
    arguments: dict[str, Mapping[str, Any]] = {}
    grasps: dict[str, list[Mapping[str, Any]]] = {}
    for event in read_events(events):
        data = event.data
        if event.kind == EventKind.TOOL_CALL and data.get("name") == "grasp_object":
            arguments[str(data.get("call_id"))] = data.get("arguments") or {}
        elif event.kind == EventKind.TOOL_RESULT and data.get("name") == "grasp_object":
            result = data.get("result")
            if not isinstance(result, Mapping):
                continue
            for grip in result.get("grips") or []:
                if grip.get("closed"):
                    args = arguments.get(str(data.get("call_id")), {})
                    grasps.setdefault(str(grip.get("arm")), []).append(args)
    for arm, calls in grasps.items():
        command = recorded.get(f"{arm}.gripper_cmd")
        tcp = recorded.get(f"{arm}.tcp")
        if command is None or tcp is None or not truth:
            continue
        command = np.asarray(command, dtype=float)
        rows = [
            r
            for r in range(1, len(command))
            if command[r - 1] >= 0.5 > command[r]
            and r < len(turns)
            and actions.get(int(turns[r])) == "grasp_object"
        ]
        for row, args in zip(rows, calls, strict=False):  # a close the run never reached: none
            before = np.flatnonzero(samples < row)
            after = np.flatnonzero(samples >= row)
            if not len(before) or not len(after):
                continue
            b, a = int(before[-1]), int(after[0])
            point = np.asarray(tcp[min(int(samples[a]), len(tcp) - 1)][:3], dtype=float)
            nearest = min(
                truth.values(), key=lambda poses: float(np.linalg.norm(poses[a][:3] - point))
            )
            turned = float(quat_angle_deg(nearest[b][3:], nearest[a][3:]))
            if turned < _TILTED_DEG:
                continue
            given = args.get("grasp")
            spec = given.strip("{}") if isinstance(given, str) else None
            found.append(TippedClose(arm, spec, str(args.get("obj") or ""), turned))
    return found


_TIPPED_DEG: Final = 30.0
"""An object whose up direction turned this far fell over."""
_TILTED_DEG: Final = 5.0
"""A tilt worth naming (less is noise of the pose)."""


def ending_text(
    ending: EndingGrasp, diagnosis: Explained | None, grasped: Mapping[str, Grasped] | None = None
) -> list[str]:
    """The grasp call a try ended on, gathered for the Reviser: the spec that ran, where the
    commanded TCP was against the object's registered center, how the gripper closed and, with the
    diagnosis, what happened to the object; what to edit."""
    receipt, name = ending.receipt, ending.spec
    which = f"the Memory's {name}" if name else "a spec written out in the call"
    lines = [f"The call that ended it: {ending.tool} on {ending.obj or 'an object'} with {which}."]
    if ending.fields:
        shown = {k: v for k, v in ending.fields.items() if k not in ("reference_q",)}
        lines.append(f"- the spec as it ran: {json.dumps(shown, separators=(',', ':'))}")
    choice = receipt.grasp
    reach = choice is not None and choice.unplannable + choice.blocked > 0
    if choice is not None:
        lines.append(f"- grasp candidate {choice.candidate + 1} of {choice.n_candidates}")
    if choice is not None and reach:
        why = [
            f"{n} {what}"
            for n, what in (
                (choice.unplannable, "out of the arm's reach"),
                (choice.blocked, "ran into the other arm"),
            )
            if n
        ]
        lines.append(
            f"- the chooser preferred {choice.unplannable + choice.blocked} candidates to it, "
            f"which it passed over ({'; '.join(why)}): reach chose this grasp, not the spec's "
            "axes, so turning or tilting the spec keeps the arm on the same reachable candidate. "
            "Change what limits the reach: the arm, the approach direction, or where the spec "
            "grasps (offset, anchor, part)"
        )
    tcp = None if receipt.commanded is None else receipt.commanded.tcp
    if tcp is not None and ending.registered is not None:
        alias, role, center, size = ending.registered
        offset = np.asarray(tcp.p) - np.asarray(center)
        if ending.relocated:  # its center is where it was found after, not what the grasp aimed at
            lines.append(
                f"- the call found {alias} {role} moved and registered it again where it was "
                "found: the registry no longer says where the grasp aimed (see the receipt's note)"
            )
        else:
            lines.append(
                f"- the commanded TCP was {_cm(float(np.linalg.norm(offset)))} from {alias} "
                f"{role}'s registered center ({_along(offset)}); registered size "
                f"{' x '.join(f'{v:.3f}' for v in size)} m (the registry as the try ended)"
            )
    if receipt.gripper is not None and receipt.gripper.measured is not None:
        held = {True: "holding", False: "holding nothing", None: "hold unknown"}
        lines.append(
            f"- after closing the gripper opening was {receipt.gripper.measured:.3f} "
            f"({held[receipt.gripper.holding]})"
        )
    fate = _fate_of(diagnosis, ending) if diagnosis is not None else None
    tipped = False
    if fate is not None:
        moved = float(np.linalg.norm(fate.moved))
        if fate.start_off_m is not None:
            lines.append(
                f"- ground truth: perception had it {_cm(fate.start_off_m)} off at the start"
            )
        if moved >= MOVED_M:
            tipped = fell(fate, _TIPPED_DEG)
            tilted = fate.tilt_deg is not None and fate.tilt_deg >= _TILTED_DEG
            how = f", tilted {fate.tilt_deg:.0f} deg{_as_demo(fate)}" if tilted else ""
            fall = " (it fell over: the approach or the close pushed it before the fingers held it)"
            lines.append(
                f"- ground truth: by the try's end it had moved {_cm(moved)} ({_along(fate.moved)})"
                f"{how}{fall if tipped else ''}"
            )
            if tilted and (square := _off_box(ending, grasped or {})):
                lines.append(square)
    target = name or "the spec the call passed"
    lines.append(
        f"Edit {target} for this failure unless the log shows the cause elsewhere (identity, "
        "perception, the arm, an earlier stage); an edit to another parameter leaves it as it was."
        + (
            " The object fell, so its center was not what was missed: look at how the fingers came "
            "in (approach, tilt, lean, closing, pre-grasp distance) before its offsets."
            if tipped and not reach
            else " The object fell where reach chose the grasp: see the line above."
            if tipped
            else ""
        )
    )
    return lines


def _off_box(ending: EndingGrasp, grasped: Mapping[str, Grasped]) -> str | None:
    """Why a tilted box is what the ending grasp's own axes did to it (its closing axis off the
    box's faces), or None."""
    obj = grasped.get(ending.obj)
    if obj is None or obj.shape != BOX or not ending.fields:
        return None
    try:
        spec = GraspSpec.model_validate(dict(ending.fields))
    except ValidationError:
        return None
    fit = face_fit(spec, obj.size)
    if fit is None or fit.tilt_deg <= FACE_TILT_DEG:
        return None
    return (
        f"- the spec's closing axis lay {fit.tilt_deg:.0f} deg off the faces of {ending.obj}, a "
        "box: fingers closing so meet an edge first and turn it. Align the closing axis with its "
        "faces (turn it back toward them); tilting it further does not fix the turn"
    )


def _fate_of(diagnosis: Explained, ending: EndingGrasp) -> Fate | None:
    names = {ending.obj, *(ending.registered[:2] if ending.registered else ())}
    for fate in diagnosis.objects:
        if fate.object is not None and (fate.object.alias in names or fate.object.role in names):
            return fate
    return None


def unmoved_lines(directory: Path) -> list[str]:
    """A try's rejections that moved nothing and went on (`hub.unmoved`, unhinted S1), one line
    each: the call, why it was rejected, and the action that ran next without a hard failure
    (its fix), if one did."""
    path = directory / EVENTS
    if not path.is_file():
        return []
    rejected: dict[str, dict[str, Any]] = {}
    fixes: dict[str, dict[str, Any]] = {}
    for event in read_events(path):
        if event.kind != EventKind.HUB_UNMOVED:
            continue
        data = event.data
        if data.get("phase") == "rejected":
            rejected[str(data["call_id"])] = data
        elif data.get("phase") == "fixed":
            fixes[str(data["after"])] = data
    lines = []
    for call_id, data in rejected.items():
        at = f"@{data['stage']}" if data.get("stage") else ""
        line = f"- {data['name']}{at} {_short(data.get('arguments'))}: {data['reason']}"
        line += f" ({_clip(str(data.get('note') or ''), 200)})"
        if (where := _unreached(directory, data)) is not None:
            line += f"; {where}"
        fix = fixes.get(call_id)
        if fix is None:
            line += "; no action ran after it without a hard failure"
        else:
            fat = f"@{fix['stage']}" if fix.get("stage") else ""
            line += f"; fixed by {fix['name']}{fat} {_short(fix.get('arguments'))}"
        lines.append(line)
    return lines


def _unreached(directory: Path, data: Mapping[str, Any]) -> str | None:
    """Where a rejected call's commanded TCP was (`hub.unmoved`'s `commanded`, world): the
    point and its offsets from the (at most three) nearest objects registered at the try's end
    (`objects.json`),
    so that a stored spec that cannot be reached is refit or moved, not called differently.
    None when the event has no commanded pose."""
    point = data.get("commanded")
    if not isinstance(point, list) or len(point) != 3:
        return None
    tcp = np.asarray(point, dtype=float)
    arm = f"{data['arm']} " if data.get("arm") else ""
    text = f"the {arm}TCP it was sent to: ({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f})"
    path = directory / OBJECTS
    found = []
    for line in json.loads(path.read_bytes()) if path.is_file() else []:
        if (m := _REGISTERED.match(str(line))) is not None:
            center = np.array([float(x) for x in m.groups()[2:5]])
            found.append((float(np.linalg.norm(tcp - center)), m[1], m[2], tcp - center))
    near = sorted(found, key=lambda f: f[0])[:3]
    if near:
        text += "; " + ", ".join(
            f"{_along(o) + ' of' if _along(o) else 'at'} {a} {r}'s center" for _, a, r, o in near
        )
        text += " (the objects as registered at the try's end)"
    return text


_SPEC_ARGUMENTS: Final = ("grasp", "place", "path", "hinge")
"""The arguments a Memory spec is passed by name (`grasp={grasp_x}`)."""
_SAME_POSE_M: Final = 0.01
"""Commanded TCPs this close are the same target (perception's error is about this)."""
_UNPLANNED: Final = frozenset({"unreachable", "plan_failed"})
_REACHED: Final = frozenset({"ok", "clamped"})


@dataclass(frozen=True)
class _Sent:
    """A model's call in a try's log: an action with the Memory spec it named, the receipt's
    reason and commanded TCP; or an information call (`spec`, `reason` None)."""

    name: str
    shown: str
    spec: str | None = None
    reason: str | None = None
    tcp: tuple[float, float, float] | None = None
    q: tuple[float, float, float, float] | None = None
    """The commanded TCP's orientation."""
    candidate: tuple[int, int] | None = None
    """The grasp candidate executed, and how many there were (`GraspChoice`)."""
    stage: str | None = None
    """The workflow stage the call's rationale named."""


def _sent(directory: Path) -> list[_Sent]:
    """A try's model calls in order, from its events."""
    path = directory / EVENTS
    if not path.is_file():
        return []
    called: dict[str, dict[str, Any]] = {}
    sent = []
    for event in read_events(path):
        data = event.data
        if event.kind == EventKind.TOOL_CALL and data.get("origin") == "model":
            called[str(data["call_id"])] = data
        elif event.kind == EventKind.TOOL_RESULT and str(data.get("call_id")) in called:
            call = called.pop(str(data["call_id"]))
            arguments = call.get("arguments") or {}
            name = str(call["name"])
            objs = arguments.get("objs") or [arguments.get("obj")]
            shown = " ".join([name, *(str(o) for o in objs if o)])
            result = data.get("result")
            if not isinstance(result, Mapping) or "reason" not in result or "ok" not in result:
                if not data.get("is_error"):
                    sent.append(_Sent(name, shown))
                continue
            receipt = ActionReceipt.model_validate(result)
            given = next(
                (arguments[k] for k in _SPEC_ARGUMENTS if isinstance(arguments.get(k), str)), None
            )
            tcp = q = None
            if receipt.commanded is not None:
                p, r = receipt.commanded.tcp.p, receipt.commanded.tcp.q
                tcp = (float(p[0]), float(p[1]), float(p[2]))
                q = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            choice = receipt.grasp
            sent.append(
                _Sent(
                    name,
                    shown,
                    None if given is None else given.strip("{}"),
                    receipt.reason.value,
                    tcp,
                    q,
                    None if choice is None else (choice.candidate, choice.n_candidates),
                    _rationale_stage(arguments),
                )
            )
    return sent


def _rationale_stage(arguments: Mapping[str, Any]) -> str | None:
    rationale = arguments.get("rationale")
    stage = rationale.get("stage") if isinstance(rationale, Mapping) else None
    return None if stage is None else str(stage)


def stuck_failure(directories: Sequence[Path]) -> tuple[str, str, str | None] | None:
    """The failure tries in a row (`directories`, oldest first) share, when they do: each one's
    first failed model call passing a Memory spec, the same tool, spec, receipt reason and
    stage in each, sent to the same point (each within `_SAME_POSE_M` of the try before), as
    (spec, reason, stage); None otherwise (or with fewer than two tries)."""
    if len(directories) < 2:
        return None
    firsts = []
    for directory in directories:
        failed = next(
            (
                c
                for c in _sent(directory)
                if c.spec and c.reason is not None and c.reason not in _REACHED
            ),
            None,
        )
        if failed is None or failed.tcp is None:
            return None
        firsts.append(failed)
    key = (firsts[0].name, firsts[0].spec, firsts[0].reason, firsts[0].stage)
    for before, after in itertools.pairwise(firsts):
        if (after.name, after.spec, after.reason, after.stage) != key:
            return None
        assert before.tcp is not None and after.tcp is not None
        if not _near(before.tcp, after.tcp):
            return None
    assert firsts[0].spec is not None and firsts[0].reason is not None
    return firsts[0].spec, firsts[0].reason, firsts[0].stage


def spec_calls(directory: Path) -> list[tuple[str, str]]:
    """A try's model actions that passed a Memory spec by name, as (spec, receipt reason)."""
    return [(c.spec, c.reason) for c in _sent(directory) if c.spec and c.reason is not None]


def _near(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return float(np.linalg.norm(np.subtract(a, b))) <= _SAME_POSE_M


def repeat_lines(directory: Path, earlier: Sequence[tuple[str, Path]]) -> list[str]:
    """A try's calls passing a Memory spec by name that could not be planned (unreachable,
    plan_failed) at a commanded TCP an earlier try (`earlier`: a name and its directory) could
    not plan the same spec at either, one line each: which tries, and, where an earlier try got
    the same call through later, at what TCP and after which information calls alone, which then
    says the spec was not what changed there. Nothing names a task: the logs say it."""
    if not earlier:
        return []
    failed: dict[tuple[str, str], tuple[float, float, float]] = {}
    for call in _sent(directory):
        if call.spec and call.reason in _UNPLANNED and call.tcp is not None:
            failed.setdefault((call.name, call.spec), call.tcp)
    lines = []
    tries = [(name, _sent(path)) for name, path in earlier]
    for (tool, spec), tcp in failed.items():
        same, fixed = [], []
        for name, calls in tries:
            first = next(
                (
                    i
                    for i, c in enumerate(calls)
                    if (c.name, c.spec) == (tool, spec)
                    and c.reason in _UNPLANNED
                    and c.tcp is not None
                    and _near(c.tcp, tcp)
                ),
                None,
            )
            if first is None:
                continue
            same.append(name)
            later = calls[first + 1 :]
            through = next(
                (
                    i
                    for i, c in enumerate(later)
                    if (c.name, c.spec) == (tool, spec) and c.reason not in _UNPLANNED
                ),
                None,
            )
            if through is None:
                continue
            between = list(dict.fromkeys(c.shown for c in later[:through] if c.reason is None))
            acted = [
                c
                for c in later[:through]
                if c.reason is not None and (c.name, c.spec) != (tool, spec)
            ]
            if acted:  # something moved in between: not only how the scene was measured
                continue
            at = later[through].tcp
            where = "" if at is None else f" at ({at[0]:+.3f}, {at[1]:+.3f}, {at[2]:+.3f})"
            fixed.append(
                f"in {name} the same call went through later{where} after only (its names) "
                + (", ".join(between) if between else "the same call again")
            )
        if not same:
            continue
        line = (
            f"- {tool} with {{{spec}}}: sent to ({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f}) "
            f"and not plannable, as in {', '.join(same)}"
        )
        if fixed:
            line += (
                "; "
                + "; ".join(fixed)
                + ": the spec stayed as it was there, what changed is how the target was "
                "measured, so make the Memory measure it that way (its OBJECTS or the stage) "
                "rather than editing the spec's numbers"
            )
        else:
            line += (
                f": calling it again does not help; change {spec} (fit_spec refits it where it "
                "was fitted, when it has drifted from its fit), its arm, or how its target is "
                "measured"
            )
        lines.append(line)
    return lines


_SAME_TURN_DEG: Final = 5.0
"""Commanded TCPs turned less than this from each other point the same way."""


def _turn_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """The angle (deg) between two orientations (unit quaternions)."""
    dot = abs(float(np.dot(a, b))) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(np.degrees(2.0 * np.arccos(min(dot, 1.0))))


def unchanged_lines(
    directory: Path, earlier: tuple[str, Path] | None, edits: Mapping[str, str]
) -> list[str]:
    """A try's failed calls passing a Memory spec the revision before it edited (`edits`: spec ->
    what the revision did to it) that the try before (`earlier`: its name and directory) sent to
    the same point (within `_SAME_POSE_M`) and that ended the same way, one line each: the edit
    did not change where the call acted, so it is no progress. Turned less than `_SAME_TURN_DEG`
    as well, the call ran as before; turned more, only its orientation changed, which did not
    change the outcome. A call that could not be planned is left to `repeat_lines`."""
    if earlier is None or not edits:
        return []
    name, path = earlier
    before = _sent(path)
    lines, seen = [], set()
    for call in _sent(directory):
        spec = call.spec
        if spec is None or spec not in edits or call.tcp is None:
            continue
        if call.reason is None or call.reason in _REACHED or call.reason in _UNPLANNED:
            continue
        if (call.name, spec) in seen:
            continue
        match = next(
            (
                c
                for c in before
                if (c.name, c.spec) == (call.name, spec)
                and c.reason == call.reason
                and c.tcp is not None
                and _near(c.tcp, call.tcp)
            ),
            None,
        )
        if match is None:
            continue
        seen.add((call.name, spec))
        assert match.tcp is not None
        apart = float(np.linalg.norm(np.subtract(match.tcp, call.tcp))) * 100
        turn = None if call.q is None or match.q is None else _turn_deg(call.q, match.q)
        where = f"{apart:.1f} cm" + ("" if turn is None else f" and {turn:.0f} deg") + " apart"
        if call.candidate is not None and call.candidate == match.candidate:
            where += f", grasp candidate {call.candidate[0] + 1} of {call.candidate[1]} both times"
        same = turn is None or turn < _SAME_TURN_DEG
        what = (
            "the edit did not change what ran"
            if same
            else "the edit only turned it, at the same point, and that did not change the outcome"
        )
        lines.append(
            f"- {call.name} with {{{spec}}}: after the revision's edit of {spec} ({edits[spec]}), "
            f"it was sent where {name} sent it ({where}) and ended as {call.reason} again: {what}, "
            "so it is no progress; change what decides where and how the call acts (the spec's "
            "position or approach, its arm, how its target is measured) or the stage, not this "
            "edit again"
        )
    return lines


def _short(arguments: object) -> str:
    """A call's arguments in one line, without its rationale."""
    if not isinstance(arguments, Mapping):
        return ""
    kept = {k: v for k, v in arguments.items() if k != "rationale"}
    return _clip(json.dumps(kept, ensure_ascii=False, separators=(",", ":")), 300)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def read_relations(directory: Path) -> dict[str, Any] | None:
    """A try's `relations.json` (None: none was read)."""
    path = directory / RELATIONS
    return json.loads(path.read_bytes()) if path.is_file() else None


def criteria_lines(
    criteria: Sequence[DoneSpec], relations: Mapping[str, Any]
) -> list[tuple[bool, str]]:
    """Each criterion against the relations a try ended with: whether it holds, and the line
    (measured, needed), with the tolerance `check_done` held it to there (capped by the objects'
    size, `perception.done.capped`, when the relations say how)."""
    return [(passed, line) for passed, _, line in criteria_reads(criteria, relations)]


def criteria_reads(
    criteria: Sequence[DoneSpec], relations: Mapping[str, Any]
) -> list[tuple[bool, float | None, str]]:
    """`criteria_lines` with each criterion's measured value (None: not read, or undecided)."""
    found = {(v["metric"], v["a"], v.get("b"), v.get("part")): v for v in relations["values"]}
    sides = relations.get("sides") or {}
    done = relations.get("done")
    config = None if done is None else DoneCheckConfig.model_validate(done)
    lines = []
    for given in criteria:
        spec = floored(given)
        if config is not None:
            spec = capped(given, sides.get(given.a), sides.get(given.b or ""), config)
        value = found.get((spec.metric, spec.a, spec.b, spec.part))
        measured = None if value is None else value["measured"]
        unit = UNITS[spec.metric][0]
        shown = (
            "not read"
            if value is None
            else f"undecided ({value.get('reason') or 'no reason'})"
            if measured is None
            else f"{measured:.3f} {unit}"
        )
        passed = verdict(spec, measured)
        cap = f" (capped from {given.tol:g} by the objects' size)" if spec.tol < given.tol else ""
        if spec.tol > given.tol:
            cap = f" (read at perception's error, not {given.tol:g})"
        lines.append(
            (
                passed,
                measured,
                f"{spec.label()}: {shown}, needs {spec.value:g} ± {spec.tol:g} {unit}{cap}",
            )
        )
    return lines


def criteria_misses(
    ends: Sequence[tuple[str, Mapping[str, Any]]], criteria: Sequence[DoneSpec]
) -> list[str]:
    """The confirmed successes among `ends` (a name, `relations.json`) that `criteria` would
    have refused, one line each with the criteria that failed there. A try ends where its success
    check first holds, as the demonstration's last frame, the criteria's target, does: the same
    kind of state, which a criterion must accept."""
    lines = []
    for name, relations in ends:
        if not relations["success"]:
            continue
        failed = [line for passed, line in criteria_lines(criteria, relations) if not passed]
        if failed:
            lines.append(f"- {name}: " + "; ".join(failed))
    return lines


def try_text(log: TryLog) -> list[str]:
    outcome = log.outcome
    verdict = "succeeded" if outcome.success else "failed"
    lines = [
        f"## {log.attempt} (Memory {log.label}): {verdict}, ended as {outcome.end.reason}"
        + (f" ({outcome.end.note})" if outcome.end.note else "")
        + f"; {outcome.turns} turns, {outcome.steps} steps"
        + (f" of {outcome.step_limit}" if outcome.step_limit else ""),
    ]
    if log.keypoints:  # S1: the official outcome and each missed keypoint's correction first
        lines += ["Keypoint report (the demonstration's keypoints against this try):"]
        lines += [f"- {line}" for line in log.keypoints]
    lines += list(log.expert)
    lines += [
        *_renamed(identified_text(log), log.aliases),
        *turns_text(log.turns, log.aliases, ending=_CALL_ID.findall(outcome.end.note or "")),
        *_renamed(
            ["Registered objects at the end:", *(f"- {line}" for line in log.objects)]
            if log.objects
            else [],
            log.aliases,
        ),
    ]
    if log.unmoved:
        lines += [
            "Calls rejected before anything moved (the attempt went on; where one came from the "
            "Memory, change it there, e.g. to what fixed it):",
            *_renamed(list(log.unmoved), log.aliases),
        ]
    if log.repeats:
        lines += [
            "Memory calls that could not be planned at the pose earlier tries could not plan "
            "either (the stored spec lands there on this scene each time):",
            *_renamed(list(log.repeats), log.aliases),
        ]
    if log.unchanged:
        lines += [
            "Memory calls the revision before this try edited that were sent where the try before "
            "sent them and failed the same way (the edit changed nothing that mattered there):",
            *_renamed(list(log.unchanged), log.aliases),
        ]
    if log.ending is not None:
        lines += ending_text(log.ending, log.diagnosis, log.grasped)
    if log.closes:
        lines += [
            "Ground truth: grasps whose close turned the object they closed on (from just before "
            "the grasp to just after it; perception does not see such a turn until it measures "
            "the object again, so what follows works from the pose before it):",
            *log.closes,
        ]
    lines += _criteria_text(log)
    if labels := failure_labels([outcome]):
        lines.append(f"Failure label: {', '.join(labels)} (from how it ended)")
    if log.diagnosis is not None:
        lines += diagnosis_text(log.diagnosis)
    return lines


def _criteria_text(log: TryLog) -> list[str]:
    """A false finish (S1's accepted after its refusals, or S2's that only the demonstration's
    end gated) asks how to tell the task is done; one that passed done criteria the official check
    did not confirm asks to tighten them, with each
    criterion and every relation measured at the try's end (a success they would have refused
    comes in `reviser_brief`'s `misses`: no Reviser runs right after a success)."""
    relations, reason = log.relations, log.outcome.end.reason
    if not log.outcome.success and (
        reason == "finish" or (reason == UNCONFIRMED and not log.criteria)
    ):
        return [
            "It finished although the task was not done (the official check never latched): "
            "find what was missing at its end and make the Memory say how to tell the task is "
            "done before finishing."
        ]
    if not log.criteria or reason != UNCONFIRMED:
        return []
    lines = [
        "Its finish passed check_done, yet the official check did not confirm success: the done "
        "criteria let a failed scene pass. Find what the scene got wrong and tighten the "
        "criterion that should have caught it (set_number value or tol), or define one more "
        "that separates this end from a success."
    ]
    if relations is None:
        return [*lines, "No relations were read at its end."]
    lines.append("The criteria at its end:")
    lines += [
        f"- {'pass' if ok else 'FAIL'} {line}"
        for ok, line in criteria_lines(log.criteria, relations)
    ]
    lines.append("Every relation read at its end:")
    lines += [f"- {_relation(value)}" for value in relations["values"]]
    return lines


def _relation(value: Mapping[str, Any]) -> str:
    a = value["a"] if value.get("part") is None else f"{value['a']}.{value['part']}"
    name = " ".join(x for x in (value["metric"], a, value.get("b")) if x)
    if value["measured"] is None:
        return f"{name}: undecided"
    return f"{name}: {value['measured']:.3f} {UNITS[value['metric']][0]}"


def identified_text(log: TryLog) -> list[str]:
    """The identification a try started from, as its Actor was shown it."""
    if not log.identified:
        return []
    found = log.outcome.identification
    how = "identified as it opened"
    if found is not None and found.status == "reused":
        how = f"reused from {found.source}, identified once for the seed"
    return [f"Objects at the start ({how}):", *log.identified]


def _renamed(lines: list[str], aliases: Mapping[str, str]) -> list[str]:
    if not aliases:
        return lines
    return [_FRAME.sub(lambda m: aliases.get(m[0], m[0]), line) for line in lines]


def turns_text(
    turns: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str] | None = None,
    *,
    ending: Collection[str] = (),
) -> list[str]:
    """A try's log as the agent saw it: per turn the observation's text, the model's text, each
    call with its arguments and the start of its output, the action's rationale, the end; frame
    aliases renamed by `aliases`. The calls in `ending` (ids: the call that ended the try) are
    shown with their whole spec and receipt, which the cut would split."""
    lines = []
    for turn in turns:
        lines.append(f"Turn {turn['turn']}:")
        observed = _text(turn.get("observation") or [])
        if observed:
            lines.append(f"  observation: {_cut(observed, _TEXT)}")
        for exchange in turn["exchanges"]:
            if exchange["text"]:
                lines.append(f"  model: {_cut(exchange['text'], _TEXT)}")
            for made in exchange["calls"]:
                arguments = made["arguments"]
                if isinstance(arguments, dict):
                    shown = {k: v for k, v in arguments.items() if k != "rationale"}
                    arguments = json.dumps(shown, ensure_ascii=False, separators=(",", ":"))
                whole = made.get("call_id") in ending
                output = _cut(_text(made["content"]), _ENDING if whole else _CALL_OUTPUT)
                shown_args = _cut(str(arguments), _ENDING if whole else _CALL_ARGS)
                lines.append(f"  {made['name']}({shown_args}) -> {output}")
        if turn.get("rationale"):
            rationale = json.dumps(turn["rationale"], ensure_ascii=False)
            lines.append(f"  rationale of {turn['action']}: {_cut(rationale, _TEXT)}")
        if turn.get("end"):
            end = turn["end"]
            note = f" ({end['note']})" if end["note"] else ""
            lines.append(f"  the episode ended: {end['reason']}{note}")
    if not aliases:
        return lines
    return [_FRAME.sub(lambda m: aliases.get(m[0], m[0]), line) for line in lines]


def diagnosis_text(explained: Explained) -> list[str]:
    """The ground truth after a try in the try's own names (`wicl.diagnosis`), for the
    Reviser alone: never write its numbers into the Memory."""
    lines = [
        "Ground truth after the try (the simulator's, in this try's names; for your diagnosis "
        "only, never for the Memory):"
    ]
    lines += [f"- {_fate(fate)}" for fate in explained.objects]
    lines += [
        f"- {_named(n)}: no task object where it was registered (another object?)"
        for n in explained.unmatched
    ]
    lines += [f"- {note}" for note in explained.notes]
    return lines


def _fate(fate: Fate) -> str:
    if fate.object is None:
        head = "a task object this try did not register"
        if fate.near is not None:
            other, distance = fate.near
            head += f" (at the start, {_cm(distance)} from {_named(other)})"
    else:
        head = _named(fate.object)
        if fate.start_off_m is not None:
            head += f": perceived {_cm(fate.start_off_m)} off at the start"
    parts = [head]
    if float(np.linalg.norm(fate.moved)) < MOVED_M:
        parts.append("did not move")
    else:
        parts.append(f"moved {_cm(float(np.linalg.norm(fate.moved)))} ({_along(fate.moved)})")
        if fell(fate, _TIPPED_DEG):
            parts.append(f"tilted {fate.tilt_deg:.0f} deg (fell over){_as_demo(fate)}")
        elif fate.tilt_deg is not None and fate.tilt_deg >= _TIPPED_DEG:
            parts.append(f"tilted {fate.tilt_deg:.0f} deg{_as_demo(fate)}")
        if fate.end_off_m is not None:
            parts.append(f"the registry had it {_cm(fate.end_off_m)} off at the end")
        for other, offset in fate.ends:
            where = _along(offset, ("above", "below"))
            parts.append(
                f"ends {_cm(float(np.linalg.norm(offset)))} from {_named(other)} ({where})"
            )
    for k, (start, end) in enumerate(fate.joints, 1):
        joint = "its joint" if len(fate.joints) == 1 else f"its joint {k}"
        went = (
            f"went from {start:.0%} to {end:.0%}"
            if abs(end - start) >= 0.02
            else f"stayed at {start:.0%}"
        )
        parts.append(f"{joint} {went} of its range")
    return "; ".join(parts)


def _as_demo(fate: Fate) -> str:
    """The demonstration's own end tilt of the object, when known: a tilt near it is the
    task's, not a fall."""
    if fate.demo_tilt_deg is None:
        return ""
    return f" (the demonstration ends it tilted {fate.demo_tilt_deg:.0f} deg from its start)"


def _named(named: Named) -> str:
    return f"{named.alias} {named.role}"


def _cm(meters: float) -> str:
    return f"{meters * 100:.1f} cm"


def _along(offset: Sequence[float], vertical: tuple[str, str] = ("up", "down")) -> str:
    """An offset in words along the robot's axes: right or left, forward or back, up or down;
    what is under half a centimeter left out."""
    words = (("right", "left"), ("forward", "back"), vertical)
    parts = [
        f"{_cm(abs(value))} {plus if value > 0 else minus}"
        for value, (plus, minus) in zip(offset, words, strict=True)
        if abs(value) >= 0.005
    ]
    return ", ".join(parts) or "about the same place"


def last_images(turns: Sequence[Mapping[str, Any]]) -> list[ContentPart]:
    """The images of the last observation in `turns`."""
    for turn in reversed(turns):
        images = [p for p in turn.get("observation") or [] if p.get("type") == "image"]
        if images:
            return [ImagePart.model_validate(part) for part in images]
    return []


def _op(record: EditRecord) -> str:
    return _cut(record.op.model_dump_json(exclude_none=True), _CALL_ARGS)


def _text(content: Sequence[Mapping[str, Any]]) -> str:
    return " / ".join(p["text"] for p in content if p.get("type") == "text")


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _xyz(p: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"
