"""Each segment fitted to a skill call with its free parameters, from the recording's truth on the
scene's measured boxes (`gt_scene.measured`), with the core's fitters (the S1 Demo Reader's): a
grasp is `grasp_object` with the TCP's pose in the object's frame where the hold was made
(`fit_grasp_pose`, `segments.taken`; a hand-over's receiving grasp with x toward the giving gripper,
its `other_tcp`), a place `place_object` with the held object's resting point from its target
(`fit_place`: the first form the demonstration bears out; a carry the same, kept: `is_open` false),
a hand-over's release `open_gripper`, a move `move_to_pose` to where the demonstration's TCP ended
(the demonstration scene's own pose: the ceiling is privileged), an arm's rest `return_home`. A
place's target (`place_target`) is, in the steps after a round's coverage, the one that coverage
fitted it with (`report_targets`, by `Segment.key`); in the coverage step, the one an earlier
coverage fitted it with where that is a measured object (the sweep's copies of the earlier rounds'
reports); else it is chosen on the boxes (`supporting`): the object it rests in, else the one it
stands on, else the nearest box within the cut's `target_m`. A touch has no skill among the entries
to fit: it is a gap (the reuse step tries every skill of the core and the other benchmarks there).

A written skill (`../builder/README.md`) is fitted by its own `fit` (`fit_written`): its
call carries the skill's name and `params`, which `run` hands to its `act`. Its `params` must be
relative (`RELATIVE_RULE`): `fit` runs again on the demonstration moved (`shifted`), and a number
that moves with it is an absolute position (`AbsoluteParams`).
"""

import functools
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.demos import gt_scene
from roboscript.embodiments.profile import PoseConventions
from roboscript.manipulation.fit import fit_grasp_pose, fit_place
from roboscript.manipulation.posemath import Constrain
from roboscript.manipulation.specs import GraspSpec, PathSpec, PlaceSpec
from roboscript_methods.skill_builder.demo_coverage.segments import CutConfig, Recording, Segment

if TYPE_CHECKING:
    from roboscript_methods.skill_builder.builder.plugins import WrittenSkill
    from roboscript_methods.skill_builder.builder.static_check import Vocabulary

TRANSITION_CORE: Final = ("return_home", "open_gripper")
"""The core entries a task plan may put between segments (`Call.before`): they take the arm
only, so a transition needs no fit."""

SKILLS = {
    "grasp": "grasp_object",
    "place": "place_object",
    "release": "open_gripper",
    "carry": "place_object",
    "move": "move_to_pose",
    "home": "return_home",
}


@dataclass(frozen=True)
class Call:
    """A segment's fitted call: the skill (None: none fits, `why`), its spec, the form fitted
    (`fit_grasp_pose`'s variant, `fit_place`'s form), what the spec cannot reproduce, for a
    touch whose object moved while held the held path its grasp then follows (`follow_path`),
    and for a place `place_object`'s `constrain`, `approach_m` and `approach_axis` (a reuse
    alternative fits them: `reuse.released_place`, with how far each `constrain` commands the ee
    from the demonstration's, `gaps`); a written skill's call, its `params` (`fit_written`)."""

    segment: Segment
    skill: str | None
    grasp: GraspSpec | None = None
    place: PlaceSpec | None = None
    form: str = ""
    residual_m: float = 0.0
    residual_deg: float = 0.0
    why: str = ""
    path: PathSpec | None = None
    constrain: Constrain = "auto"
    approach_m: float = 0.1
    approach_axis: Literal["grasp", "target_z"] | tuple[float, float, float] = "grasp"
    gaps: dict[str, tuple[float, float]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    before: tuple["Call", ...] = ()
    """A task plan's transition calls run before this one, from where the scene is
    (`../builder/README.md`, "Per-task closure"): each a written skill fitted on this segment, or
    a core entry that takes only the arm (`TRANSITION_CORE`)."""

    def record(self) -> dict[str, Any]:
        s = self.segment
        spec = self.grasp or self.place
        found = {
            "segment": s.name,
            "arm": s.arm,
            "kind": s.kind,
            "frames": [s.start, s.end],
            "obj": s.obj,
            "target": s.target,
            "rest": s.rest,
            "skill": self.skill,
            "spec": None if spec is None else spec.model_dump(mode="json", exclude_defaults=True),
            "form": self.form,
            "residual_m": round(self.residual_m, 4),
            "residual_deg": round(self.residual_deg, 2),
            "why": self.why,
            "path": None if self.path is None else self.path.model_dump(mode="json"),
            "constrain": self.constrain,
            "approach_m": round(self.approach_m, 4),
            "approach_axis": self.approach_axis,
            "gaps": {c: list(g) for c, g in self.gaps.items()},
        }
        if self.params:  # left out when empty
            found["params"] = self.params
        if self.before:  # likewise
            found["before"] = [{"skill": c.skill, "params": c.params} for c in self.before]
        return found

    def data(self, skills: Sequence["WrittenSkill"] = ()) -> dict[str, Any]:
        """The call whole (`record` rounds and leaves out), for `restored`: what a step records
        of a call it chose, so a later step runs that call rather than fitting it again on its
        own measurement of the scene, where the core can pick another form (a reuse's
        `grasp_object side/180`, chosen on the boxes it measured, may be no option of the chained
        step's). `skills` (those given the step): each written skill the call and its transitions
        run, by its content (`skill_hashes`), so a later step restores it only on the same code."""
        axis = self.approach_axis
        names = {c.skill for c in (self, *self.before) if is_written(c.form)}
        return {
            "skill": self.skill,
            "grasp": None if self.grasp is None else self.grasp.model_dump(mode="json"),
            "place": None if self.place is None else self.place.model_dump(mode="json"),
            "form": self.form,
            "residual_m": float(self.residual_m),
            "residual_deg": float(self.residual_deg),
            "why": self.why,
            "path": None if self.path is None else self.path.model_dump(mode="json"),
            "constrain": self.constrain,
            "approach_m": float(self.approach_m),
            "approach_axis": axis if isinstance(axis, str) else [float(x) for x in axis],
            "gaps": {c: [float(x) for x in g] for c, g in self.gaps.items()},
            "params": self.params,
            "before": [c.data() for c in self.before],
            "skills": {s.name: skill_hash(Path(s.dir)) for s in skills if s.name in names},
        }


def is_written(form: str) -> bool:
    """Whether a call's form is a written skill's (`fit_written`)."""
    return form.startswith("written ")


def content_hash(directory: Path) -> str:
    """A directory's files (caches left out), by their paths and contents."""
    digest = hashlib.sha256()
    files = sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


@functools.cache
def skill_hash(directory: Path) -> str:
    """A written skill's `content_hash`, once per process (a step's skills do not change)."""
    return content_hash(directory)


def restored(segment: Segment, data: dict[str, Any]) -> Call:
    """The call `Call.data` recorded, on `segment` (the recording's cut gives the same segments
    to every step; a transition is fitted on its segment, `Call.before`)."""
    axis = data["approach_axis"]
    return Call(
        segment,
        data["skill"],
        grasp=None if data["grasp"] is None else GraspSpec.model_validate(data["grasp"]),
        place=None if data["place"] is None else PlaceSpec.model_validate(data["place"]),
        form=data["form"],
        residual_m=data["residual_m"],
        residual_deg=data["residual_deg"],
        why=data["why"],
        path=None if data["path"] is None else PathSpec.model_validate(data["path"]),
        constrain=data["constrain"],
        approach_m=data["approach_m"],
        approach_axis=axis if isinstance(axis, str) else tuple(axis),
        gaps={c: (g[0], g[1]) for c, g in data["gaps"].items()},
        params=dict(data["params"]),
        before=tuple(restored(segment, b) for b in data["before"]),
    )


def runnable(
    data: dict[str, Any], skills: Sequence["WrittenSkill"], recorded: dict[str, str]
) -> bool:
    """Whether a call `Call.data` recorded (one of its transitions: `recorded`, the call's
    `skills`) runs as it ran with `skills`: a core entry, or a written skill given with the
    content it had."""
    if not is_written(data["form"]):
        return True
    given = {s.name: s for s in skills}
    skill = given.get(data["skill"])
    return skill is not None and recorded.get(skill.name) == skill_hash(Path(skill.dir))


def fit(
    segment: Segment,
    recording: Recording,
    scene: gt_scene.Scene,
    base: Pose,
    conventions: PoseConventions,
    config: CutConfig,
    targets: Mapping[str, str | None] | None = None,
    earlier: Mapping[str, str | None] | None = None,
) -> Call:
    """`segment`'s call, fitted on `scene`'s boxes at the recorded poses; a place's or a carry's
    target the one `targets` holds for it (by `Segment.key`: the round's coverage's, which the
    steps after it fit with, `report_targets`; one that is no measured object here raises), else
    the one `earlier` holds for it where that is a measured object here (an earlier coverage's:
    the coverage step's, so a near tie is chosen the same way, `place_target`), else the one it
    rests on (`supporting`)."""
    kind = segment.kind
    if kind in ("home", "release", "move"):
        return Call(segment, SKILLS[kind])
    if kind == "touch":
        return Call(segment, None, why="a close that lifted nothing: no entry presses or steadies")
    assert segment.obj is not None  # a grasp's or a place's object
    if segment.obj not in scene.boxes:
        return Call(segment, None, why=f"truth has no box for {segment.obj} (a garment, a fluid)")
    if kind == "grasp":
        made = segment.end if segment.rest is None else segment.rest  # where the hold was made
        at = scene.at(recording.poses(made), {})
        other = None
        if segment.handover:  # the receiving grasp: x toward the giving gripper
            giver = next(a for a in recording.demo.arms if a != segment.arm)
            other = recording.demo.arms[giver].tcp[made, :3]
        found = fit_grasp_pose(
            Pose.from_list(recording.demo.arms[segment.arm].tcp[made].tolist()),
            at.proxy(segment.obj),
            arm=segment.arm,
            other_tcp=other,
            base=base.matrix(),
        )
        return Call(
            segment,
            SKILLS[kind],
            grasp=found.spec,
            form=found.variant,
            residual_m=found.position_error_m,
            residual_deg=found.angle_error_deg,
        )
    assert segment.rest is not None  # a place or a carry ends somewhere
    held = scene.at(recording.poses(max(segment.end - 1, 0)), {}).proxy(segment.obj)
    rested = scene.at(recording.poses(segment.rest), {})
    segment = replace(segment, target=place_target(segment, rested, config, targets, earlier))
    target = None if segment.target is None else rested.proxy(segment.target)
    forms = fit_place(
        rested.proxy(segment.obj).pose,
        held,
        target,
        base,
        held_now=held.pose,
        arm=segment.arm,
        conventions=conventions,
    )
    name, chosen = next(
        ((n, f) for n, f in forms.items() if f.consistent), next(iter(forms.items()))
    )
    return Call(
        segment,
        SKILLS[kind],
        place=chosen.spec,
        form=name,
        residual_m=chosen.position_error_m,
        residual_deg=chosen.angle_error_deg,
    )


def place_target(
    segment: Segment,
    rested: gt_scene.Scene,
    config: CutConfig,
    targets: Mapping[str, str | None] | None,
    earlier: Mapping[str, str | None] | None,
) -> str | None:
    """A place's or a carry's target (`fit`'s): `targets`' for its key (raises where that is no
    measured object here), else `earlier`'s where it is one (None, nothing beneath, as well),
    else `supporting`'s. The cut, the seed and the layout are a demonstration's own, so an
    earlier coverage's target differs from `supporting`'s only where two supports lie nearly as
    close and measurement noise picks one (a block laid across two others); a name kept from round
    to round keeps what is keyed by it applying (earlier chains' fixes, task plans, written skills'
    cases)."""
    if targets is not None and segment.key in targets:
        pinned = targets[segment.key]
        if pinned is not None and pinned not in rested.boxes:
            raise ValueError(f"{segment.key}: the target {pinned!r} an earlier step fitted it with "
                             f"is no measured object here ({sorted(rested.boxes)})")  # fmt: skip
        return pinned
    if earlier is not None and segment.key in earlier:
        before = earlier[segment.key]
        if before is None or (before != segment.obj and before in rested.boxes
                              and before in rested.poses):  # fmt: skip
            return before
    return supporting(rested, segment.obj, config.target_m)


def report_targets(calls: Iterable[dict[str, Any]]) -> dict[str, str | None]:
    """The targets a report's calls (`Call.record`) fitted their places and carries with, by
    `Segment.key` (`fit`'s `targets`)."""
    found: dict[str, str | None] = {}
    for row in calls:
        if row["kind"] not in ("place", "carry"):
            continue
        end = f" [{row['frames'][0]}, {row['frames'][1]}]"
        onto = f" at {row['target']}" if row["target"] else ""
        if not row["segment"].endswith(onto + end):
            raise ValueError(f"{row['segment']!r}: not a name of target {row['target']!r} and "
                             f"frames {row['frames']}")  # fmt: skip
        found[row["segment"].removesuffix(onto + end) + end] = row["target"]
    return found


MARGIN_M = 0.01
"""How far past a box's footprint a resting point still lies on or in it."""


def supporting(scene: gt_scene.Scene, held: str, limit_m: float) -> str | None:
    """The object `held` rests in or on: the largest box whose volume holds its bottom centre
    (within `MARGIN_M`: a bin, a box it lies in, though it lies on something else there); else,
    of the boxes whose footprint holds it below it, the one whose top is nearest; else the
    nearest box within `limit_m`; else None."""
    point = bottom(scene, held)
    inside, under, near = [], [], []
    for name in scene.boxes:
        if name == held or name not in scene.poses:
            continue
        proxy = scene.proxy(name)
        box = proxy.pose @ proxy.whole.frame
        local = np.linalg.inv(box) @ np.array([*point, 1.0])
        half = proxy.whole.size / 2
        if np.all(np.abs(local[:2]) <= half[:2] + MARGIN_M):
            if abs(local[2]) <= half[2] + MARGIN_M and local[2] < half[2] - MARGIN_M:
                inside.append((-float(np.prod(proxy.whole.size)), name))
            elif local[2] >= half[2] - MARGIN_M:
                under.append((float(local[2] - half[2]), name))
        outside = np.maximum(np.abs(local[:3]) - half, 0.0)
        near.append((float(np.linalg.norm(outside)), name))
    for found in (inside, under):
        if found:
            return min(found)[1]
    close = [n for n in near if n[0] <= limit_m]
    return min(close)[1] if close else None


def bottom(scene: gt_scene.Scene, name: str) -> np.ndarray:
    """An object's box's bottom centre, world."""
    proxy = scene.proxy(name)
    box = proxy.pose @ proxy.whole.frame
    return np.asarray(box[:3, 3] - box[:3, 2] * proxy.whole.size[2] / 2)


RELATIVE_RULE: Final = (
    "A fit's params hold no absolute position: every position in them is relative to a perceived "
    "object, a part or the gripper, never in the world's or the robot's frame. Checked by fitting "
    "again on the demonstration moved relative to the robot and the world (its objects and the "
    "gripper's path, together): every number in params must stay the same."
)
"""The rule a written skill's fit is held to, word for word as P2's `fit_skill` holds it. A pose the
robot returns to regardless of the scene (its rest pose) goes in joint space: `shifted` keeps the
joints, so such a pose passes, and it is the robot's own, not the scene's."""
SHIFT_M: Final = (0.04, -0.03, 0.0)
"""How far `shifted` moves the demonstration: across the table, not up (a height above the table,
which stays where it is, is relative to it); every absolute x or y then moves by 3 cm or more."""
SHIFT_TOLERANCE: Final = 0.01
"""A number of `params` that changes by more than this when the demonstration moves is absolute
(a relative one changes by rounding only, or by a fit's choice of frame, well under 1 cm)."""
# The shift check does not catch a constant in the world's or the robot's frame written into
# `fit.py`, nor a position computed from the recorded joints (forward kinematics): neither moves
# with the demonstration. The gate's transfer cases cover those. The tolerance applies to every
# number alike, unitless ones and angles included (a translation leaves those unchanged).


class AbsoluteParams(ValueError):
    """A written skill's fit broke `RELATIVE_RULE`."""


def moved_pose(pose: Pose, by: np.ndarray) -> Pose:
    return Pose(p=tuple(np.add(pose.p, by).tolist()), q=pose.q)


def moved_track(track: np.ndarray, by: np.ndarray) -> np.ndarray:
    """A (N, 7) pose track (or (N, 4, 4) transforms) moved by `by`: positions only."""
    found = np.array(track, dtype=float, copy=True)
    if found.ndim == 3:
        found[:, :3, 3] += by
    else:
        found[:, :3] += by
    return found


def shifted(
    segment: Segment, recording: Recording, scene: gt_scene.Scene, by: tuple[float, ...] = SHIFT_M
) -> tuple[Segment, Recording, gt_scene.Scene]:
    """The demonstration moved by `by` relative to the robot and the world: every object's pose
    (recorded and in the scene, links too), each arm's ee and TCP path, the cameras; the joints,
    the gripper and the robot's base stay. A relative fit gives the same `params` on it. What is
    not a recording or a scene (a test's stand-in) is handed on as it is."""
    d = np.asarray(by, dtype=float)
    moved, world = recording, scene
    if isinstance(recording, Recording):
        demo = recording.demo
        arms = {a: replace(t, ee=moved_track(t.ee, d), tcp=moved_track(t.tcp, d))
                for a, t in demo.arms.items()}  # fmt: skip
        calib = {c: replace(k, cam2world=moved_track(k.cam2world, d))
                 for c, k in demo.calib.items()}  # fmt: skip
        moved = replace(
            recording,
            demo=replace(demo, arms=arms, calib=calib),
            objects={label: moved_track(track, d) for label, track in recording.objects.items()},
        )
    if isinstance(scene, gt_scene.Scene):
        world = replace(
            scene,
            poses={n: moved_pose(p, d) for n, p in scene.poses.items()},
            links={n: {k: moved_pose(p, d) for k, p in ls.items()}
                   for n, ls in scene.links.items()},
        )  # fmt: skip
    return replace(segment, tcp=moved_pose(segment.tcp, d)), moved, world


def drift(a: object, b: object, where: str = "params") -> list[str]:
    """Where `b` differs from `a`: a number by more than `SHIFT_TOLERANCE`, or anything else."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{where}'s keys"]
        return [x for k in a for x in drift(a[k], b[k], f"{where}.{k}")]
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return [f"{where}'s length"]
        return [x for k, (u, v) in enumerate(zip(a, b, strict=True))
                for x in drift(u, v, f"{where}[{k}]")]  # fmt: skip
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return [where] if abs(float(a) - float(b)) > SHIFT_TOLERANCE else []
    return [] if a == b else [where]


def relative_problem(
    skill: "WrittenSkill",
    segment: Segment,
    recording: Recording,
    scene: gt_scene.Scene,
    base: Pose,
    conventions: PoseConventions,
    params: dict[str, Any],
) -> str | None:
    """Why `params` (the fit on the demonstration) break `RELATIVE_RULE`, else None: the fit on
    the demonstration moved (`shifted`) gives other numbers, no params, or raises."""
    try:
        again = skill.fit(*shifted(segment, recording, scene), base, conventions)
    except Exception as error:  # the Agent's code
        return f"its fit raises once the demonstration moves ({fit_error(error)[:80]})"
    if again is None:
        return "its fit applies only where the demonstration lies (None once it moves)"
    found = drift(params, json.loads(json.dumps(again, default=str)))
    if not found:
        return None
    named = list(dict.fromkeys(re.sub(r"(\[\d+\])+$", "", f) for f in found))  # a vector once
    return (f"{', '.join(named[:3])} move with the demonstration (absolute positions: make them "
            "relative to an object, a part or the gripper)")  # fmt: skip


def fit_written(
    skill: "WrittenSkill",
    segment: Segment,
    recording: Recording,
    scene: gt_scene.Scene,
    base: Pose,
    conventions: PoseConventions,
) -> Call | None:
    """`skill`'s call for `segment` (a fitted call's segment: a place's target is the one `fit`
    chose, `place_target`: the round's coverage's, an earlier coverage's, else `supporting`'s),
    None where its `fit` says it does not apply. Its `params` must be data the skill could have
    been told (`params_problems`), kept as JSON gives them back (a tuple a list), and relative
    (`RELATIVE_RULE`: `AbsoluteParams`); a `fit` that raises, or `params` that are not such data,
    raise here (the caller records `fit_error`)."""
    if skill.vocabulary is None:
        raise ValueError(f"{skill.name} was loaded without a vocabulary: its params go unchecked")
    params = skill.fit(segment, recording, scene, base, conventions)
    if params is None:
        return None
    problems = params_problems(params, skill.vocabulary)
    if problems:
        raise ValueError(f"{skill.name}'s params: {'; '.join(problems[:5])}")
    params = json.loads(json.dumps(params))
    if why := relative_problem(skill, segment, recording, scene, base, conventions, params):
        raise AbsoluteParams(f"{skill.name}: {why}")
    return Call(segment, skill.name, form=f"written {skill.name}", params=params)


PARAMS_DEPTH: Final = 4
PARAMS_LEAVES: Final = 256
PARAMS_TEXT: Final = 64
"""A fit's `params` nest dicts and lists at most this deep, hold at most this many numbers and
strings, each string at most this long: a few specs' worth, never a recording or a file."""


def params_problems(params: object, vocabulary: "Vocabulary") -> list[str]:
    """What makes `params` more than data `act` could be told: a top level that is not a dict;
    a value that is not a finite number, a bool, a short string or None, or a list or a dict
    (string keys) of those, deeper or larger than the bounds above; a string (a key too) that
    names a task or an object of `vocabulary`, or holds `__`."""
    if not isinstance(params, dict):
        return [f"a {type(params).__name__}, not a dict"]
    problems: list[str] = []
    leaves = 0

    def text(value: str, where: str) -> None:
        if len(value) > PARAMS_TEXT:
            problems.append(f"{where}: a string longer than {PARAMS_TEXT}")
        if "__" in value:
            problems.append(f"{where}: a string holding `__`")
        named = [t for _, t in vocabulary.tasks_in(value)] + vocabulary.labels_in(value)
        if named:
            problems.append(f"{where}: names {named[0]!r}")

    def walk(value: object, where: str, depth: int) -> None:
        nonlocal leaves
        if depth > PARAMS_DEPTH:
            problems.append(f"{where}: nested deeper than {PARAMS_DEPTH}")
        elif isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    problems.append(f"{where}: a key that is a {type(key).__name__}")
                    continue
                text(key, f"{where}.{key}")
                walk(item, f"{where}.{key}", depth + 1)
        elif isinstance(value, list | tuple):
            for k, item in enumerate(value):
                walk(item, f"{where}[{k}]", depth + 1)
        elif isinstance(value, str):
            leaves += 1
            text(value, where)
        elif value is None or isinstance(value, int) or finite(value):
            leaves += 1
        else:
            problems.append(f"{where}: a {type(value).__name__}")

    walk(params, "params", 0)
    if leaves > PARAMS_LEAVES:
        problems.append(f"{leaves} values, more than {PARAMS_LEAVES}")
    return problems


def finite(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value)


ACT_MARK: Final = "[written act]"
"""The line a written skill's `act` is opened (`<mark> begin <name>`) and closed (`<mark> end`)
with on stderr (`run.act_written`): a job stopped between them was stopped in the skill."""
ERROR_CHARS: Final = 200


def fit_error(error: BaseException) -> str:
    """An error of the Agent's code as a report may carry it: its type and the first
    `ERROR_CHARS` characters of its message, on one line of printable characters (a message may
    quote whatever the code read)."""
    text = "".join(c if c.isprintable() else " " for c in str(error)[:ERROR_CHARS])
    return f"{type(error).__name__}: {text}"
