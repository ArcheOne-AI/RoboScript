"""`fit_skill` (docs/concepts/learning.md): S1 fits a written skill's
parameters on a segment of the demonstration, so that the Memory keeps them (a `skill_spec`
parameter, defined with `fitted` naming the segment's end keypoint) and S2 and P3 run the skill
with them (`run_skill`, by the parameter's name).

The fitters are the skillsets' (`SkillSetPlugin.fitters`, the skillsets `DemoConfig.fitters`
names), each a P1 builder's `fit.py` (`roboscript.tools.skill_fit`). Here they run at tier D on
what a real robot has: the recorded proprioception (`DEMO_TRACKS`, every frame) and the objects
this episode identified on the live scene of the demo seed (the scene the demonstration started
from), moved frame by frame by the recorded grasps (`relation.replay_grips`, as `demo_relation`
replays them; a held object rides on its gripper's recorded TCP). A fitter that reads more (the
simulator's truth) raises here, and the tool says so.

Every fit is checked relative (`tools.skill_fit.absolute_numbers`: fitted again with the
demonstrated world shifted across the table, the robot left where it is, it must not change; a
fit with a position in the world's or the robot's frame is refused) and rehearsed: the skill's
`act` builds its motions on the demonstration's scene at the segment's start (`Rehearsal`, an
action API that moves nothing), and the fit's residual is how far the commanded TCP waypoints and
the demonstrated TCP track lie apart (`residual_mm`). The tool ranks the fits by it, so that the
Agent picks a skill by how well it reproduces the segment; it shows skills by name only.

The tool is added to the `demo` toolset by the `skill_fitting` skillset (`SKILL_FITTING`), which
a stack mounts on its `demo` provider where its benchmark's skillsets have fitters; wicl's
phases show it through `demo.+`, so a stack without it shows nothing new.
"""

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Final

import anyio
import numpy as np
from pydantic import BaseModel, Field, JsonValue
from pydantic_core import to_jsonable_python

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import SkillSet
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.demos import Demo, KeypointSet, frame_alias
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning.demo import DemoTracks
from roboscript.manipulation import ObjectProxy
from roboscript.manipulation.api import Built, displaced, robot_base
from roboscript.manipulation.plan import Grip, Move
from roboscript.perception.registry import ObjectRecord, SceneRegistry, object_geometry
from roboscript.spatial.pose import pose_compose
from roboscript.tools import DEMO, DEMO_TRACKS, ObjectRef, ToolContext, tool
from roboscript.tools.skill_fit import (
    RELATIVE_RULE,
    SHIFTS_M,
    Actor,
    FitSegment,
    Fitter,
    SegmentKind,
    WrittenFit,
    absolute_numbers,
)
from roboscript.tools.skillsets import SkillSetPlugin, load_skillset, skill
from roboscript.tools.spec import ToolsetExtension
from roboscript.toolsets.scene import embodiment, stored_table
from roboscript_methods.wicl.relation import NAME, find, frames, replay_grips
from roboscript_methods.wicl.releases import demo_releases

if TYPE_CHECKING:
    from roboscript_methods.wicl.toolsets.demo import Demo as DemoToolset

SKILL_FITS: Final = "demo_skill_fits"
"""The skills' parameters `fit_skill` fitted in this episode: `{"<arm> K<n>": [{"skill": <name>,
"params": {...}, "segment": <its name>}, ...]}` by the segment's end keypoint, which
`define_param` checks a `skill_spec`'s definition against (`toolsets.demo.check_fitted`)."""
REHEARSAL_S: Final = 10.0
"""The bound on one rehearsal of a skill's `act` (it builds motions; nothing runs)."""
DIGITS: Final = 4
"""A fit's numbers are rounded to 0.1 mm (and 1e-4 of a unit elsewhere), as `demo_relation`'s
specs are: what the Reader copies into the Memory is what the edit check compares."""


class SkillFit(BaseModel):
    """One skill's fit: its parameters, or why there are none (`why`: it does not apply to the
    segment, its fit needs what perception does not give, or it carries absolute positions);
    `residual_mm`: how far the fitted call's TCP waypoints and the demonstrated TCP track lie
    apart (None: the rehearsal could not tell, `rehearsal` says why); `rank`: its place among
    the segment's fits by residual (1 the closest)."""

    skill: str
    params: dict[str, JsonValue] | None = None
    why: str | None = None
    residual_mm: float | None = None
    rehearsal: str | None = None
    rank: int | None = None


class SkillFits(BaseModel):
    """The segment fitted (`end`: its end keypoint, what `define_param`'s `fitted` names) and
    each skill's fit."""

    segment: str
    end: str
    fits: list[SkillFit]


def render_fits(result: SkillFits) -> list[ContentPart]:
    fitted = [fit for fit in result.fits if fit.params is not None]
    lines = [
        f"Segment {result.segment} (ending at {result.end}): {len(fitted)} skill(s) fit, closest "
        "first (residual: how far the fitted call's TCP waypoints and the demonstrated TCP track "
        "lie apart; choose by it, the smaller the closer):"
    ]
    for fit in sorted(fitted, key=lambda f: (f.rank is None, f.rank or 0)):
        how = (
            f"residual {fit.residual_mm:.0f} mm (rank {fit.rank})"
            if fit.residual_mm is not None
            else f"residual unknown ({fit.rehearsal})"
        )
        shown = json.dumps(fit.params, separators=(",", ":"))
        lines.append(
            f"- {fit.skill}: {how}; params {shown} (to use it, define a skill_spec with skill "
            f"{fit.skill}, value this JSON text as given, fitted {result.end}; the stage's "
            "run_skill takes it by the parameter's name)"
        )
    for fit in result.fits:
        if fit.params is None:
            lines.append(f"- {fit.skill}: no fit ({fit.why})")
    return [TextPart(text="\n".join(lines))]


@dataclass(frozen=True)
class PerceivedRecording:
    """The demonstration as a fitter reads it (`tools.skill_fit`): the recorded tracks and each
    registered object's pose per frame, as perceived and moved by the recorded grasps."""

    demo: Demo
    objects: dict[str, np.ndarray]

    def poses(self, frame: int) -> dict[str, Pose]:
        return {name: Pose.from_list(track[frame].tolist()) for name, track in self.objects.items()}


@dataclass(frozen=True)
class PerceivedScene:
    """The registered objects (`boxes`, keyed by id) at `poses`; `proxy` is an object's as the
    tools build it, at its pose here (held or not)."""

    records: Mapping[str, ObjectRecord]
    poses: dict[str, Pose]

    @property
    def boxes(self) -> Mapping[str, ObjectRecord]:
        return self.records

    def at(self, poses: Mapping[str, Pose], links: Mapping[str, Any]) -> "PerceivedScene":
        return replace(self, poses={name: poses[name] for name in self.records if name in poses})

    def proxy(self, name: str) -> ObjectProxy:
        loose = {"pose": self.poses[name], "grasps": {}, "part_grasps": {}, "part_poses": {}}
        return ObjectProxy.from_geometry(
            object_geometry(self.records[name].model_copy(update=loose))
        )


def perceived(
    keypoints: KeypointSet, registry: SceneRegistry, snapshots: Sequence[tuple[int, SceneRegistry]],
    demo: Demo,
) -> tuple[PerceivedRecording, PerceivedScene]:  # fmt: skip
    """Each object's pose at every frame: the replayed registry in force there (`snapshots`,
    (first frame, registry) in frame order), a held object on its gripper's recorded TCP."""
    n = demo.meta.frames
    starts = [first for first, _ in snapshots]
    tracks: dict[str, np.ndarray] = {name: np.zeros((n, 7)) for name in registry.objects}
    for frame in range(n):
        _, at = snapshots[bisect.bisect_right(starts, frame) - 1]
        for name, record in at.objects.items():
            if name not in tracks:
                continue
            if record.grasps:
                arm, grasp = next(iter(record.grasps.items()))
                tcp = demo.arms[arm].tcp[frame]
                tracks[name][frame] = pose_compose(tcp, grasp.as_list())
            else:
                tracks[name][frame] = record.pose.as_list()
    first = {name: Pose.from_list(track[0].tolist()) for name, track in tracks.items()}
    return PerceivedRecording(demo, tracks), PerceivedScene(dict(registry.objects), first)


def segment_of(
    keypoints: KeypointSet,
    demo: Demo,
    kind: SegmentKind,
    end: str,
    start: str | None,
    rest: str | None,
    obj: str | None,
    target: str | None,
    held_by_other: bool,
) -> FitSegment:
    """The segment the Agent names: its arm and end from `end`, its start from `start` (None:
    the arm's keypoint before, or the first frame), `rest` (None: the arm's keypoint after a
    place or a carry, the end otherwise); `closed`: no open of the arm's gripper from its last
    grip at or before the start to the end, that grip a close."""
    arm, last = find(keypoints, end)
    kps = keypoints.arms[arm]
    if start is None:
        before = [kp.frame for kp in kps if kp.frame < last.frame]
        first = before[-1] if before else 0
    else:
        side, kp = find(keypoints, start)
        if side != arm:
            raise ValueError(f"{start} is the {side} arm's; the segment is the {arm} arm's")
        first = kp.frame
    if first >= last.frame:
        raise ValueError(f"the segment starts at frame {first}, not before its end {last.frame}")
    if rest is not None:
        settled = find(keypoints, rest)[1].frame
    elif kind in ("place", "carry"):
        after = [kp.frame for kp in kps if kp.frame > last.frame]
        settled = after[0] if after else last.frame
    else:
        settled = last.frame
    grips = [kp for kp in kps if kp.grip and kp.frame <= first]
    opened = any(kp.grip and kp.cue != "close" and first < kp.frame <= last.frame for kp in kps)
    closed = bool(grips) and grips[-1].cue == "close" and not opened
    return FitSegment(
        arm=arm,
        kind=kind,
        start=first,
        end=last.frame,
        tcp=Pose.from_list(demo.arms[arm].tcp[last.frame].tolist()),
        obj=obj,
        rest=settled,
        target=target,
        handover=kind == "grasp" and held_by_other,
        closed=closed,
    )


def shifted(
    recording: PerceivedRecording, scene: PerceivedScene, segment: FitSegment,
    offset: Sequence[float],
) -> tuple[PerceivedRecording, PerceivedScene, FitSegment]:  # fmt: skip
    """The demonstration moved by `offset` (m) relative to the robot, as the P1 gate moves it
    (`tools.skill_fit.SHIFTS_M`): the objects' poses, each arm's ee and TCP path, the cameras
    and the segment's TCP; the joints and the robot base stay put."""
    d = np.array([*offset, 0.0, 0.0, 0.0, 0.0])

    def move(pose: Pose) -> Pose:
        x, y, z = np.add(pose.p, offset).tolist()
        return Pose(p=(x, y, z), q=pose.q)

    demo = recording.demo
    arms = {arm: replace(t, ee=t.ee + d, tcp=t.tcp + d) for arm, t in demo.arms.items()}
    calib = {}
    for name, c in demo.calib.items():
        cam2world = np.array(c.cam2world, dtype=float)
        cam2world[:, :3, 3] += np.asarray(offset, dtype=float)
        calib[name] = replace(c, cam2world=cam2world)
    objects = {name: track + d for name, track in recording.objects.items()}
    records = {
        name: record.model_copy(update={"pose": move(record.pose)})
        for name, record in scene.records.items()
    }
    return (
        PerceivedRecording(replace(demo, arms=arms, calib=calib), objects),
        PerceivedScene(records, {name: move(pose) for name, pose in scene.poses.items()}),
        replace(segment, tcp=move(segment.tcp)),
    )


class Rehearsal:
    """The action API as a written skill's `act` reads it (`profile`, `base` and the core
    entries that build motions), on the demonstration at the segment's start: each entry builds
    what it would (a displacement from where the arm's ee was then) and nothing moves. An entry
    it does not model raises `AttributeError`, and the fit's residual is then unknown."""

    def __init__(self, profile: EmbodimentProfile, demo: Demo, frame: int) -> None:
        self.profile = profile
        self._ee = {arm: Pose.from_list(t.ee[frame].tolist()) for arm, t in demo.arms.items()}

    @property
    def base(self) -> Pose:
        return robot_base(self.profile)

    async def move_to_pose(self, arm: Arm, pose: Pose) -> Built:
        return Built(arm, (Move(pose),))

    async def move_by(
        self, arm: Arm, x: float = 0.0, y: float = 0.0, z: float = 0.0,
        quat: Sequence[float] | None = None, move_axis: Any = "world",
    ) -> Built:  # fmt: skip
        ee_to_tcp = self.profile.arms[arm].ee_to_tcp
        target = displaced(self._ee[arm], (x, y, z), quat, move_axis, ee_to_tcp)
        return Built(arm, (Move(target, straight=True),))

    def open_gripper(self, arm: Arm, pos: float = 1.0) -> Built:
        return Built(arm, (Grip(pos),))

    def close_gripper(self, arm: Arm, pos: float = 0.0) -> Built:
        return Built(arm, (Grip(pos),))

    async def return_home(self, arm: Arm) -> Built:
        return Built(arm, ())  # home: not a waypoint of the segment

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"the rehearsal does not model api.{name}")


def residual_mm(
    built: Sequence[Built], profile: EmbodimentProfile, demo: Demo, segment: FitSegment,
    keys: Sequence[int],
) -> float | None:  # fmt: skip
    """How far the motions `built` and the demonstration lie apart over the segment, in mm: the
    larger of the farthest commanded TCP waypoint from its arm's demonstrated TCP track (frames
    `start` to `rest`) and the farthest of the segment arm's demonstrated keypoints (`keys`,
    frames) from the waypoints (a symmetric Hausdorff distance between waypoints and track); None
    when nothing moves the segment's arm."""
    last = max(segment.end, segment.rest or segment.end)
    waypoints: dict[Arm, list[np.ndarray]] = {}
    for b in built:
        for step in b.segments:
            if isinstance(step, Move):
                tcp = profile.tcp_pose(b.arm, step.target)
                waypoints.setdefault(b.arm, []).append(np.asarray(tcp.p))
    if segment.arm not in waypoints:
        return None
    far = 0.0
    for arm, points in waypoints.items():
        track = demo.arms[arm].tcp[segment.start : last + 1, :3]
        for point in points:
            far = max(far, float(np.min(np.linalg.norm(track - point, axis=1))))
    ours = np.asarray(waypoints[segment.arm])
    for frame in keys:
        point = demo.arms[segment.arm].tcp[frame, :3]
        far = max(far, float(np.min(np.linalg.norm(ours - point, axis=1))))
    return far * 1000.0


async def rehearse(
    name: str, act: Actor | None, fit: SkillFit, segment: FitSegment,
    recording: PerceivedRecording, scene: PerceivedScene, profile: EmbodimentProfile,
    keys: Sequence[int],
) -> SkillFit:  # fmt: skip
    """`fit` with its residual: the skill's `act` with the fitted params on the scene at the
    segment's start (`Rehearsal`), its motions against the demonstration (`residual_mm`)."""
    if fit.params is None:
        return fit
    if act is None:
        return fit.model_copy(update={"rehearsal": "the skillset gives no act to rehearse"})
    at = scene.at(recording.poses(segment.start), {})

    def proxy(obj: str | None) -> ObjectProxy | None:
        return at.proxy(obj) if obj is not None and obj in at.poses else None

    api = Rehearsal(profile, recording.demo, segment.start)
    try:
        with anyio.fail_after(REHEARSAL_S):
            built = await act(api, segment.arm, proxy(segment.obj), proxy(segment.target),
                              dict(fit.params))  # fmt: skip
        if not isinstance(built, list) or not all(isinstance(b, Built) for b in built):
            raise TypeError(f"act returned {type(built).__name__}, not a list of Built")
        found = residual_mm(built, profile, recording.demo, segment, keys)
    except Exception as error:  # the skill's own code: its failure is the rehearsal's answer
        why = f"its act cannot be rehearsed here: {type(error).__name__}: {error}"[:300]
        return fit.model_copy(update={"rehearsal": why})
    if found is None or not math.isfinite(found):
        return fit.model_copy(update={"rehearsal": "its act moves nothing of the segment's arm"})
    return fit.model_copy(update={"residual_mm": round(found, 1)})


def ranked(fits: Sequence[SkillFit]) -> list[SkillFit]:
    """`fits` with each measured one's rank by residual (1 the closest), in their order."""
    order = sorted(
        (fit.residual_mm, k) for k, fit in enumerate(fits) if fit.residual_mm is not None
    )
    ranks = {k: n + 1 for n, (_, k) in enumerate(order)}
    return [fit.model_copy(update={"rank": ranks.get(k)}) for k, fit in enumerate(fits)]


def run_fit(
    name: str, fitter: Fitter, segment: FitSegment, recording: PerceivedRecording,
    scene: PerceivedScene, base: Pose, conventions: Any,
) -> SkillFit:  # fmt: skip
    """One fitter on the segment; what it raises is the reason it gives no fit (it read more than
    perception gives, or the segment is not its kind)."""
    try:
        params = fitter(segment, recording, scene, base, conventions)
    except Exception as error:  # the skill's own code: its failure is its answer, reported
        what = f"{type(error).__name__}: {error}"[:300]
        return SkillFit(
            skill=name, why=f"its fit cannot run on the perceived demonstration: {what}"
        )
    if params is None:
        return SkillFit(skill=name, why="it does not apply to this segment")
    try:
        plain = json.loads(json.dumps(to_jsonable_python(params)))
    except (TypeError, ValueError) as error:
        return SkillFit(skill=name, why=f"its params are not plain data: {error}")
    if not isinstance(plain, dict):
        return SkillFit(skill=name, why="its params are not an object")
    fitted = _rounded(plain)
    again = []
    for offset in SHIFTS_M:
        moved_recording, moved_scene, moved_segment = shifted(recording, scene, segment, offset)
        try:
            other = fitter(moved_segment, moved_recording, moved_scene, base, conventions)
            again.append(_rounded(json.loads(json.dumps(to_jsonable_python(other)))))
        except Exception:  # a refit that raises breaks the rule, as one that returns None
            again.append(None)
    if absolute := absolute_numbers(fitted, again):
        shown = ", ".join(absolute[:4]) + (", ..." if len(absolute) > 4 else "")
        return SkillFit(
            skill=name,
            why=f"its params hold an absolute position ({shown} changed when the demonstration "
            f"was moved relative to the robot). {RELATIVE_RULE}",
        )
    return SkillFit(skill=name, params=fitted)


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DIGITS) + 0.0  # no -0.0
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


class SkillFitting(ToolsetExtension):
    """`fit_skill`, on the `demo` toolset."""

    extends = "demo"
    host: "DemoToolset"

    @tool(kind="pure", tier="D", render=render_fits)
    async def fit_skill(
        self,
        ctx: ToolContext,
        end: Annotated[
            str, Field(pattern=NAME, description="The segment's last keypoint, e.g. left K3.")
        ],
        kind: Annotated[
            SegmentKind,
            Field(
                description="What ends the segment: grasp (a close that holds obj), touch (a "
                "close or contact that lifts nothing), place (obj let go of at target), carry "
                "(obj moved and kept), release (obj given to the other arm), move (no event), "
                "home (back to rest)."
            ),
        ],
        start: Annotated[
            str | None,
            Field(
                pattern=NAME,
                description="The keypoint it starts at (the same arm's); null: that arm's "
                "keypoint before end, or the recording's first frame.",
            ),
        ] = None,
        obj: Annotated[
            ObjectRef | None,
            Field(description="The registered object grasped, touched, placed or carried."),
        ] = None,
        target: Annotated[
            ObjectRef | None,
            Field(description="At a place or carry, the registered object it ends at or on."),
        ] = None,
        rest: Annotated[
            str | None,
            Field(
                pattern=NAME,
                description="Where a placed or carried object rested (a keypoint); null: the "
                "arm's keypoint after end, or end.",
            ),
        ] = None,
        skill: Annotated[
            str | None,
            Field(
                description="One written skill to fit; null: every skill that has a fit, ranked."
            ),
        ] = None,
    ) -> SkillFits:
        """Fit written skills' parameters on one segment of the demonstration: the motions of
        one arm from start to end, ending in the given kind of event, as each skill would
        reproduce them relative to the objects involved. Each skill answers with its params and
        its residual (how far its fitted call's waypoints lie from the demonstrated motion), the
        fits ranked closest first, or why none (it does not apply, its fit needs what the
        demonstration's perception does not give, or it would carry absolute positions). Choose
        by residual. Define the params you use as a skill_spec parameter, fitted at end; the
        workflow's stage then calls run_skill with that skill and the parameter's name."""
        fitters = self._fitters()
        if skill is not None and skill not in fitters:
            raise ToolError(
                f"no written skill {skill!r} has a fit; those with one: {sorted(fitters)}"
            )
        fact, tracks_fact = ctx.frames.fact(DEMO), ctx.frames.fact(DEMO_TRACKS)
        if fact is None:
            raise ToolError("this episode has no demonstration")
        if tracks_fact is None:
            raise ToolError(
                "the demonstration's package has no recorded tracks (built by an older version): "
                "rebuild it to fit written skills"
            )
        if ctx.context.embodiment is None:
            raise ToolError("the episode names no embodiment, so its grippers are unknown")
        keypoints = KeypointSet.model_validate(fact)
        blob = BlobRef.model_validate(tracks_fact["blob"])
        demo = DemoTracks.model_validate_json(await ctx.blobs.aget_bytes(blob)).demo()
        registry = await ctx.frames.registry()
        for name in (obj, target):
            if name is not None and name not in registry.objects:
                raise ToolError(f"{name} is not registered in this episode")
        profile = embodiment(ctx.context.embodiment)
        robots = {}
        for frame in frames(keypoints):
            try:
                robots[frame] = (await ctx.frames.get(frame_alias(frame))).robot
            except LookupError:
                raise ToolError(f"the demonstration's {frame_alias(frame)} is missing") from None
        table = stored_table(ctx)
        released = None
        if self.host.scene is not None:  # measured on its frames, as demo_relation does
            released = await demo_releases(
                self.host.scene, ctx, keypoints, registry, robots, profile, table
            )
        events = sorted({kp.frame for kps in keypoints.arms.values() for kp in kps if kp.grip})
        snapshots = []
        for first in [0, *(frame + 1 for frame in events)]:
            at = registry.clone()
            replay_grips(keypoints, at, robots, profile, table, before=first, released=released)
            snapshots.append((first, at))
        try:
            arm = find(keypoints, end)[0]
            other: Arm = "right" if arm == "left" else "left"
            end_frame = find(keypoints, end)[1].frame
            starts = [first for first, _ in snapshots]
            then = snapshots[bisect.bisect_right(starts, end_frame) - 1][1]
            held = obj is not None and other in then.objects[obj].grasps
            segment = segment_of(keypoints, demo, kind, end, start, rest, obj, target, held)
        except (LookupError, ValueError) as error:
            raise ToolError(str(error)) from None
        recording, scene = perceived(keypoints, registry, snapshots, demo)
        base = robot_base(profile)
        chosen = {skill: fitters[skill]} if skill is not None else fitters
        last = max(segment.end, segment.rest or segment.end)
        keys = sorted(
            {segment.end}
            | {kp.frame for kp in keypoints.arms[segment.arm] if segment.start < kp.frame <= last}
        )
        fits = []
        for name, entry in chosen.items():
            fit = await anyio.to_thread.run_sync(
                run_fit, name, entry.fit, segment, recording, scene, base, profile.poses
            )
            fits.append(
                await rehearse(name, entry.act, fit, segment, recording, scene, profile, keys)
            )
        fits = ranked(fits)
        known = dict(ctx.frames.fact(SKILL_FITS) or {})
        entries = [
            {"skill": fit.skill, "params": fit.params, "segment": segment.name}
            for fit in fits
            if fit.params is not None
        ]
        known[end] = [*known.get(end, []), *entries]
        ctx.frames.set_fact(SKILL_FITS, known)
        ctx.emit("skill_fit", {"segment": segment.name, "fits": [f.model_dump() for f in fits]})
        return SkillFits(segment=segment.name, end=end, fits=fits)

    def _fitters(self) -> dict[str, WrittenFit]:
        found: dict[str, WrittenFit] = {}
        for ref in self.host.config.fitters:
            for name, fitter in load_skillset(ref).fitters.items():
                if name in found:
                    raise ToolError(f"two skillsets fit a skill named {name!r}")
                found[name] = fitter
        if not found:
            raise ToolError("no mounted skillset has written skills with a fit")
        return found


SKILL_FITTING: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="skill_fitting",
        version="1",
        layer="benchmark",
        skills=[
            skill(spec, ["parameters", "perception", "receipt"]) for spec in SkillFitting.specs()
        ],
    ),
    extensions=(SkillFitting,),
)
"""The skillset that adds `fit_skill` to `demo` (by import path:
`roboscript_methods.wicl.toolsets.skill_fit:SKILL_FITTING`)."""
