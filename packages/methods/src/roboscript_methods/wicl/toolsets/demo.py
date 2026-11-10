"""The `demo` toolset (docs/concepts/learning.md, docs/concepts/agent-runtime.md): S1's
tools on the demonstration, for the Demo Reader and hinted attempts (the runner leaves the toolset
out of unhinted ones).

`demo_relation` expresses the expert's TCP at a keypoint in the frame of an object or part
registered in this episode, as the grasp or place spec that reproduces it, with what the spec does
not reproduce (`roboscript_methods.wicl.relation`). It reads the demonstration's keypoints (the
episode's `DEMO` fact, whose TCP poses no model is shown), the robot at the recording's frames
(this episode's `frame_<n>`) and the scene registry, whose objects this episode identified on the
live scene of the demo seed, the scene the demonstration started from; never object ground truth.
Specs come back rounded to 0.1 mm and 0.1 deg, in the form the Memory's parameters take.
"""

import contextlib
import functools
import json
from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.demos import Keypoint, KeypointSet, frame_alias
from roboscript.manipulation import (
    GraspFit,
    GraspSpec,
    HingeFit,
    HingeSpec,
    PathFit,
    PathSpec,
    PlaceFit,
    PlaceSpec,
)
from roboscript.manipulation.api import robot_base
from roboscript.manipulation.specs import Name
from roboscript.memory import EditOp, ranks
from roboscript.memory.edits import DefineParam
from roboscript.memory.params import CategoryDef, NumberDef, SkillSpecDef
from roboscript.perception.registry import (
    ObjectRecord,
    SceneRegistry,
    object_geometry,
    start_geometry,
)
from roboscript.tools import (
    DEMO,
    DEMO_FITS,
    ObjectRef,
    ToolContext,
    Toolset,
    ToolsetRuntime,
    tool,
)
from roboscript.toolsets.scene import SceneAccess, SceneConfig, embodiment, stored_table
from roboscript_methods.wicl.relation import (
    NAME,
    Released,
    contacts,
    frames,
    relate,
    reproducing,
    with_contacts,
)
from roboscript_methods.wicl.releases import demo_releases
from roboscript_methods.wicl.toolsets.skill_fit import SKILL_FITS

_FUNCTIONAL = r"^(bottom|top|center|part:[a-z][a-z0-9_]*)$"


class Fitted(BaseModel):
    """One spec a relation fits: its form (a grasp's `pose`, the TCP in the whole object's
    frame, first; `plain`, a grasp's or placement's plain form, then `grasp`, `hinge`, or a
    placement's `target`, `world_axes` or `robot_frame`; `path`, the held path from the close),
    the spec, the grasp candidate it matched (e.g. `top/90`), a hinge turn's angle and straight
    moves, a held path's TCP travel and turn, and what it does not reproduce."""

    form: str
    spec: GraspSpec | PlaceSpec | HingeSpec | PathSpec
    variant: str | None = None
    angle_deg: float | None = None
    steps: int | None = None
    travel_m: float | None = None
    position_error_m: float
    angle_error_deg: float
    consistent: bool = True
    tcp_gaps: dict[str, tuple[float, float]] = {}


class DemoRelation(BaseModel):
    """The relation at a keypoint: a grasp of `obj`, a placement of the object the arm holds
    (`held`) at `obj` (null: at the robot base only), a turn of `obj`'s `part` about a hinge, or
    a retreat (`move_m`: the world displacement from the arm's keypoint before); `then`: the
    arm's next keypoint, when it is a retreat; `distances`: at a grasp's close or a placement's
    release, the action's approach distances as the demonstration moved (`approach_m` and
    `grasp_offset_m`, or `approach_m` and `stop_back_m`: the straight move from the keypoint
    before, and 0 at
    the keypoint, where the spec is fitted); `push_m`: after a press, the world displacement the
    closed gripper then made; `handover`: at a release of what the other arm holds too, that
    arm's close (a hand-over: place without opening, the other arm's grasp, then the open);
    `release`: on a hold's way to its open, why no placement is fitted here (the held path only)."""

    keypoint: str
    phase: str
    relation: Literal["grasp", "place", "turn", "retreat"]
    obj: ObjectRef | None
    held: ObjectRef | None = None
    part: str | None = None
    fits: list[Fitted]
    move_m: tuple[float, float, float] | None = None
    then: str | None = None
    distances: dict[str, float] = {}
    push_m: tuple[float, float, float] | None = None
    evidence: str = ""
    lookalikes: dict[str, tuple[int, int]] | None = None
    handover: str | None = None
    release: str | None = None


def render_relation(result: DemoRelation) -> list[ContentPart]:
    at = f"{result.keypoint} ({result.phase})"
    then = (
        f" Then {result.then}: ask demo_relation for it (the arm moving clear)."
        if result.then
        else ""
    )
    if result.relation == "retreat":
        assert result.move_m is not None  # a retreat has it
        x, y, z = result.move_m
        text = (
            f"{at}: the gripper moved clear of what it released by ({x:.3f}, {y:.3f}, {z:.3f}) m "
            "in the world (move_by x, y, z), before anything else near it."
        )
        return [TextPart(text=text)]
    if result.relation == "grasp":
        head = f"{at}: the gripper relative to {result.obj}, as a grasp spec"
    elif result.relation == "turn":
        head = f"{at}: the gripper turned about a hinge of {result.obj}'s {result.part}, as a "
        head += "hinge spec with its angle"
    elif result.relation == "place" and result.part is not None and result.held is None:
        head = f"{at}: {result.obj}'s {result.part}, which the arm holds, moved without a turn"
    elif result.release is not None:
        head = f"{at}: {result.held}, which the arm holds, carried here"
    else:
        where = f"at {result.obj}" if result.obj else "at the robot base"
        head = f"{at}: {result.held}, which the arm holds, placed {where}, as place specs"
    if any(isinstance(fit.spec, PathSpec) for fit in result.fits):
        head += ", and the held path from the close as a path spec"
    first = "pose first" if result.relation == "grasp" else "plain first"
    if result.relation == "place" and result.release is None:
        head += f"; {result.evidence}: the forms it bears out are marked consistent"
        first = "consistent first"
    lines = [head + f" (what each does not reproduce in brackets; {first}):"]
    for fit in result.fits:
        variant = f", candidate {fit.variant}" if fit.variant else ""
        if isinstance(fit.spec, PathSpec):
            variant += (
                f"; {len(fit.spec.poses)} poses, the TCP travels {fit.travel_m or 0.0:.3f} m and "
                f"turns {fit.angle_deg or 0.0:.1f} deg: follow_path carries what the arm holds "
                "along it"
            )
        elif result.relation == "place":
            variant += "; consistent" if fit.consistent else "; not borne out"
            variant += _commanded(fit.tcp_gaps)
        if fit.angle_deg is not None and not isinstance(fit.spec, PathSpec):
            variant += f"; angle_deg {fit.angle_deg:.1f} in {fit.steps} steps"
        spec = fit.spec.model_dump_json(exclude_defaults=True)
        off = f"{fit.position_error_m * 1000:.1f} mm, {fit.angle_error_deg:.1f} deg{variant}"
        lines.append(f"- {fit.form}: {spec} ({off})")
    if result.lookalikes:
        shown = ", ".join(f"{r} of {n} by {key}" for key, (r, n) in result.lookalikes.items())
        lines.append(
            f"- among the objects that look like it: {shown} (an OBJECTS order: line keeps "
            "this one where the instruction cannot tell them apart)"
        )
    if result.push_m is not None:
        x, y, z = result.push_m
        lines.append(f"- then the press: move_by ({x:.3f}, {y:.3f}, {z:.3f}) m in the world")
    if result.distances:
        shown = ", ".join(f"{name} {value:.3f}" for name, value in result.distances.items())
        along = next(
            (
                " (approach_m along the place specs' approach: the line the expert came in along, "
                "off place_object's default line from the gripper to the held object)"
                for fit in result.fits
                if isinstance(fit.spec, PlaceSpec) and fit.spec.approach is not None
            ),
            "",
        )
        lines.append(f"- distances (m, number parameters for the action): {shown}{along}")
    if result.release is not None:
        lines.append(f"- no placement here: {result.release}")
    if result.handover is not None:
        other = result.handover.split()[0]
        mine = result.keypoint.split()[0]
        lines.append(
            f"- a hand-over: at this open the {other} arm already holds {result.held} (it closed "
            f"at {result.handover}, before this open). Write it as three stages in this order: "
            f"place_object on the {mine} arm with is_open false (the placement above, which ends "
            f"where this open happens), the {other} arm's grasp ({result.handover}), then "
            f"open_gripper on the {mine} arm, which moves nothing; a place_object with is_open "
            f"true here would open before the {other} arm holds it, and a place or retreat "
            f"after its grasp would pull it out of the {other} gripper"
        )
    return [TextPart(text="\n".join(lines) + then)]


UNREPRODUCED: Final = "unreproduced"
"""The key a `DEMO_FITS` entry of a placement no `constrain` reproduces carries: the gaps, which
`define_param` names when it refuses the spec (`toolsets.memory`)."""


def fit_spec(fit: Mapping[str, Any]) -> dict[str, Any]:
    """A `DEMO_FITS` entry's spec, without what `demo_relation` noted beside it (`UNREPRODUCED`):
    what validates as the spec."""
    return {key: value for key, value in fit.items() if key != UNREPRODUCED}


def _known(fit: Fitted) -> dict[str, object]:
    """A fit as `DEMO_FITS` keeps it: the spec, and for a placement that `place_object` reproduces
    under no `constrain`, why (`UNREPRODUCED`)."""
    entry: dict[str, object] = fit.spec.model_dump(mode="json")
    if fit.tcp_gaps and not reproducing(fit.tcp_gaps):
        shown = ", ".join(f"{c} {100 * m:.1f} cm {d:.0f} deg" for c, (m, d) in fit.tcp_gaps.items())
        entry[UNREPRODUCED] = shown
    return entry


def _commanded(gaps: dict[str, tuple[float, float]]) -> str:
    """What place_object would command with a placement, against the expert's TCP, by
    constrain (`demos.relation.placed_gaps`), and the constrain that reproduces it."""
    if not gaps:
        return ""
    shown = ", ".join(f"{c} {100 * m:.1f} cm {d:.0f} deg" for c, (m, d) in gaps.items())
    fine = reproducing(gaps)
    if not fine:
        advice = (
            "no constrain reproduces it: define another form (its yaw form), or the object's "
            "frame is off"
        )
    elif "auto" in fine:
        advice = "the default constrain reproduces it"
    else:
        advice = f"pass constrain {fine[0]} in the call: the default, auto, does not reproduce it"
    return f"; place_object's TCP at stop_back_m 0 off the expert's: {shown} ({advice})"


TRUTH_MATCH_M: Final = 0.05
"""How far a registered object's center may be from the simulator's object it is fitted on."""


class DemoConfig(BaseModel):
    """`fit`: the geometry `demo_relation` fits specs on: `perceived`, the scene registry (the
    coordinates S2 uses), or `truth`, each registered object moved onto its simulator object's
    center (tier P, the `privileged` toolset; an experiment's comparison); `fit_approach`: a
    placement's fits carry the line the expert came in along onto the release where
    `place_object`'s default line is off it (`demos.relation.with_approach`; false: none does);
    `scene`: the scene toolsets' config, with which a release in the demonstration
    is measured again on its frame before a later keypoint is related
    (`releases.demo_releases`, on the scene registry, as `check_done` measures it; under
    `truth`, moved with its object; null: what an arm let go of stays where its grasp left
    it); `fitters`: the skillsets (registered names or import paths) whose written skills'
    fits `fit_skill` runs, where the `skill_fitting` skillset adds it (`toolsets.skill_fit`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fit: Literal["perceived", "truth"] = "perceived"
    fit_approach: bool = True
    scene: SceneConfig | None = None
    fitters: list[str] = []


class Demo(Toolset):
    """The demonstration's tools."""

    name = "demo"
    Config = DemoConfig
    config: DemoConfig
    scene: SceneAccess | None = None

    async def start(self, runtime: ToolsetRuntime) -> None:
        if self.config.scene is not None:
            self.scene = SceneAccess(self.config.scene, runtime)

    async def stop(self) -> None:
        if self.scene is not None:
            await self.scene.close()

    @tool(kind="pure", tier="D", render=render_relation)
    async def demo_relation(
        self,
        ctx: ToolContext,
        keypoint: Annotated[
            str,
            Field(pattern=NAME, description="A keypoint: its arm and number, e.g. left K2."),
        ],
        obj: Annotated[
            ObjectRef | None,
            Field(
                description="At a grasp, the object grasped; at a placement, the target object "
                "(null: the robot base)."
            ),
        ] = None,
        part: Annotated[
            Name | None,
            Field(description="A registered part of obj to relate to; null: the whole object."),
        ] = None,
        placement_point: Annotated[
            str,
            Field(
                pattern=_FUNCTIONAL,
                description="At a placement, the held object's point that is placed: bottom, "
                "top, center or part:<name>.",
            ),
        ] = "bottom",
    ) -> DemoRelation:
        """Relate the demonstration's gripper at a keypoint to a registered object or part, as
        the spec that reproduces it. Where the arm holds an articulated part at the keypoint, the
        hinge spec on that part's box and the angle the gripper turned since its close; where it
        holds an object (the recorded grasps carry it there), the place specs of that object at
        obj (from obj's anchor nearest its point, in obj's axes and the world's), or at the
        robot base with no obj; in both, the held path from the close to the keypoint (to the
        last before a placement's own move): a path spec for follow_path, which carries what the
        arm (or both arms) holds the way the demonstration did, arcs included; elsewhere the
        grasp spec of obj (a hand-over's receiving grasp where the other arm holds it; press
        where the close holds nothing). A release of what the other arm already holds is a
        hand-over, said with the order to write it in. Each says what it
        does not reproduce. At a retreat (the gripper leaving what it released), the move it
        made, which the workflow repeats before the other arm works near there. The
        demonstration started from this scene: its objects start where this episode first
        measured them."""
        fact = ctx.frames.fact(DEMO)
        if fact is None:
            raise ToolError("this episode has no demonstration")
        keypoints = KeypointSet.model_validate(fact)
        live = registry = await ctx.frames.registry()
        if self.config.fit == "truth":
            registry = await _on_truth(ctx, live.clone())
        if obj is not None and obj not in registry.objects:
            raise ToolError("that object is not registered in this episode")
        robots = {}
        for frame in frames(keypoints):
            try:
                robots[frame] = (await ctx.frames.get(frame_alias(frame))).robot
            except LookupError:
                raise ToolError(f"the demonstration's {frame_alias(frame)} is missing") from None
        if (final := keypoints.success_frame) is not None and final not in robots:
            with contextlib.suppress(LookupError):  # a held object's contact (`contacts`)
                robots[final] = (await ctx.frames.get(frame_alias(final))).robot
        if ctx.context.embodiment is None:
            raise ToolError("the episode names no embodiment, so its grippers are unknown")
        released = None
        if self.scene is not None:  # measured on the scene registry, as `check_done`'s are
            released = await demo_releases(
                self.scene, ctx, keypoints, live, robots, embodiment(ctx.context.embodiment),
                stored_table(ctx),
            )  # fmt: skip
            if registry is not live:
                released = _moved_with(released, live, registry)
        keypoints = with_contacts(keypoints, contacts(keypoints, robots))
        found = functools.partial(
            relate,
            keypoints,
            keypoint,
            registry,
            robots,
            embodiment(ctx.context.embodiment),
            stored_table(ctx),
            obj,
            part,
            placement_point,
            self.config.fit_approach,
            released,
        )
        try:
            relation = await anyio.to_thread.run_sync(found)
        except (LookupError, ValueError) as error:
            raise ToolError(str(error)) from None
        side, index = keypoint.split()[0], int(keypoint.split()[1].removeprefix("K"))
        after = keypoints.arms.get(side, [])[index : index + 1]  # type: ignore[call-overload]
        then = (
            f"{side} {after[0].name} (retreat)" if after and after[0].phase == "retreat" else None
        )
        if relation.retreat is not None:
            return DemoRelation(
                keypoint=keypoint,
                phase=relation.keypoint.phase,
                relation="retreat",
                obj=None,
                fits=[],
                move_m=tuple(round(v, 4) for v in relation.retreat),  # type: ignore[arg-type]
            )
        if relation.release is not None and relation.path is None:
            raise ToolError(_released(keypoint, relation.release))
        kind: Literal["grasp", "place", "turn"] = "place"
        evidence = ""
        slid = (  # a held part's move without a hinge turn: no grasp, placement or release
            relation.part is not None
            and relation.hinge is None
            and relation.held is None
            and relation.grasp is None
            and relation.release is None
        )
        if relation.hinge is not None:
            kind, fits = "turn", [as_fitted("hinge", relation.hinge)]
            if relation.path is not None:
                fits.append(as_fitted("path", relation.path))
        elif relation.grasp is not None:
            kind, fits = "grasp", [as_fitted("grasp", relation.grasp)]
            plain = None if relation.plain is None else as_fitted("plain", relation.plain)
            if plain is not None and plain.spec != fits[0].spec:  # else the grasp is plain
                fits.insert(0, plain)
            if relation.pose is not None:
                fits.insert(0, as_fitted("pose", relation.pose))
        elif relation.release is not None:
            assert relation.path is not None  # else refused above
            fits = [as_fitted("path", relation.path)]
        elif slid:  # a part the arm holds moved without a hinge turn (a drawer slid out)
            assert relation.path is not None  # else no relation
            fits = [as_fitted("path", relation.path)]
        else:
            fits = [as_fitted(form, fit) for form, fit in relation.place.items()]
            fits = [fit for i, fit in enumerate(fits) if fit.spec not in [f.spec for f in fits[:i]]]
            evidence = next((fit.evidence for fit in relation.place.values()), "")
            if relation.path is not None:
                fits.append(as_fitted("path", relation.path))
        alike = None
        if kind == "grasp" and relation.obj is not None:
            base = robot_base(embodiment(ctx.context.embodiment)).matrix()
            first = {o.id: start_geometry(o) for o in registry.objects.values()}
            starts = {o: g.center for o, g in first.items()}
            sizes = {o: g.size for o, g in first.items()}
            alike = ranks(starts, sizes, relation.obj, base[:3, 1], base[:3, 3])
        known = dict(ctx.frames.fact(DEMO_FITS) or {})
        known[keypoint] = [*known.get(keypoint, []), *(_known(fit) for fit in fits)]
        ctx.frames.set_fact(DEMO_FITS, known)
        return DemoRelation(
            distances=(
                {} if slid else _distances(keypoints.arms[relation.arm], relation.keypoint, kind)
            ),
            keypoint=keypoint,
            phase=relation.keypoint.phase,
            relation=kind,
            obj=relation.obj,
            held=relation.held,
            part=relation.part,
            fits=fits,
            then=then,
            push_m=None if relation.pushed is None else tuple(round(v, 4) for v in relation.pushed),  # type: ignore[arg-type]
            evidence=evidence,
            lookalikes=alike,
            handover=relation.handover,
            release=None if relation.release is None else _released(keypoint, relation.release),
        )


def _released(keypoint: str, release: str) -> str:
    """Why no placement is fitted at `keypoint`, a hold's keypoint before its open `release`."""
    arm = keypoint.split()[0]
    return (
        f"{keypoint} is on the way to the release: a placement is fitted where the gripper "
        f"opens, {arm} {release} (its approach_m is the straight move onto it); ask demo_relation "
        "there"
    )


def _distances(
    keypoints: list[Keypoint], kp: Keypoint, kind: Literal["grasp", "place", "turn"]
) -> dict[str, float]:
    """The approach distances of a grasp fitted at its close, or a placement at its release: the
    straight move from the arm's keypoint before (an approach or a place stop), and 0 at it."""
    names = {
        ("grasp", "close"): ("approach_m", "grasp_offset_m"),
        ("place", "open"): ("approach_m", "stop_back_m"),
    }
    pair = names.get((kind, kp.cue))
    before = keypoints[kp.index - 2] if kp.index > 1 else None
    if pair is None or before is None or before.grip:
        return {}
    distance = float(np.linalg.norm(np.subtract(kp.tcp.p, before.tcp.p)))
    return {pair[0]: _round(distance, 4), pair[1]: 0.0}


async def _on_truth(ctx: ToolContext, registry: SceneRegistry) -> SceneRegistry:
    """`registry` with each object as first measured moved onto the center of the simulator's
    object nearest it (within `TRUTH_MATCH_M`; its orientation and shape kept)."""
    try:
        found = await ctx.call("true_objects", centers=True)
    except PermissionError:
        raise ToolError("fitting on the truth needs a tier-P episode") from None
    except LookupError:
        raise ToolError("fitting on the truth needs the truth toolset mounted") from None
    centers = [np.asarray(obj.center) for obj in found.objects]  # type: ignore[attr-defined]
    for key, record in list(registry.objects.items()):
        first = record.history[0].pose if record.history else record.pose
        start = record.model_copy(update={"pose": first, "grasps": {}, "part_grasps": {}})
        here = object_geometry(start).pose[:3]
        nearest = min(centers, key=lambda c: float(np.linalg.norm(c - here)), default=None)
        if nearest is None or np.linalg.norm(nearest - here) > TRUTH_MATCH_M:
            continue
        p = np.asarray(first.p) + (nearest - here)
        moved = Pose(p=(float(p[0]), float(p[1]), float(p[2])), q=first.q)
        registry.objects[key] = record.model_copy(update={"pose": moved, "history": []})
    return registry


def _moved_with(released: Released, live: SceneRegistry, moved: SceneRegistry) -> Released:
    """`released` (measured on `live`) for `moved` (`_on_truth`'s copy): each pose shifted as
    its object's first pose was."""

    def first(record: ObjectRecord) -> np.ndarray:
        return np.asarray((record.history[0].pose if record.history else record.pose).p)

    out = {}
    for t, poses in released.items():
        out[t] = {}
        for o, pose in poses.items():
            shift = first(moved.objects[o]) - first(live.objects[o])
            p = np.asarray(pose.p) + shift
            out[t][o] = Pose(p=(float(p[0]), float(p[1]), float(p[2])), q=pose.q)
    return out


def as_fitted(form: str, fit: GraspFit | PlaceFit | HingeFit | PathFit) -> Fitted:
    if isinstance(fit, PathFit):  # rounded as fitted
        return Fitted(
            form=form,
            spec=fit.spec,
            angle_deg=_round(fit.turn_deg, 1),
            travel_m=_round(fit.travel_m, 4),
            position_error_m=fit.position_error_m,
            angle_error_deg=fit.angle_error_deg,
        )
    spec = fit.spec
    offset = tuple(_round(v, 4) for v in spec.offset_m)
    if isinstance(fit, HingeFit):
        return Fitted(
            form=form,
            spec=fit.spec.model_copy(update={"offset_m": offset}),
            angle_deg=_round(fit.angle_deg, 1),
            steps=fit.steps,
            position_error_m=fit.position_error_m,
            angle_error_deg=fit.angle_error_deg,
        )
    update: dict[str, object] = {"offset_m": offset, "yaw_deg": _round(spec.yaw_deg, 1)}
    if isinstance(spec, GraspSpec):
        update |= {"tilt_deg": _round(spec.tilt_deg, 1), "lean_deg": _round(spec.lean_deg, 1)}
        for axis in ("approach_axis", "closing_axis"):
            if (value := getattr(spec, axis)) is not None:
                update[axis] = tuple(_round(v, 4) for v in value)
    else:
        update["rpy_deg"] = tuple(_round(v, 1) for v in spec.rpy_deg)
    return Fitted(
        form=form,
        spec=spec.model_copy(update=update),
        variant=fit.variant if isinstance(fit, GraspFit) else None,
        position_error_m=fit.position_error_m,
        angle_error_deg=fit.angle_error_deg,
        consistent=fit.consistent if isinstance(fit, PlaceFit) else True,
        tcp_gaps={
            c: (_round(m, 4), _round(d, 1))
            for c, (m, d) in (fit.tcp_gaps if isinstance(fit, PlaceFit) else {}).items()
        },
    )


def _round(value: float, digits: int) -> float:
    return round(float(value), digits) + 0.0  # no -0.0


_LOOSE: Final = frozenset({"arm", "mirror", "x_toward", "region_m"})
"""A fitted spec's fields its definition may change (the arm and the search around it)."""


def check_fitted(ctx: ToolContext, op: EditOp) -> None:
    """wicl's edit check (`toolsets.memory.EditCheck`): with a demonstration in the episode,
    a spec is defined as `demo_relation` fitted it."""
    if not isinstance(op, DefineParam) or isinstance(op.param, NumberDef | CategoryDef):
        return
    if isinstance(op.param, SkillSpecDef):
        _check_skill_fitted(ctx, op.param, op.fitted)
        return
    if op.fitted is None:
        if ctx.frames.fact(DEMO) is not None:
            raise ToolError(
                "fit this spec on the demonstration first: call demo_relation at the keypoint "
                "where it happens and define the spec it returns, with fitted naming where"
            )
        return
    given = (ctx.frames.fact(DEMO_FITS) or {}).get(op.fitted, [])
    value = op.param.value.model_dump(mode="json")
    found = [fit for fit in given if _same(value, fit_spec(fit))]
    if not found:
        raise ToolError(
            f"demo_relation fitted no such spec at {op.fitted} in this episode: call it there and "
            f"define one of its specs as returned ({', '.join(sorted(_LOOSE))} may differ)"
        )
    if all(UNREPRODUCED in fit for fit in found):
        raise ToolError(
            f"place_object reproduces the expert's TCP at {op.fitted} with this placement under no "
            f"constrain ({found[0][UNREPRODUCED]} off): define a form demo_relation marks as "
            "reproduced (its yaw form under align), or the object's frame is off"
        )


def _check_skill_fitted(ctx: ToolContext, param: SkillSpecDef, fitted: str | None) -> None:
    """A written skill's parameters are defined as `fit_skill` fitted them where `fitted` says
    (`toolsets.skill_fit.SKILL_FITS`)."""
    if fitted is None:
        raise ToolError(
            "fit the skill's parameters on the demonstration first: call fit_skill on the "
            "segment it reproduces and define the params it returns, with fitted naming the "
            "segment's end keypoint"
        )
    given = (ctx.frames.fact(SKILL_FITS) or {}).get(fitted, [])
    if not any(
        fit["skill"] == param.skill and _same(json.loads(param.value), fit["params"], False)
        for fit in given
    ):
        raise ToolError(
            f"fit_skill fitted no such {param.skill} params at {fitted} in this episode: call it "
            "there and define its params as returned"
        )


def _same(a: object, b: object, top: bool = True) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - (_LOOSE if top else set())
        return all(_same(a.get(k), b.get(k), False) for k in keys)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, False) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= 1e-4
    return a == b
