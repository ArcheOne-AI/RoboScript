"""A try's ground-truth diagnosis in the try's own terms (docs/concepts/learning.md,
docs/architecture.md), made on the control plane as the try ends, while its registry is at hand
(`EpisodeDriver`, into `referee/explained.json`, through wicl's
`MethodRuntime.explain`, `explain_try`): the Reviser reads only what `explain` keeps, never the
simulator's names or a world coordinate.

Each task object the referee reports is matched to a registered object by their boxes at the
start (`match`): a pair whose boxes overlap, the smaller lying at least `CONTAINED` within the
other (a registration of a large object's front face, its center centimeters off the object's),
or whose centers lie within `MATCH_M` horizontally; the best overlap (IoU) first, then the
nearest. The referee's centers are the middle of the box around an object's geometry, as
perception's are of the box around its points; its box is its width square and its height
(`ObjectShape.size`, `referee/shapes.json`), a registered one the world box around its fit's box
where it was first measured. Without the referee's shapes, centers alone match.

Per task object: the registered object it is (its alias and role; one the try did not register by
the registered object nearest to it at the start), how far perception had it off at the start and,
once it moved, where the registry had it at the end (its grasp-following); how far its center moved
along the robot's axes (x right, y forward, z up) and how far its up direction turned (tipped
over); where it ends relative to each other registered task object; each joint's position at the
start and at the end as a fraction of its range. Registered objects that match no task object are
listed too (a wrong identity), and the env's notes,
which name no object.
"""

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.env import Diagnosis, ObjectTruth
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.robot import RobotState
from roboscript.demos import ObjectShape
from roboscript.learning.demo import DemoTruth
from roboscript.perception.registry import ObjectRecord, SceneRegistry, object_geometry
from roboscript_methods.wicl.extension import settings

MATCH_M: Final = 0.05
"""A task object is a registered one when their centers at the start lie this close,
horizontally."""
CONTAINED: Final = 0.5
"""Two boxes are one object when this share of the smaller one's volume lies in the other."""
MOVED_M: Final = 0.01
"""A task object moved when its center moved this far."""

Points = NDArray[np.float64]


class Named(BaseModel):
    """A registered object as the try's log names it: its alias and the role it was registered
    for."""

    model_config = ConfigDict(frozen=True)

    alias: str
    role: str


class Fate(BaseModel):
    """One task object after the try (see the module): `object` (None: the try did not register
    it; then `near`, the registered object nearest to it at the start, and how far); perception's
    error at the start (`start_off_m`) and, when it moved, the registry's at the end
    (`end_off_m`; None when the registry cannot tell); its center's displacement along the
    robot's axes (`moved`, meters); when it moved, its offset from each other registered task
    object at the end (`ends`); its joints' positions at the start and at the end, as fractions
    of their range."""

    model_config = ConfigDict(frozen=True)

    object: Named | None
    near: tuple[Named, float] | None = None
    start_off_m: float | None = None
    end_off_m: float | None = None
    moved: Vec3
    ends: list[tuple[Named, Vec3]] = []
    tilt_deg: float | None = None
    demo_tilt_deg: float | None = None
    """How far the demonstration turned the same object's up direction by its end (None: not
    known): a tilt near it is the task's, not a fall."""
    joints: list[tuple[float, float]] = []


class Explained(BaseModel):
    """What the Reviser reads of a try's ground truth: each task object's fate, the registered
    objects no task object matches (`unmatched`), and the env's notes."""

    model_config = ConfigDict(frozen=True)

    objects: list[Fate]
    unmatched: list[Named]
    notes: list[str]


def explain(
    diagnosis: Diagnosis,
    registry: SceneRegistry,
    alias: Callable[[str], str],
    robot: RobotState | None,
    shapes: Mapping[str, ObjectShape] | None = None,
    demo: Mapping[str, float] | None = None,
) -> Explained:
    """`diagnosis` in the terms of the try whose `registry` it was at the end (`alias`: the
    model's name of a registered object's id; `robot`: its state at the end, for objects still
    held; `shapes`: the referee's, by the simulator's names, for matching by box; `demo`: by the
    simulator's names, the tilt the demonstration ended each object with, `demo_tilts`)."""
    records = {record.id: record for record in registry.objects.values()}
    names = {obj: Named(alias=alias(obj), role=record.name) for obj, record in records.items()}
    starts = {obj: start_center(record) for obj, record in records.items()}
    centers = [(_center(t, final=False), _center(t, final=True)) for t in diagnosis.objects]
    boxes = [
        truth_box(start, (shapes or {}).get(t.name))
        for t, (start, _) in zip(diagnosis.objects, centers, strict=True)
    ]
    matched = match([start for start, _ in centers], boxes, records)
    fates = []
    for i, truth in enumerate(diagnosis.objects):
        start, end = centers[i]
        moved = end - start
        went = float(np.linalg.norm(moved)) >= MOVED_M
        obj = matched.get(i)
        near = None
        if obj is None and starts:
            nearest = min(starts, key=lambda o: _flat(start - starts[o]))
            near = (names[nearest], _flat(start - starts[nearest]))
        final = None if obj is None or not went else _end(records[obj], robot)
        others = [(names[o], _vec(end - centers[j][1])) for j, o in matched.items() if j != i]
        fates.append(
            Fate(
                object=None if obj is None else names[obj],
                near=near,
                start_off_m=None if obj is None else float(np.linalg.norm(start - starts[obj])),
                end_off_m=None if final is None else float(np.linalg.norm(end - final)),
                moved=_vec(moved),
                ends=others if went else [],
                tilt_deg=tilt(truth) if went else None,
                demo_tilt_deg=(demo or {}).get(truth.name) if went else None,
                joints=_fractions(truth),
            )
        )
    unmatched = [names[obj] for obj in records if obj not in matched.values()]
    return Explained(objects=fates, unmatched=unmatched, notes=list(diagnosis.notes))


def explain_try(
    diagnosis: Diagnosis,
    registry: SceneRegistry,
    alias: Callable[[str], str],
    robot: RobotState | None,
    shapes: Mapping[str, ObjectShape] | None,
    method: object | None,
) -> Explained:
    """`explain` with the tilts the demonstration (its ground-truth layer,
    `WiclEpisode.demo_end` of the episode's `method`) ended its objects with (`demo_tilts`):
    wicl's explainer (`MethodRuntime.explain`)."""
    demo = settings(method).demo_end
    tilts = demo_tilts(demo) if demo is not None else None
    return explain(diagnosis, registry, alias, robot, shapes, tilts)


Box = tuple[Points, Points]
"""An axis-aligned world box: its low and high corners."""


def match(
    truths: list[Points], boxes: list[Box | None], records: Mapping[str, ObjectRecord]
) -> dict[int, str]:
    """Task objects (by index; their centers and boxes at the start) to registered ones, one to
    one: pairs whose boxes overlap with the smaller `CONTAINED` in the other, or whose centers
    lie within `MATCH_M` horizontally; the largest IoU first, then the nearest."""
    starts = {obj: start_center(record) for obj, record in records.items()}
    registered = {obj: start_box(record) for obj, record in records.items()}
    pairs = []
    for i, (t, box) in enumerate(zip(truths, boxes, strict=True)):
        for obj, s in starts.items():
            distance = _flat(t - s)
            iou, contained = (0.0, 0.0) if box is None else overlap(box, registered[obj])
            if distance <= MATCH_M or contained >= CONTAINED:
                pairs.append((-iou, distance, i, obj))
    matched: dict[int, str] = {}
    for _, _, i, obj in sorted(pairs):
        if i not in matched and obj not in matched.values():
            matched[i] = obj
    return matched


def truth_box(center: Points, shape: ObjectShape | None) -> Box | None:
    """A task object's box at the start: its width square and height about its center (None
    without its size)."""
    if shape is None or shape.size is None:
        return None
    width, height = shape.size
    half = np.array([width / 2, width / 2, height / 2])
    return center - half, center + half


def start_box(record: ObjectRecord) -> Box:
    """A registered object's world box where it was first measured: around its fit's box."""
    first = record.history[0].pose if record.history else record.fit.frame
    moved = first.matrix() @ np.linalg.inv(record.fit.frame.matrix())
    obb = record.fit.obb
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    local = signs * np.asarray(obb.size) / 2
    rotation = Pose(p=obb.center, q=obb.q).matrix()
    corners = local @ rotation[:3, :3].T + rotation[:3, 3]
    world = corners @ moved[:3, :3].T + moved[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def overlap(a: Box, b: Box) -> tuple[float, float]:
    """Two boxes' IoU and the share of the smaller one's volume inside the other."""
    low, high = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = float(np.prod(np.clip(high - low, 0.0, None)))
    va, vb = (float(np.prod(np.clip(box[1] - box[0], 0.0, None))) for box in (a, b))
    union = va + vb - inter
    smaller = min(va, vb)
    return (inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def _center(truth: ObjectTruth, *, final: bool) -> Points:
    """A task object's center, else its pose's position."""
    given = truth.final_center if final else truth.initial_center
    pose = truth.final if final else truth.initial
    return np.asarray(pose.p if given is None else given, dtype=float)


def start_center(record: ObjectRecord) -> Points:
    """A registered object's center as registered: its fit's center, in its first measured
    frame."""
    first = record.history[0].pose if record.history else record.fit.frame
    moved = first.matrix() @ np.linalg.inv(record.fit.frame.matrix())
    center: Points = moved[:3, :3] @ np.asarray(record.fit.center) + moved[:3, 3]
    return center


def _end(record: ObjectRecord, robot: RobotState | None) -> Points | None:
    """Where the registry has a registered object's center at the end; None for one still held
    without the robot's state."""
    if record.grasps and robot is None:
        return None
    return np.asarray(object_geometry(record, robot).center, dtype=float)


def tilt(truth: ObjectTruth) -> float:
    """How far a task object's up direction turned between its initial and final poses (deg)."""
    return _turn_deg(truth.initial, truth.final)


def demo_tilts(layer: DemoTruth) -> dict[str, float]:
    """By the simulator's name, how far the demonstration turned each object's up direction
    between its first and last frames (deg)."""
    if not layer.frames:
        return {}
    first, last = layer.frames[0].poses, layer.frames[-1].poses
    return {name: _turn_deg(first[name], last[name]) for name in first.keys() & last.keys()}


def _turn_deg(initial: Pose, final: Pose) -> float:
    turn = final.matrix()[:3, :3] @ initial.matrix()[:3, :3].T
    return float(np.degrees(np.arccos(np.clip(turn[2, 2], -1.0, 1.0))))


def fell(fate: Fate, tipped_deg: float) -> bool:
    """Whether an object's tilt says it fell over: at least `tipped_deg` from its start, and,
    when the demonstration's own end tilt is known, that far beyond it as well."""
    if fate.tilt_deg is None or fate.tilt_deg < tipped_deg:
        return False
    return fate.demo_tilt_deg is None or fate.tilt_deg - fate.demo_tilt_deg >= tipped_deg


def _fractions(truth: ObjectTruth) -> list[tuple[float, float]]:
    return [
        ((j.initial - j.low) / (j.high - j.low), (j.final - j.low) / (j.high - j.low))
        for j in truth.joints
        if j.high > j.low
    ]


def _flat(offset: Points) -> float:
    return float(np.linalg.norm(offset[:2]))


def _vec(v: Points) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))
