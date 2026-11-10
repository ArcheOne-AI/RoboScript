"""Composites over the primitives, every inner step logged as a `perception.step` event.

`locate`: a chosen proposal (or masks) -> lift across views -> fit -> register (at the start of
an episode: identification happens once, see `perception.registry`).
`refresh`: an explicit re-measurement: find a registered object among a frame's proposals ->
lift -> fit -> store (e.g. for a Reviser on logged frames; nothing calls it automatically).
`scene`: what the predicates read: the registry's objects (held ones where their TCP holds them),
the robot and the table. Association is one to one (Hungarian) on center distance plus half the
size difference, gated at 5 cm or half the object's length when that is larger.

These are plain functions doing numpy work: async callers run them in a worker thread.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.robot import RobotState
from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.fit import Shape
from roboscript.perception.lift import Lifted, lift
from roboscript.perception.predicates import Scene
from roboscript.perception.proposals import Proposal
from roboscript.perception.registry import (
    ObjectGeometry,
    ObjectRecord,
    SceneRegistry,
    Shown,
    Step,
    object_geometry,
    silent,
)
from roboscript.perception.table import Table
from roboscript.perception.views import FrameGeometry
from roboscript.records import EventWriter

GATE = 0.05


@dataclass(frozen=True)
class StepLog:
    """Writes one `perception.step` event per inner step, `{"step": ..., **data}`: the
    composites' steps here and the registry's grasp-relation decisions (`attach`, `detach`,
    `settle`)."""

    events: EventWriter
    episode_id: EpisodeId | None = None

    def __call__(self, step: str, **data: Any) -> None:
        self.events.emit(
            EventKind.PERCEPTION_STEP, {"step": step, **data}, episode_id=self.episode_id
        )


def locate(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    name: str,
    chosen: Proposal | Mapping[str, NDArray[np.bool_]],
    *,
    config: PerceptionConfig = DEFAULTS,
    shape: Shape | Literal["auto"] = "auto",
    confidence: float = 1.0,
    ambiguous: bool = False,
    candidates: Sequence[Shown] = (),
    log: Step | None = None,
) -> ObjectRecord:
    """Register the object a chosen proposal (all its views) or masks (camera -> mask) show
    (`ambiguous`, with the `candidates` the identification's backends showed)."""
    step: Step = log or silent
    masks = chosen.masks if isinstance(chosen, Proposal) else chosen
    lifted = _lift(geometry, masks, config, step)
    record = registry.register(
        name,
        lifted,
        frame=geometry.frame,
        step=geometry.step,
        table_z=geometry.table.z,
        shape=shape,
        confidence=confidence,
        ambiguous=ambiguous,
        candidates=candidates,
        base=geometry.base,
        frames=config.frames,
        center_on_close=config.center_on_close,
    )
    _log_fit(step, record)
    step("register", object=record.id, name=name, confidence=confidence)
    return record


def refresh(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    obj_id: str,
    proposals: list[Proposal],
    *,
    robot: RobotState | None = None,
    config: PerceptionConfig = DEFAULTS,
    log: Step | None = None,
) -> ObjectRecord:
    """Re-measure a registered object in another frame (an explicit call: nothing refreshes
    automatically); a held one needs `robot` (its TCP)."""
    step: Step = log or silent
    old = registry.get(obj_id)
    expected = object_geometry(old, robot)
    found = associate({obj_id: expected}, proposals).get(obj_id)
    if found is None:
        step("associate", object=obj_id, frame=geometry.frame, region=None)
        raise LookupError(
            f"{obj_id} is not where it was in frame {geometry.frame}: locate it again"
        )
    proposal, cost = found
    step("associate", object=obj_id, frame=geometry.frame, region=proposal.id, cost=cost)
    lifted = _lift(geometry, proposal.masks, config, step)
    record = registry.refresh(
        obj_id,
        lifted,
        frame=geometry.frame,
        step=geometry.step,
        table_z=geometry.table.z,
        robot=robot,
        frames=config.frames,
    )
    _log_fit(step, record)
    return record


def scene(registry: SceneRegistry, robot: RobotState, table: Table) -> Scene:
    """What the predicates read: the registered objects as the registry has them (held ones
    where their TCP holds them; no re-detection), the robot and the table."""
    objects = {i: object_geometry(r, robot) for i, r in registry.objects.items()}
    return Scene(robot=robot, table=table, objects=objects, records=registry.objects)


def associate(
    expected: Mapping[str, ObjectGeometry], proposals: list[Proposal]
) -> dict[str, tuple[Proposal, float]]:
    """One proposal per object (and object per proposal) with its cost, for the objects that
    have one within the gate."""
    ids = list(expected)
    cost = np.full((len(ids), len(proposals)), np.inf)
    for i, obj_id in enumerate(ids):
        obj = expected[obj_id]
        gate = max(GATE, 0.5 * float(obj.size[0]))
        for j, p in enumerate(proposals):
            offset = np.asarray(p.summary.center) - obj.center
            distance = float(np.linalg.norm(offset if obj.held_by else offset[:2]))
            if distance <= gate:
                sizes = np.abs(np.sort(obj.size) - np.sort(p.summary.size))
                cost[i, j] = distance + 0.5 * float(sizes.sum())
    if not cost.size:
        return {}
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
    return {
        ids[r]: (proposals[c], float(cost[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if np.isfinite(cost[r, c])
    }


def _lift(
    geometry: FrameGeometry,
    masks: Mapping[str, NDArray[np.bool_]],
    config: PerceptionConfig,
    step: Step,
) -> Lifted:
    lifted = lift(geometry, masks, config)
    pixels = {camera: int(np.count_nonzero(m)) for camera, m in masks.items()}
    if lifted is None:
        step("lift", frame=geometry.frame, pixels=pixels, points=0)
        raise LookupError(
            f"nothing to lift in frame {geometry.frame}: no usable depth in the masks"
        )
    kept = {camera: int(m.sum()) for camera, m in lifted.masks.items()}
    step(
        "lift",
        frame=geometry.frame,
        pixels=pixels,
        kept=kept,
        points=len(lifted.points),
        kind=lifted.kind,
    )
    return lifted


def _log_fit(step: Step, record: ObjectRecord) -> None:
    f = record.fit
    step(
        "fit",
        object=record.id,
        shape=f.shape,
        center=list(f.center),
        size=list(f.obb.size),
        yaw_deg=f.yaw_deg,
        symmetry=f.symmetry,
        residual=f.residual,
        direction=None if record.direction is None else record.direction.model_dump(),
    )
