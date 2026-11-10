"""The ceiling under perception: the ceiling's objects from the D-tier perception chain instead of
truth, reported next to it. Only where the objects are changes; the plan (the fitted calls) and the
judge (the official success latch) stay the ceiling's.

`PerceivedCheck` is a `Check` whose `truth` is perception's scene, so the ceiling and `run`'s
obstacles read the objects through it:

1. As an episode opens, the objects the calls name are identified once on the anchor camera's first
   frame (`PerceptionConfig.anchor_camera`; `identify.identify_scene`: proposals, Set-of-Mark
   through the gateway's VLM, GroundingDINO and SAM 3 through the vision service, the three-way
   vote), each role a short description of the object (the benchmark's glue gives it: the ceiling
   is privileged in what it asks for, as a Memory's OBJECTS are in P2), and registered in a
   `SceneRegistry`.
2. Roles with the same description (look-alikes: three bowls) are bound to their objects by
   truth (`rebind`, Hungarian on the centres): which look-alike plays which role is the Agent's
   question in P2 (a Memory's `order:` lines), not the skills'. The rebinding is counted.
3. Each identification is scored against truth at that moment (`score`): the role's registered
   centre inside its true box (`identified`), inside the box of an object of the same description
   (`kind_identified`), and once bound, the centre's and the size's errors.
4. Before each call (the env step moved since the last look) every registered object no arm holds
   is measured again on a new frame of all cameras (`measure.measure`, `relocate` when it is not
   where it was) and taken as the manipulation toolset's coarse-then-fine refine takes one:
   `ok` and at least `MIN_CONFIDENCE` sure and moved at most `MAX_SHIFT_M`, or `relocated` and
   sure (`PerceptionConfig.measure.trust`). Held objects follow their grasp
   (`SceneRegistry.on_close` / `on_open` after each receipt, as `toolsets.scene.after_action`).

Truth is read for the scores, the look-alike binding and the judge only; a proxy the calls act
on is always perception's (`ObjectProxy.from_geometry` of the registered record).
"""

import contextlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import anyio
import numpy as np
from scipy.optimize import linear_sum_assignment

from roboscript.blobs import BlobStore
from roboscript.contracts.env import EnvControl, ObserveRequest
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import Planner
from roboscript.contracts.receipt import ActionReceipt, GripEvent
from roboscript.contracts.robot import RobotState
from roboscript.contracts.vision import Vision
from roboscript.manipulation import Built
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.perception.config import PerceptionConfig
from roboscript.perception.frames import Frame
from roboscript.perception.grounding import proposal_source
from roboscript.perception.identify import identify_scene
from roboscript.perception.measure import measure, relocate
from roboscript.perception.proposals import propose
from roboscript.perception.registry import SceneRegistry, object_geometry
from roboscript.perception.robot import RobotBody, SelfFilter
from roboscript.perception.table import Table
from roboscript.perception.views import FrameGeometry, analyze
from roboscript.perception.vision import VisionDetector, VisionSegmenter
from roboscript.perception.vlm import BoxGrounder, SomGrounder, VlmConfig
from roboscript_methods.skill_builder.core_entry.check import Bench, Check, DemoScene, Episode

MIN_CONFIDENCE: Final = 0.5
"""A re-measurement is taken at least this sure (`RefineConfig.min_confidence`'s default)."""
MAX_SHIFT_M: Final = 0.05
"""...and moving the object at most this far (`RefineConfig.max_shift_m`'s default)."""
INSIDE_M: Final = 0.01
"""A registered centre within this of a true box counts as inside it."""


@dataclass(frozen=True)
class Perceiver:
    """What perceiving needs: the gateway client (an `openai.AsyncOpenAI`) and its VLM settings,
    the vision service (None: no GroundingDINO, SAM 3 or split) and the blob store it shares with
    us, the robot's body for the self-filter (the planner's FK places it), the pipeline config."""

    client: Any
    vlm: VlmConfig
    vision: Vision | None
    blobs: BlobStore
    body: RobotBody
    config: PerceptionConfig
    log_dir: Path | None = None


class NotPerceived(LookupError):
    """A call names an object perception did not register: the perceived ceiling's failure (the
    identification's), not the infrastructure's."""


@dataclass
class PerceivedScene:
    """The registry's objects by role, as a ceiling reads `gt_scene.Scene`: `poses` (each role's
    pose now) and `proxy`; `boxes` names them for `obstacles`. No moving links."""

    registry: SceneRegistry
    ids: dict[str, str]
    robot: RobotState
    poses: dict[str, Pose] = field(init=False)
    boxes: dict[str, None] = field(init=False)

    def __post_init__(self) -> None:
        self.poses = {
            role: Pose.from_list(object_geometry(self.registry.get(i), self.robot).pose)
            for role, i in self.ids.items()
        }
        self.boxes = dict.fromkeys(self.ids)

    def proxy(self, name: str) -> ObjectProxy:
        if name not in self.ids:
            raise NotPerceived(f"{name} was not identified")
        record = self.registry.get(self.ids[name])
        return ObjectProxy.from_geometry(object_geometry(record, self.robot))

    def part_of(self, name: str, point: Any, margin: float, links: bool = False) -> str | None:
        return None  # perception registers no moving links here


def center(proxy: ObjectProxy) -> np.ndarray:
    """A proxy's whole box centre, world."""
    return np.asarray((proxy.pose @ proxy.whole.frame)[:3, 3], dtype=float)


def inside(point: np.ndarray, proxy: ObjectProxy, margin: float = INSIDE_M) -> bool:
    """Whether a world point lies in a proxy's whole box grown by `margin`."""
    box = proxy.pose @ proxy.whole.frame
    local = np.linalg.inv(box) @ np.append(point, 1.0)
    return bool(np.all(np.abs(local[:3]) <= proxy.whole.size / 2 + margin))


def rebind(
    found: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    descriptions: Mapping[str, str],
) -> dict[str, str]:
    """Role -> the role whose registration it takes: within each description that several roles
    share, the registrations (`found`: registered role -> centre) assigned one to one to the
    roles by true centre (Hungarian on distance); any other registered role keeps its own."""
    bound: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for role in truth:
        groups.setdefault(descriptions[role], []).append(role)
    for roles in groups.values():
        have = [r for r in roles if r in found]
        if len(roles) < 2:
            bound |= {r: r for r in have}
            continue
        if not have:
            continue
        cost = np.array([[np.linalg.norm(found[a] - truth[b]) for b in roles] for a in have])
        rows, cols = linear_sum_assignment(cost)
        bound |= {roles[c]: have[r] for r, c in zip(rows, cols, strict=True)}
    return bound


def score(
    role: str,
    perceived: ObjectProxy | None,
    truth: Mapping[str, ObjectProxy],
    descriptions: Mapping[str, str],
) -> dict[str, Any]:
    """A role's registration against truth: inside its own true box, inside one of the same
    description's (`kind_identified`), the centre's error (mm, whole, across, up) and the box's
    (sorted footprint sides and height)."""
    if perceived is None:
        return {"registered": False, "identified": False, "kind_identified": False}
    c = center(perceived)
    same = [n for n in truth if descriptions.get(n) == descriptions[role]]
    t = truth[role]
    d = c - center(t)
    return {
        "registered": True,
        "identified": inside(c, t),
        "kind_identified": any(inside(c, truth[n]) for n in same),
        "center_mm": round(float(np.linalg.norm(d)) * 1000, 1),
        "xy_mm": round(float(np.linalg.norm(d[:2])) * 1000, 1),
        "z_mm": round(float(d[2]) * 1000, 1),
        "size_mm": [round(float(v) * 1000, 1) for v in sides(perceived) - sides(t)],
        "yaw_deg": round(yaw_between(perceived, t), 1),
    }


def yaw_between(a: ObjectProxy, b: ObjectProxy) -> float:
    """How far two whole boxes' x axes are turned about z, either way round (0-90 degrees)."""
    xa, xb = ((p.pose @ p.whole.frame)[:2, 0] for p in (a, b))
    turn = abs(np.degrees(np.arctan2(xa[1], xa[0]) - np.arctan2(xb[1], xb[0]))) % 180.0
    return float(min(turn, 180.0 - turn))


def sides(proxy: ObjectProxy) -> np.ndarray:
    """A whole box's footprint sides (shorter first) and height."""
    return np.array([*sorted(proxy.whole.size[:2]), proxy.whole.size[2]])


def robot_at(final: RobotState, receipt: ActionReceipt, event: GripEvent) -> RobotState:
    """The robot at a grip event (`toolsets.scene._robot_at`)."""
    arms = dict(final.arms)
    for arm, path in receipt.tcp_path.items():
        if event.step < len(path):
            arms[arm] = arms[arm].model_copy(update={"tcp": path[event.step]})
    arms[event.arm] = arms[event.arm].model_copy(update={"gripper": event.opening})
    return final.model_copy(update={"arms": arms})


class PerceivedCheck(Check):
    """A `Check` whose objects are perception's (the module): `truth` returns a `PerceivedScene`,
    `run` keeps the registry's grasp relations; `identification` collects each episode's scores,
    `refines` each re-measurement's outcome."""

    def __init__(
        self,
        bench: Bench,
        demo: DemoScene,
        control: EnvControl,
        planner: Planner,
        perceiver: Perceiver,
        descriptions: Mapping[str, str],
        env_blobs: BlobStore,
    ) -> None:
        """`descriptions`: role (an object the calls name) -> what to find it by; `env_blobs`:
        the env server's blob store (its observations' images)."""
        super().__init__(bench, demo, control, planner)
        self.perceiver, self.descriptions = perceiver, dict(descriptions)
        self.env_blobs = env_blobs
        self.filter = SelfFilter(planner, perceiver.body, perceiver.config.self_filter)
        bases = [a.planner.base.p for a in bench.profile.arms.values() if a.planner is not None]
        self.base = np.mean(bases, axis=0) if bases else None
        self.identification: list[list[dict[str, Any]]] = []
        self.refines: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def episode(self) -> AsyncIterator[Episode]:
        async with super().episode() as ep:
            await self._identify(ep)
            yield ep

    async def _look(self, ep: Episode, cameras: Sequence[str] | None) -> FrameGeometry:
        seen = await ep.agent.observe(ep.opened.episode_id, ObserveRequest(cameras=None))
        frame = Frame(seen, self.env_blobs)
        shape = await self.filter.shape(seen.robot)
        table: Table | None = self._state.get("table")
        return await analyze(frame, self.perceiver.config, shape, self.base, cameras, table)

    async def _identify(self, ep: Episode) -> None:
        p = self.perceiver
        self._state = {}
        geometry = await self._look(ep, [p.config.anchor_camera])
        self._state["table"] = geometry.table
        source = proposal_source(p.config)
        registry = SceneRegistry()
        steps: list[dict[str, Any]] = []

        def log(step: str, **data: Any) -> None:
            steps.append({"step": step, **data})

        roles = {name: self.descriptions[name] for name in self._names()}
        robot = await self.robot(ep)
        found = await identify_scene(
            geometry,
            roles,
            registry,
            proposals=await source(geometry),
            som=SomGrounder(p.client, p.vlm, p.blobs, source, p.config),
            box=BoxGrounder(p.client, p.vlm, p.blobs, source, p.config),
            detector=None if p.vision is None else VisionDetector(p.vision, p.blobs),
            segmenter=None if p.vision is None else VisionSegmenter(p.vision, p.blobs),
            robot=robot,
            config=p.config,
            log=log,
        )
        assert self.measured is not None
        truth = {n: self.measured.at(ep.first.objects, {}).proxy(n) for n in roles}
        ids = {role: f.object for role, f in found.items() if f.object is not None}
        proxies = {
            role: ObjectProxy.from_geometry(object_geometry(registry.get(i), robot))
            for role, i in ids.items()
        }
        rows: list[dict[str, Any]] = [
            {"role": role, "description": roles[role], "reason": str(found[role].reason),
             "confidence": round(found[role].confidence, 3), "ambiguous": found[role].ambiguous,
             "as_registered": score(role, proxies.get(role), truth, roles)}
            for role in roles
        ]  # fmt: skip
        bound = rebind({r: center(x) for r, x in proxies.items()},
                       {r: center(x) for r, x in truth.items()}, roles)  # fmt: skip
        ids = {role: ids[by] for role, by in bound.items()}
        for role, row in zip(roles, rows, strict=True):
            row["bound_to"] = bound.get(role)
            row["rebound"] = role in bound and bound[role] != role
            got = proxies.get(bound[role]) if role in bound else None
            row["bound"] = score(role, got, truth, roles)
        for role, i in ids.items():  # the registry names each object by the role it plays
            registry.objects[i] = registry.objects[i].model_copy(update={"name": role})
        self._state |= {"registry": registry, "ids": ids, "step": None, "scene": None}
        self.identification.append(rows)
        if p.log_dir is not None:
            p.log_dir.mkdir(parents=True, exist_ok=True)
            k = len(self.identification)
            (p.log_dir / f"identify-{k}.json").write_text(
                json.dumps({"rows": rows, "steps": steps}, indent=1, default=str) + "\n",
                encoding="utf-8",
            )

    def _names(self) -> list[str]:
        return [n for n in self.descriptions if n in (self.measured.boxes if self.measured else {})]

    async def truth(self, ep: Episode) -> PerceivedScene:
        """Perception's scene now: re-measured once per env step (module, 4)."""
        robot = await self.robot(ep)
        if self._state.get("step") != robot.step:
            await self._refine(ep)
            self._state["step"] = robot.step
        robot = await self.robot(ep)
        return PerceivedScene(self._state["registry"], self._state["ids"], robot)

    async def _refine(self, ep: Episode) -> None:
        registry: SceneRegistry = self._state["registry"]
        free = [i for i, r in registry.objects.items() if not r.grasps and not r.part_grasps]
        if not free:
            return
        geometry = await self._look(ep, None)
        config = self.perceiver.config
        robot = await self.robot(ep)

        def run() -> list[dict[str, Any]]:
            rows = []
            proposals = None
            for i in free:
                found = measure(registry, geometry, i, robot=robot, pipeline=config)
                if found.reason == "not_visible":
                    proposals = proposals if proposals is not None else propose(geometry, config)
                    found = relocate(registry, geometry, i, proposals, robot=robot,
                                     pipeline=config)  # fmt: skip
                taken = (
                    found.reason == "ok"
                    and found.confidence >= MIN_CONFIDENCE
                    and found.shift_m <= MAX_SHIFT_M
                ) or (found.reason == "relocated" and found.confidence >= config.measure.trust)
                if taken:
                    registry.objects[i] = found.record
                rows.append(
                    {
                        "step": robot.step,
                        "object": registry.get(i).name,
                        "reason": str(found.reason),
                        "confidence": round(found.confidence, 3),
                        "shift_mm": round(found.shift_m * 1000, 1),
                        "taken": taken,
                    }
                )
            return rows

        self.refines += await anyio.to_thread.run_sync(run)

    async def run(self, ep: Episode, *built: Built) -> ActionReceipt:
        receipt = await super().run(ep, *built)
        registry: SceneRegistry = self._state["registry"]
        final = await self.robot(ep)
        table: Table | None = self._state.get("table")
        for event in receipt.grips:
            robot = robot_at(final, receipt, event)
            gripper = self.profile.arms[event.arm].gripper
            threshold = gripper.holding_threshold
            if event.closed and event.opening is not None and event.opening >= threshold:
                registry.on_close(
                    event.arm,
                    robot,
                    threshold=threshold,
                    closing_axis=self.profile.poses.tcp_closing_axis,
                    span_m=gripper.span_m,
                )
            else:
                registry.on_open(event.arm, robot, table)
        return receipt
