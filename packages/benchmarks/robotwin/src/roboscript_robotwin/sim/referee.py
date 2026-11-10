"""Ground truth for the control plane (judge, diagnose, gt_snapshot): the task's objects and
their poses. Never reachable from the agent plane."""

import itertools
import re
import sys
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.env import GtJoint, GtLink
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.robot import Arm
from roboscript.embodiments import EmbodimentProfile

# Attributes holding RoboTwin actors that are not task objects: the backdrop, and the distractors
# of a cluttered (randomized) table, all of them and the last one placed.
_SCENERY: Final = frozenset({"wall", "cluttered_objs", "cluttered_obj"})
_MOVING: Final = {
    "revolute": "revolute",
    "revolute_unwrapped": "revolute",
    "prismatic": "prismatic",
}
"""SAPIEN's joint types of a movable link, as `GtJoint.kind`."""
_ASSET: Final = re.compile(r"^\d+_|/.*$|\d+$")  # an asset number, a variant, a model number


@dataclass(frozen=True)
class Truth:
    """The task objects at one moment: world poses, articulated objects' joint positions, per
    arm the objects its links touch, and each object's category word (`categories`)."""

    poses: dict[str, Pose]
    joints: dict[str, list[float]]
    touched: dict[Arm, list[str]]
    categories: dict[str, str]


@dataclass(frozen=True)
class Visual:
    """A task object with no collision body, only a render body (a target square drawn on the
    table), read as the referee reads actors: its entity (`actor`), pose and name."""

    actor: Any

    def get_pose(self) -> Any:
        return self.actor.get_pose()

    def get_name(self) -> str:
        return str(self.actor.get_name())


def task_objects(task: Any) -> dict[str, Any]:
    """The task object's attributes holding RoboTwin actors of its current scene (lists of them
    too, as `name[i]`), each actor once, under the first attribute that holds it, and entities it
    holds that have a render body but no collision body (`Visual`). Left out: an attribute only
    `play_once` sets, still holding the expert pre-run's actor of a scene since replaced, or
    naming an actor again (a `last_actor` attribute), and the distractors of a cluttered
    table."""
    from envs.utils.actor_utils import Actor  # ArticulationActor is one too

    sapien = sys.modules.get("sapien")  # a task holds entities only once SAPIEN is loaded
    found: dict[str, Any] = {}
    for name, value in vars(task).items():
        if name in _SCENERY:
            continue
        if isinstance(value, Actor):
            held = {name: value}
        elif isinstance(value, list | tuple) and value and all(isinstance(v, Actor) for v in value):
            held = {f"{name}[{i}]": actor for i, actor in enumerate(value)}
        elif (
            sapien is not None
            and isinstance(value, sapien.Entity)
            and value.find_component_by_type(sapien.render.RenderBodyComponent) is not None
            and value.find_component_by_type(sapien.physx.PhysxRigidBaseComponent) is None
        ):
            held = {name: Visual(value)}
        else:
            continue
        for key, actor in held.items():
            entity = _entities(actor)[0]
            here = entity.get_scene() is task.scene
            if here and all(entity is not _entities(other)[0] for other in found.values()):
                found[key] = actor
    return found


def categories(objects: dict[str, Any]) -> dict[str, str]:
    """Each object's category word, from its RoboTwin name (`category`)."""
    return {name: category(str(actor.get_name())) for name, actor in objects.items()}


def category(asset: str) -> str:
    """A RoboTwin model or actor name as a category word, without its asset number and
    variant: `050_bell` and `050_bell/base3` are `bell`, `046_alarm-clock` is `alarm clock`."""
    return re.sub(r"[_-]+", " ", _ASSET.sub("", asset)).strip()


def poses(objects: dict[str, Any]) -> dict[str, Pose]:
    """World poses (of the root link, for articulated objects)."""
    return {name: _pose(actor.get_pose()) for name, actor in objects.items()}


def centers(objects: dict[str, Any]) -> dict[str, Vec3]:
    """The center of each object's geometry in the world: the middle of the box around the
    vertices of its render shapes, as perception's centers are the middle of the box around its
    points (an asset's pose need not sit at its middle)."""
    found: dict[str, Vec3] = {}
    for name, actor in objects.items():
        points = np.concatenate([_vertices(entity) for entity in _entities(actor)])
        if not len(points):
            raise ValueError(f"{name} has no render shape to take its center from")
        x, y, z = (points.min(axis=0) + points.max(axis=0)) / 2
        found[name] = (float(x), float(y), float(z))
    return found


def joint_positions(objects: dict[str, Any]) -> dict[str, list[float]]:
    """Joint positions of the articulated objects (laptop lids, drawers, switches, ...)."""
    from envs.utils.actor_utils import ArticulationActor

    return {
        name: [float(q) for q in actor.get_qpos()]
        for name, actor in objects.items()
        if isinstance(actor, ArticulationActor)
    }


def joint_limits(objects: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """Each articulated object's joint limits, low and high, in the order of its positions."""
    from envs.utils.actor_utils import ArticulationActor

    return {
        name: [(float(low), float(high)) for low, high in actor.get_qlimits()]
        for name, actor in objects.items()
        if isinstance(actor, ArticulationActor)
    }


def hulls(objects: dict[str, Any]) -> dict[str, list[list[Vec3]]]:
    """Each object's convex hulls in the world, one vertex list per collision shape (every
    link's, for an articulated one; a `Visual`'s render shapes): what a replay with fitted specs
    measures its boxes from and samples surfaces of."""
    return {name: _object_hulls(actor) for name, actor in objects.items()}


def links(objects: dict[str, Any], hulls: bool = True) -> dict[str, dict[str, GtLink]]:
    """The articulated objects' movable links (a revolute or prismatic joint to the parent) by
    link name: world pose, collision hulls (if `hulls`) and joint."""
    from envs.utils.actor_utils import ArticulationActor

    found: dict[str, dict[str, GtLink]] = {}
    for name, actor in objects.items():
        if not isinstance(actor, ArticulationActor):
            continue
        qpos = [float(q) for q in actor.get_qpos()]
        moving = {}
        for index, joint in enumerate(actor.actor.get_active_joints()):
            kind = _MOVING.get(str(joint.type))
            if kind is None:
                continue
            link = joint.get_child_link()
            frame = joint.get_global_pose().to_transformation_matrix().astype(float)
            low, high = (float(v) for v in joint.get_limits()[0])
            moving[str(link.get_name())] = GtLink(
                pose=_pose(link.entity.pose),
                hulls=[_listed(hull) for hull in _body_hulls(link)] if hulls else [],
                joint=GtJoint(
                    kind=kind,  # type: ignore[arg-type]
                    origin=_vec(frame[:3, 3]),
                    axis=_vec(frame[:3, 0]),  # SAPIEN joints move about or along their x axis
                    position=qpos[index],
                    limits=(_finite(low), _finite(high)),
                ),
            )
        found[name] = moving
    return found


def link_poses(objects: dict[str, Any]) -> dict[str, dict[str, Pose]]:
    """The articulated objects' movable links' world poses, by link name."""
    return {
        name: {link: truth.pose for link, truth in moving.items()}
        for name, moving in links(objects, hulls=False).items()
    }


def _object_hulls(actor: Any) -> list[list[Vec3]]:
    import sapien

    if isinstance(actor, Visual):
        render = actor.actor.find_component_by_type(sapien.render.RenderBodyComponent)
        world = actor.actor.pose.to_transformation_matrix().astype(float)
        return [_listed(_rendered(shape, world)) for shape in render.render_shapes]
    bodies = (
        actor.actor.get_links()
        if hasattr(actor.actor, "get_links")
        else [actor.actor.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)]
    )
    return [_listed(hull) for body in bodies for hull in _body_hulls(body)]


def _body_hulls(body: Any) -> list[NDArray[np.float64]]:
    """A physx body's collision shapes as convex hulls' vertices (world), one array per shape:
    a mesh's vertices, a box's corners, points on a sphere, capsule or cylinder."""
    import sapien

    physx = sapien.physx
    world = body.entity.pose.to_transformation_matrix().astype(float)
    found = []
    for shape in body.get_collision_shapes():
        if isinstance(
            shape, physx.PhysxCollisionShapeConvexMesh | physx.PhysxCollisionShapeTriangleMesh
        ):
            corners = np.asarray(shape.get_vertices(), dtype=float) * shape.get_scale()
        elif isinstance(shape, physx.PhysxCollisionShapeBox):
            corners = _SIGNS * np.asarray(shape.half_size, dtype=float)
        elif isinstance(shape, physx.PhysxCollisionShapeSphere):
            corners = _BALL * shape.radius
        elif isinstance(shape, physx.PhysxCollisionShapeCapsule):  # along its x
            ends = [_BALL * shape.radius + _along(sign * shape.half_length) for sign in (-1, 1)]
            corners = np.concatenate(ends)
        elif isinstance(shape, physx.PhysxCollisionShapeCylinder):  # along its x
            ring = _RING * shape.radius
            corners = np.concatenate([ring + _along(sign * shape.half_length) for sign in (-1, 1)])
        else:
            raise TypeError(f"{body.get_name()}: no hull for a {type(shape).__name__}")
        m = world @ shape.get_local_pose().to_transformation_matrix().astype(float)
        found.append(corners @ m[:3, :3].T + m[:3, 3])
    return found


def _rendered(shape: Any, world: Any) -> Any:
    """A render shape's vertices (a box's corners, a mesh's scaled vertices), world."""
    if hasattr(shape, "half_size"):
        corners = _SIGNS * np.asarray(shape.half_size, dtype=float)
    else:
        scale = np.asarray(shape.scale, dtype=float)
        corners = np.concatenate([np.asarray(p.vertices, dtype=float) for p in shape.parts]) * scale
    m = world @ shape.local_pose.to_transformation_matrix().astype(float)
    return corners @ m[:3, :3].T + m[:3, 3]


def _along(x: float) -> NDArray[np.float64]:
    return np.array([x, 0.0, 0.0])


def _listed(stacked: Any) -> list[Vec3]:
    return [(float(x), float(y), float(z)) for x, y, z in stacked]


_SIGNS: Final = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
_ANGLES: Final = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
_RING: Final = np.column_stack([np.zeros(16), np.cos(_ANGLES), np.sin(_ANGLES)])
"""A unit circle about x, 16 points."""
_BALL: Final = np.array(
    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    + [
        [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)]
        for a in _ANGLES
        for b in np.radians((-60.0, -30.0, 0.0, 30.0, 60.0))
    ]
)
"""Points on the unit sphere: the poles and 16 meridians at 30 deg steps."""


def _vec(v: Any) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _finite(v: float) -> float | None:
    return v if np.isfinite(v) else None


def table_top(task: Any) -> float:
    """The height of the table top: RoboTwin's table entity sits at its top surface."""
    return float(task.table.get_pose().p[2])


def contacts(
    task: Any, objects: dict[str, Any], profile: EmbodimentProfile
) -> dict[Arm, list[str]]:
    """Per arm, the objects any of its links touch (a contact with an impulse): every link
    below the arm's first joint, its fingers and wrist camera included."""
    owner = {e.per_scene_id: name for name, actor in objects.items() for e in _entities(actor)}
    arm_of = {
        link.entity.per_scene_id: arm
        for arm, links in _arm_links(task, profile).items()
        for link in links
    }
    touched: dict[Arm, set[str]] = {arm: set() for arm in profile.arms}
    for contact in task.scene.get_contacts():
        if not any(np.any(point.impulse) for point in contact.points):
            continue
        a, b = (body.entity.per_scene_id for body in contact.bodies)
        for mine, other in ((a, b), (b, a)):
            if mine in arm_of and other in owner:
                touched[arm_of[mine]].add(owner[other])
    return {arm: sorted(names) for arm, names in touched.items()}


def _vertices(entity: Any) -> NDArray[np.float64]:
    """The world vertices of an entity's render shapes (a box's corners)."""
    import sapien

    pose = entity.get_pose().to_transformation_matrix()
    found = [np.empty((0, 3))]
    for component in entity.get_components():
        if not isinstance(component, sapien.render.RenderBodyComponent):
            continue
        for shape in component.render_shapes:
            local = pose @ shape.local_pose.to_transformation_matrix()
            if isinstance(shape, sapien.render.RenderShapeTriangleMesh):
                parts = [np.asarray(part.vertices) * shape.scale for part in shape.parts]
                points = np.concatenate(parts)
            elif isinstance(shape, sapien.render.RenderShapeBox):
                points = np.array(list(itertools.product(*((-h, h) for h in shape.half_size))))
            else:
                raise TypeError(f"{entity.get_name()}: no vertices for a {type(shape).__name__}")
            found.append(points @ local[:3, :3].T + local[:3, 3])
    return np.concatenate(found)


def _entities(actor: Any) -> list[Any]:
    """The entities of an actor: itself, or an articulation's links (a `Visual`'s: its entity)."""
    from envs.utils.actor_utils import ArticulationActor

    if isinstance(actor, ArticulationActor):
        return [link.entity for link in actor.actor.get_links()]
    return [actor.actor]


def _arm_links(task: Any, profile: EmbodimentProfile) -> dict[Arm, list[Any]]:
    """Each arm's links: the child link of its first joint and every link below it."""
    out = {}
    for arm, spec in profile.arms.items():
        robot = task.robot.left_entity if arm == "left" else task.robot.right_entity
        first = next(joint for joint in robot.get_joints() if joint.get_name() == spec.joints[0])
        top = first.get_child_link().get_name()
        links = robot.get_links()
        parent = {link.get_name(): link.get_parent() for link in links}
        out[arm] = [link for link in links if top in _lineage(link.get_name(), parent)]
    return out


def _lineage(name: str, parent: dict[str, Any]) -> list[str]:
    """A link's name and its ancestors' names."""
    names = [name]
    while (up := parent[names[-1]]) is not None:
        names.append(up.get_name())
    return names


def _pose(pose: Any) -> Pose:
    return Pose.from_list([*pose.p, *pose.q])
