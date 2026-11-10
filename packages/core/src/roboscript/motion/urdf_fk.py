"""Forward kinematics on the CPU, as the cuRobo planner computes it (`CuroboPlanner.fk`): each
arm's URDF chain from its curobo yml's `base_link` to its `ee_link` (locked joints at their
values), mapped into the world as the planner maps its frame (the base pose times the inverse of
`planner_T_root`), the TCP `ee_to_tcp` ahead of the ee. It needs no GPU and no cuRobo, so a
demonstration referee's paths (`demos.paths.ArmPaths`) can be rebuilt next to a running pipeline.
Only `fk` is served (no links, no spheres)."""

import math
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import FKRequest, FKResult
from roboscript.contracts.robot import Arm
from roboscript.motion.config import CuroboConfig
from roboscript.motion.curobo import load_robot, planner_T_root
from roboscript.spatial.pose import pose_compose, pose_inverse

Array = NDArray[np.float64]


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    origin: Array  # (4, 4)
    axis: Array  # (3,)


def _rpy(r: float, p: float, y: float) -> Array:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y),
                              math.sin(y))  # fmt: skip
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _about(axis: Array, angle: float) -> Array:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    out: Array = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    return out


def _floats(element: ElementTree.Element | None, key: str, default: str) -> list[float]:
    """An element's attribute as floats (URDF: space-separated), `default` without it."""
    text = default if element is None else element.get(key, default)
    return [float(v) for v in text.split()]


def _quaternion(r: Array) -> list[float]:
    """(w, x, y, z) of a rotation matrix, w >= 0."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / norm, x / norm, y / norm, z / norm]


class UrdfChain:
    """A URDF's joints from `base` to `ee` (`pose` of a joint-value mapping: `ee` in `base`'s
    frame, (x, y, z, w, qx, qy, qz))."""

    def __init__(self, urdf: Path, base: str, ee: str) -> None:
        root = ElementTree.parse(urdf).getroot()  # a local asset file
        by_child: dict[str, tuple[str, _Joint]] = {}
        for joint in root.findall("joint"):
            origin, axis = joint.find("origin"), joint.find("axis")
            direction = np.array(_floats(axis, "xyz", "1 0 0"))
            frame = np.eye(4)
            frame[:3, :3] = _rpy(*_floats(origin, "rpy", "0 0 0"))
            frame[:3, 3] = _floats(origin, "xyz", "0 0 0")
            parent, child = joint.find("parent"), joint.find("child")
            if parent is None or child is None:
                raise ValueError(f"{urdf}: joint {joint.get('name')} without parent or child")
            by_child[str(child.get("link"))] = (
                str(parent.get("link")),
                _Joint(
                    str(joint.get("name")),
                    str(joint.get("type")),
                    frame,
                    direction / (np.linalg.norm(direction) or 1.0),  # a fixed joint: none
                ),
            )
        chain: list[_Joint] = []
        link = ee
        while link != base:
            if link not in by_child:
                raise ValueError(f"{urdf}: no chain from {base} to {ee} (stuck at {link})")
            link, found = by_child[link]
            chain.append(found)
        self.joints = chain[::-1]

    def pose(self, values: Mapping[str, float]) -> list[float]:
        t = np.eye(4)
        for joint in self.joints:
            t = t @ joint.origin
            if joint.kind in ("revolute", "continuous"):
                move = np.eye(4)
                move[:3, :3] = _about(joint.axis, values[joint.name])
                t = t @ move
            elif joint.kind == "prismatic":
                move = np.eye(4)
                move[:3, 3] = joint.axis * values[joint.name]
                t = t @ move
        return [*t[:3, 3].tolist(), *_quaternion(t[:3, :3])]


@dataclass(frozen=True)
class _Arm:
    chain: UrdfChain
    joints: list[str]
    locked: dict[str, float]
    planner_T_root: Array
    ee_to_tcp: tuple[float, float, float]


class UrdfKinematics:
    """`Planner.fk` of a cuRobo planner config's arms (`CuroboConfig`), on the CPU."""

    def __init__(self, config: CuroboConfig) -> None:
        directory, setups = config.resolve()
        self._arms: dict[Arm, _Arm] = {}
        for arm, setup in setups.items():
            robot = load_robot(setup.curobo_yml, directory, setup.joints)
            kinematics = robot["robot_cfg"]["kinematics"]
            locked = {str(k): float(v) for k, v in (kinematics.get("lock_joints") or {}).items()}
            self._arms[arm] = _Arm(
                chain=UrdfChain(
                    Path(kinematics["urdf_path"]), kinematics["base_link"], kinematics["ee_link"]
                ),
                joints=[j for j in kinematics["cspace"]["joint_names"] if j not in locked],
                locked=locked,
                planner_T_root=planner_T_root(robot["planner"]["frame_bias"], setup.yaw_rad),
                ee_to_tcp=setup.ee_to_tcp,
            )

    async def fk(self, request: FKRequest) -> FKResult:
        if request.links or request.spheres:
            raise ValueError("UrdfKinematics serves the ee and TCP only (no links, no spheres)")
        arm = self._arms[request.arm]
        world_T_planner = pose_compose(request.base.as_list(), pose_inverse(arm.planner_T_root))
        local = np.array(
            [arm.chain.pose({**arm.locked, **dict(zip(arm.joints, q, strict=True))})
             for q in request.joints]
        )  # fmt: skip
        ee = pose_compose(world_T_planner, local)
        tcp = pose_compose(ee, [*arm.ee_to_tcp, 1.0, 0.0, 0.0, 0.0])
        return FKResult(ee=_poses(ee), tcp=_poses(tcp))


def _poses(rows: Sequence[Sequence[float]]) -> list[Pose]:
    return [Pose.from_list([float(v) for v in row]) for row in rows]
