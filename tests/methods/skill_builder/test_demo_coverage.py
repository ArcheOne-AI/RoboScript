"""The demonstration coverage stage (skill_builder/demo_coverage) without a simulator: a pick and
place recording cut at its gripper events into a grasp, a place and home."""

import math

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.demos.demo import ArmTrack, Demo, DemoMeta, Provenance, tcp_from_ee
from roboscript.demos.keypoints import KeypointConfig
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.spatial.rotation import quat_from_matrix, rpy_to_matrix
from roboscript_methods.skill_builder.demo_coverage.segments import CutConfig, Recording, cut

OFFSET = (0.1, 0.0, 0.0)
CUT = CutConfig(keypoints=KeypointConfig(grip_threshold=0.99), home_m=0.15, home_deg=20.0)
"""The cut, with a back-to-origin tolerance as a benchmark's glue passes one."""


def leg(a: tuple[float, ...], b: tuple[float, ...], frames: int) -> np.ndarray:
    return np.linspace(a, b, frames)


def recording() -> Recording:
    """The left arm reaches a block, closes, lifts it, carries it over a base block, lowers it,
    opens, withdraws and returns; the right arm stands still."""
    path = [
        leg((0.0, 0.0, 1.0), (0.3, 0.0, 0.8), 60),
        leg((0.3, 0.0, 0.8), (0.3, 0.0, 0.8), 30),  # rests; closes at frame 80
        leg((0.3, 0.0, 0.8), (0.3, 0.0, 0.9), 30),
        leg((0.3, 0.0, 0.9), (0.3, 0.2, 0.9), 30),
        leg((0.3, 0.2, 0.9), (0.3, 0.2, 0.83), 20),
        leg((0.3, 0.2, 0.83), (0.3, 0.2, 0.83), 20),  # rests; opens at frame 180
        leg((0.3, 0.2, 0.83), (0.3, 0.2, 0.95), 30),
        leg((0.3, 0.2, 0.95), (0.0, 0.0, 1.0), 40),  # back where it started
    ]
    tcp_p = np.vstack(path)
    n = len(tcp_p)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    ee = np.hstack([tcp_p - np.asarray(OFFSET), q])
    grip = np.ones(n)
    grip[80:180] = 0.0
    block = np.hstack([np.tile([0.3, 0.0, 0.775], (n, 1)), q])
    block[80:180, :3] = tcp_p[80:180] - (0.0, 0.0, 0.025)
    block[180:, :3] = (0.3, 0.2, 0.805)  # rests on the base block
    base = np.hstack([np.tile([0.3, 0.2, 0.775], (n, 1)), q])
    far = np.hstack([np.tile([-0.3, 0.0, 0.775], (n, 1)), q])
    still = np.tile([-0.3, -0.35, 0.92, 1.0, 0.0, 0.0, 0.0], (n, 1))
    arms = {
        "left": ArmTrack(ee=ee, tcp=tcp_from_ee(ee, OFFSET), gripper=grip),
        "right": ArmTrack(ee=still, tcp=tcp_from_ee(still, OFFSET), gripper=np.ones(n)),
    }
    meta = DemoMeta(
        task=TaskName("t"),
        seed=0,
        instructions=["stack it"],
        frames=n,
        frame_steps=10,
        ee_to_tcp={"left": OFFSET, "right": OFFSET},
        provenance=Provenance(source="test", path="x.h5", sha256="0" * 64),
    )
    objects = {"block": block, "base": base, "far": far}
    return Recording(Demo(meta, arms), objects, np.zeros((n - 1, 14)))


def test_a_pick_and_place_cuts_into_a_grasp_a_place_and_home() -> None:
    found = cut(recording(), CUT)
    assert [(s.arm, s.kind, s.obj, s.target) for s in found] == [
        ("left", "grasp", "block", None),
        ("left", "place", "block", "base"),
        ("left", "home", None, None),
    ]
    grasp, place, home = found
    assert grasp.end == 80 and place.end == 180 and home.end == recording().demo.meta.frames - 1
    assert grasp.start < 10  # the arm moves from the start
    assert place.rest is not None and place.rest >= 180
    assert grasp.rest is not None and 80 <= grasp.rest < 92  # made as the lift begins


BASE = Pose(p=(0.0, -0.6, 0.0), q=(0.7071068, 0.0, 0.0, 0.7071068))
POSES = load_embodiment(MOCK_BIMANUAL).poses


def ee_at(rpy_deg: tuple[float, float, float], p: tuple[float, float, float]) -> Pose:
    rotation = rpy_to_matrix([math.radians(a) for a in rpy_deg])
    q = tuple(float(v) for v in quat_from_matrix(rotation))
    return Pose(p=p, q=q)  # type: ignore[arg-type]


START = ee_at((0.0, 0.0, 0.0), (0.0, -0.3, 0.9))  # where the place starts, holding the box
