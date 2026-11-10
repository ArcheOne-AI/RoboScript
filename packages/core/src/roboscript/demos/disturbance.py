"""What a demonstration did to the scene besides its task, from the referee's ground truth: how
far objects moved while no hand held them (the disturbance), the part of it with an arm link
touching them (an arm strike), and the motion of articulated joints other than the one the task
moves.

The ground truth of a recording (`SceneTruth`, `referee/truth.npz`) is per frame: the task
objects' world poses, per arm the objects its links touch, the joint positions of articulated
objects. An arm holds, from a close until its next open (that frame included), the object it
touched most while closed on something (a settled opening above the arm's `holding`
threshold, the embodiment's `Gripper.holding_threshold`); nothing
while it closes on nothing (a press). Motion into a frame does not count while the object is held
at either end, when it starts within `settle_steps` physics steps of the release, when it moves
with a held object (riding in or on it: its displacement within half the carrier's plus the
jitter), or when it is jitter: under `jitter_m` a physics step, over the steps into the frame
(`Demo.steps()`). A payload is an object no
hand held that rode on a held one and ended more than `payload_m` from its start
(balls in a bin being dumped): the task moves it, and it is left out. An articulated object's task
joint is the one that moved most.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, get_args

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt

from roboscript.contracts.robot import Arm
from roboscript.demos.demo import Array, Demo
from roboscript.demos.keypoints import runs

TRUTH_FILE: Final = "truth.npz"


class DisturbanceConfig(BaseModel):
    """Lengths in meters, `jitter_m` a physics step's; the `flag_*` values mark a demonstration
    for review. `settle_steps` covers a release: the physics steps the fingers take to open,
    while what they let go of falls or slides, plus a margin. It depends on the embodiment and
    the simulator's step, so a benchmark's demos config sets it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    settle_steps: PositiveInt
    jitter_m: PositiveFloat = 0.0005
    payload_m: PositiveFloat = 0.10
    flag_strike_m: PositiveFloat = 0.01
    flag_disturbance_m: PositiveFloat = 0.02
    flag_side_joint_rad: PositiveFloat = 0.3


@dataclass(frozen=True, eq=False)
class SceneTruth:
    """Per frame (N) of a recording: the poses (N, M, 7) of the task objects `objects`, per arm
    the objects its links touch (N, M), and each articulated object's joint positions (N, J);
    `categories` names what each object is (a category word)."""

    objects: list[str]
    poses: Array
    touched: dict[Arm, NDArray[np.bool_]]
    joints: dict[str, Array]
    categories: dict[str, str] = field(default_factory=dict)


class ObjectMotion(BaseModel):
    """`free_m`: path length while no hand held it (the counted motion); `strike_m`: the part
    of it with an arm link touching it; `net_m`: from its first to its last position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    held: bool
    payload: bool
    free_m: NonNegativeFloat
    strike_m: NonNegativeFloat
    net_m: NonNegativeFloat


class Disturbance(BaseModel):
    """Totals over the objects that are no payload; `side_joints_rad` per articulated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objects: list[ObjectMotion]
    disturbance_m: NonNegativeFloat
    strike_m: NonNegativeFloat
    side_joints_rad: dict[str, NonNegativeFloat]
    flags: list[str]


def save_truth(truth: SceneTruth, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.generic]] = {
        "objects": np.array(truth.objects, dtype=str),
        "categories": np.array([truth.categories.get(n, "") for n in truth.objects], dtype=str),
        "poses": truth.poses,
        **{f"touched.{arm}": touched for arm, touched in truth.touched.items()},
        **{f"joints.{name}": q for name, q in truth.joints.items()},
    }
    np.savez_compressed(directory / TRUTH_FILE, **arrays)  # type: ignore[arg-type]


def load_truth(directory: Path) -> SceneTruth:
    with np.load(directory / TRUTH_FILE, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    arms = [arm for arm in get_args(Arm) if f"touched.{arm}" in arrays]
    names = [str(name) for name in arrays["objects"]]
    kinds = zip(names, arrays["categories"].tolist(), strict=True)
    return SceneTruth(
        objects=names,
        poses=arrays["poses"],
        touched={arm: arrays[f"touched.{arm}"] for arm in arms},
        joints={k.removeprefix("joints."): v for k, v in arrays.items() if k.startswith("joints.")},
        categories={name: kind for name, kind in kinds if kind},
    )


def disturbance(
    demo: Demo,
    truth: SceneTruth,
    config: DisturbanceConfig,
    holding: Mapping[Arm, float],
    grip_threshold: float = 0.5,
) -> Disturbance:
    """`holding`: per arm, the measured opening above which a closed gripper holds something
    (the embodiment's `Gripper.holding_threshold`)."""
    n, m = truth.poses.shape[:2]
    steps = demo.steps()
    jitter = config.jitter_m * np.diff(steps)[:, None]  # (N - 1, 1): over the steps into a frame
    held = np.zeros((n, m), dtype=bool)
    settling = np.zeros((n - 1, m), dtype=bool)  # the motion into frames 1..N-1
    for arm, track in demo.arms.items():
        holds = closed = track.gripper < grip_threshold
        if track.opening is not None:  # closed on something, not on nothing (a press)
            holds = closed & (track.opening > holding[arm])
        touched = truth.touched[arm] & holds[:, None]
        for lo, hi in runs(closed):
            counts = touched[lo:hi].sum(axis=0)
            if counts.max() > 0:
                j = int(np.argmax(counts))
                held[lo : hi + 1, j] = True
                if hi < n:  # released at frame hi: motion starting within the settling
                    settling[hi:, j] |= steps[hi:-1] - steps[hi] < config.settle_steps
    position = truth.poses[..., :3]
    d = np.diff(position, axis=0)
    step = np.linalg.norm(d, axis=2)
    carried = held[1:] | held[:-1]
    riding = np.zeros_like(carried)
    for h in range(m):
        apart = np.linalg.norm(d - d[:, h : h + 1], axis=2)
        along = apart < 0.5 * step[:, h : h + 1] + jitter
        along[:, h] = False
        riding |= along & (carried[:, h] & (step[:, h] > 1e-4))[:, None]
    counted = ~(carried | settling | riding)
    moved = np.where(step > jitter, step, 0.0)
    touch = np.logical_or.reduce(list(truth.touched.values()))
    struck = counted & (touch[1:] | touch[:-1])
    objects = []
    for j, name in enumerate(truth.objects):
        net = float(np.linalg.norm(position[-1, j] - position[0, j]))
        ever = bool(held[:, j].any())
        objects.append(
            ObjectMotion(
                name=name,
                held=ever,
                payload=not ever and bool(riding[:, j].any()) and net > config.payload_m,
                free_m=float(moved[counted[:, j], j].sum()),
                strike_m=float(moved[struck[:, j], j].sum()),
                net_m=net,
            )
        )
    counted_objects = [o for o in objects if not o.payload]
    total = sum(o.free_m for o in counted_objects)
    strike = sum(o.strike_m for o in counted_objects)
    side = {name: _side(q) for name, q in truth.joints.items()}
    flags = []
    if strike > config.flag_strike_m:
        flags.append(f"arm strike {100 * strike:.1f} cm ({_worst(counted_objects, 'strike_m')})")
    if total > config.flag_disturbance_m:
        flags.append(f"disturbance {100 * total:.1f} cm ({_worst(counted_objects, 'free_m')})")
    flags += [
        f"{name}: {rad:.2f} rad on joints other than the one the task moves"
        for name, rad in side.items()
        if rad > config.flag_side_joint_rad
    ]
    return Disturbance(
        objects=objects, disturbance_m=total, strike_m=strike, side_joints_rad=side, flags=flags
    )


def _side(q: Array) -> float:
    """Joint motion (start to end) on all joints but the one that moved most."""
    moved = np.abs(q[-1] - q[0])
    return float(moved.sum() - moved.max()) if moved.size else 0.0


def _worst(objects: list[ObjectMotion], field: str) -> str:
    worst = max(objects, key=lambda o: getattr(o, field))
    return f"most: {worst.name} {100 * getattr(worst, field):.1f} cm"
