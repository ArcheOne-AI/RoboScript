"""learning.expert_diff: S1's privileged review of a failed attempt against the expert's state;
every verdict has its guidance."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import FrameId
from roboscript.contracts.robot import Arm
from roboscript.demos import ArmProgress, Keypoint, KeypointSet, ObjectShape, RefereeConfig
from roboscript.demos.keypoints import Tolerance
from roboscript.learning.demo import DemoTruth, TruthFrame
from roboscript.manipulation.specs import GraspSpec
from roboscript.memory import Editor, EditRules, MemoryDoc
from roboscript.memory.doc import WorkflowStage
from roboscript.memory.linter import LintConfig
from roboscript.memory.params import (
    CategoryParam,
    GraspSpecParam,
    NumberParam,
)
from roboscript.perception.registry import SceneRegistry
from roboscript.testing.scenes import TABLE_Z, Box, surface
from roboscript_methods.wicl.expert_diff import (
    Call,
    ExpertDiff,
    ReviewConfig,
    review,
)

S1_RULES = EditRules(max_numbers=None, trust_radius=False)
"""S1's edit rules, as wicl's recipe gives them (`recipes/wicl/memory.yaml`)."""

DOWN = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # the TCP's x (pointing) down
HOLD: dict[Arm, float] = {"left": 0.2, "right": 0.2}  # a settled opening holds at or above
BOX = (0.0, 0.0, 0.76)
TRAY = (0.2, 0.0, 0.75)
UP = (1.0, 0.0, 0.0, 0.0)
SHAPES = {
    "box": ObjectShape(symmetric=False, center=(0.0, 0.0, 0.0), size=(0.05, 0.04)),
    "tray": ObjectShape(symmetric=False, center=(0.0, 0.0, 0.0), size=(0.2, 0.02)),
}
FAR = {"left": (-0.3, -0.2, 0.95), "right": (0.3, -0.2, 0.95)}


def kp(index: int, frame: int, p: Sequence[float], cue: Any, phase: Any, grip: Any) -> Keypoint:
    return Keypoint(
        index=index,
        frame=frame,
        tcp=Pose(p=tuple(p), q=DOWN),
        gripper=grip,
        cue=cue,
        phase=phase,
        tolerance=Tolerance(),
    )


def expert(lift_q: tuple[float, ...] = DOWN) -> tuple[KeypointSet, DemoTruth]:
    """The left arm grasps the box from above, lifts it and sets it on the tray."""
    above = (BOX[0], BOX[1], BOX[2] + 0.1)
    lift = kp(3, 3, (0.1, 0.0, 0.9), "stop", "lift", "closed").model_copy(
        update={"tcp": Pose(p=(0.1, 0.0, 0.9), q=lift_q)}
    )
    keypoints = KeypointSet(
        task="lift_red",
        seed=7,
        frame_steps=1,
        success_frame=5,
        config_hash="synthetic",
        pointing={"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)},
        arms={
            "left": [
                kp(1, 1, above, "stop", "approach", "open"),
                kp(2, 2, BOX, "close", "grasp", "closed"),
                lift,
                kp(4, 4, (0.2, 0.0, 0.78), "open", "release", "open"),
            ]
        },
    )
    box, tray = Pose(p=BOX, q=UP), Pose(p=TRAY, q=UP)
    placed = Pose(p=(0.2, 0.0, 0.77), q=UP)
    frames = [
        TruthFrame(frame=0, what=["start"], poses={"box": box, "tray": tray}),
        TruthFrame(frame=1, what=["left K1"], poses={"box": box, "tray": tray}),
        TruthFrame(frame=2, what=["left K2"], poses={"box": box, "tray": tray}),
        TruthFrame(
            frame=3, what=["left K3"], poses={"box": Pose(p=(0.1, 0.0, 0.9), q=UP), "tray": tray}
        ),
        TruthFrame(frame=4, what=["left K4"], poses={"box": placed, "tray": tray}),
        TruthFrame(frame=5, what=["success"], poses={"box": placed, "tray": tray}),
    ]
    truth = DemoTruth(
        task="lift_red", seed=7, objects={"box": "box", "tray": "tray"}, frames=frames
    )
    return keypoints, truth


def recorded(steps: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Samples from per-sample states: `left`/`right` TCP positions (orientation `q_<arm>`,
    default DOWN), `grip_<arm>` commands, `box` position; truth at every sample."""
    data: dict[str, list[Any]] = {k: [] for k in ("step", "turn", "batch", "qpos_cmd")}
    for arm in ("left", "right"):
        for key in ("tcp", "gripper_cmd", "gripper"):
            data[f"{arm}.{key}"] = []
    data["truth.box"], data["truth.tray"] = [], []
    for n, state in enumerate(steps):
        data["step"].append(n)
        data["turn"].append(state.get("turn", n))
        data["batch"].append(n - 1)
        data["qpos_cmd"].append([0.0] * 14)
        for arm in ("left", "right"):
            q = state.get(f"q_{arm}", DOWN)
            data[f"{arm}.tcp"].append([*state.get(arm, FAR[arm]), *q])
            grip = state.get(f"grip_{arm}", 1.0)
            data[f"{arm}.gripper_cmd"].append(grip)
            data[f"{arm}.gripper"].append(state.get(f"open_{arm}", 0.9 if grip else 0.4))
        data["truth.box"].append([*state.get("box", BOX), *state.get("q_box", UP)])
        data["truth.tray"].append([*TRAY, *UP])
    out = {key: np.asarray(value, dtype=float) for key, value in data.items()}
    out["truth.sample"] = np.arange(len(steps))
    out["turn"] = out["turn"].astype(int)
    return out


def carried(arm: Arm, close: Sequence[float], to: Sequence[float]) -> list[dict[str, Any]]:
    """Approach, close at `close` (box under it), carry to `to` and release there."""
    box_at = tuple(close)
    above = (close[0], close[1], close[2] + 0.1)
    lift = (close[0], close[1], close[2] + 0.15)
    return [
        {},
        {arm: above},
        {arm: close, f"grip_{arm}": 0.0},
        {arm: lift, f"grip_{arm}": 0.0, "box": (box_at[0], box_at[1], box_at[2] + 0.15)},
        {arm: to, f"grip_{arm}": 0.0, "box": (to[0], to[1], to[2] - 0.01)},
        {arm: to, f"grip_{arm}": 1.0, "box": (to[0], to[1], to[2] - 0.01)},
    ]


SPEC = {"part": "whole", "approach": "top", "anchor": "top_center", "offset_m": [0, 0, 0]}


def memory() -> MemoryDoc:
    """The box's grasp, which the workflow's `grasp` stage passes by name."""
    grasp = GraspSpec.model_validate({**SPEC, "arm": "left"})
    stage = WorkflowStage(name="grasp", call="grasp_object(arm=<arm>, obj=box, grasp={box_grasp})")
    return MemoryDoc(
        task="lift_red",
        instruction="put the box on the tray",
        params={"box_grasp": GraspSpecParam(value=grasp, source="demo left K2")},
        workflow=[stage],
    )


async def run(
    steps: Sequence[dict[str, Any]],
    calls: Sequence[Call] = (),
    progress: dict[Arm, ArmProgress] | None = None,
    success: bool = False,
    plan: Any = None,
    lift_q: tuple[float, ...] = DOWN,
    doc: MemoryDoc | None = None,
    unreproducible: dict[str, str] | None = None,
) -> ExpertDiff:
    keypoints, truth = expert(lift_q)
    return await review(
        keypoints,
        truth,
        SHAPES,
        recorded(steps),
        calls,
        progress,
        success,
        doc or memory(),
        referee=RefereeConfig(),
        config=ReviewConfig(),
        grip_threshold=0.5,
        holding=HOLD,
        plan=plan,
        unreproducible=unreproducible,
    )


def registered(**objects: tuple[tuple[float, float, float], tuple[float, float, float]]) -> Any:
    """A first registry: blocks registered for roles, each (center, size)."""
    found = SceneRegistry()
    for role, (at, size) in objects.items():
        points = surface(Box(at, size))
        found.register(role, points, frame=FrameId("f0"), step=0, table_z=TABLE_Z)
    return found


def editor(reverts: dict[Any, Any], evidence: Sequence[str] = ()) -> Editor:
    doc = MemoryDoc(
        task="lift_red",
        instruction="lift",
        params={
            "lift_height": NumberParam(value=0.1, unit="m", bounds=(0.0, 1.0), trust=0.05),
            "frame": CategoryParam(value="world", choices=["world", "target"]),
        },
    )
    return Editor(
        doc, rules=S1_RULES, lint=LintConfig(), repair=True, reverts=reverts, evidence=evidence
    )


def _attempt(directory: Any, calls: Sequence[tuple[str, dict[str, Any], dict[str, Any]]]) -> Any:
    """An attempt's `turns.jsonl` and `tool_calls.jsonl` with one call per turn."""
    import json

    directory.mkdir(parents=True, exist_ok=True)
    turns, records = [], []
    for n, (name, arguments, result) in enumerate(calls, start=1):
        call = {"call_id": f"c{n}", "name": name, "arguments": arguments}
        turns.append({"turn": n, "exchanges": [{"call": 1, "calls": [call]}]})
        records.append({**call, "content": [], "result": result})
    for path, lines in (("turns.jsonl", turns), ("tool_calls.jsonl", records)):
        text = "".join(json.dumps(x) + "\n" for x in lines)
        (directory / path).write_text(text, encoding="utf-8")
    return directory


def articulated(drop_at: int | None) -> list[dict[str, Any]]:
    """The box never moves with the gripper (an articulated part's root); the gripper closes in
    turn 2 and holds (opening 0.22), then turns 3 and 4 move it, the hold breaking in turn
    `drop_at` (the opening falls to 0.18)."""
    steps: list[dict[str, Any]] = [
        {"turn": 0},
        {"turn": 1, "left": (0.0, 0.0, 0.86)},
        {"turn": 2, "left": BOX, "grip_left": 0.0, "open_left": 0.6},
        {"turn": 2, "left": BOX, "grip_left": 0.0, "open_left": 0.22},
        {"turn": 3, "left": (0.0, -0.02, 0.77), "grip_left": 0.0, "open_left": 0.22},
        {"turn": 3, "left": (0.0, -0.03, 0.77), "grip_left": 0.0, "open_left": 0.22},
        {"turn": 4, "left": (0.0, -0.04, 0.78), "grip_left": 0.0, "open_left": 0.22},
    ]
    if drop_at is not None:
        for state in steps:
            if state["turn"] >= drop_at:
                state["open_left"] = 0.18
    return steps


def moving(turn: int, name: str) -> Call:
    return Call(
        turn=turn,
        name=name,
        arguments={"arm": "left", "hinge": "{lid_hinge}", "rationale": {"stage": "lift"}},
        result={"reason": "lost_hold" if name == "rotate_about" else "ok", "arm": "left"},
        text="Receipt.",
    )


def test_every_verdict_has_its_guidance() -> None:
    from typing import get_args

    from roboscript_methods.wicl.expert_diff import VERDICT_TEXT, Verdict

    assert set(get_args(Verdict)) == set(VERDICT_TEXT)


def follow(turn: int, reason: str, start_m: float) -> Call:
    """A held path's call whose receipt says how far its start lay from the hold."""
    return Call(
        turn=turn,
        name="follow_path",
        arguments={"arm": "left", "path": "{lid_path}", "rationale": {"stage": "lift"}},
        result={"reason": reason, "arm": "left", "path": {"start_m": start_m}},
        text=f"Receipt: {reason}.",
    )
