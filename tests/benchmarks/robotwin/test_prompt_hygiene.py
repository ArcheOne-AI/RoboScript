"""No model-visible text of the RoboTwin stack is bound to a RoboTwin task.

A model's task-specific input comes only from the live instruction, the demonstration, the agent's
own Memory and its tools' results; the rest of what it reads is framework and recipe text, which
must hold on every task. Checked two ways:

- rendered as a model receives them, around sample data in neutral words (`texts`): every role's
  system prompt (the Actor under each stage's rules, the Demo Reader, the Reviser in S1 and S2),
  the Memory as each role reads it (`memory_read`, every section, of a draft, a Memory and an S2
  cache), every mounted tool's description and input schema (the Responses export), receipts and
  reach checks as the manipulation toolset renders them, the lint rules a rejected edit shows,
  the demonstration brief, the hinted Actor's keypoint progress, and the Reviser's briefs with
  sample tries (turns with an observation and a receipt, the keypoint report, a diagnosis, the
  cache's diff, a rejected edit);
- the fixed words of every message the framework builds for a model (`MESSAGE_SITES`: the Hub's
  refusals and transitions, the loop's, the runner's ends, every receipt reason's notes, tool
  results, the briefs): the string literals of those modules, docstrings aside.

The Memory header's task id and the instructions are the task's data, stripped before matching;
so are the names in the demonstration's ground-truth layer (`truth_layer`), which the information
boundary allows S1 roles. The denylist is every other task's id, and the words of the task names and
categories (`tasks.yaml`), of the object attributes the referee reads (`sim/referee.yaml`) and of
the simulator names the Memory lint refuses (the recipe's `simulator_name` rule), less
`ALLOWED`. Toolsets join through `scripts/run_s2.py`'s `toolsets()`; another text a model reads
joins `texts`, another module that builds one `MESSAGE_SITES`.
"""

import ast
import importlib
import importlib.util
import inspect
import itertools
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final, cast, get_args

import numpy as np
import pytest
import yaml

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionStep, StepState
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import Diagnosis, EnvControl, JointTruth, ObjectTruth
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.hub import EpisodeEnd, OpenEpisode
from roboscript.contracts.ids import EpisodeId, FrameId
from roboscript.contracts.observation import FrameEntry
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.contracts.tools import TextPart, ToolCall
from roboscript.demos import ArmProgress, Keypoint, KeypointSet, SceneTruth, Tolerance
from roboscript.demos.keypoints import Cue, Phase
from roboscript.demos.referee import Approach, KeypointProgress, describe
from roboscript.embodiments import EmbodimentProfile
from roboscript.hub import Hub
from roboscript.learning import (
    DemoPackage,
    EpisodeOutcome,
    LearningConfig,
    ProtocolRunner,
)
from roboscript.learning.demo import Keyframe, demo_truth
from roboscript.learning.opening import BRIEF
from roboscript.manipulation import GripCheck, Plan, receipt
from roboscript.memory import (
    CategoryParam,
    Editor,
    GraspSpec,
    GraspSpecParam,
    HingeSpec,
    Lesson,
    MemoryDoc,
    NumberParam,
    PlaceSpec,
    PlaceSpecParam,
    PlaceTarget,
    WorkflowStage,
    diff,
    load_recipe,
    object_roles,
    open_cache,
)
from roboscript.perception.config import PerceptionConfig
from roboscript.perception.registry import SceneRegistry
from roboscript.perception.vlm import (
    box_prompt,
    boxes_prompt,
    camera_words,
    roles_prompt,
    som_prompt,
    som_roles_prompt,
)
from roboscript.settings import repo_root
from roboscript.testing.scenes import TABLE_Z, Box, surface
from roboscript.tools import LocalProvider, to_responses_tool
from roboscript.toolsets.manipulation import (
    ManipulationConfig,
    PlannerConfig,
    Reach,
    receipt_text,
    render_reach,
    render_receipt,
)
from roboscript.toolsets.memory import MemoryTools, ReadSection
from roboscript.toolsets.privileged import Privileged
from roboscript.toolsets.sensing import (
    ArmReport,
    ImageShown,
    Observed,
    View,
    render_image,
    render_observed,
)
from roboscript_methods.wicl.diagnosis import Explained, explain
from roboscript_methods.wicl.imitate import progress_text
from roboscript_methods.wicl.roles import TryLog, demo_brief, reviser_brief, truth_brief
from roboscript_methods.wicl.toolsets.demo import DemoRelation, Fitted, render_relation
from roboscript_robotwin.benchmark import HERE, ROBOTWIN
from roboscript_robotwin.bindings import EXECUTOR
from roboscript_robotwin.config import LINT_OVERLAYS, with_perception_binding

EXAMPLE = HERE.parents[4] / "examples" / "simBenchmarks" / "RoboTwin"

TASK: Final = "beat_block_hammer"
INSTRUCTION: Final = "Grab the hammer with the left arm and beat the block with it."
"""The live instruction; `MEMORY_INSTRUCTION` the one the Memory learned from, the
demonstration's. Both are data."""
MEMORY_INSTRUCTION: Final = "Pick up the hammer and hit the block."
TASKS: Final = yaml.safe_load((HERE / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]
REFEREE: Final = yaml.safe_load((HERE / "sim" / "referee.yaml").read_text(encoding="utf-8"))[
    "tasks"
]
ALLOWED: Final = {
    "left": "an arm's name",
    "right": "an arm's name",
    "object": "what the actions act on: an object role, a place target's object",
    "target": "where a motion or a placement goes",
    "actor": "the Actor role; the actions grasp_object and place_object",
    "move": "the motion tools (move_to_pose); moving a workflow stage",
    "place": "placing: place specs, place_object, a keypoint phase",
    "lift": "the geometry tool's name (a region's pixels to world points), a generic term; a "
    "keypoint phase",
    "open": "a gripper's state",
    "turn": "the turn loop",
    "empty": "the receipt reason empty_grasp",
    "box": "a box in an image, in pixels",
    "size": "an image's size",
    "away": "a direction: the world axes' wording, a receipt's distance from its target",
    "last": "plain English: the last observation, the last rejected edits",
    "two": "a number",
    "three": "a number",
    "pick": "generic manipulation verbs used as GraspSpec purposes",
    "press": "generic manipulation verbs used as GraspSpec purposes",
    "rotate": "the action rotate_about (a gripper turn about a hinge)",
}
"""Benchmark words the framework cannot do without, each with the reason."""
MESSAGE_SITES: Final = (
    # The Hub: the pipeline's and the steps' refusals, fail-fast and redo notes, the ended
    # message, aliases, `finish`, results as text.
    "roboscript.hub.pipeline",
    "roboscript.hub.steps",
    "roboscript.contracts.hub",
    "roboscript.hub.store",
    "roboscript.hub.builtins",
    "roboscript.hub.render",
    # The own loop: nudges, refusals, its ends, the history it replays.
    "roboscript.agent.loop",
    "roboscript.agent.history",
    # The runner: episode ends, failure labels, the briefs, the keypoint progress and report.
    "roboscript.learning.episode",
    "roboscript.learning.ledger",
    "roboscript_methods.wicl.ledger",
    "roboscript_methods.wicl.roles",
    "roboscript_methods.wicl.diagnosis",
    "roboscript_methods.wicl.imitate",
    "roboscript_methods.wicl.practice",
    "roboscript.demos.referee",
    # Receipts: every reason and its notes, from execution and from refusals before it.
    "roboscript.contracts.receipt",
    "roboscript.manipulation.receipts",
    "roboscript.manipulation.executor",
    # Tool results: frames, sensing, actions and reach checks, memory reads and edits, the
    # linter's findings.
    "roboscript.tools.frames",
    "roboscript.toolsets.sensing",
    "roboscript.toolsets.manipulation",
    "roboscript.toolsets.manipulation.actions",
    "roboscript.toolsets.manipulation.results",
    "roboscript.toolsets.memory",
    "roboscript_methods.wicl.toolsets.demo",
    "roboscript_methods.wicl.relation",
    "roboscript.memory.edits",
    "roboscript.memory.linter",
    "roboscript.memory.cache",
    "roboscript.memory.objects",
    # Perception: the identification's outcome, the object reports, the scene toolsets' results
    # and refusals, the grounding model's prompts, the brief of an identification.
    "roboscript.perception.identify",
    "roboscript.perception.vlm",
    "roboscript.perception.overlay",
    "roboscript.toolsets.scene",
    "roboscript.toolsets.perception",
    "roboscript.toolsets.geometry",
    "roboscript.toolsets.worldstate",
    "roboscript.learning.opening",
    # The env's diagnosis notes, which reach the Reviser.
    "roboscript_robotwin.sim.simulator",
)
"""Modules that build text for a model; their string literals are its fixed words."""
BLOB: Final = BlobRef(id="0" * 64, size=0, media_type="image/png")
PLANNED: Final = ManipulationConfig(planner=PlannerConfig(url="http://127.0.0.1:18300"))
TCP: Final = Pose(p=(0.1, -0.1, 0.8), q=(0.7071068, 0.0, 0.7071068, 0.0))
EE: Final = Pose(p=(0.1, -0.1, 0.92), q=(0.7071068, 0.0, 0.7071068, 0.0))


def _words(name: str) -> set[str]:
    """`target_box` -> target, box; an index dropped: `block1` -> block, `bread[0]` -> bread."""
    found = (re.sub(r"(?<=[a-z])\d+$", "", t) for t in re.findall(r"[a-z0-9]+", name.lower()))
    return {word for word in found if not word.isdigit()}


SIMULATOR: Final = frozenset(
    word.lower()
    for word in load_recipe(repo_root() / "recipes" / "wicl", repo_root() / LINT_OVERLAYS["memory"])
    .lint.rules["simulator_name"]
    .words
)
"""What the simulator's software is called: robotwin, sapien, mplib."""
DENIED: Final = frozenset(
    {word for task in TASKS for word in _words(task)}
    | {word for entry in REFEREE.values() for name in entry["objects"] for word in _words(name)}
    | SIMULATOR
)
CATEGORIES: Final = frozenset(info["category"] for info in TASKS.values())
"""Matched as written (`C1`): a cache is `c1`."""


def _forms(token: str) -> tuple[str, ...]:
    """A token as itself, as an indexed name (`block1`) and as a plural (`hammers`)."""
    return token, re.sub(r"(?<=[a-z])\d+$", "", token), token.removesuffix("s")


def scan(text: str, task: str = TASK, data: Sequence[str] = ()) -> tuple[list[str], set[str]]:
    """What `text` holds of the benchmark once its data (the instructions, the Memory header's
    task id, `data`) are stripped: other tasks' ids and denied words, each with its context; and
    the allowed words it uses."""
    for datum in sorted((INSTRUCTION, MEMORY_INSTRUCTION, *data), key=len, reverse=True):
        text = text.replace(datum, " ")
    written = text.replace(f"# TASK MEMORY: {task}", "# TASK MEMORY:")
    text = written.lower()

    def context(match: re.Match[str]) -> str:
        around = match.string[max(0, match.start() - 30) : match.end() + 30]
        return f"{match[0]!r} in '...{' '.join(around.split())}...'"

    ids = (rf"(?<![a-z0-9]){re.escape(other)}(?![a-z0-9])" for other in TASKS if other != task)
    hits = [context(match) for pattern in ids for match in re.finditer(pattern, text)]
    codes = (rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])" for code in CATEGORIES)
    hits += [context(match) for pattern in codes for match in re.finditer(pattern, written)]
    used = set()
    for match in re.finditer(r"[a-z0-9]+", text):
        forms = set(_forms(match[0])) & DENIED
        used |= forms & ALLOWED.keys()
        if forms - ALLOWED.keys():
            hits.append(context(match))
    return hits, used


def literals(module: str) -> str:
    """A module's string literals, one per line, without the bare strings (docstrings)."""
    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    bare = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
    return "\n".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare
    )


def truth_layer() -> dict[str, tuple[str, list[str]]]:
    """The demonstration's ground-truth layer as S1 roles read it (`truth_brief` of the sample
    demonstration, every keypoint's moment and an articulated object's joints), by where it
    shows, with its data: the simulator's object names and categories, which the information
    boundary allows S1 roles. Its framing is scanned like the rest."""
    package = sample_package()
    frames = (package.keypoints.success_frame or 0) + 1
    names = ["hammer", "block"]
    poses = np.tile([0.0, 0.1, 0.76, 1.0, 0.0, 0.0, 0.0], (frames, len(names), 1))
    truth = SceneTruth(
        objects=names,
        poses=poses,
        touched={},
        joints={"block": np.zeros((frames, 1))},
        categories={name: name for name in names},
    )
    layer = demo_truth(package.keypoints, truth)
    return {"ground-truth layer": (_text(truth_brief(layer)), [*layer.objects, *names])}


# Samples: the data a model's text is rendered around, in neutral words.


def sample_memory() -> MemoryDoc:
    """Every section and parameter kind in neutral words, so that whatever benchmark word shows
    up around them is the framework's or the recipe's."""
    return MemoryDoc(
        task=TASK,
        instruction=MEMORY_INSTRUCTION,
        objects="- tool: the long item with a handle; find it by its colour\n- goal: what it hits",
        workflow=[
            WorkflowStage(
                name="grip",
                goal="hold the tool by its handle",
                arm="{arm_rule}",
                call=(
                    "register_part(obj=tool, part=handle); "
                    "grasp_object(arm=A, obj=tool, grasp={grip_spec})"
                ),
                done="holding(A)",
            ),
            WorkflowStage(
                name="strike",
                goal="bring the tool's head down onto the goal",
                arm="the arm holding the tool",
                call="place_object(arm=A, obj=tool, place={strike_spec})",
                done="near(A, goal, {near_m})",
            ),
        ],
        lessons=[
            Lesson(
                id="L1",
                when="the handle points toward the robot",
                do="come from the top",
                seed=10004,
                merges=1,
            )
        ],
        notes="Measure in the current scene: copy strategies, not numbers.",
        params={
            "near_m": NumberParam(
                value=0.04,
                unit="m",
                bounds=(0.01, 0.1),
                trust=0.02,
                frame="object",
                source="demo left K2",  # a spec the Demo Reader fitted
            ),
            "arm_rule": CategoryParam(value="nearest", choices=["nearest", "named"]),
            "grip_spec": GraspSpecParam(
                value=GraspSpec(
                    part="handle",
                    approach="top",
                    anchor="top_center",
                    x_toward="head",
                    offset_m=(-0.03, 0.0, -0.01),
                )
            ),
            "strike_spec": PlaceSpecParam(
                value=PlaceSpec(
                    target=PlaceTarget(kind="object", obj="goal", anchor="top_center"),
                    frame="world_axes",
                    offset_m=(0.0, 0.0, 0.05),
                    placement_point="part:head",
                )
            ),
            "rest_spec": PlaceSpecParam(
                value=PlaceSpec(
                    target=PlaceTarget(kind="landmark", landmark="table_center"),
                    orient="yaw",
                    yaw_deg=30.0,
                )
            ),
        },
    )


def sample_package() -> DemoPackage:
    """A demonstration with every keypoint phase and cue on the left arm and one keypoint on the
    right, a loaded frame and a keyframe."""
    cues = get_args(Cue)

    def keypoint(index: int, phase: Phase) -> Keypoint:
        held = phase in ("grasp", "lift", "carry", "place")
        return Keypoint(
            index=index,
            frame=10 * index,
            tcp=TCP,
            gripper="closed" if held else "open",
            cue=cues[index % len(cues)],
            phase=phase,
            tolerance=Tolerance(angle_deg=15.0 if index % 2 else None),
        )

    keypoints = KeypointSet(
        task=TASK,
        seed=7,
        frame_steps=1,
        success_frame=100,
        config_hash="sample",
        pointing={"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)},
        arms={
            "left": [keypoint(i, phase) for i, phase in enumerate(get_args(Phase), 1)],
            "right": [keypoint(1, "approach")],
        },
    )
    json_blob = BLOB.model_copy(update={"media_type": "application/json"})
    return DemoPackage(
        task=TASK,
        seed=7,
        instruction=MEMORY_INSTRUCTION,
        keypoints=keypoints,
        keyframes=[Keyframe(caption="left K2 (approach)", images=[BLOB, BLOB])],
        frames=[FrameEntry(frame_id=FrameId("f1"), alias="frame_1", step=0, observation=json_blob)],
    )


def sample_progress(package: DemoPackage) -> tuple[str, list[str]]:
    """The hinted Actor's keypoint progress after an action, and the Reviser's keypoint report:
    left K1 passed, K2 missed, K3 next; right pending."""
    near = Approach(
        sample=3, offset_m=(0.004, -0.002, 0.031), distance_m=0.0315, angle_deg=4.0, ratio=3.1
    )
    left = package.keypoints.arms["left"]
    states = [
        KeypointProgress(index=1, status="passed", at=2),
        KeypointProgress(index=2, status="missed", at=5, nearest=near),
        KeypointProgress(index=3, status="pending", nearest=near),
        *(KeypointProgress(index=kp.index, status="pending") for kp in left[3:]),
    ]
    progress = {
        "left": ArmProgress(keypoints=states, target=3),
        "right": ArmProgress(keypoints=[KeypointProgress(index=1, status="pending")], target=1),
    }
    report = [
        describe("left", kp, p) for kp, p in zip(left, states, strict=True) if p.status != "pending"
    ]
    return progress_text(package.keypoints, progress), report


def sample_receipt(profile: EmbodimentProfile) -> ActionReceipt:
    """A real receipt: the left gripper closes on nothing (`empty_grasp`)."""

    def robot(step: int, opening: float, command: float) -> RobotState:
        def arm(grip: float, cmd: float) -> ArmState:
            zeros = [0.0] * 6
            return ArmState(
                ee=EE, tcp=TCP, joints=zeros, joints_cmd=zeros, gripper=grip, gripper_cmd=cmd
            )

        arms = {"left": arm(opening, command), "right": arm(1.0, 1.0)}
        return RobotState(step=step, step_limit=400, arms=arms, qpos_cmd=[0.0] * 14)

    qpos = [0.0] * 13 + [1.0]  # the left gripper (index 6) closes, the right one stays open
    grip = GripCheck("left", 0, 0.0, grasp=True)
    plan = Plan(robot(10, 1.0, 1.0), [ActionStep(qpos=qpos)], grips=[grip], arms=["left"])
    settled = StepState(index=0, robot=robot(11, profile.arms["left"].gripper.empty, 0.0))
    return receipt(plan, [settled], "completed", profile, EXECUTOR)


def sample_try(
    directory: Path,
    attempt: int,
    label: str,
    end: str,
    grasped: ActionReceipt,
    diagnosis: Explained | None = None,
    keypoints: Sequence[str] = (),
) -> TryLog:
    """One try as the Reviser reads it: a turn with its observation, an image call and an
    action with its receipt, rendered as the model saw them; its end."""
    arms = {
        arm: ArmReport(ee=EE, tcp=TCP, gripper=1.0, gripper_cmd=1.0, holding=False)
        for arm in ("left", "right")
    }
    observed = Observed(
        frame="frame_1",
        step=10,
        step_limit=400,
        arms=arms,
        views=[View(camera="head_camera", image=BLOB)],
        overlays=["TCP dots"],
    )
    image = ImageShown(frame="frame_1", camera="head_camera", image=BLOB, overlays=[])
    rationale = {
        "stage": "grip",
        "scene": "the tool lies across the goal",
        "progress": "nothing held yet",
        "plan": "grip the handle from the top",
    }
    arguments = {"arm": "left", "obj": "obj_1", "grasp": "grip_spec", "rationale": rationale}
    calls = [
        ("get_image", {"frame": "frame_1", "camera": "head_camera"}, render_image(image)),
        ("grasp_object", arguments, render_receipt(grasped)),
    ]
    turn = {
        "turn": 1,
        "observation": [part.model_dump(mode="json") for part in render_observed(observed)],
        "exchanges": [
            {
                "text": "The tool lies across the goal; I grip its handle from the top.",
                "calls": [
                    {
                        "name": name,
                        "arguments": args,
                        "content": [part.model_dump(mode="json") for part in content],
                    }
                    for name, args, content in calls
                ],
            }
        ],
        "action": "grasp_object",
        "rationale": rationale,
        "end": {"reason": end, "note": "grasp_object (call-2)", "at": 0.0},
    }
    outcome = EpisodeOutcome(
        task=TASK,
        stage="s2",
        role="actor",
        seed=10004,
        directory=directory,
        episode_id=None,
        env_episode=None,
        instruction=INSTRUCTION,
        success=False,
        end_class="fail_fast" if end.startswith("fail_fast") else "policy",
        end=EpisodeEnd(reason=end, note="grasp_object (call-2)", at=0.0),
        steps=11,
        step_limit=400,
        turns=1,
    )
    return TryLog(
        attempt=f"try {attempt}",
        label=label,
        outcome=outcome,
        turns=[turn],
        aliases={"frame_1": f"log:{attempt}:1"},
        diagnosis=diagnosis,
        keypoints=keypoints,
    )


def sample_relations() -> str:
    """`demo_relation` at a grasp and at a placement, as the model reads them."""
    grip = GraspSpec(part="handle", approach="top", offset_m=(0.01, 0.0, 0.0), arm="left")
    rest = PlaceSpec(target=PlaceTarget(kind="object", obj="goal", anchor="top_center"))
    at_base = PlaceSpec(target=PlaceTarget(kind="robot_frame"), offset_m=(0.4, 0.1, 0.8))
    grasp = DemoRelation(
        keypoint="left K2",
        phase="grasp",
        relation="grasp",
        obj="obj_1",
        fits=[
            Fitted(
                form="grasp", spec=grip, variant="top/0", position_error_m=0.0, angle_error_deg=0.3
            )
        ],
    )
    place = DemoRelation(
        keypoint="left K5",
        phase="release",
        relation="place",
        obj="obj_2",
        held="obj_1",
        fits=[
            Fitted(form="target", spec=rest, position_error_m=0.001, angle_error_deg=0.0),
            Fitted(form="robot_frame", spec=at_base, position_error_m=0.0, angle_error_deg=0.0),
        ],
    )
    hinge = HingeSpec(part="cover", side="back", at="bottom", offset_m=(0.0, 0.01, 0.0))
    turn = DemoRelation(
        keypoint="left K4",
        phase="carry",
        relation="turn",
        obj="obj_2",
        part="cover",
        fits=[
            Fitted(
                form="hinge",
                spec=hinge,
                angle_deg=60.0,
                steps=6,
                position_error_m=0.002,
                angle_error_deg=1.0,
            )
        ],
    )
    return "\n".join(_text(render_relation(r)) for r in (grasp, place, turn))


def sample_explained() -> Explained:
    """A try's ground truth in its own terms: the referee's names of this task's objects, and an
    articulated object of another task that the try did not register, matched to registered
    objects in neutral roles (one of them matching nothing); none of the referee's names may show
    once rendered."""
    registry = SceneRegistry()
    for role, (x, y) in (("tool", (0.10, 0.0)), ("goal", (0.30, 0.10)), ("spare", (-0.20, 0.10))):
        points = surface(Box((x, y, TABLE_Z + 0.02), (0.04, 0.04, 0.04)))
        registry.register(role, points, frame=FrameId("f1"), step=0, table_z=TABLE_Z)
    first, second = REFEREE[TASK]["objects"]
    [hinged] = REFEREE["open_microwave"]["objects"]
    up, z = (1.0, 0.0, 0.0, 0.0), TABLE_Z + 0.02

    def truth(name: str, start: Vec3, end: Vec3, joints: list[JointTruth]) -> ObjectTruth:
        moved = float(np.linalg.norm(np.subtract(end, start)))
        return ObjectTruth(
            name=name,
            initial=Pose(p=start, q=up),
            final=Pose(p=end, q=up),
            displacement_m=moved,
            initial_center=start,
            final_center=end,
            joints=joints,
        )

    joint = JointTruth(initial=0.1, final=0.4, low=0.0, high=1.5)
    ground = Diagnosis(
        episode_id=EpisodeId("env-1"),
        success=False,
        objects=[
            truth(first, (0.115, 0.0, z), (0.115, 0.10, z + 0.05), []),
            truth(second, (0.30, 0.11, z), (0.30, 0.11, z), []),
            truth(hinged, (0.0, 0.25, z), (0.0, 0.25, z), [joint]),
        ],
        notes=[],
    )
    aliases = {"o1": "obj_1", "o2": "obj_2", "o3": "obj_3"}
    return explain(ground, registry, aliases.__getitem__, robot=None)


# What a model reads


def run_s2() -> Any:
    spec = importlib.util.spec_from_file_location("run_s2", EXAMPLE / "scripts" / "run_s2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def texts(runner: ProtocolRunner, blobs: LocalBlobStore, directory: Path) -> dict[str, str]:
    """Every text of the stack a model reads, by where it shows. The stack's toolsets are only
    mounted, not started, so none needs its services."""
    prompts, profile, recipe = runner.prompts, runner.profile, runner.memory
    draft, memory = MemoryDoc(task=TASK, instruction=MEMORY_INSTRUCTION), sample_memory()
    reading = Editor(draft, rules=recipe.rules("s1"), lint=recipe.lint)
    revising = Editor(memory, rules=recipe.rules("s1"), lint=recipe.lint)
    cache = open_cache(memory, budget=4, seed=10004, rules=recipe.rules("s2"), lint=recipe.lint)
    cache.edit({"op": "set_number", "param": "near_m", "value": 0.05, "why": "it stopped short"})
    rejected = cache.edit({"op": "set_number", "param": "near_m", "value": 0.09, "why": "closer"})
    cache.edit({"op": "scratch", "text": "the handle pointed toward the robot"})
    shown = {}
    for (name, doc), rules in itertools.product(
        (("draft", draft), ("Memory", memory)), prompts.stage_rules
    ):
        shown[f"Actor prompt ({rules}, {name})"] = prompts.actor(
            instruction=INSTRUCTION, memory=doc, rules=rules, profile=profile
        )
    shown["Demo Reader prompt"] = prompts.demo_reader(
        instruction=INSTRUCTION, memory=draft, profile=profile
    )
    for stage, editor in (("s1", revising), ("s2", cache)):
        shown[f"Reviser prompt ({stage})"] = prompts.reviser(
            instruction=INSTRUCTION, editor=editor, profile=profile
        )
    phases = runner.method.phases  # wicl's, as production opens its episodes
    async with Hub([LocalProvider(runner.memory_tools)], blobs=blobs, phases=phases) as hub:
        for stage, role, editor in (
            ("s1", "demo_reader", reading),
            ("s1", "reviser", revising),
            ("s2", "reviser", cache),
        ):
            shown |= await memory_reads(hub, runner.memory_tools, stage, role, editor)
    for name, tool in Hub(runner.providers, blobs=blobs, phases=phases).mounted.items():
        shown[f"tool {name}"] = "\n".join(_strings(to_responses_tool(tool.descriptor)))
    for name, rule in recipe.lint.rules.items():
        shown[f"lint rule {name}"] = f"{name}: {rule.message}"
    package = sample_package()
    progress, report = sample_progress(package)
    shown["demonstration brief"] = _text(demo_brief(package))
    shown["keypoint progress"] = progress
    grasped = sample_receipt(profile)
    shown["receipt"] = receipt_text(grasped)
    unreachable = Reach(
        reachable=False,
        reason=ReceiptReason.unreachable,
        targets={"left": TCP},
        steps=0,
        note="left: no plan to the target (unreachable) (tried 2 times)",
    )
    shown["reach check"] = _text(render_reach(unreachable))
    shown["demo relation"] = sample_relations()
    fail_fast = "fail_fast:empty_grasp@grip"
    attempt = sample_try(directory, 1, "v003", fail_fast, grasped, sample_explained(), report)
    shown["Reviser brief (s1)"] = _text(
        reviser_brief([attempt], editor=revising, diff=None, rejected=[], images=1)
    )
    tries = [
        sample_try(directory, 1, "v003", fail_fast, grasped, sample_explained()),
        sample_try(directory, 2, "c1", "step_limit", grasped),
    ]
    shown["Reviser brief (s2)"] = _text(
        reviser_brief(
            tries, editor=cache, diff=diff(memory, cache.doc), rejected=[rejected], images=1
        )
    )
    return shown | grounding()


def grounding() -> dict[str, str]:
    """What the grounding model reads (its prompts around a scene's attribute card) and what the
    Actor's brief says of the identification."""
    roles = object_roles(sample_memory())
    keyed = {f"r{i}": (role, text) for i, (role, text) in enumerate(roles.items(), 1)}
    words = PerceptionConfig.model_validate(with_perception_binding({}))
    scene = words.scene_words
    head, wrist = camera_words("head_camera", words), camera_words("left_camera", words)
    return {
        "grounding: Set-of-Mark with roles": som_roles_prompt(
            "<card>", keyed, INSTRUCTION, head, scene
        ),
        "grounding: Set-of-Mark": som_prompt("<card>", {"r1": "<description>"}, wrist, scene),
        "grounding: boxes with roles": boxes_prompt(keyed, INSTRUCTION, 640, 480, head, scene),
        "grounding: box": box_prompt("<description>", 320, 240, wrist, scene),
        "grounding: roles": roles_prompt(INSTRUCTION, ROBOTWIN.profile().description),
        "brief: identification": BRIEF,
    }


async def memory_reads(
    hub: Hub, tools: MemoryTools, stage: str, role: str, editor: Editor
) -> dict[str, str]:
    """What `memory_read` shows `role` of `editor`'s document, section by section."""
    session = await hub.open_episode(OpenEpisode(stage=stage, role=role))
    tools.attach(session.episode_id, editor)
    shown = {}
    for n, section in enumerate((None, *get_args(ReadSection))):
        arguments = {"section": section}
        call = ToolCall(
            call_id=f"read-{n}", name="memory_read", arguments=arguments, origin="model"
        )
        outcome = await hub.call(session.episode_id, call)
        assert not outcome.is_error, outcome.content
        shown[f"memory_read {section or 'all'} ({stage} {role})"] = _text(outcome.content)
    tools.detach(session.episode_id)
    await hub.close_episode(session.episode_id)
    return shown


def _text(parts: Sequence[Any]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextPart))


def _strings(value: Any) -> Iterator[str]:
    """Every string of a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _no_brain(episode: Any) -> Any:
    raise AssertionError("no episode runs here")


async def test_no_model_visible_text_is_bound_to_a_task(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    # the pipeline's S1-only tier-P toolset
    truth = LocalProvider(Privileged(cast(EnvControl, None)))
    runner = ProtocolRunner(  # the practice's prompts, recipe, profile and toolsets
        LearningConfig(run_dir=tmp_path, method="wicl", recipe=Path("recipes/wicl")),
        providers=[*run_s2().toolsets(PLANNED, demo=True), truth],
        blobs=blobs,
        controls={},
        embodiment=ROBOTWIN.embodiment,
        brains={"actor": _no_brain},  # the simulator env has no entry points to load brains from
    )
    shown = await texts(runner, blobs, tmp_path)
    assert {"tool observe", "tool together", "tool memory_edit", "tool finish"} <= shown.keys()
    assert "Reviser prompt (s2)" in shown and "memory_read diff (s2 reviser)" in shown
    assert {"tool identify", "tool check", "grounding: Set-of-Mark with roles"} <= shown.keys()
    assert {"tool grasp_object", "tool place_object", "tool rotate_about"} <= shown.keys()
    assert {"tool demo_relation", "demo relation", "tool true_objects"} <= shown.keys()
    assert "Keypoint report" in shown["Reviser brief (s1)"]
    lines = shown["Reviser brief (s1)"].splitlines()
    heading = next(i for i, line in enumerate(lines) if line.startswith("## "))
    assert lines[heading + 1].startswith("Keypoint report")  # S1's brief leads with it
    assert "Ground truth after the try" in shown["Reviser brief (s2)"]
    assert "closed on nothing" in shown["Reviser brief (s2)"]  # the receipt's note
    scanned = [(where, text, ()) for where, text in shown.items()]
    scanned += [(f"literals of {module}", literals(module), ()) for module in MESSAGE_SITES]
    scanned += [(where, text, data) for where, (text, data) in truth_layer().items()]
    found: dict[str, list[str]] = {}  # a hit -> where it shows
    used = set()
    for where, text, data in scanned:
        hits, allowed = scan(text, data=data)
        used |= allowed
        for hit in hits:
            found.setdefault(hit, []).append(where)
    report = "\n".join(f"{hit} (in {', '.join(where)})" for hit, where in found.items())
    assert not found, f"benchmark words in model-visible text:\n{report}"
    assert used == ALLOWED.keys(), f"no text uses {sorted(ALLOWED.keys() - used)}: drop them"


def test_the_scan() -> None:
    assert {"hammer", "bell", "coaster", "microwave", "alarmclock"} <= DENIED
    assert {"robotwin", "sapien", "mplib"} <= SIMULATOR <= DENIED
    assert {"C1", "C2", "C3", "C4", "C5"} == CATEGORIES
    assert ALLOWED.keys() <= DENIED  # the allowlist only exempts benchmark words
    data = f"# TASK MEMORY: {TASK}   (v003, S2, -)\n\n{INSTRUCTION}\n{MEMORY_INSTRUCTION}"
    assert scan(data) == ([], set())
    hits, used = scan("Move the hammer_head over the blocks, as in click_bell, a C3 task.")
    assert [hit.split(" in ")[0] for hit in hits] == [
        "'click_bell'",
        "'C3'",
        "'hammer'",
        "'blocks'",
        "'click'",
        "'bell'",
    ]
    assert used == {"move"} and scan("the cache c1") == ([], set())
    assert scan("the hammers and block2")[0] and scan(TASK, task="click_bell")[0]
    assert scan("hammer (hammer) at frame_1", data=["hammer"]) == ([], set())
    words = literals("roboscript.hub.builtins")  # docstrings aside
    assert "Finished: the episode is over." in words and "The Hub's own tools" not in words


def test_the_referee_knows_what_the_agent_side_does_not() -> None:
    """The arm rules of the success checks live beside the referee; the agent side's task
    metadata is the step limit and the category."""
    assert set(REFEREE) == set(ROBOTWIN.tasks) and all(t["objects"] for t in REFEREE.values())
    assert "right gripper" in REFEREE["handover_block"]["notes"]
    fields = {"step_limit", "category"}
    assert all(set(info.model_dump()) == fields for info in ROBOTWIN.tasks.values())


PRIVILEGED: Final = re.compile(
    r"ground[ -]?truth|simulator|keypoint|referee|true_objects|demo_relation|privileged", re.I
)
"""What only S1 may read: the simulator's truth, the demonstration's keypoints and fits."""
DIAGNOSIS: Final = "Ground truth after the try"
"""The S2 Reviser's rewritten diagnosis, the one privileged item S2 allows."""


def _without_diagnosis(text: str) -> str:
    """`text` less the diagnosis blocks (from their heading to the next blank line)."""
    kept, skipping = [], False
    for line in text.splitlines():
        skipping = line.startswith(DIAGNOSIS) or (skipping and line.strip() != "")
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


S1_ONLY: Final = frozenset({"demo_relation", "true_objects"})
"""Tools only S1 (and the dev stage) shows: the demonstration's fits and the simulator's objects."""


async def test_s2_and_p3_roles_read_nothing_privileged(tmp_path: Path) -> None:
    """Every S2 and P3 role's actual tool list and prompts, with the stack's toolsets and the
    privileged truth toolset mounted: no tier-P or S1-only tool, and nothing privileged in what
    it reads but the S2 Reviser's diagnosis."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    truth = LocalProvider(Privileged(cast(EnvControl, None)))  # listed only, never called
    runner = ProtocolRunner(
        LearningConfig(run_dir=tmp_path, method="wicl", recipe=Path("recipes/wicl")),
        providers=[*run_s2().toolsets(PLANNED, demo=True), truth],
        blobs=blobs,
        controls={},
        embodiment=ROBOTWIN.embodiment,
        brains={"actor": _no_brain},
    )
    prompts, profile, recipe = runner.prompts, runner.profile, runner.memory
    memory = sample_memory()
    cache = open_cache(memory, budget=4, seed=10004, rules=recipe.rules("s2"), lint=recipe.lint)
    shown: dict[str, str] = {}
    # wicl's phases, as production opens its episodes; the runner's own memory tools too
    async with Hub(runner.providers, blobs=blobs, phases=runner.method.phases) as hub:
        for stage, role in (("s2", "actor"), ("p3", "actor"), ("s2", "reviser")):
            with pytest.raises(ProtocolError, match="runs at tier D only"):
                await hub.open_episode(OpenEpisode(stage=stage, role=role, tier="P"))
            session = await hub.open_episode(OpenEpisode(stage=stage, role=role))
            tools = await hub.tools(session.episode_id)
            assert all(t.meta.tier == "D" for t in tools), (stage, role)
            names = {t.name for t in tools} | set(session.system_tools)
            assert not names & S1_ONLY, (stage, role, sorted(names & S1_ONLY))
            for t in tools:
                shown[f"{stage} {role} tool {t.name}"] = "\n".join(_strings(to_responses_tool(t)))
    for rules in ("s2", "p3"):
        shown[f"Actor prompt ({rules})"] = prompts.actor(
            instruction=INSTRUCTION, memory=memory, rules=rules, profile=profile
        )
    shown["Reviser prompt (s2)"] = prompts.reviser(
        instruction=INSTRUCTION, editor=cache, profile=profile
    )
    grasped = sample_receipt(profile)
    tries = [
        sample_try(tmp_path, 1, "v003", "fail_fast:empty_grasp@grip", grasped, sample_explained()),
        sample_try(tmp_path, 2, "c1", "step_limit", grasped),
    ]
    brief = _text(
        reviser_brief(tries, editor=cache, diff=diff(memory, cache.doc), rejected=[], images=1)
    )
    assert DIAGNOSIS in brief  # the allowed item is there, and only it is left out below
    shown["Reviser brief (s2)"] = _without_diagnosis(brief)
    found = {
        where: sorted(
            {text[max(0, m.start() - 40) : m.end() + 40] for m in PRIVILEGED.finditer(text)}
        )
        for where, text in shown.items()
        if PRIVILEGED.search(text)
    }
    assert not found, f"privileged content in S2/P3 inputs: {found}"
