"""Test support and conformance checks for learning protocols (docs/concepts/learning.md).

For tests on MockEnv without a model, where outcomes come from MockEnv's own judgement and
scripted brains decide from `Episode.extra` (the protocol's seed, attempt, revision), never from
counts or paths:

- `ScriptedEnv`: MockEnv that keeps the specs it opened, faults every open of an `open_faults`
  seed (`sim_fault`: a scene that cannot be set up), judges an episode it was told to `fault` as
  faulted, and fails to diagnose the seeds in `undiagnosable`.
- `Lifter`: the tools `lift` (grasp a box from above and lift it, finding the box through the
  env's ground truth: a test oracle no real stack mounts; the scene registry's grasp relations
  follow it, `toolsets.scene.after_action`) and `fault` (make the env fault); the Actor of
  `testing.phases.TestMethod` sees both in S1, S2 and P3.
- `write_demo_package`: a small demo package of lift_red, with or without operations and goal.
- `lift_memory`: a Memory for MockEnv's lift tasks whose `lift_height` decides success (5 cm).
- `grounding`: a scripted grounding model's answers on MockEnv (for the perception toolset).
- Brains: `acting(plan)` (an Actor that lifts, finishes, faults the env or meets a gateway error,
  as `plan(episode)` says), `lifting_actor` (lifts by the Memory's height), `finishing`,
  `editing(ops)`.

A protocol conforms when it runs a task on MockEnv with scripted brains and leaves every episode
it ran as a directory with its summary and events under the task's directory; `evaluate`-like
protocols (`writes_memory=False`) leave the Memory versions as they found them. Use the checks
from pytest by subclassing `ProtocolConformance` as a `Test*` class with a `protocol_target`
fixture, or without pytest via `roboscript.testing.precheck`.
"""

import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import numpy as np
from pydantic import BaseModel, Field

from roboscript.agent import LoopConfig, ModelError, ScriptedBrain, call
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import (
    Diagnosis,
    EpisodeOpened,
    EpisodeSpec,
    JudgeResult,
    ObserveRequest,
)
from roboscript.contracts.errors import InfraError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import BatchId, EpisodeId, TaskName, new_id
from roboscript.contracts.observation import FrameEntry
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm
from roboscript.demos import BLOBS_DIR, Keypoint, KeypointSet, Tolerance
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import COLORS, MockConfig, MockEnv
from roboscript.learning import (
    DemoPackage,
    Episode,
    EpisodeOutcome,
    Keyframe,
    Operation,
    ProtocolRunner,
    save_package,
)
from roboscript.manipulation.receipts import grip_events
from roboscript.memory import (
    FileMemoryStore,
    MemoryDoc,
    NumberParam,
    VersionSource,
    WorkflowStage,
)
from roboscript.testing.perception import (
    asked_for,
    instruction_of,
    prompt_of,
    query_roles,
    som_cards,
    som_queries,
)
from roboscript.tools import Rationale, ToolContext, Toolset, tool
from roboscript.toolsets.scene import after_action

LIFT_RED: Final = TaskName("lift_red")
RATIONALE: Final = {"stage": "lift", "scene": "a box", "progress": "start", "plan": "lift it"}
NO_OBSERVE: Final = LoopConfig(observe=False)
Act = Literal["pass", "fail", "fault", "gateway"]


class ScriptedEnv(MockEnv):
    """MockEnv with test switches (see the module docstring); `opened` keeps every spec."""

    def __init__(self, config: MockConfig, blobs: LocalBlobStore) -> None:
        super().__init__(config, blobs)
        self.opened: list[EpisodeSpec] = []
        self.open_faults: set[int] = set()
        self.undiagnosable: set[int] = set()
        self._faulted: set[EpisodeId] = set()
        self._seeds: dict[EpisodeId, int] = {}

    async def open(self, spec: EpisodeSpec) -> EpisodeOpened:
        self.opened.append(spec)
        if spec.seed in self.open_faults:
            raise InfraError("sim_fault", "the scene could not be set up")
        opened = await super().open(spec)
        self._seeds[opened.episode_id] = spec.seed
        return opened

    def fault(self, episode_id: EpisodeId) -> None:
        self._faulted.add(episode_id)

    async def judge(self, episode_id: EpisodeId) -> JudgeResult:
        judged = await super().judge(episode_id)
        return judged.model_copy(update={"faulted": episode_id in self._faulted})

    async def diagnose(self, episode_id: EpisodeId) -> Diagnosis:
        if self._seeds.get(episode_id) in self.undiagnosable:
            raise InfraError("sim_fault", "the diagnosis failed")
        return await super().diagnose(episode_id)


async def _lift_red(env: MockEnv, episode: EpisodeId, height_m: float, *, grasp: bool) -> None:
    """With `grasp`, take the red box from above with the arm on its side (gripper closed on it);
    else lift it by `height_m`: `Lifter.lift`'s steps in two halves."""
    x, y, z = (await env.gt_snapshot(episode)).objects["red"].p
    arm: Arm = "left" if x < 0 else "right"
    spec = load_embodiment(MOCK_BIMANUAL).arms[arm]
    qpos = list((await env.robot(episode)).qpos_cmd)

    def at(ee_z: float, grip: float) -> list[float]:
        q = list(qpos)
        q[spec.action[0] : spec.action[0] + 6] = [x, y, ee_z, 0.0, math.pi / 2, 0.0]
        q[spec.gripper_index] = grip
        return q

    if grasp:  # the TCP is 0.1 m below the ee
        steps = [
            ActionStep(qpos=at(z + 0.2, 1.0)),
            ActionStep(qpos=at(z + 0.1, 1.0)),
            ActionStep(qpos=at(z + 0.1, 0.0), until_gripper_stable=True),
        ]
    else:  # from the grasp: x, y as they were; z the box's as taken
        steps = [ActionStep(qpos=at(z + 0.1 + height_m, 0.0))]
    await env.execute(episode, ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps))


class Lifter(Toolset):
    """`lift` and `fault` on `ScriptedEnv`s (by task), acting on the one that holds the
    episode, `lift` finding the box through its ground truth (a test oracle)."""

    name = "lifter"

    def __init__(self, envs: Mapping[str, ScriptedEnv]) -> None:
        self.envs = envs
        self.profile = load_embodiment(MOCK_BIMANUAL)

    def _env(self, ctx: ToolContext) -> ScriptedEnv:
        return next(e for e in self.envs.values() if e.episodes.current == ctx.env_episode)

    @tool(kind="execution", tier="D")
    async def lift(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        color: Annotated[str, Field(description="The box's colour.")],
        height_m: Annotated[float, Field(ge=0.0, le=0.3, description="How high to lift it.")],
    ) -> ActionReceipt:
        """Grasp the box of that colour from above and lift it by height_m meters."""
        truth = await self._env(ctx).gt_snapshot(ctx.env_episode)
        x, y, z = truth.objects[color].p
        arm: Arm = "left" if x < 0 else "right"
        spec = self.profile.arms[arm]
        start = spec.action[0]
        before = await ctx.env.robot(ctx.env_episode)
        qpos = list(before.qpos_cmd)

        def at(ee_z: float, grip: float) -> list[float]:
            q = list(qpos)
            q[start : start + 6] = [x, y, ee_z, 0.0, math.pi / 2, 0.0]  # pointing down
            q[spec.gripper_index] = grip
            return q

        steps = [  # the TCP is 0.1 m below the ee
            ActionStep(qpos=at(z + 0.2, 1.0)),
            ActionStep(qpos=at(z + 0.1, 1.0)),
            ActionStep(qpos=at(z + 0.1, 0.0), until_gripper_stable=True),
            ActionStep(qpos=at(z + 0.1 + height_m, 0.0)),
        ]
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)
        result = await ctx.env.execute(ctx.env_episode, batch)
        path = [state.robot.arms[arm].tcp for state in result.states]
        receipt = ActionReceipt(
            reason=ReceiptReason.ok,
            arm=arm,
            steps_used=result.executed,
            tcp_path={arm: path},
            grips=grip_events(before, result.states, self.profile),
        )
        await after_action(ctx, receipt)
        return receipt

    @tool(kind="execution", tier="D")
    async def fault(self, ctx: ToolContext, rationale: Rationale) -> ActionReceipt:
        """Make the simulator fault (a test switch)."""
        self._env(ctx).fault(ctx.env_episode)
        return ActionReceipt(reason=ReceiptReason.ok, note="the simulator faulted")


def lift_memory(height: float, task: str = LIFT_RED) -> MemoryDoc:
    """A lift task's Memory: one stage whose lift height is the parameter `lift_height`."""
    return MemoryDoc(
        task=task,
        instruction="Lift the box the instruction names.",
        objects="- box: the box the instruction names; find it by its colour.",
        workflow=[
            WorkflowStage(
                name="lift",
                goal="the box held above the table",
                arm="the arm on the box's side",
                call="lift(color, height_m={lift_height})",
                done="lifted(box, {lift_height})",
            )
        ],
        params={"lift_height": NumberParam(value=height, unit="m", bounds=(0.01, 0.2), trust=0.05)},
    )


def memory_value(request: Mapping[str, Any], name: str) -> float:
    """A number parameter's value as a prompt shows it (`- name = 0.02 m`)."""
    found = re.search(rf"- {name} = (-?[0-9.]+)", request["instructions"])
    if found is None:
        raise AssertionError(f"no parameter {name} in the prompt")
    return float(found[1])


def grounding(body: Mapping[str, Any]) -> Any:
    """A grounding model for MockEnv's tasks (`ScriptedModel(grounding)`): an instruction names
    its colour's block; Set-of-Mark picks the mark whose card has the colour its words name (with
    abstract roles, the instruction's colour: one role per instruction)."""
    if asked_for(body) == "roles":
        color = next(c for c in COLORS if c in prompt_of(body))
        return {"objects": [{"role": f"{color} block", "description": f"{color} block"}]}
    cards = som_cards(body)
    roles = query_roles(body)
    words = instruction_of(body) or ""
    out: dict[str, Any] = {}
    for key, text in som_queries(body).items():
        said = text if not roles else next((f"{c} box" for c in COLORS if c in words), "")
        mark = next((m for m, color in cards.items() if color in said), None)
        out[key] = {"phrase": said, "mark": mark} if roles else mark
    return out


def color_of(request: Mapping[str, Any]) -> str:
    """The box colour the task's instruction names (the prompt's `# Task` section)."""
    task = request["instructions"].split("# Task", 1)[1]
    found = re.search("|".join(COLORS), task)
    if found is None:
        raise AssertionError("the instruction names no box colour")
    return found[0]


def acting(
    plan: Callable[[Episode], Act], *, observe: bool = False
) -> Callable[[Episode], ScriptedBrain]:
    """An Actor doing what `plan(episode)` says: `pass` (lift the instruction's box 6 cm: the
    success latch ends the episode), `fail` (finish at once), `fault` (fault the env, then
    finish), `gateway` (a gateway failure at its first model call); with `observe`, each turn
    opens with an observation (a frame)."""

    def make(episode: Episode) -> ScriptedBrain:
        what = plan(episode)
        acted = False

        async def script(request: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal acted
            if what == "gateway":
                raise ModelError("gateway_error", "the gateway failed (a test)")
            if acted or what == "fail":
                return [call("finish", note="done")]
            acted = True
            if what == "fault":
                return [call("fault", rationale=RATIONALE)]
            return [call("lift", rationale=RATIONALE, color=color_of(request), height_m=0.06)]

        return ScriptedBrain(script, None if observe else NO_OBSERVE)

    return make


def lifting_actor(episode: Episode) -> ScriptedBrain:
    """An Actor that lifts the instruction's box by the Memory's `lift_height`, then finishes
    (unless the success latch has ended the episode)."""
    lifted = False

    async def script(request: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal lifted
        if lifted:
            return [call("finish", note="done")]
        lifted = True
        height = memory_value(request, "lift_height")
        return [call("lift", rationale=RATIONALE, color=color_of(request), height_m=height)]

    return ScriptedBrain(script)


def finishing(episode: Episode) -> ScriptedBrain:
    """A brain that finishes at once, without per-turn observations."""
    return ScriptedBrain([[call("finish", note="done")]], NO_OBSERVE)


def editing(ops: Callable[[Episode], Sequence[Mapping[str, Any]]]) -> Callable[..., ScriptedBrain]:
    """A Reviser (or Demo Reader) making the edits `ops(episode)` in one reply, then finishing."""

    def make(episode: Episode) -> ScriptedBrain:
        edits = [call("memory_edit", op=dict(op)) for op in ops(episode)]
        return ScriptedBrain([[*edits, call("finish", note="revised")]], NO_OBSERVE)

    return make


@dataclass(frozen=True)
class ProtocolTarget:
    """`runner()` makes a started-to-be `ProtocolRunner` over MockEnv's lift_red with scripted
    brains, whose store holds `start` as v000; `run(runner)` runs the protocol under test and
    returns its result; `run_dir` is the runner's directory."""

    runner: Callable[[], ProtocolRunner]
    run: Callable[[ProtocolRunner], Awaitable[BaseModel]]
    run_dir: Path
    writes_memory: bool = True
    start: MemoryDoc = field(default_factory=lambda: lift_memory(0.06))


class ProtocolConformance:
    """The checks, as `test_*` methods taking a `ProtocolTarget`."""

    async def test_every_episode_leaves_its_record(self, protocol_target: ProtocolTarget) -> None:
        store = FileMemoryStore(protocol_target.run_dir)
        source = VersionSource(kind="s1")
        await store.commit(protocol_target.start, status="s1_draft", source=source)
        before = await store.versions(LIFT_RED)
        async with protocol_target.runner() as runner:
            result = await protocol_target.run(runner)
        assert isinstance(result, BaseModel)
        summaries = list((protocol_target.run_dir / LIFT_RED).rglob("summary.json"))
        assert summaries, "the protocol ran no episode"
        for path in summaries:
            outcome = EpisodeOutcome.model_validate_json(path.read_bytes())
            assert outcome.directory == path.parent
            assert (path.parent / "events.jsonl").is_file()
        after = await store.versions(LIFT_RED)
        if not protocol_target.writes_memory:
            assert [meta.hash for meta in after] == [meta.hash for meta in before]


async def write_demo_package(
    directory: Path, blobs: LocalBlobStore, *, ops: bool = True, lifted: float = 0.0
) -> None:
    """A demo package of lift_red at seed 7 in `directory`: three frames observed on MockEnv
    (through `blobs`), approach and grasp keypoints of the left arm and, with `ops`, the grasp's
    before and after views and the goal (without: a package with neither). With `lifted`,
    the red box is lifted that many meters before the last frame (as `Lifter.lift` does), which
    a third keypoint (lift) marks, so the demonstration ends with something moved; else the
    three frames show the same scene."""
    env = MockEnv(MockConfig(task=LIFT_RED, blob_dir=directory / "rec"), blobs)
    opened = await env.open(EpisodeSpec(task=LIFT_RED, seed=7))
    carried = LocalBlobStore(directory / BLOBS_DIR)
    arm: Arm = "left"
    if lifted:
        arm = (
            "left"
            if (await env.gt_snapshot(opened.episode_id)).objects["red"].p[0] < 0
            else "right"
        )
    frames, tcps = [], []
    for n in (1, 2, 3):
        if lifted and n > 1:
            await _lift_red(env, opened.episode_id, lifted, grasp=n == 2)
        seen = await env.observe(opened.episode_id, ObserveRequest())
        tcps.append(seen.robot.arms[arm].tcp)
        for ref in (b for c in seen.cameras.values() for b in (c.rgb, c.depth) if b):
            carried.put_bytes(await blobs.aget_bytes(ref), ref.media_type, ref.meta)
        blob = carried.put_bytes(seen.model_dump_json().encode(), "application/json")
        frames.append(
            FrameEntry(frame_id=seen.frame_id, alias=f"frame_{n}", step=n, observation=blob)
        )
    await env.close(opened.episode_id)
    fixed = Pose.from_list([0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0])

    def tcp(frame: int) -> Pose:  # where the arm was, in a lifting demonstration
        return tcps[frame] if lifted else fixed

    kps = [
        Keypoint(
            index=1,
            frame=0,
            tcp=tcp(0),
            gripper="open",
            cue="stop",
            phase="approach",
            tolerance=Tolerance(),
        ),
        Keypoint(
            index=2,
            frame=1,
            tcp=tcp(1),
            gripper="closed",
            cue="close",
            phase="grasp",
            tolerance=Tolerance(),
        ),
    ]
    if lifted:  # the demonstration's last keyframe shows the lift
        kps.append(
            Keypoint(
                index=3,
                frame=2,
                tcp=tcp(2),
                gripper="closed",
                cue="stop",
                phase="lift",
                tolerance=Tolerance(),
            )
        )
    keypoints = KeypointSet(
        task=LIFT_RED,
        seed=7,
        frame_steps=1,
        success_frame=2,
        config_hash="x",
        pointing={arm: (1.0, 0.0, 0.0)},
        arms={arm: kps},
    )
    image = carried.put_image(np.full((8, 8, 3), 200, dtype=np.uint8))

    def keyframe(caption: str) -> Keyframe:
        return Keyframe(caption=caption, images=[image, image])

    grasp = Operation(
        arm=arm,
        keypoint="K2",
        label="grasp",
        before=keyframe("left K2 grasp: before"),
        after=keyframe("left K2 grasp: after"),
    )
    save_package(
        directory,
        DemoPackage(
            task=LIFT_RED,
            seed=7,
            instruction="lift the red box",
            keypoints=keypoints,
            frames=frames,
            ops=[grasp] if ops else [],
            goal=keyframe("the goal") if ops else None,
        ),
    )
