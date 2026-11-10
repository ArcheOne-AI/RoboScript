"""Test support for the pipeline tests (test_pipeline.py), on MockEnv without the simulator:

- `TASKS`: MockEnv's lift_red and a lift_blue, which the jobs' env servers serve (`env.tasks`);
- `record_demo(task, directory)`: a demo package of the task's box lifted by 0.1 m at seed 7
  (MockEnv in this process), with its ground-truth layer;
- `OracleLift`: the `lifter` toolset's `lift` on a served env, finding the box through the env's
  control plane (the wiring's test-only `oracle`; no real stack mounts it);
- `wiring()`: scripted brains (a Demo Reader whose Memory lifts too little, an Actor lifting by
  the Memory's height, a Reviser that fixes the height), sensing, the oracle, a catalog that
  shows it, this script as the worker, and GPUs 0-7 with their memory free (`free_gpus`);
- `config(root, ...)`: a pipeline on MockEnv, `fake_motion.py` as each GPU's motion service and
  the launcher's config (`p3.yaml`) for `fake_robotwin`'s checkout, whose official scripts are
  `official_fakes`'s (each fails once when `RS_TEST_SERVER_FAILS` / `RS_TEST_CLIENT_FAILS` names
  a marker file that does not exist yet) and whose Agent host holds, then finishes;
- run as a script: `worker ...` runs a job as `run_pipeline.py --worker` does (`hang-once
  <marker> ...`: the first time, a worker that hangs, deaf to SIGTERM, without a heartbeat), and
  `launch <directory>` the pipeline in `directory`, with this wiring and P3's install recorded
  where `RS_TEST_P3_RECORD` says.
"""

import dataclasses
import hashlib
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Final

import anyio
import official_fakes
import yaml
from pydantic import Field
from robotwin_pipeline import PipelineConfig, Wiring, load_directory, run_pipeline, worker

from roboscript.agent.session import ModelClients
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EnvControl, EpisodeSpec, ObserveRequest
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import BatchId, TaskName, new_id
from roboscript.contracts.observation import FrameEntry
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm
from roboscript.demos import BLOBS_DIR, Keypoint, KeypointSet, Tolerance
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import TASKS as MOCK_TASKS
from roboscript.envs.mock import Lifted, MockConfig, MockEnv, MockTask
from roboscript.kernel import compose
from roboscript.learning import DemoPackage, DemoTruth, Keyframe, save_package
from roboscript.learning.demo import TRUTH_FILE, TruthFrame
from roboscript.learning.runtime import MethodRuntime
from roboscript.stack.gpus import GpuMemory
from roboscript.testing.phases import extend
from roboscript.testing.protocols import editing, lifting_actor
from roboscript.tools import LocalProvider, Rationale, ToolContext, Toolset, tool
from roboscript.toolsets.sensing import Sensing
from roboscript_methods.wicl.finish import DoneGate
from roboscript_methods.wicl.goal import demo_goal
from roboscript_methods.wicl.method import Wicl
from roboscript_robotwin.benchmark import HERE
from roboscript_robotwin.eval import install

TESTS: Final = Path(__file__).resolve().parent
REPO: Final = TESTS.parents[3]
HELPERS: Final = TESTS.parent
"""The RoboTwin tests' helper modules (`official_fakes`, `fake_motion`)."""
EXAMPLE: Final = REPO / "examples" / "simBenchmarks" / "RoboTwin"
"""Where `robotwin_pipeline` is."""
DEMO_SEED: Final = 7
TASKS: Final = {
    **MOCK_TASKS,
    TaskName("lift_blue"): MockTask(
        primary="lift the blue box", paraphrase="raise the blue block", success=Lifted(box="blue")
    ),
}
READER_OPS: Final = [
    {
        "op": "set_text",
        "section": "objects",
        "text": "- box: the box the instruction names.",
        "why": "demo",
    },
    {
        "op": "define_param",
        "name": "lift_height",
        "param": {
            "kind": "number",
            "value": 0.02,
            "unit": "m",
            "bounds": [0.01, 0.2],
            "trust": 0.05,
        },
        "why": "the expert lifted the box",
    },
    {
        "op": "add_stage",
        "stage": {
            "name": "lift",
            "goal": "the box held above the table",
            "arm": "the arm on the box's side",
            "call": "lift(color, height_m={lift_height})",
            "done": "lifted(box, {lift_height})",
        },
        "why": "the demonstration's one stage",
    },
]
FIX: Final = [
    {"op": "set_number", "param": "lift_height", "value": 0.1, "why": "success never latched"}
]
RATIONALE: Final = {"stage": "wait", "scene": "a table", "progress": "at rest", "plan": "stay"}
MOTION_SERVICE: Final = {
    "name": "motion",
    "argv": [str(HELPERS / "fake_motion.py"), "--config", "serve"],
    "interpreter": "agent",
}
"""MockPlanner as a GPU's motion service."""
"""A GPU's `vision` service for the scheduler (the jobs' toolsets here never call one)."""
FAILS_ONCE: Final = """import os
import sys
from pathlib import Path

from official_fakes import *

if __name__ == "__main__":
    marker = os.environ.get("{switch}")
    if marker and not Path(marker).exists():
        Path(marker).write_text("failed once")
        sys.exit("{main} fails once on purpose")
    {main}()
"""
"""An official script's stand-in that fails once when its switch names a marker file (importable,
as `roboscript_robotwin.eval.shard_client` imports the client's)."""
HOLD_THEN_FINISH: Final = [
    [{"name": "hold", "arguments": {"rationale": RATIONALE, "steps": 2}}],
    [{"name": "finish", "arguments": {"note": "done"}}],
]


class OracleLift(Toolset):
    """`lift` on a served MockEnv: the box's pose from the env's control plane (ground truth)."""

    name = "lifter"

    def __init__(self, control: EnvControl) -> None:
        self.control = control
        self.profile = load_embodiment(MOCK_BIMANUAL)

    @tool(kind="execution", tier="D")
    async def lift(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        color: Annotated[str, Field(description="The box's colour.")],
        height_m: Annotated[float, Field(ge=0.0, le=0.3, description="How high to lift it.")],
    ) -> ActionReceipt:
        """Grasp the box of that colour from above and lift it by height_m meters."""
        x, y, z = (await self.control.gt_snapshot(ctx.env_episode)).objects[color].p
        arm: Arm = "left" if x < 0 else "right"
        qpos = list((await ctx.env.robot(ctx.env_episode)).qpos_cmd)
        steps = [ActionStep(qpos=q) for q in _lift(qpos, self.profile.arms[arm], x, y, z, height_m)]
        steps[2] = steps[2].model_copy(update={"until_gripper_stable": True})
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)
        result = await ctx.env.execute(ctx.env_episode, batch)
        path = [state.robot.arms[arm].tcp for state in result.states]
        return ActionReceipt(
            reason=ReceiptReason.ok, arm=arm, steps_used=result.executed, tcp_path={arm: path}
        )


def _lift(
    qpos: list[float], spec: Any, x: float, y: float, z: float, height: float
) -> list[list[float]]:
    """Above the box, down to it open, closed, then up by `height` (the TCP is 0.1 m below the
    ee, pointing down)."""

    def at(ee_z: float, grip: float) -> list[float]:
        q = list(qpos)
        q[spec.action[0] : spec.action[0] + 6] = [x, y, ee_z, 0.0, math.pi / 2, 0.0]
        q[spec.gripper_index] = grip
        return q

    return [at(z + 0.2, 1.0), at(z + 0.1, 1.0), at(z + 0.1, 0.0), at(z + 0.1 + height, 0.0)]


async def free_gpus() -> dict[int, GpuMemory]:
    """GPUs 0-7 with 1 of 80 GiB in use (the tests' jobs use none)."""
    return {gpu: GpuMemory(used_mib=1024, total_mib=81920) for gpu in range(8)}


LIFTING = extend(Wicl, {"actor": ["lifter.lift"]}, after="sensing", only=["s1", "s2", "p3"])


class EndUnknown(DoneGate):
    """wicl's finish gate on episodes told nothing of the demonstration's end: these stacks
    mount no `check_done`, which the gate wants for an episode that knows it."""

    def episode(self, episode: Any, *args: Any, **kwargs: Any) -> Any:
        known = dataclasses.replace(demo_goal(episode.goal), views=None, end=())
        return super().episode(dataclasses.replace(episode, goal=known), *args, **kwargs)


class Ungated(LIFTING):  # type: ignore[misc, valid-type]
    """wicl with the oracle lift, its finish ungated (`EndUnknown`)."""

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        return dataclasses.replace(Wicl.runtime(recipe), finish=EndUnknown())


def wiring(models: ModelClients | None = None) -> Wiring:
    return Wiring(
        # the Mock's skillset on sensing: its guide is what wicl's prompts read
        toolsets=lambda env: [LocalProvider(Sensing(), None, ["mock"])],
        worker=[sys.executable, str(TESTS / "pipeline_fakes.py"), "worker"],
        models=models,
        oracle=lambda control: LocalProvider(OracleLift(control)),
        brains={
            "demo_reader": editing(lambda _: READER_OPS),
            "actor": lifting_actor,
            "reviser": editing(lambda _: FIX),
        },
        method=Ungated,
        gpu_memory=free_gpus,
    )


async def record_demo(task: TaskName, directory: Path) -> None:
    """The oracle lift of the task's box by 0.1 m at the demo seed, observed before and after
    every step, as a package with its ground-truth layer in `directory`."""
    recording = directory.parent / "recording"
    blobs = LocalBlobStore(recording)
    env = MockEnv(MockConfig(task=task, blob_dir=recording, tasks=TASKS), blobs)
    opened = await env.open(EpisodeSpec(task=task, seed=DEMO_SEED))
    episode, color = opened.episode_id, TASKS[task].success.box
    x, y, z = (await env.gt_snapshot(episode)).objects[color].p
    arm: Arm = "left" if x < 0 else "right"
    qpos = list((await env.robot(episode)).qpos_cmd)
    observations = [await env.observe(episode, ObserveRequest())]
    for n, q in enumerate(_lift(qpos, load_embodiment(MOCK_BIMANUAL).arms[arm], x, y, z, 0.1)):
        step = ActionStep(qpos=q, until_gripper_stable=n == 2)
        await env.execute(episode, ActionBatch(batch_id=BatchId(new_id("b")), steps=[step]))
        observations.append(await env.observe(episode, ObserveRequest()))
    assert (await env.judge(episode)).success
    await env.close(episode)
    frames = []
    for n, observation in enumerate(observations, 1):
        blob = await blobs.aput_bytes(observation.model_dump_json().encode(), "application/json")
        frames.append(
            FrameEntry(
                frame_id=observation.frame_id,
                alias=f"frame_{n}",
                step=observation.step,
                observation=blob,
            )
        )

    def keypoint(index: int, frame: int, cue: Any, phase: Any, gripper: Any) -> Keypoint:
        tcp = observations[frame].robot.arms[arm].tcp
        return Keypoint(
            index=index,
            frame=frame,
            tcp=tcp,
            gripper=gripper,
            cue=cue,
            phase=phase,
            tolerance=Tolerance(),
        )

    keypoints = KeypointSet(
        task=task,
        seed=DEMO_SEED,
        frame_steps=1,
        success_frame=4,
        config_hash="synthetic",
        pointing={"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)},
        arms={
            arm: [
                keypoint(1, 1, "stop", "approach", "open"),
                keypoint(2, 3, "close", "grasp", "closed"),
                keypoint(3, 4, "stop", "lift", "closed"),
            ]
        },
    )
    keyframes = []
    for kp in keypoints.arms[arm]:
        rgb = await blobs.aget_array(observations[kp.frame].cameras["overhead"].rgb)
        keyframes.append(
            Keyframe(caption=f"{arm} {kp.name} ({kp.phase})", images=[await blobs.aput_image(rgb)])
        )
    package = DemoPackage(
        task=task,
        seed=DEMO_SEED,
        instruction=opened.instruction or "",
        keypoints=keypoints,
        keyframes=keyframes,
        frames=frames,
    )
    save_package(directory, package)
    carried = LocalBlobStore(directory / BLOBS_DIR)  # a package carries what it references
    refs = [image for keyframe in keyframes for image in keyframe.images]
    refs += [entry.observation for entry in frames]
    refs += [b for o in observations for c in o.cameras.values() for b in (c.rgb, c.depth) if b]
    for ref in refs:
        carried.put_bytes(await blobs.aget_bytes(ref), ref.media_type)
    box = Pose.from_list([x, y, z, 1.0, 0.0, 0.0, 0.0])
    truth = DemoTruth(
        task=task,
        seed=DEMO_SEED,
        objects={color: "box"},
        frames=[TruthFrame(frame=0, what=["start"], poses={color: box})],
    )
    (directory / TRUTH_FILE).write_text(truth.model_dump_json(), encoding="utf-8")


def config(root: Path, tasks: list[str], variants: dict[str, Any], **more: Any) -> PipelineConfig:
    """A pipeline on MockEnv in `root` (its runs in `root/runs`, demos from `root/demos`, P3's
    config `root/p3.yaml`, P3 on block 1): GPUs 4 and 5 (MockEnv and MockPlanner use none), S1
    exits after one unhinted pass, S2 converges after two first-try passes (ungated: no
    `check_done` among these toolsets), respawns and restarts
    wait 0.2 s; `more` is merged in."""
    p3 = {
        "interpreters": {"agent": sys.executable, "sim": sys.executable},
        "sim_pythonpath": [
            str(REPO / "packages" / "core" / "src"),
            str(HERE.parent),
            str(TESTS),
            str(HELPERS),
        ],
        "host": {
            "recipe": str(REPO / "recipes" / "wicl"),
            "episodes": {
                "embodiment": str(HERE / "embodiments" / "aloha_agilex.yaml"),
                "bridge": {"observe_timeout_s": 10, "report_timeout_s": 30},
            },
            "toolsets": [{"name": "sensing"}, {"name": "hold"}],
            # its Actor holds in p3, ungated: wicl's P3 needs check_done mounted
            "method": "roboscript.testing.phases:TestMethod",
            "memory_access": "bound",  # the Memory of the task id
            "brain": {"name": "scripted", "config": {"replies": HOLD_THEN_FINISH}},
        },
        "server_ready_timeout_s": 60,
        "stop_grace_s": 5,
    }
    (root / "p3.yaml").write_text(yaml.safe_dump(p3), encoding="utf-8")
    values = {
        "root": str(root / "runs"),
        "benchmark": "mock",
        "tasks": tasks,
        "variants": variants,
        "protocols": {"imitate": {"exit_passes": 1}, "practice": {"stop_passes": 2}},
        "demo": str(root / "demos" / "{task}" / "package"),
        "done_goal": False,  # the jobs' toolsets here offer no check_done
        "learning": {
            "method": "wicl",
            "recipe": str(REPO / "recipes" / "wicl"),
            "max_infra_reruns": 1,
        },
        "env": {"tasks": {name: task.model_dump(mode="json") for name, task in TASKS.items()}},
        "interpreters": {"agent": sys.executable, "sim": sys.executable},
        "sim_pythonpath": [str(REPO / "packages" / "core" / "src"), str(HERE.parent)],
        "gpus": [4, 5],
        "max_per_gpu": 3,
        "max_runs": 4,
        "per_gpu": [MOTION_SERVICE],
        "env_ready_timeout_s": 60,
        "p3": {"config": str(root / "p3.yaml")},
        "respawn_waits_s": [0.2, 0.2],
        "service_restart_waits_s": [0.2],
        "min_free_gb": 0,
        "status_interval_s": 0.5,
    }
    return PipelineConfig.model_validate(compose(values, more))


def fake_robotwin(directory: Path) -> tuple[Path, install.Expected]:
    """A RoboTwin checkout whose official scripts are the stand-ins of `official_fakes`."""
    root = directory / "RoboTwin"
    for name in ("script", "policy", "task_config", "envs"):
        (root / name).mkdir(parents=True)
    scripts = {
        "policy_model_server.py": ("server_main", "RS_TEST_SERVER_FAILS"),
        "eval_policy_client.py": ("client_main", "RS_TEST_CLIENT_FAILS"),
    }
    for name, (main, switch) in scripts.items():
        (root / "script" / name).write_text(
            FAILS_ONCE.format(main=main, switch=switch), encoding="utf-8"
        )
    (root / "script" / "test_render.py").write_text(official_fakes.TEST_RENDER, encoding="utf-8")
    (root / "envs" / "_base_task.py").write_text("# the task base class\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "script", "envs"], ["commit", "-qm", "stand-ins"]):
        _git(root, *args)
    files = {
        f"script/{n}": hashlib.sha256((root / "script" / n).read_bytes()).hexdigest()
        for n in scripts
    }
    return root, install.Expected(commit=_git(root, "rev-parse", "HEAD"), files=files)


def _git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    install.record_path = lambda: Path(os.environ["RS_TEST_P3_RECORD"])
    if sys.argv[1] == "worker":
        raise SystemExit(worker.main(wiring(), sys.argv[2:]))
    if sys.argv[1] == "hang-once":
        marker = Path(sys.argv[2])
        if not marker.exists():
            marker.write_text("hung once", encoding="utf-8")
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(600)
        raise SystemExit(worker.main(wiring(), sys.argv[3:]))
    assert sys.argv[1] == "launch", sys.argv
    directory = Path(sys.argv[2])
    anyio.run(run_pipeline, load_directory(directory), wiring(), directory)
