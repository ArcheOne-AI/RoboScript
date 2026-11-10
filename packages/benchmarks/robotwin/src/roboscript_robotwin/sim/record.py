"""Re-recording a chosen demonstration at its seed with our cameras (docs/concepts/learning.md),
in the simulator env.

The episode opens the way S1's do (`Simulator.open`: our task config, eval mode, the expert
pre-run with retries, the demonstration's instruction reused), so the scene and the attributes some
tasks judge by are what S1 will see. A recorded joint trajectory is replayed, one
`take_action` per frame a strategy sends (`roboscript.demos.replay_frames`, `s1` first), and
after every action the recorder keeps the robot state, the three RGB-D cameras (observations
through the env host's own encoding, `roboscript_robotwin.frames`), the physics steps simulated so
far (`Simulator.stepping`) and, apart from them, the referee's ground truth (object poses,
articulation joints, arm contacts). Frame 0 is the start. The success frame is the first frame
whose action latched the task's success; the replay stops there, since RoboTwin's `take_action`
does nothing after the latch. SAPIEN's render cache is cleared every episode.

Replaying the recorded commands is deterministic here; re-running the expert is not (cuRobo), so
it is only done for `GLOB_TASKS`, which draw their object variant with
`np.random.choice(glob.glob(...))` in an order the filesystem decides: the expert data may show
another variant than `setup_demo(seed)` builds here. Their trajectory comes from re-running the
expert at the seed (`Simulator.expert_trajectory`, at the expert data's frame cadence), their
seen instruction from the scene of that run (`official.instruction`, drawn with seeded
generators).

`python -m roboscript_robotwin.sim.record` records one candidate in a process of its own
(`examples/simBenchmarks/RoboTwin/scripts/record_demos.py` runs it), which imports no learning
code.
"""

import argparse
import contextlib
import os
import random
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import NDArray

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.ids import BatchId, FrameId, TaskName
from roboscript.contracts.observation import FrameEntry
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos import (
    BLOBS_DIR,
    ArmTrack,
    CameraCalib,
    CameraTrack,
    Demo,
    DemoManifest,
    DemoMeta,
    DemosConfig,
    Provenance,
    SceneTruth,
    TaskDemo,
    arm_track,
    frame_alias,
    replay_frames,
    save_demo,
    save_frames,
    save_truth,
)
from roboscript.demos.manifest import content_sha256
from roboscript.demos.replay import commands
from roboscript.embodiments import EmbodimentProfile
from roboscript.settings import load_config_ref
from roboscript_robotwin.benchmark import ROBOTWIN, TASK_CONFIGS
from roboscript_robotwin.config import robotwin_root
from roboscript_robotwin.demos.expert import load_episode
from roboscript_robotwin.demos.records import Attempt, Outcome, RecordedTask, read, write
from roboscript_robotwin.frames import observation
from roboscript_robotwin.sim import official, referee
from roboscript_robotwin.sim.simulator import Simulator

GLOB_TASKS: Final = frozenset(
    {
        "place_a2b_left",
        "place_a2b_right",
        "place_object_scale",
        "place_object_stand",
        "put_object_cabinet",
    }
)
"""Tasks whose object variant RoboTwin draws from an unsorted `glob`, an upstream nondeterminism.
This is adapter data for re-recording a demonstration, on the simulator side. The Agent never
reads it."""


@dataclass(frozen=True)
class Source:
    """A joint trajectory to replay: action vectors (N, action_dim) at `frame_steps` physics
    steps a frame, the instruction that goes with it, and where it came from."""

    commands: NDArray[np.float64]
    frame_steps: int
    instruction: str
    provenance: Provenance


class Unsolvable(Exception):
    """The expert pre-run failed at the seed every time: S1 could not open it either."""


@dataclass(frozen=True)
class Recording:
    """A replay that latched success: the demo (with observations), its frames and truth."""

    demo: Demo
    frames: list[FrameEntry]
    truth: SceneTruth
    seconds: float


def record_task(
    sim: Simulator,
    entry: TaskDemo,
    episode: int,
    seed: int,
    expert_root: Path,
    out: Path,
    config: DemosConfig,
) -> RecordedTask:
    """Record the candidate `episode` (at `seed`) of the demonstration `entry` chose (the chosen
    one or one ranked after it) into `out`: `recording/`, `referee/truth.npz`, a glob task's
    `expert/`, and `record.json`, which keeps the attempts at earlier candidates. The expert
    pre-run must pass `prerun_checks` runs in a row first: S1 opens the demo seed at every
    attempt, and a seed whose pre-run often fails leaves S1 unable to open it. The strategies
    then run in order; a candidate that does not record joins the rejected ones."""
    rec = config.recording
    earlier = read(out)
    attempts = [] if earlier is None else list(earlier.attempts)
    notes = [] if earlier is None else list(earlier.notes)
    rejected = [] if earlier is None else list(earlier.rejected)
    glob = sim.task in GLOB_TASKS
    started = time.monotonic()
    if (passed := sim.prerun_passes(seed, rec.prerun_checks)) < rec.prerun_checks:
        attempts.append(_attempt(episode, seed, None, "pre-run failed", started))
        notes.append(f"episode {episode}: the expert pre-run passed {passed} runs in a row")
        return write(
            out,
            RecordedTask(
                task=sim.task,
                category=entry.category,
                recorded=False,
                episode=episode,
                seed=seed,
                source="expert_rerun" if glob else "expert_data",
                instruction=entry.instruction,
                attempts=attempts,
                notes=notes,
                rejected=[*rejected, episode],
            ),
        )
    started = time.monotonic()
    if glob:
        source = expert_source(sim, seed, rec.expert_attempts, episode, out / "expert")
    else:
        demo = load_episode(expert_root, sim.task, episode, sim.profile)
        steps, provenance = demo.meta.frame_steps, demo.meta.provenance
        source = Source(commands(demo, sim.profile), steps, demo.meta.instructions[0], provenance)
    base = {
        "task": sim.task,
        "category": entry.category,
        "episode": episode,
        "seed": seed,
        "source": "expert_rerun" if glob else "expert_data",
        "instruction": entry.instruction if source is None else source.instruction,
    }
    if source is None:
        attempts.append(_attempt(episode, seed, None, "expert failed", started))
        notes.append(f"episode {episode}: the expert failed {rec.expert_attempts} runs")
        return write(
            out,
            RecordedTask(
                **base,
                recorded=False,
                attempts=attempts,
                notes=notes,
                rejected=[*rejected, episode],
            ),
        )
    if episode != entry.episode:
        notes.append(f"recording episode {episode}, a candidate after the chosen {entry.episode}")
    spec = EpisodeSpec(
        task=sim.task,
        seed=seed,
        stage="demo",
        instruction=source.instruction,
        prerun_attempts=rec.prerun_attempts,
    )
    staging = out / "recording.partial"
    grip = config.keypoints.grip_threshold
    for strategy in rec.strategies:
        if staging.exists():
            shutil.rmtree(staging)
        started = time.monotonic()
        try:
            recording = replay(sim, source, spec, strategy, rec.tail_holds, grip, staging)
        except Unsolvable:
            attempts.append(_attempt(episode, seed, strategy, "pre-run failed", started))
            notes.append(f"episode {episode}: the pre-run failed {rec.prerun_attempts} times")
            break
        if recording is None:
            attempts.append(_attempt(episode, seed, strategy, "no success latch", started))
            continue
        meta = recording.demo.meta
        attempts.append(_attempt(episode, seed, strategy, "success", started, meta.frames))
        _keep(recording, staging, out)
        recorded = RecordedTask(
            **base,
            recorded=True,
            strategy=strategy,
            frames=meta.frames,
            success_frame=meta.success_frame,
            attempts=attempts,
            notes=notes,
            rejected=rejected,
        )
        return write(out, recorded)
    if staging.exists():
        shutil.rmtree(staging)
    failed = RecordedTask(
        **base, recorded=False, attempts=attempts, notes=notes, rejected=[*rejected, episode]
    )
    return write(out, failed)


def _attempt(
    episode: int,
    seed: int,
    strategy: str | None,
    outcome: Outcome,
    started: float,
    frames: int | None = None,
) -> Attempt:
    return Attempt(
        episode=episode,
        seed=seed,
        strategy=strategy,
        outcome=outcome,
        frames=frames,
        seconds=time.monotonic() - started,
    )


def _keep(recording: Recording, staging: Path, out: Path) -> None:
    """The staged recording becomes `out/recording`, its ground truth `out/referee` (replacing an
    earlier candidate's)."""
    save_demo(recording.demo, staging)
    save_frames(staging, recording.frames)
    for old in (out / "recording", out / "referee"):
        if old.exists():
            shutil.rmtree(old)
    save_truth(recording.truth, out / "referee")
    staging.rename(out / "recording")


def expert_source(
    sim: Simulator, seed: int, attempts: int, episode: int, directory: Path
) -> Source | None:
    """Re-run the expert at `seed` (a candidate of the expert data's `episode`); its run is
    saved in `directory` as a demo, the reference of the recording's keypoints. None if every
    run failed."""
    run = sim.expert_trajectory(seed, attempts)
    if run is None:
        return None
    states, info = run
    instruction = seen_instruction(sim.task, info, seed)
    steps = int(yaml.safe_load(sim.task_config.read_text(encoding="utf-8"))["save_freq"])
    script = sim.root / "envs" / f"{sim.task}.py"
    demo = Demo(
        DemoMeta(
            task=sim.task,
            seed=seed,
            instructions=[instruction],
            frames=len(states),
            frame_steps=steps,
            ee_to_tcp={arm: spec.ee_to_tcp for arm, spec in sim.profile.arms.items()},
            provenance=Provenance(
                source="robotwin.expert_rerun",
                path=script,
                sha256=content_sha256(script),
                episode=episode,
            ),
        ),
        tracks(states, sim.profile),
    )
    save_demo(demo, directory)
    arrays = directory / "arrays.npz"
    return Source(
        commands=np.array([state.qpos_cmd for state in states]),
        frame_steps=steps,
        instruction=instruction,
        provenance=Provenance(
            source="rs_robotwin.expert_rerun",
            path=arrays,
            sha256=content_sha256(arrays),
            episode=episode,
        ),
    )


def seen_instruction(task: str, info: Any, seed: int) -> str:
    """The official generator's seen instruction for the expert's episode `info`, drawn with
    Python's and numpy's generators seeded by `seed` (RoboTwin never seeds Python's)."""
    random.seed(seed)
    np.random.seed(seed)
    return official.instruction(task, info, "seen")


def replay(
    sim: Simulator,
    source: Source,
    spec: EpisodeSpec,
    strategy: str,
    tail_holds: int,
    grip_threshold: float,
    directory: Path,
) -> Recording | None:
    """Open `spec`, replay `source` with `strategy`, writing the observations' blobs under
    `directory`: the recording if the success latched (within `tail_holds` holds of the last
    frame after the trajectory), else None; `Unsolvable` if the expert pre-run failed."""
    started = time.monotonic()
    sent = replay_frames(strategy, source.commands, sim.profile, grip_threshold)
    blobs = LocalBlobStore(directory / BLOBS_DIR)
    cameras = list(sim.profile.cameras)
    with quiet():
        opened = sim.open(spec)
    if not opened.solvable:
        raise Unsolvable(f"{spec.task} seed {spec.seed}")
    states: list[RobotState] = []
    frames: list[FrameEntry] = []
    calib: dict[str, list[tuple[Any, Any]]] = {name: [] for name in cameras}
    truths: list[referee.Truth] = []
    sources: list[int] = []
    physics: list[int] = []

    def capture(source_frame: int, steps: int) -> None:
        views, robot, instruction = sim.observe(cameras)
        n = len(frames)
        frame_id = FrameId(f"demo-{spec.task}-{spec.seed}-{n}")
        seen = observation(views, robot, instruction, blobs, frame_id=frame_id)
        blob = blobs.put_bytes(seen.model_dump_json().encode(), "application/json")
        frames.append(
            FrameEntry(frame_id=frame_id, alias=frame_alias(n), step=robot.step, observation=blob)
        )
        states.append(robot)
        sources.append(source_frame)
        physics.append(steps)
        for name, view in views.items():
            calib[name].append((view.K, view.cam2world))
        truths.append(sim.truth())

    latched = False
    with sim.stepping() as stepped:
        capture(0, stepped())
        for i, f in enumerate([*sent, *[sent[-1]] * tail_holds], start=1):
            step = ActionStep(qpos=source.commands[f].tolist())
            with quiet():
                result = sim.execute(ActionBatch(batch_id=BatchId(f"record-{i}"), steps=[step]))
            if result.stopped == "error":
                raise RuntimeError(f"{spec.task} seed {spec.seed}: {result.error}")
            if not result.states:  # the step limit
                break
            capture(f, stepped())
            if latched := sim.judge()[0]:
                break
    sim.close()
    if not latched:
        return None
    stride = int(np.median(np.diff([0, *sent])))
    demo = Demo(
        DemoMeta(
            task=spec.task,
            seed=spec.seed,
            instructions=[opened.instruction or source.instruction],
            frames=len(frames),
            frame_steps=source.frame_steps * stride,
            ee_to_tcp={arm: s.ee_to_tcp for arm, s in sim.profile.arms.items()},
            cameras={
                name: CameraTrack(width=s.width, height=s.height)
                for name, s in sim.profile.cameras.items()
            },
            success_frame=len(frames) - 1,
            replay=strategy,
            observations=True,
            provenance=source.provenance,
        ),
        tracks(states, sim.profile),
        {
            name: CameraCalib(np.array([k for k, _ in c]), np.array([w for _, w in c]))
            for name, c in calib.items()
        },
        np.array(sources),
        np.array(physics),
    )
    return Recording(demo, frames, scene_truth(truths), time.monotonic() - started)


def tracks(states: list[RobotState], profile: EmbodimentProfile) -> dict[Arm, ArmTrack]:
    """Per arm, the recorded robot states as an arm track (commands and measurements)."""
    return {
        arm: arm_track(
            [s.arms[arm].ee.as_list() for s in states],
            [s.arms[arm].gripper_cmd for s in states],
            spec.ee_to_tcp,
            joints=[s.arms[arm].joints_cmd for s in states],
            joints_measured=[s.arms[arm].joints for s in states],
            opening=[s.arms[arm].gripper for s in states],
        )
        for arm, spec in profile.arms.items()
    }


def scene_truth(truths: list[referee.Truth]) -> SceneTruth:
    """`Simulator.truth()` of every frame as arrays."""
    first = truths[0]
    names = list(first.poses)
    return SceneTruth(
        objects=names,
        poses=np.array([[t.poses[name].as_list() for name in names] for t in truths]),
        touched={
            arm: np.array([[name in t.touched[arm] for name in names] for t in truths])
            for arm in first.touched
        },
        joints={name: np.array([t.joints[name] for t in truths]) for name in first.joints},
        categories=first.categories,
    )


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """RoboTwin prints a progress line per action and chatter while setting up."""
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        yield


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record one candidate demonstration of a task.")
    ap.add_argument("--manifest", type=Path, required=True, help="select_demos.py's choice")
    ap.add_argument("--out", type=Path, required=True, help="the task's directory")
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True, help="the expert data's episode")
    ap.add_argument("--seed", type=int, required=True, help="the episode's seed")
    ap.add_argument("--config", default="demos", help="a config name or YAML file")
    ap.add_argument("--set", action="append", default=[], help="a config override key=value")
    args = ap.parse_args(argv)
    config = load_config_ref(args.config, DemosConfig, overrides=args.set)
    manifest = DemoManifest.model_validate_json(args.manifest.read_text("utf-8"))
    root = robotwin_root()
    if root is None:
        raise SystemExit("set ROBOTWIN_ROOT in the environment or .env")
    task = TaskName(args.task)
    sim = Simulator(root.resolve(), TASK_CONFIGS["clean"], ROBOTWIN.profile(), task)
    sim.start()
    entry, source = manifest.tasks[task], Path(manifest.source_root)
    record_task(sim, entry, args.episode, args.seed, source, args.out, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
