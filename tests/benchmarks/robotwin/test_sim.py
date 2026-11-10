"""The RoboTwin env server on the real simulator (marker `sim`: the simulator env and a GPU).

Host processes serve the session, one per task (beat_block_hammer; open_laptop for the pre-run
guard), started the way a supervisor starts them: from the repository, with the token and
ROBOTWIN_ROOT in the environment and no RS_ENV_FILE. The beat_block_hammer host passes the env
conformance suite; then the RoboTwin particulars: calibration against ground truth, the robot at
rest, the settled opening of an empty grasp, the official files our data derive from. Timings go
to `pytest.log`.

    ROBOTWIN_ROOT=... RS_EXPERT_DEMO_ROOT=... python -m pytest -m sim tests/benchmarks/robotwin
"""

import contextlib
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
import yaml
from pydantic import SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import (
    EnvAgent,
    EnvControl,
    EpisodeOpened,
    EpisodeSpec,
    ObserveRequest,
)
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.ids import BatchId, TaskName, new_id
from roboscript.rpc import Ready, client, read_ready
from roboscript.spatial.camera import backproject_world, project
from roboscript.testing.env import EnvConformance, EnvTarget
from roboscript_robotwin.benchmark import HERE, ROBOTWIN, TASK_CONFIGS
from roboscript_robotwin.config import expert_demo_root, robotwin_root

pytestmark = pytest.mark.sim
log = logging.getLogger(__name__)
TASK = TaskName("beat_block_hammer")
LAPTOP = TaskName("open_laptop")
TABLE_Z = 0.74  # clean task configs do not move the table
REFEREE = yaml.safe_load((HERE / "sim" / "referee.yaml").read_text(encoding="utf-8"))["tasks"]


@dataclass(frozen=True)
class Host:
    ready: Ready
    token: SecretStr
    blobs: LocalBlobStore
    root: Path
    solvable: int


def connect(interface: type[Any], url: str, token: SecretStr | None) -> Any:
    return client(interface, url, token=token, timeout=120)


def _settings() -> Path:
    """The expert demonstrations' root; skips without it or the RoboTwin checkout."""
    root = expert_demo_root()
    if robotwin_root() is None or root is None:
        pytest.skip("set ROBOTWIN_ROOT and RS_EXPERT_DEMO_ROOT (environment or .env)")
    return root


def _demo_seeds(task: TaskName) -> list[int]:
    root = _settings()
    return [
        int(seed)
        for seed in (root / task / "aloha-agilex_clean_50" / "seed.txt")
        .read_text(encoding="utf-8")
        .split()
    ]


@contextlib.contextmanager
def _run_host(directory: Path, task: TaskName) -> Iterator[tuple[Ready, SecretStr]]:
    """`roboscript_robotwin.sim.host` serving `task`, stopped at the end."""
    token = secrets.token_urlsafe(24)
    config = directory / "robotwin.yaml"
    config.write_text(
        yaml.safe_dump({"task": task, "blob_dir": str(directory / "blobs")}), encoding="utf-8"
    )
    _settings()  # skips without ROBOTWIN_ROOT or RS_EXPERT_DEMO_ROOT
    env = {k: v for k, v in os.environ.items() if k != "RS_ENV_FILE"}
    env |= {"RS_ENV_TOKEN": token, "ROBOTWIN_ROOT": str(robotwin_root())}
    ready_file, output = directory / "ready.json", directory / "host.log"
    command = [sys.executable, "-m", "roboscript_robotwin.sim.host", "--config", str(config)]
    started = time.monotonic()
    with output.open("wb") as out:
        process = subprocess.Popen(
            [*command, "--ready-file", str(ready_file)],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    try:
        while (ready := read_ready(ready_file)) is None:
            if process.poll() is not None or time.monotonic() - started > 300:
                tail = output.read_text(errors="replace", encoding="utf-8")[-3000:]
                pytest.fail(f"the {task} host did not come up (exit {process.poll()}):\n{tail}")
            time.sleep(0.5)
        log.info("%s host ready in %.1f s", task, time.monotonic() - started)
        yield ready, SecretStr(token)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.fixture(scope="session")
def host(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Host]:
    """The beat_block_hammer host and a seed of its expert demo that the expert solves."""
    _settings()
    root = robotwin_root()
    assert root is not None
    directory = tmp_path_factory.mktemp("robotwin")
    with _run_host(directory, TASK) as (ready, token):
        solvable = anyio.run(_solvable_seed, ready.url, token, _demo_seeds(TASK))
        yield Host(ready, token, LocalBlobStore(directory / "blobs"), root, solvable)


@pytest.fixture(scope="session")
def laptop(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Ready, SecretStr]]:
    with _run_host(tmp_path_factory.mktemp("laptop"), LAPTOP) as served:
        yield served


async def _solvable_seed(url: str, token: SecretStr, seeds: list[int]) -> int:
    """The first seed whose expert pre-run passes twice in a row (its GPU planning is not
    deterministic, so one pass says little)."""
    control = connect(EnvControl, url, token)
    for seed in seeds[:10]:
        for attempt in range(2):
            started = time.monotonic()
            opened = await control.open(EpisodeSpec(task=TASK, seed=seed))
            seconds = time.monotonic() - started
            log.info("open seed %d: solvable=%s in %.2f s", seed, opened.solvable, seconds)
            if not opened.solvable:
                break
            if attempt == 1:
                await control.close(opened.episode_id)
                return seed
    raise AssertionError(f"the expert solves none of {seeds[:10]} twice")


async def _open(
    control: EnvControl, seed: int, instruction: str | None = None, task: TaskName = TASK
) -> EpisodeOpened:
    """Open a seed the expert usually solves, with up to three pre-runs."""
    spec = EpisodeSpec(task=task, seed=seed, instruction=instruction, prerun_attempts=3)
    return await control.open(spec)


class TestRoboTwinConformance(EnvConformance):
    exhaust_step_limit = True
    traces_expert = True

    @pytest.fixture
    def env_target(self, host: Host) -> EnvTarget:
        return EnvTarget(
            url=host.ready.url,
            token=host.token,
            connect=connect,
            profile=ROBOTWIN.profile(),
            task=TASK,
            seed=host.solvable,
        )


async def test_an_episode_of_beat_block_hammer(host: Host) -> None:
    profile = ROBOTWIN.profile()
    control = connect(EnvControl, host.ready.url, host.token)
    timings: dict[str, float] = {}
    with _timed(timings, "open"):
        opened = await _open(control, host.solvable)
    official = yaml.safe_load(
        (host.root / "task_config" / "_eval_step_limit.yml").read_text(encoding="utf-8")
    )
    assert opened.instruction and opened.step_limit == official[TASK] == 400
    agent = connect(EnvAgent, opened.agent_url, opened.agent_token)
    with _timed(timings, "observe"):
        obs = await agent.observe(opened.episode_id, ObserveRequest())
    with _timed(timings, "observe again"):
        await agent.observe(opened.episode_id, ObserveRequest())
    with _timed(timings, "robot"):
        await agent.robot(opened.episode_id)
    front = agent.observe(opened.episode_id, ObserveRequest(cameras=["front_camera"]))
    with pytest.raises(ProtocolError, match="no cameras"):  # not shown to the agent
        await front
    assert set(obs.cameras) == {"head_camera", "left_camera", "right_camera"}
    head = obs.cameras["head_camera"]
    assert (head.camera.width, head.camera.height) == (640, 480)
    depth = host.blobs.get_array(head.depth)
    assert depth.dtype == np.float32 and np.mean(depth > 0) > 0.99
    assert depth[depth > 0].min() > 0.3 and depth.max() < 3.0  # meters, not millimeters

    # Calibration: the block's top center (ground truth) projects onto a pixel whose depth
    # back-projects to it; wrist cameras see the table top at its height.
    block = np.array((await control.gt_snapshot(opened.episode_id)).objects["block"].p)
    top = block + np.array([0.0, 0.0, 0.025])
    K, cam2world = np.array(head.camera.K), np.array(head.camera.cam2world)
    uv, _, visible = project(top, K, cam2world)
    u, v = np.round(uv).astype(int)
    assert visible and 0 <= u < 640 and 0 <= v < 480
    mask = np.zeros_like(depth, dtype=bool)
    mask[v, u] = True
    seen, _ = backproject_world(depth, K, cam2world, mask)
    assert np.linalg.norm(seen[0] - top) < 0.01, (seen[0], top)
    for name in ("left_camera", "right_camera"):
        wrist = obs.cameras[name]
        points, _ = backproject_world(
            host.blobs.get_array(wrist.depth),
            np.array(wrist.camera.K),
            np.array(wrist.camera.cam2world),
        )
        assert np.mean(np.abs(points[:, 2] - TABLE_Z) < 0.005) > 0.3, name

    for arm in obs.robot.arms.values():  # at rest, open
        assert arm.joints is not None and np.allclose(arm.joints, arm.joints_cmd, atol=1e-3)
        assert arm.gripper_cmd == 1.0 and arm.gripper is not None and arm.gripper > 0.95

    hold = obs.robot.qpos_cmd
    with _timed(timings, "execute (1 hold)"):
        held = await agent.execute(opened.episode_id, _batch([hold]))
    assert (held.stopped, held.executed, held.states[0].robot.step) == ("completed", 1, 1)
    verdict = await control.judge(opened.episode_id)
    assert (verdict.success, verdict.steps_used) == (False, 1)
    names = {truth.name for truth in (await control.diagnose(opened.episode_id)).objects}
    assert names == set(REFEREE[TASK]["objects"]) == {"hammer", "block"}

    closed = list(hold)
    closed[profile.arms["left"].gripper_index] = profile.arms["left"].gripper.closed
    with _timed(timings, "execute (close until stable)"):
        grasp = await agent.execute(opened.episode_id, _batch([closed], stable=True, repeats=6))
    settled = grasp.states[-1].robot.arms["left"].gripper
    assert settled == pytest.approx(profile.arms["left"].gripper.empty, abs=1e-3)
    assert grasp.executed < 7, grasp.executed  # settled before the cap of 1 + 6
    log.info(
        "beat_block_hammer seed %d: %s; closing on nothing settled at %.5f after %d actions",
        host.solvable,
        ", ".join(f"{name} {seconds:.3f} s" for name, seconds in timings.items()),
        settled,
        grasp.executed,
    )
    with _timed(timings, "reopen the same seed"):
        await _open(control, host.solvable, opened.instruction)
    log.info("reopen the same seed %.2f s", timings["reopen the same seed"])


async def test_the_expert_trace_of_beat_block_hammer(host: Host) -> None:
    """The traced pre-run: a grasp of the hammer (pre-grasp, a locked approach, a close), then
    moves, in RoboTwin's API; the table top at 0.74 m in the clean config."""
    control = connect(EnvControl, host.ready.url, host.token)
    spec = EpisodeSpec(task=TASK, seed=host.solvable, prerun_attempts=3, trace_expert=True)
    opened = await control.open(spec)
    trace = await control.expert_trace(opened.episode_id)
    assert (await control.gt_snapshot(opened.episode_id)).table_top_z == pytest.approx(TABLE_Z)
    moves = [call for call in trace.calls if call.kind == "move" and not call.refused]
    [arm] = moves[0].actions
    assert moves[0].api == {arm: "grasp_object"} and moves[0].sim_steps > 0
    pre, approach, close = moves[0].actions[arm]
    assert (pre.constraint, approach.constraint, close.gripper) == ("free", "orientation", 0.0)
    assert {api for call in moves for api in call.api.values()} <= {
        "grasp_object",
        "place_object",
        "move_by",
        "move_to_pose",
        "return_home",
        "open_gripper",
        "close_gripper",
        "raw",
    }
    # How the API built them: the hammer, its arguments, the true poses, and where the
    # placement would leave the hammer; the snapshot's convex hulls, one per collision shape.
    grasp = moves[0].built[arm]
    assert grasp.obj == "hammer" and grasp.args["approach_m"] == pytest.approx(0.12)
    assert set(grasp.objects) == {"hammer", "block"}
    placing = next(c for c in moves if c.api.get(arm) == "place_object").built[arm]
    assert placing.placed is not None and placing.args["placement_frame"] == 0
    assert len(placing.args["target_pose"]) == 7
    hulls = (await control.gt_snapshot(opened.episode_id, geometry=True)).hulls
    assert set(hulls) == {"hammer", "block"} and all(len(hull) >= 4 for hull in hulls["block"])
    assert not (await control.gt_snapshot(opened.episode_id)).hulls  # only when asked for
    log.info(
        "beat_block_hammer expert trace: %s", [(c.kind, c.api, c.sim_steps) for c in trace.calls]
    )


async def test_the_pre_run_guard_on_a_task_judged_by_the_experts_arm(
    laptop: tuple[Ready, SecretStr],
) -> None:
    """open_laptop's success check reads the arm the expert pre-run chose (`arm_tag`): an episode
    may skip the pre-run only to reopen the seed whose pre-run passed last."""
    ready, token = laptop
    control = connect(EnvControl, ready.url, token)
    for seed in _demo_seeds(LAPTOP)[:5]:
        first = await _open(control, seed, task=LAPTOP)
        if first.solvable:
            break
    assert first.solvable and first.instruction and 1 <= first.prerun_attempts <= 3

    async def hold_and_judge(opened: EpisodeOpened) -> None:
        agent = connect(EnvAgent, opened.agent_url, opened.agent_token)
        rest = (await agent.robot(opened.episode_id)).qpos_cmd
        held = await agent.execute(opened.episode_id, _batch([rest]))
        assert held.stopped == "completed"  # check_success ran every physics step: arm_tag set
        verdict = await control.judge(opened.episode_id)
        assert (verdict.success, verdict.faulted, verdict.steps_used) == (False, False, 1)

    await hold_and_judge(first)
    reopen = EpisodeSpec(task=LAPTOP, seed=seed, repeat=True, instruction=first.instruction)
    again = await control.open(reopen)  # try 2 of the same seed: no pre-run needed
    assert again.solvable and again.prerun_attempts == 0 and again.instruction == first.instruction
    await hold_and_judge(again)
    other = EpisodeSpec(task=LAPTOP, seed=seed + 1, repeat=True, instruction="open it")
    with pytest.raises(ProtocolError, match="seed whose pre-run passed last"):
        await control.open(other)
    with pytest.raises(ProtocolError, match="serves task 'open_laptop', not 'beat_block_hammer'"):
        await control.open(EpisodeSpec(task=TASK, seed=seed))


def test_the_profile_matches_robotwin() -> None:
    root = _root()
    profile = ROBOTWIN.profile()
    config = yaml.safe_load(
        (root / "assets/embodiments/aloha-agilex/config.yml").read_text(encoding="utf-8")
    )
    for index, (arm, spec) in enumerate(profile.arms.items()):
        assert spec.joints == config["arm_joints_name"][index]
        assert spec.ee_link == config["move_group"][index]
        assert spec.gripper.joint == config["gripper_name"][index]["base"]
        assert list(spec.gripper.scale) == config["gripper_scale"]
        assert spec.ee_to_tcp == (0.12, 0.0, 0.0)  # robot.py _trans_endpose: tcp - ee, a constant
        assert spec.home == config["homestate"][index]  # return_home's pose, after settling
        assert spec.planner is not None
        assert spec.planner.base.as_list() == pytest.approx(config["robot_pose"][0], abs=1e-3)
        for asset in (spec.planner.curobo, spec.planner.urdf, spec.planner.srdf):
            assert asset is not None and profile.asset(asset).is_file(), (arm, asset)
        curobo = yaml.safe_load(profile.asset(spec.planner.curobo).read_text(encoding="utf-8"))
        assert curobo["robot_cfg"]["kinematics"]["ee_link"] == spec.ee_link
    cameras = yaml.safe_load((root / "task_config/_camera_config.yml").read_text(encoding="utf-8"))
    for spec in profile.cameras.values():
        assert (spec.width, spec.height) == (cameras[spec.type]["w"], cameras[spec.type]["h"])


def test_task_configs_are_the_official_ones_with_our_changes() -> None:
    root = _root()
    for mode, ours in TASK_CONFIGS.items():
        expected = yaml.safe_load(
            (root / "task_config" / f"demo_{mode}.yml").read_text(encoding="utf-8")
        )
        expected["camera"]["head_camera_type"] = "Large_D435"
        expected["data_type"]["depth"] = True
        expected.update(collect_data=False, eval_video_log=False)
        assert yaml.safe_load(ours.read_text(encoding="utf-8")) == expected, mode


def test_the_task_list_is_the_official_one() -> None:
    root = _root()
    limits = yaml.safe_load(
        (root / "task_config" / "_eval_step_limit.yml").read_text(encoding="utf-8")
    )
    assert {task: info.step_limit for task, info in ROBOTWIN.tasks.items()} == limits
    instructions = {
        path.stem for path in (root / "description" / "task_instruction").glob("*.json")
    }
    assert set(ROBOTWIN.tasks) == instructions


def test_the_referee_file_names_attributes_of_the_task_code() -> None:
    """Each object `sim/referee.yaml` lists is an attribute its task's code assigns."""
    root = _root()
    for task, entry in REFEREE.items():
        code = (root / "envs" / f"{task}.py").read_text(encoding="utf-8")
        for name in entry["objects"]:
            attribute = name.partition("[")[0]
            assigned = rf"self\.{attribute}\b(?:\s*:[^=\n]*)?\s*(?:=(?!=)|\.append\()"
            assert re.search(assigned, code), f"{task}: no self.{attribute} in envs/{task}.py"


def _root() -> Path:
    root = robotwin_root()
    if root is None:
        pytest.skip("set ROBOTWIN_ROOT (environment or .env)")
    return root


def _batch(qpos: list[list[float]], *, stable: bool = False, repeats: int = 3) -> ActionBatch:
    steps = [ActionStep(qpos=q, until_gripper_stable=stable, max_repeats=repeats) for q in qpos]
    return ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)


@contextlib.contextmanager
def _timed(timings: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - started
