"""The expert's traced API calls replayed through our executor, as demonstration validation runs
it (`roboscript_robotwin.demos.validate`): `replay` (one task and seed: one row), its configuration
`ReplayConfig`, and the services it runs on (`motion_services`, `env_server`, `sim_interpreters`).
"""

import contextlib
import math
import time
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
)

from roboscript.contracts.env import (
    EnvAgent,
    EnvControl,
    EpisodeOpened,
    EpisodeSpec,
    ExpertCall,
)
from roboscript.contracts.executor import executor_overrides, with_overrides
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import Planner, PlanRequest, PlanResult
from roboscript.contracts.receipt import HARD_FAILURES, ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos import gt_scene
from roboscript.demos.gt_scene import obstacles
from roboscript.embodiments import EmbodimentProfile
from roboscript.manipulation import Executor, ExecutorConfig, Grip, Move, Segment
from roboscript.rpc import client
from roboscript.stack import Interpreters, ServiceSpec, StackConfig, up
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.bindings import EXECUTOR


class Placement(BaseModel):
    """How to write one placed object's PlaceSpec instead of the default rule: its target (an
    object, or `robot` for the robot base frame), the axes of its offset, a region radius, the
    placement point, and whether the other arm places at the mirror image (`PlaceSpec.mirror`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str | None = None
    frame: Literal["target", "world_axes"] | None = None
    region_m: float = Field(default=0.0, ge=0.0)
    placement_point: str | None = None
    mirror: bool = False


_READY_S: Final = 300.0
"""A service not ready by then has hung."""


HOLD_SIM_STEPS: Final = 50
"""Simulation steps of one hold (`take_action` idles an unmoved arm for 50)."""


PRESS_M: Final = 0.01
"""A displacement or pose move lowering the TCP this much or more is meant to touch."""


_CONTACT_LAST: Final = ("grasp_object", "place_object")


_STOPPING: Final = frozenset(
    {
        ReceiptReason.plan_failed,
        ReceiptReason.unreachable,
        ReceiptReason.invalid_args,
        ReceiptReason.budget_exhausted,
    }
)
"""Receipts after which nothing more can run: the action was refused, or the steps are used."""


_FAILED: Final = frozenset({*HARD_FAILURES, *_STOPPING})


_RECEIPT_FIELDS: Final = {"reason", "arm", "steps_used", "deviation_m", "note"}


class Seeds(BaseModel):
    """Seeds from `start` on until `per_task` are solvable or `max_tried` were tried."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: NonNegativeInt = 10000
    per_task: PositiveInt = 5
    max_tried: PositiveInt = 15
    only: tuple[NonNegativeInt, ...] = ()
    """These seeds in this order instead of `start` on (in the `api` and `perc` variants the first
    solvable one fits the specs), still until `per_task`."""

    pinned: Path | None = None
    """A seed file (relative to the repository root): per task the
    fitting seed and the transfer seeds, with the RoboTwin commit they were found on. Missing, it
    is written from this run's solvable seeds; present, every task runs exactly those seeds (none
    replaced when a pre-run fails this time), so that runs compare on the same seeds: whether the
    expert's pre-run solves a seed is not repeatable (the GPU planner). The fitting seed is the
    task's demonstration seed where the file names one (`PinnedSeeds.fitting`: the
    single-demonstration setting), so that paired runs fit their specs on the same scene."""

    def order(self) -> list[int]:
        """The seeds to try, in order."""
        return list(self.only) or list(range(self.start, self.start + self.max_tried))


class ReplayConfig(BaseModel):
    """The replay's configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    motion_config: Path = Path("examples/simBenchmarks/RoboTwin/configs/motion.example.yaml")
    """The planner service's config, relative to the repository root."""
    gpus: list[int] = Field(default=[0], min_length=1)
    envs_per_gpu: PositiveInt = 5
    """Env servers per GPU, one task each at a time."""
    mode: Literal["clean", "randomized"] = "clean"
    tasks: list[TaskName] = []
    """Empty: all of RoboTwin's tasks."""
    seeds: Seeds = Seeds()
    prerun_attempts: PositiveInt = 3
    fit_prerun_attempts: PositiveInt = 10
    """The expert's pre-run attempts on a seed the specs are to be fitted on (the pinned
    demonstration seed, which the expert solved when it was recorded), so that a run fits on
    another seed only when the expert fails it this often."""
    tail_holds: NonNegativeInt = 4
    """Holds after the last call, for objects still settling into success."""
    executor: ExecutorConfig = EXECUTOR
    """RoboTwin's executor tuning (`roboscript_robotwin.bindings.EXECUTOR`) with the config's
    `executor:` section laid over it field by field."""
    out_dir: Path = Path("runs/replay")
    """Relative to the repository root; each run writes a timestamped directory in it."""
    sim_python: Path | None = None
    """The simulator env's interpreter for the services; None is the one running the replay."""
    variant: Literal["exec", "api", "perc"] = "exec"
    """`exec` replays the traced targets; `api` rebuilds grasps and placements from specs fitted
    on each task's first solvable seed; `perc` has a scripted brain call the Actor's tools with
    those specs on perception's scene registry."""
    obstacles: bool = False
    """`exec`: free moves keep clear of the objects no arm holds, their boxes measured from ground
    truth (`gt_scene`) at each call, as the Actor's toolset sets perceived ones."""
    instruction: Literal["seen", "unseen"] = "unseen"
    """`perc`: the instruction the objects are identified by."""
    priors: Path | None = None
    """`perc`: the category prior cache (one JSON file per kind,
    relative to the repository root; a kind not cached is asked once and kept). Set: each role's
    kind's front goes with the axis question as context (`identify_objects`' `fronts`); None: no
    priors."""
    end_targets: str | None = None
    """`perc`: a task's demonstration end as targets (`perception.endstate`; an S1 attempt's
    `end_targets.json`, `{task}` in the path filled in). Set, a seed whose replay leaves an object
    off the targets gets the end loop: `check_done`, then a grasp and a `nudge` per object off,
    up to `END_ROUNDS` times. None, or no file for the task: no end loop."""
    vision_config: Path = Path("configs/vision.yaml")
    """`perc`: the vision service's config (`blob_dir` unset: a stack's own)."""
    vision_python: Path | None = None
    """`perc`: the vision env's interpreter."""
    perc_variants: list[str] = ["perc"]
    """`perc`: the variants to run, each on every task and seed."""
    demo_packages: str | None = None
    """`perc`'s `demo` variants: the demonstration package of each task, a directory pattern
    with `{task}` relative to the repository root (`data/demos/recorded/{task}/package`)."""
    perc_items: list[str] = []
    """`perc`: only these `<task>/<variant>` items (empty: every task with every variant), to
    run again those that infrastructure failed."""
    mirror: bool = True
    """`api`: a grasp spec serves the other arm mirrored (false: as written)."""
    parts: bool = False
    """`api`: specs name the object's fixed part candidates too (a handle, a rim), not only the
    moving links (whole object or part is a per-task category, S2's to learn; reported apart)."""
    placements: dict[TaskName, dict[str, Placement]] = {}
    """`api`: per task and placed object, how to write its PlaceSpec instead of the default
    rule (the object holding the expert's target point, else the robot base frame)."""
    fitted_from: Path | None = None
    """An earlier `api` or `perc` run directory whose fitting rows' specs every task replays (per
    task, and in `perc` per variant), none fitted here: paired runs then replay the same specs
    (`perc`'s are fitted on perception, which differs run to run).
    The earlier run's fitting seed still runs, as a fitting row (no transfer); a task it fitted
    nothing for fits here as usual."""
    compare_to: Path | None = None
    """`api`: an `exec` run directory whose results the table sets beside this run's; `perc`: an
    earlier `perc` run compared on the transfer seeds both replayed."""
    repeats: PositiveInt = 1
    """`perc`: runs of each seed (each a row with its `repeat`), for effects smaller than the
    spread between runs; the tables then give per task the mean and spread over the repeats."""
    gpu_claims: Path | None = None
    """The claims directory other tenants of the GPUs write
    (None: none, and env servers are placed by count alone). With it, env servers are placed as
    the pipeline places its jobs (`roboscript.orchestrate.slots.Slots`): on a GPU whose memory in
    use, plus what our env servers and services hold that nvidia-smi does not show yet, plus what
    others claimed, plus `env_gb` stays within `gpu_cap_gb`, claiming there and publishing our holds
    every 20 s; a task waits for such room (`perc`: also a seed whose simulator ran out of GPU
    memory, before it runs again, without spending one of its infra reruns), `oom_wait_s` in all
    per task."""
    env_gb: PositiveInt = 9
    """GiB an env server holds."""
    gpu_cap_gb: PositiveInt = 46
    """GiB of a GPU in use (by anyone) plus held and claimed that a placement may reach."""
    motion_gb: NonNegativeFloat = 2.5
    """GiB a GPU's motion service holds."""
    vision_gb: NonNegativeFloat = 8.5
    """`perc`: GiB a GPU's vision service holds."""
    service_idle_s: PositiveFloat = 300.0
    """A GPU's services (motion; `perc`: and vision) are attached to (or started) when the
    first env server is placed there and let go once none was there for this long."""
    shared_stacks: Path | None = Path("/dev/shm/gpu_stacks")
    """The registry directory of the shared per-GPU stacks (`roboscript.stack.shared`): every
    run and pipeline of the same service code on the machine uses one stack per GPU. None: a
    stack private to this run, in its directory."""
    shared_linger_s: PositiveFloat = 60.0
    """How long a shared stack outlives its last tenant."""
    oom_wait_s: NonNegativeFloat = 3600.0
    """Seconds a task (`perc`: a task and variant) may wait for GPU room in all; after that an
    env server that finds none is an infra row, and running out of memory spends reruns."""
    resume: Path | None = None
    """A run directory to continue (relative to the repository root), e.g. with fewer env servers:
    its rows stay, each task goes on after the seeds it tried (an episode stopped in flight runs
    again), and the table and summary cover every row. The seeds and variant must be the same."""

    @field_validator("executor", mode="before")
    @classmethod
    def _over_binding(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return with_overrides(EXECUTOR, executor_overrides(value))
        return value

    @field_validator("tasks")
    @classmethod
    def _known(cls, tasks: list[TaskName]) -> list[TaskName]:
        if unknown := sorted(set(tasks) - set(ROBOTWIN.tasks)):
            raise ValueError(f"RoboTwin has no tasks {unknown}")
        return tasks


def segments(
    call: ExpertCall, state: RobotState, profile: EmbodimentProfile
) -> dict[Arm, list[Segment]]:
    """The executor's segments for one traced `move` or `grippers` call, from the robot's state
    before it (which says where a move starts, to tell pressing moves)."""
    out: dict[Arm, list[Segment]] = {}
    for arm, actions in call.actions.items():
        api = call.api.get(arm, "raw")
        moves = [i for i, action in enumerate(actions) if action.target is not None]
        z = state.arms[arm].tcp.p[2]
        held = call.openings.get(arm, 0.0) >= profile.arms[arm].gripper.holding_threshold
        out[arm] = []
        for i, action in enumerate(actions):
            if action.target is None:
                assert action.gripper is not None
                out[arm].append(Grip(action.gripper, grasp=held and action.gripper < 0.5))
                continue
            tcp_z = profile.tcp_pose(arm, action.target).p[2]
            if api in _CONTACT_LAST:
                contact = i == moves[-1]
            else:
                contact = api != "return_home" and tcp_z <= z - PRESS_M
            straight = action.constraint == "orientation"
            out[arm].append(Move(action.target, straight=straight, contact=contact))
            z = tcp_z
    return out


async def replay(
    control: EnvControl,
    agent_of: Callable[[EpisodeOpened], EnvAgent],
    planner: Planner,
    profile: EmbodimentProfile,
    task: TaskName,
    seed: int,
    config: ReplayConfig,
) -> dict[str, Any]:
    """Open `task` at `seed` with the expert traced and replay its calls: one table row."""
    started = time.monotonic()
    spec = EpisodeSpec(
        task=task, seed=seed, prerun_attempts=config.prerun_attempts, trace_expert=True
    )
    opened = await control.open(spec)
    row: dict[str, Any] = {
        "task": task,
        "seed": seed,
        "solvable": opened.solvable,
        "prerun_attempts": opened.prerun_attempts,
    }
    if not opened.solvable:
        return row
    trace = await control.expert_trace(opened.episode_id)
    first = await control.gt_snapshot(opened.episode_id, geometry=config.obstacles)
    scene = gt_scene.measured(first) if config.obstacles else None
    agent = agent_of(opened)
    timed = _TimedPlanner(planner)
    executor = Executor(
        agent, opened.episode_id, timed, profile, config.executor, table_top_z=first.table_top_z
    )
    receipts: list[ActionReceipt] = []
    what: list[dict[str, Any]] = []  # which call each receipt answers
    for index, call in enumerate(trace.calls):
        if call.refused:
            continue
        if call.kind == "hold":
            receipt = await executor.hold(math.ceil(call.sim_steps / HOLD_SIM_STEPS))
        else:
            robot = await agent.robot(opened.episode_id)
            per_arm = segments(call, robot, profile)
            if scene is not None:
                now = await control.gt_snapshot(opened.episode_id)
                links = {n: {k: v.pose for k, v in ls.items()} for n, ls in now.links.items()}
                executor.obstacles = obstacles(
                    scene.at(now.objects, links), robot, config.executor.obstacle_margin_m
                )
            receipt = await executor.run(per_arm.get("left", ()), per_arm.get("right", ()))
        receipts.append(receipt)
        what.append(
            {
                "call": index,
                "api": call.api or call.kind,
                "expected": expected(receipt, call, profile),
            }
        )
        if executor.succeeded or receipt.reason in _STOPPING:
            break
    if not executor.succeeded and receipts and receipts[-1].reason not in _STOPPING:
        receipts.append(await executor.hold(config.tail_holds))
        what.append({"call": None, "api": "tail", "expected": False})
    counted = [r for r, called in zip(receipts, what, strict=True) if not called["expected"]]
    fail_fast = next((str(r.reason) for r in counted if r.reason in HARD_FAILURES), None)
    verdict = await control.judge(opened.episode_id)
    await control.close(opened.episode_id)
    row.update(
        trace=trace.model_dump(mode="json"),
        success=verdict.success,
        failure=None if verdict.success else failure(counted),
        fail_fast=fail_fast,
        take_actions=verdict.steps_used,
        step_limit=verdict.step_limit,
        calls=len(receipts),
        expert_calls=sum(not call.refused for call in trace.calls),
        seconds=round(time.monotonic() - started, 1),
        plans=timed.plans,
        plan_seconds=round(timed.seconds, 2),
        receipts=[
            {**called, **r.model_dump(mode="json", include=_RECEIPT_FIELDS)}
            for called, r in zip(what, receipts, strict=True)
        ],
    )
    return row


class _TimedPlanner:
    """A planner that counts its `plan` calls and the seconds they take (the replay's plan time)."""

    def __init__(self, planner: Planner) -> None:
        self._planner, self.plans, self.seconds = planner, 0, 0.0

    async def plan(self, request: PlanRequest) -> PlanResult:
        started = time.monotonic()
        try:
            return await self._planner.plan(request)
        finally:
            self.plans += 1
            self.seconds += time.monotonic() - started

    def __getattr__(self, name: str) -> Any:
        return getattr(self._planner, name)


def expected(receipt: ActionReceipt, call: ExpertCall, profile: EmbodimentProfile) -> bool:
    """A lost hold the expert's own run had too: its fingers were empty after this call
    (a gripper can slide off a drawer handle as it pulls)."""
    arm = receipt.arm
    if receipt.reason != ReceiptReason.lost_hold or arm is None or arm not in call.openings:
        return False
    return call.openings[arm] < profile.arms[arm].gripper.holding_threshold


def failure(receipts: Sequence[ActionReceipt]) -> str:
    """Why a replay failed: the first hard failure or refusal, else the first move that stopped
    short or deviated, else the judge (every action went as planned, the task was not done)."""
    for reasons in (_FAILED, {ReceiptReason.stopped_short, ReceiptReason.deviated}):
        for receipt in receipts:
            if receipt.reason in reasons:
                return str(receipt.reason)
    return "judge"


def sim_interpreters(root: Path, python: Path) -> tuple[Interpreters, list[Path]]:
    """The interpreters of a stack whose services all run in the simulator env (`python`), and
    their PYTHONPATH: the core's and vision package's sources and the RoboTwin example."""
    pythonpath = [root / "packages/core/src", root / "packages/vision/src", *ROBOTWIN.pythonpath]
    return Interpreters(agent=python, sim=python), pythonpath


@contextlib.asynccontextmanager
async def motion_services(
    gpus: Sequence[int],
    motion_config: Path,
    interpreters: Interpreters,
    pythonpath: list[Path],
    run_dir: Path,
    http: httpx.AsyncClient,
) -> AsyncIterator[dict[int, Planner]]:
    """A motion service per GPU (logging under `run_dir`), as planner clients by GPU."""
    services = [
        ServiceSpec(
            name=f"motion{gpu}",
            module="roboscript.motion",
            interpreter="sim",
            args=["--config", str(motion_config)],
            gpu=gpu,
            ready_timeout_s=_READY_S,
        )
        for gpu in gpus
    ]
    stack = StackConfig(
        interpreters=interpreters, sim_pythonpath=pythonpath, run_dir=run_dir, services=services
    )
    async with up(stack) as running:
        yield {
            gpu: client(
                Planner,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
                running.endpoints[f"motion{gpu}"],
                token=None,
                http=http,
                timeout=300,
            )
            for gpu in gpus
        }


@contextlib.asynccontextmanager
async def env_server(
    task: TaskName,
    mode: str,
    gpu: int,
    directory: Path,
    interpreters: Interpreters,
    pythonpath: list[Path],
    token: SecretStr,
    http: httpx.AsyncClient,
    blob_dir: Path | None = None,
    on_start: Callable[[str, int], None] | None = None,
) -> AsyncIterator[tuple[EnvControl, Callable[[EpisodeOpened], EnvAgent]]]:
    """An env server of `task` on `gpu` (its config and log in `directory`, its blobs in
    `blob_dir`, else there; `on_start` hears of its process, `up`'s): its control plane, and the
    agent plane of an episode it opened."""
    service = ServiceSpec(
        name="env",
        module=ROBOTWIN.env_server,
        interpreter="sim",
        args=["--config", str(_env_config(directory, task, mode))],
        gpu=gpu,
        ready_timeout_s=_READY_S,
    )
    stack = StackConfig(
        interpreters=interpreters,
        sim_pythonpath=pythonpath,
        run_dir=directory,
        blob_dir=blob_dir,
        services=[service],
    )
    async with up(stack, on_start=on_start) as running:
        control = client(
            EnvControl,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            running.urls["env"]["env.control"],
            token=token,
            http=http,
            timeout=900,
        )

        def agent_of(opened: EpisodeOpened) -> EnvAgent:
            return client(
                EnvAgent,  # type: ignore[type-abstract]
                opened.agent_url,
                token=opened.agent_token,
                http=http,
                timeout=300,
            )

        yield control, agent_of


def _env_config(directory: Path, task: TaskName, mode: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "env.yaml"
    path.write_text(yaml.safe_dump({"task": task, "mode": mode}), encoding="utf-8")
    return path
