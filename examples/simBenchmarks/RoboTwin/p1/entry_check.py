"""P1-2's core entry check on RoboTwin (the shared check:
`roboscript_methods.skill_builder.core_entry`): RoboTwin's glue. The env server is RoboTwin's
(`roboscript_robotwin.sim.host` in the simulator interpreter, `--python`), started here for the
task in clean mode; the scene is the task's demonstration seed (`demo_export.py`'s export in
`--recordings`), its movable objects every object its truth tracks but the articulated ones (whose
joints it records); the planner is the motion service `--planner` names (`build.py` keeps one per
GPU). The executor is bound by RoboTwin's binding data alone (`skills.bindings_only`), the pose
targets drawn from a probed workspace (`--workspace`, `scripts/probe_workspace.py`'s table). Every
object the demonstration grasps (its cut, `CUT`) is grasped, then the other graspable ones; calls
are judged by the criteria measured on the runs (`<out>/criteria.yaml`: `--measure-criteria`),
the pool by the gates given (`--summarize REPORTS --gates FILE`).

Writes `<out>/<task>.json`. Runs in the repository's env:

    ROBOTWIN_ROOT=<RoboTwin> CUDA_VISIBLE_DEVICES=0 uv run python entry_check.py \\
        --task place_empty_cup --python <sim env>/bin/python --recordings <recordings> \\
        --planner http://127.0.0.1:<port> --workspace <workspace.json> --out entry/
    uv run python entry_check.py --measure-criteria entry/run1 entry/run2 --out entry/
    uv run python entry_check.py --rejudge entry/run2 --recordings <recordings> --out entry/
    uv run python entry_check.py --summarize entry/run2 --gates <gates.yaml>
"""

import argparse
import asyncio
import contextlib
import json
import math
import os
import secrets
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, TypeVar

import httpx
import yaml
from demo_export import recorded
from pydantic import SecretStr

from roboscript.contracts.env import EnvControl, EpisodeSpec
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import TIMEOUTS, Planner
from roboscript.demos.keypoints import KeypointConfig
from roboscript.manipulation.workspace import Workspace, load_workspace
from roboscript.stack import Interpreters
from roboscript_methods.skill_builder.core_entry.check import (
    Bench,
    Criteria,
    DemoScene,
    connect,
    run_check,
)
from roboscript_methods.skill_builder.core_entry.criteria import measure
from roboscript_methods.skill_builder.core_entry.report import rejudge, summarize
from roboscript_methods.skill_builder.demo_coverage.segments import CutConfig, cut
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.demos.replay import env_server, sim_interpreters
from roboscript_robotwin.skills.bindings_only import BINDINGS_ONLY

T = TypeVar("T")
P1: Final = Path(__file__).resolve().parent
REPO: Final = P1.parents[3]
MODE: Final = "clean"
"""The scene mode every P1 episode opens in: the demonstrations' (`record_demos.py`)."""
ORIGIN_POSITION_M: Final = 0.005
ORIGIN_ROTATION: Final = 0.05
"""RoboTwin's own back-to-origin tolerance. Its `back_to_origin` (`envs/_base_task.py`, lines
1390-1395) is a planned `move` to the arm's original pose with no tolerance of its own; its planner
(`envs/robot/planner.py`, lines 75-79) builds cuRobo's `MotionGenConfig` with the defaults of
`curobo/wrap/reacher/motion_gen.py` (lines 177-178): `position_threshold` 0.005 m (the position
error's norm) and `rotation_threshold` 0.05 (the norm of the vector part of the quaternion
error, `curobolib/cpp/pose_distance_kernel.cu`'s rotation distance)."""


def origin_tolerance() -> tuple[float, float]:
    """RoboTwin's back-to-origin tolerance as the shared check takes it, metres per axis and
    degrees: a position error of norm 0.005 m keeps each axis within it, and a quaternion error
    whose vector part has norm r is a turn of 2 asin(r)."""
    return ORIGIN_POSITION_M, math.degrees(2.0 * math.asin(ORIGIN_ROTATION))


CUT: Final = CutConfig(
    keypoints=KeypointConfig(),
    home_m=origin_tolerance()[0],
    home_deg=origin_tolerance()[1],
)
"""The core's keypoints as the demonstration package reads them; an arm is home within RoboTwin's
back-to-origin tolerance."""
CRITERIA: Final = "criteria.yaml"
UNPROBED: Final = Workspace(embodiment="aloha_agilex", low=(0.0, 0.0, 0.0), cell_m=1.0,
                            shape=(0, 0, 0), reachable={"left": [], "right": []})  # fmt: skip
"""The workspace a check that draws no pose target is given (the coverage's episodes): no cell
reached. The entry check's `move_to_pose` group needs a probed one (`--workspace`)."""


def criteria(directory: Path | None) -> Criteria:
    """The criteria measured on the runs (`<directory>/criteria.yaml`, `--measure-criteria`);
    no directory: the shared check's defaults. A directory without the file is an error, not the
    defaults: a job that cannot read it (outside its sandbox) would judge by the wrong criteria."""
    if directory is None:
        return Criteria()
    path = directory / CRITERIA
    if not path.is_file():
        raise SystemExit(f"{path}: no such file (measure it: entry_check.py --measure-criteria)")
    return Criteria.load(path)


def bench(criteria_dir: Path | None = None, workspace: Path | None = None) -> Bench:
    """RoboTwin's profile, its executor binding (`bindings_only`), the probed workspace (none:
    `UNPROBED`), RoboTwin's back-to-origin tolerance and the criteria."""
    config = BINDINGS_ONLY.manifest.bindings.executor
    assert config is not None  # RoboTwin binds its executor
    reach = UNPROBED if workspace is None else load_workspace(workspace)
    return Bench(ROBOTWIN.profile(), config, reach, origin_tolerance(), criteria(criteria_dir))


def demo_scene(recordings: Path, task: str, seed: int | None = None) -> DemoScene:
    """The task's demonstration scene (its seed, its objects but the articulated ones, the ones
    its cut grasps), or the scene of another seed of the task (a transfer case's: its export in
    `<task>/transfer/`)."""
    rec = recorded(recordings / task)
    if seed is not None and seed != rec.source["seed"]:
        found = [d for d in (recordings / task / "transfer").glob(f"*-{seed}")]
        if len(found) != 1:
            raise SystemExit(f"{recordings / task}: no transfer export of seed {seed}")
        rec = recorded(found[0])
    articulated = set(rec.source.get("articulated", []))
    movable = frozenset(label for label in rec.objects if label not in articulated)
    picked = tuple(dict.fromkeys(s.obj for s in cut(rec, CUT) if s.kind == "grasp" and s.obj))
    spec = EpisodeSpec(task=TaskName(task), seed=int(rec.source["seed"]))
    source = {"seed": spec.seed, "episode": rec.source.get("episode"), "mode": MODE}
    return DemoScene(spec, movable, source, picked)


def demo_picks(recordings: Path) -> dict[str, list[str]]:
    """Per exported task, the objects its demonstration grasps (`--rejudge`)."""
    return {d.name: list(demo_scene(recordings, d.name).picked)
            for d in sorted(recordings.iterdir()) if (d / "source.json").is_file()}  # fmt: skip


def planner(url: str) -> Planner:
    return connect(Planner, url, None, timeouts=TIMEOUTS)  # type: ignore[no-any-return]


def gpu() -> int:
    """The job's GPU: the one CUDA_VISIBLE_DEVICES names."""
    found = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not found.isdigit():
        raise SystemExit("set CUDA_VISIBLE_DEVICES to one GPU")
    return int(found)


@contextlib.asynccontextmanager
async def host(python: Path, task: str, directory: Path) -> AsyncIterator[EnvControl]:
    """RoboTwin's env server for `task` (its config, logs and blobs in `directory`), started with
    the simulator interpreter `python` on the job's GPU and a control token of its own: its
    control plane."""
    token = secrets.token_urlsafe(24)
    os.environ["RS_ENV_TOKEN"] = token  # the server reads it; nothing else holds one here
    _, pythonpath = sim_interpreters(REPO, python)
    interpreters = Interpreters(agent=Path(sys.executable), sim=python)
    async with httpx.AsyncClient(trust_env=False) as http:
        server = env_server(TaskName(task), MODE, gpu(), directory, interpreters, pythonpath,
                            SecretStr(token), http)  # fmt: skip
        async with server as (control, _):
            yield control


def hosted(
    python: Path, task: str, directory: Path, work: Callable[[EnvControl], Awaitable[T]]
) -> T:
    """`work` on the control plane of an env server of `task` (`host`), run to its end."""

    async def run() -> T:
        async with host(python, task, directory) as control:
            return await work(control)

    return asyncio.run(run())


async def check(args: argparse.Namespace, control: EnvControl) -> dict[str, Any]:
    scene = demo_scene(args.recordings, args.task)
    measured = args.out if (args.out / CRITERIA).is_file() else None  # a first run: defaults
    found = bench(measured, args.workspace)
    report = await run_check(found, scene, control, planner(args.planner), args.objects)
    return report | {"planner": args.planner}


def measure_criteria(runs: list[Path], out: Path) -> None:
    """Write `<out>/criteria.yaml`: each criterion measured on these runs' reports
    (`core_entry.gaps`)."""
    found = [
        json.loads(p.read_text(encoding="utf-8")) for d in runs for p in sorted(d.glob("*.json"))
    ]
    record = {"runs": [str(d) for d in runs], "reports": len(found), "criteria": measure(found)}
    out.mkdir(parents=True, exist_ok=True)
    (out / CRITERIA).write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    for name, entry in record["criteria"].items():
        how = "measured" if entry["measured"] else f"not measured ({entry['instead']})"
        print(f"{name}: {entry['value']} {how}; gap {entry.get('gap')} ratio {entry.get('ratio')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task")
    parser.add_argument("--python", type=Path, help="the simulator env's interpreter")
    parser.add_argument("--recordings", type=Path, help="demo_export.py's --out")
    parser.add_argument("--planner", help="the motion service's URL")
    parser.add_argument("--workspace", type=Path, help="the probed workspace (probe_workspace.py)")
    parser.add_argument("--objects", type=int, default=6, help="graspable objects per task")
    parser.add_argument("--out", type=Path, default=Path("entry"), help="reports and criteria")
    parser.add_argument("--measure-criteria", type=Path, nargs="+", metavar="REPORTS")
    parser.add_argument("--rejudge", type=Path, metavar="REPORTS", help="by criteria.yaml")
    parser.add_argument("--summarize", type=Path, metavar="REPORTS", help="by --gates")
    parser.add_argument("--gates", type=Path, help="--summarize: the gates' YAML")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if args.measure_criteria:
        measure_criteria([p.resolve() for p in args.measure_criteria], args.out)
    if args.rejudge:
        if not args.recordings:
            raise SystemExit("--rejudge: give --recordings")
        found = criteria(args.out)
        picks = demo_picks(args.recordings.resolve())
        rejudge(args.rejudge, found, picks, f"judged again by {CRITERIA} {asdict(found)}")
    if args.summarize:
        if not args.gates:
            raise SystemExit("--summarize: give --gates")
        summarize(args.summarize, args.gates)
    if args.measure_criteria or args.rejudge or args.summarize:
        return
    if not (args.task and args.python and args.recordings and args.planner and args.workspace):
        raise SystemExit("give --task, --python, --recordings, --planner and --workspace")
    if not os.environ.get("ROBOTWIN_ROOT"):
        raise SystemExit("set ROBOTWIN_ROOT")
    args.recordings, args.workspace = args.recordings.resolve(), args.workspace.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report = hosted(args.python, args.task, args.out / "hosts" / args.task,
                    lambda control: check(args, control))  # fmt: skip
    path = args.out / f"{args.task}.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()
