"""Re-record each task's chosen demonstration on this machine, with our cameras, and build and
validate the demo package S1 learns from (docs/concepts/learning.md).

    python examples/simBenchmarks/RoboTwin/scripts/record_demos.py --demos DIR [--out DIR/recorded]
        [--task beat_block_hammer ...] [--gpus 0,1] [--per-gpu 4] [--config demos]
        [--set recording.every=5] [--transfer N]

Reads DIR/manifest.json (select_demos.py's choice per task, reviewed picks applied) and
writes --out (default DIR/recorded; layout: `roboscript_robotwin.demos.recorded`). Per task,
candidate after candidate until one is accepted: an automatic pick, then the episodes ranked after
it (DIR/<task>/scores.json, best first); a reviewed pick (`reviewed`) alone, never replaced:

1. record it in a simulator process of its own (`python -m roboscript_robotwin.sim.record`): a
   candidate whose expert pre-run fails once in `recording.prerun_checks` runs, or whose replay
   never latches the official success, is rejected;
2. review it in another process (keypoints, disturbance, package, video); both processes log to
   <out>/<task>/worker.log;
3. compare its keypoints with those of the replayed trajectory (FK of its commanded joints);
4. validate it (`roboscript_robotwin.demos.validate`, up to `recording.validation_attempts`
   replays): a candidate that fails every attempt is rejected. A package rebuilt since is validated
   again.

Keypoint and disturbance flags only mark a task for review. Tasks run in parallel, --per-gpu at a
time on each GPU, dealt round-robin (--gpus defaults to CUDA_VISIBLE_DEVICES); each holds one
simulator process at a time (its recording, then its validation's env server), besides one motion
service per GPU; the services' logs go to <out>/services/. Infrastructure failing a task (a process
that crashed, an env server that did not start or failed twice) leaves it for a resumed run, which
keeps what is done; the other tasks go on. At the end: `summary.json` (rejected and flagged tasks
first) and `index.html`.

`--transfer N` records, instead of the chosen demonstrations, up to N other demonstrations per task
(the P1 builder's transfer cases, never an S1 input) into `<out>/transfer/<task>/<seed>/`, each
in the chosen one's layout and through the same steps: the task's other ranked episodes, best
first, whose seed lies in the demonstration block `DEMO_SEEDS` and is not the chosen seed
(`transfer_seed`); a candidate is kept once its replay latched the official success and its
package validated, and the next one is tried otherwise. `<out>/transfer/summary.json` lists them.

Every process it starts runs in a process group of its own. SIGTERM, and SIGHUP unless ignored
(`nohup`), stop the run: each group is stopped (SIGTERM, then SIGKILL after a grace) and the
script exits with 128 plus the signal's number, leaving what is done for a resumed run.

Run from the repository root with the simulator env's Python, with ROBOTWIN_ROOT
and the env servers' control token RS_ENV_TOKEN in the environment or .env (or the file
RS_ENV_FILE names); settings from configs/demos.yaml (copy configs/demos.example.yaml) or
--config (a name under configs/ or a YAML file).
"""

import argparse
import collections
import functools
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Final

import anyio
import httpx
from pydantic import TypeAdapter

from roboscript.contracts.errors import InfraError
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import Planner
from roboscript.demos import DemoManifest, DemosConfig, RankedEpisode, TaskDemo
from roboscript.envs import control_token
from roboscript.learning.seeds import SeedRange
from roboscript.settings import load_config_ref, repo_root
from roboscript.stack import StackError, Stopped, cancel_on_signals, stop_groups
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.demos import replay
from roboscript_robotwin.demos.recorded import (
    compare_task,
    compared,
    index_page,
    read,
    reference,
    reject,
    render_page,
    review_task,
    reviewed,
    summary,
    summary_row,
    validation,
)
from roboscript_robotwin.demos.validate import Validation, validate

log = logging.getLogger("record_demos")
STOP_GRACE_S: Final = 10.0
"""How long a worker's process group may take to exit after SIGTERM before SIGKILL."""
DEMO_SEEDS: Final = SeedRange(start=0, count=68)
"""The demonstration block of the seed split (demos 0-67, S2 from 10000, internal evaluation
from 20000): transfer demonstrations come from it, so they never touch a learning or evaluation
seed."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--demos", type=Path, required=True, help="the directory of manifest.json")
    ap.add_argument("--out", type=Path, help="where to write (default: DEMOS/recorded)")
    ap.add_argument("--task", action="append", default=[], help="a task (default: all)")
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--config", default="demos", help="a config name or YAML file")
    ap.add_argument("--set", action="append", default=[], help="a config override key=value")
    ap.add_argument(
        "--transfer", type=int, default=0, metavar="N", help="record N transfer demos per task"
    )
    ap.add_argument("--review", help=argparse.SUPPRESS)  # a recorded task, in its own process
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config_ref(args.config, DemosConfig, overrides=args.set)
    demos = args.demos.resolve()
    manifest = DemoManifest.model_validate_json((demos / "manifest.json").read_text("utf-8"))
    out = (args.out or demos / "recorded").resolve()
    if args.review:
        anyio.run(review_task, out / args.review, config, ROBOTWIN.profile())
        return 0
    tasks = args.task or sorted(manifest.tasks)
    if unknown := sorted(set(tasks) - set(manifest.tasks)):
        ap.error(f"{demos / 'manifest.json'} has no choice for {unknown}")
    if args.transfer < 0:
        ap.error("--transfer: a count of demonstrations, 0 or more")
    gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]
    if not gpus:
        ap.error("name the GPUs: --gpus or CUDA_VISIBLE_DEVICES")
    settings = ["--config", args.config, *(f"--set={override}" for override in args.set)]
    runner = _Run(manifest, demos / "manifest.json", out, config, settings, args.transfer)
    try:
        anyio.run(runner.all, tasks, gpus, args.per_gpu)
    except Stopped as stopped:
        log.error("%s: every process it started is stopped; a resumed run goes on", stopped)
        return 128 + stopped.signal
    if args.transfer:
        rows = [summary_row(d, 1) for t in tasks for d in transfer_dirs(out, TaskName(t))]
        text = json.dumps(summary(rows), indent=1) + "\n"
        (out / "transfer" / "summary.json").write_text(text, "utf-8")
        found = {t: [r.seed for r in rows if r.task == t and r.status == "accepted"] for t in tasks}
        print(json.dumps({"transfer": found}, indent=1))
        return 0 if all(len(seeds) == args.transfer for seeds in found.values()) else 1
    rows = [
        summary_row(out / t, len(candidates(demos, manifest.tasks[TaskName(t)], TaskName(t))))
        for t in sorted(manifest.tasks)
        if read(out / t) is not None
    ]
    (out / "summary.json").write_text(json.dumps(summary(rows), indent=1) + "\n", "utf-8")
    (out / "index.html").write_text(index_page(rows), "utf-8")
    print(json.dumps({k: v for k, v in summary(rows).items() if k != "rows"}, indent=1))
    return 0


def candidates(demos: Path, entry: TaskDemo, task: TaskName) -> list[tuple[int, int]]:
    """The (episode, seed) pairs to try: the chosen one, then, for an automatic pick, every
    episode ranked after it (select_demos.py's `scores.json`); a reviewed pick is not replaced."""
    chosen = (entry.episode, entry.seed)
    if entry.source == "reviewed":
        return [chosen]
    text = (demos / task / "scores.json").read_bytes()
    ranking = TypeAdapter(list[RankedEpisode]).validate_json(text)
    return [chosen, *((r.episode, r.seed) for r in ranking if r.episode != entry.episode)]


def transfer_seed(seed: int, chosen: int) -> bool:
    """Whether `seed` may be a transfer demonstration's: in `DEMO_SEEDS`, not the chosen seed."""
    return DEMO_SEEDS.holds(seed) and seed != chosen


def transfer_candidates(demos: Path, entry: TaskDemo, task: TaskName) -> list[tuple[int, int]]:
    """The (episode, seed) pairs a task's transfer demonstrations are tried from: every ranked
    episode (`scores.json`, best first) but the chosen one, at a seed `transfer_seed` allows."""
    ranking = TypeAdapter(list[RankedEpisode]).validate_json(
        (demos / task / "scores.json").read_bytes()
    )
    return [
        (r.episode, r.seed)
        for r in ranking
        if r.episode != entry.episode and transfer_seed(r.seed, entry.seed)
    ]


def transfer_dirs(out: Path, task: TaskName) -> list[Path]:
    """A task's transfer recordings, by seed."""
    found = [d for d in (out / "transfer" / task).glob("*") if read(d) is not None]
    return sorted(found, key=lambda d: int(d.name))


class _Run:
    """Tasks in parallel over the GPUs' slots, each through its candidates."""

    def __init__(
        self,
        manifest: DemoManifest,
        path: Path,
        out: Path,
        config: DemosConfig,
        settings: list[str],
        transfer: int = 0,
    ) -> None:
        self.manifest, self.path, self.out, self.config = manifest, path, out, config
        self.transfer = transfer  # transfer demonstrations per task instead of the chosen one
        self.settings = settings  # --config and --set, for the processes it starts
        self.profile = ROBOTWIN.profile()
        self.token = control_token()  # the env servers read the same from .env

    async def all(self, tasks: list[str], gpus: list[int], per_gpu: int) -> None:
        """Every task; SIGTERM or SIGHUP cancel it, the services and workers stopped, and then
        raise `Stopped`."""
        root = repo_root()
        self.interpreters, self.pythonpath = replay.sim_interpreters(root, Path(sys.executable))
        queue = collections.deque(TaskName(task) for task in tasks)
        motion = root / replay.ReplayConfig().motion_config
        with cancel_on_signals(signal.SIGTERM, signal.SIGHUP):
            async with httpx.AsyncClient(trust_env=False) as http:
                self.http = http
                services = replay.motion_services(
                    gpus, motion, self.interpreters, self.pythonpath, self.out / "services", http
                )
                async with services as planners, anyio.create_task_group() as group:

                    async def work(gpu: int) -> None:
                        step = self.transfers if self.transfer else self.task
                        while queue:
                            await step(queue.popleft(), gpu, planners[gpu])

                    for slot in range(len(gpus) * per_gpu):
                        group.start_soon(work, gpus[slot % len(gpus)])

    async def task(self, task: TaskName, gpu: int, planner: Planner) -> None:
        """Candidates in order until one is accepted, from where a resumed run left off."""
        out, entry = self.out / task, self.manifest.tasks[task]
        for episode, seed in candidates(self.path.parent, entry, task):
            accepted = await self.candidate(task, out, episode, seed, gpu, planner)
            if accepted is None:
                return
            if accepted:
                log.info("%s: episode %d accepted", task, episode)
                return
        log.info("%s: every candidate was rejected", task)

    async def transfers(self, task: TaskName, gpu: int, planner: Planner) -> None:
        """`transfer` demonstrations of `task`, each at a seed of its own, from where a resumed
        run left off."""
        entry, kept = self.manifest.tasks[task], 0
        for episode, seed in transfer_candidates(self.path.parent, entry, task):
            out = self.out / "transfer" / task / str(seed)
            accepted = await self.candidate(task, out, episode, seed, gpu, planner)
            if accepted is None:
                return
            if accepted:
                kept += 1
                log.info(
                    "%s: transfer seed %d accepted (%d of %d)", task, seed, kept, self.transfer
                )
                if kept == self.transfer:
                    return
        log.info("%s: %d of %d transfer demonstrations", task, kept, self.transfer)

    async def candidate(
        self, task: TaskName, out: Path, episode: int, seed: int, gpu: int, planner: Planner
    ) -> bool | None:
        """Record, review, compare and validate the candidate `episode` into `out`: whether it
        was accepted, None when a step failed for infrastructure (a resumed run goes on)."""
        recorded = read(out)
        if recorded is not None and episode in recorded.rejected:
            return False
        if recorded is None or recorded.episode != episode or not recorded.recorded:
            if not await self.process(out, self.record(task, out, episode, seed), gpu):
                log.error("%s: recording episode %d failed, see worker.log", task, episode)
                return None
            recorded = read(out)
            assert recorded is not None and recorded.episode == episode
            if not recorded.recorded:
                log.info("%s: episode %d did not record", task, episode)
                return False
        if not reviewed(out):
            review = str(out.relative_to(self.out))
            await self.process(out, [*self.command(), "--review", review], None)
            if not reviewed(out):
                log.error("%s: reviewing episode %d failed, see worker.log", task, episode)
                return None
        if (diff := compared(out)) is None or diff.episode != episode:
            expert_root, profile = Path(self.manifest.source_root), self.profile
            load = functools.partial(reference, out, recorded, expert_root, profile)
            replayed = await anyio.to_thread.run_sync(load)
            await compare_task(out, replayed, self.config, planner, profile)
        checked = validation(out)
        if checked is None:
            checked = await self.validate(task, out, episode, gpu, planner)
            if checked is None:
                return None  # infrastructure: a resumed run validates it again
            render_page(out)
        if checked.passed:
            return True
        reject(out, checked)
        render_page(out)
        log.info("%s: episode %d failed validation", task, episode)
        return False

    def command(self) -> list[str]:
        """This script, for a process of its own."""
        demos = ["--demos", str(self.path.parent), "--out", str(self.out)]
        return [sys.executable, __file__, *demos, *self.settings]

    def record(self, task: TaskName, out: Path, episode: int, seed: int) -> list[str]:
        """The simulator process that records a candidate into `out`."""
        where = ["--manifest", str(self.path), "--out", str(out)]
        what = ["--task", task, "--episode", str(episode), "--seed", str(seed)]
        record = "roboscript_robotwin.sim.record"
        return [sys.executable, "-m", record, *where, *what, *self.settings]

    async def process(self, out: Path, command: list[str], gpu: int | None) -> bool:
        """Run `command` for the candidate in `out` (on `gpu`, or on none) in a process group of
        its own, logging to `out`/worker.log: whether it exited cleanly. Cancelled, it stops the
        whole group."""
        await anyio.Path(out).mkdir(parents=True, exist_ok=True)
        with (out / "worker.log").open("ab") as output:
            process = await anyio.open_process(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)},
                start_new_session=True,  # its children (the review's ffmpeg) stop with it
            )
        try:
            return await process.wait() == 0
        finally:
            if process.returncode is None:
                with anyio.CancelScope(shield=True):
                    await stop_groups([(process, STOP_GRACE_S)])

    async def validate(
        self, task: TaskName, out: Path, episode: int, gpu: int, planner: Planner
    ) -> Validation | None:
        """Validate the candidate in `out` on an env server of the task's own; once more when
        infrastructure failed it, else None."""
        rec = self.config.recording
        logs = self.out / "services" / out.relative_to(self.out)
        for attempt in (1, 2):
            server = replay.env_server(
                task,
                "clean",
                gpu,
                logs,
                self.interpreters,
                self.pythonpath,
                self.token,
                self.http,
            )
            try:
                async with server as (control, agent_of):
                    return await validate(
                        control,
                        agent_of,
                        planner,
                        self.profile,
                        out / "package",
                        episode,
                        out / "referee",
                        replay.ReplayConfig(),
                        attempts=rec.validation_attempts,
                        spacing=(rec.path_step_rad, rec.path_step_m),
                        grip_threshold=self.config.keypoints.grip_threshold,
                    )
            except (StackError, InfraError) as error:  # this task's env server failed
                log.error("%s: validating episode %d (%d of 2): %s", task, episode, attempt, error)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
