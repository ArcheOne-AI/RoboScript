"""The pipeline's scheduler: each task's S1, then each variant's S2 and P3
(`robotwin_pipeline.run`), as jobs in worker processes of their own
(`robotwin_pipeline.worker`), in parallel under the GPU and concurrency limits,
resumable, and stopped cleanly by SIGTERM, or SIGHUP unless ignored (`nohup`).

```python
directory = new_directory(config)  # <root>/<stamp>/ with pipeline.yaml
summary = await run_pipeline(config, wiring, directory, gateway=GatewayCheck(url, token))
config = load_directory(directory)  # to launch it again
```

1. Before anything starts (`preflight`): one launch per directory (`.lock`); no job that started
   would change under the config (`definition`); no process group an earlier launch recorded is
   alive (after a launch that was killed: listed, with the command to stop them; records of
   groups that are gone are cleared); the interpreters, `ROBOTWIN_ROOT`, the env servers' token;
   for the pending jobs, their env configs, demo packages and ground-truth layers, protocol
   parameters (the recipe's with the config's) and brains, P3's config for every GPU, its Agent
   host (config, brain, registered toolsets) and install; the GPUs `nvidia-smi` lists (one it
   does not is skipped for the launch); the gateway's `/readyz`, and `/v1/models` serving every
   model the jobs use (the brains', and the scene toolsets' VLM).
2. A GPU's services (`per_gpu`: the motion service, a vision service) start when a job first needs
   the GPU (no GPU whose services run or come up has room for the waiting jobs) and it has room for
   them all and the job (`service_gb` each, counted as for jobs below; a GPU others load joins when
   it has room), and stop once no job of ours was on it for `service_idle_s`; each in a stack of
   its own (`services/gpu<n>/<service>/`), all with the pipeline's blob store. One that dies
   restarts on its port after each of `service_restart_waits_s`, so that the URLs the GPU's running
   jobs hold stay valid (no job is placed there meanwhile); if that port is taken, it restarts on a
   new one and the GPU's running jobs are stopped as infrastructure failures (to be respawned with
   the new URL), as they are when the GPU goes down for good: no job keeps calling a dead URL. One
   that came up and then was not ready (`/readyz`) for `service_unready_s` straight is stopped and
   restarts the same way; one ready for `service_healthy_s` straight has its restarts back. One
   whose log shows it ran out of GPU memory restarts when there is room again, without counting a
   restart, for `pressure_retry_s` at most. If every GPU goes down before one has all its services
   up, the launch stops before any job starts. The jobs get their GPU's service URLs: S2's scene
   toolsets and P3's Agent host cross-check with its vision service.
3. Jobs: each task's S1, then its variants' S2 and P3 as each is ready (a variant whose S1 did not
   pass is `blocked`), each in a worker process on the first GPU in `gpus` order with room for its
   places (one, `P3_PLACES` per shard for P3): `free_gb_per_place` per place under `gpu_cap_gb` less
   `gpu_margin_gb` (GB as nvidia-smi's MiB read: `MIB_PER_GB`), counting the memory in use
   (`nvidia-smi`, `roboscript.stack.gpus`), each placement of ours as reserved at the
   larger of its estimate and the peak its processes showed, for as long as it runs, less what they
   show, and what others claim in the shared `gpu_claims` file (where our placements are claimed
   too, each also in the pipeline's `gpu_claims.mine`, so that a launch again tells its own claims
   from others'; every `PUBLISH_S` our claims there are replaced by what we hold now that nvidia-smi
   does not show, a line per GPU); `max_per_gpu` places of ours at most, a sanity bound. A task's
   jobs reserve the most its jobs of as many places showed, once one ran for `PEAK_LEARN_S`
   (`gpu_peaks.json`; a task none of whose jobs of as many places ran yet: for a sharded P3 block
   its shards times its task's unsharded blocks', else the most any task's did), plus
   `peak_headroom_gb`, instead of the estimate; a running job's placement drops to its own peak plus
   that headroom once it ran `PEAK_LEARN_S`. A GPU that a reading shows over the cap less the margin
   takes no job (nor service) of ours for `OVER_BACKOFF_S`. Waiting jobs place in turn: those that
   ran before (in this launch or one before it) first, then those that have not, among each the
   later stages (S2, P3) of tasks whose S1 is done before S1s, so that tasks finish before new ones
   start (`placement_order: by_stage`: S1s before S2s before P3s, those that ran before first in
   each). Under `p3.randomized: spare`, a variant's randomized P3 blocks are a unit of their own
   after its clean ones (also of a variant that had ended without them): its turn comes after every
   other (`SPARE_RANK`), and it places, or starts a GPU's services, only while every job waiting for
   room is such a unit (a job not yet ready, e.g. a variant whose S1 runs, does not wait), so that
   any clean block goes first; the status shows it waiting (`SPARE_NOTE`). A spare unit that runs
   goes on while jobs that come after it take the next room; once one of them waited
   `spare_preempt_after_s` for room that stopping spare units on a GPU would give it, the newest of
   those are stopped as a launch's stop stops a job (P3's server and client together, the open block
   set aside), `preempted`, not failed: no respawn is used, and each waits for spare room again and
   resumes its block from the episodes it finished (`PREEMPT_POLL_S`, `Slots.preempt`). With
   `p3.spare_parallel`, that unit is a job of its own, the variant's `SPARE` lane (its records in
   `<variant>/P3/randomized/`: state, worker, log), asking for spare room as soon as the Memory
   froze, so it may run while the variant's clean blocks run; each of the two stops, cleans up,
   respawns and is preempted on its own records and blocks only, and the variant is done once
   both ended (failed, if its lane failed). At most
   `max_runs` run at once, and none starts while the disk has less than `min_free_gb` free (the
   status carries the alert). A worker that dies, or that gives no heartbeat for `watchdog_s`
   (killed), leaves its job failed on infrastructure and what it started, which the scheduler stops.
   A job infrastructure failed is `stopped` while it waits for each of `respawn_waits_s`, then
   respawned from its own records; if its logs show it ran out of GPU memory, it waits for room and
   retries without using a respawn, for `pressure_retry_s` at most (the status reports it). Any
   other failure is not retried. An error of the scheduler's own while it handles a job fails that
   job alone.
4. `status.json` and `status.md` every `status_interval_s` and at the end; `summary.json` and
   `summary.md` once every job has ended (`robotwin_pipeline.report`).

`launches.jsonl` records each launch (its pid, the code's version, its GPUs, the jobs, the errors
they had). Launching again on the directory continues every job that has not ended (queued,
stopped for any reason, blocked, running when its launch died, failed on infrastructure) and with
`retry_failed` those that failed otherwise, each afresh: its earlier error and failure cleared.
Each launch first brings every variant whose S2 stopped at its seed cap without converging in
line with `p3.unconverged` (`run.gate_variant`): under `best_memory` (the default) one that stopped
with no pick recorded gets its best first-try version frozen (`run.pick_s2`; P3 blocks it ran on
another version are moved aside to run again) and is taken up at P3, and one with no first-try
success ends `not_converged` (P3 counted 0); under `stop` each ends `not_converged`, whatever P3 it
had started (its lane too): no P3 of it runs, its blocks done before stay on disk and do not count;
under `last` such a variant is taken up at P3 on the Memory frozen at the cap. `p3_modes` reopens
the variants that froze for P3 in those modes (C2R later: `randomized`), new blocks beside the
done ones, and runs nothing else.
"""

import contextlib
import fcntl
import json
import logging
import os
import signal
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Final

import anyio
import anyio.abc
import httpx
import yaml
from pydantic import SecretStr

from roboscript.contracts.ids import TaskName
from roboscript.envs import control_token
from roboscript.kernel import entrypoints
from roboscript.learning import (
    LearningConfig,
    brain_maker,
    load_demo_truth,
    load_package,
    set_aside,
)
from roboscript.learning.agent_host import AgentHostConfig
from roboscript.learning.config import BrainSpec
from roboscript.learning.demo import PACKAGE_FILE
from roboscript.memory import FileMemoryStore, LintConfig, lint, load_recipe, read_doc
from roboscript.memory import errors as lint_errors
from roboscript.orchestrate import worker
from roboscript.orchestrate.graph import ComposedGraph, Records, run_graph, walk
from roboscript.orchestrate.jobs import PIDS, RunKey, RunState, pending
from roboscript.orchestrate.launch import WORKER, Launch, LaunchSettings, alive
from roboscript.records import write_atomic
from roboscript.settings import (
    load_config_file,
    load_config_ref,
    repo_root,
    runs_path,
    source_sha,
)
from roboscript.stack import (
    Interpreters,
    ServiceSpec,
    StackConfig,
    cancel_on_signals,
)
from roboscript.stack.gpus import GpuProbe, LogMarks, nvidia_smi
from roboscript.stack.shared import (
    StackPlan,
    stack_key,
)
from roboscript.toolsets.scene import SceneConfig
from roboscript_methods.wicl.imitate import EXIT, ImitateConfig, ImitateResult, resume_plan
from roboscript_methods.wicl.jobs import PHASES
from roboscript_methods.wicl.memory.arms import unbind_arms
from roboscript_methods.wicl.ops import FALLBACK, S1, imitate_values, practice_values, s2_stop
from roboscript_methods.wicl.practice import PracticeConfig
from roboscript_methods.wicl.roles import demo_context_brief
from roboscript_robotwin.config import robotwin_root
from roboscript_robotwin.eval import install
from roboscript_robotwin.eval import jobs as p3_jobs
from roboscript_robotwin.eval.blocks import (
    BLOCK_RUN,
    LANE_DIRS,
    SPARE,
    block_plan,
    close_blocks,
    open_blocks,
    owned_modes,
    p3_pending,
)
from roboscript_robotwin.eval.blocks import redo_p3 as redo_p3_blocks
from roboscript_robotwin.eval.job import p3_config
from roboscript_robotwin.eval.jobs import (
    CLEAN,
    PHASE,
    RANDOMIZED,
)
from roboscript_robotwin.eval.launch import (
    ENDED,
    VISION,
    P3Config,
    P3Run,
    host_config,
    scene_names,
)
from robotwin_pipeline import report
from robotwin_pipeline.config import CONFIG, PipelineConfig, check_run
from robotwin_pipeline.run import (
    BENCHMARKS,
    BLOBS,
    Mode,
    Wiring,
    check_source,
    definition,
    end_targets_due,
    env_config,
    evaluation_factory,
    gate_variant,
    job_dir,
    method,
    one_line,
    p3_gated,
    p3_held,
    run_keys,
    run_models,
    s2_exit,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewayCheck:
    """The model gateway's URL (`.../v1`), the token that lists its models, and the models the
    wiring's toolsets ask besides the brains' (e.g. the scene toolsets' VLM, `vlm.model`)."""

    url: str
    token: SecretStr
    models: frozenset[str] = frozenset()


def new_directory(config: PipelineConfig) -> Path:
    """A new `<root>/<stamp>/` with the config (`pipeline.yaml`)."""
    directory = runs_path(config.root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True)
    _save(directory, config)
    return directory


async def run_pipeline(
    config: PipelineConfig,
    wiring: Wiring,
    directory: Path,
    *,
    retry_failed: bool = False,
    gateway: GatewayCheck | None = None,
    note: Mapping[str, Any] | None = None,
    p3_modes: Sequence[Mode] | None = None,
    s1_only: bool = False,
    redo_p3: Sequence[tuple[TaskName, Mode]] = (),
) -> dict[str, Any]:
    """Run the pending jobs of `directory` (see the module docstring; `note` joins the launch's
    line in `launches.jsonl`); the summary once every job has ended. The code must name its
    version, and a resume run the version the directory started with (`run.check_source`). A
    signal stops it with `roboscript.stack.Stopped`, after its cleanup. `redo_p3`: (task, mode)
    whose variants' P3 blocks of that mode are discarded and run again from scratch
    (`eval.blocks.redo_p3`)."""
    if p3_modes is not None and not config.p3.blocks:
        raise ValueError("p3.blocks is empty: there is no P3 block to evaluate")
    check_source(directory)
    note = dict(note or {})
    if unknown := sorted({t for t, _ in redo_p3} - set(config.tasks)):
        raise ValueError(f"--redo-p3 {unknown}: none of the pipeline's tasks")
    with _lock(directory):
        # the prechecks (seed split, env capabilities, the plugins'), before anything below
        # moves, prunes or rewrites the directory's records
        await check_run(config)
        if redo_p3 and (alive := leftovers(directory)):
            raise RuntimeError(
                f"processes of an earlier launch in {directory} are still running "
                f"({', '.join(where for _, where in alive)}): nothing is moved aside under them"
            )
        variants = list(config.variants)
        # before the gate, which may freeze another version: a config that changes the jobs'
        # definitions (e.g. `p3.unconverged`) is refused before it moves anything of theirs
        _check_definitions(config, directory, s1_only=s1_only)
        gated = {
            f"{task}/{variant}": how
            for task in config.tasks
            for variant in variants
            if (how := gate_variant(config, directory, RunKey(task, variant))) is not None
        }
        redone_p3 = {
            f"{task}/{variant}/{mode}": moved
            for task, mode in redo_p3
            for variant in variants
            if (moved := redo_p3_blocks(directory, task, variant, mode))
        }
        closed = await _close_cut(config, directory, p3_modes, s1_only)
        pending = await _pending_jobs(config, directory, retry_failed, p3_modes, s1_only)
        host = preflight(config, wiring, directory, pending)
        probe = wiring.gpu_memory or nvidia_smi
        gpus, skipped = await usable_gpus(config, probe) if pending else (config.gpus, {})
        _save(directory, config)  # what a later launch continues
        had = _settle(directory, pending)
        launch = _Launch(
            config,
            wiring,
            directory,
            retry_failed,
            gpus,
            skipped,
            p3_modes,
            s1_only=s1_only,
        )
        more = {"gpus": gpus, "skipped": skipped, "p3_modes": p3_modes}
        if closed:
            more["closed_p3"] = closed
        more["gated"] = gated
        more["redone_p3"] = redone_p3
        more["s1_only"] = s1_only
        more["errors"] = had
        if launch.key is not None:
            more["shared_stacks"] = {"registry": str(config.shared_stacks), "key": launch.key}
        _record_launch(directory, pending, retry_failed, {**(note or {}), **more})
        try:
            with cancel_on_signals(signal.SIGTERM, signal.SIGHUP):
                if gateway is not None:
                    models = _models(config, wiring, host) | gateway.models
                    async with httpx.AsyncClient(trust_env=False) as http:
                        await check_gateway(http, gateway, models)
                if pending:
                    await launch.run()
        finally:
            report.write_status(directory, config, launch.extra())
        return await anyio.to_thread.run_sync(report.write_summary, directory, config)


async def _close_cut(
    config: PipelineConfig,
    directory: Path,
    p3_modes: Sequence[Mode] | None,
    s1_only: bool,
) -> dict[str, list[str]]:
    """Before the launch counts its jobs: the variants' P3 blocks whose set-aside runs finished
    the episodes the config asks for are closed from them (`eval.blocks.close_blocks`, nothing
    run), so no job starts for them. Open blocks are set aside first (`prune`); with processes of an
    earlier launch alive nothing is done here (`preflight` refuses the launch). The blocks
    closed, by job."""
    if s1_only or not config.p3.blocks or leftovers(directory):
        return {}
    prune(directory)
    closed = {}
    for task in config.tasks:
        for variant in config.variants:
            run_dir = directory / task / variant
            if not (run_dir / "P3").is_dir():
                continue
            names = await close_blocks(config.p3, run_dir, task, p3_modes or config.p3.modes)
            if names:
                log.info("%s/%s: P3 blocks closed from set-aside runs: %s", task, variant, names)
                closed[f"{task}/{variant}"] = names
    return closed


def preflight(
    config: PipelineConfig, wiring: Wiring, directory: Path, pending: Sequence[RunKey]
) -> AgentHostConfig | None:
    """What would fail a job later, checked now (see the module docstring); P3's Agent host
    config, when a pending job may reach P3."""
    if alive := leftovers(directory):
        kill = " ".join(f"-{pid}" for pid, _ in alive)
        shown = "\n".join(f"  {where} (process group {pid})" for pid, where in alive)
        raise RuntimeError(
            f"processes of an earlier launch in {directory} are still running:\n{shown}\n"
            f"stop them (kill -TERM -- {kill}) and launch again"
        )
    prune(directory)
    _interpreters(config.interpreters, CONFIG)
    root = robotwin_root()
    if root is None:
        raise ValueError("set ROBOTWIN_ROOT in the environment or .env")
    if not pending:
        return None
    for gpu in config.gpus:
        for spec in config.per_gpu:
            _service_stack(config, directory, gpu, spec)  # validates the interpreters
    if config.shared_stacks is not None:
        stack_key(shared_plan(config))  # the services' code and configs can be read
    _check_vision(config)
    if getattr(config.interpreters, BENCHMARKS[config.benchmark][0].interpreter) is None:
        raise ValueError(f"{config.benchmark}'s env servers need interpreters.sim")
    control_token()
    learning = LearningConfig.model_validate({**config.learning, "run_dir": directory})
    if wiring.brains is None:
        for spec in learning.brains.values():
            brain_maker(spec)
    recipe = config.recipe().protocols
    for key in pending:
        env_config(config, key.task, directory / BLOBS)
        if key.variant == S1:
            _check_demo(config, recipe, key.task, directory)
        else:
            values = {**recipe.get("practice", {}), **practice_values(config, key)}
            practice = PracticeConfig.model_validate(values)
            if practice.demo_truth is not None and load_demo_truth(practice.demo_truth) is None:
                raise ValueError(f"{practice.demo_truth} has no ground-truth layer (practice)")
            if practice.demo is not None:  # a package without the level's content fails here
                demo_context_brief(load_package(practice.demo), practice.demo_context)
    variants = [key for key in pending if key.variant != S1]
    if not (config.p3.blocks and variants):
        return None
    base = run_models(config, load_config_file(config.p3.config, P3Config))
    _interpreters(base.interpreters, str(config.p3.config))
    # Stand-in URLs: what is checked is the config each job's launcher would get.
    vision = "http://127.0.0.1:2" if VISION in {s.name for s in config.per_gpu} else None
    launchers = [
        p3_config(base, directory, gpu, "http://127.0.0.1:1", vision, directory / BLOBS)
        for gpu in config.gpus
    ]
    check = directory / "p3-check"  # never written: the host config names its records
    run = P3Run(task=variants[0].task, seed=config.p3.blocks[0], manifest=check, out_dir=check)
    host = host_config(run, launchers[0], check)
    registered = set(entrypoints.names("roboscript.toolsets"))
    if missing := sorted({spec.name for spec in host.toolsets} - registered):
        raise ValueError(f"P3's Agent host mounts toolsets that are not registered: {missing}")
    brain_maker(host.brain)
    if wiring.brains is None:  # tests' brains replace both
        check_actor_brain(config, learning, host.brain)
    install.check_installed(root.resolve())
    return host


def check_actor_brain(config: PipelineConfig, learning: LearningConfig, brain: BrainSpec) -> None:
    """P3's Actor is S2's: the launcher's host brain (`p3.config`, outside the layered config)
    must equal `learning.brains.actor` (name and the whole config: model, reasoning effort and
    the rest), unless `p3.actor_brain_differs` says otherwise, which the run's pipeline.yaml
    records."""
    actor = learning.brains.get("actor", BrainSpec())
    if brain == actor or config.p3.actor_brain_differs:
        return
    raise ValueError(
        f"P3's Actor brain in {config.p3.config} ({brain.model_dump()}) differs from "
        f"learning.brains.actor ({actor.model_dump()}), which S2 practised with: make them "
        "equal, or set p3.actor_brain_differs: true"
    )


async def usable_gpus(config: PipelineConfig, probe: GpuProbe) -> tuple[list[int], dict[int, str]]:
    """The config's GPUs a launch uses, in order, and those it skips, with why: the ones `probe`
    does not list. A GPU others load is used: it takes services and jobs as it has room."""
    memory = await probe()
    skipped = {gpu: "nvidia-smi does not list it" for gpu in config.gpus if gpu not in memory}
    if skipped:
        log.warning("skipping GPUs %s", skipped)
    gpus = [gpu for gpu in config.gpus if gpu not in skipped]
    if not gpus:
        raise RuntimeError(f"no GPU to run on: {skipped}")
    return gpus, skipped


def _check_vision(config: PipelineConfig) -> None:
    """A GPU's vision service reads the images its callers put in the pipeline's blob store,
    which its stack gives it: its config must not name another."""
    for spec in config.per_gpu:
        if spec.module != "roboscript_vision" or "--config" not in spec.args:
            continue
        from roboscript_vision.config import VisionConfig  # its config only: no torch here

        ref = spec.args[spec.args.index("--config") + 1]
        if load_config_ref(ref, VisionConfig).blob_dir is not None:
            raise ValueError(f"per_gpu {spec.name}: leave blob_dir unset in {ref} (the pipeline's)")


def _check_demo(
    config: PipelineConfig, recipe: Mapping[str, dict[str, Any]], task: str, directory: Path
) -> None:
    if any((directory / task / S1 / task / "S1").glob(f"*/{EXIT}")):
        return  # S1 exited: its package was read
    demo = config.demo_of(task)
    values = {**recipe.get("imitate", {}), **config.protocol("imitate", task), "demo": demo}
    imitate = ImitateConfig.model_validate(values)
    if not (demo / PACKAGE_FILE).is_file():
        raise FileNotFoundError(f"{task}: no demo package in {demo}")
    if (package := load_package(demo)).task != task:
        raise ValueError(f"{demo} demonstrates {package.task}, not {task}")
    if imitate.demo_truth and load_demo_truth(demo) is None:
        raise ValueError(f"{demo} has no ground-truth layer (imitate.demo_truth is on)")


async def check_gateway(http: httpx.AsyncClient, gateway: GatewayCheck, models: set[str]) -> None:
    """The gateway answers `/readyz` and serves every one of `models` (`/v1/models`)."""
    base = gateway.url.rstrip("/")
    root = base.removesuffix("/v1")
    bearer = {"authorization": f"Bearer {gateway.token.get_secret_value()}"}
    try:
        ready = await http.get(f"{root}/readyz", timeout=10)
        listed = await http.get(f"{base}/models", headers=bearer, timeout=10)
    except httpx.TransportError as error:
        raise RuntimeError(f"the model gateway at {root} does not answer: {error!r}") from None
    if ready.status_code != 200:
        raise RuntimeError(f"the model gateway at {root} is not ready ({ready.status_code})")
    if listed.status_code != 200:
        raise RuntimeError(f"the model gateway at {root} lists no models ({listed.status_code})")
    served = {item["id"] for item in listed.json()["data"]}
    if missing := sorted(models - served):
        raise RuntimeError(f"the model gateway at {root} does not serve {missing}")


def _models(config: PipelineConfig, wiring: Wiring, host: AgentHostConfig | None) -> set[str]:
    """The models the jobs' brains name, the config's (unless the wiring replaces them) and P3's
    Agent host's, and the VLM of the host's scene toolsets."""
    specs: list[BrainSpec] = []
    if wiring.brains is None:
        learning = LearningConfig.model_validate({**config.learning, "run_dir": config.root})
        specs += learning.brains.values()
    if host is not None:
        specs.append(host.brain)
    models = {model for spec in specs if isinstance(model := spec.config.get("model"), str)}
    if host is not None:
        scenes = scene_names(spec.name for spec in host.toolsets)
        scene = [spec.config for spec in host.toolsets if spec.name in scenes]
        models |= {SceneConfig.model_validate(values).vlm.model for values in scene}
    return models


def leftovers(directory: Path) -> list[tuple[int, str]]:
    """The process groups that launches in `directory` recorded and that are alive: the GPUs'
    services, the workers, their env servers (`.roboscript/pids.json`, `worker.json`) and P3's
    open blocks (`pids.json`)."""
    return sorted((pid, where) for pid, where in _recorded(directory).items() if alive(pid))


def prune(directory: Path) -> None:
    """Clear the records of groups that are gone (so that a reused pid never looks like one of
    them later), setting aside the P3 blocks they were open in."""
    for path in _record_files(directory):
        if not any(alive(pid) for pid in json.loads(path.read_bytes()).values()):
            path.unlink()
    for run_dir in directory.glob("*/*"):
        for out in open_blocks(run_dir):
            set_aside(out)


def _record_files(directory: Path) -> list[Path]:
    """The pid records: a GPU's services (`services/gpu<n>/<service>/`), a job's env stack and
    worker, a lane's worker (`run.LANE_DIRS`)."""
    patterns = [f"*/*/{PIDS}", f"*/*/{WORKER}", f"services/*/*/{PIDS}"]
    patterns += [f"*/*/{lane}/{WORKER}" for lane in LANE_DIRS.values()]
    return [path for pattern in patterns for path in directory.glob(pattern)]


def _recorded(directory: Path) -> dict[int, str]:
    groups: dict[int, str] = {}
    for path in _record_files(directory):
        owner = path.parent.parent.relative_to(directory)  # the job's (or service's) records
        groups |= {pid: f"{owner}: {name}" for name, pid in json.loads(path.read_bytes()).items()}
    for run_dir in directory.glob("*/*"):
        for out in open_blocks(run_dir):
            if (out / "pids.json").is_file():
                pids = json.loads((out / "pids.json").read_bytes())
                owner = out.relative_to(directory)
                groups |= {pid: f"{owner}: {name}" for name, pid in pids.items()}
    return groups


class _Launch(Launch):
    """One launch of the pipeline's jobs (`roboscript.orchestrate.launch.Launch`): wicl's
    jobs with RoboTwin's evaluation, the pipeline's workers, records and status."""

    def __init__(
        self,
        config: PipelineConfig,
        wiring: Wiring,
        directory: Path,
        retry_failed: bool,
        gpus: Sequence[int] | None = None,
        skipped: Mapping[int, str] | None = None,
        modes: Sequence[Mode] | None = None,
        *,
        s1_only: bool = False,
    ) -> None:
        self.config, self.wiring = config, wiring
        self.retry_failed, self.modes, self.s1_only = retry_failed, modes, s1_only
        super().__init__(
            launch_settings(config, gpus),
            directory,
            blobs=directory / BLOBS,
            plan=None if config.shared_stacks is None else shared_plan(config),
            gpu_memory=wiring.gpu_memory,
            processes=wiring.processes,
            skipped=skipped,
        )

    async def _tasks(self) -> None:
        """The launch's job graph (`job_graph`), run to its end."""
        graph = job_graph(
            self.config, self.directory, self.retry_failed, self.modes, s1_only=self.s1_only
        )
        await run_graph(graph, self._run)

    def _command(
        self,
        key: RunKey,
        phase: str,
        gpu: int,
        urls: Mapping[str, str],
        modes: Sequence[str] | None,
    ) -> list[str]:
        return worker.command(self.wiring.worker, self.directory, phase, key, gpu, urls, modes)

    def _write_status(self, extra: dict[str, Any]) -> None:
        report.write_status(self.directory, self.config, extra)

    def _dir(self, key: RunKey) -> Path:
        """A job's records (`run.job_dir`: a lane's apart)."""
        return job_dir(self.directory, key)

    def _content(self, key: RunKey) -> Path:
        """The directory a job works in: its variant's (a lane's too)."""
        return self.directory / key.task / key.variant

    def _owned(self, key: RunKey) -> list[Mode] | None:
        """The P3 modes whose blocks a job's processes run (None: all): a `--p3-mode` launch
        runs no lane beside the variant's job, and S1 runs no P3."""
        if self.modes is not None or key.variant == S1:
            return None
        return owned_modes(self.config.p3, key)

    def _logs(self, key: RunKey) -> LogMarks:
        """A job's logs, as they are now: a lane's own; the variant's job's, not its lane's."""
        if key.lane is not None:
            return LogMarks(self._dir(key))
        lanes = [self._content(key) / lane for lane in LANE_DIRS.values()]
        return LogMarks(self._dir(key), apart=lanes if self.config.p3.parallel else ())

    def _job_groups(
        self, run_dir: Path, records: Path | None = None, modes: Sequence[str] | None = None
    ) -> list[int]:
        return _job_groups(run_dir, records, modes)  # type: ignore[arg-type]

    def _open_blocks(self, run_dir: Path, modes: Sequence[str] | None = None) -> list[Path]:
        return open_blocks(run_dir, modes)  # type: ignore[no-any-return]

    def _ended_groups(self, run_dir: Path, modes: Sequence[str] | None = None) -> list[int]:
        """The shard clients its open P3 blocks' runs record ended for good (`ENDED`)."""
        files = [out / ENDED for out in open_blocks(run_dir, modes)]  # type: ignore[arg-type]
        return [pid for path in files if path.is_file() for pid in _pids(path)]

    def _service_stack(self, gpu: int, spec: ServiceSpec, port: int | None = None) -> StackConfig:
        return _service_stack(self.config, self.directory, gpu, spec, port)


def launch_settings(config: PipelineConfig, gpus: Sequence[int] | None = None) -> LaunchSettings:
    """What a launch of the config runs under, on `gpus` (None: the config's)."""
    names = set(LaunchSettings.model_fields) - {"gpus", "preempt", "pythonpath"}
    return LaunchSettings.model_validate(
        {
            **{name: getattr(config, name) for name in names},
            "gpus": list(config.gpus if gpus is None else gpus),
            "preempt": config.p3.spare,
            "pythonpath": [Path(__file__).resolve().parents[1]],  # where `robotwin_pipeline` is
        }
    )


def shared_plan(config: PipelineConfig) -> StackPlan:
    """What a GPU's shared stack runs for the pipeline (`shared_stacks`): its `per_gpu` services
    from this repository, their memory (`service_gb`) claimed in `gpu_claims`, restarted as its
    own would be."""
    return StackPlan(
        services=config.per_gpu,
        interpreters=config.interpreters,
        sim_pythonpath=config.sim_pythonpath,
        vision_pythonpath=config.vision_pythonpath,
        root=repo_root(),
        mib={spec.name: config.service_mib(spec.name) for spec in config.per_gpu},
        claims=config.gpu_claims,
        restart_waits_s=config.service_restart_waits_s,
        pressure_retry_s=config.pressure_retry_s,
        healthy_s=config.service_healthy_s,
        linger_s=config.shared_linger_s,
    )


def _service_stack(
    config: PipelineConfig, directory: Path, gpu: int, spec: ServiceSpec, port: int | None = None
) -> StackConfig:
    """A GPU's service (`<name>-gpu<n>`) in a stack of its own (`services/gpu<n>/<name>/`), with
    the pipeline's blob store, on `port` (None: any free one)."""
    args = [*spec.args, *(["--port", str(port)] if port is not None else [])]
    named = ServiceSpec.model_validate(
        {**spec.model_dump(), "name": f"{spec.name}-gpu{gpu}", "gpu": gpu, "args": args}
    )
    return StackConfig(
        interpreters=config.interpreters,
        sim_pythonpath=config.sim_pythonpath,
        vision_pythonpath=config.vision_pythonpath,
        run_dir=directory / "services" / f"gpu{gpu}" / spec.name,
        blob_dir=directory / BLOBS,
        services=[named],
    )


async def _pending_jobs(
    config: PipelineConfig,
    directory: Path,
    retry_failed: bool,
    modes: Sequence[Mode] | None,
    s1_only: bool = False,
) -> list[RunKey]:
    """The jobs this launch takes up, read from its job graph (`job_graph`) with the records as
    they are now (`graph.walk`, `ComposedGraph.taken_up`): a unit it takes up whose S1 has passed
    or may yet (none with `s1_only`), and under `p3.randomized: spare` a variant that ended with
    randomized blocks to run (under `p3.spare_parallel`, its lane, `SPARE`, once its Memory froze,
    beside the variant's job); with `modes`, the variants that froze and have P3 blocks of `modes`
    to run."""
    graph = job_graph(config, directory, retry_failed, modes, s1_only=s1_only)
    jobs = graph.taken_up(await walk(graph))
    if modes is not None and not s1_only:  # it reports what it leaves out
        for task in config.tasks:
            for variant in config.variants:
                key = RunKey(task, variant)
                if key not in jobs and end_targets_due(config, directory, key) is not None:
                    log.warning(
                        "%s: left out of this P3-mode launch: the end targets of S2's best "
                        "Memory are read first, in a launch that runs its practice phase",
                        key.name,
                    )
    return jobs


def job_graph(
    config: PipelineConfig,
    directory: Path,
    retry_failed: bool,
    modes: Sequence[Mode] | None,
    *,
    s1_only: bool = False,
) -> ComposedGraph:
    """A launch's job graph: wicl's jobs (`run.method`, `Wicl.jobs`) of each task, with RoboTwin's
    evaluation of each variant (`P3Jobs`), held from P3 by `run.p3_held`, once each phase has a
    body (`worker.check_kinds`: else nothing starts); a variant whose Memory froze practises again
    while the end targets of S2's pick are to be read (`run.end_targets_due`; its practice returns
    at once, then reads them)."""
    factory = evaluation_factory()
    evaluation = factory(config.p3, gated=partial(p3_held, config), modes=modes)
    learning = method()
    learn = learning.jobs(
        list(config.variants),
        evaluation,
        s1_only=s1_only,
        evaluate_only=modes is not None,
        practice_again=lambda key: end_targets_due(config, directory, key) is not None,
    )
    kinds = worker.job_kinds(getattr(learning, "kinds", {}), factory.kinds)
    if problems := worker.check_kinds(kinds, learn.stages):
        raise ValueError(f"the launch's job bodies: {'; '.join(problems)}")
    records = Records(directory, partial(job_dir, directory), retry_failed)
    graph = ComposedGraph(learn.stages, records)
    for task in config.tasks:
        learn.add(graph, task)
    return graph


STAGE_ORDER: Final = (*PHASES, PHASE)
"""The stages in the order `by_stage` places them (`PipelineConfig.placement_order`)."""


def _reopens(
    config: PipelineConfig,
    run_dir: Path,
    state: RunState,
    retry_failed: bool,
    modes: Sequence[Mode],
) -> bool:
    """`P3Jobs`' rule for a variant that froze, with this pipeline's S2 gate and the end targets
    of S2's picked version read first (`run.p3_held`)."""
    gated = partial(p3_held, config)
    return p3_jobs.reopens(config.p3, run_dir, state, retry_failed, modes, gated)


def _lane_due(
    config: PipelineConfig, directory: Path, key: RunKey, retry_failed: bool, frozen: bool
) -> bool:
    """`P3Jobs`' rule for a variant's spare lane, with this pipeline's S2 gate and the end targets
    of S2's picked version read first (`run.p3_held`)."""
    lane_key = RunKey(key.task, key.variant, SPARE)
    main = RunState.load(directory / key.task / key.variant, key.main)
    lane = RunState.load(job_dir(directory, lane_key), lane_key)
    run_dir, gated = directory / key.task / key.variant, partial(p3_held, config)
    return p3_jobs.lane_due(config.p3, run_dir, main, lane, retry_failed, frozen, gated)


def _spare_due(config: PipelineConfig, run_dir: Path, state: RunState) -> bool:
    """Whether a variant that ended (`done`) has randomized blocks to run as a spare unit."""
    return state.status == "done" and _reopens(config, run_dir, state, False, RANDOMIZED)


def plan(
    config: PipelineConfig,
    directory: Path,
    *,
    retry_failed: bool = False,
    s1_only: bool = False,
    redo_p3: Sequence[tuple[TaskName, Mode]] = (),
    p3_modes: Sequence[Mode] | None = None,
) -> dict[str, Any]:
    """What a launch with these options would do (`--dry-run`), read from `directory` alone: no
    marker is written, nothing is moved or started. Per job (`jobs`, in start order): its status
    now and what the launch would do with it (`action`), with an S1's `imitate` mode and, for one
    it resumes, its finished attempts as the schedule replays them (`resume_plan`: the verdicts
    as counted, the attempt the replay exits at), with a variant's P3 blocks of the config's
    modes (or `p3_modes`) at their counts (`p3`, `eval.blocks.block_plan`: done, closed from a
    set-aside run, resumed or started); a variant whose S2 did not converge says what it goes on
    with (`run.s2_exit`) or that it ends `not_converged`, no P3 (`run.p3_gated`); and
    `definitions`, the jobs whose settings the config would change (which the launch
    refuses)."""
    learning = LearningConfig.model_validate({**config.learning, "run_dir": config.root})
    recipe = config.recipe().protocols.get("imitate", {})
    linter = load_recipe(learning.recipe, learning.lint_overlay).lint
    jobs: list[dict[str, Any]] = []
    for task in config.tasks:
        s1_dir = directory / task / S1
        state = RunState.load(s1_dir, RunKey(task, S1))
        exits = sorted((s1_dir / task / "S1").glob(f"*/{EXIT}"))
        ended = ImitateResult.model_validate_json(exits[-1].read_bytes()) if exits else None
        row: dict[str, Any] = {"job": f"{task}/{S1}", "status": state.status}
        if ended is not None:
            row["exit"] = {"passed": ended.passed, "attempts": ended.attempts, "v": ended.version}
        if state.status == "done":  # its exit stands: what it would resume with is moot
            values = {**recipe, **config.protocol("imitate", task)}
        else:
            values = {**recipe, **imitate_values(config, task)}
            row["max_attempts"] = values.get("max_attempts")
        if state.status == "done":
            row["action"] = "keep" + (" (fallback)" if (s1_dir / FALLBACK).is_file() else "")
        elif not pending(state, retry_failed):
            row["action"] = f"skip ({state.status})"
        else:
            row |= _resume_row(task, s1_dir, values, linter)
        jobs.append(row)
        for variant in config.variants:
            key = RunKey(task, variant)
            vstate = RunState.load(directory / task / variant, key)
            vrow: dict[str, Any] = {"job": key.name, "status": vstate.status}
            vdir = directory / task / variant
            gated = p3_gated(config, vdir, task)  # it ends not_converged
            if s1_only:
                vrow["action"] = "held (--s1-only)"
            elif gated:
                vrow["action"] = _gated_action(config, vdir, task, vstate)
            elif vstate.status == "not_converged":  # it may go on now
                found = s2_exit(config, vdir, task) or {}
                if found.get("s2_exit") == "best_memory":
                    vrow["action"] = f"S2's best Memory v{found['version']:03d}: its P3 goes on"
                else:
                    vrow["action"] = (
                        f"reopened (p3.unconverged {config.p3.unconverged}): P3 goes on"
                    )
            elif config.p3.parallel:
                vrow |= _parallel_row(config, directory, key, vstate, retry_failed)
            elif pending(vstate, retry_failed):
                vrow["action"] = "after S1: S2, then P3"
                if (
                    config.p3.spare
                    and vstate.stage == "p3"
                    and not p3_pending(config.p3, vdir, CLEAN)
                ):
                    vrow["action"] = "P3 randomized on spare GPU room"
                elif config.p3.spare:
                    vrow["action"] += " (randomized on spare GPU room)"
            elif config.p3.spare and _spare_due(config, directory / task / variant, vstate):
                vrow["action"] = "P3 randomized on spare GPU room"
            else:
                vrow["action"] = f"skip ({vstate.status})"
            redone = [m for t, m in redo_p3 if t == task]
            for mode in redone:
                folder = directory / task / variant / "P3" / mode
                names = sorted(p.name for p in folder.glob("block*") if BLOCK_RUN.fullmatch(p.name))
                runs = (
                    p3_modes is not None and mode in p3_modes
                    if p3_modes is not None or mode not in config.p3.modes
                    else mode == "clean" or config.p3.spare or pending(vstate, retry_failed)
                )
                how = "" if runs else f" (runs with --p3-mode {mode})"
                vrow["action"] = f"redo P3 {mode} from scratch{how}"
                vrow["moved_aside"] = [*vrow.get("moved_aside", []), *names]
            if config.p3.blocks and (vdir / "P3").is_dir() and not gated:
                modes = p3_modes or config.p3.modes
                vrow["p3"] = block_plan(config.p3, vdir, [m for m in modes if m not in redone])
                vrow["p3"] |= {
                    f"{m}/block{b}": f"start ({config.p3.episodes_of(m)} episodes)"
                    for m in redone
                    for b in config.p3.blocks
                }
            jobs.append(vrow)
    try:
        _check_definitions(config, directory, s1_only=s1_only)
        definitions = "unchanged"
    except ValueError as error:
        definitions = str(error)
    return {"jobs": jobs, "definitions": definitions}


def _parallel_row(
    config: PipelineConfig, directory: Path, key: RunKey, state: RunState, retry_failed: bool
) -> dict[str, Any]:
    """A variant's plan row under `p3.spare_parallel`: its own job, and its lane (`spare`: its
    status and whether it starts now)."""
    run_dir = directory / key.task / key.variant
    lane = RunKey(key.task, key.variant, SPARE)
    lane_state = RunState.load(job_dir(directory, lane), lane)
    runs = pending(state, retry_failed)
    frozen = state.stage == "p3"
    due = _lane_due(config, directory, key, retry_failed, frozen)
    if not frozen and runs:
        action = "after S1: S2, then P3 clean, its randomized lane beside it once S2 froze"
    elif runs and p3_pending(config.p3, run_dir, CLEAN):
        action = "P3 clean" + (f", and P3 randomized beside it ({lane.name})" if due else "")
    elif due:
        action = f"P3 randomized on spare GPU room ({lane.name})"
    elif runs:
        action = "P3 done: ends done"
    else:
        action = f"skip ({state.status})"
    found: dict[str, Any] = {"action": action}
    if frozen or lane_state.launches:
        found["spare"] = {
            "status": lane_state.status,
            "launches": lane_state.launches,
            "starts": due,
        }
    return found


def _gated_action(config: PipelineConfig, run_dir: Path, task: TaskName, state: RunState) -> str:
    """What a launch does with a variant whose S2 did not converge (`run.gate_variant`)."""
    stop = s2_stop(run_dir, task) or {}
    k = "" if stop.get("k") is None else f" at k={stop['k']}"
    found = block_plan(config.p3, run_dir, config.p3.modes) if (run_dir / "P3").is_dir() else {}
    done = sorted(name for name, how in found.items() if how.startswith("done"))
    kept = f"; P3 done before kept, not counted: {', '.join(done)}" if done else ""
    was = "" if state.status == "not_converged" else f" (was {state.status}, stage {state.stage})"
    if (s2_exit(config, run_dir, task) or {}).get("s2_exit") == "none":
        return (
            f"stop before P3: not_converged{was}: S2 stopped{k} with no first-try success, "
            "counted 0 in P3"
        )
    return (
        f"stop before P3: not_converged{was}: S2 stopped{k} without converging, no P3 or lane "
        f"starts{kept}"
    )


def plan_text(found: Mapping[str, Any]) -> str:
    """`plan`'s result as lines: one per job, then the counts of S1 actions and the definitions."""
    lines = []
    for job in found["jobs"]:
        parts = [job["job"], job["status"], job["action"]]
        if "max_attempts" in job:
            parts.append(f"cap {job['max_attempts']}")
        if "moved_aside" in job:
            parts.append(f"moves aside {', '.join(job['moved_aside'])}")
        if job.get("verdicts"):
            parts.append(job["verdicts"])
        if job.get("p3"):
            parts.append("P3 " + "; ".join(f"{k}: {v}" for k, v in job["p3"].items()))
        if lane := job.get("spare"):
            starts = "starts" if lane["starts"] else "nothing to run"
            parts.append(f"{SPARE} lane {lane['status']} ({lane['launches']} launches): {starts}")
        lines.append(" | ".join(str(part) for part in parts))
    kinds = Counter(
        job["action"].split(":")[0].split(" (")[0]
        for job in found["jobs"]
        if job["job"].endswith(f"/{S1}")
    )
    lines += ["", "S1: " + ", ".join(f"{kind} {n}" for kind, n in sorted(kinds.items()))]
    lines.append(f"definitions: {found['definitions']}")
    return "\n".join(lines)


def _resume_row(
    task: TaskName, s1_dir: Path, values: Mapping[str, Any], linter: LintConfig
) -> dict[str, Any]:
    """An S1 that would run: fresh, or resumed from its finished attempts (`resume_plan`)."""
    imitate = ImitateConfig.model_validate(values)
    seeds = [p for p in (s1_dir / task / "S1").glob("*") if p.is_dir() and p.name.isdigit()]
    if not seeds:
        return {"action": "run: a fresh S1 (from the demonstration)"}
    try:
        found = resume_plan(seeds[0], imitate)
    except ValueError as error:
        return {"action": f"refused on resume: {one_line(error)}"}
    rows = found["attempts"]
    unhinted = [r for r in rows if not r["hinted"]]
    passes = sum(bool(r["passed"]) for r in unhinted)
    shown = " ".join(("H" if r["hinted"] else "") + ("P" if r["passed"] else "f") for r in rows)
    if found["exit_at"] is not None:
        action = f"resume: the replay exits at attempt {found['exit_at']}, then done criteria"
    else:
        action = f"resume: attempt {found['next']} on, to {imitate.max_attempts}"
    more = f"{len(rows)} finished ({len(rows) - len(unhinted)} hinted), {passes} unhinted passes"
    if (errors := _lint_errors(task, s1_dir, linter, found["memory"])) is not None:
        if errors and found["exit_at"] is not None:
            action += f"; held: its Memory fails the linter ({errors} errors), S1 goes on, fixed"
        elif errors:
            action += f"; fixed on load first ({errors} lint errors)"
        else:
            action += "; lint-clean"
    return {
        "action": f"{action} [{more}]",
        "verdicts": shown,
    }


def _lint_errors(task: TaskName, s1_dir: Path, linter: LintConfig, reads: str | None) -> int | None:
    """How many errors the current linter finds on the Memory a resume goes on with (`reads`, a
    hash in the S1's store), its arm bindings rewritten as on load: more than none run the
    lint-fix pass first (`imitate`'s fix on load); None without one to read (read-only)."""
    if reads is None:
        return None
    for version in FileMemoryStore(s1_dir).folder(task).glob("v[0-9][0-9][0-9]"):
        if json.loads((version / "meta.json").read_bytes())["hash"] == reads:
            doc = read_doc(version)
            return len(lint_errors(lint(unbind_arms(doc).doc, linter)))
    raise ValueError(f"{task}: no stored Memory {reads[:12]} in {s1_dir}")


def _check_definitions(config: PipelineConfig, directory: Path, s1_only: bool = False) -> None:
    """A job that started keeps what it started with (`s1_only`: S1s alone, as a launch of S1s
    runs no variant, whose settings a launch that runs it checks)."""
    changed = []
    for key in run_keys(config):
        if s1_only and key.variant != S1:
            continue
        state = RunState.load(directory / key.task / key.variant, key)
        if state.definition is None:
            continue
        now = json.loads(json.dumps(definition(config, key), default=str))
        was = dict(state.definition)
        if isinstance(was.get("p3"), dict):  # a definition does not count P3's episodes
            was["p3"] = {k: v for k, v in was["p3"].items() if k != "episodes"}
        if "site_configs" not in was:  # nothing recorded to compare
            now.pop("site_configs", None)
        if diff := sorted(k for k in now.keys() | was.keys() if now.get(k) != was.get(k)):
            changed.append(f"{key.name} ({', '.join(diff)})")
    if changed:
        raise ValueError(
            f"the config changes jobs that started: {'; '.join(changed)}; keep their settings, "
            "or start a new pipeline directory"
        )


def _settle(directory: Path, jobs: Sequence[RunKey]) -> dict[str, str]:
    """This launch's jobs that have not ended (stopped, left running or blocked, failed and taken
    up again) start afresh, without an earlier launch's error or failure (its launches, respawns
    and running time kept): `stopped` if it ran before (it waits to run again), else `queued`;
    the errors they had, by job (the launch's record keeps them)."""
    had = {}
    for key in jobs:
        run_dir = job_dir(directory, key)
        state = RunState.load(run_dir, key)
        if state.status == "done":  # a variant P3 reopens
            continue
        if state.error:
            had[key.name] = state.error
        state.status = "stopped" if state.launches > 0 else "queued"
        state.started, state.failure, state.error = None, None, None
        state.save(run_dir)
    return had


def _save(directory: Path, config: PipelineConfig) -> None:
    dumped = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    write_atomic(directory / CONFIG, dumped.encode())


def _record_launch(
    directory: Path, jobs: Sequence[RunKey], retry_failed: bool, note: Mapping[str, Any]
) -> None:
    line = {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
        "source": source_sha(),
        "retry_failed": retry_failed,
        "jobs": [key.name for key in jobs],
        **note,
    }
    with (directory / "launches.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(line) + "\n")


def _job_groups(
    run_dir: Path, records: Path | None = None, modes: Sequence[Mode] | None = None
) -> list[int]:
    """The process groups a job's records (in `records`, default `run_dir`) name: its env
    stack's, its worker's (reaped by now) and its open P3 blocks' (of `modes`; None: all)."""
    records = run_dir if records is None else records
    files = [records / PIDS, *(out / "pids.json" for out in open_blocks(run_dir, modes))]
    return [pid for path in files if path.is_file() for pid in _pids(path)]


def _pids(path: Path) -> list[int]:
    """The process groups a record of names and groups lists."""
    return list(json.loads(path.read_bytes()).values())


def _interpreters(interpreters: Interpreters, where: str) -> None:
    for path in (interpreters.agent, interpreters.sim, interpreters.vision):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"no interpreter {path} (interpreters in {where})")


@contextlib.contextmanager
def _lock(directory: Path) -> Iterator[None]:
    """One launch at a time per directory (the lock goes with the process, however it ends)."""
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another launch is running {directory}") from None
        yield
