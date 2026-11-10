"""The pipeline's jobs, each run by a worker process of its own
(`robotwin_pipeline.worker`, by the job bodies wicl and RoboTwin register for its phase, on
`Host`): a task's S1 in `<root>/<stamp>/<task>/S1/`, a variant's S2
and P3 in `<root>/<stamp>/<task>/<variant>/`. Each directory is a harness run directory
(`ProtocolRunner`: `run.json`, `events.jsonl`, `<task>/{memory,S1,S2}/`) with the pipeline's records
beside: `pipeline.json` (the state), `configs/env.yaml` and `logs/env.log` (its env server),
`.roboscript/pids.json` (the process groups its env stack started, while they run),
`worker.log`, `P3/` and, when it failed, `error.txt`. Frames and images go to the pipeline's blob
store (`BLOBS`), which a GPU's vision service reads.

- `imitate` (S1, one simulator place): the task's demo package on an env server of its own; the
  version S1 exits with is every variant's start. An S1 cut short resumes (`resume`): its
  finished attempts, Memory versions and exit window stay, and only the attempt it cut short,
  set aside as `attempt<n>-aborted<k>`, runs again. With `s1_fallback`, an S1 that runs out of
  attempts yields its best Memory all the same (`ImitateConfig.fallback`, under
  `s1_fallback_rule`; `exit.json`'s `fallback`, `converged: false`, `s1_exit: best_memory`): the
  job writes `fallback.json` naming it (`ops.FALLBACK`), S1 is `done`, and its variants start
  from it; under the `success` rule an S1 without any official success has none and ends
  `not_passed` (`s1_exit: none`), which the summary counts as 0 in P3.
- `practice` (S2, one place): the variant's directory gets the S1's Memory versions and its
  passed `exit.json` (the S1 -> S2 gate); S2 then runs on an env server of its own, freezes the
  Memory it ends with, and resumes at its ledger position (`resume`; a stopped practice returns
  its result). One that stopped at its seed cap without converging goes on to P3 with the Memory
  version of the best first-try record (`p3.unconverged: best_memory`, the default: `pick_s2`
  freezes it in place of the cap's and records `S2/s2_exit.json`), or ends `not_converged` when no
  first try succeeded (it counts 0 in P3); under `p3.unconverged: stop` it ends `not_converged`
  instead of going on to P3 (`p3_gated`).
- `evaluate` (P3, `P3_PLACES` places): the frozen Memory (`P3/manifest.json`) through the launcher
  (`roboscript_robotwin.eval.launch`) on the job's GPU, planning with that GPU's motion service and
  cross-checking with its vision service (in the pipeline's blob store), into
  `P3/<mode>/block<b>/` (the config's modes, or those a launch names: `--p3-mode`, or the
  scheduler: clean, then randomized as a spare unit under `p3.randomized: spare`; a job that
  leaves blocks of the config's modes to run ends `queued`). A block is
  done once its `result.json` has no configuration error and its client exited 0 with the
  official rate written (`_result.txt`) or was cut on purpose (`p3.episodes`) after its count of
  ended episodes; any other is set aside and run again, from the episodes the set-aside run that
  got furthest finished (the seeds checked as before, only the rest evaluated; `P3Run.resume`);
  one whose client exited 0 without the rate, or cut short of its count, fails the job on
  infrastructure (respawned, and resumed from its finished episodes). A launch that fails
  before its client starts is retried, `P3_STARTS` launches in all.

A job's state: `queued` (a variant also between S2 and P3), `running`, `done`, `not_passed` (S1 did
not exit and found no fallback), `not_converged` (a variant whose S2
stopped at its seed cap without converging and may not go on: no first-try success at all under
`p3.unconverged: best_memory`, any under `stop`; no P3 starts, its lane's neither; `gate_variant`
marks one at each launch, picks the best Memory of one that stopped
with no pick recorded, and reopens it once it may go on), `blocked` (a variant whose task's S1 did
not pass), `failed` (the error in a line, the traceback in `error.txt`, and `failure`: `infra` when
infrastructure failed it, which the scheduler respawns and a launch again takes up, else `error`,
retried only on request), `stopped` (by a signal, its worker's death while the pipeline stopped, or
waiting to run again) or `preempted` (a spare unit the scheduler stopped for a job that waited for
its room: it waits for spare room again). A launch again takes up each job that has not ended
(`pending`) afresh, its earlier error cleared (`schedule._settle`). The worker beats `HEARTBEAT`
while it runs.
`definition` is what the job was started with, which a later launch may not change.
"""

import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx
import yaml
from pydantic import BaseModel, SecretStr

from roboscript import rpc
from roboscript.agent.session import ModelClients
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.env import EnvControl
from roboscript.contracts.errors import InfraError, RSError
from roboscript.contracts.ids import RoleName, TaskName
from roboscript.contracts.jobs import EVALUATIONS, EvaluationFactory, JobRequest
from roboscript.contracts.motion import Planner
from roboscript.contracts.provenance import Provenance
from roboscript.envs import Benchmark, control_token
from roboscript.envs.mock import MOCK, MockConfig
from roboscript.kernel import entrypoints
from roboscript.learning import (
    InfraExhausted,
    LearningConfig,
    ProtocolRunner,
    control_client,
)
from roboscript.learning.agent_host import access_of
from roboscript.learning.compose import compose_run, open_sinks
from roboscript.learning.config import BrainSpec
from roboscript.learning.demo import import_blobs, load_package
from roboscript.learning.episode import BrainFactory
from roboscript.learning.method import Declaration, Method, load_method
from roboscript.memory import (
    FileMemoryStore,
    build_manifest,
    object_roles,
    read_doc,
    write_manifest,
)
from roboscript.memory.library import index_from_manifests, write_index
from roboscript.memory.store import FROZEN_FILE
from roboscript.orchestrate.jobs import (
    LAUNCHES,
    PIDS,
    STATE,
    Failure,
    PidRecord,
    RunKey,
    RunState,
)
from roboscript.records import write_atomic
from roboscript.settings import (
    config_ref_path,
    load_config_file,
    repo_root,
    require_source_sha,
    same_source,
    source_sha,
)
from roboscript.stack import ServiceSpec, StackConfig, StackError, up
from roboscript.stack.gpus import GpuProbe, ProcessTable
from roboscript.tools import LocalProvider, ToolProvider
from roboscript.toolsets.privileged import Privileged, mountable
from roboscript_methods.wicl.extension import END_TARGETS
from roboscript_methods.wicl.ledger import Ledger, best_version, version_records
from roboscript_methods.wicl.ops import (
    S1,
    S2_EXIT,
    UNCONVERGED,
    practice_values,
    s2_stop,
    unconverged_note,
)
from roboscript_methods.wicl.practice import (
    LEDGER,
    STOP,
    read_end_targets,
)
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import RoboTwinConfig
from roboscript_robotwin.eval.blocks import LANE_DIRS, MANIFEST, SPARE, redo_p3
from roboscript_robotwin.eval.job import InfraFailure, evaluate_blocks, p3_config
from roboscript_robotwin.eval.launch import VISION, P3Config, scene_names
from roboscript_robotwin.eval.p4 import check_settings, p4_launcher, p4_provenance, read_block
from robotwin_pipeline.config import PipelineConfig, brain_with

log = logging.getLogger(__name__)

BENCHMARKS: Final[dict[str, tuple[Benchmark, type[BaseModel]]]] = {
    "robotwin": (ROBOTWIN, RoboTwinConfig),
    "mock": (MOCK, MockConfig),
}
"""Each benchmark's descriptor and the config model of its env server."""
BLOBS: Final = "blobs"
"""The pipeline's blob store (`<root>/<stamp>/blobs`), which every job, env server and GPU service
shares: a GPU's vision service reads the images its callers put there, by reference."""
PLANNER_TIMEOUT_S: Final = 300.0
"""Planner calls queue on a GPU's one service, behind every job placed there."""

Mode = Literal["clean", "randomized"]
METHOD: Final = "wicl"
"""The method whose jobs the pipeline runs (a `roboscript.methods` entry point)."""
EVALUATION: Final = "robotwin"
"""The evaluation a launch runs of each variant (an `EVALUATIONS` entry point): RoboTwin's
official P3 (`P3Jobs`), whichever env S1 and S2 run on."""


def method() -> Method:
    """The method whose jobs the pipeline runs (`METHOD`, loaded by name as a third-party one
    would be): the launch composes its jobs, a worker runs their bodies."""
    return cast("type[Method]", load_method(METHOD))()


def evaluation_factory() -> EvaluationFactory:
    """The evaluation the pipeline runs of each variant (`EVALUATION`, loaded by name as a
    third-party one would be)."""
    return cast(EvaluationFactory, entrypoints.load(EVALUATIONS, EVALUATION))


def job_dir(directory: Path, key: RunKey) -> Path:
    """A job's records: its variant's (or S1's) directory, or its lane's in it."""
    run_dir = directory / key.task / key.variant
    return run_dir if key.lane is None else run_dir / LANE_DIRS[key.lane]


@dataclass(frozen=True)
class RunEnv:
    """What a job's toolsets get: its task and GPU, the GPU's planner (a client, and `motion`,
    its URL: the service `PipelineConfig.planner` names) and the URLs of its services by name
    (`per_gpu`: the motion service, a vision service), the pipeline's blob store and the model
    its scene toolsets' VLM asks (`PipelineConfig.vlm_model`; None: as their config names it).
    Never the env's control plane."""

    task: TaskName
    gpu: int
    planner: Planner
    motion: str
    services: Mapping[str, str]
    blobs: LocalBlobStore
    vlm_model: str | None = None


@dataclass(frozen=True)
class Wiring:
    """What a pipeline's jobs run with besides its config: the toolsets of a job besides `memory`
    (a list of its own per job, e.g. `run_s2.toolsets`), the command that runs a job in a process
    of its own with this same wiring (`worker`, completed by
    `robotwin_pipeline.worker`'s arguments), the gateway's model client factory, and for
    tests brains and a method in place of the config's (`LearningConfig.method`), and `oracle`: a
    toolset reading the env's ground truth through its control plane, which no real stack mounts,
    and `gpu_memory` and `processes`, how the scheduler reads the GPUs' memory and the process table
    (None: `nvidia-smi`, `ps`)."""

    toolsets: Callable[[RunEnv], Sequence[ToolProvider]]
    worker: Sequence[str]
    models: ModelClients | None = None
    brains: Mapping[RoleName, BrainFactory] | None = None
    method: type[Declaration] | None = None
    oracle: Callable[[EnvControl], ToolProvider] | None = None
    gpu_memory: GpuProbe | None = None
    processes: ProcessTable | None = None


class EndTargetsUnread(RuntimeError):
    """The end targets of the version S2's pick froze could not be read (`refresh_end_targets`:
    the gateway or the perception's VLM calls failed): infrastructure, the job is respawned."""


INFRA: Final = (
    InfraFailure,
    EndTargetsUnread,
    InfraExhausted,
    StackError,
    InfraError,
    httpx.TransportError,
    TimeoutError,
    ConnectionError,
)
"""Errors of infrastructure: a job they fail is respawned; any other error is not."""


def run_keys(config: PipelineConfig) -> list[RunKey]:
    """Every job's key, in start order: each task's S1, then its variants."""
    return [RunKey(task, v) for task in config.tasks for v in (S1, *config.variants)]


S1_KNOBS: Final = ("hinted_max_turns",)
"""The `imitate` parameters a later launch may change for an S1 that started (`definition`):
they bound attempts still to run, not what the finished ones were."""


def definition(config: PipelineConfig, key: RunKey) -> dict[str, Any]:
    """What a job is made of, which a later launch must not change once it started: the
    learning, env and GPU services (the planner, the vision cross-check), the protocol
    parameters but S1's `S1_KNOBS`, S1's demo package, the scene VLM's model when the run names
    one (`vlm_model`), S1's fallback rule and what an unconverged S2 does (`p3.unconverged`) when
    not their defaults, and P3's settings but its modes (C2R is
    a later launch's, `--p3-mode`), and its
    episodes per block (a later launch may evaluate
    more: a block cut shorter runs again, `block_done`), its shards and how they merge, and its
    profile (each block
    records the one it runs with, `block_profile`, and a resumed block keeps it), with the
    launcher config's content
    hash, and the content hashes of the machine's toolset and service configs a job reads as it
    starts (`site_configs`)."""
    per_gpu = [spec.model_dump(mode="json") for spec in config.per_gpu]
    made: dict[str, Any] = {
        "benchmark": config.benchmark,
        "env": config.env,
        "learning": config.learning,
        "per_gpu": per_gpu,
        "site_configs": site_configs(config),
    }
    # each only when it is not its default, so that a job on the defaults keeps its hash
    if config.vlm_model is not None:
        made["vlm_model"] = config.vlm_model
    if key.variant == S1:
        imitate = config.protocol("imitate", key.task)
        imitate = {k: v for k, v in imitate.items() if k not in S1_KNOBS}
        if config.s1_fallback_rule != "success":
            made["s1_fallback_rule"] = config.s1_fallback_rule
        return {**made, "demo": str(config.demo_of(key.task)), "imitate": imitate}
    practice = practice_values(config, key)
    later = {"randomized", "episodes", "episodes_randomized", "shards", "strict_shards"}
    later |= {"spare_parallel"}
    later |= {"unconverged"}
    p3 = config.p3.model_dump(mode="json", exclude=later)
    launcher = config.p3.config
    p3["config_sha256"] = _sha256(launcher) if config.p3.blocks and launcher.is_file() else None
    if config.p3.unconverged != "best_memory":
        made["s2_unconverged"] = config.p3.unconverged
    return {**made, "practice": practice, "p3": p3}


def job_provenance(
    made: dict[str, Any], key: RunKey, profile: dict[str, Any] | None = None
) -> Provenance:
    """What a job's episodes record as their code and behaviour config: the code's
    revision, the variant (`S1` for the task's S1) and the sha256 of what the job is made of
    (`definition`, as its first launch recorded it); a P3 block under a profile other than the
    current one (`block_profile`) is `<variant>+<profile>`, hashed with the profile's overrides."""
    name, content = key.variant, made
    if profile is not None and profile.get("profile", "current") != "current":
        name = f"{key.variant}+{profile['profile']}"
        content = {"definition": made, "overrides": profile.get("overrides") or {}}
    digest = hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
    return Provenance(source=source_sha(), config=name, config_hash=digest)


def demo_host(practice: Mapping[str, Any]) -> dict[str, Any]:
    """The Agent host's demonstration settings for a variant's P3: those its S2 practised with
    (its demonstration's end targets too, `practice.end_targets`), and its latch refusals when the
    practice sets them (`practice.latch_finish`, one switch for S2 and P3; unset, both are on by
    default)."""
    level = practice.get("demo_context", "none")
    shown = {} if level == "none" else {"demo_context": level, "demo": practice["demo"]}
    goal = {"done_goal": practice["done_goal"]} if "done_goal" in practice else {}
    end = {"end_targets": practice["end_targets"]} if practice.get("end_targets") else {}
    latch = {}
    if "latch_finish" in practice:
        latch = {"episodes": {"latch_finish": practice["latch_finish"]}}
    return {**shown, **goal, **end, **latch}


def p3_demo(config: PipelineConfig, key: RunKey, run_dir: Path) -> dict[str, Any]:
    """A variant's P3 demonstration settings (`demo_host`), with the demonstration's end targets
    its S2 read (`PracticeConfig.read_end_targets`, `S2/end_targets.json`) unless the practice
    names a file of its own; none without the package's end state (`done_goal`)."""
    demo = demo_host(practice_values(config, key))
    read = run_dir / key.task / "S2" / END_TARGETS
    if "end_targets" not in demo and "done_goal" in demo and read.is_file():
        demo["end_targets"] = str(read)
    return demo


def run_models(config: PipelineConfig, launcher: P3Config) -> P3Config:
    """`launcher` (P3's) with the run's models (`PipelineConfig.model`): its Agent host brain
    takes `model` and `context_windows` as `learning.brains` do (`config.brain_with`), and with
    `model` set also the S2 Actor brain's `tool_choice` (so that P3's Actor asks the model as S2's
    did, without a P3 file of the run's own); its scene toolsets' VLM `vlm_model`; unchanged when
    the config names none."""
    if config.model is None and config.vlm_model is None and not config.context_windows:
        return launcher
    values = launcher.model_dump(mode="json")
    host = dict(values["host"])
    if "brain" in host:
        host["brain"] = brain_with(host["brain"], config.model, config.context_windows)
        actor = (config.learning.get("brains") or {}).get("actor") or {}
        choice = (actor.get("config") or {}).get("tool_choice")
        if config.model is not None and choice is not None and takes_tool_choice(host["brain"]):
            host["brain"]["config"] = {**host["brain"]["config"], "tool_choice": choice}
    if (vlm := config.vlm_model) is not None:
        scene = scene_names(spec.name for spec in launcher.toolsets())
        host["toolsets"] = [
            {**spec, "config": scene_with(spec.get("config") or {}, vlm)}
            if spec["name"] in scene
            else spec
            for spec in host.get("toolsets", [])
        ]
    return P3Config.model_validate({**values, "host": host})


def takes_tool_choice(entry: Mapping[str, Any]) -> bool:
    """Whether a brain entry's brain has a `tool_choice` setting."""
    name = entry.get("name", BrainSpec.model_fields["name"].default)
    return "tool_choice" in entrypoints.load("roboscript.brains", name).Config.model_fields


def scene_with(values: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Scene toolset config values whose VLM asks `model`."""
    return {**values, "vlm": {**(values.get("vlm") or {}), "model": model}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SITE_TOOLSETS: Final = ("manipulation", "scene")
"""The toolset configs under configs/ a job reads as it starts (`run_s2.toolsets`)."""


def site_configs(config: PipelineConfig) -> dict[str, str | None]:
    """The sha256 of each machine config a job reads at its start, outside the pipeline's
    config: the toolsets' (`SITE_TOOLSETS`, configs/<name>.yaml; None: absent, the defaults) and
    each GPU service's `--config` (`per_gpu`: the motion and vision services'), by name, so that
    an edit to one changes the job's `definition` and a resume refuses it."""
    found: dict[str, str | None] = {}
    for name in SITE_TOOLSETS:
        path = repo_root() / "configs" / f"{name}.yaml"
        found[name] = _sha256(path) if path.is_file() else None
    for spec in config.per_gpu:
        if "--config" in spec.args:
            ref = spec.args[spec.args.index("--config") + 1]
            try:
                path = config_ref_path(ref)
            except FileNotFoundError:
                found[spec.name] = None
                continue
            path = path if path.is_absolute() else repo_root() / path  # services run from there
            found[spec.name] = _sha256(path) if path.is_file() else None
    return found


def env_config(config: PipelineConfig, task: TaskName, blob_dir: Path) -> BaseModel:
    """A job's env server config: the benchmark's default, the config's `env`, the task."""
    benchmark, model = BENCHMARKS[config.benchmark]
    values = {**benchmark.env_config, **config.env, "task": task, "blob_dir": blob_dir}
    return model.model_validate(values)


def failure_of(error: BaseException) -> Failure:
    """Infrastructure (`INFRA`, or a service's unexpected failure, `internal`) or not."""
    internal = isinstance(error, RSError) and error.code == "internal"
    return "infra" if isinstance(error, INFRA) or internal else "error"


def check_dirty(label: str, allow: bool = False) -> dict[str, Any]:
    """A launch runs a commit: a source label of a dirty tree (`-dirty`) is refused unless `allow`
    (`--allow-dirty`), which the launch records (what this returns
    joins its line in `launches.jsonl`)."""
    if not label.endswith("-dirty"):
        return {}
    if not allow:
        raise ValueError(
            f"the code is a dirty tree ({label}): commit and sync it, or pass --allow-dirty, "
            "which the launch records"
        )
    return {"dirty_source": label}


def check_source(directory: Path) -> None:
    """The code version a launch in `directory` runs, checked: the code must name one
    (`source_sha`: `RS_SOURCE_SHA` or the checkout's source record), so that every record says what
    ran; a resume runs the version of the directory's first launch (`launches.jsonl`)."""
    current = require_source_sha()
    path = directory / LAUNCHES
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if not lines:
        return
    first = json.loads(lines[0]).get("source")
    if first is not None and same_source(str(first), current):
        return
    raise ValueError(
        f"{directory} was first launched with code {first}, this is {current}: a resume runs "
        "the code the run started with (check out that version)"
    )


S2Exit = Literal["converged", "best_memory", "none"]
"""How a variant's S2 ended for P3 (`ops.S2_EXIT`)."""


def s2_exit(config: PipelineConfig, run_dir: Path, task: TaskName) -> dict[str, Any] | None:
    """How a variant's S2 ended for P3: its `S2_EXIT` record, else what `pick_s2` would record
    now (nothing written; a dry run and the report read it too); None while S2 has not stopped,
    and for a stop at the cap under `p3.unconverged` `stop` or `last`."""
    folder = run_dir / task / "S2"
    if (path := folder / S2_EXIT).is_file():
        return cast(dict[str, Any], json.loads(path.read_bytes()))
    if not (folder / STOP).is_file():
        return None
    stop = json.loads((folder / STOP).read_bytes())
    if stop["reason"] == "converged":
        return {"s2_exit": "converged", "k": stop["k"], "version": stop["version"]}
    if stop["reason"] != UNCONVERGED or config.p3.unconverged != "best_memory":
        return None
    rows = Ledger(folder / LEDGER).rows()
    best = best_version(rows)
    records = {f"v{v:03d}": list(kn) for v, kn in sorted(version_records(rows).items())}
    return {
        "s2_exit": "best_memory" if best is not None else "none",
        "rule": "best_memory",
        "k": stop["k"],
        "version": None if best is None else best["version"],
        "picked": best,
        "records": records,
        "instead_of": {"version": stop["version"], "hash": stop["hash"]},
    }


def pick_s2(config: PipelineConfig, directory: Path, key: RunKey) -> dict[str, Any] | None:
    """Record how a variant's S2 ended for P3 (`S2_EXIT`, once; `s2_exit`). At the seed cap
    under `p3.unconverged: best_memory`, the version with the
    best first-try record (`wicl.ledger.best_version`) is frozen in place of the cap's
    (`FileMemoryStore.refreeze`, the cap's record kept aside); when that changes the frozen
    version, P3 blocks done before (they evaluated the cap's) are moved aside to run again
    (`redo_p3`, `redone_p3`) and the P3 manifest is written afresh. From the recorded ledger
    alone: nothing runs again. The record; None as `s2_exit`."""
    run_dir = directory / key.task / key.variant
    folder = run_dir / key.task / "S2"
    if (path := folder / S2_EXIT).is_file():
        return cast(dict[str, Any], json.loads(path.read_bytes()))
    record = s2_exit(config, run_dir, key.task)
    if record is None:
        return None
    if record["s2_exit"] == "best_memory":
        store = FileMemoryStore(run_dir)
        old = store.folder(key.task) / FROZEN_FILE
        before = json.loads(old.read_bytes()) if old.is_file() else {}
        picked = {k: record[k] for k in ("picked", "instead_of", "records")}
        frozen = store.refreeze(
            key.task,
            record["version"],
            reason=UNCONVERGED,
            summary={**before.get("summary", {}), "s2_exit": "best_memory", **picked},
            vision=before.get("vision"),
        )
        record["hash"] = frozen.hash
        # against the cap's version (stop.json, never rewritten): a pick again after a crash
        # before `S2_EXIT` was written discards the cap's P3 all the same
        if frozen.hash != record["instead_of"]["hash"]:
            (run_dir / MANIFEST).unlink(missing_ok=True)
            moved = {
                mode: redo_p3(directory, key.task, key.variant, mode)
                for mode in ("clean", "randomized")
            }
            record["redone_p3"] = {mode: names for mode, names in moved.items() if names}
        log.warning(
            "%s: S2 stopped at its seed cap (k=%s): P3 evaluates v%03d (%d of %d first tries, "
            "Wilson lower %.3f) instead of v%03d",
            key.name,
            record["k"],
            record["version"],
            record["picked"]["successes"],
            record["picked"]["n"],
            record["picked"]["lower"],
            record["instead_of"]["version"],
        )
    elif record["s2_exit"] == "none":
        log.warning("%s: S2 stopped at its seed cap with no first-try success: no P3", key.name)
    record["at"] = time.time()
    write_atomic(path, (json.dumps(record) + "\n").encode())
    return record


END_ASIDE: Final = "end_targets.before"
"""`refresh_end_targets` keeps a replaced `S2/end_targets.json` as `end_targets.before<n>.json`."""


def end_targets_due(config: PipelineConfig, directory: Path, key: RunKey) -> Path | None:
    """The demo package whose end `S2/end_targets.json` must be read again before P3 runs
    (`refresh_end_targets`): S2 stopped at its cap and `pick_s2` froze another version than the
    cap's, whose roles the file (read for the cap's, or not at all) does not name; None when P3
    reads no end targets (no `done_goal`, which `p3_demo` reads them with, a practice file of its
    own, or `read_end_targets` off), the package has no frames to read them on (as
    `practice.ensure_end`: P3 goes on without) or the file names the picked version's roles.
    From the files alone."""
    run_dir = directory / key.task / key.variant
    path = run_dir / key.task / "S2" / S2_EXIT
    if not path.is_file():
        return None
    record = json.loads(path.read_bytes())
    if record["s2_exit"] != "best_memory" or record["version"] == record["instead_of"]["version"]:
        return None
    values = practice_values(config, key)
    package = values.get("done_goal")
    if package is None or values.get("end_targets") or not values.get("read_end_targets", True):
        return None
    if not load_package(Path(package)).frames:
        return None
    doc = read_doc(FileMemoryStore(run_dir).folder(key.task) / f"v{record['version']:03d}")
    read = run_dir / key.task / "S2" / END_TARGETS
    if read.is_file() and json.loads(read.read_bytes()).get("roles") == sorted(object_roles(doc)):
        return None
    return Path(package)


async def refresh_end_targets(
    config: PipelineConfig, runner: ProtocolRunner, directory: Path, key: RunKey
) -> None:
    """Read the demonstration's end targets again for the version `pick_s2` froze when their
    roles differ from the file's (`end_targets_due`), as a practice reads them for the version it
    freezes (`practice.read_end_targets`, which scripts/read_end_targets.py calls too), so that P3
    never runs with the cap version's targets, of which only some carry over. The file it
    replaces is kept as `end_targets.before<n>.json`; a reading that fails puts it back and fails
    the job on infrastructure (`EndTargetsUnread`: respawned; P3 does not start), unless no
    perception toolset is mounted (a configuration error)."""
    package = end_targets_due(config, directory, key)
    if package is None:
        return
    run_dir = directory / key.task / key.variant
    folder = run_dir / key.task / "S2"
    picked = json.loads((folder / S2_EXIT).read_bytes())["version"]
    doc = read_doc(FileMemoryStore(run_dir).folder(key.task) / f"v{picked:03d}")
    read, aside = folder / END_TARGETS, None
    if read.is_file():
        n = 1
        while (aside := folder / f"{END_ASIDE}{n}.json").exists():
            n += 1
        read.rename(aside)
    values = practice_values(config, key)
    await import_blobs(package, runner.blobs)
    targets, why = await read_end_targets(
        runner.driver,
        folder,
        load_package(package),
        doc,
        attempts=values.get("end_targets_attempts", 2),
        timeout_s=values.get("end_targets_timeout_s", 2700.0),
        overwrite=True,
    )
    if targets is None:
        if aside is not None:
            aside.rename(read)
        unread = RuntimeError if why == "no perception toolset mounted" else EndTargetsUnread
        raise unread(
            f"{key.name}: the demonstration's end targets for S2's best Memory v{picked:03d} "
            f"were not read ({why}): P3 does not start with the cap version's"
        )
    log.warning("%s: %d end targets read for v%03d", key.name, len(targets), picked)


def p3_gated(config: PipelineConfig, run_dir: Path, task: TaskName) -> bool:
    """Whether a variant may not go on to P3: its S2 stopped at the seed cap without converging
    and either no counted first try succeeded (`p3.unconverged: best_memory`, `s2_exit` none) or
    any stop at the cap ends it (`stop`). Read from the files alone (a dry run and the report read
    it too)."""
    stop = s2_stop(run_dir, task)
    if stop is None or stop["reason"] != UNCONVERGED:
        return False
    found = s2_exit(config, run_dir, task)
    if found is not None and found["s2_exit"] in ("best_memory", "none"):
        return found["s2_exit"] == "none"
    return config.p3.unconverged == "stop"


def p3_held(config: PipelineConfig, run_dir: Path, task: TaskName) -> bool:
    """Whether a variant's P3 waits (`P3Jobs`' `gated`): it may not go on (`p3_gated`), or the end
    targets of the version S2's pick froze are still to be read in its practice phase
    (`end_targets_due`)."""
    if p3_gated(config, run_dir, task):
        return True
    key = RunKey(task, run_dir.name)
    return end_targets_due(config, run_dir.parent.parent, key) is not None


def gate_variant(config: PipelineConfig, directory: Path, key: RunKey) -> str | None:
    """At a launch, bring a variant's state (and its lane's, `SPARE`) in line with
    `p3.unconverged`. One whose S2 stopped at the cap with no pick recorded gets its pick now
    (`pick_s2`; a P3 it ran on another version is moved aside and queued again, `picked`). One
    that may not go on (`p3_gated`) ends `not_converged` (whatever P3 it had started or finished:
    those blocks stay on disk), and its lane too unless it ended; a `not_converged` one that may
    go on is `queued` at P3 again (its lane too). "stopped before P3", "picked v<n>" or
    "reopened"; None when nothing changed."""
    run_dir = directory / key.task / key.variant
    state = RunState.load(run_dir, key)
    lane = RunKey(key.task, key.variant, SPARE)
    lane_dir = job_dir(directory, lane)
    lane_state = RunState.load(lane_dir, lane) if (lane_dir / STATE).is_file() else None
    had = (run_dir / key.task / "S2" / S2_EXIT).is_file()
    picked = None if had or state.status == "running" else pick_s2(config, directory, key)
    cap = ((picked or {}).get("instead_of") or {}).get("version")
    if picked is not None and picked["s2_exit"] == "best_memory" and picked["version"] != cap:
        # its P3 (if any ran) evaluated the cap's version: the blocks moved aside run again
        for job_state, where in ((state, run_dir), (lane_state, lane_dir)):
            if job_state is not None and job_state.status == "done":
                job_state.stage = "p3"
                job_state.end(where, "queued")
        log.warning("%s: P3 runs again on v%03d", key.name, picked["version"])
        if state.status != "not_converged":
            return f"picked v{picked['version']:03d}"
    if p3_gated(config, run_dir, key.task):
        if state.status == "not_converged":
            return None
        note = unconverged_note(run_dir, key.task)
        state.failure = None
        state.end(run_dir, "not_converged", note)
        if lane_state is not None and lane_state.status not in ("done", "not_converged"):
            lane_state.failure = None
            lane_state.end(lane_dir, "not_converged", note)
        log.warning("%s: %s", key.name, note)
        return "stopped before P3"
    if state.status != "not_converged":
        return None
    state.failure = None
    state.stage = "p3"
    state.end(run_dir, "queued")
    if lane_state is not None and lane_state.status == "not_converged":
        lane_state.end(lane_dir, "queued")
    if picked is not None and picked["s2_exit"] == "best_memory":
        log.warning("%s: S2's best Memory v%03d: its P3 goes on", key.name, picked["version"])
        return f"picked v{picked['version']:03d}"
    log.warning("%s: its S2 may go on to P3 (p3.unconverged %s)", key.name, config.p3.unconverged)
    return "reopened"


class Host:
    """The pipeline as a job's worker runs it (`roboscript.orchestrate.worker.WorkerHost`, and
    what the job kinds need: wicl's `LearnHost`, RoboTwin's `EvalHost`), under `config`, with
    `wiring`, in the pipeline's `directory`: the job's records (`job_dir`), its `definition`, how an
    error fails it (`failure_of`), a learning job's runner on an env server of its own (`serve`,
    `_served`, `_runner`), S2's best-Memory pick and its end targets after practice
    (`after_practice`: `pick_s2`, `refresh_end_targets`), the P3 gate (`p3_gated`) and P3
    (`evaluate`, `_evaluate`)."""

    def __init__(self, config: PipelineConfig, wiring: Wiring, directory: Path) -> None:
        self.config, self.wiring, self.directory = config, wiring, directory
        self.heartbeat_s = config.watchdog_s / 10
        self.units = list(config.variants)
        self.p3 = config.p3
        # the stage a unit's evaluation records as it starts: its first job's (RoboTwin's one)
        first = next(iter(evaluation_factory().kinds.values()))
        self.evaluation_stage = first.stage if config.p3.blocks else None

    def job_dir(self, key: RunKey) -> Path:
        return job_dir(self.directory, key)

    def definition(self, key: RunKey) -> dict[str, Any]:
        return definition(self.config, key)

    def failure(self, error: BaseException) -> Failure:
        return failure_of(error)

    @asynccontextmanager
    async def serve(
        self, request: JobRequest, made: dict[str, Any] | None
    ) -> AsyncIterator[ProtocolRunner]:
        """A learning job's runner on an env server of its own; `made`: what the job is made of
        (`definition`), which its episodes' provenance names."""
        key = RunKey.of(request)
        run_dir = key.unit_dir(self.directory)
        provenance = None if made is None else job_provenance(made, key)
        blobs = self.directory / BLOBS
        async with httpx.AsyncClient(trust_env=False) as http:
            served = _served(
                self.config,
                key,
                run_dir,
                request.gpu,
                request.services,
                blobs,
                control_token(),
                http,
            )
            async with (
                served as (env, control),
                _runner(self.config, self.wiring, run_dir, env, control, provenance) as runner,
            ):
                yield runner

    async def after_practice(self, runner: ProtocolRunner, key: RunKey) -> None:
        pick_s2(self.config, self.directory, key)  # at the cap: the best first-try version, frozen
        await refresh_end_targets(
            self.config, runner, self.directory, key
        )  # its roles' end targets

    def gated(self, run_dir: Path, task: TaskName) -> bool:
        return p3_gated(self.config, run_dir, task)

    async def evaluate(
        self, request: JobRequest, modes: Sequence[Mode], made: dict[str, Any] | None
    ) -> None:
        key = RunKey.of(request)
        run_dir = key.unit_dir(self.directory)
        blobs = self.directory / BLOBS
        await _evaluate(
            self.config, key, run_dir, request.gpu, request.services, blobs, modes, made
        )


@asynccontextmanager
async def _served(
    config: PipelineConfig,
    key: RunKey,
    run_dir: Path,
    gpu: int,
    services: Mapping[str, str],
    blobs: Path,
    token: SecretStr,
    http: httpx.AsyncClient,
) -> AsyncIterator[tuple[RunEnv, EnvControl]]:
    """The job's env server on `gpu` (`configs/env.yaml`, `logs/env.log`), with the pipeline's
    blob store `blobs`, recorded while it runs."""
    benchmark, _ = BENCHMARKS[config.benchmark]
    path = run_dir / "configs" / "env.yaml"
    dumped = env_config(config, key.task, blobs).model_dump(mode="json")
    write_atomic(path, yaml.safe_dump(dumped, sort_keys=False).encode())
    service = ServiceSpec(
        name="env",
        module=benchmark.env_server,
        interpreter=benchmark.interpreter,
        args=["--config", str(path)],
        gpu=gpu,
        ready_timeout_s=config.env_ready_timeout_s,
    )
    stack = StackConfig(
        interpreters=config.interpreters,
        sim_pythonpath=config.sim_pythonpath,
        run_dir=run_dir,
        blob_dir=blobs,
        services=[service],
    )
    record = PidRecord(run_dir / PIDS)
    try:
        async with up(stack, on_start=record.add) as running:
            learning = LearningConfig.model_validate({**config.learning, "run_dir": run_dir})
            control = control_client(running.urls["env"]["env.control"], token, learning, http)
            planner = rpc.client(
                Planner,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
                services[config.planner],
                token=None,
                http=http,
                timeout=PLANNER_TIMEOUT_S,
            )
            motion = services[config.planner]
            blobs = LocalBlobStore(running.blob_dir)
            env = RunEnv(key.task, gpu, planner, motion, services, blobs, config.vlm_model)
            yield env, control
    finally:
        record.clear()  # stopped with the stack


@asynccontextmanager
async def _runner(
    config: PipelineConfig,
    wiring: Wiring,
    run_dir: Path,
    env: RunEnv,
    control: EnvControl,
    provenance: Provenance | None = None,
) -> AsyncIterator[ProtocolRunner]:
    """The job's runner, started, with the run's plugins (`config.plugins`, over the built-in
    guards; its event sinks open in `run_dir`)."""
    benchmark = BENCHMARKS[config.benchmark][0]
    if missing := mountable(benchmark.capabilities):  # the privileged toolset needs truth
        raise ValueError(f"{benchmark.name}: the privileged toolset cannot mount: {missing}")
    providers = [*wiring.toolsets(env), LocalProvider(Privileged(control))]  # tier P: S1 alone
    if wiring.oracle is not None:
        providers.append(wiring.oracle(control))
    async with compose_run(config.plugins) as composed:
        with open_sinks(composed.sinks, run_dir) as sinks:
            learning = {**config.learning, "run_dir": run_dir, "provenance": provenance}
            runner = ProtocolRunner(
                LearningConfig.model_validate(learning),
                providers=providers,
                blobs=env.blobs,
                controls={env.task: control},
                embodiment=benchmark.embodiment,
                brains=wiring.brains,
                models=wiring.models,
                planner=env.planner,
                method=wiring.method,
                recipe=config.recipe(),
                pipeline=composed.pipeline,
                sinks=sinks,
                offered=benchmark.capabilities,
            )
            async with runner:
                yield runner


async def _evaluate(
    config: PipelineConfig,
    key: RunKey,
    run_dir: Path,
    gpu: int,
    services: Mapping[str, str],
    blobs: Path,
    modes: Sequence[Mode],
    made: dict[str, Any] | None = None,
) -> None:
    """P3 of the frozen Memory (`roboscript_robotwin.eval.job.evaluate_blocks`), through the
    launcher config of the job (`p3_config`: its GPU's services, the variant's demonstration
    settings, `p3_demo`), its episodes' provenance the job's under each block's profile
    (`job_provenance`). A unit a P4 run prepared (`roboscript_robotwin.eval.p4`: `P3/p4.json`)
    evaluates its arm's block: the arm's Agent host method and document, its episodes' provenance
    the arm's (`p4_provenance`); every other unit runs P3 as it is."""
    manifest = run_dir / MANIFEST
    if not manifest.is_file():
        write_manifest(manifest, await build_manifest(FileMemoryStore(run_dir), [key.task]))
    base = run_models(config, load_config_file(config.p3.config, P3Config))
    motion, vision = services[config.planner], services.get(VISION)
    demo = p3_demo(config, key, run_dir)
    launcher = p3_config(base, run_dir, gpu, motion, vision, blobs, demo)
    launcher = with_library_index(config, key, run_dir, launcher)
    block = read_block(run_dir)  # a unit a P4 run prepared: its arm's block
    if block is not None:
        check_settings(config.p3, block)
        launcher = p4_launcher(launcher, run_dir, block)

    def provenance(profile: dict[str, Any]) -> Provenance | None:
        made_by = None if made is None else job_provenance(made, key, profile)
        return made_by if block is None else p4_provenance(made_by, block)

    pins = None if block is None else block.pins  # shared by the block's arms
    await evaluate_blocks(config.p3, key, run_dir, manifest, launcher, modes, provenance, pins)


LIBRARY_INDEX: Final = "P3/library.json"
"""A unit's P3 in library mode: the library index it was given (every task's frozen Memory)."""

NO_MEMORY: Final = frozenset({"not_passed", "not_converged", "blocked"})
"""A unit that ended so has no frozen Memory, and the library goes without it."""


def with_library_index(
    config: PipelineConfig, key: RunKey, run_dir: Path, launcher: P3Config
) -> P3Config:
    """In library mode with no index named (`host.memory_library.index`), `launcher` with the
    library of the run's variant: every task's frozen P3 Memory (`P3/manifest.json`, each unit its
    own store), written to the unit's `LIBRARY_INDEX`, never one task's manifest alone. A task
    still to freeze one fails the job: P3 in library mode runs once every task's S2 has ended
    (launch it again then, `--p3-mode`), so that every task's episodes see the same library."""
    host = launcher.host
    named = (host.get("memory_library") or {}).get("index")
    if access_of(host) != "library" or named is not None:
        return launcher
    stamp = run_dir.parent.parent
    pairs, waiting = [], []
    for task in config.tasks:
        unit = RunKey(TaskName(task), key.variant).unit_dir(stamp)
        manifest = unit / MANIFEST
        if manifest.is_file():
            pairs.append((manifest, unit))
        elif RunState.load(unit, key).status not in NO_MEMORY:
            waiting.append(task)
    if waiting:
        raise ValueError(
            f"library mode: P3 needs every task's frozen Memory, and {waiting} have none yet; "
            "launch P3 again once their S2 has ended (or name host.memory_library.index)"
        )
    path = run_dir / LIBRARY_INDEX
    write_index(path, index_from_manifests(pairs))
    library = {**(host.get("memory_library") or {}), "index": str(path)}
    values = launcher.model_dump(mode="json")
    return P3Config.model_validate({**values, "host": {**host, "memory_library": library}})


def one_line(error: BaseException) -> str:
    """The error in one line: its type and the first line of its message."""
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0]}" if text else type(error).__name__
