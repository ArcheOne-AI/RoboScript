"""One official RoboTwin evaluation of RoboScript (docs/concepts/learning.md): a task x seed
block x config, what RoboTwin's `policy/*/eval_double_env.sh` scripts do, in Python.

```python
run = P3Run(task="beat_block_hammer", seed=0, manifest=Path("runs/p3/manifest.json"),
            out_dir=Path("runs/p3/beat_block_hammer/clean/seed0"))
result = anyio.run(evaluate, run, load_config_file(path, P3Config))
```

1. Check the install again (`roboscript_robotwin.eval.install.check_installed`: the links, the
   commit, the official scripts, no uncommitted change) and read the Memory manifest, whose id is
   the `ckpt_setting`. The run directory must be new or empty.
2. Start the Agent host (`roboscript.learning.agent_host`, agent venv) under the stack
   supervisor, with a bearer token issued for the run in a file only its owner can read, which
   the host and the policy server read (never the environment or a command line). The host's
   `manipulation` toolset plans with a shared motion service (`motion.url`) or with one the
   launcher starts for the run before the host (`motion.config`: `roboscript.motion`, simulator
   env, the run's GPU; the toolset reads its ready file), and keeps targets above the table
   fallback of the run's scene config (`table_top_z`) until perception estimates the table. The
   scene toolsets (perception, geometry, worldstate) filter the robot out with the same motion
   service and cross-check with a vision service: a shared one (`vision.url`, whose blob store,
   `vision.blob_dir`, the run's processes then use, since images reach the service there by
   reference), or one the launcher starts for the run (`vision.config`, vision env), before the
   host, on a port chosen for the run, on `vision_gpu`, else the least loaded of `gpus`.
3. Start RoboTwin's `script/policy_model_server.py` on a free port and wait for its own "Model
   server started on localhost:<port>" line, then `script/eval_policy_client.py`: both in the
   simulator env, from `ROBOTWIN_ROOT`, each in its own process group, with `instruction_type:
   unseen` from `deploy_policy.yml`. The server binds its port seconds after the probe found it
   free, so another launch may take it in between: a bind that fails because the address is in
   use stops that server and starts another on a new port, `PORT_ATTEMPTS` times in all; any
   other traceback, or the server's exit, fails the run at once. Only our server's own line counts,
   so a client never talks to another launch's server.
   With `shards` = K > 1 there are K servers (`--shard k`, so each relays to its own host slot)
   and K clients, started one after another (`roboscript_robotwin.eval.shard_client`: the official
   client's `main` with a loop that runs the expert check on every seed and evaluates the episodes
   whose index is k modulo K; every shard so finds the seeds one client would, in its order). The
   expert check of a seed does not repeat on some tasks, so the shards pin its verdicts in
   `shards/verdicts.jsonl` (the first shard to check a seed decides; the others follow and skip
   the seeds of other shards), which a resumed run starts from (a copy). Runs given `pins`
   share one verdicts file instead (`P3Run.pins`: P4's arms on one block).
4. Wait for the client (a run of `max_episodes` runs the shard client alone, which stops after
   exactly that many episodes; the run stops it early once the host has failed the run on a
   configuration error, its `config_error.json`: every episode would end alike), stop the server and
   the host, move the official result directory into the run directory, remove the directories under
   `eval_result/<task>` that our clients made and that are empty now, and write `result.json` (with
   the configuration error, after which `evaluate` raises). A finished run never fails there: if the
   lock does not come in time, or moving fails, the official directory stays where it is, which
   `result.json` names. Shards: each client's result directory goes to `official/shard<k>`, and
   `merge_shards` checks that the shards found the same solvable seeds and writes
   `official/shards.json` (the episodes by index, seed, shard and success) and the block's
   `official/_result.txt`, which the launcher writes in the official format (the successes of all
   shards over the official 100, stamped as shard 0's).

`P3Run.resume` names an interrupted run of the same block: the episodes its official clients
finished (`finished`) are carried (`carried.json`), every seed still gets the expert check, and the
clients (`shard_client --carry`, one shard alone too) evaluate only the others; the carried
episodes' host records and videos are brought over (`carry_records`), so the block's result is an
uninterrupted block's. An episode infrastructure lost (`lost`: the relay said so; the official
client's log shows our policy's `INFRA_LOST` line) has no task outcome: the resumed run evaluates
it again on its seed and index (`carried.json`'s `rerun`) and carries every finished one, before
and after it; its lost attempt's host records come along apart (`<slot>.lost-from-<run>`) as
history, not counted, and `result.json` holds one final outcome per index (`outcomes`). A resume
refuses an episode lost `MAX_LOSSES` runs in a row (`LostRepeatedly`), and a block with a lost
episode is not done until a resumed run evaluates it (`robotwin_pipeline.run.block_done`). With
more than one shard, the shards' splits and uneven stops still cut what is carried (`_agreed`); a
block's final rerun uses one shard. `close` instead ends a block from a set-aside run that finished
enough episodes, without starting anything: its result is a block cut on purpose with every episode
that run finished.

A task whose Memory was learned with the vision service's cross-check (the manifest's `vision`)
runs only with a vision service and the perception toolset, so that P3 identifies objects as
learning did.

The client names its result directory `eval_result/<task>/RoboScript/<config>/<manifest
id>/<start time to the second>`, and the runs of one task share the parents: so they take turns
(a lock per checkout and task, `~/.cache/roboscript/locks/`; runs of other tasks go on) from
starting a client until its directory exists and that second has passed, and to remove
directories. Which directories under `eval_result/<task>` our clients made (that one included, if
it was absent) is recorded beside the lock, for all runs of the task: whichever run finds one
empty removes it; a directory that existed before stays, and so does `eval_result` itself, which
the clients of every task share.

SIGTERM, and SIGHUP unless ignored (`nohup`), stops a run the same way (processes stopped, result
directory moved out) and then raises `roboscript.stack.Stopped`.

Run directory: `agent_host.yaml`, `pids.json` (the process groups, as each starts), `ended.json`
(with `release_shards`: the shard clients that ended after their episodes, by name, their process
groups), `logs/` (`motion.log`, `vision.log`, `agent-host.log`, `policy_server.log`, `client.log`;
with shards `policy_server.shard<k>.log`, `client.shard<k>.log`), `shards/` (the shard clients'
records, `shard<k>.jsonl`), `host/` (the host's records: `events.jsonl`, `<slot>/ep<NNN>/`),
`blobs/` (unless a shared vision service's store holds them), `official/` (the official result
directory: `_result.txt`, the videos; with shards `shard<k>/` each, the merged `_result.txt` and
`shards.json`) and `result.json`.
"""

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import anyio
import anyio.abc
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    FiniteFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    TypeAdapter,
    model_validator,
)
from typing_extensions import Self

from roboscript.agent.host import CONFIG_ERROR, infra_lost
from roboscript.contracts.ids import ManifestId, TaskName
from roboscript.contracts.provenance import Provenance
from roboscript.kernel import compose, entrypoints
from roboscript.learning.agent_host import AgentHostConfig, ToolsetSpec
from roboscript.learning.config import BrainSpec
from roboscript.memory import read_manifest
from roboscript.records import replay, write_atomic
from roboscript.settings import repo_root, runs_path, write_secret
from roboscript.stack import (
    Interpreters,
    ServiceSpec,
    StackConfig,
    cancel_on_signals,
    least_loaded,
    ready_path,
    stop_groups,
    up,
)
from roboscript.tools.skillsets import load_skillset
from roboscript.toolsets.scene import SceneConfig
from roboscript_robotwin.benchmark import P3_TASK_CONFIGS, ROBOTWIN
from roboscript_robotwin.config import (
    robotwin_root,
    with_perception_binding,
    with_perception_place,
)
from roboscript_robotwin.eval import install
from roboscript_robotwin.eval.wire import INFRA_LOST

log = logging.getLogger(__name__)

HOST: Final = "agent-host"
MOTION: Final = "motion"
VISION: Final = "vision"
PROFILE: Final = "profile.json"
"""A run's profile record (`P3Run.profile`)."""


def scene_names(names: Iterable[str]) -> set[str]:
    """The scene toolsets among the registered toolsets `names`: those configured as the scene
    toolsets are (`Config` is `SceneConfig`: the core's perception, geometry and worldstate, and
    a method's own). They filter the robot out with the motion service and
    cross-check with the vision service."""
    found = set()
    for name in names:
        try:
            toolset = entrypoints.load("roboscript.toolsets", name)
        except LookupError as error:
            raise ValueError(str(error)) from error
        if getattr(toolset, "Config", None) is SceneConfig:
            found.add(name)
    return found


MOTION_MODULE: Final = "roboscript.motion"
MOTION_READY_S: Final = 600.0  # cuRobo builds and warms up its planners first
STARTED: Final = "Model server started on localhost:{port}"
"""What RoboTwin's policy server prints once it listens (after `get_model`, in a thread that
binds the port: a failed bind leaves the process running, silent but for a traceback)."""
TRACEBACK: Final = "Traceback (most recent call last)"
ADDRESS_IN_USE: Final = f"[Errno {errno.EADDRINUSE}] {os.strerror(errno.EADDRINUSE)}"
"""How the server's traceback ends when another process took the port first."""
PORT_ATTEMPTS: Final = 4  # the first port and 3 more
CLIENT: Final = "client"
SERVER: Final = "policy_server"
OFFICIAL: Final = "official"
SHARDS: Final = "shards"  # the shard clients' records (`roboscript_robotwin.eval.shard_client`)
CARRIED: Final = "carried.json"  # the episodes a resumed run carries (`P3Run.resume`)
VERDICTS: Final = "verdicts.jsonl"  # in SHARDS: the expert's pinned verdicts (`shard_client`)
CLAIMS: Final = "claims.jsonl"  # in SHARDS: which shard took what (`shard_client.Claims`)
LOST: Final = "lost-from"
"""Host slots `<slot>.lost-from-<run>`: the records of episodes infrastructure lost in an
interrupted run, which a resumed run evaluates again; history, not counted."""
ENDED: Final = "ended.json"
"""The shard clients a run's launcher may count ended for good (`P3Run.release_shards`)."""
CLOSED: Final = "closed.json"  # a block closed from a set-aside run without running (`close`)
_RATE_LINE: Final = re.compile(r"Success rate: (\d+)/(\d+)\D[^\n]*?current seed: (\d+)")
_LOG_LINE: Final = re.compile(rf"({re.escape(INFRA_LOST)})|{_RATE_LINE.pattern}")
"""Our policy's mark of an episode infrastructure lost, or the official client's rate line."""
MAX_LOSSES: Final = 3
"""How many runs in a row may lose one episode to infrastructure before a resume refuses to run
it again (`evaluate`): a loss that keeps coming back is a fault for a human to look at, not a
loop."""
_ANSI: Final = re.compile(r"\x1b\[[0-9;]*m")
Process = anyio.abc.Process
Mode = Literal["clean", "randomized"]
_TOOLSETS: Final = TypeAdapter(list[ToolsetSpec])


class P3Run(BaseModel):
    """One evaluation. `seed` is the official seed block: 0 evaluates seeds from 100000, 1 from
    200000. `max_episodes` evaluates only the block's first episodes, carried ones included (the
    shard client, one shard alone too, stops after them: a block cut on purpose, `truncated`).
    `shards` > 1 evaluates the block's episodes in as many server and client pairs at once
    (`roboscript_robotwin.eval.shard_client`), on the seeds one official client would evaluate.
    `resume`, an interrupted run of the same block (set aside), carries the episodes it finished
    (`finished`): the clients check every seed as before and evaluate only the others. `profile`: a
    record of the settings the run's Agent host runs with (the pipeline's `blocks.block_profile`),
    written to `profile.json` as the run starts. `provenance`: the code and behaviour config every
    episode summary of the run's host records (`HostConfig.provenance`; None: the code's revision
    alone); carried episodes keep their own run's. `pins`: a verdicts file
    (`shard_client.Verdicts`) the run shares with other runs of the same block instead of its own
    `shards/verdicts.jsonl` (None), so that the runs evaluate the same seeds at the same indices
    whichever checks a seed first (P4's arms on one block, `roboscript_robotwin.eval.p4`); the run
    then always runs the shard client, and a resumed run follows the shared verdicts as they stand
    rather than a copy of its own. `steal_shards` (with `shards` > 1): the shards take the episodes
    as they come free (`shard_client --claims`, `shards/claims.jsonl`) rather than by index modulo
    `shards`, so a shard whose episodes end early does not idle while the slowest finishes its own.
    `release_shards` (with `shards` > 1): its job starts no shard again after this run, so each
    shard client that ends after its episodes is recorded in `ended.json`, from which the job's
    launcher gives back what it held (`roboscript.orchestrate.slots.Reservation`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    seed: NonNegativeInt
    mode: Mode = "clean"
    manifest: Path
    out_dir: Path
    max_episodes: PositiveInt | None = None
    shards: PositiveInt = 1
    strict_shards: bool = True
    steal_shards: bool = False
    resume: Path | None = None
    profile: dict[str, Any] | None = None
    provenance: Provenance | None = None
    pins: Path | None = None
    release_shards: bool = False


class P3Motion(BaseModel):
    """The motion service the host's manipulation toolset plans with, exactly one of: `url`, a
    running one to share (one per GPU serves any number of episodes), or `config`, the config of
    one the launcher starts for the run on its GPU (a YAML file or a name under configs/)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str | None = None
    config: str | None = None

    @model_validator(mode="after")
    def _one_service(self) -> Self:
        if (self.url is None) == (self.config is None):
            raise ValueError("set exactly one of motion.url and motion.config")
        return self


class P3Vision(BaseModel):
    """The vision service the host's scene toolsets cross-check with, exactly one of: `url`, a
    running one to share (one per GPU serves any number of episodes), with `blob_dir`, the blob
    store it shares with its callers, which the run's processes then use instead of their own
    (images reach the service there, by reference); or `config`, the config of one the launcher
    starts for the run on `vision_gpu` (a YAML file or a name under configs/), with the run's
    store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str | None = None
    blob_dir: Path | None = None
    config: str | None = None

    @model_validator(mode="after")
    def _one_service(self) -> Self:
        if (self.url is None) == (self.config is None):
            raise ValueError("set exactly one of vision.url and vision.config")
        if (self.url is None) != (self.blob_dir is None):
            raise ValueError(
                "a shared vision service (vision.url) needs its store, vision.blob_dir"
            )
        return self


class P3Config(BaseModel):
    """`examples/simBenchmarks/RoboTwin/configs/p3.example.yaml`: the interpreters (a vision one
    for a vision service the launcher starts), the official processes' PYTHONPATH (also a started
    motion service's) and GPU, the GPU of a vision service it starts (None: the least loaded of
    `gpus` as the run starts, `roboscript.stack.least_loaded`), the GPUs the machine lends the
    evaluation, the motion service (None: the host mounts neither the manipulation toolset nor the
    scene toolsets), the vision service the scene toolsets cross-check with (None: no cross-check),
    the table height the manipulation toolset keeps targets above per scene config until perception
    estimates the table (clean scenes have their table at 0.74 m; randomized ones lower it by up to
    3 cm, so their fallback is the floor, 0.71, which refuses nothing above the table), the Agent
    host's config without what each run sets (`manifest`, `episodes.records`, the services and
    table height of the manipulation and scene toolsets; the blob store is the run's), how long the
    policy server may take to start (which also bounds the wait for the checkout's lock and for the
    client's result directory), and how long a process group may take to stop before it is killed.
    Relative paths are relative to the repository root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    interpreters: Interpreters
    sim_pythonpath: list[Path]
    gpu: NonNegativeInt = 0
    vision_gpu: NonNegativeInt | None = None
    gpus: list[NonNegativeInt] = [0]
    motion: P3Motion | None = None
    vision: P3Vision | None = None
    table_top_z: dict[Mode, FiniteFloat] = {"clean": 0.74, "randomized": 0.71}
    host: dict[str, Any]
    server_ready_timeout_s: PositiveFloat = 300.0
    stop_grace_s: PositiveFloat = 10.0

    @model_validator(mode="after")
    def _sim_interpreter(self) -> Self:
        if self.interpreters.sim is None:
            raise ValueError("set interpreters.sim: RoboTwin's scripts run in the simulator env")
        return self

    @model_validator(mode="before")
    @classmethod
    def _perception_binding(cls, data: Any) -> Any:
        """The Agent host's scene toolsets perceive with RoboTwin's perception binding as S1 and
        S2 do, and its manipulation toolset finds the demonstration's table with its anchor
        camera and table area (`table_perception`)."""
        if not isinstance(data, dict) or not isinstance(data.get("host"), dict):
            return data
        if "toolsets" not in data["host"]:
            return data
        specs = data["host"].get("toolsets", [])
        scene = scene_names(s["name"] for s in specs if isinstance(s, dict) and "name" in s)

        def bound(spec: Any) -> Any:
            if not isinstance(spec, dict):
                return spec
            config = spec.get("config") or {}
            if spec.get("name") == "manipulation" and spec.get("url") is None:
                table = with_perception_place(config.get("table_perception") or {})
                return {**spec, "config": {**config, "table_perception": table}}
            if spec.get("name") not in scene:
                return spec
            pipeline = with_perception_binding(config.get("pipeline") or {})
            return {**spec, "config": {**config, "pipeline": pipeline}}

        return {**data, "host": {**data["host"], "toolsets": [bound(s) for s in specs]}}

    @model_validator(mode="after")
    def _motion_with_manipulation(self) -> Self:
        names = {spec.name for spec in self.toolsets()}
        mounted = "manipulation" in names
        scene = scene_names(names)
        if (mounted or scene) and self.motion is None:
            raise ValueError(
                "the host mounts the manipulation or scene toolsets: set motion to plan with (the "
                "scene toolsets filter the robot out with it)"
            )
        if self.motion is not None and not (mounted or scene):
            raise ValueError("motion is set, but the host mounts no manipulation toolset")
        if mounted and set(self.table_top_z) != {"clean", "randomized"}:
            raise ValueError("set table_top_z for both scene configs, clean and randomized")
        return self

    @model_validator(mode="after")
    def _vision_with_scene(self) -> Self:
        if self.vision is None:
            return self
        if not scene_names(spec.name for spec in self.toolsets()):
            raise ValueError("vision is set, but the host mounts no scene toolset")
        if self.vision.config is not None and self.interpreters.vision is None:
            raise ValueError("vision.config starts a vision service: set interpreters.vision")
        return self

    @model_validator(mode="after")
    def _lent_gpus(self) -> Self:
        for name in ("gpu", "vision_gpu"):
            value = getattr(self, name)
            if value is not None and value not in self.gpus:
                raise ValueError(f"{name} {value} is not one of the GPUs lent, gpus {self.gpus}")
        return self

    def toolsets(self) -> list[ToolsetSpec]:
        """The host's toolsets (`host.toolsets`), validated."""
        return _TOOLSETS.validate_python(self.host.get("toolsets", []))


class P3Result(BaseModel):
    """What a run produced: the official result directory (moved into the run's) and success rate
    (None when the run was cut short), the host's episode summaries in order, how many of them
    infrastructure failed (`infra` in a summary: a failed call of the client or such an end;
    the official rate counts them as failures, like any other), the configuration error that
    stopped the run (`<end reason>: <note>` of the host's `config_error.json`), the client's exit
    code (with shards: the first nonzero one) and the process group ids of the host, the policy
    server and the client (`policy_server.shard<k>`, `client.shard<k>` with shards). An episode
    whose reset never reached the host has no summary; the policy server's log names it.
    `outcomes`: the block's final outcome per episode index, as the official clients recorded it
    (index, seed, success; `lost` with `losses` when infrastructure lost its last attempt: no
    task outcome, a resume evaluates it again); `lost_episodes`: the host's summaries of
    attempts infrastructure lost in earlier runs of the block, kept as history, not counted.
    `outcomes_error`: why the outcomes could not be read (None: read), in which case `outcomes`
    is empty and says nothing about lost episodes. `skillsets`: by toolset, the skillsets the
    host mounted on it, name -> version; `brain`: the Agent host's (the Actor's) brain, name and
    config (empty or None: not recorded)."""

    model_config = ConfigDict(frozen=True)

    task: TaskName
    seed: int
    task_config: str
    manifest: ManifestId
    official_dir: Path | None
    success_rate: float | None
    episodes: list[dict[str, Any]]
    infra_episodes: int
    truncated: bool
    config_error: str | None = None
    client_exit: int | None
    pids: dict[str, int]
    shards: PositiveInt = 1
    outcomes: list[dict[str, Any]] = []
    outcomes_error: str | None = None
    lost_episodes: list[dict[str, Any]] = []
    brain: dict[str, Any] | None = None
    skillsets: dict[str, dict[str, str]] = {}


@dataclass
class _Launched:
    """What a run started, for its cleanup: the process groups (recorded in `pids.json` as they
    start) and each client's result directory in RoboTwin, by the client's name."""

    out: Path
    processes: dict[str, Process] = field(default_factory=dict)
    pids: dict[str, int] = field(default_factory=dict)
    result_dirs: dict[str, Path | None] = field(default_factory=dict)

    def clients(self) -> list[Process]:
        return [p for name, p in self.processes.items() if name.split(".")[0] == CLIENT]

    def ended(self, name: str) -> None:
        """The process `name` ended after its work, for good (`ENDED`)."""
        path = self.out / ENDED
        found = json.loads(path.read_bytes()) if path.is_file() else {}
        found[name] = self.pids[name]
        write_atomic(path, (json.dumps(found, indent=2) + "\n").encode())

    def started(self, name: str, pid: int, process: Process | None = None) -> None:
        self.pids[name] = pid
        if process is not None:
            self.processes[name] = process
        write_atomic(self.out / "pids.json", (json.dumps(self.pids, indent=2) + "\n").encode())


async def evaluate(run: P3Run, config: P3Config) -> P3Result:
    """Run one official evaluation (see the module docstring) and write its `result.json`."""
    root = robotwin_root()
    if root is None:
        raise ValueError("set ROBOTWIN_ROOT in the environment or .env")
    root = root.resolve()
    install.check_installed(root)
    if config.vision is not None and config.vision.config is not None and config.vision_gpu is None:
        vision_gpu = await least_loaded(config.gpus)
        lent = config.gpus
        log.info("the vision service goes on GPU %d, the least loaded of %s", vision_gpu, lent)
        config = config.model_copy(update={"vision_gpu": vision_gpu})
    manifest = read_manifest(_absolute(run.manifest))
    if run.task not in manifest.tasks:
        raise ValueError(f"manifest {manifest.id} has no Memory for {run.task}")
    mounted = {spec.name for spec in config.toolsets()}
    if manifest.tasks[run.task].vision and (config.vision is None or "perception" not in mounted):
        raise ValueError(
            f"{run.task}'s Memory was learned with the vision service's cross-check: mount the "
            "perception toolset and set vision (a shared service or one to start), so that P3 "
            "identifies objects as learning did"
        )
    out = _absolute(runs_path(run.out_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: each run needs a directory of its own")
    source = None if run.resume is None else _absolute(run.resume)
    task_config = P3_TASK_CONFIGS[run.mode].stem
    carry, carried, rerun = carry_plan(run, out, source, task_config)
    verdicts = out / SHARDS / VERDICTS if run.pins is None else _absolute(run.pins)
    if run.pins is None and source is not None and (source / SHARDS / VERDICTS).is_file():
        window = [*carried, *rerun]
        carry_verdicts(source / SHARDS / VERDICTS, verdicts, window)
    if run.profile is not None:
        write_atomic(out / PROFILE, (json.dumps(run.profile, indent=2) + "\n").encode())
    # The expert's verdicts on the block's seeds come from the GPU planner, so a replay follows the
    # recording's (the shard clients pin them: `shard_client.Verdicts`).
    tape, kept = replay.active(), f"p3-verdicts-{run.mode}-block{run.seed}.jsonl"
    pinned = tape.kept(kept) if tape is not None and tape.mode == "replay" else None
    if (
        pinned is not None
        and source is None
        and not (verdicts.is_file() and verdicts.stat().st_size)
    ):
        write_atomic(verdicts, pinned)
    launched = _Launched(out)
    try:
        with cancel_on_signals(signal.SIGTERM, signal.SIGHUP):
            truncated = await _run(run, config, root, task_config, manifest.id, launched, carry)
    finally:
        with anyio.CancelScope(shield=True):
            official = await _move_out(root, run.task, launched, config.server_ready_timeout_s)
            if source is not None and (carried or rerun):
                carry_records(source, out, official, carried, run.shards)
            if tape is not None and tape.mode == "record" and verdicts.is_file():
                tape.keep(kept, verdicts.read_bytes())
    exits = [client.returncode for client in launched.clients()]
    exit_code = next((code for code in exits if code != 0), exits[0] if exits else None)
    if run.shards > 1 and official is not None:
        merge_shards(
            out, run.shards, full=not truncated and exit_code == 0, strict=run.strict_shards
        )
    skillsets = mounted_skillsets(config.toolsets())
    brain = BrainSpec.model_validate(config.host.get("brain", {})).model_dump(mode="json")
    result = _collect(
        run, task_config, manifest.id, official, truncated, exit_code, launched, skillsets, brain
    )
    if result.config_error is not None:
        raise RuntimeError(
            f"the run stopped on a configuration error ({result.config_error}): see "
            f"{out}/result.json"
        )
    if exit_code not in (None, 0):  # a cut run's too: its clients stop after their episodes
        raise RuntimeError(f"the official client exited with {exit_code}: see {out}/logs")
    return result


def carry_plan(
    run: P3Run, out: Path, source: Path | None, task_config: str
) -> tuple[Path | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """What a run of block `out` resumed from `source` carries and evaluates again
    (`resume_plan`), written to `carried.json` when it is either (also when nothing finished but
    episodes were lost: their losses count on): that file (None: nothing to carry) and both."""
    carried, rerun = ([], []) if source is None else resume_plan(source)
    if not (carried or rerun):
        return None, carried, rerun
    carry = out / CARRIED
    record = {
        "source": str(source),
        "task": run.task,
        "task_config": task_config,
        "st_seed": 100000 * (1 + run.seed),
        "episodes": carried,
        "rerun": rerun,
    }
    write_atomic(carry, (json.dumps(record, indent=1) + "\n").encode())
    log.info("%s: %d episodes carried from %s", run.task, len(carried), source)
    if rerun:  # no task outcome; each runs again on its seed and index
        log.warning("%s: infrastructure lost episodes %s: evaluated again", run.task, rerun)
    return carry, carried, rerun


def carry_verdicts(source: Path, target: Path, carried: list[dict[str, Any]]) -> None:
    """The pinned expert verdicts (`shard_client.Verdicts`) a resumed run starts from: those on
    seeds up to its last carried one. The verdicts after it were pinned by clients that ran
    other episodes before (on some tasks the check depends on that), and the shard that
    evaluates such a seed runs its expert check again right before its episode: pinned solvable
    where its own check fails, it has no episode to run, so the resumed run checks them
    afresh."""
    last = max((int(episode["seed"]) for episode in carried), default=None)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [
        line
        for line in lines
        if line.strip() and last is not None and int(json.loads(line)["seed"]) <= last
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(target, "".join(kept).encode())


async def _run(
    run: P3Run,
    config: P3Config,
    root: Path,
    task_config: str,
    manifest: ManifestId,
    launched: _Launched,
    carry: Path | None = None,
) -> bool:
    """Start the host, the server and the client, and wait for the client; whether the run was
    cut (`max_episodes`, or a configuration error). With shards, `carry` (carried episodes),
    `max_episodes` or `pins`, the clients are `shard_client`'s, one shard alone too."""
    out = launched.out
    verdicts = out / SHARDS / VERDICTS if run.pins is None else _absolute(run.pins)
    stealing = run.steal_shards and run.shards > 1
    token_file = out / ".roboscript" / "host.token"
    write_secret(token_file, SecretStr(secrets.token_urlsafe(32)))
    try:
        services, vision = vision_service(config, out)
        host = host_config(run, config, out, vision)
        if config.motion is not None and config.motion.config is not None:
            services.append(
                ServiceSpec(
                    name=MOTION,
                    module=MOTION_MODULE,
                    interpreter="sim",
                    args=["--config", config.motion.config],
                    gpu=config.gpu,
                    ready_timeout_s=MOTION_READY_S,
                    stop_grace_s=config.stop_grace_s,
                )
            )
        host_file = out / "agent_host.yaml"
        dumped = yaml.safe_dump(host.model_dump(mode="json"), sort_keys=False)
        write_atomic(host_file, dumped.encode())
        service = ServiceSpec(
            name=HOST,
            module="roboscript.learning.agent_host",
            args=["--config", str(host_file), "--token-file", str(token_file)],
            depends_on=[spec.name for spec in services],
            stop_grace_s=config.stop_grace_s,
        )
        stack = StackConfig(
            interpreters=config.interpreters,
            sim_pythonpath=[_absolute(path) for path in config.sim_pythonpath],
            vision_pythonpath=[
                repo_root() / "packages/core/src",
                repo_root() / "packages/vision/src",
            ],
            run_dir=out,
            blob_dir=blob_store(config, out),
            services=[*services, service],
        )
        async with up(stack, on_start=launched.started) as running:
            env = _environment(config, running.blob_dir)
            episode = overrides(run.task, task_config, manifest, run.seed)
            relay = ["--host_url", running.endpoints[HOST], "--host_token_file", str(token_file)]
            official = root / "eval_result" / run.task / install.POLICY / task_config / manifest
            deploy = ["--config", f"policy/{install.POLICY}/deploy_policy.yml"]
            try:
                pairs = []
                for shard in range(run.shards):
                    name = "" if run.shards == 1 else f".shard{shard}"
                    sharded = [] if run.shards == 1 else ["--shard", str(shard)]
                    args = [*deploy, *episode, *relay, *sharded]
                    server, port = await _start_server(config, root, env, launched, args, name)
                    script = ["script/eval_policy_client.py"]
                    # the official main, its episodes: a block with shards, carried episodes or
                    # a count stops after its episodes itself, exactly
                    sharded_client = run.shards > 1 or carry is not None or run.pins is not None
                    if sharded_client or run.max_episodes is not None:
                        record = out / SHARDS / f"shard{shard}.jsonl"
                        script = [
                            *("-m", "roboscript_robotwin.eval.shard_client"),
                            *("--shard", str(shard)),
                            *("--of", str(run.shards), "--record", str(record)),
                            *([] if run.max_episodes is None else ["--episodes"]),
                            *([] if run.max_episodes is None else [str(run.max_episodes)]),
                            *([] if carry is None else ["--carry", str(carry)]),
                            *("--verdicts", str(verdicts)),
                            *(["--claims", str(out / SHARDS / CLAIMS)] if stealing else []),
                            "--",
                        ]
                    script += ["--port", str(port), *deploy, *episode]
                    client = await _start_client(
                        config, root, env, launched, run.task, official, script, name
                    )
                    pairs.append((client, server))
                if run.shards == 1:  # a shard client stops after its episodes itself
                    cut = await _wait(client, server, out)
                    return cut or run.max_episodes is not None

                def ended(k: int) -> None:
                    launched.ended(f"{CLIENT}.shard{k}")

                told = ended if run.release_shards else None
                return await _wait_shards(pairs, out, told) or run.max_episodes is not None
            finally:
                with anyio.CancelScope(shield=True):
                    groups = reversed(launched.processes.values())
                    await stop_groups([(process, config.stop_grace_s) for process in groups])
    finally:
        token_file.unlink(missing_ok=True)


def mounted_skillsets(toolsets: list[ToolsetSpec]) -> dict[str, dict[str, str]]:
    """By toolset of the host, the skillsets mounted on it (`_with_skillsets`): name -> version."""
    found = {}
    for spec in _with_skillsets(toolsets):
        manifests = [load_skillset(ref).manifest for ref in spec.skillsets]
        found[spec.name] = {m.name: m.version for m in manifests}
    return found


def _with_skillsets(toolsets: list[ToolsetSpec]) -> list[ToolsetSpec]:
    """Each toolset with RoboTwin's skillsets mounted (`ROBOTWIN.skillsets`), as S1 and S2 mount
    them, each once (a host config may name them too: configs/agent_host.example.yaml)."""
    return [
        spec.model_copy(
            update={"skillsets": list(dict.fromkeys([*ROBOTWIN.skillsets, *spec.skillsets]))}
        )
        for spec in toolsets
    ]


def host_config(
    run: P3Run, config: P3Config, out: Path, vision: str | None = None
) -> AgentHostConfig:
    """The Agent host's config for `run` into `out`: `config.host` with the run's manifest and
    records, every toolset with RoboTwin's skillsets; for the manipulation toolset, the motion
    service to plan with (the shared one, or the ready file of the run's own) and the table
    fallback of the run's scene config; for the scene toolsets, the same motion service (their
    self-filter) and the vision service at `vision` (their cross-check; None: none)."""
    episodes: dict[str, Any] = {"records": str(out / "host")}
    if run.provenance is not None:
        episodes["provenance"] = run.provenance.model_dump(mode="json")
    per_run = {"manifest": str(_absolute(run.manifest)), "episodes": episodes}
    host = AgentHostConfig.model_validate(compose(config.host, per_run))
    host = host.model_copy(update={"toolsets": _with_skillsets(host.toolsets)})
    motion = config.motion
    if motion is None:
        return host
    shared = motion.url is not None
    ready = str(ready_path(out, MOTION))
    planner = {"url": motion.url} if shared else {"ready_file": ready}
    manipulation = {"planner": planner, "table_top_z": config.table_top_z[run.mode]}
    scene = {
        "planner": motion.url,
        "planner_ready_file": None if shared else ready,
        "vision": vision,
    }
    scenes = scene_names(spec.name for spec in host.toolsets)
    toolsets = [
        spec.model_copy(update={"config": {**spec.config, **manipulation}})
        if spec.name == "manipulation"
        else spec.model_copy(update={"config": {**spec.config, **scene}})
        if spec.name in scenes
        else spec
        for spec in host.toolsets
    ]
    return host.model_copy(update={"toolsets": toolsets})


def blob_store(config: P3Config, out: Path) -> Path:
    """The blob store of a run's processes: a shared vision service's (images reach it there),
    else the run's own."""
    shared = None if config.vision is None else config.vision.blob_dir
    return out / "blobs" if shared is None else _absolute(shared)


def vision_service(config: P3Config, out: Path) -> tuple[list[ServiceSpec], str | None]:
    """The vision service the host's scene toolsets cross-check with, and its URL: a shared one
    (nothing to start), one started for the run on a port chosen for it, or none."""
    vision = config.vision
    if vision is None or vision.url is not None:
        return [], None if vision is None else vision.url
    if config.vision_gpu is None:
        raise ValueError("set vision_gpu, or let `evaluate` pick the least loaded of gpus")
    ref = str(vision.config)
    if Path(ref).suffix in (".yaml", ".yml"):
        ref = str(_absolute(Path(ref)))
    port = _free_port()
    # started by its declared module with its config as given, validated where it loads; its
    # store is the stack's (`RS_BLOB_DIR`), whatever the config names
    spec = ServiceSpec(
        name=VISION,
        module="roboscript_vision",
        interpreter="vision",
        args=["--config", ref, "--set", "blob_dir=null", "--port", str(port)],
        gpu=config.vision_gpu,
        ready_timeout_s=600,
        stop_grace_s=config.stop_grace_s,
    )
    return [spec], f"http://127.0.0.1:{port}"


def overrides(task: str, task_config: str, manifest: ManifestId, seed: int) -> list[str]:
    """The `--overrides` of both official scripts: the official fields of the run. The client
    `eval`s every value (`mf-...` names an undefined variable, so a manifest id stays a string);
    the server gets the host's fields after these."""
    return [
        *("--overrides", "--task_name", task, "--task_config", task_config),
        *("--ckpt_setting", manifest, "--seed", str(seed)),
    ]


def _environment(config: P3Config, blob_dir: Path) -> dict[str, str]:
    """The official processes' environment: the simulator env's PYTHONPATH (nothing of ours is
    installed there), no user site, the GPU, where `.env` is and the stack's blob store."""
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(str(_absolute(path)) for path in config.sim_pythonpath),
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore::UserWarning",  # as RoboTwin's own scripts run them
        "CUDA_VISIBLE_DEVICES": str(config.gpu),
        "RS_ROOT": str(repo_root()),
        "RS_BLOB_DIR": str(blob_dir),
    }


async def _start(
    config: P3Config, root: Path, env: dict[str, str], log: Path, script: list[str]
) -> Process:
    """An official script in the simulator env, from `root`, in its own process group, its
    output in `log`."""
    command = [str(config.interpreters.sim), *script]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        output.write(f"$ {shlex.join(command)}\n".encode())
        output.flush()
        return await anyio.open_process(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


class _PortTaken(Exception):
    """The policy server's bind failed: another process took the port after the probe."""


async def _start_client(
    config: P3Config,
    root: Path,
    env: dict[str, str],
    launched: _Launched,
    task: str,
    official: Path,
    script: list[str],
    name: str,
) -> Process:
    """The client `script` (`client<name>`, logging to `client<name>.log`), under the task's lock
    from its start until its result directory exists and that second has passed (the next client
    names another); the directories it will make under `eval_result/<task>` are recorded first,
    its result directory (under `official`) once it exists."""
    top, wait_s = root / "eval_result" / task, config.server_ready_timeout_s
    async with _checkout_lock(root, task, wait_s):
        before = _entries(official)
        _record_made(root, task, _absent(official, top))
        log_file = launched.out / "logs" / f"{CLIENT}{name}.log"
        client = await _start(config, root, env, log_file, script)
        launched.started(f"{CLIENT}{name}", client.pid, client)
        found: Path | None = None
        try:
            found = await _new_entry(official, before, client, wait_s)
        finally:
            if found is None:  # stop the client before it makes one
                with anyio.CancelScope(shield=True):
                    await stop_groups([(client, config.stop_grace_s)])
                    found = max(_entries(official) - before, default=None)
            launched.result_dirs[f"{CLIENT}{name}"] = found
        await anyio.sleep(1.05 - time.time() % 1.0)
    return client


async def _start_server(
    config: P3Config,
    root: Path,
    env: dict[str, str],
    launched: _Launched,
    args: list[str],
    name: str = "",
) -> tuple[Process, int]:
    """The policy server `policy_server<name>` (with `args` after its port) listening on a port
    found free, and that port: a port taken between the probe and the server's bind costs that
    server, and another port is tried, `PORT_ATTEMPTS` times in all."""
    server_log = launched.out / "logs" / f"{SERVER}{name}.log"
    for attempt in range(1, PORT_ATTEMPTS + 1):
        port = _free_port()
        offset = server_log.stat().st_size if server_log.exists() else 0
        script = ["script/policy_model_server.py", "--port", str(port), *args]
        server = await _start(config, root, env, server_log, script)
        launched.started(f"{SERVER}{name}", server.pid, server)
        try:
            await _server_started(server, server_log, offset, port, config.server_ready_timeout_s)
            return server, port
        except _PortTaken:
            with anyio.CancelScope(shield=True):
                await stop_groups([(server, config.stop_grace_s)])
            log.warning(
                "port %d was taken before the policy server bound it (%d of %d)",
                port,
                attempt,
                PORT_ATTEMPTS,
            )
    raise RuntimeError(
        f"the policy server found its port taken {PORT_ATTEMPTS} times:\n{_tail(server_log)}"
    )


async def _server_started(
    server: Process, log: Path, offset: int, port: int, seconds: float
) -> None:
    """Wait for the policy server's own line that it listens on `port` (`get_model` ran: the host
    answered), in its log from byte `offset` on (its own part). A bind that failed because the
    address is in use is `_PortTaken`; another traceback or its exit fails at once, and so does
    silence past `seconds`."""
    started, deadline = STARTED.format(port=port), anyio.current_time() + seconds

    async def written() -> str:
        return (await anyio.Path(log).read_bytes())[offset:].decode(errors="replace")

    while started not in (text := await written()):
        if server.returncode is not None or TRACEBACK in text:
            await anyio.sleep(0.2)  # the rest of the traceback
            if ADDRESS_IN_USE in await written():
                raise _PortTaken(port)
            how = "exited" if server.returncode is not None else "failed"
            raise RuntimeError(f"the policy server {how} before listening:\n{_tail(log)}")
        if anyio.current_time() > deadline:
            raise TimeoutError(f"the policy server did not listen within {seconds:g} s")
        await anyio.sleep(0.2)


@contextlib.asynccontextmanager
async def _checkout_lock(root: Path, task: str, seconds: float) -> AsyncIterator[None]:
    """The lock of a RoboTwin checkout's `eval_result/<task>` among this user's runs, whatever
    RoboScript checkout they come from; waiting more than `seconds` for it fails."""
    path = _lock_path(root, task)
    deadline = anyio.current_time() + seconds
    with path.open("a", encoding="utf-8") as handle:  # closing it releases the lock
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if anyio.current_time() > deadline:
                    raise TimeoutError(
                        f"another run held the lock of {task} in {root} ({path}) for {seconds:g} s"
                    ) from None
                await anyio.sleep(0.2)
        yield


def _lock_path(root: Path, task: str) -> Path:
    """Per user, not per RoboScript checkout (two worktrees may drive one RoboTwin), and by the
    checkout's real path (a symbolic link to it names the same lock) and the task."""
    key = hashlib.sha256(f"{root.resolve()}\0{task}".encode()).hexdigest()[:16]
    path = Path.home() / ".cache" / "roboscript" / "locks" / f"robotwin-{key}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _entries(directory: Path) -> set[Path]:
    return set(directory.iterdir()) if directory.is_dir() else set()


async def _new_entry(
    directory: Path, before: set[Path], client: Process, seconds: float
) -> Path | None:
    """The result directory the client makes at its start (None if it exits first)."""
    deadline = anyio.current_time() + seconds
    while not (new := _entries(directory) - before):
        if client.returncode is not None:
            return None
        if anyio.current_time() > deadline:
            raise TimeoutError(f"the official client made no result directory in {seconds:g} s")
        await anyio.sleep(0.2)
    return max(new)


async def _wait(client: Process, server: Process, out: Path) -> bool:
    """Wait for the client to finish; whether it was stopped early: once the host failed the run
    on a configuration error."""
    while client.returncode is None:
        if server.returncode is not None:
            raise RuntimeError(
                f"the policy server exited ({server.returncode}) during the evaluation:\n"
                + _tail(out / "logs" / "policy_server.log")
            )
        if (out / "host" / CONFIG_ERROR).is_file():
            return True
        with anyio.move_on_after(1.0):
            await client.wait()
    return False


async def _wait_shards(
    pairs: list[tuple[Process, Process]],
    out: Path,
    ended: Callable[[int], None] | None = None,
) -> bool:
    """Wait for every shard client to finish (each stops after its episodes); whether the run was
    stopped early, once the host failed it on a configuration error. A server that exits, or a
    client that fails, fails the run (the others are stopped then). `ended(k)` is told of each
    shard client `k` that exited 0, once."""
    told: set[int] = set()
    while any(client.returncode is None for client, _ in pairs):
        for k, (client, server) in enumerate(pairs):
            if client.returncode == 0 and ended is not None and k not in told:
                told.add(k)
                ended(k)
            if server.returncode is not None:
                raise RuntimeError(
                    f"policy server {k} exited ({server.returncode}) during the evaluation:\n"
                    + _tail(out / "logs" / f"{SERVER}.shard{k}.log")
                )
            if client.returncode not in (None, 0):
                return False  # `evaluate` raises on its exit code
        if (out / "host" / CONFIG_ERROR).is_file():
            return True
        await anyio.sleep(1.0)
    return False


def host_summaries(out: Path, lost: bool = False) -> list[Path]:
    """The host's episode summaries that count (with `lost`: those kept as history only)."""
    found = sorted((out / "host").glob("*/ep*/summary.json"))
    return [path for path in found if _lost_slot(path.parent.parent.name) == lost]


async def _move_out(root: Path, task: str, launched: _Launched, seconds: float) -> Path | None:
    """Once a client ran: under the task's lock (another client of the task may be making
    directories), move its result directory to `<run>/official` (a shard client's to
    `<run>/official/shard<k>`) and remove the directories our clients made that are empty now.
    Where the result is now, if there was one: a finished run never fails here, so if the lock
    does not come in time or moving fails, a result directory stays where it is (of one client:
    that directory; of shards: the run's `official`, which `merge_shards` then finds
    incomplete)."""
    found = {n: d for n, d in launched.result_dirs.items() if d is not None and d.is_dir()}
    if not launched.clients():
        return None
    official = launched.out / OFFICIAL
    sharded = any("." in name for name in launched.result_dirs)  # `client.shard<k>`
    try:
        async with _checkout_lock(root, task, seconds):
            for name, directory in found.items():
                shard = name.split(".", 1)[-1]
                shutil.move(directory, official / shard if sharded else official)
                found[name] = official
            _remove_made(root, task)
    except (TimeoutError, OSError, ValueError):  # the lock, a move, the record
        log.exception("the official result stays at %s", sorted(map(str, found.values())))
    if sharded:
        return official if official.is_dir() else None
    return found.get(CLIENT)


def merge_shards(out: Path, shards: int, full: bool, strict: bool = True) -> None:
    """The shards' episodes as one block: `official/shards.json` (each evaluated episode's index,
    seed, shard and success, in order, and the solvable seeds) and, when `full` and every shard
    evaluated all its episodes and wrote its `_result.txt`, the official `official/_result.txt`
    with the block's rate (its successes over the official episode count, as the client writes
    it), stamped as shard 0's. Each shard ran every seed's expert check: shards that disagree
    on the solvable seeds (in their common part) fail the merge, unless not `strict`: then the
    merge warns, keeps each index's episode from the shard that evaluated it and records
    `"disagreed": true` (a rough estimate, not the official seed order)."""
    records = [_shard_record(out / SHARDS / f"shard{k}.jsonl") for k in range(shards)]
    if not all(records):  # a shard client that never began its loop
        return
    headers = [lines[0] for lines in records]
    test_num, count = headers[0]["test_num"], headers[0]["episodes"]
    solvable = [[(r["index"], r["seed"]) for r in lines[1:] if "index" in r] for lines in records]
    common = min(map(len, solvable))
    disagreed = any(seeds[:common] != solvable[0][:common] for seeds in solvable)
    if disagreed and not strict:
        log.warning("%s: the shards found different solvable seeds; merged anyway", out)
    elif disagreed:
        raise RuntimeError(
            f"the shards found different solvable seeds (see {out / SHARDS}): the expert check "
            "is not deterministic here"
        )
    episodes = sorted(
        (
            {"index": r["index"], "seed": r["seed"], "shard": k, "success": r["success"]}
            for k, lines in enumerate(records)
            for r in lines[1:]
            if r["evaluated"]
        ),
        key=lambda e: e["index"],
    )
    complete = [e["index"] for e in episodes] == list(range(count))
    successes = sum(e["success"] for e in episodes)
    official = out / OFFICIAL
    merged = {
        "shards": shards,
        "test_num": test_num,
        "episodes": count,
        "complete": complete,
        "successes": successes,
        "solvable": [seed for _, seed in max(solvable, key=len)],
        "evaluated": episodes,
        "disagreed": disagreed,
    }
    write_atomic(official / "shards.json", (json.dumps(merged, indent=1) + "\n").encode())
    first = official / "shard0" / "_result.txt"
    written = all((official / f"shard{k}" / "_result.txt").is_file() for k in range(shards))
    if full and complete and count == test_num and written:
        head = first.read_text(encoding="utf-8").rsplit("\n\n", 1)[0]
        text = f"{head}\n\n{successes / test_num}"
        write_atomic(official / "_result.txt", text.encode())


def resumable(out: Path) -> Path | None:
    """The set-aside run of block `out` (`<name>-aborted<n>`) to resume from: the one that ended
    the most episodes (`finished`, and `lost`: their losses carry on), the latest of those; None
    when none ended one. The choice looks at how many episodes ended, never at how they ended. A
    set-aside run whose records cannot be read as one official sequence is left where it is and not
    chosen (a warning): with none to choose, the block starts over."""
    best: tuple[int, int, Path] | None = None
    for aside in out.parent.glob(f"{out.name}-aborted*"):
        if (n := re.fullmatch(rf"{re.escape(out.name)}-aborted(\d+)", aside.name)) is None:
            continue
        try:
            done, gone = _ended(aside)
        except (OSError, ValueError, KeyError, TypeError) as error:  # JSONDecodeError too
            log.warning("%s: not resumable, kept as it is and not carried from (%s)", aside, error)
            continue
        key = (len(done) + len(gone), int(n[1]), aside)
        if key[0] and (best is None or key[:2] > best[:2]):
            best = key
    return None if best is None else best[2]


def finished(block: Path) -> list[dict[str, Any]]:
    """The episodes a run of a block finished, as the official clients recorded them (index,
    seed and success each, by index): the shard clients' records (`shards/`, carried ones
    included), else the official client's own lines in `logs/client.log` (its rate after each
    episode: successes/episodes and the seed), and the episodes its `carried.json` carried that
    its records do not show yet (a resumed run cut while it replayed). An episode whose client
    did not get as far as that is not finished (the host's records are not the official count),
    nor is one infrastructure lost (`lost`), which leaves a gap that a resumed run fills.
    Shards whose expert checks disagreed (`_agreed`) are finished only up to where they split,
    and shards that stopped unevenly only up to the first episode that did not end (a resumed
    run replays the verdicts up to its last carried seed): the episodes from there on are not,
    and a resumed run evaluates them again."""
    return _ended(block)[0]


class LostRepeatedly(Exception):
    """Infrastructure lost an episode of a block `MAX_LOSSES` runs in a row: the block stops for a
    human to look at why. Not a `RuntimeError`: no launcher or scheduler retries it as a start
    that failed or as infrastructure, and the job fails with this message."""


def resume_plan(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """What a run resumed from `source` carries (`finished`) and evaluates again (`lost`);
    `LostRepeatedly` for an episode lost `MAX_LOSSES` runs in a row (a fault for a human, not a
    loop)."""
    carried, rerun = _ended(source)
    if worn := [e for e in rerun if e["losses"] >= MAX_LOSSES]:
        raise LostRepeatedly(
            f"infrastructure lost episodes {worn} (index, seed, runs in a row that lost it) "
            f"{MAX_LOSSES} runs in a row, the last {source}: look at why before resuming the block"
        )
    return carried, rerun


def lost(block: Path) -> list[dict[str, Any]]:
    """The episodes of a run of a block that infrastructure lost and no later attempt in it
    ended (no task outcome): index, seed and `losses`, how many runs in a row lost it
    (counted on through `carried.json`'s `rerun`). A resumed run evaluates exactly these again,
    on their seed and index, and carries every finished one."""
    return _ended(block)[1]


def _ended(block: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    done: dict[int, tuple[int, bool]] = {}
    gone: dict[int, int] = {}  # lost: index -> seed
    records = sorted((block / SHARDS).glob("shard*.jsonl"))
    if records:
        shards, cut = _agreed(block)
        for lines in shards:
            for line in lines[1:]:
                # a rerun whose expert check kept failing is lost again: it could not be run
                again = line.get("rerun") and line.get("error") is not None
                if not (line.get("evaluated") or again) or int(line["index"]) >= cut:
                    continue
                index, seed = int(line["index"]), int(line["seed"])
                if (line.get("infra") and not line.get("success")) or again:  # a success stands
                    gone[index] = seed
                else:
                    done[index] = (seed, bool(line["success"]))
    else:
        successes, marked = 0, False
        log_file = block / "logs" / f"{CLIENT}.log"
        text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else ""
        for found in _LOG_LINE.finditer(_ANSI.sub("", text)):
            if found[1] is not None:  # our policy's mark: this episode was lost
                marked = True
                continue
            wins, episodes, seed = int(found[2]), int(found[3]), int(found[4])
            index = episodes - 1
            if index != len(done) + len(gone):  # the official loop counts them one by one
                raise ValueError(f"{log_file}: episode {episodes} after {len(done) + len(gone)}")
            if marked and wins <= successes:  # a success stands (a client may have
                gone[index] = seed  # marked it lost all the same)
            else:
                done[index] = (seed, wins > successes)
            successes, marked = wins, False
    carried = _carried_record(block)
    for episode in carried.get("episodes", []):  # carried, not replayed yet: finished still
        index = int(episode["index"])
        if index not in done and index not in gone:
            done[index] = (int(episode["seed"]), bool(episode["success"]))
    before = {int(e["index"]): e for e in carried.get("rerun", [])}
    losses = {}
    for index, episode in before.items():
        if index in done:
            continue  # its rerun ended
        if index not in gone:  # not reached: still lost
            gone[index] = int(episode["seed"])
            losses[index] = int(episode.get("losses", 1))
        else:
            losses[index] = int(episode.get("losses", 1)) + 1
    episodes = [{"index": i, "seed": s, "success": w} for i, (s, w) in sorted(done.items())]
    lost_ = [
        {"index": i, "seed": s, "losses": losses.get(i, 1)}
        for i, s in sorted(gone.items())
        if i not in done
    ]
    seeds = [seed for _, seed in sorted([*((i, s) for i, (s, _) in done.items()), *gone.items()])]
    if seeds != sorted(set(seeds)):
        raise ValueError(f"{block}: the finished episodes' seeds do not rise with their index")
    return episodes, lost_


def _carried_record(block: Path) -> dict[str, Any]:
    """A resumed run's `carried.json` (empty for a run that carried nothing)."""
    path = block / CARRIED
    record: dict[str, Any] = json.loads(path.read_bytes()) if path.is_file() else {}
    return record


def carry_records(
    source: Path, out: Path, official: Path | None, carried: list[dict[str, Any]], shards: int
) -> None:
    """Bring the carried episodes' records from `source` into the resumed run `out`: the host's
    records of the episodes its clients ended (`host/<slot>.from-<source>/ep<n>`, so that the
    run's records count them; those infrastructure lost under `host/<slot>.lost-from-<source>/`,
    history only; their `index` is the source host's own count: of each slot, as many of its
    ended episodes, in order, as its client ended, `_finished_per_client`; those the source carried
    itself come as they are) and the official videos, into `official` as the client would have
    named them (`shard<k>/` with shards). Linked where the file system lets it, else copied; a
    record that cannot be brought is logged, and the run stands."""
    try:
        for name, episode in _host_carried(source):
            target = out / "host" / name / episode.name
            if not target.exists():
                shutil.copytree(episode, target, copy_function=_link)
        if official is None or not (source / OFFICIAL).is_dir():
            return
        for episode in carried:
            video = f"episode{episode['index']}.mp4"
            found = [source / OFFICIAL / video, *sorted((source / OFFICIAL).glob(f"*/{video}"))]
            where = official if shards == 1 else official / f"shard{episode['index'] % shards}"
            have = [path for path in found if path.is_file()]
            if have and where.is_dir() and not (where / video).exists():
                _link(have[0], where / video)
    except (OSError, ValueError):
        log.exception("the carried episodes' records from %s are not all in %s", source, out)


def _host_carried(source: Path) -> list[tuple[str, Path]]:
    """The host's records a resumed run brings from `source` (`carry_records`): the slot name
    they go under and each ended episode's directory. A `source` that resumed a run itself and
    stopped before it brought that run's records (they come as a run ends: a run killed outright
    has not) brings them from there (its `carried.json`'s source, recursively)."""
    own = _own_indices(source)
    gone = {episode["index"] for episode in lost(source)}  # routed by the clients' records
    found: dict[tuple[str, str], Path] = {}
    record = source / CARRIED
    origin = Path(json.loads(record.read_bytes())["source"]) if record.is_file() else None
    if origin is not None and origin.is_dir():
        found |= {(name, ep.name): ep for name, ep in _host_carried(origin)}
    for slot in sorted(p for p in (source / "host").glob("*") if p.is_dir()):
        carried_on = _lost_slot(slot.name) or ".from-" in slot.name  # brought as they are
        name = slot.name if carried_on else f"{slot.name}.from-{source.name}"
        shard = re.search(r"\.shard(\d+)$", slot.name)
        indices = [] if carried_on else own.get(0 if shard is None else int(shard[1]), [])
        ended = 0
        for episode in sorted(slot.glob("ep*")):
            summary = episode / "summary.json"
            if not summary.is_file() or (not carried_on and ended == len(indices)):
                continue
            if json.loads(summary.read_text(encoding="utf-8")).get("official") is None:
                continue
            where = name
            if not carried_on:  # its client's episode, in order; one it lost is history
                if indices[ended] in gone:
                    where = f"{slot.name}.{LOST}-{source.name}"
                ended += 1
            found[where, episode.name] = episode
    return [(name, episode) for (name, _), episode in sorted(found.items())]


def host_indices(block: Path) -> dict[int, Path]:
    """The host directory (`host/<slot>/ep<n>`) of each official episode index a run of a block
    holds records of or would bring, walked as `_host_carried` walks a source: first the run its
    `carried.json` resumed or closed from (recursively; found as a sibling of `block` by name, so
    that a copied run tree reads its own copy), then its slots: a slot of its own client holds its
    ended episodes (summary with an official result) in the order that client ended them
    (`_own_indices`, by shard), those infrastructure lost left out; a slot brought from another
    run (`<slot>.from-<run>`) holds episodes found along that chain, and its copy is the one
    returned. So a resumed run stopped before it brought its carried records points into its
    source. `.lost-from-` slots are history and left out. The host's own `index` in a summary is
    its count per slot, not this one."""
    found: dict[int, Path] = {}
    for path, index in _indexed(block).values():
        if found.get(index, path) != path:
            raise ValueError(f"{block}: episode {index} has records {found[index]} and {path}")
        found[index] = path
    return found


def _indexed(source: Path) -> dict[tuple[str, str], tuple[Path, int]]:
    """`host_indices` keyed as `_host_carried(source)` names the episodes it brings (slot name
    in the resumed run, episode directory name) -> (directory, index); lost ones left out."""
    found: dict[tuple[str, str], tuple[Path, int]] = {}
    record = source / CARRIED
    if record.is_file():
        origin = source.parent / Path(json.loads(record.read_bytes())["source"]).name
        if origin.is_dir():
            found |= _indexed(origin)
    own = _own_indices(source)
    gone = {episode["index"] for episode in lost(source)}
    for slot in sorted(p for p in (source / "host").glob("*") if p.is_dir()):
        if _lost_slot(slot.name):
            continue
        if ".from-" in slot.name:  # brought as they are: indexed along the chain
            for episode in sorted(slot.glob("ep*")):
                key = (slot.name, episode.name)
                if key in found:
                    found[key] = (episode, found[key][1])
                elif _ended_on_host(episode):
                    raise FileNotFoundError(
                        f"{episode}: brought from a run not on {source}'s carried.json chain"
                    )
            continue
        shard = re.search(r"\.shard(\d+)$", slot.name)
        indices = own.get(0 if shard is None else int(shard[1]), [])
        ended = 0
        for episode in sorted(slot.glob("ep*")):
            summary = episode / "summary.json"
            if not summary.is_file() or ended == len(indices):
                continue
            if not _ended_on_host(episode):
                continue
            if indices[ended] not in gone:
                found[f"{slot.name}.from-{source.name}", episode.name] = (episode, indices[ended])
            ended += 1
    return found


def _ended_on_host(episode: Path) -> bool:
    summary = episode / "summary.json"
    if not summary.is_file():
        return False
    return json.loads(summary.read_text(encoding="utf-8")).get("official") is not None


def closable(source: Path) -> int:
    """How many episodes a block closed from the set-aside run `source` would count (`close`):
    those its official clients finished (`finished`), at most as many as the host's records it
    brings (`carry_records`) but those infrastructure lost, whose count is the block's."""
    counted = [name for name, _ in _host_carried(source) if not _lost_slot(name)]
    return min(len(_closed(source)), len(counted))


def _closed(source: Path) -> list[dict[str, Any]]:
    """The episodes a block closed from `source` carries: those it finished before the first
    one infrastructure lost (a closed block runs nothing, so it cannot fill that gap)."""
    first = min((e["index"] for e in lost(source)), default=None)
    return [e for e in finished(source) if first is None or e["index"] < first]


def _lost_slot(name: str) -> bool:
    """Whether a host slot holds episodes infrastructure lost, carried as history."""
    return f".{LOST}-" in name


def close(run: P3Run, source: Path) -> P3Result:
    """Close block `run.out_dir` from `source`, a set-aside run of it, without running anything:
    its finished episodes are carried as a resumed run carries them (`carried.json`, the host's
    records and videos, `carry_records`), `closed.json` records where from, and `result.json` is
    a block cut on purpose (`truncated`, no official rate) whose episodes are every one carried.
    For a block whose set-aside run finished at least the episodes a launch asks for
    (`blocks.close_blocks`)."""
    out = _absolute(runs_path(run.out_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: each run needs a directory of its own")
    manifest = read_manifest(_absolute(run.manifest))
    task_config = P3_TASK_CONFIGS[run.mode].stem
    carried = _closed(source)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "source": str(source),
        "task": run.task,
        "task_config": task_config,
        "st_seed": 100000 * (1 + run.seed),
        "episodes": carried,
    }
    write_atomic(out / CARRIED, (json.dumps(record, indent=1) + "\n").encode())
    if run.profile is not None:
        write_atomic(out / PROFILE, (json.dumps(run.profile, indent=2) + "\n").encode())
    official = out / OFFICIAL
    for shard in range(run.shards) if run.shards > 1 else [None]:
        (official if shard is None else official / f"shard{shard}").mkdir(parents=True)
    carry_records(source, out, official, carried, run.shards)
    closed = {"from": str(source), "finished": len(carried), "wanted": run.max_episodes}
    write_atomic(out / CLOSED, (json.dumps(closed, indent=1) + "\n").encode())
    log.info("%s: block closed from %s with %d episodes", run.task, source, len(carried))
    return _collect(run, task_config, manifest.id, official, True, None, _Launched(out))


def _finished_per_client(block: Path) -> dict[int, int]:
    """How many episodes each client of a run ended itself, those infrastructure lost included
    (carried ones not counted), by shard (0 for an unsharded client): its host slot's ended
    episodes in order (`_host_carried`)."""
    return {shard: len(indices) for shard, indices in _own_indices(block).items()}


def _own_indices(block: Path) -> dict[int, list[int]]:
    """The episode indices each client of a run ended itself, in the order it ended them (those
    infrastructure lost included, carried ones not), by shard (0 for an unsharded client)."""
    if not sorted((block / SHARDS).glob("shard*.jsonl")):
        if _carried_record(block):  # a block closed from another run: its client ran none
            return {0: []}
        done, gone = _ended(block)  # the official client's own log, one by one
        return {0: sorted(e["index"] for e in [*done, *gone])}
    shards, cut = _agreed(block)
    return {
        int(lines[0]["shard"]): [
            int(line["index"])
            for line in lines[1:]
            if line.get("evaluated") and not line.get("carried") and int(line["index"]) < cut
        ]
        for lines in shards
    }


def _agreed(block: Path) -> tuple[list[list[dict[str, Any]]], float]:
    """The shard clients' records of a run (those begun) and the index up to which they found
    the same solvable seeds (infinite when they never disagreed). Each shard runs the expert check
    on every seed itself, and on some tasks the check does not repeat (it depends on what the
    client ran before: an episode or another check), so the shards' sequences can split; each
    shard's evaluated episodes are then of its own sequence, and those from the split on are not
    one official block's. A seed or an index the shards saw differently splits them at the
    smallest index involved; where only one shard got, its sequence stands."""
    shards = [
        lines
        for path in sorted((block / SHARDS).glob("shard*.jsonl"))
        if (lines := _shard_record(path))
    ]
    by_seed: dict[int, set[int | None]] = {}
    by_index: dict[int, set[int]] = {}
    for lines in shards:
        for line in lines[1:]:
            index = None if line.get("index") is None else int(line["index"])
            by_seed.setdefault(int(line["seed"]), set()).add(index)
            if index is not None:
                by_index.setdefault(index, set()).add(int(line["seed"]))
    split = [i for i, seeds in by_index.items() if len(seeds) > 1]
    split += [i for seen in by_seed.values() if len(seen) > 1 for i in seen if i is not None]
    cut: float = min(split, default=float("inf"))
    ended = {
        int(line["index"])
        for lines in shards
        for line in lines[1:]
        if line.get("evaluated") or (line.get("rerun") and line.get("error") is not None)
    }
    hole = next(i for i in range(len(ended) + 1) if i not in ended)
    if hole < cut and any(i > hole for i in ended):  # one shard stopped behind another
        log.warning(
            "%s: episode %d did not end, so the episodes after it are not carried: a resumed run "
            "replays the expert's verdicts up to the last carried seed and cannot leave a hole",
            block,
            hole,
        )
        cut = hole
    if split and cut == min(split):
        dropped = sorted(
            int(line["index"])
            for lines in shards
            for line in lines[1:]
            if line.get("evaluated") and int(line["index"]) >= cut
        )
        log.warning(
            "%s: the shards' expert checks found different solvable seeds from episode %d on; "
            "the episodes from there (%s) are not finished",
            block,
            cut,
            dropped,
        )
    return shards, cut


def _link(source: Path | str, target: Path | str) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _shard_record(path: Path) -> list[dict[str, Any]]:
    """A shard client's record: its header, then a line per seed checked (none: not begun)."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _absent(directory: Path, top: Path) -> list[Path]:
    """`directory` and its parents down from `top` that do not exist (yet)."""
    chain = [d for d in (directory, *directory.parents) if d == top or top in d.parents]
    return [d for d in chain if not d.exists()]


def _made_path(root: Path, task: str) -> Path:
    """Beside the task's lock: the directories our clients made that are still there."""
    return _lock_path(root, task).with_suffix(".made.json")


def _record_made(root: Path, task: str, directories: list[Path]) -> None:
    """Add `directories` to the record (under the task's lock)."""
    path = _made_path(root, task)
    made = set(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else set()
    made |= {str(directory) for directory in directories}
    write_atomic(path, json.dumps(sorted(made), indent=1).encode())


def _remove_made(root: Path, task: str) -> None:
    """Remove the recorded directories that are empty now (or gone), deepest first, and keep the
    rest recorded (under the task's lock)."""
    path = _made_path(root, task)
    if not path.is_file():
        return
    kept = []
    for directory in sorted(
        map(Path, json.loads(path.read_text(encoding="utf-8"))), key=lambda d: -len(d.parts)
    ):
        try:
            directory.rmdir()
        except FileNotFoundError:  # never made (a client that failed first), or removed
            continue
        except OSError as error:
            if error.errno != errno.ENOTEMPTY:
                raise
            kept.append(str(directory))
    if kept:
        write_atomic(path, json.dumps(sorted(kept), indent=1).encode())
    else:
        path.unlink()


def _collect(
    run: P3Run,
    task_config: str,
    manifest: ManifestId,
    official: Path | None,
    truncated: bool,
    client_exit: int | None,
    launched: _Launched,
    skillsets: dict[str, dict[str, str]] | None = None,
    brain: dict[str, Any] | None = None,
) -> P3Result:
    rate = None
    if official is not None and (official / "_result.txt").is_file() and not truncated:
        rate = float(
            (official / "_result.txt").read_text(encoding="utf-8").split()[-1]
        )  # successes / 100
    episodes = [
        json.loads(path.read_text(encoding="utf-8")) for path in host_summaries(launched.out)
    ]
    history = [
        json.loads(path.read_text(encoding="utf-8")) for path in host_summaries(launched.out, True)
    ]
    outcomes_error = None
    try:
        done, gone = _ended(launched.out)
    except (OSError, ValueError, KeyError, TypeError) as error:  # result.json is written anyway
        log.warning("%s: no per-episode outcomes (%s)", launched.out, error)
        done, gone = [], []
        outcomes_error = f"{type(error).__name__}: {error}"
    outcomes = sorted([*done, *({**e, "lost": True} for e in gone)], key=lambda e: e["index"])
    failed = launched.out / "host" / CONFIG_ERROR
    end = json.loads(failed.read_text(encoding="utf-8"))["end"] if failed.is_file() else None
    result = P3Result(
        task=run.task,
        seed=run.seed,
        task_config=task_config,
        manifest=manifest,
        official_dir=official,
        success_rate=rate,
        episodes=episodes,
        infra_episodes=sum(1 for episode in episodes if infra_lost(episode)),
        truncated=truncated,
        config_error=None if end is None else f"{end['reason']}: {end['note']}",
        client_exit=client_exit,
        pids=launched.pids,
        shards=run.shards,
        outcomes=outcomes,
        outcomes_error=outcomes_error,
        lost_episodes=history,
        skillsets=skillsets or {},
        brain=brain,
    )
    write_atomic(launched.out / "result.json", (result.model_dump_json(indent=2) + "\n").encode())
    return result


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def _tail(path: Path, lines: int = 20) -> str:
    return "\n".join(path.read_text(errors="replace", encoding="utf-8").splitlines()[-lines:])


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path
