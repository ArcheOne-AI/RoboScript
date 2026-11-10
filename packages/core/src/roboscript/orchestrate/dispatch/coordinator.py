"""The coordinator: assigns the tasks to the workers once, starts each worker's unit (one pipeline
launch over its tasks and GPUs), watches them, moves each finished task's directory and the blobs
it names into its own run root (copy, verify by manifest, then delete on the worker, keeping only
tombstones), resumes a unit whose process ended with tasks unfinished, hands a lost worker's
tasks to the others, drains a finished unit (its records and remaining blobs, then its directory)
and writes one status across the workers.

    <root>/<name>/  dispatch.yaml   the config it runs (as given)
                    pipeline.yaml   the pipeline config with every task (for the pipeline's own
                                    summary over this directory)
                    state.json      assignment, units, tasks, workers (`DispatchState`)
                    status.{json,md}, events.jsonl, coordinator.json (its pid)
                    <task>/         a moved task: the one copy of its records
                    blobs/          the blobs the moved tasks name
                    .units/<unit>/  a drained unit's own records (launches, logs, its config)
                    .checkpoints/<task>/  an unfinished task's last copy (`checkpoint: true`)
                    .incoming/      a move in progress
"""

import contextlib
import json
import os
import shutil
import signal
import time
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

import anyio
import yaml
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.dispatch import (
    AgentReply,
    DigestRequest,
    Manifest,
    ManifestRequest,
    PurgeRequest,
    RemoveReply,
    RemoveRequest,
    StartReply,
    StartRequest,
    StatusReply,
    StatusRequest,
    StopReply,
    StopRequest,
    UnitPaths,
)
from roboscript.contracts.ids import TaskName
from roboscript.orchestrate.dispatch.agent import code_sha, sha_matches, start_token
from roboscript.orchestrate.dispatch.assign import load_policy
from roboscript.orchestrate.dispatch.config import (
    DispatchConfig,
    WorkerSpec,
    load_pipeline,
    local_keys,
    provenance,
    worker_config,
)
from roboscript.orchestrate.dispatch.files import (
    blob_path,
    named_blobs,
    path_digest,
    tree_files,
    verify,
    verify_blobs,
)
from roboscript.orchestrate.dispatch.transport import Transport, TransportError, open_transport
from roboscript.records import write_atomic
from roboscript.settings import repo_root

STATE: Final = "state.json"
AGENT: Final = "roboscript.orchestrate.dispatch.agent"
UnitStatus = Literal["new", "running", "waiting", "ended", "drained", "orphaned"]
TaskStatus = Literal["assigned", "moved", "stuck"]


class DispatchFailed(RuntimeError):
    """The coordinator cannot go on (another coordinator runs, wrong version)."""


class AgentRefused(RuntimeError):
    """A worker's agent refused or failed a request (`AgentReply.error`)."""


class UnitState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    worker: str
    tasks: list[TaskName]
    dir: str
    status: UnitStatus = "new"
    pid: int | None = None
    launches: int = 0
    relaunches: int = 0
    relaunch_at: float | None = None
    error: str | None = None


class TaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker: str
    unit: str
    status: TaskStatus = "assigned"
    rows: list[dict[str, Any]] = []
    checkpoint: str | None = None
    """The rows' statuses when the task was last checkpointed."""
    moved_at: str | None = None
    note: str | None = None


class WorkerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unreachable_since: float | None = None
    lost: bool = False
    error: str | None = None


class DispatchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha: str
    provenance: str
    pinned: dict[str, str]
    """Pinned path (task-expanded) -> digest, from the coordinator's code copy."""
    units: dict[str, UnitState]
    tasks: dict[TaskName, TaskState]
    workers: dict[str, WorkerState]
    started_at: str


def _utc(at: float | None = None) -> str:
    moment = datetime.fromtimestamp(time.time() if at is None else at, timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class Coordinator:
    """One dispatch, from its directory. `transports` opens a worker's transport (the config's
    by default); `code_root` is the coordinator's own code copy (the repository's), whose
    version must be the dispatch's and whose pinned files every worker must match."""

    def __init__(
        self,
        config: DispatchConfig,
        *,
        transports: Callable[[WorkerSpec], Transport] | None = None,
        code_root: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.directory = config.directory
        self.workers = {w.name: w for w in config.workers}
        self._open = transports or (lambda w: open_transport(w.transport, w.host))
        self._transports: dict[str, Transport] = {}
        self.code_root = code_root or repo_root()
        self.clock = clock
        self.pipeline = load_pipeline(config)
        self.provenance = provenance(self.pipeline, local_keys(config.unit))
        self._moves = anyio.CapacityLimiter(config.max_parallel_moves)
        self.state = self._load_or_create()

    # -- setup ---------------------------------------------------------------------------------

    def _load_or_create(self) -> DispatchState:
        sha, dirty = code_sha(self.code_root)
        if dirty or not sha_matches(sha, self.config.sha):
            raise DispatchFailed(
                f"the coordinator's code {self.code_root} is at {sha}{' (dirty)' if dirty else ''}"
                f", not the dispatch's {self.config.sha}"
            )
        path = self.directory / STATE
        if path.is_file():
            state = DispatchState.model_validate_json(path.read_bytes())
            if state.provenance != self.provenance or not sha_matches(state.sha, self.config.sha):
                raise DispatchFailed(f"{self.directory} runs another version or pipeline config")
            return state
        self.directory.mkdir(parents=True, exist_ok=True)
        dumped = yaml.safe_dump(self.pipeline, sort_keys=False).encode()
        write_atomic(self.directory / self.config.unit.config_file, dumped)
        config = self.config.model_dump(mode="json")
        write_atomic(
            self.directory / "dispatch.yaml", yaml.safe_dump(config, sort_keys=False).encode()
        )
        policy = load_policy(self.config.policy)
        assigned = policy.assign(self.config.tasks, self.config.cost, self.config.workers)
        units: dict[str, UnitState] = {}
        tasks: dict[TaskName, TaskState] = {}
        for worker in self.config.workers:
            if not assigned[worker.name]:
                continue
            unit = self._new_unit(worker, assigned[worker.name], units)
            units[unit.name] = unit
            for task in unit.tasks:
                tasks[task] = TaskState(worker=worker.name, unit=unit.name)
        seeds = self.config.seeds
        if seeds is not None and (missing := [t for t in tasks if not (seeds / t).is_dir()]):
            raise DispatchFailed(f"{seeds} holds no prepared directory of {missing}")
        pinned = {}
        for spec in self.config.unit.pinned:
            for name in _expand(spec.path, self.config.tasks):
                digest = path_digest(self.code_root / name)
                if digest is None:
                    raise DispatchFailed(f"pinned {name} is missing in {self.code_root}")
                pinned[name] = digest
        state = DispatchState(
            sha=self.config.sha,
            provenance=self.provenance,
            pinned=pinned,
            units=units,
            tasks=tasks,
            workers={w.name: WorkerState() for w in self.config.workers},
            started_at=_utc(),
        )
        self.state = state
        self._save()
        self._event("assigned", assignment={u.name: u.tasks for u in units.values()})
        return state

    def _new_unit(
        self, worker: WorkerSpec, tasks: list[TaskName], units: Mapping[str, UnitState]
    ) -> UnitState:
        n = 1 + sum(u.worker == worker.name for u in units.values())
        name = f"{worker.name}-u{n}"
        unit_dir = f"{worker.run_root}/{self.config.name}/{name}"
        return UnitState(name=name, worker=worker.name, tasks=list(tasks), dir=unit_dir)

    def _save(self) -> None:
        write_atomic(self.directory / STATE, (self.state.model_dump_json(indent=1) + "\n").encode())

    def _event(self, kind: str, **fields: Any) -> None:
        line = canonical_json({"at": _utc(), "event": kind, **fields}) + "\n"
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(line)

    # -- the agent -------------------------------------------------------------------------------

    def _transport(self, worker: WorkerSpec) -> Transport:
        if worker.name not in self._transports:
            self._transports[worker.name] = self._open(worker)
        return self._transports[worker.name]

    async def _call(self, worker: WorkerSpec, command: str, request: BaseModel) -> dict[str, Any]:
        found = await self._transport(worker).run(
            [worker.python, "-m", AGENT, command],
            cwd=worker.code_dir,
            input=request.model_dump_json().encode(),
            timeout_s=self.config.command_timeout_s,
        )
        try:
            reply = AgentReply.model_validate_json(found.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as error:
            stderr = found.stderr.decode(errors="replace").strip()[-2000:]
            raise AgentRefused(
                f"{worker.name} {command}: no reply (exit {found.returncode}): {stderr}"
            ) from error
        if not reply.ok:
            raise AgentRefused(f"{worker.name} {command}: {reply.kind}: {reply.error}")
        return reply.reply or {}

    def _paths(self, unit: UnitState) -> UnitPaths:
        return UnitPaths(run_root=self.workers[unit.worker].run_root, unit_dir=unit.dir)

    # -- the loop ----------------------------------------------------------------------------------

    def done(self) -> bool:
        settled = all(t.status != "assigned" for t in self.state.tasks.values())
        units = all(
            u.status in ("drained", "orphaned") or (u.status == "ended" and u.error)
            for u in self.state.units.values()
        )
        return settled and units

    async def run(self) -> dict[str, Any]:
        """Rounds until every task moved (or is stuck); the status of the last."""
        self._claim()
        try:
            while True:
                await self.step()
                if self.done():
                    await self._summary()
                    return self.write_status()
                await anyio.sleep(self.config.poll_s)
        finally:
            (self.directory / "coordinator.json").unlink(missing_ok=True)

    def _claim(self) -> None:
        path = self.directory / "coordinator.json"
        if path.is_file():
            other = json.loads(path.read_bytes())
            if start_token(other["pid"]) == other["token"]:
                raise DispatchFailed(f"coordinator pid {other['pid']} runs {self.directory}")
        token = start_token(os.getpid())
        write_atomic(path, json.dumps({"pid": os.getpid(), "token": token}).encode())

    async def step(self) -> dict[str, Any]:
        """One round over every worker (at once), then the status."""
        async with anyio.create_task_group() as group:
            for worker in self.config.workers:
                group.start_soon(self._round, worker)
        return self.write_status()

    async def _round(self, worker: WorkerSpec) -> None:
        mine = self.state.workers[worker.name]
        try:
            for unit in [u for u in self.state.units.values() if u.worker == worker.name]:
                try:
                    await self._unit_round(worker, unit)
                except TransportError:
                    raise
                except (AgentRefused, RuntimeError) as error:  # reached, but this round failed
                    unit.error = str(error)  # (a copy whose file vanished, ...): the next retries
                    self._event("agent_error", unit=unit.name, error=str(error))
        except TransportError as error:
            now = self.clock()
            if mine.unreachable_since is None:
                mine.unreachable_since = now
                self._event("unreachable", worker=worker.name, error=str(error))
            mine.error = str(error)
            if not mine.lost and now - mine.unreachable_since >= self.config.lost_after_s:
                await self._lose(worker)
            self._save()
            return
        if mine.unreachable_since is not None or mine.error:
            self._event("reachable", worker=worker.name)
        mine.unreachable_since, mine.error = None, None
        self._save()

    async def _unit_round(self, worker: WorkerSpec, unit: UnitState) -> None:
        if unit.status == "drained" or (unit.status == "ended" and unit.error):
            return  # drained, or refused or given up on: its tasks are stuck, left for a human
        if unit.status == "orphaned":  # its worker was lost and is back: stop what still runs
            reply = StopReply.model_validate(
                await self._call(
                    worker,
                    "stop",
                    StopRequest(unit=self._paths(unit), grace_s=self.config.unit.stop_grace_s),
                )
            )
            if reply.was_alive:
                self._event("stopped_orphan", unit=unit.name, pid=reply.pid)
            return
        if unit.status == "new" and not await self._start(worker, unit):
            return
        status = StatusReply.model_validate(
            await self._call(
                worker,
                "status",
                StatusRequest(unit=self._paths(unit), status_files=self.config.unit.status_files),
            )
        )
        unit.pid, unit.error = status.pid, None
        rows = self._rows_by_task(status.rows)
        for task in unit.tasks:
            state = self.state.tasks[task]
            if state.status != "assigned" or state.unit != unit.name:
                continue
            if task in rows:
                state.rows = rows[task]
            if state.rows and all(r.get("status") in self.config.unit.terminal for r in state.rows):
                await self._move(worker, unit, task)
            elif self.config.checkpoint and self._checkpoint_due(state):
                await self._checkpoint(worker, unit, task)
        open_tasks = [t for t in unit.tasks if self._open_here(t, unit)]
        if status.alive:
            unit.status, unit.relaunch_at = "running", None
        elif open_tasks:
            await self._after_exit(worker, unit, open_tasks)
        else:
            await self._drain(worker, unit)
        self._save()

    def _open_here(self, task: TaskName, unit: UnitState) -> bool:
        state = self.state.tasks[task]
        return state.unit == unit.name and state.status == "assigned"

    def _rows_by_task(self, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        found: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            task = str(row.get("run", "")).split("/", 1)[0]
            found.setdefault(task, []).append(row)
        return found

    async def _start(self, worker: WorkerSpec, unit: UnitState) -> bool:
        """Start or resume the unit (its tasks' prepared directories pushed before its first
        launch, `DispatchConfig.seeds`); False when the worker refused it (its tasks are
        stuck)."""
        await self._ship(worker, unit)
        if self.config.seeds is not None and unit.launches == 0:
            await self._push_seeds(worker, unit, self.config.seeds)
        config = worker_config(self.pipeline, self.config.unit, worker, unit.tasks)
        argv = [part.replace("{python}", worker.python) for part in self.config.unit.argv]
        request = StartRequest(
            unit=self._paths(unit),
            sha=self.config.sha,
            provenance=self.provenance,
            config=config,
            config_file=self.config.unit.config_file,
            local_keys=local_keys(self.config.unit),
            argv=argv,
            env={**self.config.unit.env, **worker.env},
            pinned=self._pinned(unit.tasks),
            gateway_url=worker.gateway_url,
            tasks=list(unit.tasks),
        )
        try:
            reply = StartReply.model_validate(await self._call(worker, "start", request))
        except AgentRefused as error:
            unit.status, unit.error = "ended", str(error)
            for task in unit.tasks:
                if self._open_here(task, unit):
                    self.state.tasks[task].status = "stuck"
                    self.state.tasks[task].note = "the worker refused the unit"
            self._event("refused", unit=unit.name, error=str(error))
            self._save()
            return False
        unit.pid, unit.launches, unit.status, unit.error = (
            reply.pid,
            reply.launches,
            "running",
            None,
        )
        self._event(
            "started", unit=unit.name, pid=reply.pid, launches=reply.launches, started=reply.started
        )
        self._save()
        return True

    async def _push_seeds(self, worker: WorkerSpec, unit: UnitState, seeds: Path) -> None:
        """Each of the unit's tasks' prepared directory, into the unit's (again after a push cut
        short: the same files)."""
        transport, timeout = self._transport(worker), self.config.command_timeout_s
        for task in unit.tasks:
            source = seeds / task
            files = tree_files(source)
            await transport.push(source, f"{unit.dir}/{task}", files, timeout_s=timeout)
            self._event("seeded", unit=unit.name, task=task, files=len(files), source=str(seeds))

    def _pinned(self, tasks: list[TaskName]) -> dict[str, str]:
        names = {n for spec in self.config.unit.pinned for n in _expand(spec.path, tasks)}
        return {name: self.state.pinned[name] for name in sorted(names)}

    async def _ship(self, worker: WorkerSpec, unit: UnitState) -> None:
        shipped = []
        for spec in self.config.unit.pinned:
            if not spec.ship:
                continue
            names = _expand(spec.path, unit.tasks)
            found = await self._call(worker, "digest", DigestRequest(paths=names))
            for name in names:
                if found[name] == self.state.pinned[name]:
                    continue
                source = self.code_root / name
                if source.is_dir():
                    await self._transport(worker).push(
                        source,
                        f"{worker.code_dir}/{name}",
                        tree_files(source),
                        timeout_s=self.config.command_timeout_s,
                    )
                else:
                    await self._transport(worker).push(
                        source.parent,
                        f"{worker.code_dir}/{Path(name).parent}",
                        [source.name],
                        timeout_s=self.config.command_timeout_s,
                    )
                shipped.append(name)
        if shipped:
            self._event("shipped", worker=worker.name, paths=shipped)

    async def _after_exit(
        self, worker: WorkerSpec, unit: UnitState, open_tasks: list[TaskName]
    ) -> None:
        waits = self.config.relaunch_waits_s
        now = self.clock()
        if unit.status == "running":
            unit.status = "waiting"
            if unit.relaunches >= len(waits):
                unit.status, unit.error = (
                    "ended",
                    f"ended {len(waits) + 1} times with tasks unfinished",
                )
                for task in open_tasks:
                    self.state.tasks[task].status = "stuck"
                    self.state.tasks[
                        task
                    ].note = "its unit ended with it unfinished: left on the worker"
                self._event("stuck", unit=unit.name, tasks=open_tasks)
                return
            unit.relaunch_at = now + waits[unit.relaunches]
            self._event(
                "exited", unit=unit.name, open=open_tasks, relaunch_in_s=waits[unit.relaunches]
            )
            return
        if unit.status == "waiting" and unit.relaunch_at is not None and now >= unit.relaunch_at:
            unit.relaunches += 1
            unit.relaunch_at = None
            await self._start(worker, unit)

    # -- moving --------------------------------------------------------------------------------

    def _blobs(self) -> Path:
        return self.directory / (self.config.unit.blobs or "blobs")

    async def _fetch(
        self, worker: WorkerSpec, unit: UnitState, task: TaskName, target: Path
    ) -> Manifest:
        """Copy a task's directory into `target` (fresh) and the blobs it names into the run
        root's store, both verified; its manifest."""
        found = Manifest.model_validate(
            await self._call(
                worker,
                "manifest",
                ManifestRequest(unit=self._paths(unit), path=task, blobs=self.config.unit.blobs),
            )
        )
        await anyio.to_thread.run_sync(shutil.rmtree, target, True)
        transport, timeout = self._transport(worker), self.config.command_timeout_s
        await transport.pull(f"{unit.dir}/{task}", target, list(found.files), timeout_s=timeout)
        if wrong := await anyio.to_thread.run_sync(verify, target, found):
            raise AgentRefused(f"{task}: the copy does not match its manifest: {wrong[:5]}")
        blobs = self._blobs()
        missing = [b for b in found.blobs if not blob_path(blobs, b).is_file()]
        if missing and self.config.unit.blobs is not None:
            names = [f"{b[:2]}/{b}" for b in missing]
            await transport.pull(
                f"{unit.dir}/{self.config.unit.blobs}", blobs, names, timeout_s=timeout
            )
            if wrong := await anyio.to_thread.run_sync(verify_blobs, blobs, missing):
                raise AgentRefused(f"{task}: blobs did not copy intact: {wrong[:5]}")
        return found

    async def _move(self, worker: WorkerSpec, unit: UnitState, task: TaskName) -> None:
        incoming = self.directory / ".incoming" / task
        marker = incoming.with_name(f"{task}.manifest.json")
        target = self.directory / task
        if target.exists():
            raise DispatchFailed(f"{target} exists already: a task moves once")
        if marker.is_file():  # removed on the worker before a crash: finish from the copy
            found = Manifest.model_validate_json(marker.read_bytes())
            if incoming.is_dir() and not verify(incoming, found):
                self._finish_move(unit, task, incoming, marker, found, None)
                return
            marker.unlink()
        async with self._moves:
            try:
                found = await self._fetch(worker, unit, task, incoming)
                write_atomic(marker, found.model_dump_json().encode())
                removed = RemoveReply.model_validate(
                    await self._call(
                        worker,
                        "remove",
                        RemoveRequest(
                            unit=self._paths(unit),
                            task=task,
                            digest=found.digest,
                            keep=self.config.unit.tombstones,
                            blobs=self.config.unit.blobs,
                            blob_min_bytes=self.config.unit.blob_min_bytes,
                        ),
                    )
                )
            except TransportError:
                raise
            except (AgentRefused, RuntimeError) as error:  # changed or vanished while copied, or
                marker.unlink(missing_ok=True)  # a bad copy: next round
                self._event("move_retry", task=task, error=str(error))
                return
        self._finish_move(unit, task, incoming, marker, found, removed)

    def _finish_move(
        self,
        unit: UnitState,
        task: TaskName,
        incoming: Path,
        marker: Path,
        found: Manifest,
        removed: RemoveReply | None,
    ) -> None:
        os.replace(incoming, self.directory / task)
        marker.unlink()
        with contextlib.suppress(OSError):  # the last move in progress: no leftover directory
            incoming.parent.rmdir()
        shutil.rmtree(self.directory / ".checkpoints" / task, ignore_errors=True)
        state = self.state.tasks[task]
        state.status, state.moved_at = "moved", _utc()
        self._event(
            "moved",
            task=task,
            unit=unit.name,
            files=len(found.files),
            blobs=len(found.blobs),
            digest=found.digest,
            removed=None if removed is None else removed.model_dump(),
        )
        self._save()

    def _checkpoint_due(self, state: TaskState) -> bool:
        key = canonical_json(
            sorted(str(r.get("status")) + ":" + str(r.get("stage")) for r in state.rows)
        )
        ended = any(r.get("status") in self.config.unit.terminal for r in state.rows)
        return ended and key != state.checkpoint

    async def _checkpoint(self, worker: WorkerSpec, unit: UnitState, task: TaskName) -> None:
        state = self.state.tasks[task]
        target = self.directory / ".checkpoints" / task
        try:
            async with self._moves:
                found = await self._fetch(worker, unit, task, target.with_name(f"{task}.new"))
        except TransportError:
            raise
        except (AgentRefused, RuntimeError) as error:
            self._event("checkpoint_retry", task=task, error=str(error))
            return
        if target.exists():
            shutil.rmtree(target)
        os.replace(target.with_name(f"{task}.new"), target)
        state.checkpoint = canonical_json(
            sorted(str(r.get("status")) + ":" + str(r.get("stage")) for r in state.rows)
        )
        self._event("checkpoint", task=task, files=len(found.files))

    async def _drain(self, worker: WorkerSpec, unit: UnitState) -> None:
        """A unit whose process ended and whose tasks all moved: its own records to `.units/`,
        its remaining blobs to the store (both verified), then its directory goes."""
        transport, timeout = self._transport(worker), self.config.command_timeout_s
        paths = self._paths(unit)
        blobs = self.config.unit.blobs
        exclude = [*unit.tasks, *([blobs] if blobs else [])]
        records = Manifest.model_validate(
            await self._call(worker, "manifest", ManifestRequest(unit=paths, exclude=exclude))
        )
        target = self.directory / ".units" / unit.name
        await transport.pull(unit.dir, target, list(records.files), timeout_s=timeout)
        if wrong := verify(target, records):
            raise AgentRefused(f"{unit.name}: records did not copy intact: {wrong[:5]}")
        if blobs:
            left = Manifest.model_validate(
                await self._call(worker, "manifest", ManifestRequest(unit=paths, path=blobs))
            )
            store = self._blobs()
            names = [n for n in left.files if not (store / n).is_file()]
            await transport.pull(f"{unit.dir}/{blobs}", store, names, timeout_s=timeout)
            if wrong := verify_blobs(store, [Path(n).name for n in names]):
                raise AgentRefused(f"{unit.name}: blobs did not copy intact: {wrong[:5]}")
        await self._call(worker, "purge", PurgeRequest(unit=paths))
        unit.status = "drained"
        self._event("drained", unit=unit.name, records=len(records.files))

    # -- lost workers -----------------------------------------------------------------------------

    async def _lose(self, worker: WorkerSpec) -> None:
        """Hand the worker's unfinished tasks to the live workers (from their checkpoints, else
        afresh), each target's share as a new unit; the worker's units are orphaned (stopped by
        exact pid if it comes back, their directories left for a human)."""
        mine = self.state.workers[worker.name]
        mine.lost = True
        live = [w for w in self.config.workers if not self.state.workers[w.name].lost]
        loads: dict[str, float] = {}
        for task, state in self.state.tasks.items():
            if state.status == "assigned":
                loads[state.worker] = loads.get(state.worker, 0.0) + self.config.cost(task)
        policy = load_policy(self.config.policy)
        moved: dict[str, list[TaskName]] = {}
        for unit in [u for u in self.state.units.values() if u.worker == worker.name]:
            if unit.status == "drained":
                continue
            unit.status = "orphaned"
            for task in unit.tasks:
                if not self._open_here(task, unit):
                    continue
                if not live:
                    self.state.tasks[task].status = "stuck"
                    self.state.tasks[task].note = "its worker was lost and no other is live"
                    continue
                target = policy.reassign(task, self.config.cost, live, loads)
                loads[target] = loads.get(target, 0.0) + self.config.cost(task)
                moved.setdefault(target, []).append(task)
        for target, tasks in moved.items():
            unit = self._new_unit(self.workers[target], tasks, self.state.units)
            self.state.units[unit.name] = unit
            for task in tasks:
                state = self.state.tasks[task]
                state.worker, state.unit, state.rows = target, unit.name, []
            self._event("reassigned", lost=worker.name, unit=unit.name, tasks=tasks)
        self._save()
        for target, tasks in moved.items():
            unit = next(
                u
                for u in self.state.units.values()
                if u.worker == target and u.status == "new" and u.tasks == tasks
            )
            try:
                await self._seed(self.workers[target], unit)
            except TransportError as error:  # the target's own round reports it
                self._event("seed_failed", unit=unit.name, error=str(error))

    async def _seed(self, worker: WorkerSpec, unit: UnitState) -> None:
        """Push the checkpoints of a new unit's tasks (and the blobs they name) to it."""
        transport, timeout = self._transport(worker), self.config.command_timeout_s
        for task in unit.tasks:
            source = self.directory / ".checkpoints" / task
            if not source.is_dir():
                continue
            files = [
                f
                for f in tree_files(source)
                if not any(Path(f).match(p) for p in self.config.unit.strip)
            ]
            await transport.push(source, f"{unit.dir}/{task}", files, timeout_s=timeout)
            if self.config.unit.blobs:
                names = sorted(named_blobs(source, files, self._blobs()))
                await transport.push(
                    self._blobs(),
                    f"{unit.dir}/{self.config.unit.blobs}",
                    [f"{b[:2]}/{b}" for b in names],
                    timeout_s=timeout,
                )
            self._event("seeded", unit=unit.name, task=task, files=len(files))

    # -- status ----------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The pipeline's status format across workers: every job's latest row (tagged with its
        worker and unit) and the counts by status, with the workers and tasks."""
        rows, workers = [], {}
        for state in self.state.tasks.values():
            for row in state.rows:
                rows.append(
                    {**row, "worker": state.worker, "unit": state.unit, "task_state": state.status}
                )
        for name, worker in self.workers.items():
            mine = self.state.workers[name]
            units = [u for u in self.state.units.values() if u.worker == name]
            workers[name] = {
                "host": worker.host,
                "gpus": worker.gpus,
                "lost": mine.lost,
                "unreachable_since": None
                if mine.unreachable_since is None
                else _utc(mine.unreachable_since),
                "error": mine.error,
                "units": [
                    u.model_dump(
                        include={"name", "status", "pid", "launches", "relaunches", "error"}
                    )
                    for u in units
                ],
                "tasks": dict(
                    Counter(t.status for t in self.state.tasks.values() if t.worker == name)
                ),
            }
        alerts = [f"worker {n} lost" for n, w in self.state.workers.items() if w.lost]
        alerts += [
            f"{t}: stuck ({s.note})" for t, s in self.state.tasks.items() if s.status == "stuck"
        ]
        return {
            "directory": str(self.directory),
            "updated": _utc(),
            "runs": dict(Counter(row.get("status") for row in rows)),
            "tasks": dict(Counter(t.status for t in self.state.tasks.values())),
            "workers": workers,
            "alerts": alerts,
            "rows": rows,
        }

    def write_status(self) -> dict[str, Any]:
        found = self.status()
        write_atomic(self.directory / "status.json", (json.dumps(found, indent=1) + "\n").encode())
        write_atomic(self.directory / "status.md", status_table(found).encode())
        return found

    async def _summary(self) -> None:
        argv = self.config.unit.summary_argv
        if argv is None or not any(t.status == "moved" for t in self.state.tasks.values()):
            return
        command = [part.replace("{dir}", str(self.directory)) for part in argv]
        done = await anyio.run_process(
            command, cwd=self.code_root, check=False, env={**os.environ, **self.config.unit.env}
        )
        self._event(
            "summary", exit=done.returncode, stderr=done.stderr.decode(errors="replace")[-2000:]
        )

    async def stop_all(self) -> list[dict[str, Any]]:
        """Stop the running coordinator of this dispatch (else it would resume what stops), then
        every unit (each by its exact pid); the dispatch stays resumable."""
        path = self.directory / "coordinator.json"
        if path.is_file():
            other = json.loads(path.read_bytes())
            if other["pid"] != os.getpid() and start_token(other["pid"]) == other["token"]:
                os.kill(other["pid"], signal.SIGTERM)
                for _ in range(300):  # a minute at most
                    if start_token(other["pid"]) != other["token"]:
                        break
                    await anyio.sleep(0.2)
                else:
                    raise DispatchFailed(f"coordinator pid {other['pid']} did not stop")
                self._event("stopped_coordinator", pid=other["pid"])
            path.unlink(missing_ok=True)
        stopped = []
        for unit in self.state.units.values():
            if unit.status in ("drained", "new"):
                continue
            worker = self.workers[unit.worker]
            reply = StopReply.model_validate(
                await self._call(
                    worker,
                    "stop",
                    StopRequest(unit=self._paths(unit), grace_s=self.config.unit.stop_grace_s),
                )
            )
            stopped.append({"unit": unit.name, **reply.model_dump()})
            if reply.was_alive and unit.status in ("running", "waiting"):
                unit.status = "new"  # a resume starts it again without using up a relaunch
        self._event("stopped", units=stopped)
        self._save()
        return stopped


def _expand(path: str, tasks: list[TaskName]) -> list[str]:
    """A pinned path for each task when it names `{task}`, else itself."""
    return [path.replace("{task}", t) for t in tasks] if "{task}" in path else [path]


def status_table(found: Mapping[str, Any]) -> str:
    counts = ", ".join(
        f"{n} {s}" for s, n in sorted(found["runs"].items(), key=lambda i: str(i[0]))
    )
    tasks = ", ".join(f"{n} {s}" for s, n in sorted(found["tasks"].items()))
    lines = [
        f"# Dispatch {found['directory']}",
        "",
        f"Updated {found['updated']}: tasks {tasks}; jobs {counts}.",
        "",
    ]
    lines += [f"**ALERT: {alert}**" for alert in found["alerts"]]
    lines += ["", "| worker | host | GPUs | units | tasks |", "|---|---|---|---|---|"]
    for name, worker in found["workers"].items():
        units = "; ".join(
            f"{u['name']} {u['status']} (pid {u['pid']}, {u['launches']} launches)"
            for u in worker["units"]
        )
        tasks_here = ", ".join(f"{n} {s}" for s, n in sorted(worker["tasks"].items()))
        state = (
            " LOST" if worker["lost"] else (" unreachable" if worker["unreachable_since"] else "")
        )
        lines.append(
            f"| {name}{state} | {worker['host']} | {worker['gpus']} | {units} | {tasks_here} |"
        )
    lines += ["", "| job | worker | status | stage | GPU | task |", "|---|---|---|---|---|---|"]
    for row in found["rows"]:
        gpu = "-" if row.get("gpu") is None else row.get("gpu")
        stage = row.get("stage") or "-"
        lines.append(
            f"| {row.get('run')} | {row['worker']} | {row.get('status')} | {stage} | {gpu} | "
            f"{row['task_state']} |"
        )
    return "\n".join(lines) + "\n"
