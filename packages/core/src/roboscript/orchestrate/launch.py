"""One launch of a run's jobs (docs/concepts/operations.md, `orchestrate`): the GPUs'
services, the jobs' workers placed on the GPUs (`Slots`), respawned while infrastructure fails them,
watched (heartbeat), preempted when spare units hold room others wait for, and the status. What
is a pipeline's own (its job graph, worker command, records' layout, status) it supplies by
subclassing `Launch`, and its config by `LaunchSettings`."""

import abc
import contextlib
import json
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

import anyio
import anyio.abc
import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import JobSpec
from roboscript.learning.episode import set_aside
from roboscript.orchestrate.graph import one_line
from roboscript.orchestrate.jobs import HEARTBEAT, PIDS, PidRecord, RunKey, RunState
from roboscript.orchestrate.slots import MIB_PER_GB, PEAK_LEARN_S, Placement, Reservation, Slots
from roboscript.records import write_atomic
from roboscript.settings import source_sha
from roboscript.stack import ServiceSpec, StackConfig, StackError, stall, stop_groups, up
from roboscript.stack.gpus import (
    CLAIMS_MINE,
    GpuProbe,
    LogMarks,
    ProcessTable,
    SharedClaims,
    nvidia_smi,
    ps,
)
from roboscript.stack.shared import (
    Lease,
    StackPlan,
    StackRegistry,
    StackSet,
    Unit,
    follow_set,
    plan_units,
    stack_key,
)

log = logging.getLogger(__name__)

WORKER: Final = ".roboscript/worker.json"
WORKER_STOP_S: Final = 120.0
"""How long a worker may take after SIGTERM to stop what it started (P3's processes, an env
server) and record its end, before it is killed."""
GROUP_STOP_S: Final = 10.0
DISK_POLL_S: Final = 60.0
STOPPED: Final = ("running", "stopped")
"""How a job the scheduler stopped may have ended: stopped by its worker, or not at all."""
WATCH_POLL_S: Final = 30.0
"""How often at most the scheduler looks at a worker's heartbeat."""
IDLE_POLL_S: Final = 10.0
"""How often at most a GPU's services look whether they were idle for `service_idle_s`."""
PUBLISH_S: Final = 20.0
"""How often a launch publishes its holds in the shared claims file (`Slots.publish`), well
within the `HOLD_TTL_S` its lines count for."""
PEAKS: Final = "gpu_peaks.json"
PREEMPT_POLL_S: Final = 10.0
"""How often the scheduler looks whether a job waits long for room that spare units hold."""
PRESSURE_WAIT_S: Final = 30.0
"""How long a job or service that ran out of GPU memory waits before it asks for room again: the
memory its processes held is freed by then, and the bursts that crowded it out have settled."""
READY_POLL_S: Final = 10.0
"""How often at most the launch asks a running GPU service's `/readyz` (`watch_ready`)."""
READY_TIMEOUT_S: Final = 5.0
"""How long at most a GPU service's `/readyz` may take to answer before it counts as not
answering (`watch_ready`)."""


@dataclass
class Worker:
    """A running job's worker process on its GPU, and why the scheduler stops it (None: it runs
    on)."""

    gpu: int
    process: anyio.abc.Process
    reservation: Reservation | None = None
    why: str | None = None
    preempted: bool = False
    wake: anyio.Event = field(default_factory=anyio.Event)

    def interrupt(self, why: str, preempted: bool = False) -> None:
        if self.why is None:
            self.why, self.preempted = why, preempted
            self.wake.set()


class LaunchSettings(BaseModel):
    """What a launch runs under, as the pipeline's config sets it: its GPUs and their memory
    (GB as nvidia-smi's MiB read: `MIB_PER_GB`), the services each GPU runs (`per_gpu`, each
    reserving `service_gb`, else `free_gb_per_place`), how jobs and services are retried, the
    watchdog, the disk's floor, the status's interval, whether spare units are preempted
    (`preempt`), and what a worker's PYTHONPATH starts with (`pythonpath`). A GPU service that
    came up and then has not been ready for `service_unready_s` straight is stopped and restarts
    as one that died does; one ready for `service_healthy_s` straight has its restarts back
    (`watch_ready`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpus: list[NonNegativeInt]
    gpu_cap_gb: PositiveFloat
    gpu_margin_gb: NonNegativeFloat
    gpu_claims: Path | None
    shared_stacks: Path | None
    free_gb_per_place: NonNegativeFloat
    peak_headroom_gb: NonNegativeFloat
    service_gb: dict[str, NonNegativeFloat]
    max_per_gpu: PositiveInt
    max_runs: PositiveInt
    per_gpu: list[ServiceSpec]
    respawn_waits_s: list[PositiveFloat]
    pressure_retry_s: PositiveFloat
    service_restart_waits_s: list[PositiveFloat]
    service_idle_s: PositiveFloat
    watchdog_s: PositiveFloat
    min_free_gb: NonNegativeFloat
    status_interval_s: PositiveFloat
    spare_preempt_after_s: PositiveFloat
    placement_order: Literal["finish_tasks", "by_stage"]
    preempt: bool
    pythonpath: list[Path] = []
    service_unready_s: PositiveFloat = 60.0
    service_healthy_s: PositiveFloat = 1800.0

    def service_mib(self, name: str) -> float:
        """The memory (MiB) a GPU's service `name` reserves as it starts."""
        return self.service_gb.get(name, self.free_gb_per_place) * MIB_PER_GB


class Launch(abc.ABC):
    """One launch's jobs, services and alerts (see the module docstring). What is a pipeline's
    own it supplies by subclassing: the launch's job graph (`_tasks`, run with `run_graph` on
    `_run`), a job's worker command (`_command`), its records and working directory (`_dir`,
    `_content`), the modes of the blocks its processes run (`_owned`), its logs (`_logs`), the
    process groups its records name and its open blocks (`_job_groups`, `_open_blocks`), a GPU's
    service stack (`_service_stack`) and the status (`_write_status`)."""

    def __init__(
        self,
        settings: LaunchSettings,
        directory: Path,
        *,
        blobs: Path,
        plan: StackPlan | None = None,
        gpu_memory: GpuProbe | None = None,
        processes: ProcessTable | None = None,
        skipped: Mapping[int, str] | None = None,
    ) -> None:
        config = self.settings = settings
        self.directory = directory
        self.gpus = list(config.gpus)
        self.skipped = dict(skipped or {})
        self.stacks, self.plan, self.key, self.lease = None, None, None, None
        self.units: list[Unit] = []  # the services' stacks (shared service by service)
        if config.shared_stacks is not None:
            assert plan is not None, "shared stacks run the launch's stack plan"
            self.stacks = StackRegistry(config.shared_stacks)
            self.plan = plan
            self.key = stack_key(self.plan)
            self.units = plan_units(self.plan)
            self.lease = Lease.mine(str(directory), blobs)
        self.stack_of: dict[int, str] = {}  # each GPU's shared stack, while we hold it
        self.peaks = Peaks(directory / PEAKS)
        self.slots = Slots(
            self.gpus,
            config.max_per_gpu,
            config.max_runs,
            config.free_gb_per_place * MIB_PER_GB,
            gpu_memory or nvidia_smi,
            cap_mib=(config.gpu_cap_gb - config.gpu_margin_gb) * MIB_PER_GB,
            claims=(
                None
                if config.gpu_claims is None
                else SharedClaims(config.gpu_claims, directory / CLAIMS_MINE)
            ),
            processes=processes or ps,
            start=self._start,
            services_mib=sum(config.service_mib(spec.name) for spec in config.per_gpu),
            headroom_mib=config.peak_headroom_gb * MIB_PER_GB,
            services_of=None if self.stacks is None else self._services_of,
            by_stage=config.placement_order == "by_stage",
        )
        self._group: anyio.abc.TaskGroup | None = None
        self.respawned: Counter[RunKey] = Counter()
        self.pressure: dict[str, dict[str, float]] = {}  # by job or service: retries, since
        self.running: dict[RunKey, Worker] = {}
        self.alerts: dict[str, str] = {}
        self.services: dict[int, dict[str, dict[str, Any]]] = {
            gpu: {s.name: {"state": "off", "restarts": 0, "error": None} for s in config.per_gpu}
            for gpu in self.gpus
        }
        self.urls: dict[int, dict[str, str]] = {gpu: {} for gpu in self.gpus}
        self.source = source_sha()

    def extra(self) -> dict[str, Any]:
        """What the status adds: each GPU's services, places of ours and memory as the scheduler
        counts it (`Slots.ledger`), the GPUs skipped, the jobs and services waiting for room
        after running out of memory, and the alerts."""
        gpus = {}
        for gpu, services in self.services.items():
            states = {service["state"] for service in services.values()}
            order = ("down", "waiting", "starting", "restarting", "up", "off")
            gpus[gpu] = {
                "state": next((s for s in order if s in states), "up"),
                "places": self.slots.used[gpu],
                **self.slots.ledger(gpu),
                "services": {n: dict(v) for n, v in services.items()},
            }
            if self.stacks is not None:
                gpus[gpu]["stack"] = self.stack_of.get(gpu)
        now = anyio.current_time()
        pressure = {
            name: {
                "retries": int(entry["retries"]),
                "waited_s": round(now - entry["since"]),
                "limit_s": self.settings.pressure_retry_s,
            }
            for name, entry in self.pressure.items()
        }
        alerts = [*self.alerts.values()]
        if self.slots.probe_error is not None:
            alerts.append(f"placing jobs by our count alone: {self.slots.probe_error}")
        return {
            "gpus": gpus,
            "gpu_cap_gb": self.settings.gpu_cap_gb,
            "skipped": self.skipped,
            "pressure": pressure,
            "alerts": alerts,
        }

    async def run(self) -> None:
        up_now = True
        async with anyio.create_task_group() as group:
            self._group = group
            group.start_soon(self._monitor)
            group.start_soon(self._publish)
            if self.settings.preempt:
                group.start_soon(self._preempt)

            async def none_up() -> None:
                nonlocal up_now
                await self.slots.none_opened()
                up_now = False
                group.cancel_scope.cancel()  # the jobs waiting for room

            group.start_soon(none_up)
            await self._tasks()
            group.cancel_scope.cancel()  # the services and the monitor
        if not up_now:
            errors = {
                gpu: {name: state["error"] for name, state in services.items() if state["error"]}
                for gpu, services in self.services.items()
            }
            raise RuntimeError(f"no GPU's services came up: {errors}")

    def _start(self, gpu: int) -> None:
        """Start the GPU's services: a job waits for them (`Slots`)."""
        assert self._group is not None, "a job waits only while the launch runs"
        self._group.start_soon(self._gpu, gpu)

    async def _gpu(self, gpu: int) -> None:
        """The GPU's services, once it has room for them all (`service_gb` each; looked at every
        `MEMORY_TTL_S`, so that a GPU others load joins when they leave it room), until no job
        of ours was on the GPU for `service_idle_s`: then they stop (and start again when a job
        needs the GPU)."""
        specs, states = self.settings.per_gpu, self.services[gpu]
        if not specs:
            self.slots.open(gpu, {})
            return
        if self.stacks is not None:
            await self._shared_gpu(gpu)
            return
        for state in states.values():
            state["state"] = "waiting"
        mibs = [self.settings.service_mib(spec.name) for spec in specs]
        reservations = await self.slots.reserve(gpu, mibs)
        idle, stopping = self.settings.service_idle_s, False
        async with anyio.create_task_group() as group:
            for spec, reservation in zip(specs, reservations, strict=True):
                states[spec.name]["state"] = "starting"
                group.start_soon(self._service, gpu, spec, reservation)
            while not self.slots.idle(gpu, idle):  # noqa: ASYNC110 (a time, not an event)
                await anyio.sleep(min(idle / 10, IDLE_POLL_S))
            stopping = True
            log.info("GPU %d: no job of ours for %g s, its services stop", gpu, idle)
            group.cancel_scope.cancel()
        if stopping:
            self.urls[gpu].clear()
            for state in states.values():
                state.update(state="off", error=None)
            self.slots.stopped(gpu)

    async def _service(self, gpu: int, spec: ServiceSpec, reservation: Reservation) -> None:
        """A GPU's service until cancelled (see the module docstring), started on `reservation`
        and on a new one each restart. The GPU takes jobs while all its services are up. One
        that ran out of memory restarts without counting, for `pressure_retry_s` at most; one
        that stays not ready restarts too (`watch_ready`)."""
        state, waits = self.services[gpu][spec.name], self.settings.service_restart_waits_s
        port: int | None = None  # the port the GPU's jobs call, once it came up
        while True:
            stack = self._service_stack(gpu, spec, port)
            record = PidRecord(stack.run_dir / PIDS)
            logs = LogMarks(stack.run_dir)

            def started(
                name: str, pid: int, record: PidRecord = record, held: Reservation = reservation
            ) -> None:
                record.add(name, pid)
                held.roots.add(pid)  # its own session: its group is its processes

            try:
                async with up(stack, on_start=started) as running:
                    url = running.endpoints[f"{spec.name}-gpu{gpu}"]
                    port = urlsplit(url).port
                    self.urls[gpu][spec.name] = url
                    if len(self.urls[gpu]) == len(self.settings.per_gpu):
                        self.slots.open(gpu, self.urls[gpu])
                    state.update(state="up", error=None)
                    await watch_ready(self.settings, f"GPU {gpu}'s {spec.name}", url, state)
            except Exception as error:  # it did not come up, died, or stayed not ready
                if not isinstance(error, (StackError, OSError)):
                    log.exception("GPU %d: its %s failed", gpu, spec.name)
                self.urls[gpu].pop(spec.name, None)
                self.slots.close(gpu)
                self.slots.release(reservation)
                state["error"] = one_line(error)
                name = f"GPU {gpu} {spec.name}"
                if logs.out_of_memory() and self._pressured(name):
                    wait = PRESSURE_WAIT_S
                    state["state"] = "restarting"
                    log.error("%s ran out of GPU memory, waits for room: %s", name, error)
                elif state["restarts"] >= len(waits):
                    state["state"] = "down"
                    self.slots.fail(gpu, f"{spec.name}: {state['error']}")
                    self._interrupt(gpu, f"GPU {gpu}'s {spec.name} is down: {state['error']}")
                    log.error("GPU %d takes no more jobs: its %s failed: %s", gpu, spec.name, error)
                    return
                else:
                    wait = waits[state["restarts"]]
                    state.update(state="restarting", restarts=state["restarts"] + 1)
                    log.error(
                        "GPU %d: its %s failed, restarting in %g s: %s", gpu, spec.name, wait, error
                    )
                await anyio.sleep(wait)
                if port is not None and not _port_free(port):
                    log.error("GPU %d: port %d is taken: its %s moves", gpu, port, spec.name)
                    self._interrupt(
                        gpu, f"GPU {gpu}'s {spec.name} restarts on a new port ({port} is taken)"
                    )
                    port = None
                (reservation,) = await self.slots.reserve(
                    gpu, [self.settings.service_mib(spec.name)]
                )
            finally:
                record.clear()  # the stack stopped its group

    async def _shared_gpu(self, gpu: int) -> None:
        """The GPU's shared stack (`shared_stacks`), attached to, or started once the GPU has room
        for its services (`Slots.enter`), and held until no job of ours was on the GPU for
        `service_idle_s`. The GPU takes jobs while the stack is up; a service that moved to a new
        port stops the GPU's running jobs as infrastructure failures (respawned with its new
        URL), as the stack's end does; a stack that is down takes the GPU out."""
        assert self.stacks is not None and self.lease is not None
        stacks, states, idle = self.stacks, self.services[gpu], self.settings.service_idle_s
        for state in states.values():
            state["state"] = "waiting"
        record: StackSet = await self.slots.enter(gpu, partial(self._enter, gpu))
        self.stack_of[gpu] = record.id
        last, idled = record, False
        try:
            async with anyio.create_task_group() as group:

                async def watch() -> None:
                    nonlocal last
                    async for now in follow_set(stacks, [r.id for r in record.records]):
                        for name, service in now.services.items():
                            if name in states:
                                states[name].update(
                                    state=service.state,
                                    restarts=service.restarts,
                                    error=service.error,
                                )
                        if now.state == "up":
                            if last.state == "up" and now.generation != last.generation:
                                why = f"GPU {gpu}'s shared stack {now.id} moved a service"
                                self._interrupt(gpu, why)
                            self.urls[gpu] = dict(now.urls)
                            self.slots.open(gpu, now.urls)
                        elif last.state == "up":  # coming up, it counts as coming (`_demand`)
                            self.slots.close(gpu)
                        last = now
                    group.cancel_scope.cancel()  # it ended

                group.start_soon(watch)
                while not self.slots.idle(gpu, idle):  # noqa: ASYNC110 (a time, not an event)
                    await anyio.sleep(min(idle / 10, IDLE_POLL_S))
                idled = True
                log.info("GPU %d: no job of ours for %g s, leaving stack %s", gpu, idle, record.id)
                group.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                async with stacks.hold():
                    stacks.detach_set(record, self.lease.tenant)
            self.urls[gpu].clear()
            self.stack_of.pop(gpu, None)
        if last.state == "down":
            error = f"shared stack {last.id}: {last.error}"
            self.slots.fail(gpu, error)
            self._interrupt(gpu, f"GPU {gpu}'s {error}")
            log.error("GPU %d takes no more jobs: %s", gpu, error)
            return
        if not idled:
            self._interrupt(gpu, f"GPU {gpu}'s shared stack {record.id} stopped")
        for state in states.values():
            state.update(state="off", error=None)
        self.slots.stopped(gpu)

    async def _enter(self, gpu: int, room: bool) -> StackSet | None:
        """Attach to the GPU's shared stacks, one per service, or start those none runs if there
        is `room` for them (under the claims lock, where their memory is claimed for their
        keepers)."""
        assert self.stacks is not None and self.lease is not None
        checked = self._services_of(gpu)  # what `Slots.enter` found `room` for
        claimed = self.settings.gpu_claims is not None
        async with self.stacks.hold() as stacks:
            return await stacks.enter_set(
                gpu, self.units, self.lease, room=room, checked_mib=checked, claimed=claimed
            )

    def _services_of(self, gpu: int) -> float:
        """What the GPU's services still need there: nothing where shared stacks run them."""
        assert self.stacks is not None
        return self.stacks.missing_mib(gpu, self.units)

    def _need(self, peak: str, weight: int, shards: int = 1) -> float:
        """The memory (MiB) a job reserves as it starts: the most its task's jobs of as many
        places showed in this pipeline; for a P3 block of `shards` shards none ran yet, `shards`
        times what its task's unsharded blocks showed (each shard is one such block); else the
        most any task's jobs of as many places showed (a task's P3 runs once, so its own peak
        never helps it); plus `peak_headroom_gb`; else `free_gb_per_place` a place."""
        learned = self.peaks.get(peak)
        if learned is None and shards > 1:
            one = self.peaks.get(f"{peak.rsplit('/', 1)[0]}/{weight // shards}")
            learned = None if one is None else one * shards
        if learned is None:
            learned = self.peaks.most(weight)
        if learned is None:
            return self.settings.free_gb_per_place * MIB_PER_GB * weight
        return learned + self.settings.peak_headroom_gb * MIB_PER_GB

    def _pressured(self, name: str) -> bool:
        """Count a failure of `name` (a job or a GPU's service) that ran out of GPU memory: True
        while it is within `pressure_retry_s` of its first such failure (it retries free)."""
        now = anyio.current_time()
        entry = self.pressure.setdefault(name, {"retries": 0, "since": now})
        if now - entry["since"] > self.settings.pressure_retry_s:
            return False
        entry["retries"] += 1
        return True

    def _interrupt(self, gpu: int, why: str) -> None:
        """Stop the GPU's running jobs as infrastructure failures (respawned elsewhere or with
        the GPU's new URLs)."""
        for key, running in self.running.items():
            if running.gpu == gpu and running.why is None:
                log.warning("%s: stopping it: %s", key.name, why)
                running.interrupt(why)

    @abc.abstractmethod
    async def _tasks(self) -> None:
        """The launch's job graph, run to its end (`run_graph` on `_run`)."""

    async def _run(self, spec: JobSpec) -> RunState:
        key = RunKey(TaskName(spec.task), spec.unit, spec.lane)
        return await self._job(key, spec)

    async def _job(self, key: RunKey, spec: JobSpec) -> RunState:
        """Run a job in a worker (its spec's `modes`, the config's by default), respawned while
        infrastructure fails it (`respawn_waits_s`), `stopped` while it waits (so that a launch
        again takes it up if this one dies). A failure its logs show ran out of GPU memory waits
        for room and retries without counting, for `pressure_retry_s` at most. A job that ran in
        an earlier launch (which left it pending, `_settle`) waits in turn with those that ran
        before, and says so. A `spare` unit waits behind every other job (`Slots`), `queued` (a
        variant that had ended is taken up again) and saying so (its spec's `note`). A job holds
        its spec's `weight` places, in its `stage`'s turn (`first`: a task's first job)."""
        weight, waits = spec.weight, self.settings.respawn_waits_s
        spare, modes = spec.spare, spec.modes
        turn = self.slots.turn(first=spec.first, stage=spec.stage, spare=spare)
        peak = f"{key.task}/{weight}"  # its task's jobs of as many places take alike
        state = self._state(key)
        if state.launches > 0 and state.status == "stopped":
            turn = turn.again()
            state.error = "waits for GPU room to run again"
            state.save(self._dir(key))
        if spare:
            if state.status == "done":
                state.status = "queued"
            state.error = spec.note
            state.save(self._dir(key))
        while True:
            await self._disk()
            need = self._need(peak, weight, spec.shards)
            place = await self.slots.take(weight, turn, need, key.name)
            if place is None:
                state = self._state(key)
                state.failure = "infra"
                state.end(self._dir(key), "failed", f"every GPU is down: {self.slots.down}")
                return state
            place.reservation.parts = spec.shards  # each gives its room back as it ends
            logs = self._logs(key)
            began = anyio.current_time()
            try:
                state = await self._spawn(key, spec.phase, place, modes)
            finally:
                if anyio.current_time() - began >= PEAK_LEARN_S:
                    self.peaks.record(peak, place.reservation.peak)
                self.slots.give(place, weight)
            if state.status == "preempted":  # spare again, from its finished episodes
                state.error = f"{spec.note} after it was {state.error}"
                state.save(self._dir(key))
                turn = turn.again()
                continue
            if not (state.status == "failed" and state.failure == "infra"):
                return state
            cause = state.error
            if logs.out_of_memory() and self._pressured(key.name):
                entry = self.pressure[key.name]
                waited = anyio.current_time() - entry["since"]
                log.warning("%s ran out of GPU memory (%s); waits for room", key.name, state.error)
                state.status = "stopped"
                state.error = (
                    f"waits for GPU room after out of memory (retry {entry['retries']:.0f}, "
                    f"{waited:.0f} of {self.settings.pressure_retry_s:g} s): {state.error}"
                )
                state.save(self._dir(key))
                await anyio.sleep(PRESSURE_WAIT_S)
            elif self.respawned[key] < len(waits):
                wait = waits[self.respawned[key]]
                self.respawned[key] += 1
                log.warning(
                    "%s: infrastructure failed it (%s); again in %g s", key.name, state.error, wait
                )
                state.respawns += 1
                state.status, state.error = "stopped", f"respawns in {wait:g} s: {state.error}"
                state.save(self._dir(key))
                await anyio.sleep(wait)
                state.error = f"waits for GPU room to respawn: {cause}"
                state.save(self._dir(key))
            else:
                return state
            turn = turn.again()  # before the jobs that have not run yet

    async def _spawn(
        self, key: RunKey, phase: str, place: Placement, modes: Sequence[str] | None = None
    ) -> RunState:
        """A worker for the job (P3 in `modes`, the config's by default), watched (`_watch`),
        its processes (its group, its env stack's) showing its placement's memory; when it ends,
        what it left is stopped and its end recorded."""
        run_dir, gpu, urls = self._dir(key), place.gpu, place.urls  # its records
        content, owned = self._content(key), self._owned(key)
        modes = modes if phase == "evaluate" else None
        command = self._command(key, phase, gpu, urls, modes)
        env = {**os.environ, "PYTHONPATH": self._pythonpath(), "PYTHONUNBUFFERED": "1"}
        if self.source is not None:
            env["RS_SOURCE_SHA"] = self.source
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / HEARTBEAT).unlink(missing_ok=True)  # an earlier worker's
        launches = self._state(key).launches  # the worker counts its own once it begins
        with (run_dir / "worker.log").open("ab") as output:
            output.write(f"$ {shlex.join(command)}\n".encode())
            output.flush()
            process = await anyio.open_process(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,  # its own group: the launch stops it with its cleanup
            )
        record = PidRecord(run_dir / WORKER)
        record.add("worker", process.pid)
        place.reservation.roots.add(process.pid)
        place.reservation.groups = lambda: self._job_groups(content, run_dir, owned)
        place.reservation.ended = lambda: self._ended_groups(content, owned)
        running = self.running[key] = Worker(gpu, process, place.reservation)
        log.info("%s: %s on GPU %d (worker %d)", key.name, phase, gpu, process.pid)
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(self._watch, key, running, run_dir)
                await process.wait()
                group.cancel_scope.cancel()
        finally:
            del self.running[key]
            with anyio.CancelScope(shield=True):
                stopping = process.returncode is None  # the launch stops
                if stopping:
                    await stop_groups([(process, WORKER_STOP_S)])
                await self._clean(content, run_dir, owned)
                record.clear()
                state = self._state(key)
                began = state.launches > launches
                if stopping:
                    if began and state.status == "running":
                        state.end(run_dir, "stopped")
                elif running.preempted and (not began or state.status in STOPPED):
                    state.end(run_dir, "preempted", running.why)
                elif running.why is not None and (not began or state.status in STOPPED):
                    state.failure = "infra"
                    state.end(run_dir, "failed", running.why)
                elif not began or state.status == "running":  # it died before it recorded an end
                    state.failure = "infra"
                    before = "" if began else " before it began the job"
                    state.end(run_dir, "failed", f"its worker {_exit(process.returncode)}{before}")
        return state

    async def _watch(self, key: RunKey, running: Worker, run_dir: Path) -> None:
        """Stop the worker (SIGTERM, then SIGKILL after `WORKER_STOP_S`) once the scheduler
        interrupts it or its heartbeat is older than `watchdog_s`."""
        beat, since, limit = run_dir / HEARTBEAT, time.time(), self.settings.watchdog_s
        while running.why is None:
            with anyio.move_on_after(min(limit / 10, WATCH_POLL_S)):
                await running.wake.wait()
            if running.why is None and time.time() - max(since, _mtime(beat)) > limit:
                log.error("%s: its worker gave no heartbeat for %g s", key.name, limit)
                running.interrupt(f"its worker gave no heartbeat for {limit:g} s (killed)")
        await stop_groups([(running.process, WORKER_STOP_S)])

    async def _clean(
        self, content: Path, records: Path, modes: Sequence[str] | None = None
    ) -> None:
        """Stop what a job's worker left running, clear its records and set aside its open P3
        blocks (of `modes`, those it runs: not a lane's beside it)."""
        groups = [pid for pid in self._job_groups(content, records, modes) if alive(pid)]
        if groups:
            log.warning("stopping what %s left: process groups %s", records, groups)
            await _stop(groups)
        (records / PIDS).unlink(missing_ok=True)
        for out in self._open_blocks(content, modes):
            set_aside(out)

    async def _disk(self) -> None:
        """Wait while the disk has less than `min_free_gb` free, alerting in the status."""
        while (free := shutil.disk_usage(self.directory).free / 1e9) < self.settings.min_free_gb:
            if "disk" not in self.alerts:
                self.alerts["disk"] = (
                    f"only {free:.0f} GB free on the disk of {self.directory} (below "
                    f"{self.settings.min_free_gb:g} GB): no new job starts"
                )
                log.error("ALERT: %s", self.alerts["disk"])
            await anyio.sleep(DISK_POLL_S)
        self.alerts.pop("disk", None)

    async def _monitor(self) -> None:
        """The status every `status_interval_s`; a status that cannot be written stops no job."""
        while True:
            try:
                extra = self.extra()
                await anyio.to_thread.run_sync(self._write_status, extra)
            except Exception:
                log.exception("could not write the status of %s", self.directory)
            await anyio.sleep(self.settings.status_interval_s)

    async def _preempt(self) -> None:
        """Every `PREEMPT_POLL_S`, stop the spare units `Slots.preempt` picks: their workers are
        stopped as a launch's stop stops them (P3's server and client together, the open block
        set aside), and their jobs wait for spare room again (`_job`)."""
        while True:
            await anyio.sleep(PREEMPT_POLL_S)
            for reservation, waiter in await self.slots.preempt(
                self.settings.spare_preempt_after_s
            ):
                for key, running in self.running.items():
                    if running.reservation is reservation:
                        why = f"preempted for {waiter} (waited for its room)"
                        log.warning("%s: %s", key.name, why)
                        running.interrupt(why, preempted=True)

    async def _publish(self) -> None:
        """Our holds in the shared claims file every `PUBLISH_S`; a failure to write them stops
        no job (the lines published before expire)."""
        while True:
            try:
                await self.slots.publish()
            except OSError:
                log.exception("could not publish our GPU holds in %s", self.settings.gpu_claims)
            await anyio.sleep(PUBLISH_S)

    def _pythonpath(self) -> str:
        """A worker's: the settings' `pythonpath` before the launch's own."""
        own = os.environ.get("PYTHONPATH")
        return os.pathsep.join(p for p in (*map(str, self.settings.pythonpath), own) if p)

    @abc.abstractmethod
    def _command(
        self,
        key: RunKey,
        phase: str,
        gpu: int,
        urls: Mapping[str, str],
        modes: Sequence[str] | None,
    ) -> list[str]:
        """The command of a job's worker on `gpu`, with the GPU's service `urls`."""

    @abc.abstractmethod
    def _write_status(self, extra: dict[str, Any]) -> None:
        """Write the launch's status, with what it adds (`extra`)."""

    @abc.abstractmethod
    def _dir(self, key: RunKey) -> Path:
        """A job's records."""

    @abc.abstractmethod
    def _content(self, key: RunKey) -> Path:
        """The directory a job works in."""

    @abc.abstractmethod
    def _owned(self, key: RunKey) -> Sequence[str] | None:
        """The modes whose blocks a job's processes run (None: all)."""

    @abc.abstractmethod
    def _logs(self, key: RunKey) -> LogMarks:
        """A job's logs, as they are now."""

    @abc.abstractmethod
    def _job_groups(
        self, run_dir: Path, records: Path | None = None, modes: Sequence[str] | None = None
    ) -> list[int]:
        """The process groups a job's records (in `records`, default `run_dir`) name."""

    def _ended_groups(self, run_dir: Path, modes: Sequence[str] | None = None) -> list[int]:
        """The process groups of a job's parts (its spec's `shards`) that its records declare
        ended for good (of `modes`' blocks): their room goes back (`Reservation`). None here."""
        return []

    @abc.abstractmethod
    def _open_blocks(self, run_dir: Path, modes: Sequence[str] | None = None) -> list[Path]:
        """A job's blocks that are open (of `modes`; None: all), set aside as it ends."""

    @abc.abstractmethod
    def _service_stack(self, gpu: int, spec: ServiceSpec, port: int | None = None) -> StackConfig:
        """A GPU's service in a stack of its own, on `port` (None: any free one)."""

    def _state(self, key: RunKey) -> RunState:
        return RunState.load(self._dir(key), key)


class Peaks:
    """The most GPU memory (MiB) the jobs of each task and weight showed, kept in `path` across
    launches (only increases)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.found: dict[str, float] = json.loads(path.read_bytes()) if path.is_file() else {}

    def get(self, key: str) -> float | None:
        return self.found.get(key)

    def most(self, weight: int) -> float | None:
        """The most any task's jobs of `weight` places showed."""
        found = [mib for key, mib in self.found.items() if key.endswith(f"/{weight}")]
        return max(found, default=None)

    def record(self, key: str, mib: float) -> None:
        if mib > self.found.get(key, 0.0):
            self.found[key] = round(mib)
            write_atomic(self.path, json.dumps(self.found, indent=1, sort_keys=True).encode())


async def watch_ready(settings: LaunchSettings, name: str, url: str, state: dict[str, Any]) -> None:
    """Watch a GPU service that came up (its `/readyz` answered 200) until it has not been ready
    for `service_unready_s` straight (503, or no answer), then raise `StackError`, on which its
    stack stops its process group and it restarts as one that died does: a service can stay alive
    refusing every call (a latched failure, a wedged GPU context), which an exit alone would never
    restart. The time the host stood still on I/O or memory between two polls that ended without
    an answer does not count, `service_unready_s` at most in all (`stack.stall.Excuse`): a
    stalled disk freezes a service's event loop (its access log),
    and a restart would not mend that, but the pressure counts other processes' I/O too,
    so a service that is gone is given up after twice `service_unready_s` at worst; an answer of
    503 always counts. Once it has been ready for
    `service_healthy_s` straight, its `restarts` in `state` go back to 0, so that rare failures
    over a long run do not use up `service_restart_waits_s`."""
    unready_s, healthy_s = settings.service_unready_s, settings.service_healthy_s
    poll, timeout = min(READY_POLL_S, unready_s / 6), min(READY_TIMEOUT_S, unready_s)
    ready_since: float | None = anyio.current_time()
    unready_since: float | None = None
    excuse = stall.Excuse(unready_s)  # begun again each time unready begins
    async with httpx.AsyncClient(trust_env=False) as http:
        while True:
            await anyio.sleep(poll)
            try:
                response = await http.get(f"{url}/readyz", timeout=timeout)
                reason = None if response.status_code == 200 else f"answers {response.status_code}"
            except httpx.TransportError:
                reason = "does not answer"
            now = anyio.current_time()
            if reason is None:
                unready_since = None
                ready_since = now if ready_since is None else ready_since
                if state["restarts"] and now - ready_since >= healthy_s:
                    log.info("%s was ready for %g s: its restarts are back", name, healthy_s)
                    state["restarts"] = 0
                continue
            ready_since = None
            if unready_since is None:
                unready_since, excuse = now, stall.Excuse(unready_s)
                log.warning("%s: /readyz %s", name, reason)
            waited = now - unready_since - excuse.mark(reason == "does not answer")
            if waited >= unready_s:
                raise StackError(f"{name} was not ready for {waited:.0f} s (/readyz {reason})")


def _port_free(port: int) -> bool:
    """Whether a service can listen on `port` again (bound as `roboscript.rpc.serve` binds)."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:  # no heartbeat yet
        return 0.0


async def _stop(groups: Sequence[int]) -> None:
    """SIGTERM to each process group, SIGKILL to those still there after `GROUP_STOP_S`."""
    for group in groups:
        _signal(group, signal.SIGTERM)
    with anyio.move_on_after(GROUP_STOP_S):
        while any(alive(group) for group in groups):  # noqa: ASYNC110 - other processes'
            await anyio.sleep(0.1)
    for group in groups:
        _signal(group, signal.SIGKILL)


def alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):  # gone, or another user's group now
        return False
    return True


def _signal(group: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):  # the whole group is gone
        os.killpg(group, sig)


def _exit(code: int | None) -> str:
    return (
        f"was killed by {signal.Signals(-code).name}" if code and code < 0 else f"exited ({code})"
    )
