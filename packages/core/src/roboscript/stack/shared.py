"""Service stacks shared by the processes of a machine (docs/concepts/embodied-core.md): one per
GPU and code version, started when the first tenant needs it and stopped once none has used it for a
while.

A tenant (a pipeline launch, a tool) that needs a GPU's services (a planner, a vision service)
attaches to the stack that runs the same services on that GPU (`StackRegistry.enter`), and starts
one only when there is none and the GPU has room for it. Stacks are the same when their
`stack_key` is: the services' commands, interpreters and environments, the content of the files
and configs their arguments name (a YAML config by what it says, not its comments or layout, and
without the knobs that only set how many calls it admits, `THROUGHPUT_KNOBS`), and the source of
every first-party module they import (`code_files`), so that a branch that changes a service's code
gets a stack of its own while branches that do not share one.

A plan's services are shared one by one: `plan_units` cuts a plan into one unit per service (a
plan whose services `depends_on` each other stays one unit), each with a `stack_key` of its own,
so that a run whose motion code differs from a stack's still attaches to the vision service that
runs on the GPU and starts only its own motion service. A tenant holds a `StackSet`, the stacks of
its plan's units on one GPU (`StackRegistry.enter_set`, `follow_set`), which reads as one record.

A stack is run by its keeper, a process of its own (`roboscript.stack.keeper`, in a session of
its own, so that it outlives the tenant that started it): it starts each service in a stack of its
own (`roboscript.stack.up`), restarts one that dies on the same port, so that the URLs its tenants
hold stay valid (a new port, when that one is taken, bumps the record's `generation`), publishes in
the machine's claims file (`SharedClaims`) the memory its services will take but do not show yet,
and stops the stack once it has had no live tenant for `StackPlan.linger_s`. A tenant that finds
the keeper of its stack dead starts another (`follow`), which stops what the dead one left.

`directory` (the registry, best on a tmpfs such as /dev/shm/gpu_stacks) holds `lock` (flock:
every change, `flocked`) and
a directory per stack: `stack.json` (`StackRecord`: its plan, state, services' URLs, keeper),
`tenants/<tenant>.json` (a `Lease`: the process that holds it and its blob store), `roots/<tenant>`
(that blob store, where the services read their callers' images and write their outputs:
`roboscript.blobs.BlobRoots`, `RS_BLOB_ROOTS`), `services/<name>/` (each service's stack: logs,
ready file), `keeper.log`, and the keeper's claims (`gpu_claims.mine`). A lease counts while its
process lives (the same pid and start time). Simple tenants use `use_stack`; the pipeline's
scheduler attaches within its own placing.
"""

import ast
import contextlib
import hashlib
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import anyio
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
)

import roboscript
from roboscript.contracts.canonical import canonical_json
from roboscript.records import write_atomic
from roboscript.stack.config import Interpreters, ServiceSpec
from roboscript.stack.gpus import CLAIMS_MINE, GpuProbe, SharedClaims, flocked, nvidia_smi
from roboscript.stack.supervisor import StackError

log = logging.getLogger(__name__)

SCHEMA: Final = 1
"""The registry's format: records of another are not read (a branch that changes it runs its
own stacks)."""
POLL_S: Final = 2.0
"""How often a keeper looks at its tenants, and a tenant (`follow`) at its stack."""
SPAWN_S: Final = 60.0
"""How long a keeper may take to take up its stack once spawned, before it counts as dead."""
MAX_SPAWNS: Final = 3
"""Keepers spawned for a stack in a row without its services coming up, before it is `down`."""
ROOM_POLL_S: Final = 10.0
"""How often `use_stack` looks again for room to start a stack."""
_SPAWN: Final = (
    "import subprocess, sys\n"
    "log = open(sys.argv[1], 'ab')\n"
    "child = subprocess.Popen(sys.argv[2:], start_new_session=True, stdin=subprocess.DEVNULL,"
    " stdout=log, stderr=subprocess.STDOUT)\n"
    "print(child.pid)\n"
)
"""Starts the keeper and exits: the keeper, orphaned, is reaped by init, never left a zombie of
the tenant that started it."""
THROUGHPUT_KNOBS: Final = frozenset({"max_pending"})
"""Top-level keys of a service's YAML config that only set how many calls it admits at once (more
fail as `busy`, retryable): they change what a stack can carry, not what its services compute, so
they are not in `stack_key` (the stack that was started first sets them for its tenants)."""
_NAME: Final = re.compile(r"[^A-Za-z0-9_.-]+")
HEALTHY_S: Final = 1800.0
"""`StackPlan.healthy_s` by default."""

State = Literal["starting", "up", "restarting", "stopping", "down"]


class StackPlan(BaseModel):
    """What a shared stack runs: its `services` (no `gpu`, no `--port`: the stack sets both), their
    interpreters and PYTHONPATHs (`agent` services import the tenant's own `roboscript`), the
    repository they run from (`root`: their working directory and `RS_ROOT`, where configs named
    by name are), and how the keeper runs them: the memory each takes (`mib`, MiB, claimed in
    `claims` until the service shows it), the waits before each restart of one that died
    (`restart_waits_s`; after the last it is down, and so is the stack), how long one that ran out
    of GPU memory restarts without counting (`pressure_retry_s`), how long one must be ready
    straight to have its restarts back (`healthy_s`), and how long the stack outlives its last
    tenant (`linger_s`).

    `healthy_s` is not in `stack_key`: as with `THROUGHPUT_KNOBS`, the tenant that started the
    stack sets it for all its tenants. A record leaves it out while it has its default
    (`HEALTHY_S`), so that readers that forbid fields they do not know can read the record then."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    services: list[ServiceSpec] = Field(min_length=1)
    interpreters: Interpreters
    sim_pythonpath: list[Path] = []
    vision_pythonpath: list[Path] = []
    root: Path
    mib: dict[str, float] = {}
    claims: Path | None = None
    restart_waits_s: list[float] = [10.0, 60.0, 300.0]
    pressure_retry_s: float = 3600.0
    healthy_s: float = Field(default=HEALTHY_S, gt=0)
    linger_s: float = 60.0

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if self.healthy_s == HEALTHY_S:
            data.pop("healthy_s", None)
        return data


class Proc(BaseModel):
    """A process, told from a later one with the same pid by its start time (`/proc`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pid: int
    start: str | None = None

    @classmethod
    def of(cls, pid: int) -> "Proc":
        return cls(pid=pid, start=_stat(pid)[1])

    def alive(self) -> bool:
        """Whether it runs (a zombie does not)."""
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # another user's: not this one
            return False
        state, start = _stat(self.pid)
        return state != "Z" and (self.start is None or start == self.start)


class ServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["starting", "up", "restarting", "down"] = "starting"
    url: str | None = None
    port: int | None = None
    pid: int | None = None  # its process group
    restarts: int = 0
    error: str | None = None


class StackRecord(BaseModel):
    """A shared stack (`stack.json`): `generation` counts the times a service's URL changed."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SCHEMA
    id: str
    gpu: int
    key: str
    plan: StackPlan
    state: State = "starting"
    services: dict[str, ServiceState] = {}
    keeper: Proc | None = None
    spawned: float = 0.0  # when its keeper was spawned (time.time())
    spawns: int = 0  # keepers spawned since its services were last all up
    generation: int = 0
    error: str | None = None
    created: str = ""
    by: str = ""  # the tenant that started it

    @property
    def urls(self) -> dict[str, str]:
        return {name: s.url for name, s in self.services.items() if s.url is not None}


class Lease(BaseModel):
    """A tenant's hold on a stack: its process, a label for people and the blob store its calls'
    blobs are in (None: it passes none)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str
    label: str
    proc: Proc
    blob_dir: Path | None = None

    @classmethod
    def mine(cls, label: str, blob_dir: Path | None) -> "Lease":
        """This process's lease, named by `label` and the pid."""
        pid = os.getpid()
        name = _NAME.sub("-", label).strip("-")[-80:] or "tenant"
        return cls(tenant=f"{name}-{pid}", label=label, proc=Proc.of(pid), blob_dir=blob_dir)


@dataclass(frozen=True)
class StackSet:
    """The stacks a tenant holds on one GPU, one per unit of its plan (`plan_units`), read as one
    record: `state` is `up` when all are, `down` or `stopping` when one is; `generation` changes
    when a service's URL did in any of them; `services`, `urls` and `error` are theirs together."""

    records: tuple[StackRecord, ...]

    @property
    def id(self) -> str:
        return "+".join(record.id for record in self.records)

    @property
    def state(self) -> State:
        states = {record.state for record in self.records}
        for state in ("down", "stopping", "restarting", "starting"):
            if state in states:
                return state
        return "up"

    @property
    def generation(self) -> int:
        return sum(record.generation for record in self.records)

    @property
    def services(self) -> dict[str, ServiceState]:
        return {name: s for record in self.records for name, s in record.services.items()}

    @property
    def urls(self) -> dict[str, str]:
        return {name: url for record in self.records for name, url in record.urls.items()}

    @property
    def error(self) -> str | None:
        return next((record.error for record in self.records if record.error), None)


@dataclass(frozen=True)
class Unit:
    """A stack's worth of a plan: one service (see `plan_units`) and its `stack_key`."""

    plan: StackPlan
    key: str


def plan_units(plan: StackPlan) -> list[Unit]:
    """`plan` cut into one unit per service, each keyed apart (`stack_key`), so that stacks are
    shared service by service. Services that `depends_on` each other start together: such a plan
    is one unit. Compute once and keep: a key reads the services' code."""
    if len(plan.services) == 1 or any(spec.depends_on for spec in plan.services):
        return [Unit(plan, stack_key(plan))]
    units = []
    for spec in plan.services:
        mib = {name: v for name, v in plan.mib.items() if name == spec.name}
        part = plan.model_copy(update={"services": [spec], "mib": mib})
        units.append(Unit(part, stack_key(part)))
    return units


class StackRegistry:
    """The shared stacks in `directory` (see the module docstring). Everything but `read` and
    `live` runs under `hold`."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._unreadable: set[Path] = set()

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator["StackRegistry"]:
        async with flocked(self.directory / "lock"):
            yield self

    def path(self, stack_id: str) -> Path:
        return self.directory / stack_id

    def read(self, stack_id: str) -> StackRecord | None:
        path = self.path(stack_id) / "stack.json"
        try:
            return StackRecord.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            return None
        except ValidationError as error:  # another schema, a branch of its own
            if path not in self._unreadable:
                log.warning("%s: not a stack this code reads: %s", path, error)
                self._unreadable.add(path)
            return None

    def records(self) -> list[StackRecord]:
        found = (self.read(p.parent.name) for p in sorted(self.directory.glob("*/stack.json")))
        return [record for record in found if record is not None]

    def live(self, gpu: int, key: str) -> StackRecord | None:
        """The stack of `key` on `gpu` that runs or comes up (its keeper may have died: `follow`
        starts another)."""
        for record in self.records():
            if record.gpu == gpu and record.key == key and record.state in _RUNNING:
                return record
        return None

    def write(self, record: StackRecord) -> None:
        directory = self.path(record.id)
        directory.mkdir(parents=True, exist_ok=True)
        write_atomic(directory / "stack.json", record.model_dump_json(indent=1).encode())

    def remove(self, stack_id: str) -> None:
        (self.path(stack_id) / "stack.json").unlink(missing_ok=True)

    def tenants(self, stack_id: str) -> list[Lease]:
        """The stack's live leases; those of processes that ended go."""
        live = []
        for path in sorted((self.path(stack_id) / "tenants").glob("*.json")):
            try:
                lease = Lease.model_validate_json(path.read_bytes())
            except (FileNotFoundError, ValidationError):
                continue
            if lease.proc.alive():
                live.append(lease)
            else:
                self.detach(stack_id, lease.tenant)
        return live

    def attach(self, stack_id: str, lease: Lease) -> None:
        """Hold the stack with `lease`, its blob store listed for the services (`roots/`)."""
        directory = self.path(stack_id)
        (directory / "tenants").mkdir(parents=True, exist_ok=True)
        (directory / "roots").mkdir(parents=True, exist_ok=True)
        if lease.blob_dir is not None:
            root = str(lease.blob_dir.absolute())
            write_atomic(directory / "roots" / lease.tenant, root.encode())
        payload = lease.model_dump_json(indent=1).encode()
        write_atomic(directory / "tenants" / f"{lease.tenant}.json", payload)

    def detach(self, stack_id: str, tenant: str) -> None:
        directory = self.path(stack_id)
        (directory / "tenants" / f"{tenant}.json").unlink(missing_ok=True)
        (directory / "roots" / tenant).unlink(missing_ok=True)

    def live_set(self, gpu: int, units: Sequence[Unit]) -> StackSet | None:
        """The stacks of all `units` on `gpu`, if each runs or comes up."""
        found = [self.live(gpu, unit.key) for unit in units]
        if any(record is None for record in found):
            return None
        return StackSet(tuple(record for record in found if record is not None))

    def missing_mib(self, gpu: int, units: Sequence[Unit]) -> float:
        """The memory (MiB) the `units` that no stack on `gpu` runs yet still need there."""
        return sum(
            sum(unit.plan.mib.values()) for unit in units if self.live(gpu, unit.key) is None
        )

    async def enter_set(
        self,
        gpu: int,
        units: Sequence[Unit],
        lease: Lease,
        *,
        room: bool,
        checked_mib: float,
        claimed: bool,
    ) -> StackSet | None:
        """Attach `lease` to the stack of each of `units` on `gpu`, starting those that none runs
        when there is `room`, which the caller found for `checked_mib`: if more is missing by now
        (another tenant's stack went), nothing is done and the result is None, as it is when there
        is no room. Under `hold` (and, with `claimed`, the claims lock)."""
        missing = [unit for unit in units if self.live(gpu, unit.key) is None]
        if missing and (not room or self.missing_mib(gpu, units) > checked_mib + 1e-6):
            return None
        records = []
        for unit in units:
            record = self.live(gpu, unit.key)
            if record is None:
                record = await self.create(gpu, unit.key, unit.plan, lease.label, claimed=claimed)
            self.attach(record.id, lease)
            records.append(record)
        return StackSet(tuple(records))

    def detach_set(self, stacks: StackSet, tenant: str) -> None:
        for record in stacks.records:
            self.detach(record.id, tenant)

    async def enter(
        self, gpu: int, key: str, plan: StackPlan, lease: Lease, *, room: bool, claimed: bool
    ) -> StackRecord | None:
        """Attach `lease` to the stack of `key` on `gpu`, else, when the GPU has `room` for one,
        start it (`create`) and attach; None: neither. `claimed`: the caller holds the claims
        file's lock (`SharedClaims.hold`), so the new stack's memory is claimed there."""
        record = self.live(gpu, key)
        if record is None:
            if not room:
                return None
            record = await self.create(gpu, key, plan, lease.label, claimed=claimed)
        self.attach(record.id, lease)
        return record

    async def create(
        self, gpu: int, key: str, plan: StackPlan, by: str, *, claimed: bool
    ) -> StackRecord:
        """A new stack of `plan` on `gpu`, its keeper spawned; with `claimed`, the services'
        memory is claimed in `plan.claims` as the keeper's (it replaces the claim with what it
        holds as it publishes)."""
        prefix = f"gpu{gpu}-{key[:12]}-"
        taken = [p.name[len(prefix) :] for p in self.directory.glob(f"{prefix}*")]
        number = max([int(n) for n in taken if n.isdigit()], default=0) + 1
        record = StackRecord(
            id=f"{prefix}{number}",
            gpu=gpu,
            key=key,
            plan=plan,
            services={spec.name: ServiceState() for spec in plan.services},
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            by=by,
        )
        self.write(record)
        if claimed and plan.claims is not None and (mib := sum(plan.mib.values())) > 0:
            SharedClaims(plan.claims, self.path(record.id) / CLAIMS_MINE).add(gpu, mib)
        return await self.spawn(record)

    async def spawn(self, record: StackRecord) -> StackRecord:
        """Start a keeper for `record` (a new stack, or one whose keeper died); after
        `MAX_SPAWNS` in a row without its services coming up, it is `down` instead."""
        if record.spawns >= MAX_SPAWNS:
            record.state = "down"
            record.error = f"its keeper died {record.spawns} times: see {self._keeper_log(record)}"
            self.write(record)
            return record
        command = [
            sys.executable,
            "-c",
            _SPAWN,
            str(self._keeper_log(record)),
            sys.executable,
            "-m",
            "roboscript.stack.keeper",
            str(self.directory),
            record.id,
        ]
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(_keeper_path()),
            "RS_ROOT": str(record.plan.root),
        }
        done = await anyio.to_thread.run_sync(
            lambda: subprocess.run(
                command,
                cwd=record.plan.root,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                check=True,
            )
        )
        record.keeper = Proc.of(int(done.stdout.decode().split()[-1]))
        record.spawned, record.spawns = time.time(), record.spawns + 1
        self.write(record)
        log.info("stack %s: keeper %d spawned", record.id, record.keeper.pid)
        return record

    def keeper_alive(self, record: StackRecord) -> bool:
        if record.keeper is not None and record.keeper.alive():
            return True
        return record.keeper is None and time.time() - record.spawned < SPAWN_S

    def _keeper_log(self, record: StackRecord) -> Path:
        return self.path(record.id) / "keeper.log"


_RUNNING: Final = ("starting", "up", "restarting")


def _keeper_path() -> list[str]:
    """The keeper's PYTHONPATH: the `roboscript` this process runs, before what it had."""
    core = str(Path(roboscript.__file__).resolve().parents[1])
    return [core, *(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)]


async def _current(registry: StackRegistry, stack_id: str) -> StackRecord | None:
    """The stack's record, its keeper replaced when it died (under `registry.hold`)."""
    record = registry.read(stack_id)
    if record is not None and record.state in _RUNNING and not registry.keeper_alive(record):
        log.warning("stack %s: its keeper is gone, starting another", stack_id)
        record = await registry.spawn(record)
    return record


_VIEW: Final = {"state", "generation", "services", "keeper"}


async def follow(
    registry: StackRegistry, stack_id: str, *, poll_s: float = POLL_S
) -> AsyncIterator[StackRecord]:
    """The stack's record whenever its state, a service's or its keeper changes (the first at
    once); a keeper that died is replaced (`StackRegistry.spawn`). Ends after a `down` or `stopping`
    record, or when the record is gone."""
    seen = None
    while True:
        async with registry.hold():
            record = await _current(registry, stack_id)
        if record is None:
            return
        view = record.model_dump_json(include=_VIEW)
        if view != seen:
            seen = view
            yield record
        if record.state not in _RUNNING:
            return
        await anyio.sleep(poll_s)


async def follow_set(
    registry: StackRegistry, stack_ids: Sequence[str], *, poll_s: float = POLL_S
) -> AsyncIterator["StackSet"]:
    """`follow` for a set of stacks: the `StackSet` whenever one of them changes (the first at
    once). Ends after a set that is `down` or `stopping` (one of its stacks is), or when a record
    is gone."""
    seen = None
    while True:
        async with registry.hold():
            records = [await _current(registry, stack_id) for stack_id in stack_ids]
        found = [record for record in records if record is not None]
        if len(found) < len(records):
            return
        now = StackSet(tuple(found))
        view = [record.model_dump_json(include=_VIEW) for record in found]
        if view != seen:
            seen = view
            yield now
        if now.state not in _RUNNING:
            return
        await anyio.sleep(poll_s)


@dataclass
class Attached:
    """A tenant's stacks (`use_stack`): their set as last seen; `urls` once its services are up."""

    record: StackSet

    @property
    def urls(self) -> dict[str, str]:
        return self.record.urls


@contextlib.asynccontextmanager
async def use_stack(
    directory: Path,
    plan: StackPlan,
    gpu: int,
    lease: Lease,
    *,
    claims: SharedClaims | None = None,
    cap_mib: float = math.inf,
    probe: GpuProbe = nvidia_smi,
) -> AsyncIterator[Attached]:
    """Hold the shared stacks of `plan` on `gpu` for the body, from when its services are up: the
    ones running, else those started when the GPU has room for them (its memory in use, plus what
    others claim in `claims`, plus the `mib` of the services no stack runs yet, within `cap_mib`;
    looked at every `ROOM_POLL_S`). The set is kept current (a service that moved changes
    `urls`); the body is cancelled, and `StackError` raised, when a stack goes down. The lease
    goes on exit."""
    registry, units = StackRegistry(directory), plan_units(plan)
    found = None
    while found is None:
        async with contextlib.AsyncExitStack() as locks:
            held = None if claims is None else await locks.enter_async_context(claims.hold())
            need = registry.missing_mib(gpu, units)
            room = held is None or await _room(probe, held, gpu, need, cap_mib)
            async with registry.hold():
                found = await registry.enter_set(
                    gpu, units, lease, room=room, checked_mib=need, claimed=bool(held)
                )
        if found is None:
            await anyio.sleep(ROOM_POLL_S)
    attached, failed = Attached(found), None
    try:
        async with anyio.create_task_group() as group:
            up = anyio.Event()

            async def watch() -> None:
                nonlocal failed
                async for now in follow_set(registry, [r.id for r in found.records]):
                    attached.record = now
                    if now.state == "up":
                        up.set()
                failed = attached.record.error or f"stack {found.id} is {attached.record.state}"
                group.cancel_scope.cancel()

            group.start_soon(watch)
            await up.wait()
            yield attached
            group.cancel_scope.cancel()
    finally:
        with anyio.CancelScope(shield=True):
            async with registry.hold():
                registry.detach_set(found, lease.tenant)
    if failed is not None:
        raise StackError(f"the shared stack {found.id} on GPU {gpu} stopped: {failed}")


async def _room(
    probe: GpuProbe, claims: SharedClaims, gpu: int, mib: float, cap_mib: float
) -> bool:
    memory = await probe()
    if gpu not in memory:
        return False
    claimed = claims.others().get(gpu, 0.0) * 1024  # GiB, the claims file's unit
    return memory[gpu].used_mib + claimed + mib <= min(cap_mib, memory[gpu].total_mib)


def stack_key(plan: StackPlan) -> str:
    """What makes two stacks the same (see the module docstring), as a sha256 hex digest: not
    paths (another checkout of the same code shares the stack), nor how the keeper runs it.

    The code is the import closure of each service (`code_files`). Where that cannot be worked out
    (a module that does not parse, a file that cannot be read) the key falls back to every file
    under the service's roots (`all_files`): wider, so stacks are shared less, never wrongly."""
    services = []
    for spec in sorted(plan.services, key=lambda s: s.name):
        roots = _roots(plan, spec)
        script = Path(spec.argv[0]) if spec.argv else None
        if script is not None and not script.is_absolute():
            script = plan.root / script
        try:
            code = {"closure": code_files(spec.module, script, roots)}
        except (OSError, SyntaxError, ValueError, StopIteration) as error:
            log.warning(
                "stack key of %s: no import closure (%s), keying on every file", spec.name, error
            )
            code = {"all_files": all_files([*roots, *([script.parent] if script else [])])}
        interpreter = _interpreter(plan, spec)
        services.append(
            {
                "name": spec.name,
                "module": spec.module,
                "argv": [_arg(a, plan.root) for a in spec.argv or []],
                "args": [_arg(a, plan.root) for a in spec.args],
                "interpreter": str(interpreter),
                "environment": _environment(interpreter),
                "env": spec.env,
                "code": code,
            }
        )
    return hashlib.sha256(
        canonical_json({"schema": SCHEMA, "services": services}).encode()
    ).hexdigest()


def all_files(roots: Sequence[Path]) -> dict[str, str]:
    """Every file under `roots` but caches and VCS directories, as sha256 hex digests by the root's
    place in `roots` and the path in it (not its own path: another checkout shares), `stack_key`'s
    fallback; a root that is not a directory counts as empty."""
    found: dict[str, str] = {}
    for number, root in enumerate(roots):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            parts = path.relative_to(root).parts
            if path.is_file() and not any(p in ("__pycache__", ".git") for p in parts):
                found[f"{number}:{'/'.join(parts)}"] = _sha(path)
    return found


def _environment(interpreter: Path) -> dict[str, str]:
    """The Python environment an absolute interpreter path names: where it really lives (the same
    path may be re-pointed) and its `pyvenv.cfg`, which says which Python and site-packages the
    environment is. A relative path (the tenant's own `.venv`) is named by itself."""
    if not interpreter.is_absolute():
        return {}
    found = {"real": str(interpreter.resolve())}
    for directory in (interpreter.parent.parent, interpreter.resolve().parent.parent):
        config = directory / "pyvenv.cfg"
        with contextlib.suppress(OSError):
            found["pyvenv.cfg"] = _sha(config)
            break
    return found


def code_files(module: str | None, script: Path | None, roots: Sequence[Path]) -> dict[str, str]:
    """The first-party code a service runs, `python -m <module>` or `python <script>`: every
    module found under `roots` (its PYTHONPATH; a script's directory too) that it imports,
    directly or not, by any import statement (also one inside a function), and each package's
    data files, as sha256 hex digests by path relative to its root."""
    roots = [*roots, *([script.parent] if script is not None else [])]
    todo: list[tuple[str, Path]] = []
    found: dict[str, str] = {}
    seen: set[str] = set()

    def add(name: str) -> None:
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            prefix = ".".join(parts[:end])
            if prefix not in seen:
                seen.add(prefix)
                if (located := _module_file(prefix, roots)) is not None:
                    todo.append((prefix, located))

    if module is not None:
        add(module)
        add(f"{module}.__main__")  # `-m` on a package runs it
    if script is not None and script.is_file():
        todo.append(("__main__", script))
    while todo:
        name, path = todo.pop()
        root = next(r for r in roots if path.is_relative_to(r))
        found[str(path.relative_to(root))] = _sha(path)
        if path.name == "__init__.py":
            for data in _package_data(path.parent):
                found[str(data.relative_to(root))] = _sha(data)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for imported in _imports(path, package):
            add(imported)
    return dict(sorted(found.items()))


def _imports(path: Path, package: str) -> list[str]:
    """The modules a file's import statements name (for `from a import b`, `a` and `a.b`)."""
    names = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                base = ".".join(parts[: len(parts) - node.level + 1])
                base = ".".join(p for p in (base, node.module or "") if p)
            else:
                base = node.module or ""
            if base:
                names.append(base)
                names += [f"{base}.{alias.name}" for alias in node.names if alias.name != "*"]
    return names


def _module_file(name: str, roots: Sequence[Path]) -> Path | None:
    parts = name.split(".")
    for root in roots:
        for candidate in (
            root.joinpath(*parts, "__init__.py"),
            root.joinpath(*parts[:-1], f"{parts[-1]}.py"),
        ):
            if candidate.is_file():
                return candidate
    return None


def _package_data(directory: Path) -> list[Path]:
    """A package's files other than modules, also in its subdirectories that are no packages."""
    found = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        if path.is_dir():
            if not (path / "__init__.py").exists():
                found += [
                    p
                    for p in sorted(path.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts
                ]
        elif path.suffix not in (".py", ".pyc"):
            found.append(path)
    return found


def _roots(plan: StackPlan, spec: ServiceSpec) -> list[Path]:
    if spec.interpreter == "sim":
        return [p if p.is_absolute() else plan.root / p for p in plan.sim_pythonpath]
    if spec.interpreter == "vision":
        return [p if p.is_absolute() else plan.root / p for p in plan.vision_pythonpath]
    return [Path(roboscript.__file__).resolve().parents[1]]


def _interpreter(plan: StackPlan, spec: ServiceSpec) -> Path:
    if isinstance(spec.interpreter, Path):
        return spec.interpreter
    named = getattr(plan.interpreters, spec.interpreter)
    if named is None:
        raise ValueError(f"service {spec.name!r} needs interpreters.{spec.interpreter}")
    return Path(named)


def _arg(arg: str, root: Path) -> str:
    """An argument as it counts for the key: a file it names by its content, a config it names
    (`configs/<name>.yaml`, as `load_config_ref` finds it) by the config's content; a YAML file by
    the data it holds, without `THROUGHPUT_KNOBS` (comments, layout and key order do not count;
    one that does not parse counts by its bytes)."""
    path = Path(arg)
    for candidate in (
        path if path.is_absolute() else root / path,
        root / "configs" / f"{arg}.yaml",
    ):
        with contextlib.suppress(OSError, ValueError):
            if candidate.is_file():
                return _config_digest(candidate)
    return arg


def _config_digest(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(raw)
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in THROUGHPUT_KNOBS}
            return "yaml:" + hashlib.sha256(canonical_json(data).encode()).hexdigest()
        except (yaml.YAMLError, TypeError, ValueError):
            pass
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat(pid: int) -> tuple[str | None, str | None]:
    """A process's state and start time (`/proc/<pid>/stat`), None where there is no `/proc`."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None, None
    fields = stat.rsplit(")", 1)[1].split()
    return fields[0], fields[19]
