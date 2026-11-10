"""The keeper of a shared stack: `python -m roboscript.stack.keeper <registry> <stack id>`, spawned
by `StackRegistry.spawn` in a session of its own (docs/concepts/operations.md).

It takes up the stack only if its record names it (a keeper spawned twice leaves), stops what a
keeper before it left running (a service's process group whose command names this stack), then
runs each service in a stack of its own on the record's GPU with the tenants' blob stores listed
(`RS_BLOB_ROOTS`), and records its URL. One that dies restarts on its port after each of the
plan's `restart_waits_s` (on a new port if that one is taken, bumping `generation`), or, if its
log shows it ran out of GPU memory, after `PRESSURE_WAIT_S` without counting, for
`pressure_retry_s` at most; one that stays up but fails `/readyz` `UNHEALTHY_CHECKS` times in a
row (checked every `HEALTH_S`; checks that got no answer while the host stood still on I/O or
memory are excused, `UNHEALTHY_CHECKS` at most, `stack.stall.Excuse`) restarts the same way, the
keeper owning that check because a service cannot always exit on its own (a wedged GPU context
after an out-of-memory); after the last wait it is `down`, and so is the stack, which then stops.
One ready for the plan's
`healthy_s` straight has its restarts back. Every `PUBLISH_S` it replaces its claims in the
plan's claims file with the memory its services will take and do not show yet (the larger of the
plan's `mib` and the most their processes showed, less what they show). Once no tenant's lease
has lived for the plan's `linger_s`, or on SIGTERM, it stops the services and removes the record
(a `down` one stays, for its error).
"""

import argparse
import contextlib
import logging
import os
import signal
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import anyio
import httpx

from roboscript.kernel.signals import Stopped, cancel_on_signals
from roboscript.stack import stall
from roboscript.stack.config import ServiceSpec, StackConfig
from roboscript.stack.gpus import (
    CLAIMS_MINE,
    GpuProbe,
    LogMarks,
    ProcessTable,
    SharedClaims,
    family,
    nvidia_smi,
    ps,
)
from roboscript.stack.shared import (
    POLL_S,
    SPAWN_S,
    Proc,
    ServiceState,
    StackRecord,
    StackRegistry,
    State,
)
from roboscript.stack.supervisor import StackError, up

log = logging.getLogger(__name__)

PUBLISH_S: Final = 20.0
PRESSURE_WAIT_S: Final = 30.0
LEFTOVER_STOP_S: Final = 10.0
HEALTH_S: Final = 10.0  # between `/readyz` checks of a running service
HEALTH_TIMEOUT_S: Final = 5.0
UNHEALTHY_CHECKS: Final = 3  # consecutive failed checks that make a service restart


@dataclass
class _Hold:
    """A service's memory as the keeper claims it: its estimate, the processes it started and
    the most they showed."""

    mib: float
    roots: set[int] = field(default_factory=set)
    peaks: dict[int, float] = field(default_factory=dict)
    peak: float = 0.0

    def held(self, shown: dict[int, float], table: dict[int, tuple[int, int]]) -> float:
        pids = family(table, self.roots)
        now = {pid: shown[pid] for pid in pids if pid in shown}
        self.peaks = {pid: max(self.peaks.get(pid, 0.0), mib) for pid, mib in now.items()}
        self.peak = max(self.peak, sum(self.peaks.values()))
        return max(0.0, max(self.mib, self.peak) - sum(now.values()))


class Keeper:
    """One stack's services, tenants and claims (see the module docstring)."""

    def __init__(
        self,
        registry: StackRegistry,
        record: StackRecord,
        *,
        probe: GpuProbe = nvidia_smi,
        processes: ProcessTable = ps,
    ) -> None:
        self.registry, self.id, self.gpu = registry, record.id, record.gpu
        self.plan, self.directory = record.plan, registry.path(record.id)
        self.probe, self.processes = probe, processes
        self.holds = {
            spec.name: _Hold(self.plan.mib.get(spec.name, 0.0)) for spec in self.plan.services
        }
        self.ports = {name: state.port for name, state in record.services.items()}
        self.restarts = {spec.name: 0 for spec in self.plan.services}
        self.down = anyio.Event()

    async def run(self) -> None:
        """Until the stack has no tenant for `linger_s`, or is down."""
        async with anyio.create_task_group() as group:
            for spec in self.plan.services:
                group.start_soon(self._service, spec)
            group.start_soon(self._publish)
            await self._tenancy()
            group.cancel_scope.cancel()

    async def close(self) -> None:
        """The record goes (a `down` one stays), and so do the claims."""
        async with self.registry.hold():
            record = self.registry.read(self.id)
            if record is not None and record.state != "down":
                self.registry.remove(self.id)
        if self.plan.claims is not None:
            claims = SharedClaims(self.plan.claims, self.directory / CLAIMS_MINE)
            async with claims.hold():
                claims.publish({})

    async def _tenancy(self) -> None:
        idle = None
        while True:
            async with self.registry.hold():
                record = self.registry.read(self.id)
                if record is None or record.state in ("down", "stopping"):
                    return
                if self.registry.tenants(self.id):
                    idle = None
                elif idle is None:
                    idle = anyio.current_time()
                elif anyio.current_time() - idle >= self.plan.linger_s:
                    record.state = "stopping"
                    self.registry.write(record)
                    log.info("stack %s: no tenant for %g s, stopping", self.id, self.plan.linger_s)
                    return
            with anyio.move_on_after(POLL_S):
                await self.down.wait()

    async def _service(self, spec: ServiceSpec) -> None:
        name, waits = spec.name, self.plan.restart_waits_s
        pressure_since = None
        while True:
            stack = self._stack(spec, self.ports[name])
            logs, hold = LogMarks(stack.run_dir), self.holds[name]
            hold.roots.clear()

            def started(_: str, pid: int, hold: _Hold = hold) -> None:
                hold.roots.add(pid)  # its own session: its group is its processes

            try:
                async with up(stack, on_start=started) as running:
                    url = running.endpoints[name]
                    port = urlsplit(url).port
                    moved = self.ports[name] is not None and port != self.ports[name]
                    self.ports[name] = port
                    await self._set(
                        name,
                        ServiceState(
                            state="up",
                            url=url,
                            port=port,
                            pid=running.pids[name],
                            restarts=self.restarts[name],
                        ),
                        moved=moved,
                    )
                    log.info("stack %s: %s up at %s", self.id, name, url)
                    await self._watch_health(name, url)
            except Exception as error:  # it did not come up, or died
                line = " ".join(str(error).split())[:500]
                now = anyio.current_time()
                if (
                    logs.out_of_memory()
                    and now - (pressure_since or now) <= self.plan.pressure_retry_s
                ):
                    pressure_since = pressure_since or now
                    wait = PRESSURE_WAIT_S
                elif (restarts := self.restarts[name]) >= len(waits):
                    log.error("stack %s: %s is down: %s", self.id, name, line)
                    await self._set(name, ServiceState(state="down", restarts=restarts, error=line))
                    self.down.set()
                    return
                else:
                    wait, self.restarts[name] = waits[restarts], restarts + 1
                if not isinstance(error, (StackError, OSError)):
                    log.exception("stack %s: %s failed", self.id, name)
                log.error("stack %s: %s failed, restarting in %g s: %s", self.id, name, wait, line)
                state = ServiceState(
                    state="restarting",
                    port=self.ports[name],
                    restarts=self.restarts[name],
                    error=line,
                )
                await self._set(name, state)
                await anyio.sleep(wait)
                if self.ports[name] is not None and not _port_free(self.ports[name]):
                    log.error(
                        "stack %s: port %s is taken: %s moves", self.id, self.ports[name], name
                    )
                    self.ports[name] = None

    async def _watch_health(self, name: str, url: str) -> None:
        """Until `UNHEALTHY_CHECKS` checks of `/readyz` in a row fail, then raise `StackError`,
        which restarts the service like an exit (a service that ran out of GPU memory stays up
        but answers 503 for good, so exits are not the only way one is lost). A check that got no
        answer after the host stood still on I/O or memory is excused, a check for every
        `HEALTH_S` stalled, no more than `UNHEALTHY_CHECKS` nor the checks that got no answer
        (`stack.stall.Excuse`, as the pipeline's own services: `orchestrate.launch.watch_ready`);
        a 503 always counts. Once it was ready for the plan's `healthy_s` straight, its restarts
        are back to 0."""
        deadline = UNHEALTHY_CHECKS * HEALTH_S
        failed, unanswered, excused, ready_since = 0, 0, 0, anyio.current_time()
        excuse = stall.Excuse(deadline)
        async with httpx.AsyncClient(trust_env=False) as http:
            while True:
                await anyio.sleep(HEALTH_S)
                try:
                    response = await http.get(f"{url}/readyz", timeout=HEALTH_TIMEOUT_S)
                    reason = (
                        None if response.status_code == 200 else f"answers {response.status_code}"
                    )
                except httpx.TransportError:
                    reason = "does not answer"
                now = anyio.current_time()
                if reason is not None:
                    silent = reason == "does not answer"
                    failed, unanswered, ready_since = failed + 1, unanswered + int(silent), now
                    # a stall excuses checks that got no answer, never a 503
                    excused = min(unanswered, int(excuse.mark(silent) // HEALTH_S))
                else:
                    failed, unanswered, excused, excuse = 0, 0, 0, stall.Excuse(deadline)
                    if self.restarts[name] and now - ready_since >= self.plan.healthy_s:
                        back = "stack %s: %s was ready for %g s, its restarts are back"
                        log.info(back, self.id, name, self.plan.healthy_s)
                        await self._restarts_back(name)
                if reason is not None:
                    log.warning(
                        "stack %s: %s /readyz %s (%d/%d, %d excused for host stalls)",
                        self.id, name, reason, failed - excused, UNHEALTHY_CHECKS, excused,
                    )  # fmt: skip
                if failed - excused >= UNHEALTHY_CHECKS:
                    message = f"service {name} /readyz {reason} {failed} times in a row"
                    message += f" ({excused} excused for host stalls)" if excused else ""
                    raise StackError(message)

    async def _restarts_back(self, name: str) -> None:
        self.restarts[name] = 0
        async with self.registry.hold():
            record = self.registry.read(self.id)
            if record is not None and name in record.services:
                record.services[name].restarts = 0
                self.registry.write(record)

    async def _set(self, name: str, service: ServiceState, *, moved: bool = False) -> None:
        async with self.registry.hold():
            record = self.registry.read(self.id)
            if record is None:
                return
            record.services[name] = service
            record.state = _state(record)
            record.error = service.error if service.state == "down" else None
            if moved:
                record.generation += 1
            if record.state == "up":
                record.spawns = 0
            self.registry.write(record)

    async def _publish(self) -> None:
        """What the services will take and do not show yet, in the claims file."""
        if self.plan.claims is None:
            return
        claims = SharedClaims(self.plan.claims, self.directory / CLAIMS_MINE)
        while True:
            try:
                memory, table = await self.probe(), await self.processes()
            except RuntimeError as error:
                log.error("stack %s: cannot read the GPUs: %s", self.id, error)
            else:
                shown = dict(memory[self.gpu].processes) if self.gpu in memory else {}
                held = sum(hold.held(shown, dict(table)) for hold in self.holds.values())
                try:
                    async with claims.hold():
                        claims.publish({self.gpu: held})
                except OSError:
                    log.exception("stack %s: could not publish its claims", self.id)
            await anyio.sleep(PUBLISH_S)

    def _stack(self, spec: ServiceSpec, port: int | None) -> StackConfig:
        plan = self.plan

        def absolute(paths: list[Path]) -> list[Path]:
            return [p if p.is_absolute() else plan.root / p for p in paths]

        named = spec.model_copy(
            update={
                "gpu": self.gpu,
                "args": [*spec.args, *(["--port", str(port)] if port is not None else [])],
                "env": {**spec.env, "RS_BLOB_ROOTS": str(self.directory / "roots")},
            }
        )
        return StackConfig(
            interpreters=plan.interpreters,
            sim_pythonpath=absolute(plan.sim_pythonpath),
            vision_pythonpath=absolute(plan.vision_pythonpath),
            run_dir=self.directory / "services" / spec.name,
            blob_dir=self.directory / "blobs",
            services=[named],
        )


def _state(record: StackRecord) -> State:
    states = [s.state for s in record.services.values()]
    if "down" in states:
        return "down"
    if all(s == "up" for s in states):
        return "up"
    if all(s == "starting" for s in states):
        return "starting"
    return "restarting"


async def _named(registry: StackRegistry, stack_id: str, me: int) -> StackRecord | None:
    """The record once it names this process its keeper; None if it names another that lives,
    or none in time, or the stack is gone."""
    with anyio.move_on_after(SPAWN_S):
        while True:
            async with registry.hold():
                record = registry.read(stack_id)
                if record is None or record.state in ("down", "stopping"):
                    return None
                if record.keeper is not None:
                    if record.keeper.pid == me:
                        return record
                    if record.keeper.alive():
                        return None
            await anyio.sleep(0.2)
    return None


def _leftovers(registry: StackRegistry, record: StackRecord) -> list[int]:
    """The services a keeper before this one left running: process groups the record names whose
    command still names this stack's directory (not a reused pid)."""
    mark, groups = str(registry.path(record.id)), []
    for state in record.services.values():
        if state.pid is None or not Proc(pid=state.pid).alive():
            continue
        with contextlib.suppress(OSError):
            command = Path(f"/proc/{state.pid}/cmdline").read_bytes().decode(errors="replace")
            if mark in command:
                groups.append(state.pid)
    return groups


async def _stop_leftovers(registry: StackRegistry, record: StackRecord) -> None:
    """Stop what `_leftovers` finds."""
    groups = _leftovers(registry, record)
    for group in groups:
        log.warning("stack %s: stopping process group %d a keeper before left", record.id, group)
        _signal(group, signal.SIGTERM)
    with anyio.move_on_after(LEFTOVER_STOP_S):
        while any(Proc(pid=group).alive() for group in groups):  # noqa: ASYNC110 - other processes
            await anyio.sleep(0.1)
    for group in groups:
        _signal(group, signal.SIGKILL)


async def keep(
    directory: Path, stack_id: str, *, probe: GpuProbe = nvidia_smi, processes: ProcessTable = ps
) -> None:
    """Keep the stack `stack_id` of the registry in `directory` (see the module docstring)."""
    registry = StackRegistry(directory)
    record = await _named(registry, stack_id, os.getpid())
    if record is None:
        log.info("stack %s is not this keeper's (%d): leaving", stack_id, os.getpid())
        return
    await _stop_leftovers(registry, record)
    keeper = Keeper(registry, record, probe=probe, processes=processes)
    try:
        with cancel_on_signals(signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            await keeper.run()
    finally:
        with anyio.CancelScope(shield=True):
            await keeper.close()


def _port_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _signal(group: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, sig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("registry", type=Path, help="the shared stacks' registry directory")
    parser.add_argument("stack", help="the stack's id")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    with contextlib.suppress(Stopped):
        anyio.run(keep, args.registry, args.stack)


if __name__ == "__main__":
    main()
