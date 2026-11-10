"""GPU placement for jobs (docs/concepts/operations.md, `orchestrate`): simulator places
on the GPUs whose services are up, by our count and by the memory the GPUs show (`Slots`), each
placement holding a `Reservation` while its job runs, jobs waiting in `Turn`s. Which turn a job gets
(its stage's rank, a spare unit) is the caller's."""

import contextlib
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar

import anyio

from roboscript.stack.gpus import GpuMemory, GpuProbe, ProcessTable, SharedClaims, family, ps

log = logging.getLogger(__name__)
T = TypeVar("T")

MIB_PER_GB: Final = 1000
"""A config's GB: 1000 of the MiB nvidia-smi reports, so that a cap of 46 is the 46000 MiB it
shows (not 46 GiB, 47104 MiB)."""
MEMORY_TTL_S: Final = 10.0
"""How long a reading of the GPUs' memory places jobs, and how often a job waiting for memory
looks again."""
PEAK_LEARN_S: Final = 300.0
"""A job that ran this long showed its task's memory (its scenes loaded, episodes opened): its
peak sets what the task's next jobs reserve (`Launch._need`)."""
OVER_BACKOFF_S: Final = 120.0
"""How long a GPU takes no job of ours after a reading showed its memory in use over the cap less
the margin (`Slots.over`): what grew there is not ours to count (another tenant, a simulator past
its peak), so it settles before we add to it."""


@dataclass(eq=False)
class Reservation:
    """Memory held on a GPU for a placement or a service, from `since` until it is released: the
    larger of `mib` (the estimate: a simulator keeps growing as it loads scenes long after it
    started) and `peak`, less what its processes show now (`seen`, at the last reading). Its
    processes are those of `roots` (pids or process groups, and those `groups` names as they
    come) and their descendants; `peaks` is the most each live one showed, and `peak` the most
    their peaks came to together, kept when one ends: a simulator frees and takes memory again as
    it loads each episode's scene, and a job's env server that is started again grows back to
    where it was. A job's placement `settles`: once it ran `PEAK_LEARN_S` and showed a peak, its
    estimate drops to that peak plus the slots' headroom (an estimate made before its task's peak
    was learned holds no more than the job takes).

    A job of `parts` alike parts (a P3 block's shards) gives back what a part held once the job
    declares it ended for good (`ended`: the process groups of its parts that ended, each a
    part, none of which the job starts again): its places (`weight`, `released` of them given
    back so far) and its share of the estimate, and the most its processes showed (`shares`,
    each process's group and most, kept after it ends) leaves the peak (`freed`). One part's
    places stay held until the job ends. A process that ended without the job declaring it (a
    crash, a service its stack restarts) keeps its share."""

    gpu: int
    mib: float
    since: float
    roots: set[int] = field(default_factory=set)
    groups: Callable[[], Iterable[int]] | None = None
    seen: float = 0.0
    peaks: dict[int, float] = field(default_factory=dict)
    peak: float = 0.0
    settles: bool = False
    weight: int = 0
    spare: bool = False
    preempting: bool = False
    parts: int = 1
    ended: Callable[[], Iterable[int]] | None = None
    shares: dict[int, tuple[int, float]] = field(default_factory=dict)
    gone: set[int] = field(default_factory=set)
    counted: int = 0
    released: int = 0
    freed: float = 0.0


SPARE_RANK: Final = 6
"""The rank of a spare unit's turn (a job the pipeline marks `spare`) that ran before, one more
for one that has not: behind every other turn of either placement order."""


@dataclass(frozen=True, order=True)
class Turn:
    """A job's place in the queue for GPU room, lower `rank` first, each rank in the order its
    jobs first asked. `finish_tasks`: the jobs that ran before and wait to run again first, then
    those that have not run; among each, a later stage of a started task (S2, P3) before an S1,
    so that tasks finish before new ones start (ranks 0 to 3). `by_stage`: S1s, then S2s, then
    P3s; among each, those that ran before first (ranks 0 to 5). A spare unit's turn
    (`SPARE_RANK` and up) comes after all of those, and places only while no other job waits
    (`Slots`). `again` is the turn of a job that ran and asks again (`rerun`, its rank then)."""

    rank: int
    number: int
    rerun: int = field(default=0, compare=False)

    def again(self) -> "Turn":
        return Turn(self.rerun, self.number, self.rerun)

    @property
    def spare(self) -> bool:
        return self.rank >= SPARE_RANK


@dataclass(eq=False)
class _Waiter:
    turn: Turn
    weight: int
    need: float
    name: str = ""
    since: float = 0.0


@dataclass(frozen=True)
class Placement:
    """A job's GPU, its services' URLs and the memory it reserved there."""

    gpu: int
    urls: dict[str, str]
    reservation: Reservation


class Slots:
    """Simulator places on the GPUs whose services are up, filled in `gpus` order: a job of
    `weight` places goes on the first GPU with room, `per_gpu` places of ours at most and, when a
    `probe` reads the GPUs' memory (at most every `MEMORY_TTL_S`), `need_mib` per place within
    `cap_mib`: the memory in use, plus what our reservations hold that their processes do not
    show now (`Reservation`: the larger of the estimate and the peak, for as long as the job
    runs; `processes` tells whose processes are whose), plus what others claimed in the shared
    `claims` file; `jobs` jobs at once. `take` waits for room and returns the placement; None
    when every GPU is down for good after one came up. Waiting jobs place in their `Turn`s: one
    may take room only while no job ahead of it could take room now (a smaller job fills room a
    bigger one ahead of it cannot use). A GPU's services start on demand: when waiting jobs need
    more places than the open GPUs and those coming up can give, `start` is called for the first
    GPU without services that has room for their `services_mib` and the first such job (without
    `start`, only the GPUs `open` names take jobs); `reserve` waits for room for them, and a GPU
    whose services `stopped` can start again. With `services_of`, what a GPU's services still
    need there (0 where a shared stack already runs them), and `enter` waits for a shared stack
    in their stead. A probe that fails places by our count alone (`probe_error`, which the status
    shows). A GPU whose reading shows more in use than `cap_mib` takes nothing of ours for
    `OVER_BACKOFF_S` after (`over`). With `headroom_mib`, a job's placement `settles` to its own
    peak plus that headroom once it ran `PEAK_LEARN_S`. With `by_stage`, turns rank by stage
    (`Turn`). A spare turn (`turn(spare=True)`) places only while every job waiting is spare,
    and starts a GPU's services only then; once placed, it runs on when other jobs come (they
    take the next room) unless one of them waits long for room only spare units hold
    (`preempt`)."""

    def __init__(
        self,
        gpus: Sequence[int],
        per_gpu: int,
        jobs: int,
        need_mib: float = 0.0,
        probe: GpuProbe | None = None,
        *,
        cap_mib: float = math.inf,
        claims: SharedClaims | None = None,
        processes: ProcessTable = ps,
        start: Callable[[int], None] | None = None,
        services_mib: float = 0.0,
        services_of: Callable[[int], float] | None = None,
        headroom_mib: float | None = None,
        by_stage: bool = False,
    ) -> None:
        self.by_stage = by_stage
        self.used = dict.fromkeys(gpus, 0)
        self.urls: dict[int, dict[str, str]] = {}  # the GPUs whose services are up
        self.started: set[int] = set()  # the GPUs whose services run or come up
        self.opening: set[int] = set()  # started, not up yet since
        self.unreserved: set[int] = set()  # started, their services' memory not reserved yet
        self.opened = False  # whether a GPU's services came up
        self.since: dict[int, float] = {}  # when each GPU last had no job of ours
        self.start, self.services_mib = start, services_mib
        self.services_of = services_of or (lambda gpu: services_mib)
        self.down: dict[int, str] = {}  # the GPUs out for good, and why
        self.per_gpu, self.max_jobs, self.need_mib = per_gpu, jobs, need_mib
        self.probe, self.cap_mib, self.claims = probe, cap_mib, claims
        self.headroom_mib = headroom_mib
        self.processes = processes
        self.memory: dict[int, GpuMemory] | None = None  # the last reading
        self.claimed: dict[int, float] = {}  # GiB others claimed, as last read
        self.reservations: list[Reservation] = []
        self.over: dict[int, float] = {}  # when a reading last showed a GPU over the cap
        self.probe_error: str | None = None
        self.jobs = 0
        self._read = -math.inf
        self.waiting: list[_Waiter] = []
        self._turns = 0
        self._placing = anyio.Lock()
        self._changed = anyio.Event()

    def turn(self, first: bool = False, stage: int | None = None, spare: bool = False) -> Turn:
        """A new job's turn, after every turn given before of its rank: a `spare` unit's behind
        every other; with `by_stage`, by its `stage` (`STAGE_ORDER`'s index); else a task's
        `first` stage (S1) behind the later stages of tasks that started."""
        self._turns += 1
        if spare:
            return Turn(SPARE_RANK + 1, self._turns, SPARE_RANK)
        if self.by_stage and stage is not None:
            return Turn(2 * stage + 1, self._turns, 2 * stage)
        rank = 3 if first else 2
        return Turn(rank, self._turns, rank % 2)

    async def take(
        self, weight: int, turn: Turn | None = None, need: float | None = None, name: str = ""
    ) -> Placement | None:
        """A placement of `weight` places needing `need` MiB (default `need_mib` per place), for
        the job `name`."""
        need = self.need_mib * weight if need is None else need
        waiter = _Waiter(turn or self.turn(), weight, need, name, anyio.current_time())
        self.waiting.append(waiter)
        try:
            while len(self.down) < len(self.used):
                async with self._placing:
                    await self._memory()
                    async with self._claims() as claims:
                        if waiter.turn.spare and self._others_wait():
                            pass  # it waits for the queue to empty (its leaving wakes us)
                        elif (gpu := self._pick(weight, need)) is not None:
                            if not self._ahead(waiter):
                                self.used[gpu] += weight
                                self.jobs += 1
                                reservation = self._reserve(gpu, need, claims)
                                reservation.settles = self.headroom_mib is not None
                                reservation.weight = weight
                                reservation.spare = waiter.turn.spare
                                return Placement(gpu, dict(self.urls[gpu]), reservation)
                            self._wake()  # a job ahead fits: it places first
                        else:
                            self._demand()
                await self._wait()
            while self.start is not None and not self.opened:  # the launch stops (`none_opened`)
                await self._wait()
            return None
        finally:
            self.waiting.remove(waiter)
            self._wake()  # the queue moved

    def give(self, placement: Placement, weight: int) -> None:
        """The placement of `weight` places ends (less the places it gave back already)."""
        self.used[placement.gpu] -= weight - placement.reservation.released
        if self.used[placement.gpu] == 0:
            self.since[placement.gpu] = anyio.current_time()
        self.jobs -= 1
        self.release(placement.reservation)

    async def reserve(self, gpu: int, mibs: Sequence[float]) -> list[Reservation]:
        """Wait until `gpu` has room for all of `mibs` (by memory, not by places), and reserve
        each."""
        while True:
            async with self._placing:
                await self._memory()
                async with self._claims() as claims:
                    if self._room(gpu, sum(mibs)):
                        self.unreserved.discard(gpu)
                        return [self._reserve(gpu, mib, claims) for mib in mibs]
            await self._wait()

    async def enter(self, gpu: int, attach: Callable[[bool], Awaitable[T | None]]) -> T:
        """Wait until `attach(room)`, called under the placing and claims locks with whether
        `gpu` has room for the services still to start there (`services_of`) now, gives a result
        (shared stacks attached to, or started where there is room: `Launch._enter`)."""
        while True:
            async with self._placing:
                await self._memory()
                async with self._claims():
                    found = await attach(self._room(gpu, self.services_of(gpu)))
                    if found is not None:
                        self.unreserved.discard(gpu)
                        return found
            await self._wait()

    async def preempt(self, after_s: float) -> list[tuple[Reservation, str]]:
        """The spare placements to stop, each with the job it makes room for: for the first job
        in turn that is not a spare unit, waited `after_s` or longer and fits nowhere now, the
        newest spare placements on the first GPU (in `gpus` order, services up) where stopping
        them gives it room (places, jobs at once and memory: what they hold, shown or reserved),
        as few as do; none while placements stopped before are still held (`preempting`), so
        that one wait stops no more than it needs, nor while a GPU's services come up."""
        async with self._placing:
            await self._memory()
            async with self._claims():
                if self.opening or any(r.preempting for r in self.reservations):
                    return []  # a GPU coming up, or what was stopped, may give it room yet
                now = anyio.current_time()
                for waiter in sorted(self.waiting, key=lambda w: w.turn):
                    if waiter.turn.spare or now - waiter.since < after_s:
                        continue
                    if self._pick(waiter.weight, waiter.need) is not None:
                        continue  # room is not what it waits for
                    for gpu in self.used:
                        if (chosen := self._freeing(gpu, waiter)) is not None:
                            for reservation in chosen:
                                reservation.preempting = True
                            return [(reservation, waiter.name) for reservation in chosen]
        return []

    def _freeing(self, gpu: int, waiter: _Waiter) -> list[Reservation] | None:
        """The newest spare placements on `gpu` whose stopping gives `waiter` room there, as few
        as do (None: none do)."""
        if gpu not in self.urls or self._backing_off(gpu):
            return None
        spares = sorted(
            (r for r in self.reservations if r.gpu == gpu and r.spare),
            key=lambda r: r.since,
            reverse=True,
        )
        room = self._room_mib(gpu)
        chosen: list[Reservation] = []
        for reservation in spares:
            chosen.append(reservation)
            places = self.used[gpu] - sum(r.weight for r in chosen) + waiter.weight
            jobs = self.jobs - len(chosen) < self.max_jobs
            freed = sum(r.seen + self._held(r) for r in chosen)
            memory = self.memory is None or (room is not None and waiter.need <= room + freed)
            if places <= self.per_gpu and jobs and memory:
                return chosen
        return None

    async def publish(self) -> None:
        """Replace our claims in the shared claims file with what our reservations hold now that
        nvidia-smi does not show yet, per GPU: other tenants see the room they really have."""
        if self.claims is None:
            return
        async with self._placing:
            await self._memory()
            async with self._claims() as claims:
                assert claims is not None
                claims.publish({gpu: self._reserved_mib(gpu) for gpu in self.used})

    def release(self, reservation: Reservation) -> None:
        with contextlib.suppress(ValueError):  # settled already
            self.reservations.remove(reservation)
        self._wake()

    def open(self, gpu: int, urls: Mapping[str, str]) -> None:
        """The GPU's services are up at `urls`."""
        self.urls[gpu] = dict(urls)
        self.started.add(gpu)
        self.opening.discard(gpu)
        self.opened = True
        self.since.setdefault(gpu, anyio.current_time())
        self._wake()

    def close(self, gpu: int) -> None:
        """The GPU's services are down for now: no job goes there (nor counts on it coming)."""
        self.urls.pop(gpu, None)
        self.opening.discard(gpu)
        self._wake()  # a job waiting for it may need another GPU

    def fail(self, gpu: int, error: str) -> None:
        """The GPU takes no more jobs."""
        self.urls.pop(gpu, None)
        self.opening.discard(gpu)
        self.down[gpu] = error
        self._wake()

    def idle(self, gpu: int, seconds: float) -> bool:
        """Whether the GPU's services are up and no job of ours was on it for `seconds`: if so,
        it takes no job from now on (its services stop, then `stopped`)."""
        now, since = anyio.current_time(), self.since.get(gpu, math.inf)
        if gpu not in self.urls or self.used[gpu] > 0 or now - since < seconds:
            return False
        self.urls.pop(gpu)
        return True

    def stopped(self, gpu: int) -> None:
        """The GPU's services stopped (their reservations go: no job of ours is on it): a job
        that needs it starts them again."""
        self.urls.pop(gpu, None)
        self.reservations = [r for r in self.reservations if r.gpu != gpu]
        for gpus in (self.started, self.opening, self.unreserved):
            gpus.discard(gpu)
        self.since.pop(gpu, None)
        self._wake()

    async def none_opened(self) -> None:
        """Wait until every GPU is down for good before one's services came up."""
        while self.opened or len(self.down) < len(self.used):
            await self._changed.wait()

    def ledger(self, gpu: int) -> dict[str, float | bool | None]:
        """The GPU's memory in GB (`MIB_PER_GB`) as the scheduler counts it: in use (all of it,
        and what our placements' processes show of it), reserved by us and not yet shown, claimed
        by others, the room left under the cap (None without a reading), and whether it backs
        off (`OVER_BACKOFF_S`)."""
        found = None if self.memory is None else self.memory.get(gpu)
        room = self._room_mib(gpu)
        ours = sum(r.seen for r in self.reservations if r.gpu == gpu)
        return {
            "used_gb": None if found is None else round(found.used_mib / MIB_PER_GB, 1),
            "ours_gb": round(ours / MIB_PER_GB, 1),
            "reserved_gb": round(self._reserved_mib(gpu) / MIB_PER_GB, 1),
            "claimed_gb": round(self.claimed.get(gpu, 0.0) * 1024 / MIB_PER_GB, 1),
            "room_gb": None if room is None else round(room / MIB_PER_GB, 1),
            "backing_off": self._backing_off(gpu),
        }

    def _others_wait(self) -> bool:
        """Whether a job that is not a spare unit waits for room."""
        return any(not other.turn.spare for other in self.waiting)

    def _ahead(self, waiter: _Waiter) -> bool:
        """Whether a job ahead of `waiter` in the queue could take room now."""
        return any(
            other.turn < waiter.turn and self._pick(other.weight, other.need) is not None
            for other in self.waiting
        )

    def _demand(self) -> None:
        """Start a GPU's services if the waiting jobs that can place nowhere now (as many as may
        run) need more places than the GPUs coming up can give."""
        if self.start is None or self.jobs >= self.max_jobs:
            return
        others = self._others_wait()  # spare units ask for services only when no other job waits
        blocked = sorted(
            (
                w
                for w in self.waiting
                if not (others and w.turn.spare) and self._pick(w.weight, w.need) is None
            ),
            key=lambda w: w.turn,
        )[: self.max_jobs - self.jobs]
        coming = sum(self._capacity(gpu) for gpu in self.opening)
        if sum(w.weight for w in blocked) <= coming:
            return
        cold = [gpu for gpu in self.used if gpu not in self.started and gpu not in self.down]
        services = {gpu: self.services_of(gpu) for gpu in cold}
        for waiter in blocked:
            for gpu in cold:
                if self._room(gpu, services[gpu] + waiter.need):
                    self.started.add(gpu)
                    self.opening.add(gpu)
                    if services[gpu] > 0:
                        self.unreserved.add(gpu)
                    self.start(gpu)
                    self._wake()  # the other waiting jobs count it as coming
                    return

    def _capacity(self, gpu: int) -> int:
        """The places a GPU coming up will have, by our count and its room (its services'
        memory taken, reserved or not yet)."""
        places = self.per_gpu - self.used[gpu]
        if self.memory is None or self.need_mib <= 0:
            return places
        room = self._room_mib(gpu) or 0.0
        if gpu in self.unreserved:
            room -= self.services_mib
        return max(0, min(places, int(room // self.need_mib)))

    def _pick(self, weight: int, need: float) -> int | None:
        if self.jobs >= self.max_jobs:
            return None
        for gpu in self.used:
            places = self.used[gpu] + weight <= self.per_gpu
            if gpu in self.urls and places and self._room(gpu, need):
                return gpu
        return None

    def _room(self, gpu: int, mib: float) -> bool:
        """Whether `mib` more fits on `gpu` under the cap (always, without a reading), and the
        GPU does not back off."""
        room = self._room_mib(gpu)
        if self.memory is None:
            return True
        return room is not None and mib <= room and not self._backing_off(gpu)

    def _backing_off(self, gpu: int) -> bool:
        return anyio.current_time() - self.over.get(gpu, -math.inf) < OVER_BACKOFF_S

    def _room_mib(self, gpu: int) -> float | None:
        """What is left on `gpu` under the cap: its memory in use, what our reservations hold
        that their processes do not show now and what others claimed (None: no reading)."""
        if self.memory is None or (found := self.memory.get(gpu)) is None:
            return None
        committed = found.used_mib + self._reserved_mib(gpu) + self.claimed.get(gpu, 0.0) * 1024
        # claims are in GiB
        return min(self.cap_mib, found.total_mib) - committed

    def _reserved_mib(self, gpu: int) -> float:
        return sum(self._held(r) for r in self.reservations if r.gpu == gpu)

    def _reserve(self, gpu: int, mib: float, claims: SharedClaims | None) -> Reservation:
        reservation = Reservation(gpu, mib, anyio.current_time())
        self.reservations.append(reservation)
        if claims is not None and mib > 0:
            claims.add(gpu, mib)
        return reservation

    def _held(self, reservation: Reservation) -> float:
        """What a reservation holds that its processes do not show now: of its peak, not what its
        ended parts' processes showed, nor less than its live processes' peaks."""
        peak = max(reservation.peak - reservation.freed, sum(reservation.peaks.values()))
        return max(0.0, max(reservation.mib, peak) - reservation.seen)

    def _shrink(self, reservation: Reservation) -> None:
        """Give back what the parts its job declared ended held (`Reservation`)."""
        if reservation.ended is None or reservation.parts <= 1:
            return
        new = set(reservation.ended()) - reservation.gone
        if not new:
            return
        reservation.gone |= new
        reservation.freed += sum(mib for group, mib in reservation.shares.values() if group in new)
        count = min(len(reservation.gone), reservation.parts - 1)
        if count > reservation.counted:
            part = (reservation.weight + reservation.released) // reservation.parts
            back = part * count - reservation.released
            self.used[reservation.gpu] -= back
            reservation.weight -= back
            reservation.released += back
            left = reservation.parts - reservation.counted
            reservation.mib *= (reservation.parts - count) / left
            reservation.counted = count
        self._wake()

    @contextlib.asynccontextmanager
    async def _claims(self) -> AsyncIterator[SharedClaims | None]:
        """The shared claims file held, others' claims read into `claimed` (None: no file)."""
        if self.claims is None:
            yield None
            return
        async with self.claims.hold() as claims:
            self.claimed = claims.others()
            yield claims

    async def _memory(self) -> None:
        if self.probe is None:
            for reservation in self.reservations:
                self._shrink(reservation)
            return
        if anyio.current_time() - self._read < MEMORY_TTL_S:
            return
        try:
            memory = dict(await self.probe())
            table = await self.processes() if self.reservations else {}
        except RuntimeError as error:
            if self.probe_error is None:
                log.error("placing jobs by our count alone: %s", error)
            self.memory, self.probe_error = None, str(error)
        else:
            self.memory, self.probe_error = memory, None
            for gpu in self.used:
                if gpu in memory and memory[gpu].used_mib > self.cap_mib:
                    if not self._backing_off(gpu):
                        log.warning(
                            "GPU %d: %.1f GB in use, over the cap less the margin: no job of "
                            "ours goes there for %g s",
                            gpu,
                            memory[gpu].used_mib / MIB_PER_GB,
                            OVER_BACKOFF_S,
                        )
                    self.over[gpu] = anyio.current_time()
            for reservation in self.reservations:
                groups = reservation.groups() if reservation.groups is not None else ()
                pids = family(table, {*reservation.roots, *groups})
                shown = memory[reservation.gpu].processes if reservation.gpu in memory else {}
                now = {pid: shown[pid] for pid in pids if pid in shown}
                peaks = reservation.peaks
                reservation.peaks = {pid: max(peaks.get(pid, 0.0), mib) for pid, mib in now.items()}
                for pid, mib in reservation.peaks.items():
                    reservation.shares[pid] = (table[pid][1], mib)
                reservation.seen = sum(now.values())
                reservation.peak = max(reservation.peak, sum(reservation.peaks.values()))
                ran = anyio.current_time() - reservation.since
                if reservation.settles and reservation.peak > 0 and ran >= PEAK_LEARN_S:
                    assert self.headroom_mib is not None
                    reservation.mib = min(reservation.mib, reservation.peak + self.headroom_mib)
                self._shrink(reservation)
        self._read = anyio.current_time()

    async def _wait(self) -> None:
        with anyio.move_on_after(math.inf if self.probe is None else MEMORY_TTL_S):
            await self._changed.wait()

    def _wake(self) -> None:
        self._changed.set()
        self._changed = anyio.Event()
