"""The P4 round driver (docs/concepts/learning.md): evaluate -> count -> reflect -> freeze -> next
round, over a seed chain (`roboscript.learning.seeds`), resumable, append-only.

A P4 run is a directory. `p4.json` (`P4Plan`) is its frozen plan: the code, the base run, each
task's Task Memory (the version the base run's P3 read), the seed chain, the episodes, shards and
modes, the method, the recipe and models, the gates, and where round 1's document came from; adding
rounds appends a segment and never changes one. Round r reflects on the records of its blocks
(round 1: the base run's first blocks, block 1 by default or blocks 1 to k
(`new_split(first_read=k)`), imported as `rounds/r1/`; round r > 1: both arms of round r-1's
evaluation block) into `G<r>`, frozen in `rounds/r<r>/rsi/`, and is evaluated on its own block;
each segment's final block is evaluated with the document of its last round, and no reflection ever
reads a final block. Two arms per evaluated block, paired by seed: the baseline (no document:
`baseline/block<N>/`) and the RSI arm (`rounds/r<r>/eval/block<N>/`, `final/block<N>/`).

Two lanes run at once: the baseline lane evaluates the baseline's blocks in order (every round's
and every final block), from the start, and starts its next block early once every stage being
evaluated has all its tasks started (`Evaluator.waiting`), so that it fills the machines a stage's
last long tasks leave idle without taking them from a stage that still waits for room; the RSI lane
goes round by round. After both arms of round r's block are evaluated, the round is counted
(`counters.update_counters` on the paired clean outcomes and the entry ids the RSI arm's Actor
cited: `rounds/r<r>/G<r>.counted.json`, `counts.json`) and reported (`rounds/r<r>/report.md`,
`report.json`). A round named in the plan's `gates` then waits for an approval (`approve`, appended
to `approvals.jsonl`) before round r+1 reflects; the baseline lane goes on. A run whose lanes have
nothing left but a gate ends ("waiting"), and a resume after the approval goes on.

Every step is decided from the files it leaves (an evaluated stage's `p4-done.json`, a round's
frozen document, its counts and report), so a run killed at any point resumes where it stopped;
`state.json` records the phases for people and `status`. The seed guard: the chain is checked when
the plan is made and loaded (`SeedSplit`: no overlap, no final block read, no round reading its own
evaluation block), each reflection's seeds lie in its round's read blocks (`check_reflection`, in
`reflect`) and each evaluated stage's seeds in its block, else the run stops.

A plan may evaluate only some of its stages (`P4Plan.only`, e.g. one variant's RSI arm on one
block, paired afterwards against another run's baseline on the same seeds): the stages left out are
never evaluated, a round is counted only when both its arms are evaluated here, no round is
reflected after the last one it evaluates, and the run writes no final report.

The benchmark is the evaluator's (`Evaluator`); the reflection's wiring (Hub, brains, models,
images) the reflector's (`Reflector`). This module knows neither.
"""

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import anyio
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.ids import TaskName
from roboscript.learning.eval_view import read_eval_episode
from roboscript.learning.seeds import SeedRange, SeedSplit, check_reflection, extend_chain
from roboscript.orchestrate.dispatch.agent import start_token
from roboscript.records import write_atomic
from roboscript_methods.rsi.counters import PairedOutcome, update_counters
from roboscript_methods.rsi.digest import digest_episode
from roboscript_methods.rsi.doc import EpisodeKey, RsiDoc
from roboscript_methods.rsi.recipe import RsiRecipe
from roboscript_methods.rsi.reflect import RECORD, Reflected
from roboscript_methods.rsi.report import ArmEpisode, final_report, render, round_report
from roboscript_methods.rsi.store import FileRsiStore, round_name

log = logging.getLogger(__name__)

PLAN: Final = "p4.json"
STATE: Final = "state.json"
APPROVALS: Final = "approvals.jsonl"
CLAIM: Final = "driver.json"
DONE: Final = "p4-done.json"
"""An evaluated stage's record, in its directory: written once its evaluator finished it."""
STORE: Final = "rsi"
"""A round's document store, under `rounds/r<r>/`."""
FIRST_READ: Final = 1
"""Round 1 reflects on the base run's blocks from block 1 (docs/concepts/learning.md): block 1 alone
by default."""

Arm = Literal["baseline", "rsi"]
Mode = str


def block_name(block: int) -> str:
    return f"block{block}"


def block_number(name: str) -> int:
    if not name.startswith("block") or not name[5:].isdigit():
        raise ValueError(f"{name!r} is not a block name (block<N>)")
    return int(name[5:])


def gate_name(round: int) -> str:
    return f"r{round}"


class Segment(BaseModel):
    """The plan's rounds up to `rounds` and the final block `final` they are tested on, made at
    `at` (UTC)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rounds: PositiveInt
    final: NonNegativeInt
    at: str


class Imported(BaseModel):
    """Round 1's document, reflected elsewhere (`import_round`): the directory it was copied
    from and the sha256 of its `G1.json`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    sha256: str


class P4Plan(BaseModel):
    """`p4.json`: see the module. `tasks`: each task's Task Memory; `excluded`: the tasks left out
    and why; `split`: the chain (block names `block<N>`); `models`: the reflection's and the
    evaluation's, as the launcher records them; `gates`: rounds (`r<N>`) that wait for an
    approval after they are counted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    code: str = Field(min_length=1)
    base: str = Field(min_length=1)
    tasks: dict[TaskName, ArtifactRef] = Field(min_length=1)
    excluded: dict[TaskName, str] = {}
    split: SeedSplit
    segments: tuple[Segment, ...] = Field(min_length=1)
    episodes: int = Field(ge=1, le=100)
    shards: PositiveInt
    eval_modes: tuple[Mode, ...]
    final_modes: tuple[Mode, ...]
    count_mode: Mode
    method: str = Field(min_length=1)
    recipe: str = Field(min_length=1)
    models: dict[str, Any] = {}
    gates: tuple[str, ...] = ()
    imported: Imported | None = None
    only: tuple[str, ...] = ()
    """The stages this run evaluates, by name (`stages`); empty: every one. A round's document
    past round 1 is reflected on both arms of the round before it, so an RSI stage of round
    r > 1 needs both arms of rounds 1 to r-1 in it."""

    @model_validator(mode="after")
    def _chain(self) -> Self:
        rounds = list(self.split.reflects)
        if len(rounds) != self.segments[-1].rounds:
            raise ValueError(f"the chain has {len(rounds)} rounds, the segments {self.segments}")
        first = self.split.reflects[rounds[0]]
        if first != _first_reads(len(first)):
            raise ValueError(
                f"round 1 reflects on {list(first)}, not on blocks from {block_name(FIRST_READ)} "
                "on in order"
            )
        finals = tuple(block_name(s.final) for s in self.segments)
        if finals != self.split.final:
            raise ValueError(f"final blocks {self.split.final}, the segments' {finals}")
        if self.split.transductive:
            raise ValueError("a P4 run never evaluates on a block it reflected on (transductive)")
        if self.count_mode not in self.eval_modes or self.count_mode not in self.final_modes:
            raise ValueError(f"the count mode {self.count_mode} is evaluated in every block")
        known = {gate_name(r) for r in range(1, len(rounds) + 1)}
        if unknown := sorted(set(self.gates) - known):
            raise ValueError(f"gates {unknown} name no round of {sorted(known)}")
        if self.only:
            names = {s.name for s in self.stages()}
            if unknown := sorted(set(self.only) - names):
                raise ValueError(f"only names no stage of the plan: {unknown} (stages {names})")
            for stage in self.stages():
                if stage.arm != "rsi" or stage.name not in self.only or (stage.round or 1) == 1:
                    continue
                for r in range(1, stage.round or 1):
                    block = block_name(self.eval_block(r))
                    need = {f"baseline/{block}", f"rounds/r{r}/eval/{block}"}
                    if lacking := sorted(need - set(self.only)):
                        raise ValueError(
                            f"{stage.name} reads round {stage.round}'s document, reflected on "
                            f"round {r}'s arms: only lacks {lacking}"
                        )
        return self

    def selected(self, stage: "Stage") -> bool:
        """Whether this run evaluates `stage` (`only`)."""
        return not self.only or stage.name in self.only

    @property
    def rounds(self) -> int:
        return self.segments[-1].rounds

    def eval_block(self, round: int) -> int:
        return block_number(list(self.split.reflects)[round - 1])

    def read_blocks(self, round: int) -> tuple[int, ...]:
        return tuple(
            block_number(b) for b in self.split.reflects[block_name(self.eval_block(round))]
        )

    def stages(self) -> list["Stage"]:
        """Every stage the plan evaluates: the baseline's in block order, then the RSI arm's."""
        found = []
        for r in range(1, self.rounds + 1):
            block = self.eval_block(r)
            found.append(
                Stage(
                    name=f"baseline/{block_name(block)}",
                    arm="baseline",
                    block=block,
                    modes=self.eval_modes,
                )
            )
            found.append(
                Stage(
                    name=f"rounds/r{r}/eval/{block_name(block)}",
                    arm="rsi",
                    round=r,
                    block=block,
                    modes=self.eval_modes,
                )
            )
        for segment in self.segments:
            found.append(
                Stage(
                    name=f"baseline/{block_name(segment.final)}",
                    arm="baseline",
                    block=segment.final,
                    modes=self.final_modes,
                )
            )
            found.append(
                Stage(
                    name=f"final/{block_name(segment.final)}",
                    arm="rsi",
                    round=segment.rounds,
                    block=segment.final,
                    modes=self.final_modes,
                )
            )
        baseline = sorted((s for s in found if s.arm == "baseline"), key=lambda s: s.block)
        return baseline + [s for s in found if s.arm == "rsi"]


class Stage(BaseModel):
    """One evaluated block of an arm: its directory under the run (`name`), the round whose
    document it reads (None: the baseline), the block and its modes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arm: Arm
    round: PositiveInt | None = None
    block: NonNegativeInt
    modes: tuple[Mode, ...]

    @property
    def final(self) -> bool:
        return self.name.startswith("final/")


class StageEpisode(ArmEpisode):
    """An episode of a stage, as its evaluator read it: its mode and records directory (None:
    lost before it had one)."""

    mode: Mode
    directory: Path | None = None


class DocSource(BaseModel):
    """The document a stage reads: its frozen ref and the store directory it is loaded from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: ArtifactRef
    store: Path


class Evaluator(Protocol):
    """A benchmark's evaluation of a stage. `evaluate` runs it to the end (a stage started before
    resumes; a stage finished returns at once); `episodes` reads its episodes (the first
    `P4Plan.episodes` of each task, by official index, in each of its modes); `waiting`: whether
    some task of a stage being evaluated has not started (or waits to start again)."""

    async def evaluate(self, stage: Stage, doc: DocSource | None) -> None: ...

    def episodes(self, stage: Stage) -> list[StageEpisode]: ...

    def waiting(self, stage: Stage) -> bool: ...


@dataclass(frozen=True)
class RoundInput:
    """What a round reflects on: its number and directory (`rounds/r<r>/`, where it writes), the
    chain, the round's evaluation block's name, its episodes (each with its key and arm, of the
    read blocks only), the directories they were evaluated in (their blob stores), and the
    document it starts from (the previous round's after its counts)."""

    round: int
    directory: Path
    split: SeedSplit
    evaluation: str
    episodes: Sequence[tuple[StageEpisode, EpisodeKey]]
    sources: Sequence[Path]
    start: RsiDoc


Reflector = Callable[[RoundInput], Awaitable[Reflected]]


def new_split(rounds: int, block: Callable[[int], SeedRange], first_read: int = 1) -> SeedSplit:
    """The chain of `rounds` rounds whose first reflects on blocks 1 to `first_read` (k): round r
    is evaluated on block k+r, then a final block."""
    if first_read < 1:
        raise ValueError(f"round 1 reflects on at least one block, not {first_read}")
    reads = _first_reads(first_read)
    first = SeedSplit(blocks={name: block(block_number(name)) for name in reads})
    return extend_chain(first, rounds, _candidates(block, FIRST_READ + first_read), reads=reads)


def _first_reads(count: int) -> tuple[str, ...]:
    return tuple(block_name(FIRST_READ + i) for i in range(count))


def _candidates(
    block: Callable[[int], SeedRange], n: int = FIRST_READ + 1
) -> Iterable[tuple[str, SeedRange]]:
    while True:
        yield block_name(n), block(n)
        n += 1


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_plan(directory: Path, plan: P4Plan) -> None:
    write_atomic(directory / PLAN, (plan.model_dump_json(indent=2) + "\n").encode())


def load_plan(directory: Path) -> P4Plan:
    return P4Plan.model_validate_json((directory / PLAN).read_bytes())


def add_rounds(directory: Path, rounds: int, block: Callable[[int], SeedRange]) -> P4Plan:
    """Append `rounds` rounds on the next unused blocks and a new final block (`extend_chain`):
    the earlier rounds, their final blocks and the stages run on them stay. Refused while a
    driver runs the directory."""
    if (other := _claimant(directory)) is not None:
        raise RuntimeError(f"driver pid {other} runs {directory}: stop it before adding rounds")
    plan = load_plan(directory)
    split = extend_chain(plan.split, rounds, _candidates(block))
    added = block_number(split.final[-1])
    segment = Segment(rounds=plan.rounds + rounds, final=added, at=_utc())
    grown = P4Plan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "split": split.model_dump(mode="json"),
            "segments": [*plan.segments, segment],
        }
    )
    write_plan(directory, grown)
    _append(
        directory / "plan-changes.jsonl", {"at": segment.at, "added_rounds": rounds, "final": added}
    )
    return grown


def import_round(directory: Path, source: Path, plan: P4Plan | None = None) -> Imported:
    """Copy a round-1 reflection made elsewhere (that run's directory: G1.json,
    G1.md, edits, reflection.json, inputs/, reflect/, its store rsi/ ...) into `rounds/r1/`, which
    must not exist yet; its `G1.json` must be the store's frozen G1. With `plan`, the reflection
    (`reflection.json`) must have been made for the plan's round-1 evaluation block and read only
    seeds of the blocks round 1 reflects on (`check_reflection`)."""
    target = directory / "rounds" / "r1"
    if target.exists():
        raise ValueError(f"{target} exists: round 1 is imported once")
    name = round_name(1)
    doc_file = source / f"{name}.json"
    if not doc_file.is_file() or not (source / STORE).is_dir():
        raise ValueError(f"{source} is not a round-1 directory (no {name}.json or {STORE}/)")
    marker = source / STORE / name / "frozen.json"
    if not marker.is_file():
        raise ValueError(f"{source / STORE} has no frozen {name}")
    frozen = ArtifactRef.model_validate(json.loads(marker.read_bytes())["ref"])
    written = RsiDoc.model_validate_json(doc_file.read_bytes())
    if written.content_hash() != frozen.hash:
        raise ValueError(f"{doc_file} is not the frozen {name} of {source / STORE}")
    if plan is not None:
        record = json.loads((source / RECORD).read_bytes())
        evaluation = block_name(plan.eval_block(1))
        if (made := record["config"]["evaluation"]) != evaluation:
            raise ValueError(
                f"{source} was reflected for {made}, the plan evaluates round 1 on {evaluation}"
            )
        check_reflection(plan.split, evaluation, record["seeds"])
    shutil.copytree(source, target)
    digest = hashlib.sha256(doc_file.read_bytes()).hexdigest()
    return Imported(source=str(source), sha256=digest)


def approve(directory: Path, gate: str) -> None:
    """Record the approval of `gate` (`r<N>`): a running driver picks it up, a resume reads it."""
    plan = load_plan(directory)
    if gate not in plan.gates:
        raise ValueError(f"{gate} is not one of the run's gates {list(plan.gates)}")
    _append(directory / APPROVALS, {"gate": gate, "at": _utc()})


def approved(directory: Path) -> set[str]:
    path = directory / APPROVALS
    if not path.is_file():
        return set()
    return {
        json.loads(line)["gate"] for line in path.read_text(encoding="utf-8").splitlines() if line
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


def running(directory: Path) -> int | None:
    """The pid of the driver running `directory` (None: none; a recorded process that is gone,
    or whose pid another process took since, runs nothing)."""
    path = directory / CLAIM
    if not path.is_file():
        return None
    claim = json.loads(path.read_bytes())
    return int(claim["pid"]) if start_token(claim["pid"]) == claim["token"] else None


def _claimant(directory: Path) -> int | None:
    """Another process running `directory`."""
    pid = running(directory)
    return pid if pid != os.getpid() else None


def run_status(directory: Path, phases: dict[str, str] | None = None) -> dict[str, Any]:
    """Each stage's and round's phase, from the files (`phases`: a running driver's, for what
    the files do not show yet), written to `state.json`."""
    plan, phases = load_plan(directory), phases or {}
    stages = {
        s.name: "done"
        if (directory / s.name / DONE).is_file()
        else phases.get(s.name, "pending" if plan.selected(s) else "skipped")
        for s in plan.stages()
    }
    rounds = {}
    for r in range(1, plan.rounds + 1):
        here = directory / "rounds" / f"r{r}"
        if (here / "report.json").is_file():
            phase = "counted"
            if gate_name(r) in plan.gates:
                phase = "approved" if gate_name(r) in approved(directory) else "gated"
        elif (here / STORE / round_name(r) / "frozen.json").is_file():
            phase = "frozen"
        else:
            phase = phases.get(gate_name(r), "pending")
        rounds[gate_name(r)] = phase
    found: dict[str, Any] = {"stages": stages, "rounds": rounds, "at": _utc()}
    found["finished"] = (directory / "report.md").is_file()
    found["waiting"] = [g for g, v in rounds.items() if v == "gated"]
    found["driver"] = running(directory)
    write_atomic(directory / STATE, (json.dumps(found, indent=2) + "\n").encode())
    return found


@dataclass
class Driver:
    """Runs a P4 directory (see the module). `recipe`: the RSI recipe (the counting rules and
    the digest's); `poll_s`: how often a lane waiting at a gate reads the approvals."""

    directory: Path
    evaluator: Evaluator
    reflector: Reflector
    recipe: RsiRecipe
    poll_s: float = 60.0
    _done: dict[str, anyio.Event] = field(default_factory=dict)
    _phases: dict[str, str] = field(default_factory=dict)

    async def run(self) -> dict[str, Any]:
        """Both lanes until every stage is evaluated and every round counted, or until the RSI
        lane waits at a gate and the baseline lane is done; the status."""
        if (other := _claimant(self.directory)) is not None:
            raise RuntimeError(f"driver pid {other} runs {self.directory}")
        claim = {"pid": os.getpid(), "token": start_token(os.getpid()), "at": _utc()}
        write_atomic(self.directory / CLAIM, json.dumps(claim).encode())
        try:
            plan = load_plan(self.directory)
            self._done = {s.name: anyio.Event() for s in plan.stages()}
            for stage in plan.stages():
                # a stage left out (`only`) is never evaluated: nothing waits for it
                if self._stage_done(stage) or not plan.selected(stage):
                    self._done[stage.name].set()
            async with anyio.create_task_group() as group:
                group.start_soon(self._baseline_lane, plan)
                group.start_soon(self._rsi_lane, plan)
            if all(self._stage_done(s) for s in plan.stages()) and all(
                (self._round_dir(r) / "report.json").is_file() for r in range(1, plan.rounds + 1)
            ):
                await self._final(plan)
            return self.status()
        finally:
            (self.directory / CLAIM).unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        return run_status(self.directory, self._phases)

    # -- lanes

    async def _baseline_lane(self, plan: P4Plan) -> None:
        async with anyio.create_task_group() as group:
            for stage in (s for s in plan.stages() if s.arm == "baseline"):
                if self._stage_done(stage) or not plan.selected(stage):
                    continue
                group.start_soon(self._evaluate, plan, stage, None)
                await self._started(plan, stage)

    async def _started(self, plan: P4Plan, stage: Stage) -> None:
        """Wait until `stage` is evaluated, or until it and every other stage being evaluated have
        all their tasks started (`Evaluator.waiting`)."""
        while not self._done[stage.name].is_set():
            evaluating = [
                s
                for s in plan.stages()
                if self._phases.get(s.name) == "running" and not self._done[s.name].is_set()
            ]
            if stage in evaluating and not any(self.evaluator.waiting(s) for s in evaluating):
                return
            await anyio.sleep(self.poll_s)

    async def _rsi_lane(self, plan: P4Plan) -> None:
        wanted = [s.round or 0 for s in plan.stages() if s.arm == "rsi" and plan.selected(s)]
        async with anyio.create_task_group() as finals:
            for r in range(1, max(wanted, default=0) + 1):
                if r > 1 and not await self._passed_gate(plan, r - 1):
                    return
                doc = await self._document(plan, r)
                stage = self._stage(plan, f"rounds/r{r}/eval/{block_name(plan.eval_block(r))}")
                # the round's own block first: its count (and the gate or reflection after it)
                # waits on it, the final does not
                if plan.selected(stage):
                    await self._evaluate(plan, stage, doc)
                for segment in plan.segments:
                    final = self._stage(plan, f"final/{block_name(segment.final)}")
                    if segment.rounds == r and plan.selected(final):
                        finals.start_soon(self._evaluate, plan, final, doc)
                baseline = self._stage(plan, f"baseline/{block_name(plan.eval_block(r))}")
                if not (plan.selected(stage) and plan.selected(baseline)):
                    continue  # one arm only here: nothing to count
                await self._done[baseline.name].wait()
                await self._count(plan, r, stage, baseline)

    async def _passed_gate(self, plan: P4Plan, round: int) -> bool:
        """Whether round `round`'s gate (if any) is approved; waits for it while the baseline
        lane runs, and gives up (False) once that lane is done."""
        gate = gate_name(round)
        if gate not in plan.gates:
            return True
        baseline = [self._done[s.name] for s in plan.stages() if s.arm == "baseline"]
        while gate not in approved(self.directory):
            self._phases[gate] = "gated"
            self.status()
            if all(event.is_set() for event in baseline):
                log.warning(
                    "round %s waits for its approval: run_p4.py resume --approve %s", round, gate
                )
                return False
            await anyio.sleep(self.poll_s)
        return True

    # -- steps

    async def _evaluate(self, plan: P4Plan, stage: Stage, doc: DocSource | None) -> None:
        if self._stage_done(stage):
            return
        self._phases[stage.name] = "running"
        self.status()
        await self.evaluator.evaluate(stage, doc)
        episodes = self.evaluator.episodes(stage)
        self._check_stage(plan, stage, episodes)
        record = {
            "stage": stage.model_dump(mode="json"),
            "doc": None if doc is None else doc.ref.model_dump(mode="json"),
            "episodes": len(episodes),
            "at": _utc(),
        }
        write_atomic(
            self.directory / stage.name / DONE, (json.dumps(record, indent=2) + "\n").encode()
        )
        self._done[stage.name].set()
        self.status()

    def _check_stage(self, plan: P4Plan, stage: Stage, episodes: Sequence[StageEpisode]) -> None:
        """The seed guard of an evaluation: every episode in the stage's block, of a planned task
        and mode, at most the plan's episodes per task and mode."""
        name = block_name(stage.block)
        seeds = [e.seed for e in episodes]
        if stage.final:
            if name not in plan.split.final:
                raise ValueError(f"{stage.name}: {name} is not a final block of the chain")
            block = plan.split.blocks[name]
            if outside := sorted({s for s in seeds if not block.holds(s)}):
                raise ValueError(f"{stage.name}: seeds {outside} lie outside {name}")
        elif stage.arm == "rsi":
            read = [block_name(b) for b in plan.read_blocks(stage.round or 0)]
            check_reflection(plan.split, name, [], seeds)
            if name in read:
                raise ValueError(f"{stage.name}: round {stage.round} reflected on {name}")
        else:
            block = plan.split.blocks[name]
            if outside := sorted({s for s in seeds if not block.holds(s)}):
                raise ValueError(f"{stage.name}: seeds {outside} lie outside {name}")
        counts: dict[tuple[str, str], int] = {}
        for e in episodes:
            if e.task not in plan.tasks:
                raise ValueError(f"{stage.name}: {e.task} is not a task of the plan")
            if e.mode not in stage.modes:
                raise ValueError(f"{stage.name}: an episode in mode {e.mode}")
            counts[(e.task, e.mode)] = counts.get((e.task, e.mode), 0) + 1
        if over := sorted(k for k, n in counts.items() if n > plan.episodes):
            raise ValueError(f"{stage.name}: more than {plan.episodes} episodes of {over}")

    async def _document(self, plan: P4Plan, round: int) -> DocSource:
        """Round `round`'s frozen document: imported (round 1) or reflected now on both arms of
        the previous round's block, from its counted document."""
        here = self._round_dir(round)
        store = FileRsiStore(here / STORE)
        name = round_name(round)
        if (frozen := await store.frozen(name)) is not None:
            return DocSource(ref=frozen, store=here / STORE)
        if round == 1:
            raise ValueError(f"{here}: round 1's document is imported (run_p4.py new --import-g1)")
        previous = round - 1
        start = RsiDoc.model_validate_json(
            (self._round_dir(previous) / f"{round_name(previous)}.counted.json").read_bytes()
        )
        reads = plan.read_blocks(round)
        stages = [
            s
            for s in plan.stages()
            if not s.final and s.block in reads and s.arm in ("baseline", "rsi")
        ]
        episodes: list[tuple[StageEpisode, EpisodeKey]] = []
        for stage in stages:
            for e in self.evaluator.episodes(stage):
                if e.mode != plan.count_mode or e.directory is None:
                    continue
                key = EpisodeKey(
                    block=block_name(stage.block), task=e.task, episode=e.index, arm=stage.arm
                )
                episodes.append((e, key))
        self._phases[gate_name(round)] = "reflecting"
        self.status()
        here.mkdir(parents=True, exist_ok=True)
        evaluation = block_name(plan.eval_block(round))
        reflected = await self.reflector(
            RoundInput(
                round=round,
                directory=here,
                split=plan.split,
                evaluation=evaluation,
                episodes=episodes,
                sources=[self.directory / s.name for s in stages],
                start=start,
            )
        )
        if (frozen := await store.frozen(name)) is None or frozen != reflected.ref:
            raise RuntimeError(f"round {round}'s reflection froze no {name} in {here / STORE}")
        return DocSource(ref=frozen, store=here / STORE)

    async def _count(self, plan: P4Plan, round: int, stage: Stage, baseline: Stage) -> None:
        here = self._round_dir(round)
        if (here / "report.json").is_file():
            return
        doc = await FileRsiStore(here / STORE).load_round(round)
        rsi = self._with_citations(stage, self.evaluator.episodes(stage))
        base = self.evaluator.episodes(baseline)
        mode = plan.count_mode
        pairs = _paired(base, rsi, mode)
        counted, counts = update_counters(doc, pairs, self.recipe.rules, round)
        report = round_report(
            round=round,
            block=block_name(stage.block),
            mode=mode,
            baseline=[e for e in base if e.mode == mode],
            rsi=[e for e in rsi if e.mode == mode],
            doc=doc,
            counted=counted,
            counts=counts,
        )
        write_atomic(
            here / f"{round_name(round)}.counted.json",
            (counted.model_dump_json(indent=2) + "\n").encode(),
        )
        write_atomic(here / "counts.json", (counts.model_dump_json(indent=2) + "\n").encode())
        write_atomic(here / "report.md", render(report).encode())
        write_atomic(here / "report.json", (json.dumps(report, indent=2) + "\n").encode())
        self.status()

    def _with_citations(self, stage: Stage, episodes: Sequence[StageEpisode]) -> list[StageEpisode]:
        """The RSI arm's episodes with the entry ids each Actor cited (`digest_episode`)."""
        found = []
        for e in episodes:
            if e.directory is None:
                found.append(e)
                continue
            read = read_eval_episode(e.directory, e.seed)
            if read is None:
                found.append(e)
                continue
            key = EpisodeKey(
                block=block_name(stage.block), task=e.task, episode=e.index, arm=stage.arm
            )
            cited = digest_episode(read, key, stage.arm, self.recipe.reflect).cited
            found.append(e.model_copy(update={"cited": tuple(cited)}))
        return found

    async def _final(self, plan: P4Plan) -> None:
        """The run's report (`report.md`, `report.json`): every round's, and each final block's
        in each mode, and the gap ledger (`gaps.md`: each round's)."""
        reports = [
            json.loads((self._round_dir(r) / "report.json").read_bytes())
            for r in range(1, plan.rounds + 1)
        ]
        for segment in plan.segments:
            rsi_stage = self._stage(plan, f"final/{block_name(segment.final)}")
            base_stage = self._stage(plan, f"baseline/{block_name(segment.final)}")
            store = FileRsiStore(self._round_dir(segment.rounds) / STORE)
            doc = await store.load_round(segment.rounds)  # what the final block's Actor read
            rsi = self.evaluator.episodes(rsi_stage)
            base = self.evaluator.episodes(base_stage)
            for mode in plan.final_modes:
                report = round_report(
                    round=segment.rounds,
                    block=block_name(segment.final),
                    mode=mode,
                    baseline=[e for e in base if e.mode == mode],
                    rsi=[e for e in rsi if e.mode == mode],
                    doc=doc,
                )
                where = self.directory / rsi_stage.name / f"report-{mode}"
                write_atomic(where.with_suffix(".md"), render(report).encode())
                write_atomic(
                    where.with_suffix(".json"), (json.dumps(report, indent=2) + "\n").encode()
                )
                reports.append(report)
        write_atomic(self.directory / "report.md", final_report(reports).encode())
        write_atomic(
            self.directory / "report.json", (json.dumps(reports, indent=2) + "\n").encode()
        )
        gaps = []
        for r in range(1, plan.rounds + 1):
            path = self._round_dir(r) / "gaps.md"
            if path.is_file():
                gaps += [f"## Round {r}", "", path.read_text(encoding="utf-8").strip(), ""]
        write_atomic(self.directory / "gaps.md", ("\n".join(gaps) + "\n").encode())

    # -- helpers

    def _round_dir(self, round: int) -> Path:
        return self.directory / "rounds" / f"r{round}"

    def _stage(self, plan: P4Plan, name: str) -> Stage:
        return next(s for s in plan.stages() if s.name == name)

    def _stage_done(self, stage: Stage) -> bool:
        return (self.directory / stage.name / DONE).is_file()


def _paired(
    baseline: Sequence[StageEpisode], rsi: Sequence[StageEpisode], mode: Mode
) -> list[PairedOutcome]:
    """The (task, seed) pairs with an outcome in both arms in `mode`, with the RSI arm's
    citations."""
    base = {(e.task, e.seed): e for e in baseline if e.mode == mode and e.success is not None}
    found = []
    for e in rsi:
        if e.mode != mode or e.success is None or (e.task, e.seed) not in base:
            continue
        found.append(
            PairedOutcome(
                task=e.task,
                seed=e.seed,
                rsi_success=e.success,
                baseline_success=bool(base[(e.task, e.seed)].success),
                cited=e.cited,
            )
        )
    return found


def make_plan(
    *,
    name: str,
    code: str,
    base: str,
    tasks: dict[TaskName, ArtifactRef],
    excluded: dict[TaskName, str],
    rounds: int,
    block: Callable[[int], SeedRange],
    episodes: int,
    shards: int,
    method: str,
    recipe: str,
    models: dict[str, Any],
    eval_modes: Sequence[Mode],
    final_modes: Sequence[Mode],
    count_mode: Mode,
    gates: Sequence[str] = (),
    imported: Imported | None = None,
    taken: dict[str, SeedRange] | None = None,
    only: Sequence[str] = (),
    first_read: int = 1,
) -> P4Plan:
    """A new plan of `rounds` rounds, the first reflecting on blocks 1 to `first_read`
    (`new_split`). `taken`: the seed blocks the base run's learning used (its demonstrations,
    practice, internal evaluation), which no block of the chain may overlap."""
    split = new_split(rounds, block, first_read)
    if taken:
        SeedSplit(blocks={**taken, **split.blocks}, reflects=split.reflects, final=split.final)
    segment = Segment(rounds=rounds, final=block_number(split.final[-1]), at=_utc())
    return P4Plan(
        name=name,
        code=code,
        base=base,
        tasks=tasks,
        excluded=excluded,
        split=split,
        segments=(segment,),
        episodes=episodes,
        shards=shards,
        eval_modes=tuple(eval_modes),
        final_modes=tuple(final_modes),
        count_mode=count_mode,
        method=method,
        recipe=recipe,
        models=models,
        gates=tuple(gates),
        imported=imported,
        only=tuple(only),
    )
