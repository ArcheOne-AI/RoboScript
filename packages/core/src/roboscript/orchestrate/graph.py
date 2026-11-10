"""Job graphs (docs/how-to/add-a-method.md, `Method.jobs()`): what a run schedules, as nodes and
edges a method (and the benchmark that evaluates it) declare, and `run_graph`, which runs them.

A node is a job (a worker process on a GPU's simulator places: `JobSpec`) or an action (a short
step of the method's own on the run's records, no GPU). Its edges name the nodes it follows and
the outcomes of each it accepts: a node runs once every node it follows has ended with an
accepted outcome; an edge that does not accept its node's outcome rejects the node, which then
ends `rejected` (and so on down the graph). An outcome is the status the node's records show
once it ran (`JobGraph.status`; an action's own), `skipped` when the graph found it not due as it
became ready (`JobGraph.due`, asked then and not before: the records may have changed since the
graph was made, a job of the same task having run), `error` when it raised (the graph's `failed`
records it), or `rejected`. An edge says what a skipped node means to the node after it
(`Edge.skip`).

`run_graph` starts the nodes in their order, each waiting for those it follows, so that nodes
that become ready together ask for room in that order (`Slots.turn`); the graph's own order is
the only order there is. `walk` says what a run would do with the records as they are now,
acting on nothing (a launch's dry run and its list of jobs: `ComposedGraph.taken_up`)."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import anyio

from roboscript.contracts.jobs import (
    ANY,
    ERROR,
    REJECTED,
    SKIPPED,
    Edge,
    JobNode,
    JobSpec,
    Outcomes,
    Part,
)
from roboscript.orchestrate.jobs import RunKey, RunState, pending

log = logging.getLogger(__name__)

RUNS: Final = "runs"
"""`walk`'s outcome of a node that would run (a job started, an action acted out): what it ends
with is not known before it ran."""


class JobGraph(Protocol):
    """A run's jobs: `nodes` in order, and `stages`, the phases in the order `by_stage` places
    their jobs. The graph decides, as a node becomes ready, whether it is due (`due`, with the
    outcomes so far), acts out an action node (`act`, its outcome), reads a job's outcome from
    its records once it ran (`status`), and records a node that raised (`failed`)."""

    nodes: Sequence[JobNode]
    stages: Sequence[str]

    async def due(self, node: JobNode, outcomes: Outcomes) -> bool: ...

    async def act(self, node: JobNode, outcomes: Outcomes) -> str: ...

    def status(self, node: JobNode) -> str: ...

    def failed(self, node: JobNode, error: Exception) -> None: ...


RunJob = Callable[[JobSpec], Awaitable[Any]]
"""Runs a job node's job to its end (the scheduler: room, a worker, respawns)."""


def check_graph(graph: JobGraph) -> list[str]:
    """What is wrong with a graph, one line each (empty: it runs): ids unique, every edge after a
    node before it (so the graph is acyclic and its order a run order), stages in range."""
    lines, seen = [], set()
    for node in graph.nodes:
        if node.id in seen:
            lines.append(f"node {node.id!r} declared twice")
        for edge in node.after:
            if edge.node not in seen:
                lines.append(f"node {node.id!r} follows {edge.node!r}, not a node before it")
        if node.job is not None and node.job.stage >= len(graph.stages):
            lines.append(f"node {node.id!r}: stage {node.job.stage} of {len(graph.stages)}")
        seen.add(node.id)
    return lines


async def run_graph(graph: JobGraph, run_job: RunJob) -> dict[str, str]:
    """Run every node of `graph` (see the module docstring); the outcome of each, by id."""
    if wrong := check_graph(graph):
        raise ValueError(f"the job graph does not run: {'; '.join(wrong)}")
    outcomes: dict[str, str] = {}
    ended = {node.id: anyio.Event() for node in graph.nodes}

    async def one(node: JobNode) -> None:
        try:
            for edge in node.after:
                await ended[edge.node].wait()
            outcomes[node.id] = await _outcome(graph, node, outcomes, run_job)
        finally:
            ended[node.id].set()

    async with anyio.create_task_group() as group:
        for node in graph.nodes:
            group.start_soon(one, node)
    return outcomes


async def _outcome(graph: JobGraph, node: JobNode, outcomes: Outcomes, run_job: RunJob) -> str:
    for edge in node.after:
        if not _accepts(graph, edge, outcomes[edge.node]):
            return REJECTED
    try:
        if not await graph.due(node, outcomes):
            return SKIPPED
        if node.action is not None:
            return await graph.act(node, outcomes)
        assert node.job is not None
        await run_job(node.job)
        return graph.status(node)
    except Exception as error:  # the scheduler's own failure: that node's, the others go on
        graph.failed(node, error)
        return ERROR


async def walk(graph: JobGraph) -> dict[str, str]:
    """What `run_graph` would do with the records as they are now, acting on nothing: per node,
    by id, `RUNS`, `SKIPPED` (not due), `REJECTED`, or `ERROR` (asking whether it is due raised,
    which a run records as the node's failure, `JobGraph.failed`). A node that runs may end with
    any outcome, so an edge after one accepts it when it accepts any outcome at all; a node is
    asked whether it is due with the records as they are now (not as the nodes before it would
    leave them)."""
    if wrong := check_graph(graph):
        raise ValueError(f"the job graph does not run: {'; '.join(wrong)}")
    outcomes: dict[str, str] = {}
    for node in graph.nodes:
        outcomes[node.id] = await _walked(graph, node, outcomes)
    return outcomes


async def _walked(graph: JobGraph, node: JobNode, outcomes: Outcomes) -> str:
    for edge in node.after:
        before = outcomes[edge.node]
        if not (bool(edge.accept) if before == RUNS else _accepts(graph, edge, before)):
            return REJECTED
    try:
        due = await graph.due(node, outcomes)
    except Exception as error:  # a run fails the node (`_outcome`); the walk records nothing
        log.warning("%s: asking whether it is due failed: %s", node.id, one_line(error))
        return ERROR
    return RUNS if due else SKIPPED


def _accepts(graph: JobGraph, edge: Edge, before: str) -> bool:
    if before == REJECTED:  # passes down the graph, whatever an edge accepts
        return False
    if before == SKIPPED:
        if edge.skip == "status":
            before = graph.status(graph_node(graph, edge.node))
        else:
            return edge.skip == "pass"
    return ANY in edge.accept or before in edge.accept


def graph_node(graph: JobGraph, id: str) -> JobNode:
    """The node `id` of `graph`."""
    return next(node for node in graph.nodes if node.id == id)


@dataclass(frozen=True)
class Unit:
    """What a part sees of a node as it becomes ready: its key and step, whether the launch took
    its unit up when the unit's first node of this launch became ready (`began`: its state then
    was `pending`), the outcomes so far, and the run's records (`state`, `directory`)."""

    key: RunKey
    step: str
    began: bool
    outcomes: Outcomes
    records: "Records"

    def state(self, key: RunKey | None = None) -> RunState:
        return self.records.state(key or self.key)

    def directory(self, key: RunKey | None = None) -> Path:
        return self.records.directory(key or self.key)


@dataclass(frozen=True)
class Records:
    """Where each job's records lie (`directory`: the unit's, or its lane's), and the run's
    `retry_failed` (whether a job that failed otherwise than on infrastructure is taken up)."""

    root: Path
    directory: Callable[[RunKey], Path]
    retry_failed: bool = False

    def state(self, key: RunKey) -> RunState:
        return RunState.load(self.directory(key), key)


class ComposedGraph:
    """A job graph its parts add nodes to (`add`), in order, each node `<key>:<step>` owned by the
    part that added it; a job node's outcome is its key's status, and a node the scheduler
    failed fails its key (`broken`)."""

    def __init__(self, stages: Sequence[str], records: Records) -> None:
        self.stages = tuple(stages)
        self.records = records
        self.nodes: list[JobNode] = []
        self._owner: dict[str, tuple[Part, RunKey, str]] = {}
        self._began: dict[RunKey, bool] = {}

    def add(
        self, part: Part, key: RunKey, step: str, after: Sequence[Edge] = (), **what: Any
    ) -> str:
        """Add a node of `part` (`job`: a `JobSpec`, or `action`: a name); its id."""
        node = f"{key.name}:{step}"
        self.nodes.append(JobNode.model_validate({"id": node, "after": list(after), **what}))
        self._owner[node] = (part, key, step)
        return node

    def _unit(self, node: JobNode, outcomes: Outcomes) -> tuple[Part, Unit]:
        part, key, step = self._owner[node.id]
        main = RunKey(key.task, key.variant)
        if main not in self._began:  # the unit's chain begins: as the launch takes it up
            self._began[main] = pending(self.records.state(main), self.records.retry_failed)
        return part, Unit(key, step, self._began[main], outcomes, self.records)

    async def due(self, node: JobNode, outcomes: Outcomes) -> bool:
        part, unit = self._unit(node, outcomes)
        return await part.due(unit)

    async def act(self, node: JobNode, outcomes: Outcomes) -> str:
        part, unit = self._unit(node, outcomes)
        return await part.act(unit)

    def status(self, node: JobNode) -> str:
        return self.records.state(self._owner[node.id][1]).status

    def taken_up(self, walked: Outcomes) -> list[RunKey]:
        """The jobs a launch takes up, by `walk`'s outcomes, in the order of each one's first node:
        a unit the launch takes up (`Unit.began`) with a node that runs (or that the run fails,
        `ERROR`), and a job that does without waiting on another job of this launch (e.g. of a
        unit that had ended, or a lane beside its unit)."""
        keys = list(dict.fromkeys(key for _, key, _ in self._owner.values()))
        taken = set()
        for node in self.nodes:
            key = self._owner[node.id][1]
            if walked[node.id] not in (RUNS, ERROR):  # an error: the run fails the job
                continue
            began = key.lane is None and self._began.get(key, False)
            waits = any(
                walked[edge.node] == RUNS and graph_node(self, edge.node).job is not None
                for edge in node.after
            )
            if began or (node.job is not None and not waits):
                taken.add(key)
        return [key for key in keys if key in taken]

    def failed(self, node: JobNode, error: Exception) -> None:
        """The scheduler failed while it handled the node: its job fails (`error`), the others go
        on."""
        key = self._owner[node.id][1]
        log.error("%s: the scheduler failed it", key.name, exc_info=error)
        try:
            state = self.records.state(key)
            state.failure = "error"
            state.end(self.records.directory(key), "failed", f"scheduler: {one_line(error)}")
        except Exception:  # its records cannot be written either: the log has it
            log.exception("%s: its failure could not be recorded", key.name)


def one_line(error: BaseException) -> str:
    """The error in one line: its type and the first line of its message."""
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0]}" if text else type(error).__name__
