"""`roboscript.orchestrate.graph`: nodes run in their order once what they follow ended with an
accepted outcome; an unaccepted outcome rejects down the graph."""

from collections.abc import Sequence
from typing import Any

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import (
    ANY,
    REJECTED,
    Edge,
    JobNode,
    JobSpec,
    Outcomes,
)
from roboscript.orchestrate.graph import (
    ComposedGraph,
    JobGraph,
    Unit,
    run_graph,
)
from roboscript.orchestrate.jobs import RunKey


def job(phase: str, stage: int = 0) -> JobSpec:
    return JobSpec(task="t", unit="u", phase=phase, stage=stage)


class Toy:
    """A graph whose jobs end with the status `ends` gives them; `not_due` nodes are skipped,
    `raises` nodes raise as they run; `records` is each node's status as its records show it."""

    stages: Sequence[str] = ("a", "b")

    def __init__(self, nodes: list[JobNode], **given: Any) -> None:
        self.nodes = nodes
        self.ends: dict[str, str] = given.get("ends", {})
        self.not_due: set[str] = given.get("not_due", set())
        self.raises: set[str] = given.get("raises", set())
        self.records: dict[str, str] = given.get("records", {})
        self.failures: list[str] = []
        self.ran: list[str] = []

    async def due(self, node: JobNode, outcomes: Outcomes) -> bool:
        if node.id in self.raises:
            raise RuntimeError(f"{node.id} failed")
        return node.id not in self.not_due

    async def act(self, node: JobNode, outcomes: Outcomes) -> str:
        self.ran.append(node.id)
        return "acted"

    def status(self, node: JobNode) -> str:
        return self.records.get(node.id, self.ends.get(node.id, "done"))

    def failed(self, node: JobNode, error: Exception) -> None:
        self.failures.append(node.id)


async def run(graph: JobGraph) -> tuple[dict[str, str], list[str]]:
    order: list[str] = []

    async def run_job(spec: JobSpec) -> None:
        order.append(spec.phase)

    return await run_graph(graph, run_job), order


async def test_nodes_run_in_order_after_what_they_follow() -> None:
    a = JobNode(id="a", job=job("a"))
    b = JobNode(id="b", job=job("b", 1), after=[Edge(node="a", accept=frozenset({"done"}))])
    c = JobNode(id="c", job=job("c", 1), after=[Edge(node="a", accept=frozenset({"done"}))])
    d = JobNode(id="d", action="settle", after=[Edge(node="b", accept=frozenset({ANY}))])
    graph = Toy([a, b, c, d])
    outcomes, order = await run(graph)
    assert order == ["a", "b", "c"] and graph.ran == ["d"]
    assert outcomes == {"a": "done", "b": "done", "c": "done", "d": "acted"}


async def test_an_unaccepted_outcome_rejects_down_the_graph() -> None:
    a = JobNode(id="a", job=job("a"))
    b = JobNode(id="b", job=job("b"), after=[Edge(node="a", accept=frozenset({"done"}))])
    c = JobNode(id="c", job=job("c"), after=[Edge(node="b", accept=frozenset({ANY}))])
    blocked = JobNode(
        id="block", action="block", after=[Edge(node="a", accept=frozenset({"not_passed"}))]
    )
    graph = Toy([a, b, c, blocked], ends={"a": "not_passed"})
    outcomes, order = await run(graph)
    assert order == ["a"] and graph.ran == ["block"]
    assert (outcomes["b"], outcomes["c"]) == (REJECTED, REJECTED)


class Chain:
    """A part whose units run `learn` (due while the unit began) then `check` (due when
    `checks` names its task, whether or not the unit began), and a lane `side` beside a unit
    that follows its `learn`."""

    def __init__(self, checks: set[str]) -> None:
        self.checks = checks

    def add(self, graph: ComposedGraph, task: str) -> None:
        key = RunKey(TaskName(task), "u")
        learn = graph.add(self, key, "learn", job=job("a"))
        after = Edge(node=learn, accept=frozenset({"queued"}), skip="pass")
        graph.add(self, key, "check", [after], job=job("b", 1))
        graph.add(self, RunKey(TaskName(task), "u", "lane"), "side", [after], job=job("b", 1))

    async def due(self, unit: Unit) -> bool:
        if unit.step == "learn":
            return unit.began
        return unit.key.task in self.checks

    async def act(self, unit: Unit) -> str:
        raise AssertionError("no action")
