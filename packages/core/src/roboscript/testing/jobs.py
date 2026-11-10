"""Conformance of a method's jobs (`Method.jobs`, `orchestrate.graph.TaskJobs`) and a benchmark's
evaluation (`orchestrate.graph.Evaluation`, registered as an `EVALUATIONS` entry point,
`check_evaluation`): the graph they compose for some tasks runs (`check_graph`: ids unique, every
node after nodes before it, stages in range), and each job node's phase is one of the graph's
stages at its index. Their job bodies (`JobKind`: `Method.kinds`, `EvaluationFactory.kinds`,
`orchestrate.worker.check_kinds`) are one per phase, each with the stage its records show."""

from collections.abc import Sequence
from pathlib import Path

from roboscript.contracts.jobs import (
    EVALUATIONS,
    Edge,
    Evaluation,
    EvaluationFactory,
    JobSpec,
    TaskJobs,
)
from roboscript.kernel import entrypoints
from roboscript.orchestrate.graph import ComposedGraph, Records, check_graph
from roboscript.orchestrate.jobs import RunKey
from roboscript.orchestrate.worker import check_kinds


def check_jobs(jobs: TaskJobs, tasks: Sequence[str], root: Path) -> list[str]:
    """What is wrong with the graph `jobs` compose for `tasks` over records in `root`, one line
    each (empty: it runs)."""
    graph = ComposedGraph(jobs.stages, Records(root, lambda key: _directory(root, key)))
    for task in tasks:
        jobs.add(graph, task)
    lines = check_graph(graph)
    for node in graph.nodes:
        job = node.job
        if job is not None and graph.stages[job.stage] != job.phase:
            lines.append(f"node {node.id!r}: phase {job.phase!r} at stage {job.stage}")
    return lines


def check_evaluation(name: str, tasks: Sequence[str], root: Path) -> list[str]:
    """What is wrong with the evaluation registered as `name` (`EVALUATIONS`), under its
    settings' defaults, after one learning job per task (`check_jobs`), one line each (empty: it
    runs)."""
    factory: EvaluationFactory = entrypoints.load(EVALUATIONS, name)
    evaluation = factory(factory.settings(), gated=lambda directory, task: False)
    if not evaluation.phases:
        return [f"evaluation {name!r} runs no phase"]
    lines = check_kinds(factory.kinds, evaluation.phases)
    return lines + check_jobs(_Learn(evaluation), tasks, root)


class _Learn:
    """One learning job per task, then the evaluation of its unit."""

    def __init__(self, evaluation: Evaluation) -> None:
        self.evaluation = evaluation
        self.stages = ("learn", *evaluation.phases)

    def add(self, graph: ComposedGraph, task: str) -> None:
        key = RunKey(task, "unit")
        job = JobSpec(task=task, unit="unit", phase="learn", stage=0, first=True)
        learned = Edge(node=graph.add(self, key, "learn", job=job), accept=frozenset({"done"}))
        self.evaluation.add(graph, key, learned, learned)

    async def due(self, unit: object) -> bool:
        return True

    async def act(self, unit: object) -> str:
        return "done"


def _directory(root: Path, key: RunKey) -> Path:
    unit = root / key.task / key.variant
    return unit if key.lane is None else unit / key.lane
