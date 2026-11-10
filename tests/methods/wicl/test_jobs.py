"""wicl's jobs (`Wicl.jobs`) compose a graph that runs, with any evaluation after each
unit: S1, its block, each unit's S2, in that order; S1s only, or evaluations only."""

from pathlib import Path

from roboscript.contracts.jobs import Edge
from roboscript.learning.method import load_method
from roboscript.orchestrate.graph import ComposedGraph, Records, Unit
from roboscript.orchestrate.jobs import RunKey
from roboscript.orchestrate.worker import check_kinds
from roboscript.testing.jobs import check_jobs
from roboscript_methods.wicl.jobs import PHASES
from roboscript_methods.wicl.method import Wicl


class Evaluate:
    """An evaluation of one job per unit, after the unit learned."""

    phases = ("evaluate",)

    def add(self, graph: ComposedGraph, key: RunKey, learned: Edge, begun: Edge) -> None:
        spec = {"task": key.task, "unit": key.variant, "phase": "evaluate", "stage": 2}
        graph.add(self, key, "evaluate", [learned], job=spec)

    async def due(self, unit: Unit) -> bool:
        return True

    async def act(self, unit: Unit) -> str:
        return "done"


def test_wicl_jobs_compose_a_graph_that_runs(tmp_path: Path) -> None:
    jobs = Wicl().jobs(["default", "v2"], Evaluate())
    assert jobs.stages == ("imitate", "practice", "evaluate")
    assert check_jobs(jobs, ["red", "blue"], tmp_path) == []
    graph = ComposedGraph(jobs.stages, _records(tmp_path))
    jobs.add(graph, "red")
    assert [node.id for node in graph.nodes] == [
        "red/S1:imitate",
        "red/S1:block",
        "red/default:practice",
        "red/default:evaluate",
        "red/v2:practice",
        "red/v2:evaluate",
    ]
    only = ComposedGraph(jobs.stages, _records(tmp_path))
    Wicl().jobs(["default"], Evaluate(), s1_only=True).add(only, "red")
    assert [node.id for node in only.nodes] == ["red/S1:imitate", "red/S1:block"]
    evaluated = ComposedGraph(jobs.stages, _records(tmp_path))
    Wicl().jobs(["default"], Evaluate(), evaluate_only=True).add(evaluated, "red")
    assert [node.id for node in evaluated.nodes][2:] == ["red/default:evaluate"]


def _records(root: Path) -> Records:
    return Records(root, lambda key: root / key.task / key.variant)


def test_wicl_gives_a_job_body_for_each_of_its_phases() -> None:
    """A worker finds wicl's job bodies through its registration (`Method.kinds`)."""
    kinds = load_method("wicl").kinds  # type: ignore[attr-defined]
    assert check_kinds(kinds, PHASES) == []
    assert {phase: kind.stage for phase, kind in kinds.items()} == {
        "imitate": "s1",
        "practice": "s2",
    }
