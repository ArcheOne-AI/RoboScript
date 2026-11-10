"""wicl's declaration: it loads, its eval phase is D only, and the information boundary's
negative tests hold for it (docs/architecture.md). Its protocols and its part
of a pipeline's report pass the core's conformance suites (`roboscript.testing.protocols`,
`roboscript.testing.report`)."""

from pathlib import Path
from typing import Any

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import OpenEpisode
from roboscript.hub import Hub
from roboscript.learning import Evaluate
from roboscript.learning.method import load_method
from roboscript.orchestrate.jobs import RunKey
from roboscript.testing import precheck
from roboscript.testing.methods import MethodConformance
from roboscript.testing.protocols import (
    LIFT_RED,
    ProtocolConformance,
    ProtocolTarget,
    acting,
    editing,
    lift_memory,
    lifting_actor,
)
from roboscript.testing.report import check_report_part
from roboscript_methods.wicl.method import Wicl
from roboscript_methods.wicl.ops import S1
from roboscript_methods.wicl.practice import Practice
from roboscript_methods.wicl.report import LearnReport
from roboscript_methods.wicl.toolsets.demo import check_fitted


class TestWiclConformance(MethodConformance):
    @pytest.fixture
    def method_target(self) -> type:
        return Wicl


async def test_the_conformance_suite_runs_without_pytest() -> None:
    results = await precheck(MethodConformance(), Wicl)
    assert results and all(why is None for why in results.values()), results


def test_wicl_is_registered() -> None:
    assert load_method("wicl") is Wicl
    assert load_method("roboscript_methods.wicl.method:Wicl") is Wicl


def test_phases_kinds_and_tiers() -> None:
    kinds = {phase.name: phase.kind for phase in Wicl.phases}
    assert kinds == {"s1": "learn", "s2": "learn", "p3": "eval", "dev": "learn"}
    by_name = {phase.name: phase for phase in Wicl.phases}
    assert {by_name["s1"].tier(role) for role in by_name["s1"].visibility} == {"P"}
    assert {by_name["s2"].tier(role) for role in by_name["s2"].visibility} == {"D"}
    assert list(by_name["p3"].visibility) == ["actor"]
    # S2 is held to tier D as P3 is (`PhaseSpec.d_only`): what it learns runs in P3
    assert {phase.name for phase in Wicl.phases if phase.holds_d} == {"s2", "p3"}
    assert by_name["s2"].d_only


async def test_s2_refuses_a_privileged_episode(tmp_path: Path) -> None:
    async with Hub([], blobs=LocalBlobStore(tmp_path), phases=Wicl.phases) as hub:
        with pytest.raises(ProtocolError, match="tier D only"):
            await hub.open_episode(OpenEpisode(stage="s2", role="reviser", tier="P"))
        session = await hub.open_episode(OpenEpisode(stage="s1", role="reviser", tier="P"))
        assert [tool.name for tool in session.tools] == ["finish"]


def test_the_runtime_checks_spec_edits_against_the_fitted_specs(tmp_path: Path) -> None:
    """wicl's Memory edit check (`toolsets.demo.check_fitted`) comes with its runtime
    (`MethodRuntime.edit_checks`), which the runner's memory toolset runs."""
    assert Wicl.runtime(tmp_path).edit_checks == (check_fitted,)


class TestPracticeConforms(ProtocolConformance):
    @pytest.fixture
    def protocol_target(self, setup: Any) -> ProtocolTarget:
        brains = {"actor": lifting_actor, "reviser": editing(lambda _: [])}
        return ProtocolTarget(
            runner=lambda: setup.runner(brains),
            run=lambda runner: runner.run(Practice(), LIFT_RED, {"start": 0, "max_seeds": 1}),
            run_dir=setup.run_dir,
            start=lift_memory(0.02),  # try 1 fails: the Reviser runs, tries 2 and 3 too
        )


class TestEvaluateConforms(ProtocolConformance):
    @pytest.fixture
    def protocol_target(self, setup: Any) -> ProtocolTarget:
        return ProtocolTarget(
            runner=lambda: setup.runner({"actor": lifting_actor}),
            run=lambda runner: runner.run(Evaluate(), LIFT_RED, {"version": 0, "count": 2}),
            run_dir=setup.run_dir,
            writes_memory=False,
        )


async def test_the_suite_runs_without_pytest(setup: Any) -> None:
    target = ProtocolTarget(
        runner=lambda: setup.runner({"actor": acting(lambda _: "fail")}),
        run=lambda runner: runner.run(Evaluate(), LIFT_RED, {"version": 0, "count": 1}),
        run_dir=setup.run_dir,
        writes_memory=False,
    )
    assert await precheck(ProtocolConformance(), target) == {
        "test_every_episode_leaves_its_record": None
    }


def test_wicl_reports_its_learning(tmp_path: Path) -> None:
    part = LearnReport(lambda run_dir, task: None)
    keys = [RunKey("red", S1), RunKey("red", "default"), RunKey("blue", S1)]
    assert check_report_part(part, keys, tmp_path) == []
    assert list(part.readings) == ["s1", "s2"]
