"""A P4 round's report (`rsi.report`): pairs by task and seed, McNemar's exact test, the bootstrap
interval, and harm made easy to see."""

import pytest

from roboscript.contracts.ids import TaskName
from roboscript_methods.rsi.doc import RsiDoc
from roboscript_methods.rsi.report import (
    ArmEpisode,
    bootstrap_interval,
    mcnemar_exact,
    pair_up,
    render,
    round_report,
)


def arm(task: str, outcomes: list[bool | None], voided: bool = False) -> list[ArmEpisode]:
    return [
        ArmEpisode(
            task=TaskName(task),
            index=i,
            seed=200000 + i,
            success=s,
            voided=voided and s is None,
            infra=s is None and not voided,
            tokens=10,
        )
        for i, s in enumerate(outcomes)
    ]


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)
    assert mcnemar_exact(3, 3) == 1.0


def test_pairs_by_seed_and_the_unpaired() -> None:
    pairs, unpaired = pair_up(arm("a", [True, False, None]), arm("a", [True, True, True]))
    assert [(p.seed, p.baseline, p.rsi) for p in pairs] == [
        (200000, True, True),
        (200001, False, True),
    ]
    assert unpaired == {
        "baseline": ["a@200002: lost to the infrastructure"],
        "rsi": ["a@200002: the other arm has no outcome"],
    }
    with pytest.raises(ValueError, match="twice"):
        pair_up(arm("a", [True]) * 2, [])


def test_the_report_lists_harmed_tasks_first() -> None:
    baseline = arm("hurt", [True, True, True, False]) + arm("fine", [False, False])
    rsi = arm("hurt", [False, False, True, False]) + arm("fine", [True, False])
    report = round_report(
        round=1, block="block2", mode="clean", baseline=baseline, rsi=rsi, doc=RsiDoc()
    )
    assert (report["pairs"], report["rsi_lost"], report["rsi_won"], report["harmed"]) == (
        6,
        2,
        1,
        ["hurt"],
    )
    assert report["tasks"]["hurt"] == {
        "pairs": 4,
        "baseline": 3,
        "rsi": 1,
        "rsi_lost": 2,
        "rsi_won": 0,
    }
    low, high = report["diff_ci95"]
    assert low <= -1 / 6 <= high
    shown = render(report)
    assert shown.index("| hurt | 4 | 3 | 1 | 2 | 0 |") < shown.index("## Per task")
    assert render(report) == shown  # a fixed resampling seed: the same report again


def test_no_pairs_no_interval() -> None:
    assert bootstrap_interval([]) is None
