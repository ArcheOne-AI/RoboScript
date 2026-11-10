"""The best-Memory fallback rules. S1's (`imitate.fallback_ranking`, from attempt files alone):
only versions an attempt with an official success read qualify, by unhinted successes, then
successes in all, then the later version; no success at all leaves none (S1 stops). Set-aside
attempts and infrastructure ends never count. S2's (`ledger.best_version`): the Memory version
of the best first-try record over the counted seeds whose try 1 read it, for an S2 that stops at
its seed cap."""

import json
from pathlib import Path
from typing import Any

from roboscript.learning.ledger import wilson
from roboscript.memory import VersionMeta
from roboscript_methods.wicl.imitate import FallbackRule, fallback_ranking
from roboscript_methods.wicl.ledger import LedgerRow, best_version, version_records


def _version(n: int) -> VersionMeta:
    return VersionMeta.model_construct(version=n, hash=f"h{n}")


def _keypoints(statuses: list[tuple[str, str]]) -> dict[str, Any]:
    keypoints = [
        {"index": i + 1, "status": status, "kind": kind}
        for i, (kind, status) in enumerate(statuses)
    ]
    return {"verdict": {"passed": False}, "progress": {"left": {"keypoints": keypoints}}}


def _attempt(
    folder: Path,
    name: str,
    version: int,
    *,
    hinted: bool = False,
    passed: bool = False,
    success: bool = False,
    end_class: str = "policy",
    keypoints: dict[str, Any] | None = None,
) -> None:
    attempt = folder / name
    (attempt / "referee").mkdir(parents=True)
    config = {"memory": {"hash": f"h{version}"}, "protocol": {"hinted": hinted}}
    (attempt / "config.json").write_text(json.dumps(config), encoding="utf-8")
    summary = {"success": success, "end_class": end_class}
    (attempt / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    verdict = keypoints or {"verdict": {"passed": passed}, "progress": {}}
    verdict["verdict"]["passed"] = passed
    (attempt / "referee" / "keypoints.json").write_text(json.dumps(verdict), encoding="utf-8")


def _order(folder: Path, n: int, rule: FallbackRule = "success") -> list[int]:
    versions = [_version(v) for v in range(n)]
    return [meta.version for _, meta, _ in fallback_ranking(folder, versions, rule)]


def test_set_aside_and_infrastructure_attempts_do_not_count(tmp_path: Path) -> None:
    _attempt(tmp_path, "attempt1-aborted1", 0, passed=True, success=True)
    _attempt(tmp_path, "attempt2", 0, passed=True, success=True, end_class="infra")
    _attempt(tmp_path, "attempt3", 1, success=True)
    assert _order(tmp_path, 2) == [1]


def test_success_rule_unhinted_successes_beat_hinted_ones(tmp_path: Path) -> None:
    """v0: two hinted successes; v1: one unhinted success; v2: referee passes, no success (left
    out); v3: nothing (left out)."""
    _attempt(tmp_path, "attempt1", 0, hinted=True, success=True)
    _attempt(tmp_path, "attempt2", 0, hinted=True, success=True)
    _attempt(tmp_path, "attempt3", 1, success=True)
    _attempt(tmp_path, "attempt4", 2, passed=True)
    _attempt(tmp_path, "attempt5", 3)
    assert _order(tmp_path, 4, "success") == [1, 0]
    score, meta, evidence = fallback_ranking(tmp_path, [_version(1)], "success")[0]
    assert (score, meta.version) == ((1, 1, 1), 1)
    assert evidence == {"attempts": [3], "rule": "success"}


def test_success_rule_ties_go_to_more_successes_then_the_later_version(tmp_path: Path) -> None:
    _attempt(tmp_path, "attempt1", 0, success=True)
    _attempt(tmp_path, "attempt2", 0, hinted=True, success=True)
    _attempt(tmp_path, "attempt3", 1, success=True)  # v1: one unhinted, one in all
    _attempt(tmp_path, "attempt4", 2, success=True)  # v2: the same as v1, later
    _attempt(tmp_path, "attempt5", 2)
    assert _order(tmp_path, 3, "success") == [0, 2, 1]


def test_success_rule_with_only_hinted_successes(tmp_path: Path) -> None:
    _attempt(tmp_path, "attempt1", 0, hinted=True, success=True)
    _attempt(tmp_path, "attempt2", 1, hinted=True, success=True)
    _attempt(tmp_path, "attempt3", 1, hinted=True, success=True)
    _attempt(tmp_path, "attempt4", 2, hinted=True, success=True)
    _attempt(tmp_path, "attempt5", 3, passed=True)
    assert _order(tmp_path, 4, "success") == [1, 2, 0]


def test_success_rule_without_any_success_has_no_fallback(tmp_path: Path) -> None:
    """Referee passes and keypoint progress do not qualify; a success that ended on
    infrastructure or was set aside does not either."""
    further = _keypoints([("grip", "passed"), ("path", "passed")])
    _attempt(tmp_path, "attempt1", 0, passed=True, keypoints=further)
    _attempt(tmp_path, "attempt2", 1, hinted=True, passed=True)
    _attempt(tmp_path, "attempt3-aborted1", 1, success=True)
    _attempt(tmp_path, "attempt3", 1, success=True, end_class="infra")
    assert fallback_ranking(tmp_path, [_version(0), _version(1)], "success") == []


def row(seed: int, version: int, first: bool | None, *, counted: bool = True) -> LedgerRow:
    return LedgerRow(
        seed=seed,
        k=seed if counted else None,
        solvable=counted,
        instruction=None,
        tries=[],
        first_try_pass=first,
        streak_after=0,
        outcome="pass" if first else "skip" if counted else "unsolvable",
        memory_before=version,
        memory_after=version,
    )


def rows(*records: tuple[int, int, int]) -> list[LedgerRow]:
    """`(version, successes, seeds)` as counted rows, in that order."""
    found = []
    for version, k, n in records:
        found += [row(len(found) + 1, version, i < k) for i in range(n)]
    return found


def test_the_wilson_lower_bound_ranks_not_the_rate() -> None:
    """1 of 1 has the higher rate, 5 of 10 the higher lower bound: the earlier version wins."""
    found = rows((1, 5, 10), (2, 1, 1))
    assert version_records(found) == {1: (5, 10), 2: (1, 1)}
    assert wilson(5, 10)[0] > wilson(1, 1)[0]
    best = best_version(found)
    assert best is not None
    assert (best["version"], best["successes"], best["n"]) == (1, 5, 10)
    assert best["lower"] == round(wilson(5, 10)[0], 6)
