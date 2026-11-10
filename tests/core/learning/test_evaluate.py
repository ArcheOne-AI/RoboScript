"""`evaluate`: a Memory version on a seed list, one try each, no writes, Wilson intervals."""

import json
from typing import Any

import pytest

from roboscript.learning import Evaluate, EvaluateResult, wilson
from roboscript.memory import VersionSource
from roboscript.testing.protocols import LIFT_RED, acting, lift_memory


def test_wilson() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(8, 10)
    assert (round(low, 3), round(high, 3)) == (0.49, 0.943)
    assert wilson(3, 3) == (pytest.approx(0.4385, abs=1e-4), 1.0)
    assert wilson(0, 5)[0] == 0.0


async def test_a_version_on_seeds(setup: Any) -> None:
    await setup.start(lift_memory(0.06))
    await setup.store.commit(
        lift_memory(0.08), status="edited", source=VersionSource(kind="edited"), parent=0
    )
    actor = acting(lambda e: "fail" if e.extra["seed"] == 20001 else "pass")
    async with setup.runner({"actor": actor}) as runner:
        with pytest.raises(ValueError, match="no frozen Memory"):
            await runner.run(Evaluate(), LIFT_RED)
        await setup.store.freeze(LIFT_RED, 1, reason="manual")
        seeds = [20000, 10001, 20001, 20002]  # 10001 is unsolvable
        result = await runner.run(Evaluate(), LIFT_RED, {"seeds": seeds})
        other = await runner.run(
            Evaluate(), LIFT_RED, {"seeds": [20000], "version": 0, "name": "again"}
        )
        opened = len(setup.env.opened)
        with pytest.raises(ValueError, match="again exists"):
            await runner.run(Evaluate(), LIFT_RED, {"seeds": [20002], "name": "again"})
        assert len(setup.env.opened) == opened  # refused before any episode
    assert isinstance(result, EvaluateResult) and isinstance(other, EvaluateResult)
    assert (result.version, result.n, result.successes, result.unsolvable) == (1, 3, 2, [10001])
    assert result.rate == pytest.approx(2 / 3) and result.wilson95 == wilson(2, 3)
    assert [t.directory for t in result.tries] == [
        f"eval/v001/{seed}/try1" for seed in (20000, 20001, 20002)
    ]
    saved = json.loads(
        (setup.run_dir / LIFT_RED / "eval" / "v001" / "results.json").read_text(encoding="utf-8")
    )
    assert saved["successes"] == 2
    assert (other.version, other.n) == (0, 1)
    assert [meta.version for meta in await setup.store.versions(LIFT_RED)] == [0, 1]
