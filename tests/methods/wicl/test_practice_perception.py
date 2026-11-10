"""S2 `practice` with the perception toolset mounted (a scripted model): a finish the criteria
passed but the latch did not fails the try."""

import json
import re
from pathlib import Path
from typing import Any

from roboscript.agent import ScriptedBrain, call
from roboscript.memory import DoneSpecParam
from roboscript.perception.criteria import DoneSpec
from roboscript.records import read_events
from roboscript.testing.perception import ScriptedModel, asked_for
from roboscript.testing.protocols import (
    LIFT_RED,
    NO_OBSERVE,
    RATIONALE,
    acting,
    color_of,
    editing,
    grounding,
    lift_memory,
)
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import LocalProvider
from roboscript.toolsets.perception import Perception
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.worldstate import WorldState
from roboscript_methods.wicl.ledger import Ledger, LedgerRow
from roboscript_methods.wicl.practice import Practice, PracticeConfig, PracticeResult
from roboscript_methods.wicl.toolsets.done import Done


def by_try(plan: dict[tuple[int, int], str], default: str = "fail") -> Any:
    return acting(lambda e: plan.get((e.extra["seed"], e.extra["attempt"]), default))  # type: ignore[arg-type, return-value]


def rows(setup: Any) -> list[LedgerRow]:
    return Ledger(setup.run_dir / LIFT_RED / "S2" / "ledger.jsonl").rows()


async def practice(
    setup: Any,
    model: ScriptedModel,
    brains: dict[str, Any],
    scene: SceneConfig | None = None,
    **config: Any,
) -> PracticeResult:
    extra = [
        LocalProvider(Perception(), SceneConfig(pipeline=PERCEPTION) if scene is None else scene)
    ]
    async with setup.runner(brains, extra=extra, models=model.client) as runner:
        result = await runner.run(Practice(), LIFT_RED, {"start": 0, **config})
    assert isinstance(result, PracticeResult)
    return result


def marks(model: ScriptedModel) -> int:
    return sum(asked_for(body) == "set_of_mark" for body in model.requests)


def seed_dir(setup: Any, seed: int) -> Path:
    return Path(setup.run_dir / LIFT_RED / "S2" / str(seed))


LIFTED_A_LITTLE = DoneSpec(metric="gap_z", a="box", b="table", value=0.03, tol=0.02)


def criteria_memory() -> Any:
    """lift_memory(0.02) with a done criterion a 2 cm lift passes (no latch there: a false
    finish the criteria let through)."""
    doc = lift_memory(0.02)
    param = DoneSpecParam(value=LIFTED_A_LITTLE)
    return doc.replace(params={**doc.params, "box_up": param})


def checking(heights: dict[int, float]) -> Any:
    """An Actor lifting the box by `heights[seed]` (default 2 cm), then checking and finishing."""

    def make(episode: Any) -> ScriptedBrain:
        turn = 0

        async def script(request: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal turn
            turn += 1
            if turn == 1:
                height = heights.get(episode.extra["seed"], 0.02)
                return [call("lift", rationale=RATIONALE, color=color_of(request), height_m=height)]
            return [call("check_done")] if turn == 2 else [call("finish", note="done")]

        return ScriptedBrain(script, NO_OBSERVE)

    return make


async def gated(setup: Any, brains: dict[str, Any], **config: Any) -> PracticeResult:
    extra = [
        LocalProvider(Perception(), SceneConfig(pipeline=PERCEPTION)),
        LocalProvider(WorldState(), SceneConfig(pipeline=PERCEPTION)),
        LocalProvider(Done(), SceneConfig(pipeline=PERCEPTION)),
    ]
    model = ScriptedModel(grounding)
    async with setup.runner(brains, extra=extra, models=model.client) as runner:
        result = await runner.run(Practice(), LIFT_RED, {"start": 0, **config})
    assert isinstance(result, PracticeResult)
    return result


async def test_a_finish_the_criteria_passed_but_the_latch_did_not_fails_the_try(
    setup: Any,
) -> None:
    """Under done criteria a finish needs a passing check_done; one the official check does not
    confirm fails the try (fail-fast), and the Reviser is asked to tighten the criteria."""
    await setup.start(criteria_memory())
    await gated(setup, {"actor": checking({}), "reviser": editing(lambda e: [])}, max_seeds=1)
    [row] = rows(setup)
    assert [t.end for t in row.tries] == ["finish_unconfirmed"] * 3
    assert row.failure_labels == ["finish unconfirmed"] * 3
    try1 = seed_dir(setup, 10000) / "try1"
    # each try's finish was first refused `finish_refusals` times while the latch had not fired
    results = [e.data for e in read_events(try1 / "events.jsonl") if e.kind == "tool.result"]
    refused = [r for r in results if r["name"] == "finish" and r["code"] == "finish_refused"]
    latch = [r for r in refused if "official success check is false" in json.dumps(r["content"])]
    assert len(latch) == PracticeConfig.model_fields["finish_refusals"].default
    relations = json.loads((try1 / "relations.json").read_text(encoding="utf-8"))
    assert relations["success"] is False
    brief = (seed_dir(setup, 10000) / "cache" / "reviser" / "r1" / "llm_calls.jsonl").read_text(
        encoding="utf-8"
    )
    assert "Its finish passed check_done, yet the official check did not confirm" in brief
    assert "pass gap_z box table" in brief
    # A Reviser that leaves the criteria as they were has its finish refused once
    # (`PracticeConfig.false_finish_edit`).
    events = (seed_dir(setup, 10000) / "cache" / "reviser" / "r1" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "try 1 finished although the official check did not confirm success" in events


_HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,64}\b")  # ids and hashes, not numbers


def _records(setup: Any, paths: list[Path]) -> dict[str, Any]:
    """The files at `paths` (JSON, JSON lines) by their path under the run directory."""
    out: dict[str, Any] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        key = str(path.relative_to(setup.run_dir))
        if path.suffix == ".jsonl":
            out[key] = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            out[key] = json.loads(text)
    return out
