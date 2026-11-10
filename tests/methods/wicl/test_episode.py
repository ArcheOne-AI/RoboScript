"""The episode driver with wicl's runtime on MockEnv: S1's finish gate (wicl's done gate) refuses
a finish short of success, naming the gap to the demonstration's end."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from roboscript.agent import ScriptedBrain, call, read_lines
from roboscript.contracts.geometry import Pose
from roboscript.learning.demo import DemoTruth, TruthFrame
from roboscript.learning.runtime import MethodRuntime
from roboscript.testing.phases import TestMethod
from roboscript.testing.protocols import (
    LIFT_RED,
    RATIONALE,
    lift_memory,
)
from roboscript_methods.wicl.diagnosis import explain_try
from roboscript_methods.wicl.extension import UNCONFIRMED, WiclEpisode
from roboscript_methods.wicl.finish import DoneGate


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_test_episode", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


actor_episode = _load(
    Path(__file__).resolve().parents[2] / "core" / "learning" / "test_episode.py"
).actor_episode


class Explaining(TestMethod):
    """The test method with wicl's reading of a try's ground truth."""

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        return replace(super().runtime(recipe), explain=explain_try)


class DoneGated(TestMethod):
    """The test method with wicl's finish gate."""

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        return replace(super().runtime(recipe), finish=DoneGate())


async def test_s1_refuses_a_finish_short_of_success_with_the_gap_to_the_demonstration_s_end(
    setup: Any,
) -> None:
    """S1's privileged guard refuses each finish while the official check is false, up to
    `finish_refusals`, saying how far the ground truth is from the demonstration's success
    frame; the finish after those ends the attempt, failed."""
    lift = [call("lift", rationale=RATIONALE, color="red", height_m=0.02)]
    finish = [call("finish", note="done")]
    end = Pose(p=(0.0, 0.0, 0.9), q=(1.0, 0.0, 0.0, 0.0))
    target = DemoTruth(
        task=LIFT_RED,
        seed=10000,
        objects={"red": "box"},
        frames=[
            TruthFrame(frame=0, what=["start"], poses={"red": end}),
            TruthFrame(frame=9, what=["success"], poses={"red": end}),
        ],
    )
    directory = setup.run_dir / "attempt"
    brains = {"actor": lambda _: ScriptedBrain([lift, finish, finish, finish])}
    setup.method = DoneGated  # S1's finish rule is the method's (wicl's done gate)
    async with setup.runner(brains) as runner:
        episode = actor_episode(runner, directory, lift_memory(0.02))
        spec = episode.spec.model_copy(update={"stage": "s1"}) if episode.spec else None
        episode = replace(
            episode,
            stage="s1",
            spec=spec,
            finish_refusals=2,
            method=WiclEpisode(official_finish=True, finish_target=target),
        )
        outcome = await runner.driver.run(episode)
    # past its refusals the finish is not let through: the attempt ends as a failure
    assert (outcome.success, outcome.end.reason) == (False, UNCONFIRMED)
    assert outcome.end_class == "fail_fast"
    results = [
        e["data"] for e in read_lines(directory / "events.jsonl") if e["kind"] == "tool.result"
    ]
    refused = [r for r in results if r["name"] == "finish" and r["code"] == "finish_refused"]
    assert len(refused) == 2
    said = json.dumps(refused[0]["content"])
    assert "official success check is false" in said and "the box is" in said
    assert "cm off" in said and "red" not in said.split("Against")[-1]
