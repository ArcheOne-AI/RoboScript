"""The episode driver on MockEnv: one episode end to end, its directory, how it ends, and how
infrastructure outcomes are re-run (a voided run's Memory edits undone)."""

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from roboscript.agent import ScriptedBrain, call, read_lines
from roboscript.contracts.env import (
    EpisodeOpened,
    EpisodeSpec,
    JudgeResult,
)
from roboscript.contracts.hub import EpisodeEnd
from roboscript.contracts.ids import EpisodeId
from roboscript.envs.mock import TASKS, MockConfig
from roboscript.learning import (
    Episode,
    EpisodeOutcome,
    InfraExhausted,
    ProtocolRunner,
    RoleConfig,
)
from roboscript.learning.episode import EXPLAINED, voids
from roboscript.learning.runtime import MethodRuntime
from roboscript.memory import MemoryDoc
from roboscript.records import read_events
from roboscript.testing.phases import TestMethod
from roboscript.testing.protocols import (
    LIFT_RED,
    RATIONALE,
    ScriptedEnv,
    acting,
    finishing,
    lift_memory,
    lifting_actor,
)


def actor_episode(
    runner: ProtocolRunner, directory: Path, doc: MemoryDoc, seed: int = 10000, **extra: Any
) -> Episode:
    def prompt(opened: EpisodeOpened | None) -> tuple[str, None]:
        assert opened is not None and opened.instruction is not None
        text = runner.prompts.actor(
            instruction=opened.instruction, memory=doc, rules="s2", profile=runner.profile
        )
        return text, None

    spec = EpisodeSpec(task=LIFT_RED, seed=seed, stage="s2", prerun_attempts=3)
    return Episode(
        task=LIFT_RED,
        stage="s2",
        role="actor",
        directory=directory,
        prompt=prompt,
        spec=spec,
        diagnose=True,
        memory=doc,
        memory_label="v007",
        extra={"seed": seed, **extra},
    )


async def test_a_successful_episode_leaves_its_directory(setup: Any) -> None:
    directory = setup.run_dir / "episode"
    async with setup.runner({"actor": lifting_actor}) as runner:
        outcome = await runner.driver.run(actor_episode(runner, directory, lift_memory(0.06)))
    # The success latch ended the episode after the lift, before the actor could finish.
    assert (outcome.success, outcome.end_class, outcome.end.reason) == (True, "success", "success")
    assert outcome.turns == 1 and outcome.steps == outcome.success_step == 5
    files = {path.name for path in directory.iterdir()}
    assert files >= {"config.json", "events.jsonl", "llm_calls.jsonl", "turns.jsonl"}
    assert files >= {"physics.npz", "memory_before.md", "summary.json", "referee"}
    calls = read_lines(directory / "tool_calls.jsonl")
    assert [(c["name"], c["origin"], c["refused"]) for c in calls] == [
        ("observe", "system", False),
        ("lift", "model", False),
    ]
    assert calls[1]["arguments"]["height_m"] == 0.06 and calls[1]["result"]["reason"] == "ok"
    assert EpisodeOutcome.model_validate_json((directory / "summary.json").read_bytes()) == outcome
    assert {path.name for path in (directory / "referee").iterdir()} == {"success.json"}
    with np.load(directory / "physics.npz") as physics:
        assert physics["step"].tolist() == [0, 1, 2, 3, 4, 5]  # the start, then every step
        assert physics["turn"].tolist() == [0, 1, 1, 1, 1, 1]
        assert physics["right.tcp"].shape == physics["left.tcp"].shape == (6, 7)
    events = list(read_events(directory / "events.jsonl"))
    kinds = {event.kind for event in events}
    assert kinds >= {"episode.open", "env.observe", "tool.call", "model.request", "episode.end"}
    assert len({event.trace.trace_id for event in events if event.trace}) == 1
    [observed] = [event for event in events if event.kind == "env.observe"]
    assert observed.data["turn"] == 1  # the turn the frame was taken in
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert config["memory"] == {"label": "v007", "hash": outcome.memory}
    assert config["protocol"] == {"seed": 10000} and "lift" in config["tools"]
    assert config["env"]["prerun_attempts"] == 3 and config["role"] == "actor"
    assert "lift_height = 0.06 m" in (directory / "memory_before.md").read_text(encoding="utf-8")


class Seen(BaseModel):
    objects: list[str]


def name_objects(
    diagnosis: Any, registry: Any, alias: Any, robot: Any, shapes: Any, method: Any
) -> BaseModel:
    """A placeholder explainer (`MethodRuntime.explain`): the objects the referee diagnosed."""
    return Seen(objects=sorted(obj.name for obj in diagnosis.objects))


class Explaining(TestMethod):
    """The test method with a placeholder explainer."""

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        return replace(super().runtime(recipe), explain=name_objects)


async def test_a_failure_is_diagnosed_under_referee(setup: Any) -> None:
    setup.method = Explaining
    directory = setup.run_dir / "episode"
    async with setup.runner({"actor": lifting_actor}) as runner:
        outcome = await runner.driver.run(actor_episode(runner, directory, lift_memory(0.02)))
    # No done criteria in this Memory: its finish is no gated claim, just a failure.
    assert (outcome.success, outcome.end_class, outcome.end.reason) == (False, "policy", "finish")
    diagnosis = json.loads((directory / "referee" / "diagnostics.json").read_text(encoding="utf-8"))
    assert {obj["name"] for obj in diagnosis["objects"]} == {"red", "green", "blue"}
    explained = (directory / "referee" / EXPLAINED).read_text(encoding="utf-8")
    assert Seen.model_validate_json(explained).objects == ["blue", "green", "red"]
    sent = (directory / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "displacement" not in sent and "diagnos" not in sent
    assert "success.json" not in sent  # nothing of the referee


async def test_the_step_limit_ends_the_episode(tmp_path: Path, setup: Any) -> None:
    task = TASKS[LIFT_RED].model_copy(update={"step_limit": 7})
    config = MockConfig(task=LIFT_RED, blob_dir=tmp_path / "blobs", tasks={LIFT_RED: task})
    setup.env = ScriptedEnv(config, setup.blobs)
    lift = [call("lift", rationale=RATIONALE, color="red", height_m=0.02)]
    twice = [lift, lift, [call("finish", note="done")]]
    directory = setup.run_dir / "episode"
    async with setup.runner({"actor": lambda _: ScriptedBrain(twice)}) as runner:
        outcome = await runner.driver.run(actor_episode(runner, directory, lift_memory(0.02)))
    assert (outcome.end_class, outcome.end.reason, outcome.steps) == ("policy", "step_limit", 7)


async def test_infra_reruns_are_capped(setup: Any) -> None:
    async with setup.runner({"actor": acting(lambda _: "gateway")}, max_infra_reruns=1) as runner:
        with pytest.raises(InfraExhausted, match="2 runs in a row"):
            await runner.driver.run_counted(
                actor_episode(runner, setup.run_dir / "try1", lift_memory(0.06))
            )


async def test_the_bounds_follow_the_task_step_limit(setup: Any) -> None:
    """The Actor's from its env's step limit (MockEnv: 40), the env-less Reviser's from the one
    its episode carries."""
    scaled = {"turns": 0.05, "max_turns": 10, "model_calls": 0.5, "wall_clock_s": 60.0}
    roles = {
        role: RoleConfig.model_validate(
            {"budgets": {"max_turns": 1}, "limits": {"max_model_calls": 5}, "per_step": scaled}
        )
        for role in ("actor", "reviser")
    }

    def reviser(directory: Path, step_limit: int | None) -> Episode:
        return Episode(
            task=LIFT_RED,
            stage="s2",
            role="reviser",
            directory=directory,
            prompt=lambda _: ("Revise, then finish.", None),
            step_limit=step_limit,
        )

    def bounds(directory: Path) -> tuple[int, int, float]:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        role = config["role_config"]
        return (
            role["budgets"]["max_turns"],
            role["limits"]["max_model_calls"],
            role["budgets"]["wall_clock_s"],
        )

    brains = {"actor": lifting_actor, "reviser": finishing}
    async with setup.runner(brains, roles=roles) as runner:
        await runner.driver.run(actor_episode(runner, setup.run_dir / "a", lift_memory(0.06)))
        await runner.driver.run(reviser(setup.run_dir / "r", 400))
        await runner.driver.run(reviser(setup.run_dir / "r0", None))
        capped = dataclasses.replace(reviser(setup.run_dir / "rc", 400), max_turns=4)
        await runner.driver.run(capped)
        loose = dataclasses.replace(reviser(setup.run_dir / "rl", None), max_turns=4)
        await runner.driver.run(loose)
    assert bounds(setup.run_dir / "a") == (2, 20, 2400.0)
    assert bounds(setup.run_dir / "r") == (10, 200, 24000.0)  # 20 turns, capped at 10
    assert bounds(setup.run_dir / "r0")[:2] == (1, 5)  # no step limit: the configured bounds
    # An episode's own cap (S1's hinted attempts, `ImitateConfig.hinted_max_turns`): the smaller.
    assert bounds(setup.run_dir / "rc")[:2] == (4, 200)
    assert bounds(setup.run_dir / "rl")[:2] == (1, 5)


async def test_an_unsolvable_seed_opens_no_episode(setup: Any) -> None:
    directory = setup.run_dir / "try1"
    async with setup.runner({"actor": finishing}) as runner:
        episode = actor_episode(runner, directory, lift_memory(0.06), seed=10001)
        outcome = await runner.driver.run(episode)
    assert (outcome.end_class, outcome.solvable, outcome.episode_id) == ("unsolvable", False, None)
    assert {path.name for path in directory.iterdir()} == {"events.jsonl", "summary.json"}
    assert not read_lines(directory / "events.jsonl")


def test_an_unanswered_infrastructure_error_voids_only_where_the_success_did_not_latch() -> None:
    """`voids`: an Actor's episode is void after an unanswered infrastructure error unless the
    task's success latched, also when the official check passed while something other than the
    latch ended the episode (the latch fired as the client held still, say); never one that ended
    on infrastructure already, was cancelled or ended on a configuration error."""

    def end(reason: str) -> EpisodeEnd:
        return EpisodeEnd(reason=reason, at=0.0)

    def judged(success: bool) -> JudgeResult:
        return JudgeResult(
            episode_id=EpisodeId("e"), success=success, success_step=None, steps_used=5,
            step_limit=40,
        )  # fmt: skip

    assert voids(end("finish"), judged(False)) and voids(end("max_turns"), None)
    assert not voids(end("finish"), judged(True))  # passed, closed on something else
    assert not voids(end("step_limit"), judged(True))
    assert not voids(end("success"), None)
    for reason in ("infra_error", "env_ended", "cancelled", "model_error:config:bad"):
        assert not voids(end(reason), judged(False)), reason
