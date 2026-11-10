"""The runner identifies once as an episode opens (the perception toolset mounted, a scripted
model): an Actor's Memory names abstract roles and the live instruction binds them, and any other
failure is a configuration error that stops the run."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from roboscript.agent import read_lines
from roboscript.contracts.env import EpisodeOpened, EpisodeSpec
from roboscript.learning import Episode, ProtocolRunner
from roboscript.learning.opening import SAVED, Reuse, load
from roboscript.memory import MemoryDoc
from roboscript.records import read_events
from roboscript.testing.perception import ScriptedModel, asked_for, instruction_of
from roboscript.testing.protocols import LIFT_RED, acting, grounding, lift_memory
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import LocalProvider
from roboscript.toolsets.perception import Perception
from roboscript.toolsets.scene import SceneConfig


def episode(
    runner: ProtocolRunner,
    directory: Path,
    role: str,
    memory: MemoryDoc,
    reuse: Reuse | None = None,
    spec: EpisodeSpec | None = None,
) -> Episode:
    def prompt(opened: EpisodeOpened | None) -> tuple[str, None]:
        assert opened is not None and opened.instruction is not None
        if role == "actor":
            text = runner.prompts.actor(
                instruction=opened.instruction, memory=memory, rules="s2", profile=runner.profile
            )
        else:
            text = runner.prompts.demo_reader(
                instruction=opened.instruction, memory=memory, profile=runner.profile
            )
        return text, None

    stage = "s2" if role == "actor" else "s1"
    return Episode(
        task=LIFT_RED,
        stage=stage,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        directory=directory,
        prompt=prompt,
        spec=spec or EpisodeSpec(task=LIFT_RED, seed=10000, stage=stage),  # type: ignore[arg-type]
        memory=memory,
        reuse=reuse,
    )


EYE = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def perception(**config: Any) -> LocalProvider:
    return LocalProvider(
        Perception(), SceneConfig.model_validate({"pipeline": PERCEPTION, **config})
    )


async def test_an_actor_starts_from_what_its_roles_and_instruction_identify(setup: Any) -> None:
    model = ScriptedModel(grounding)
    memory = lift_memory(0.06)  # "- box: the box the instruction names; ..."
    directory = setup.run_dir / "actor"
    brain = acting(lambda _: "fail", observe=True)  # observe, then finish
    async with setup.runner({"actor": brain}, extra=[perception()], models=model.client) as runner:
        outcome = await runner.driver.run(episode(runner, directory, "actor", memory))
    calls = read_lines(directory / "tool_calls.jsonl")
    assert [(c["name"], c["origin"]) for c in calls] == [
        ("identify", "system"),
        ("observe", "system"),
        ("finish", "model"),
    ]
    arguments = calls[0]["arguments"]
    assert arguments["instruction"] == outcome.instruction
    assert arguments["roles"] == [{"role": "box", "description": memory.objects[7:]}]
    [request] = model.requests
    assert asked_for(request) == "set_of_mark" and instruction_of(request) == outcome.instruction
    observed = " ".join(part.get("text", "") for part in calls[1]["content"])
    assert "Registered objects:" in observed and "- obj_1 box: center (" in observed
    first = (directory / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "The objects were identified once" in first and "Objects in frame_1:" in first
    assert outcome.end.reason == "finish" and outcome.end_class == "policy"  # no done criteria
    assert outcome.identification is not None and outcome.identification.status == "identified"
    assert (outcome.perception_calls, outcome.perception_usage.input_tokens) == (1, 1000)
    assert outcome.tokens >= 1020  # the perception call counts in the episode's cost
    saved = load(directory)
    assert saved is not None and saved.source == "actor" and (directory / SAVED).is_file()
    events = list(read_events(directory / "events.jsonl"))
    [clock] = [e for e in events if e.kind == "episode.clock"]
    assert clock.data["reason"] == "identification" and clock.data["spent_s"] >= 0


@pytest.mark.parametrize(
    ("objects", "refuse", "found"),
    [
        (None, True, "identify: tool_error"),  # every model request refused
        (  # past the identification's limits, which the linter enforces on every edit
            "- box: " + "x" * 250,
            False,
            "OBJECTS line 1: the description has 250 characters, more than 200",
        ),
        ("- box: fine\nnot a role line", False, "OBJECTS line 2"),  # a malformed section
    ],
)
async def test_any_other_failure_is_a_configuration_error(
    setup: Any, objects: str | None, refuse: bool, found: str
) -> None:
    def refusing(body: dict[str, Any]) -> Any:
        return httpx.Response(400, json={"error": {"message": "no"}}) if refuse else grounding(body)

    model = ScriptedModel(refusing)
    memory = lift_memory(0.06)
    if objects is not None:
        memory = memory.model_copy(update={"objects": objects})
    directory = setup.run_dir / "config"
    brain = acting(lambda _: "pass", observe=True)
    async with setup.runner({"actor": brain}, extra=[perception()], models=model.client) as runner:
        with pytest.raises(ValueError, match="a configuration error"):
            await runner.driver.run_counted(episode(runner, directory, "actor", memory))
    summary = (directory / "summary.json").read_text(encoding="utf-8")
    assert '"end_class": "config"' in summary and "config_error:identify" in summary
    assert found in summary
    kinds = {e.kind for e in read_events(directory / "events.jsonl")}
    assert "model.request" not in kinds  # the brain never ran
