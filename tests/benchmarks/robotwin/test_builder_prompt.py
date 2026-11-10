"""The skill builder's prompt names no RoboTwin task: the builder is the same for every benchmark,
and the tasks it builds for reach it as cases, never as prompt text."""

import textwrap

import yaml

from roboscript_methods.skill_builder.builder.codex import PROMPT
from roboscript_robotwin.benchmark import HERE


def test_the_prompt_is_generic() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    for placeholder in ("$case_runs", "$at_once", "$wall_h", "$tokens_m"):
        assert placeholder in text
    listed = yaml.safe_load((HERE / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]
    tasks = {t.lower() for t in listed}
    assert len(tasks) == 50
    words = set(text.lower().replace("`", " ").split())
    assert not (tasks & words), "the builder prompt names a task"
    assert textwrap.dedent(text) == text
