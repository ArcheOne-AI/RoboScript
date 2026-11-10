"""A run's composition from its plugins (guards, event sinks, prechecks) through the kernel, and
the built-in prechecks."""

import json
from pathlib import Path

from roboscript.hub import DEFAULT_PIPELINE
from roboscript.learning.compose import compose_run, open_sinks
from roboscript.learning.events import EventRouter
from roboscript.learning.prechecks import (
    DEFAULT_PRECHECKS,
)
from roboscript.learning.runspec import PluginItem
from roboscript.testing.plugins import PrecheckConformance, SinkConformance


async def test_no_plugins_is_the_builtin_run() -> None:
    async with compose_run([]) as run:
        assert run.pipeline == DEFAULT_PIPELINE
        assert run.sinks == ()
        assert [p.name for p, _ in run.prechecks] == [p.name for p in DEFAULT_PRECHECKS]


async def test_plugins_add_a_guard_a_sink_and_a_precheck(tmp_path: Path) -> None:
    plugins = [
        PluginItem(group="guards", name="rate_limit", overrides={"max_calls": 5}),
        PluginItem(group="event_sinks", name="jsonl", overrides={"file": "copy.jsonl"}),
        PluginItem(group="prechecks", name="gpu_headroom"),
    ]
    async with compose_run(plugins) as run:
        assert run.pipeline.pre[-1].__name__ == "rate_limit"
        assert run.pipeline.guards == DEFAULT_PIPELINE.guards
        assert [p.name for p, _ in run.prechecks][-1] == "gpu_headroom"
        with (
            open_sinks(run.sinks, tmp_path) as sinks,
            EventRouter(tmp_path / "events.jsonl", sinks=sinks) as router,
        ):
            router.emit("probe", {"n": 1})
    copied = [
        json.loads(line)
        for line in (tmp_path / "copy.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    logged = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert copied == logged and logged[0]["kind"] == "probe"


class TestJsonl(SinkConformance):
    sink_name = "jsonl"


class TestEnvCapabilities(PrecheckConformance):
    precheck_name = "env_capabilities"


class TestSeedSplit(PrecheckConformance):
    precheck_name = "seed_split"


class TestSkillsets(PrecheckConformance):
    precheck_name = "skillsets"
