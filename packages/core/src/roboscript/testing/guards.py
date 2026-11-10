"""The guard conformance suite (`roboscript.guards`): a registered guard is a `Guard`, its
default config validates and survives a config file (JSON), its step has the shape its phase
needs, it composes from its plugin item (as a run's `RunSpec.plugins` and a served Hub's
`HubServerConfig.guards` name it; a built-in is in every pipeline already), and a Hub over the
built-in pipeline with the guard added still serves a plain call.

```python
class TestRateLimit(GuardConformance):
    guard_name = "rate_limit"
```

A guard that is not registered (a package that is not installed) is named by its import path,
`guard_name = "mypackage.guards:MY_GUARD"`, as a run's plugin item names it.
"""

import inspect
import json
import tempfile
from pathlib import Path
from typing import ClassVar

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.tools import ToolCall
from roboscript.hub import DEFAULT_PIPELINE, GUARDS, Guard, Hub, assemble, load_guard
from roboscript.learning.compose import compose_run
from roboscript.learning.runspec import PluginItem
from roboscript.testing import unit
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider


class GuardConformance:
    """`guard_name`: the guard's name in `roboscript.guards`, or its import path
    `package.module:attribute`."""

    guard_name: ClassVar[str]

    def _guard(self) -> Guard:
        return load_guard(self.guard_name)

    @unit
    async def test_registered(self) -> None:
        guard = self._guard()
        if ":" not in self.guard_name:  # registered: under its own name
            assert guard.name == self.guard_name
        step = guard.make(guard.Config())
        assert inspect.iscoroutinefunction(step)
        wanted = 2 if guard.phase == "around" else 1
        assert len(inspect.signature(step).parameters) == wanted

    async def test_it_composes_from_its_item(self) -> None:
        guard = self._guard()
        defaults = guard.Config()
        assert guard.Config.model_validate(json.loads(defaults.model_dump_json())) == defaults
        if guard.name in vars(GUARDS):  # built in: every pipeline has it, none adds it twice
            async with compose_run([]) as run:
                assert run.pipeline == DEFAULT_PIPELINE
            return
        item = PluginItem(
            group="guards", name=self.guard_name, overrides=json.loads(defaults.model_dump_json())
        )
        async with compose_run([item]) as run:
            phase = guard.phase
            added = len(getattr(run.pipeline, phase)) - len(getattr(DEFAULT_PIPELINE, phase))
        assert added == 1

    async def test_a_hub_with_it_serves_a_call(self) -> None:
        guard = self._guard()
        builtin = list(vars(GUARDS).values())
        guards = builtin if guard.name in vars(GUARDS) else [*builtin, guard]
        with tempfile.TemporaryDirectory() as directory:
            blobs = LocalBlobStore(Path(directory))
            hub = Hub(
                [LocalProvider(Mock())],
                blobs=blobs,
                phases=TEST_PHASES,
                pipeline=assemble(guards),
            )
            async with hub:
                session = await hub.open_episode(OpenEpisode(stage="s2"))
                call = ToolCall(call_id="c1", name="look", arguments={}, origin="model")
                outcome = await hub.call(session.episode_id, call)
        assert not outcome.is_error, outcome.content
