"""Conformance suites of the run's in-process plugins: event sinks (`roboscript.event_sinks`)
and prechecks (`roboscript.prechecks`). Guards have theirs in `roboscript.testing.guards`.

```python
class TestConsoleSummary(SinkConformance):
    sink_name = "console_summary"
```

A plugin that is not registered (a package that is not installed) is named by its import path,
`sink_name = "mypackage.sinks:MY_SINK"`, as a run's plugin item names it.
"""

import tempfile
import time
from pathlib import Path
from typing import ClassVar

from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.events import Event
from roboscript.learning.prechecks import RunFacts, load_precheck, run_prechecks
from roboscript.learning.runspec import RunSpec
from roboscript.records.sinks import load_sink
from roboscript.testing import unit


class SinkConformance:
    """`sink_name`: the sink's name in `roboscript.event_sinks` (or its import path
    `package.module:attribute`). With its default config it opens in a run directory, takes
    events, flushes and closes, writing only inside that directory."""

    sink_name: ClassVar[str]

    @unit
    async def test_takes_events(self) -> None:
        sink = load_sink(self.sink_name)
        if ":" not in self.sink_name:  # registered: under its own name
            assert sink.name == self.sink_name
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            opened = sink.make(sink.Config(), run_dir)
            for seq in range(3):
                opened.write(Event(ts=time.time(), seq=seq, service="test", kind="probe"))
            opened.flush()
            opened.close()
            assert _beside(run_dir) == []


def _beside(run_dir: Path) -> list[Path]:
    return [path for path in run_dir.parent.iterdir() if path != run_dir]


class PrecheckConformance:
    """`precheck_name`: the precheck's name in `roboscript.prechecks` (or its import path
    `package.module:attribute`). With its default config it passes a run that is fine (`facts`,
    which a subclass may replace)."""

    precheck_name: ClassVar[str]
    facts: ClassVar[RunFacts] = RunFacts(
        spec=RunSpec.model_validate(
            {
                "env": {"name": "mock"},
                "embodiment": {"name": "mock_bimanual"},
                "method": {"name": "wicl"},
                "tasks": ["lift_red"],
                "seeds": {"blocks": {"s2": {"start": 10000, "count": 60}}},
            }
        ),
        offered=EnvCapabilities(success={"latch"}, reset={"seed"}),
        required=EnvCapabilities(success={"latch"}),
    )

    @unit
    async def test_passes_a_fine_run(self) -> None:
        precheck = load_precheck(self.precheck_name)
        if ":" not in self.precheck_name:  # registered: under its own name
            assert precheck.name == self.precheck_name
        assert await run_prechecks([(precheck, {})], self.facts) == []
