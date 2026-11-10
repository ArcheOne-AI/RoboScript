"""In-process plugins of your own: a Hub guard and an event sink.

Guards, event sinks and prechecks are not servers. They run in the run's own process, are
registered in an entry-point group (`roboscript.guards`, `roboscript.event_sinks`,
`roboscript.prechecks`), or named by import path (`custom_plugin.plugins:FORBID_ARM`), and are
listed in a run's `plugins`. Each is plain data: a name, a
pydantic config and a function that makes the working part from the validated config.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from roboscript.contracts.events import Event, EventKind
from roboscript.hub.pipeline import Call, Guard, Reject, Step
from roboscript.records.sinks import EventSink, Sink

# ---- a guard: refuse the calls that name an arm this run keeps off limits -------------------


class ForbidArmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)  # an unknown key is an error

    arms: list[str] = []


def forbid_arm(config: ForbidArmConfig) -> Step:
    async def step(call: Call) -> None:
        arm = call.request.arguments.get("arm")
        if arm in config.arms:
            # The code is stable (tests and logs match on it); the message is what the model reads.
            raise Reject("arm_off_limits", f"The {arm} arm is not available in this run.")

    step.__name__ = "forbid_arm"
    return step


# Phase `pre` is before the action lock and the budget: a refused call takes no turn. With no
# `after`, the guard goes to the end of its phase, when arguments are validated.
FORBID_ARM: Final = Guard(
    name="forbid_arm", phase="pre", make=lambda config: forbid_arm(config), Config=ForbidArmConfig
)

# ---- an event sink: count the tools the model called, written when the run closes -----------


class TallyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = "tool_tally.json"


class ToolTally:
    def __init__(self, config: TallyConfig, run_dir: Path) -> None:
        self.path = run_dir / config.file  # a sink writes only inside the run directory
        self.counts: Counter[str] = Counter()

    def write(self, event: Event) -> None:
        if event.kind == EventKind.TOOL_CALL and event.data["origin"] == "model":
            self.counts[event.data["name"]] += 1

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(self.counts), sort_keys=True), encoding="utf-8")


def _make_tally(config: TallyConfig, run_dir: Path) -> EventSink:
    return ToolTally(config, run_dir)


TOOL_TALLY: Final = Sink(name="tool_tally", Config=TallyConfig, make=_make_tally)
