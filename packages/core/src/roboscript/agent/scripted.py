"""`ScriptedBrain`: the own loop with scripted replies in place of a model, to test protocols,
toolsets and the loop itself without one (docs/concepts/agent-runtime.md). Its requests are built
and recorded exactly as a model's would be; only nothing is sent.

```python
brain = ScriptedBrain([[call("get_robot_state")], "text only", [call("finish", note="done")]])
brain = ScriptedBrain.from_config(ScriptedConfig(replay=Path("runs/.../turns.jsonl")))
```
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from roboscript.agent.loop import LoopConfig, ModelError, Reply, run_loop
from roboscript.agent.recording import read_lines
from roboscript.agent.session import AgentSession, BrainResult
from roboscript.contracts.ids import new_id

Step = Sequence[dict[str, Any]] | str
"""A reply: output items (`call`, `say`, `think`) or plain text."""
Script = Sequence[Step] | Callable[[dict[str, Any]], Awaitable[Step]]
"""Replies in order, or a function of the request (as sent) that returns the reply."""


class ScriptedCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ScriptedConfig(LoopConfig):
    """The loop's settings and the script: `replies`, each a list of calls or a text reply, or
    `replay`, an earlier episode's `turns.jsonl` whose replies are played again in order."""

    replies: list[list[ScriptedCall] | str] = Field(default_factory=list)
    replay: Path | None = None

    @model_validator(mode="after")
    def _one_script(self) -> Self:
        if bool(self.replies) == (self.replay is not None):
            raise ValueError("give either replies or replay")
        return self


def call(name: str, /, **arguments: Any) -> dict[str, Any]:
    """A function call of `name` with `arguments`."""
    return {
        "type": "function_call",
        "call_id": new_id("call"),
        "name": name,
        "arguments": json.dumps(arguments),
    }


def say(text: str) -> dict[str, Any]:
    """A text reply."""
    content = [{"type": "output_text", "text": text, "annotations": []}]
    return {"type": "message", "id": new_id("msg"), "role": "assistant", "content": content}


def think(summary: str = "") -> dict[str, Any]:
    """A reasoning item with (made up) encrypted content, as a reasoning model returns it."""
    return {
        "type": "reasoning",
        "id": new_id("rs"),
        "summary": [{"type": "summary_text", "text": summary}] if summary else [],
        "encrypted_content": f"enc-{new_id('x')}",
    }


def replayed(turns: Path) -> list[Step]:
    """The replies an episode's `turns.jsonl` records: each exchange's text and calls, the calls
    with their arguments as sent (also ones that were not a JSON object)."""
    steps: list[Step] = []
    for turn in read_lines(turns):
        for exchange in turn["exchanges"]:
            items = [say(exchange["text"])] if exchange["text"] else []
            for made in exchange["calls"]:
                arguments = made["arguments"]
                text = arguments if isinstance(arguments, str) else json.dumps(arguments)
                items.append({**call(made["name"]), "arguments": text})
            steps.append(items)
    return steps


class ScriptedBrain:
    """Answers each model call with the next step of `script` (a script out of steps ends the
    episode as `script_exhausted`) or with `script(request)`."""

    Config = ScriptedConfig

    def __init__(
        self, script: Script, config: LoopConfig | None = None, model: str = "scripted"
    ) -> None:
        self.script = script
        self.config = LoopConfig() if config is None else config
        self.model = model
        self._next = 0

    @classmethod
    def from_config(cls, config: ScriptedConfig) -> "ScriptedBrain":
        if config.replay is not None:
            return cls(replayed(config.replay), config)
        script: list[Step] = [
            reply if isinstance(reply, str) else [call(c.name, **c.arguments) for c in reply]
            for reply in config.replies
        ]
        return cls(script, config)

    async def run(self, session: AgentSession) -> BrainResult:
        options = {
            "model": self.model,
            "parallel_tool_calls": False,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        return await run_loop(session, self._reply, options, self.config)

    async def _reply(self, request: dict[str, Any]) -> Reply:
        if callable(self.script):
            step = await self.script(request)
        elif self._next < len(self.script):
            step = self.script[self._next]
            self._next += 1
        else:
            raise ModelError("script_exhausted", f"all {len(self.script)} replies are used")
        output = [say(step)] if isinstance(step, str) else list(step)
        return Reply(output=output, meta={"status": "completed"})
