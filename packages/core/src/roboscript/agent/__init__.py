"""The own agent loop (docs/concepts/learning.md, docs/concepts/agent-runtime.md): brains that
drive a model through the Hub.

```python
session = AgentSession(episode=await hub.open_episode(spec), hub=hub, instructions=prompt,
                       blobs=blobs, recorder=Recorder(episode_dir, events), models=openai_client)
result = await ResponsesBrain(ResponsesConfig(reasoning_effort="low")).run(session)
```

- `session`: `AgentSession`, `Limits`, `Brain`, `BrainResult` (its end reasons)
- `loop`: the turn state machine brains share (`run_loop`, `LoopConfig`, `Reply`, `ModelError`)
- `history`: what a request shows of the episode (`HistoryPolicy`, `build_input`)
- `context`: the context budget every request is held to (`ContextBudget`, `ImageCost`)
- `recording`: `llm_calls.jsonl`, `turns.jsonl`, model events, `rebuild_requests`
- `responses`: `ResponsesBrain`, the reference brain; `scripted`: `ScriptedBrain` (a script, a
  config, or a replay of an episode's `turns.jsonl`)
- `prompts`: `render_prompt` for the method's templates (`recipes/wicl/prompts/`)
- `host`: `AgentHostService`, the agent's side of episodes a benchmark's own client drives (P3),
  served as the `agent.host` plane (process: `roboscript.learning.agent_host`)
"""

from roboscript.agent.context import ContextBudget, ImageCost
from roboscript.agent.history import HistoryPolicy
from roboscript.agent.loop import LoopConfig, ModelError, Reply, run_loop
from roboscript.agent.prompts import placeholders, render_prompt
from roboscript.agent.recording import Recorder, read_lines, rebuild_requests
from roboscript.agent.responses import ResponsesBrain, ResponsesConfig
from roboscript.agent.scripted import (
    ScriptedBrain,
    ScriptedCall,
    ScriptedConfig,
    call,
    replayed,
    say,
    think,
)
from roboscript.agent.session import AgentSession, Brain, BrainResult, Limits, Usage

__all__ = [
    "AgentSession",
    "Brain",
    "BrainResult",
    "ContextBudget",
    "HistoryPolicy",
    "ImageCost",
    "Limits",
    "LoopConfig",
    "ModelError",
    "Recorder",
    "Reply",
    "ResponsesBrain",
    "ResponsesConfig",
    "ScriptedBrain",
    "ScriptedCall",
    "ScriptedConfig",
    "Usage",
    "call",
    "placeholders",
    "read_lines",
    "rebuild_requests",
    "render_prompt",
    "replayed",
    "run_loop",
    "say",
    "think",
]
