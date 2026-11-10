"""The Tool Hub (docs/concepts/agent-runtime.md): mounts toolsets, keeps episodes, and passes every
tool call through the guard pipeline.

```python
async with Hub([LocalProvider(Mock())], blobs=store, phases=method.phases) as hub:
    session = await hub.open_episode(OpenEpisode(stage="s2", budgets=Budgets(max_turns=45)))
    outcome = await hub.call(session.episode_id, ToolCall(call_id="c1", name="observe",
                                                          arguments={}, origin="model"))
```
"""

from roboscript.hub.config import HubConfig, RenderConfig
from roboscript.hub.hub import Hub
from roboscript.hub.pipeline import Call, Guard, Pipeline, Reject, assemble, load_guard
from roboscript.hub.steps import DEFAULT_GUARDS, DEFAULT_PIPELINE, GUARDS
from roboscript.hub.store import ActionNotes, FinishGuard
from roboscript.hub.visibility import Phases, Tool

__all__ = [
    "DEFAULT_GUARDS",
    "DEFAULT_PIPELINE",
    "GUARDS",
    "ActionNotes",
    "Call",
    "FinishGuard",
    "Guard",
    "Hub",
    "HubConfig",
    "Phases",
    "Pipeline",
    "Reject",
    "RenderConfig",
    "Tool",
    "assemble",
    "load_guard",
]
