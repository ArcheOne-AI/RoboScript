"""Task-level dispatch over several machines, like DDP for pipelines
(docs/concepts/operations.md, docs/how-to/dispatch.md): each worker runs one
pipeline launch (a unit) over its share of the tasks on its GPUs; finished tasks move to the
coordinator's run root, the one copy.

- `config`: `DispatchConfig`, `WorkerSpec`, `UnitSpec`, `load_dispatch`, `provenance`
- `assign`: the assignment policies (`roboscript.dispatch_policies`; built-in `balanced`)
- `transport`: how a worker is reached (`roboscript.dispatch_transports`; `ssh`, `local`)
- `agent`: the worker side, run over the transport in the worker's code copy
- `coordinator`: `Coordinator` (assign, start, watch, move, resume, reassign, drain, status)
- `files`: manifests and their checks
"""

from roboscript.orchestrate.dispatch.config import (
    DispatchConfig,
    PinnedPath,
    UnitSpec,
    WorkerSpec,
    load_dispatch,
)
from roboscript.orchestrate.dispatch.coordinator import Coordinator, DispatchFailed

__all__ = [
    "Coordinator",
    "DispatchConfig",
    "DispatchFailed",
    "PinnedPath",
    "UnitSpec",
    "WorkerSpec",
    "load_dispatch",
]
