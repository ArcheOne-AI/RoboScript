"""The process supervisor (docs/concepts/operations.md):
`async with up("stack.yaml") as running:`.

Starts each service of a stack with its interpreter (the agent venv, the simulator env or the
vision env) and the stack's blob store directory (`RS_BLOB_DIR`), gates on its ready file and
`/readyz`, records `<run_dir>/.roboscript/endpoints.json`, fails fast when a service exits, and
stops every process group on exit, with a bound (`stop_groups`, also for processes started
elsewhere). `cancel_on_signals` turns signals such as SIGTERM, and SIGHUP unless ignored
(`nohup`), into the cancellation of a body, so that its cleanup runs; `on_signals`, the registry
under it (`roboscript.kernel.signals`), also serves `rpc.serve`. `least_loaded` picks a GPU for
a service whose config names none.
"""

from roboscript.kernel.signals import Stopped, cancel_on_signals, on_signals
from roboscript.stack.config import Interpreters, ServiceSpec, StackConfig
from roboscript.stack.gpus import least_loaded
from roboscript.stack.supervisor import Running, StackError, ready_path, stop_groups, up

__all__ = [
    "Interpreters",
    "Running",
    "ServiceSpec",
    "StackConfig",
    "StackError",
    "Stopped",
    "cancel_on_signals",
    "least_loaded",
    "on_signals",
    "ready_path",
    "stop_groups",
    "up",
]
