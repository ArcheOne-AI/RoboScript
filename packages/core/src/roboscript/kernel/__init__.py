"""Composition kernel (docs/concepts/agent-runtime.md): typed services, plugins whose registrations
are undone at unload, hooks with fixed dispatch modes, entry points and config composition; and
`roboscript.kernel.signals`, signals as requests to stop (shared by `rpc.serve` and the stack)."""

from roboscript.kernel.bus import EventBus, Hook, Mode, Revoke
from roboscript.kernel.compose import compose
from roboscript.kernel.context import Context, KernelError, PluginContext, ServiceKey
from roboscript.kernel.plugin import Kernel, NoConfig, Plugin

__all__ = [
    "Context",
    "EventBus",
    "Hook",
    "Kernel",
    "KernelError",
    "Mode",
    "NoConfig",
    "Plugin",
    "PluginContext",
    "Revoke",
    "ServiceKey",
    "compose",
]
