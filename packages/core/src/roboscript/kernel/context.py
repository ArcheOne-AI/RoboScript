"""Typed services by key, and the per-plugin view that undoes its registrations at unload."""

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Generic, TypeVar, cast

from roboscript.kernel.bus import EventBus, Hook, Revoke

T = TypeVar("T")


class KernelError(Exception):
    """Services or plugins that cannot work together as declared or configured."""


class ServiceKey(Generic[T]):
    """Names a service of type T. Keys compare by identity: declare each one once, as a constant."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"ServiceKey({self.name!r})"


class Context:
    """Services by key plus the hook bus that plugins share."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self.bus = EventBus() if bus is None else bus
        self._services: dict[ServiceKey[Any], Any] = {}

    def provide(self, key: ServiceKey[T], value: T) -> Revoke:
        if key in self._services:
            raise KernelError(f"service {key.name!r} is already provided")
        self._services[key] = value
        return Revoke(lambda: self._services.pop(key))

    def get(self, key: ServiceKey[T]) -> T:
        try:
            return cast(T, self._services[key])
        except KeyError:
            raise LookupError(
                f"no service {key.name!r}; is the plugin that provides it loaded?"
            ) from None

    def has(self, key: ServiceKey[Any]) -> bool:
        return key in self._services

    def on(self, hook: Hook[T], handler: Callable[..., Any], *, priority: int = 0) -> Revoke:
        """Register a handler on the shared bus."""
        return self.bus.on(hook, handler, priority=priority)


class PluginContext(Context):
    """What a plugin's `setup` receives: the shared services and bus, where everything registered
    through it is undone, in reverse order, when the plugin unloads."""

    def __init__(
        self,
        parent: Context,
        plugin: str,
        provides: tuple[ServiceKey[Any], ...],
        stack: AsyncExitStack,
    ) -> None:
        self.bus = parent.bus
        self._services = parent._services
        self._plugin = plugin
        self._provides = provides
        self._stack = stack

    def provide(self, key: ServiceKey[T], value: T) -> Revoke:
        if key not in self._provides:
            raise KernelError(
                f"plugin {self._plugin!r} provides {key.name!r} without declaring it in `provides`"
            )
        return self._track(super().provide(key, value))

    def on(self, hook: Hook[T], handler: Callable[..., Any], *, priority: int = 0) -> Revoke:
        return self._track(super().on(hook, handler, priority=priority))

    def defer(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        """Run `callback` at unload (awaited if it returns an awaitable), in reverse order with
        the plugin's other registrations: close clients, stop servers, undo other registries."""
        self._stack.push_async_callback(_call, callback)

    async def enter_async_context(self, cm: AbstractAsyncContextManager[T]) -> T:
        """Enter `cm` now and exit it at unload, in reverse order with the plugin's other
        registrations (an HTTP client, a server's lifespan)."""
        return await self._stack.enter_async_context(cm)

    def _track(self, revoke: Revoke) -> Revoke:
        self._stack.callback(revoke)
        return revoke


async def _call(callback: Callable[[], Awaitable[None] | None]) -> None:
    result = callback()
    if result is not None:
        await result
