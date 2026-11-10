"""roboscript.kernel: services, hooks and the plugin lifecycle."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from roboscript.kernel import (
    Context,
    EventBus,
    Hook,
    Kernel,
    KernelError,
    NoConfig,
    Plugin,
    PluginContext,
    ServiceKey,
)

X = ServiceKey[str]("x")
Y = ServiceKey[str]("y")
Z = ServiceKey[str]("z")
SEEN = Hook[str]("seen", "emit")

Setup = Callable[[PluginContext, Any], Awaitable[None]]


def make_plugin(
    name: str,
    log: list[str],
    *,
    requires: Sequence[ServiceKey[Any]] = (),
    provides: Sequence[ServiceKey[Any]] = (),
    config: type[BaseModel] = NoConfig,
    setup: Setup | None = None,
) -> Plugin:
    """A plugin that provides each declared key (value: its name) and logs load and unload."""

    async def default_setup(ctx: PluginContext, cfg: Any) -> None:
        log.append(f"load {name}")
        for key in provides:
            ctx.provide(key, name)
        ctx.on(SEEN, log.append)
        ctx.defer(lambda: log.append(f"unload {name}"))

    async def run_setup(self: Plugin, ctx: PluginContext, cfg: Any) -> None:
        await (setup or default_setup)(ctx, cfg)

    attributes = {
        "name": name,
        "requires": tuple(requires),
        "provides": tuple(provides),
        "Config": config,
        "setup": run_setup,
    }
    return type(f"{name}_plugin", (Plugin,), attributes)()


def test_context_provides_gets_and_revokes() -> None:
    ctx = Context()
    assert not ctx.has(X)
    revoke = ctx.provide(X, "value")
    assert ctx.has(X) and ctx.get(X) == "value"
    with pytest.raises(KernelError, match="'x' is already provided"):
        ctx.provide(X, "other")
    revoke()
    revoke()  # a second call does nothing
    assert not ctx.has(X)
    with pytest.raises(LookupError, match="no service 'x'"):
        ctx.get(X)
    ctx.provide(X, "again")
    assert ctx.get(X) == "again"


async def test_emit_runs_observers_in_priority_order_and_logs_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus, seen = EventBus(), []

    def broken(value: str) -> None:
        raise RuntimeError("observer broke")

    async def late(value: str) -> None:
        seen.append(f"late {value}")

    bus.on(SEEN, broken)
    bus.on(SEEN, late, priority=-1)
    bus.on(SEEN, lambda value: seen.append(f"second {value}"))
    bus.on(SEEN, lambda value: seen.append(f"first {value}"), priority=10)
    with caplog.at_level(logging.ERROR, logger="roboscript.kernel.bus"):
        assert await bus.call(SEEN, "x") is None
    assert seen == ["first x", "second x", "late x"]
    assert "observer broke" in caplog.text


async def test_kernel_loads_providers_first_and_unloads_in_reverse() -> None:
    log: list[str] = []

    async def uses_x(ctx: PluginContext, cfg: Any) -> None:
        log.append(f"load b with {ctx.get(X)}")
        ctx.provide(Y, "b")
        ctx.defer(lambda: log.append("unload b"))

    plugins = [
        make_plugin("c", log, requires=[Y]),
        make_plugin("b", log, requires=[X], provides=[Y], setup=uses_x),
        make_plugin("a", log, provides=[X]),
        make_plugin("d", log),
    ]
    kernel = Kernel(plugins)
    assert [plugin.name for plugin in kernel.order] == ["a", "b", "c", "d"]
    async with kernel as ctx:
        assert log == ["load a", "load b with a", "load c", "load d"]
        assert ctx.get(Y) == "b"
    assert log[4:] == ["unload d", "unload c", "unload b", "unload a"]


@pytest.mark.parametrize(
    ("plugins", "message"),
    [
        (lambda log: [make_plugin("a", log, requires=[X])], "'a' requires 'x', which no plugin"),
        (
            lambda log: [make_plugin("a", log, provides=[X]), make_plugin("b", log, provides=[X])],
            "'x' is provided by both 'a' and 'b'",
        ),
        (
            lambda log: [
                make_plugin("a", log, requires=[Y], provides=[X]),
                make_plugin("b", log, requires=[Z], provides=[Y]),
                make_plugin("c", log, requires=[Y], provides=[Z]),
            ],
            "cycle: b -> c -> b",  # a only waits on the cycle
        ),
        (lambda log: [make_plugin("a", log, requires=[X], provides=[X])], "cycle: a -> a"),
        (lambda log: [make_plugin("a", log), make_plugin("a", log)], "'a' is listed twice"),
    ],
    ids=["missing", "duplicate provider", "cycle", "self cycle", "duplicate name"],
)
def test_wiring_errors_are_raised_before_any_plugin_runs(
    plugins: Callable[[list[str]], list[Plugin]], message: str
) -> None:
    log: list[str] = []
    with pytest.raises(KernelError, match=message):
        Kernel(plugins(log))
    assert log == []


async def test_a_failing_setup_unloads_everything_loaded_so_far() -> None:
    log: list[str] = []
    ctx = Context()

    async def breaks(plugin_ctx: PluginContext, cfg: Any) -> None:
        plugin_ctx.provide(Y, "partial")
        plugin_ctx.defer(lambda: log.append("unload b"))
        raise RuntimeError("setup broke")

    kernel = Kernel(
        [
            make_plugin("a", log, provides=[X]),
            make_plugin("b", log, provides=[Y], setup=breaks),
            make_plugin("c", log),
        ],
        ctx=ctx,
    )
    with pytest.raises(RuntimeError, match="setup broke"):
        async with kernel:
            pass
    assert log == ["load a", "unload b", "unload a"]
    assert not ctx.has(X) and not ctx.has(Y) and ctx.bus.hooks() == []
