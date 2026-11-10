"""Test support for RoboScript and its plugins.

`roboscript.testing.pytest_plugin` records every pytest session under `test-results/`;
`roboscript.testing.gateway` serves the model gateway in front of scripted fake upstreams;
`roboscript.testing.env`, `roboscript.testing.planner`, `roboscript.testing.brain` and
`roboscript.testing.perception` hold the env, planner, brain, grounder and segmenter
conformance suites (and the scripted model and vision service perception tests use),
`roboscript.testing.memory` and `roboscript.testing.artifacts` the Memory store and artifact
store suites, `roboscript.testing.transport` and `roboscript.testing.assignment` those of
dispatch's transports and assignment policies, `roboscript.testing.embodiments`,
`roboscript.testing.toolsets`, `roboscript.testing.benchmarks` and
`roboscript.testing.routing` those of embodiment profiles, toolsets, benchmark descriptors and
gateway routing policies, `roboscript.testing.scenes` synthetic RGB-D
scenes, and `roboscript.testing.external` the suite for external brains over MCP (a served Hub
behind the gateway and a scripted upstream; needs the `mcp` extra). `roboscript.testing.host`
holds `MockClient`, a benchmark's own evaluation client over `MockEnv` (for the Agent host), and
`roboscript.testing.hold` a toolset whose `hold` executes through the env. Conformance suites
are classes of async `test_*` checks that take a target; pytest runs them in `Test*` subclasses,
`precheck` without pytest. Checks marked `unit` need no running episode; the others are
`integration`. A suite whose subject is named by a class attribute (guards, event sinks, prechecks)
has checks without a target: `precheck(SubclassWithTheName())`.

A check that does not apply to its target, by what the target declares (an env with no expert
pre-run, say), ends with `not_applicable(reason)`: pytest reports it skipped, `precheck` as
`not applicable: <reason>`.

`registered(group, distribution)` lists the names a distribution registers in an entry-point
group: the built-ins a suite is run against, through the entry point, as a third-party package's
suite would be.
"""

import inspect
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import metadata
from typing import Any, Final, NoReturn

NOT_APPLICABLE: Final = "not applicable: "
"""How `precheck` reports a check that does not apply to its target (followed by the reason)."""

_IN_PRECHECK: ContextVar[bool] = ContextVar("roboscript_in_precheck", default=False)


def unit(check: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a check of the `unit` tier."""
    check.tier = "unit"  # type: ignore[attr-defined]
    return check


class NotApplicable(Exception):
    """A check does not apply to its target (raised by `not_applicable`)."""


def not_applicable(reason: str) -> NoReturn:
    """End a check that does not apply to its target, by what the target declares: under pytest
    the check is skipped with `reason`; run by `precheck` (or anywhere else) it raises
    `NotApplicable`, which `precheck` reports as `not applicable: <reason>`."""
    pytest = sys.modules.get("pytest")
    if pytest is not None and not _IN_PRECHECK.get():
        pytest.skip(reason)
    raise NotApplicable(reason)


def registered(group: str, distribution: str = "roboscript") -> list[str]:
    """The names `distribution` registers in entry-point group `group`, sorted; empty where it is
    not installed with its entry points (the simulator env runs the sources by `PYTHONPATH`)."""
    return sorted(
        entry.name
        for entry in metadata.entry_points(group=group)
        if entry.dist is not None and entry.dist.name == distribution
    )


@contextmanager
def raises(error: type[BaseException], match: str) -> Iterator[None]:
    """`pytest.raises(error, match=match)` for suites that run without pytest: the block raises
    `error` with a message `match` (a regular expression) is found in."""
    try:
        yield
    except error as caught:
        if re.search(match, str(caught)) is None:
            raise AssertionError(
                f"{type(caught).__name__} does not match {match!r}: {caught}"
            ) from caught
        return
    raise AssertionError(f"{error.__name__} was not raised")


async def precheck(
    suite: object, target: object = None, tiers: Iterable[str] = ("unit", "integration")
) -> dict[str, str | None]:
    """Run the checks of `suite` in the given tiers against `target`, without pytest: check name
    -> None if it passed, `not applicable: <reason>` if it does not apply to the target
    (`not_applicable`), else why it failed. A check that takes no target is run without one."""
    wanted = set(tiers)
    results: dict[str, str | None] = {}
    inside = _IN_PRECHECK.set(True)
    try:
        for name in [n for n in dir(suite) if n.startswith("test_")]:
            check = getattr(suite, name)
            if getattr(check, "tier", "integration") not in wanted:
                continue
            try:
                await (check(target) if inspect.signature(check).parameters else check())
                results[name] = None
            except NotApplicable as reason:
                results[name] = f"{NOT_APPLICABLE}{reason}"
            except Exception as error:
                results[name] = f"{type(error).__name__}: {error}"
    finally:
        _IN_PRECHECK.reset(inside)
    return results
