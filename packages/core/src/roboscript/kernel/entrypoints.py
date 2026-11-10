"""Extension points: package entry points (listed without importing, loaded by name) and
in-repo references by import path (`package.module:attribute`), e.g. benchmark examples."""

import importlib
from importlib import metadata
from typing import Any

from roboscript.kernel.context import KernelError


def names(group: str) -> list[str]:
    """The names registered in `group`, sorted; imports nothing."""
    return sorted(_entries(group))


def names_in(group: str, package: str) -> list[str]:
    """The names registered in `group` whose objects live in the top-level `package` (by the
    module their entry names), sorted; imports nothing."""
    entries = _entries(group).items()
    return sorted(name for name, entry in entries if entry.module.partition(".")[0] == package)


def load(group: str, name: str) -> Any:
    """Import and return the object registered as `name` in `group`."""
    entries = _entries(group)
    if name not in entries:
        raise LookupError(
            f"nothing named {name!r} in entry-point group {group!r}; registered: {sorted(entries)}"
        )
    return entries[name].load()


def load_ref(ref: str) -> Any:
    """Import and return the object `package.module:attribute` names (the attribute may be
    dotted), e.g. `my_package.benchmark:BENCHMARK`."""
    module, sep, attribute = ref.partition(":")
    if not sep or not module or not attribute:
        raise ValueError(f"{ref!r} is not a reference like 'package.module:attribute'")
    obj = importlib.import_module(module)
    for name in attribute.split("."):
        obj = getattr(obj, name)
    return obj


def _entries(group: str) -> dict[str, metadata.EntryPoint]:
    found: dict[str, metadata.EntryPoint] = {}
    for entry in metadata.entry_points(group=group):
        if entry.name in found:
            raise KernelError(
                f"{entry.name!r} in entry-point group {group!r} is registered by both "
                f"{_distribution(found[entry.name])} and {_distribution(entry)}"
            )
        found[entry.name] = entry
    return found


def _distribution(entry: metadata.EntryPoint) -> str:
    return entry.dist.name if entry.dist is not None else f"<unknown> ({entry.value})"
