"""Config composition on plain dicts: deep merges, then dotted `key.path=value` overrides."""

import copy
from collections.abc import Collection, Sequence
from typing import Any

import yaml


def compose(
    base: dict[str, Any],
    *patches: dict[str, Any],
    overrides: Sequence[str] = (),
    replace: Collection[str] = (),
) -> dict[str, Any]:
    """Merge `patches` into a copy of `base` in order (dicts merge recursively, anything else
    replaces), then apply `overrides` such as `gateway.cooldown_s=60`, whose values are parsed as
    YAML (`60` is an int, `true` a bool, `null` None, `[a, b]` a list; quote other strings that
    contain `: `). `replace`: dotted key paths (`*` stands for any one key) whose value a patch
    that names them replaces whole instead of merging into, so that a later patch can drop keys
    an earlier one set (`learning.brains.*`: each brain a later patch names is its alone). The
    inputs are not modified."""
    result = copy.deepcopy(base)
    patterns = [tuple(pattern.split(".")) for pattern in replace]
    for patch in patches:
        _merge(result, patch, (), patterns)
    for override in overrides:
        _apply(result, override)
    return result


def _merge(
    into: dict[str, Any],
    patch: dict[str, Any],
    path: tuple[str, ...] = (),
    replace: Sequence[tuple[str, ...]] = (),
) -> None:
    for key, value in patch.items():
        where = (*path, str(key))
        whole = any(_matches(where, pattern) for pattern in replace)
        if not whole and isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value, where, replace)
        else:
            into[key] = copy.deepcopy(value)


def _matches(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if len(path) != len(pattern):
        return False
    return all(p in ("*", k) for k, p in zip(path, pattern, strict=True))


def _apply(tree: dict[str, Any], override: str) -> None:
    key, sep, text = override.partition("=")
    path = key.strip().split(".")
    if not sep or not all(path):
        raise ValueError(f"override {override!r} must look like key.path=value")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"override {override!r}: the value is not valid YAML") from error
    node = tree
    for depth, name in enumerate(path[:-1], start=1):
        node = node.setdefault(name, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {override!r}: {'.'.join(path[:depth])} is not a mapping")
    node[path[-1]] = value
