"""`catalog.lock` (docs/concepts/agent-runtime.md): the mounted tools as one hash, which every
episode token carries and run manifests record.

A lock lists each mounted tool by qualified name with its descriptor hash (name, description and
input schema: what a model is shown) and its tier, and the method's phases (which roles see which
tools, in what order, at which tier, under which rules; a `toolset.+` entry as the tools it stands
for on this mount, `toolset.tool` each, so that the lock names what the roles see). A served Hub
started with a lock file refuses to run when its tools differ from it; after an intended change,
`write_lock` records the new state.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import ToolName
from roboscript.contracts.phase import PhaseSpec
from roboscript.hub.visibility import Tool, added_tools, mounted_skillsets
from roboscript.records import write_atomic

SCHEMA: Final = "rs-catalog-lock/2"


def catalog_lock(tools: Mapping[ToolName, Tool], phases: Mapping[str, PhaseSpec]) -> dict[str, Any]:
    entries = {
        tool.qualified: {"hash": tool.descriptor.hash, "tier": tool.meta.tier}
        for tool in tools.values()
    }
    by_toolset: dict[str, list[ToolName]] = {}
    for name, tool in tools.items():
        by_toolset.setdefault(tool.meta.toolset, []).append(name)
    skillsets = mounted_skillsets(tools)
    declared = []
    for phase in phases.values():
        # `optional` decides only whether a start fails, never what an episode sees, and
        # `d_only` only refuses what the tiers locked here already rule out: both left out, so
        # that marking an entry optional or a phase D only keeps the lock
        data = phase.model_dump(mode="json", exclude={"optional", "d_only"})
        data["visibility"] = {
            role: [named for entry in entries for named in _named(entry, by_toolset, skillsets)]
            for role, entries in data["visibility"].items()
        }
        declared.append(data)
    return {"schema": SCHEMA, "tools": entries, "phases": declared}


def _named(
    entry: str, by_toolset: Mapping[str, list[ToolName]], skillsets: Mapping[str, Sequence[str]]
) -> list[str]:
    toolset, _, name = entry.partition(".")
    if name != "+":
        return [entry]
    mounted, refs = by_toolset.get(toolset, []), skillsets.get(toolset, ())
    return [f"{toolset}.{tool}" for tool in added_tools(toolset, mounted, refs)]


def lock_hash(lock: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(lock).encode("utf-8")).hexdigest()


def write_lock(path: Path, lock: Mapping[str, Any]) -> None:
    write_atomic(path, (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def check_lock(path: Path, lock: Mapping[str, Any]) -> None:
    """Raise ValueError naming every tool that differs from the lock file at `path`."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved == lock:
        return
    old, new = saved.get("tools", {}), lock["tools"]
    changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
    raise ValueError(
        f"{path}: the mounted tools differ from the lock ({', '.join(changed) or 'schema'}); "
        "if the change is intended, write a new lock with roboscript.hub.lock.write_lock"
    )
