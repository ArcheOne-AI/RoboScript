"""The information-boundary step of skillset validation (docs/how-to/add-a-skillset.md).

A skillset's tools run while the Agent executes, so what they compute must come from data a real
robot has. `check_boundary` reads, without importing anything, the sources of
a skillset's implementation (the modules of its extensions; for a core skillset, those of the
toolsets whose tools it describes) and every module they import, directly or not, within their own
top-level package or the core's (`roboscript`), parents included, and reports:

- an import of the control plane or of privileged modules (`FORBIDDEN`, plus an env's own, e.g.
  its simulator side);
- a use of the control plane's names (`NAMES`) in the implementation modules themselves;
- a tool it adds that is not tier D (a skillset's tools run where the Agent executes).

Imports into other packages (third-party ones) are checked against the forbidden list but not
followed. The core's layering and a benchmark's agent and sim sides are also import-linter's
contracts (`.importlinter`); this check is what a built skillset, which has no contract of its
own, gets, and it catches a route through the core's test helpers (`roboscript.testing`).
"""

import ast
import importlib.util
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

from roboscript.tools import Toolset
from roboscript.tools.skillsets import SkillSetPlugin, load_skillset

FORBIDDEN: Final = (
    "roboscript.toolsets.privileged",
    "roboscript.envs",
    "roboscript.learning",
    "roboscript.orchestrate",
    "roboscript.demos.referee",
    "roboscript.demos.gt_scene",
)
"""Modules a skillset's implementation never imports: the privileged toolset, the env side
(servers, the Mock simulator, the control token), the harness, the referee and the ground-truth
scene."""
CORE: Final = "roboscript"
NAMES: Final = frozenset({"EnvControl", "GtSnapshot", "gt_snapshot", "Diagnosis"})
"""The control plane's names: its protocol, its ground-truth snapshot and diagnosis."""


def check_boundary(
    skillset: str | SkillSetPlugin,
    hosts: Mapping[str, type[Toolset]] | None = None,
    forbidden: Sequence[str] = (),
) -> list[str]:
    """What crosses the information boundary in `skillset` (a registered name, an import path
    or a plugin), one line each (empty: nothing): see the module. `hosts`, the core toolsets by
    name, gives a core skillset's implementation (the toolsets that have its tools); `forbidden`
    adds an env's privileged modules to `FORBIDDEN`."""
    plugin = load_skillset(skillset) if isinstance(skillset, str) else skillset
    name = plugin.manifest.name
    lines = [
        f"{name}.{spec.name}: tier {spec.meta.tier}, a skillset's tools are tier D"
        for ext in plugin.extensions
        for spec in ext.specs()
        if spec.meta.tier != "D"
    ]
    skills = {skill.name for skill in plugin.manifest.skills}
    modules = [ext.__module__ for ext in plugin.extensions]
    modules += [
        host.__module__
        for host in (hosts or {}).values()
        if plugin.manifest.layer == "core" and skills & {spec.name for spec in host.specs()}
    ]
    banned = (*FORBIDDEN, *forbidden)
    for module in dict.fromkeys(modules):
        lines += [f"{name}: {line}" for line in _scan(module, banned)]
    return lines


def _scan(root: str, banned: Sequence[str]) -> list[str]:
    """Imports of `banned` modules by `root` and what it imports within its top-level package,
    and `NAMES` used in `root` itself."""
    tops = {root.partition(".")[0], CORE}
    lines: list[str] = []
    hits: dict[tuple[str, str], str] = {}  # (module, banned prefix): the shortest name imported
    seen: set[str] = set()
    todo = [root, *_parents(root)]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _source(module)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for imported in _imports(tree, package):
            if hit := next((b for b in banned if _within(imported, b)), None):
                known = hits.setdefault((module, hit), imported)
                hits[module, hit] = min(known, imported, key=len)
            elif imported.partition(".")[0] in tops:
                todo += [imported, *_parents(imported)]
        if module == root:
            lines += [f"{module} names {used}" for used in sorted(_names(tree) & NAMES)]
    lines += [
        f"{module} imports {name} ({hit} is off limits)" for (module, hit), name in hits.items()
    ]
    return sorted(lines)


def _imports(tree: ast.AST, package: str) -> Iterable[str]:
    """The absolute names of the modules `tree` imports; `from a import b` gives `a` and `a.b`
    (b may be a module)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                anchor = ".".join(parts[: len(parts) - node.level + 1])
                base = f"{anchor}.{base}" if base else anchor
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names if alias.name != "*")


def _names(tree: ast.AST) -> set[str]:
    """The identifiers `tree` uses: names, attributes and imported names."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.rpartition(".")[2])
    return found


def _within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _parents(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts))]


def _source(module: str) -> Path | None:
    """`module`'s source file, found under its top-level package's directory without importing
    anything (None: not a module, e.g. a name imported from a package)."""
    top, _, rest = module.partition(".")
    spec = importlib.util.find_spec(top)
    if spec is None or not spec.submodule_search_locations:
        return None
    for directory in spec.submodule_search_locations:
        base = Path(directory).joinpath(*rest.split(".")) if rest else Path(directory)
        for path in (base / "__init__.py", base.with_suffix(".py")):
            if path.is_file():
                return path
    return None
