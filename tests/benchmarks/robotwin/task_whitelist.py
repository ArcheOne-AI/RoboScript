"""The AST check that keeps RoboTwin's task object behind the shim
(docs/architecture.md): where a module's top-level functions use their
task-object parameter."""

import ast
from collections.abc import Callable
from pathlib import Path

Chain = tuple[str, ...]


def task_uses(
    path: Path, name: str, passes: Callable[[ast.Call, set[str]], bool]
) -> tuple[set[Chain], list[int]]:
    """The attribute chains read on the parameter `name` of the module's top-level functions
    (`task.robot.get_qpos` -> `("robot", "get_qpos")`), and the lines where `name` is used
    otherwise: outside a function taking it, rebound, or not as an argument of a call that
    `passes(call, local_function_names)` accepts."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    local = {function.name for function in functions}
    used: set[Chain] = set()
    stray: list[int] = []
    inside: set[ast.AST] = set()
    for function in functions:
        names = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == name]
        inside.update(names)
        if names and name not in {arg.arg for arg in function.args.args}:
            stray.append(function.lineno)
        for use in names:
            node: ast.AST = use
            chain: list[str] = []
            while isinstance(parents.get(node), ast.Attribute):
                node = parents[node]
                chain.append(node.attr)  # type: ignore[attr-defined]
            call = parents.get(use)
            handed = isinstance(call, ast.Call) and use in call.args and passes(call, local)
            if chain:
                used.add(tuple(chain))
            elif isinstance(use.ctx, ast.Store) or not handed:
                stray.append(use.lineno)
    stray += [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == name and n not in inside
    ]
    return used, stray


def to_local(call: ast.Call, local: set[str]) -> bool:
    """A call of a function of the same module."""
    return isinstance(call.func, ast.Name) and call.func.id in local


def to_shim(call: ast.Call, local: set[str]) -> bool:
    """A call `shim.<function>(...)`."""
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "shim"
    )
