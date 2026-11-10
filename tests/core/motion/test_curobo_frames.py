"""The frames `curobo.GRAPH_INTERPOLATION` matches exist in the installed cuRobo: each module
defines a function (or a method) of that name, and `find_paths` interpolates outside the `try`
around its search. Imports cuRobo, without a GPU."""

import ast
import importlib
import inspect
import textwrap

import pytest

from roboscript.motion.curobo import GRAPH_INTERPOLATION

pytestmark = pytest.mark.sim


def _defined(module: str) -> set[str]:
    """The names of the functions and methods defined in `module`."""
    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize(("module", "name"), GRAPH_INTERPOLATION)
def test_the_installed_curobo_defines_the_frame(module: str, name: str) -> None:
    pytest.importorskip("curobo")
    assert name in _defined(module)


def test_find_paths_interpolates_outside_the_try_around_its_search() -> None:
    pytest.importorskip("curobo")
    from curobo.graph.graph_base import GraphPlanBase

    tree = ast.parse(textwrap.dedent(inspect.getsource(GraphPlanBase.find_paths)))
    guarded = {
        call.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for statement in node.body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    called = {
        call.func.attr
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert "_find_paths" in guarded
    assert "get_interpolated_trajectory" in called - guarded
