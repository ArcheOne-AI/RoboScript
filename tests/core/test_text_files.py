"""Text files are read and written as UTF-8 whatever the locale (a benchmark's code and logs may
hold non-ASCII text, and a shell's locale need not be UTF-8): every `read_text`, `write_text` and
text-mode `open` in the packages, the examples and the tests names its encoding."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this checkout
SOURCES = ("packages", "examples", "tests")


def test_text_files_name_their_encoding() -> None:
    unnamed = [
        f"{path.relative_to(ROOT)}:{node.lineno}"
        for top in SOURCES
        for path in sorted((ROOT / top).rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and _locale_encoded(node)
    ]
    assert not unnamed, f'pass encoding="utf-8" at {", ".join(unnamed)}'


def _locale_encoded(call: ast.Call) -> bool:
    """`Path.read_text` / `write_text`, or the built-in `open` in text mode, without an
    encoding (a keyword or in its place)."""
    if any(keyword.arg == "encoding" for keyword in call.keywords):
        return False
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
        return len(call.args) < (1 if func.attr == "read_text" else 2)
    if not (isinstance(func, ast.Name) and func.id == "open"):
        return False
    given = [k.value for k in call.keywords if k.arg == "mode"] or call.args[1:2]
    binary = bool(given) and isinstance(given[0], ast.Constant) and "b" in str(given[0].value)
    return not binary and len(call.args) < 4
