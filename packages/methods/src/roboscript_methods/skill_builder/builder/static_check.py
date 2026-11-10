"""The gate's static checks on a written skill's directory (`README.md`, "The gate", item 1), on
the source alone: nothing the Agent wrote is imported here (`plugins.validate` imports it only
after these pass, in a child process).

- layout: `skill.py`, `fit.py` and `cases.yaml` are there (what the files define is the contract
  check, `plugins.load_one`);
- imports: an allowlist per file (`SKILL_IMPORTS`, `FIT_IMPORTS`: a few pure standard-library
  modules, numpy and the core's contracts, spatial and manipulation; `fit.py` also the coverage
  stage's own segment and fit helpers); no relative import;
- escapes, in both files: none of `DENIED_NAMES` (`getattr`, `eval`, `open`, `format`, `print`,
  `SystemExit`, ...); no class pattern in a `match` (`case Api(executor=e)`, `case
  object(__class__=c)` read attributes no `.x` shows); no dunder attribute but `__name__`, no
  private attribute (`_x`), no frame or traceback attribute (`f_`, `tb_`, `gi_`, `cr_`, `ag_`), no
  `format` method, no attribute or imported name that is a module the allowlist leaves out (`os`
  reached through an allowed module) or that reads or writes files (`IO`: numpy's `load`, `save*`,
  `fromfile`, `tofile`, `memmap`, `loadtxt`, `genfromtxt`, `lib`; a core loader `load_*`; pydantic's
  `parse_file`), evaluates text (`get_type_hints`, `ForwardRef`) or reads every field of a dataclass
  (`asdict`, `astuple`, `fields`: the action API is one); no string literal holding `__` (a dunder
  spelled as data); no dunder method defined but `__init__`, `__post_init__`; no attribute assigned
  or deleted but on `self` (no monkeypatching, no edit of the API, the recording or the scene it is
  handed);
- `skill.py` does not run what it builds (`.run`, `.executor`: the caller runs each motion,
  judged, `demo_coverage.run.explain`);
- no task-specific content in either file, against the vocabulary the
  benchmark's glue supplies (`cover_demos.py --vocab`): a task name anywhere, comments
  included, its words joined by `_`, space or `-`; an object label (and its stem, the label
  without a trailing index: `block_2`, `block`) as a word of a string literal other than a
  docstring, where a label keys behaviour (`obj.name == ...`). Terms shorter than
  `MIN_TERM` characters are not checked, nor a label that is a word of the API itself
  (`api_words`: a name or a string of the allowed core modules, a builtin, a keyword; a benchmark
  may label objects `object1`, `target0`, `frame`): code needs those words whatever the task. The
  same vocabulary checks a fit's `params` (`demo_coverage.fitting.fit_written`);
- no position in `skill.py`: a number that is not a whole number and is larger in magnitude than
  `TOLERANCE` is a value the demonstration gave (a place, a height, an offset), which `fit`
  supplies through `params`; smaller ones are tolerances and clearances, whole numbers are counts
  and degrees, and `HALF` is a midpoint. Constant arithmetic is folded first (`37 * 0.01`,
  `np.array([42, -18]) / 100`), as is string concatenation for the label check (`"blo" + "ck"`):
  defence in depth, the gate's transfer cases are what tell a relative skill from a replay.

    python -m roboscript_methods.skill_builder.builder.static_check DIR... --vocab FILE
    (exit 1 on a finding)
"""

import argparse
import ast
import builtins
import functools
import importlib.util
import json
import keyword
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import yaml

STDLIB: Final = frozenset({
    "__future__", "abc", "bisect", "collections", "copy", "dataclasses", "enum", "functools",
    "heapq", "itertools", "math", "statistics", "typing",
})  # fmt: skip
CORE: Final = ("roboscript.contracts", "roboscript.spatial", "roboscript.manipulation")
SKILL_IMPORTS: Final = (*CORE, "numpy")
FIT_IMPORTS: Final = (
    *SKILL_IMPORTS,
    "json",
    "roboscript_methods.skill_builder.demo_coverage.segments",
    "roboscript_methods.skill_builder.demo_coverage.fitting",
)
"""`fit.py` reads the recording and the truth scene it is handed, through these."""
DENIED_NAMES: Final = frozenset({
    "__import__", "breakpoint", "compile", "delattr", "eval", "exec", "format", "getattr",
    "globals", "help", "input", "locals", "open", "setattr", "vars",
    "print", "exit", "quit", "SystemExit", "KeyboardInterrupt", "GeneratorExit", "BaseException",
})  # fmt: skip
"""`print` and the exits: a process's output and exit status are no skill's to write (the
validator's rows, a case's end); `BaseException` and its non-`Exception` kinds pass the
callers' `except Exception`."""
FRAME: Final = re.compile(r"^(f|tb|gi|cr|ag)_")
IO: Final = re.compile(
    r"^(load|save|dump|read|write)(_|txt$|z|$)|^(open|fromfile|tofile|memmap|genfromtxt|"
    r"fromregex|lib|ctypeslib|DataSource|f2py|distutils|testing|parse_file|get_type_hints|"
    r"ForwardRef|format|format_map|asdict|astuple|fields)$"
)
"""Names that read or write a file or evaluate text; `ActionApi`'s own methods are exempt
(`open_gripper`)."""
REACHED: Final = frozenset({
    "asyncio", "builtins", "code", "codeop", "concurrent", "ctypes", "dbm", "faulthandler",
    "fileinput", "ftplib", "gc", "glob", "http", "importlib", "imp", "inspect", "io",
    "linecache", "marshal", "mmap", "multiprocessing", "nt", "operator", "os", "pathlib",
    "pickle", "pkgutil", "posix", "pty", "pwd", "resource", "runpy", "select", "selectors",
    "shelve", "shutil", "signal", "site", "smtplib", "socket", "sqlite3", "ssl", "subprocess",
    "sys", "sysconfig", "tarfile", "tempfile", "threading", "traceback", "tracemalloc", "types",
    "urllib", "webbrowser", "xmlrpc", "zipfile", "zipimport",
})  # fmt: skip
"""Modules that reach a process, a file, the network or the interpreter, as an allowed module's
attribute (`typing.sys`, `api_module.os`): imports are an allowlist, these are its back door."""
DUNDER_METHODS: Final = frozenset({"__init__", "__post_init__"})
RUNS: Final = frozenset({"run", "executor"})
TOLERANCE: Final = 0.05
HALF: Final = 0.5
MIN_TERM: Final = 3


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: [{self.rule}] {self.message}"


@dataclass(frozen=True)
class Vocabulary:
    tasks: frozenset[str]
    labels: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """`{tasks: [...], labels: [...]}` (YAML or JSON)."""
        found = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not (isinstance(found, dict) and {"tasks", "labels"} <= set(found)):
            raise SystemExit(f"{path}: give a mapping with lists `tasks` and `labels`")
        return cls.of(found["tasks"], found["labels"])

    @classmethod
    def of(cls, tasks: Iterable[str], labels: Iterable[str]) -> "Vocabulary":
        words = {w.lower() for label in labels for w in (label, stem(label))}
        return cls(
            frozenset(t.lower() for t in tasks if len(t) >= MIN_TERM),
            frozenset(w for w in words if len(w) >= MIN_TERM) - api_words(),
        )

    def tasks_in(self, text: str) -> list[tuple[int, str]]:
        """(offset, task) of each task named in `text`, its words joined by `_`, space or `-`."""
        found = []
        for task in sorted(self.tasks):
            words = r"[_\s-]+".join(map(re.escape, task.split("_")))
            pattern = rf"(?<![A-Za-z0-9]){words}(?![A-Za-z0-9])"
            found += [(m.start(), task) for m in re.finditer(pattern, text, re.I)]
        return found

    def labels_in(self, text: str) -> list[str]:
        """The labels that are words of `text`."""
        return sorted({w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text)} & self.labels)


def stem(label: str) -> str:
    """A label without its trailing index: `block_2`, `box0`, `cat0_1` -> `block`, `box`, `cat`."""
    return re.sub(r"[_\d]+$", "", label)


@functools.lru_cache(maxsize=1)
def api_words() -> frozenset[str]:
    """The words a skill writes because the API it may use does: the names and the non-docstring
    strings' words of the core modules it may import, the builtins, the keywords."""
    words = {w.lower() for w in dir(builtins)} | set(keyword.kwlist)
    for name in CORE:
        found = importlib.util.find_spec(name)
        assert found is not None and found.submodule_search_locations is not None
        for path in Path(next(iter(found.submodule_search_locations))).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = {id(d) for d in docstring_nodes(tree)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    words.add(node.id.lower())
                elif isinstance(node, ast.Attribute):
                    words.add(node.attr.lower())
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    words.add(node.name.lower())
                elif isinstance(node, ast.arg):
                    words.add(node.arg.lower())
                elif isinstance(node, ast.Constant) and id(node) not in docstrings:
                    words |= {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", str(node.value))}
    return frozenset(words)


@functools.lru_cache(maxsize=1)
def api_methods() -> frozenset[str]:
    from roboscript.manipulation import ActionApi  # the core's, never the Agent's code

    return frozenset(n for n in dir(ActionApi) if not n.startswith("_"))


def check(directory: Path, vocabulary: Vocabulary) -> list[Finding]:
    """Every finding on the skill in `directory`; none: it passes."""
    missing = [f for f in ("skill.py", "fit.py", "cases.yaml") if not (directory / f).is_file()]
    if missing:
        return [Finding(str(directory), 0, "layout", f"missing {', '.join(missing)}")]
    found = []
    for name, imports in (("skill.py", SKILL_IMPORTS), ("fit.py", FIT_IMPORTS)):
        path = directory / name
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, str(path))
        except SyntaxError as error:
            found.append(Finding(name, error.lineno or 0, "syntax", str(error.msg)))
            continue
        for at, task in vocabulary.tasks_in(source):
            line = source.count("\n", 0, at) + 1
            found.append(Finding(name, line, "task", f"names the task {task!r}"))
        found += labels(name, tree, vocabulary)
        found += escapes(name, tree, imports)
        if name == "skill.py":
            found += runs(name, tree) + positions(name, tree)
    return sorted(found, key=lambda f: (f.file, f.line, f.rule))


def labels(file: str, tree: ast.AST, vocabulary: Vocabulary) -> list[Finding]:
    docstrings = {id(d) for d in docstring_nodes(tree)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = None if id(node) in docstrings else node.value
        else:
            text = joined(node) if isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Call)) else None
        for word in vocabulary.labels_in(text or ""):
            found.append(Finding(file, node.lineno, "label", f"names the object {word!r}"))
    return sorted(set(found), key=lambda f: (f.line, f.message))


def joined(node: ast.AST) -> str | None:
    """The text a string expression of literals makes (`"blo" + "ck"`, `"ab" * 2`, an f-string's
    literal parts, `"".join(["blo", "ck"])`), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values if isinstance(v, ast.Constant)]
        return "".join(str(p) for p in parts) if len(parts) > 1 else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = joined(node.left), joined(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        text, times = joined(node.left), node.right
        if text is not None and isinstance(times, ast.Constant) and type(times.value) is int:
            return text * min(times.value, 2)
        return None
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join" and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):  # fmt: skip
        separator, items = joined(node.func.value), [joined(e) for e in node.args[0].elts]
        if separator is not None and all(i is not None for i in items):
            return separator.join(i for i in items if i is not None)
    return None


def docstring_nodes(tree: ast.AST) -> list[ast.Constant]:
    found = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):  # a statement right after an assignment documents it, too
            for statement in body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                    found.append(statement.value)
    return found


def escapes(file: str, tree: ast.AST, imports: tuple[str, ...]) -> list[Finding]:
    """The ways out of the file's tier: imports, names, attributes, dunder strings, mutation."""
    docstrings = {id(d) for d in docstring_nodes(tree)}
    found = []

    def add(node: ast.AST, rule: str, message: str) -> None:
        found.append(Finding(file, getattr(node, "lineno", 0), rule, message))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in imported(node):
                if not allowed(name, imports):
                    add(node, "import", f"imports {name}")
                elif whys := [w for p in name.split(".")[1:] if (w := denied_attribute(p))]:
                    add(node, "import", f"imports {name}: {whys[0]}")
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            add(node, "call", f"uses {node.id}")
        elif isinstance(node, ast.MatchClass):
            add(node, "attribute", "a class pattern reads the matched object's attributes")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            add(node, "dunder", f"uses {node.id}")
        elif isinstance(node, ast.Attribute) and (why := denied_attribute(node.attr)):
            rule = "dunder" if node.attr.startswith("__") else "attribute"
            add(node, rule, f"reaches .{node.attr}: {why}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value and id(node) not in docstrings:
                add(node, "dunder", "a string holds `__`")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name not in DUNDER_METHODS:
                add(node, "dunder", f"defines {node.name}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.Attribute) and root(part) != "self":
                        add(node, "mutates", f"assigns .{part.attr} of {root(part) or 'a value'}")
                        break
    return found


def imported(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Each dotted name the statement imports: `from a.b import c` -> `a.b.c`."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = "." * node.level + (node.module or "")
    return [f"{module}.{alias.name}" for alias in node.names]


def denied_attribute(name: str) -> str | None:
    if name.startswith("__"):
        return None if name == "__name__" else "a dunder"
    if name.startswith("_"):
        return "private"
    if FRAME.match(name):
        return "a frame or traceback's"
    if name in REACHED:
        return "a module the allowlist leaves out"
    if IO.match(name) and name not in api_methods():
        return "reads, writes or evaluates"
    return None


def root(node: ast.expr) -> str | None:
    """The name an attribute chain starts from (`a.b.c` -> `a`), None for a call's result."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def allowed(module: str, imports: tuple[str, ...]) -> bool:
    """`module` (or the object `from module import`s: `numpy.pi`) is one `imports` allows."""
    if module.startswith("."):
        return False
    if any(module == a or module.startswith(a + ".") for a in imports):
        return True
    return module.split(".")[0] in STDLIB


def runs(file: str, tree: ast.AST) -> list[Finding]:
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [p for name in imported(node) for p in name.split(".")]
        for name in set(names) & RUNS:
            message = f"uses {name}: the caller runs the motions act returns"
            found.append(Finding(file, node.lineno, "runs", message))
    return found


def positions(file: str, tree: ast.AST) -> list[Finding]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            values = [float(node.value)]
        elif isinstance(node, ast.BinOp):
            values = folded(node) or []
        else:
            continue
        for value in values:
            if abs(value) > TOLERANCE and abs(value) != HALF and not value.is_integer():
                found.add(Finding(file, node.lineno, "position", f"literal {value:g}: give it "
                                  "through params (fit)"))  # fmt: skip
    return sorted(found, key=lambda f: (f.line, f.message))


ARRAYS: Final = frozenset({"array", "asarray", "list", "tuple", "float"})
"""Calls that keep their literal arguments' values (`np.array([42, -18])`)."""
OPERATORS: Final = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Pow: lambda a, b: a**b,
}
FOLD_MAX: Final = 64
"""Values a folded expression may hold (more: not folded)."""


def folded(node: ast.AST) -> list[float] | None:
    """The numbers a constant numeric expression evaluates to (element-wise over literal
    sequences, a scalar broadcast), else None."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return [float(node.value)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = folded(node.operand)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return None if inner is None else [sign * v for v in inner]
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [folded(e) for e in node.elts]
        if all(i is not None for i in items):
            flat = [v for i in items if i is not None for v in i]
            return flat if len(flat) <= FOLD_MAX else None
        return None
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
            node.func, "id", None)  # fmt: skip
        return folded(node.args[0]) if name in ARRAYS else None
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = folded(node.left), folded(node.right)
        if left is None or right is None or (len(left) != len(right) and 1 not in (
                len(left), len(right))):  # fmt: skip
            return None
        n = max(len(left), len(right))
        a, b = (left * n if len(left) == 1 else left), (right * n if len(right) == 1 else right)
        try:
            values = [float(OPERATORS[type(node.op)](x, y)) for x, y in zip(a, b, strict=True)]
        except (ZeroDivisionError, OverflowError):
            return None
        return values if all(math.isfinite(v) for v in values) else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dirs", type=Path, nargs="+", help="written skills' directories")
    parser.add_argument("--vocab", type=Path, required=True, help="`cover_demos.py --vocab`'s")
    parser.add_argument("--json", action="store_true", help="findings as JSON lines")
    args = parser.parse_args()
    vocabulary = Vocabulary.load(args.vocab)
    any_found = False
    for d in args.dirs:
        for finding in check(d.resolve(), vocabulary):
            any_found = True
            print(json.dumps({"dir": str(d)} | asdict(finding)) if args.json else f"{d}/{finding}")
    sys.exit(1 if any_found else 0)


if __name__ == "__main__":
    main()
