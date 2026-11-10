"""A written skill (`README.md`, "A written skill") as the coverage and reuse steps run it: its
directory's
`skill.py` and `fit.py` imported by path, the contract checked on what they define.

The files are the builder Agent's, a boundary: every way they break the contract
is reported at load, all of them at once, so the driver can hand the list back to the session
instead of failing a coverage run halfway.

Importing them runs them, so the driver never does it in its own process: `validate` runs `python -m
roboscript_methods.skill_builder.builder.plugins --validate DIR... --vocab FILE --json` under the
argv prefix it is given (a sandbox), which runs the static checks (`static_check.check`) and imports
a skill only when they pass. The child reads a nonce from its stdin before it imports anything and
marks each row with it; `validate` takes only marked rows, so a skill that prints a row of its own
and exits, past the static checks, passes nothing. `load_skills` is for the processes that run
skills (a case, a sweep), sandboxed themselves; it takes the vocabulary a fit's `params` are checked
against (`demo_coverage.fitting.fit_written`).

    python -m roboscript_methods.skill_builder.builder.plugins
        --validate DIR... --vocab FILE [--json]   # exit 1 unless all pass
    (--json: a nonce on stdin first; each row is `<nonce> <json>`)
"""

import argparse
import asyncio
import importlib.util
import inspect
import json
import re
import secrets
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, get_args

import yaml

from roboscript.contracts.skills import SkillSpec
from roboscript.manipulation import ActionApi
from roboscript_methods.skill_builder.builder.static_check import Vocabulary, check
from roboscript_methods.skill_builder.demo_coverage.fitting import fit_error
from roboscript_methods.skill_builder.demo_coverage.segments import Kind

FILES: Final = ("skill.py", "fit.py", "cases.yaml")
NAME: Final = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
"""`SkillSpec.name`'s own rule: the skill is a tool name."""
KINDS: Final = frozenset(get_args(Kind))
ACT_PARAMS: Final = ("api", "arm", "obj", "target", "params")
FIT_PARAMS: Final = ("segment", "recording", "scene", "base", "poses")
CORE_NAMES: Final = frozenset(n for n in dir(ActionApi) if not n.startswith("_"))
"""A written skill may not take an entry's name: a call's `skill` would no longer say which ran."""
VALIDATE_S: Final = 300.0
"""A validation that takes longer is stuck (an import that loops): the skill fails it."""


class SkillError(ValueError):
    """What is wrong with the written skills, one line per problem."""


@dataclass(frozen=True, eq=False)
class WrittenSkill:
    name: str
    kinds: tuple[str, ...]
    spec: SkillSpec
    act: Callable[..., Any]
    """`async act(api, arm, obj, target, params) -> list[Built]`"""
    fit: Callable[..., dict[str, Any] | None]
    """`fit(segment, recording, scene, base, poses) -> dict | None`"""
    cases: tuple[str, ...]
    """`<task>/<segment name>`: the gaps it was written for."""
    dir: Path
    vocabulary: Vocabulary | None = None
    """What its `params` may not name (`fitting.fit_written`); None: it cannot be fitted."""


def skill_dirs(dirs: Iterable[Path]) -> list[Path]:
    """Each given directory that holds a `skill.py`, and of one that does not, its subdirectories
    that do (the driver keeps the accepted skills side by side in one directory)."""
    found = []
    for d in dirs:
        d = Path(d).resolve()
        if (d / "skill.py").is_file():
            found.append(d)
        elif d.is_dir():
            found += sorted(p.parent for p in d.glob("*/skill.py"))
        else:
            raise SkillError(f"{d}: no such directory")
    return found


def load_skills(dirs: Iterable[Path], vocabulary: Vocabulary) -> list[WrittenSkill]:
    """Every written skill under `dirs` (`skill_dirs`), in that order, its fits' `params` checked
    against `vocabulary`; `SkillError` lists every problem found."""
    skills, problems = [], []
    for d in skill_dirs(dirs):
        found = load_one(d, vocabulary)
        if isinstance(found, WrittenSkill):
            skills.append(found)
        else:
            problems += [f"{d}: {p}" for p in found]
    seen: dict[str, Path] = {}
    for skill in skills:
        if skill.name in seen:
            problems.append(f"{skill.dir}: {skill.name!r} is also {seen[skill.name]}'s name")
        seen.setdefault(skill.name, skill.dir)
    if problems:
        raise SkillError("\n".join(problems))
    return skills


def load_one(d: Path, vocabulary: Vocabulary | None = None) -> WrittenSkill | list[str]:
    """The skill in `d` (importing it: in a sandboxed process only), or what is wrong with it."""
    missing = [f for f in FILES if not (d / f).is_file()]
    if missing:
        return [f"missing {', '.join(missing)}"]
    problems: list[str] = []
    try:
        skill = module(d, "skill")
    except Exception as error:  # the Agent's code: reported, never raised into the run
        return [f"skill.py does not import: {fit_error(error)}"]
    try:
        fitting = module(d, "fit")
    except Exception as error:
        return [f"fit.py does not import: {fit_error(error)}"]
    name = getattr(skill, "NAME", None)
    if not isinstance(name, str) or not NAME.match(name):
        problems.append(f"NAME {name!r} is not a snake_case tool name")
    elif name != d.name:
        problems.append(f"NAME {name!r} is not the directory's name {d.name!r}")
    elif name in CORE_NAMES:
        problems.append(f"NAME {name!r} is a core entry's name")
    kinds = getattr(skill, "KINDS", None)
    if not (isinstance(kinds, tuple) and kinds and set(kinds) <= KINDS):
        problems.append(f"KINDS {kinds!r} is not a non-empty tuple of {sorted(KINDS)}")
    spec = getattr(skill, "SPEC", None)
    if not isinstance(spec, SkillSpec):
        problems.append(f"SPEC is a {type(spec).__name__}, not a SkillSpec")
    elif spec.name != name:
        problems.append(f"SPEC.name {spec.name!r} is not NAME {name!r}")
    act = getattr(skill, "act", None)
    if not inspect.iscoroutinefunction(act):
        problems.append("act is not an async function")
    elif (got := parameters(act)) != ACT_PARAMS:
        problems.append(f"act takes {got}, not {ACT_PARAMS}")
    fit = getattr(fitting, "fit", None)
    if not callable(fit) or inspect.iscoroutinefunction(fit):
        problems.append("fit.py has no plain function fit")
    elif (got := parameters(fit)) != FIT_PARAMS:
        problems.append(f"fit takes {got}, not {FIT_PARAMS}")
    cases = read_cases(d / "cases.yaml", problems)
    if problems:
        return problems
    assert act is not None and fit is not None and isinstance(spec, SkillSpec)
    return WrittenSkill(str(name), tuple(kinds), spec, act, fit, cases, d, vocabulary)  # type: ignore[arg-type]


def module(d: Path, stem: str) -> ModuleType:
    """`d/<stem>.py` imported under a name of its own: two skills' `fit.py` never collide, and
    the module is in `sys.modules` (dataclasses and pydantic look their module up there)."""
    name = f"_written_skill_{d.name}_{stem}"
    spec = importlib.util.spec_from_file_location(name, d / f"{stem}.py")
    assert spec is not None and spec.loader is not None
    found = importlib.util.module_from_spec(spec)
    sys.modules[name] = found
    try:
        spec.loader.exec_module(found)
    except BaseException:
        del sys.modules[name]
        raise
    return found


def parameters(function: Callable[..., Any]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def read_cases(path: Path, problems: list[str]) -> tuple[str, ...]:
    try:
        found = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        problems.append(f"cases.yaml is not YAML: {error}")
        return ()
    gaps = found.get("gaps") if isinstance(found, dict) else None
    if not (isinstance(gaps, list) and gaps and all(isinstance(g, str) for g in gaps)):
        problems.append("cases.yaml has no list `gaps` of `<task>/<segment>` strings")
        return ()
    bad = [g for g in gaps if not case_parts(g)]
    if bad:
        problems.append(f"cases.yaml: {bad} are not `<task>/<segment>`")
    return tuple(gaps)


MODES: Final = ("chained", "stages")
"""A gap id's mode, after `#` (`case_mode`): `chained`, the segment run from where the plan's
calls before it left one rollout from the reset (`demo_coverage.run.chained_case`); `stages`, the
same, also judged by the benchmark's stages left after it."""


def case_mode(segment: str) -> tuple[str, str | None]:
    """A case's segment name and its mode (`MODES`; None: from the segment's own start)."""
    name, sep, mode = segment.rpartition("#")
    return (name, mode) if sep and name and mode in MODES else (segment, None)


def own_gap(gap: str) -> str:
    """A gap id without its mode (`case_mode`): its segment's own gap, run from its start."""
    task, _, segment = gap.partition("/")
    return f"{task}/{case_mode(segment)[0]}"


def case_parts(case: str) -> tuple[str, str] | None:
    """`<task>/<segment name>` split at its first `/` (a segment's name holds no `/`, its
    object's label may not either), or None."""
    task, _, segment = case.partition("/")
    return (task, segment) if task and segment and "/" not in segment else None


def validated(d: Path, vocabulary: Vocabulary) -> dict[str, Any]:
    """Gate step 1 on the skill in `d`, in this process: the static checks, then, only if they
    pass, the import and the contract. `ok`: both passed."""
    row: dict[str, Any] = {"dir": str(d), "name": d.name}
    row["static"] = [str(f) for f in check(d, vocabulary)]
    if row["static"]:
        return row | {"contract": [], "ok": False}
    loaded = load_one(d, vocabulary)
    if isinstance(loaded, list):
        return row | {"contract": loaded, "ok": False}
    return row | {"contract": [], "ok": True, "kinds": list(loaded.kinds),
                  "cases": list(loaded.cases)}  # fmt: skip


async def validate(
    dirs: Sequence[Path],
    vocab: Path,
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> list[dict[str, Any]]:
    """`validated` on each of `dirs` in a child process: `argv` is how to start Python there
    (`[sys.executable]`, or a sandbox's command ending in a Python), `env` and `cwd` its
    environment (`roboscript_methods` importable). One row per directory, in order; a
    child that fails or hangs (`VALIDATE_S`) fails every directory it did not report."""
    command = [*argv, "-m", "roboscript_methods.skill_builder.builder.plugins", "--validate",
               *map(str, dirs), "--vocab", str(vocab), "--json"]  # fmt: skip
    nonce = secrets.token_hex(16)
    child = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=None if env is None else dict(env),
        cwd=cwd,
    )
    try:
        out, err = await asyncio.wait_for(child.communicate(nonce.encode() + b"\n"), VALIDATE_S)
        why = f"the validator exited {child.returncode}"
    except asyncio.TimeoutError:
        child.kill()
        out, err = await child.communicate()
        why = f"the validator ran past {VALIDATE_S:.0f} s"
    rows = {}
    for line in out.decode(errors="replace").splitlines():
        mark, _, text = line.partition(" ")
        if mark == nonce:
            row = json.loads(text)
            rows[row["dir"]] = row
    tail = err.decode(errors="replace").strip().splitlines()[-1:]
    failed = [f"{why}: {fit_error(RuntimeError(tail[0]))}" if tail else why]
    unreported = {"static": [], "contract": failed, "ok": False}
    return [rows.get(str(d)) or {"dir": str(d), "name": d.name} | unreported for d in dirs]


def main() -> None:
    parser = argparse.ArgumentParser(description="validate written skills (gate step 1)")
    parser.add_argument("--validate", type=Path, nargs="+", required=True, metavar="DIR")
    parser.add_argument("--vocab", type=Path, required=True, help="`cover_demos.py --vocab`'s")
    parser.add_argument("--json", action="store_true", help="one JSON line per directory")
    args = parser.parse_args()
    nonce = sys.stdin.readline().strip() if args.json else ""  # before any skill is imported
    vocabulary = Vocabulary.load(args.vocab)
    rows = [validated(d, vocabulary) for d in args.validate]
    for row in rows:
        if args.json:
            print(nonce, json.dumps(row), flush=True)
        else:
            print(f"{row['dir']}: {'ok' if row['ok'] else 'fails'}")
            for problem in row["static"] + row["contract"]:
                print(f"  {problem}")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
