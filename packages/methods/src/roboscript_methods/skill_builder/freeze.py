"""Freezing a builder run's skills as a skillset (P1, the last step), through the gate's static
checks (`builder.static_check`), the interface check (`roboscript.skills.verify.check_interface`)
and the skillset store (`SkillSetStore`).

`freeze_run(run, ...)` refuses a run whose output is not sandboxed work (`sandbox.unsandboxed`:
its `run.json` records no sandbox mode or `none`, or a summary is marked `UNSANDBOXED`), then
checks every accepted skill statically, reads each one's `SPEC` in a child process started with
the given argv (a sandbox: importing a skill runs the Agent's code, never in this process), checks
the manifest against the embodiment, and commits and freezes it. The manifest records the skills'
specs, never how they were built; the skills' code stays in the run's `accepted/`.

    python -m roboscript_methods.skill_builder.freeze --specs DIR...
        (the child: a nonce on stdin first; each row is `<nonce> <json>`)
"""

import argparse
import asyncio
import json
import secrets
import sys
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any, Final

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.skills import (
    BindingName,
    Bindings,
    SkillSet,
    SkillSpec,
    StepResult,
    VerificationReport,
)
from roboscript.embodiments import EmbodimentProfile
from roboscript.skills.store import SkillSetStore
from roboscript.skills.verify import check_interface
from roboscript_methods.skill_builder.builder.plugins import load_one, skill_dirs
from roboscript_methods.skill_builder.builder.sandbox import unsandboxed
from roboscript_methods.skill_builder.builder.static_check import Vocabulary, check

ACCEPTED: Final = "accepted"
"""Where a builder run keeps the skills it accepted (`builder.driver`)."""
SPECS_S: Final = 300.0
"""A child that reads specs longer than this is stuck (an import that loops)."""


class FreezeRefused(ValueError):
    """Why a run's skills are not frozen, one line per reason."""


async def freeze_run(
    run: Path,
    *,
    name: str,
    version: str,
    vocabulary: Vocabulary,
    profile: EmbodimentProfile,
    store: SkillSetStore,
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    skills: Path | None = None,
    envs: Sequence[str] = (),
    embodiments: Sequence[str] = (),
    bindings: Bindings | None = None,
    provided: Set[BindingName] = frozenset(),
    reason: str = "built",
) -> ArtifactRef:
    """Freeze the skills `run` accepted (`skills`: another directory of skill directories) as
    version `version` of skillset `name` (layer `built`) in `store`; its ref. `argv` starts Python
    in the sandbox the specs are read in (`sandbox.wrap(..., network=False)` ending in an
    interpreter with `roboscript_methods`), `env` its environment; `vocabulary`, the task names and
    labels the static checks refuse; `profile` and `provided` (the bindings the skillsets mounted
    beside it give), the interface check; `envs`, `embodiments`, `bindings` (None: none), the
    manifest's.
    `FreezeRefused` says every reason it is not frozen."""
    if problems := unsandboxed(run):
        raise FreezeRefused(
            "a run without the sandbox is never frozen into the store; rerun it sandboxed:\n"
            + "\n".join(problems)
        )
    directory = run / ACCEPTED if skills is None else skills
    if not directory.is_dir():
        raise FreezeRefused(f"{directory}: no such directory of skills")
    dirs = skill_dirs([directory])
    if not dirs:
        raise FreezeRefused(f"{directory} holds no skill (a directory with a skill.py)")
    findings = [f"{d.name}: {f}" for d in dirs for f in check(d, vocabulary)]
    if findings:
        raise FreezeRefused("static checks:\n" + "\n".join(findings))
    specs = await read_specs(dirs, argv, env)
    bindings = bindings or Bindings()
    manifest = SkillSet(
        name=name,
        version=version,
        layer="built",
        envs=list(envs),
        embodiments=list(embodiments),
        skills=specs,
        bindings=bindings,
    )
    lines = check_interface(manifest, profile, set(provided) | bindings.provided())
    if lines:
        raise FreezeRefused("interface:\n" + "\n".join(lines))
    report = VerificationReport(steps={"interface": StepResult(passed=True)})
    try:
        ref = await store.commit(manifest.model_copy(update={"report": report}))
        return await store.freeze(ref, reason=reason)
    except ValueError as error:  # the store's refusal: another content, frozen already
        raise FreezeRefused(str(error)) from error


async def read_specs(
    dirs: Sequence[Path], argv: Sequence[str], env: Mapping[str, str] | None = None
) -> list[SkillSpec]:
    """Each skill's `SPEC`, in order, read by `python -m roboscript_methods.skill_builder.freeze
    --specs` started with `argv` (a sandbox); `FreezeRefused` lists the skills it could not load."""
    command = [*argv, "-m", "roboscript_methods.skill_builder.freeze", "--specs", *map(str, dirs)]
    nonce = secrets.token_hex(16)
    child = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=None if env is None else dict(env),
    )
    try:
        out, err = await asyncio.wait_for(child.communicate(nonce.encode() + b"\n"), SPECS_S)
    except asyncio.TimeoutError:
        child.kill()
        await child.communicate()
        raise FreezeRefused(f"reading the specs ran past {SPECS_S:.0f} s") from None
    rows: dict[str, dict[str, Any]] = {}
    for line in out.decode(errors="replace").splitlines():
        mark, _, text = line.partition(" ")
        if mark == nonce:  # rows a skill printed itself carry no nonce
            row = json.loads(text)
            rows[row["dir"]] = row
    problems, specs = [], []
    for d in dirs:
        row = rows.get(str(d))
        if row is None:
            tail = err.decode(errors="replace").strip().splitlines()[-1:]
            problems.append(f"{d.name}: not read (exit {child.returncode}) {' '.join(tail)}")
        elif row["problems"]:
            problems += [f"{d.name}: {p}" for p in row["problems"]]
        else:
            specs.append(SkillSpec.model_validate(row["spec"]))
    if problems:
        raise FreezeRefused("skills that do not load:\n" + "\n".join(problems))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="print written skills' SPECs (freeze's child)")
    parser.add_argument("--specs", type=Path, nargs="+", required=True, metavar="DIR")
    args = parser.parse_args()
    nonce = sys.stdin.readline().strip()  # before any skill is imported
    for d in args.specs:
        loaded = load_one(d)
        row: dict[str, Any] = {"dir": str(d)}
        if isinstance(loaded, list):
            row |= {"problems": loaded, "spec": None}
        else:
            row |= {"problems": [], "spec": loaded.spec.model_dump(mode="json")}
        print(nonce, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
