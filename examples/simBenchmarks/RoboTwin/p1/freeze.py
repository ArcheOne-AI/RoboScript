"""Freeze a finished RoboTwin build (`build.py`'s run) into a skillset: the skills its gate
accepted (`<run>/accepted/`, where their code stays) as version `--version` of skillset `--name`
in the skillset store at `--store` (`skill_builder.freeze.freeze_run`). The skills mount beside
RoboTwin's binding data (`roboscript_robotwin.skills.bindings_only`), whose bindings they may use.
A smoke run is a wiring check and is never frozen; neither is a run whose jobs ran without the
sandbox (`freeze_run` refuses it). Each skill's `SPEC` is read in the sandbox, without network.

    uv run python freeze.py --run <runs>/<run> --name robotwin --version v1 --store <store>
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from roboscript.contracts.artifact import ArtifactRef
from roboscript.skills.store import SkillSetStore
from roboscript_methods.skill_builder.builder.sandbox import wrap
from roboscript_methods.skill_builder.builder.static_check import Vocabulary
from roboscript_methods.skill_builder.freeze import ACCEPTED, FreezeRefused, freeze_run
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.skills.bindings_only import BINDINGS_ONLY

VOCABULARY = "vocabulary.json"
"""The static checks' vocabulary a build writes into its run (`build.RoboTwin.vocab`)."""


def frozen_run(run: Path) -> dict[str, Any]:
    """`run.json` of a build run that is no smoke run, or SystemExit saying why it is not frozen
    (the sandbox is `freeze_run`'s to check)."""
    meta = run / "run.json"
    if not meta.is_file():
        raise SystemExit(f"{run}: no run.json (not a build run)")
    record = json.loads(meta.read_text(encoding="utf-8"))
    builder = record.get("builder")
    if not isinstance(builder, dict):
        raise SystemExit(f"{run}: run.json records no build (`builder`): not a build.py run")
    if builder.get("smoke", True):
        raise SystemExit(f"{run}: a smoke run is a wiring check, never frozen")
    if not (run / VOCABULARY).is_file():
        raise SystemExit(f"{run}: no {VOCABULARY} (the build writes it at its gate)")
    return dict(record)


async def freeze(run: Path, name: str, version: str, store: Path) -> ArtifactRef:
    """The skillset frozen from `run`'s accepted skills, in the store at `store`: its ref."""
    profile = ROBOTWIN.profile()
    reads = [d for d in [run / ACCEPTED] if d.is_dir()]  # none: `freeze_run` says so
    argv = wrap([sys.executable], write=[], read=reads, env={"LANG": "C.UTF-8"}, network=False)
    return await freeze_run(
        run,
        name=name,
        version=version,
        vocabulary=Vocabulary.load(run / VOCABULARY),
        profile=profile,
        store=SkillSetStore(store),
        argv=argv,
        envs=["robotwin"],
        embodiments=[profile.name],
        provided=BINDINGS_ONLY.manifest.bindings.provided(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="a finished build.py run")
    parser.add_argument("--name", required=True, help="the frozen skillset's name")
    parser.add_argument("--version", required=True, help="its version (a file name)")
    parser.add_argument("--store", type=Path, required=True, help="the skillset store's root")
    args = parser.parse_args(argv)
    run = args.run.absolute()
    frozen_run(run)
    try:
        ref = asyncio.run(freeze(run, args.name, args.version, args.store.absolute()))
    except FreezeRefused as e:
        raise SystemExit(f"not frozen: {e}") from None
    print(f"frozen: {ref.name} {ref.version} {ref.hash}")


if __name__ == "__main__":
    main()
