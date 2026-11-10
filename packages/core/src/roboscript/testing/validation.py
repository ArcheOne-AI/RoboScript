"""Skillset validation, the four steps of docs/how-to/add-a-skillset.md, as one call (`validate`)
that returns the report a manifest carries (`contracts.skills.VerificationReport`):

1. interface: the manifest is valid (its model) and every skill fits the target robot with the
   bindings mounted with it (`skills.check_interface`);
2. information boundary: its implementation imports no control-plane or privileged module and its
   tools are tier D (`testing.boundary.check_boundary`);
3. reachable ceiling: a scripted planner that may read the ground truth (`CeilingPlanner`) calls
   only the skillset's tools over some seeds; the share of episodes the env judges a success is
   an upper bound on what an Agent reaches with them (`ceiling`). Run on MockEnv here; a
   benchmark's own ceiling is the benchmark's to run;
4. freeze: once the first three pass, the manifest, with their report, is committed to a
   `skills.store.SkillSetStore` and frozen there.

A builder's `verify` phase calls `validate` (`learning.builder`). The planner is privileged by
design: the ceiling bounds what the tools can do, not what perception gives an Agent.

`validate_registered` runs them on a skillset by its registered name (or import path) with the
core's toolsets as its hosts, and so does the command (exit status 0 when every step that ran
passed):

    python -m roboscript.testing.validation SKILLSET --embodiment NAME_OR_YAML
        [--also SKILLSET ...] [--forbidden MODULE ...]
        [--ceiling package.module:factory --store DIR]

`--also`: the skillsets mounted with it (their bindings count); `--forbidden`: an env's own
privileged modules; `--ceiling`: a function of no arguments that returns the `CeilingRun`, and
`--store`: the `SkillSetStore` directory the skillset is frozen in once the ceiling passes.
Without `--ceiling`, steps 1 and 2 run and nothing is frozen.
"""

import argparse
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import anyio

from roboscript.blobs import BlobStore, LocalBlobStore
from roboscript.contracts.env import EnvAgent, EnvControl, EpisodeSpec, GtSnapshot
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import TaskName
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.skills import Bindings, SkillSet, StepResult, VerificationReport
from roboscript.contracts.tools import EnvBinding, ToolCall
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.hub import Hub
from roboscript.hub.builtins import FINISH
from roboscript.kernel import entrypoints
from roboscript.skills import SkillSetStore, check_interface
from roboscript.testing.boundary import check_boundary
from roboscript.tools import Toolset
from roboscript.tools.provider import ToolProvider
from roboscript.tools.skillsets import SkillSetPlugin, load_skillset

TOOLSETS: Final = "roboscript.toolsets"
CEILING = "ceiling"
"""The phase the ceiling's episodes run in: the planner (role `actor`) sees the skillset's tools
only."""


@dataclass(frozen=True)
class CeilingEpisode:
    """One ceiling episode as the planner sees it: the ground truth at the start, the tools it
    may call (the skillset's) and `call(tool, **arguments)`, through the Hub."""

    truth: GtSnapshot
    tools: tuple[str, ...]
    call: Callable[..., Awaitable[CallOutcome]]


class CeilingPlanner(Protocol):
    """Drives one episode to the task's goal with the skillset's tools, reading the truth."""

    async def __call__(self, episode: CeilingEpisode) -> None: ...


@dataclass(frozen=True)
class CeilingRun:
    """What the ceiling runs: `providers()` builds the stack's providers with the skillset
    mounted (fresh for each run); `env`, the env's control plane, and `env_client`, how the
    Hub's toolsets reach its agent plane; `task` and `seeds`; the `planner`; `min_ratio`, the
    success share below which the step fails (0: it reports, never fails)."""

    providers: Callable[[], Sequence[ToolProvider]]
    env: EnvControl
    env_client: Callable[[EnvBinding], EnvAgent]
    task: TaskName
    seeds: Sequence[int]
    planner: CeilingPlanner
    embodiment: str
    min_ratio: float = 0.0
    blobs: BlobStore | None = None


@dataclass(frozen=True)
class ValidationTarget:
    """The skillset (`ref`, registered name or import path), the robot it is for, the
    skillsets mounted with it (`also`, their bindings count), the core toolsets by name
    (`hosts`, a core skillset's implementation), an env's privileged modules beyond the core's
    (`forbidden`) and the ceiling to run (None: the step is skipped, and the skillset is not
    frozen)."""

    ref: str
    profile: EmbodimentProfile
    also: Sequence[str] = ()
    hosts: Mapping[str, type[Toolset]] = field(default_factory=dict)
    forbidden: Sequence[str] = ()
    ceiling: CeilingRun | None = None


async def ceiling(plugin: SkillSetPlugin, run: CeilingRun) -> StepResult:
    """The share of `run.seeds` whose episode the env judges a success when `run.planner`
    drives the skillset's tools (its manifest's skills) and nothing else."""
    skills = {skill.name for skill in plugin.manifest.skills}
    providers = list(run.providers())
    entries = [f"{p.name}.{s.name}" for p in providers for s in p.specs() if s.name in skills]
    if missing := sorted(skills - {entry.partition(".")[2] for entry in entries}):
        return StepResult(passed=False, detail=f"not mounted: {missing}")
    shown = {"actor": [*entries, FINISH]}  # a phase lists finish; the planner never calls it
    phase = PhaseSpec.model_validate(
        {"name": CEILING, "kind": "learn", "visibility": shown, "tiers": {"actor": "P"}}
    )
    with tempfile.TemporaryDirectory() as directory:
        blobs = run.blobs or LocalBlobStore(Path(directory))
        async with Hub(providers, blobs=blobs, phases=[phase], env_client=run.env_client) as hub:
            successes = 0
            for seed in run.seeds:
                successes += await _episode(hub, run, seed)
    ratio = successes / len(run.seeds)
    detail = f"{successes} of {len(run.seeds)} seeds of {run.task}"
    return StepResult(passed=ratio >= run.min_ratio, detail=detail, value=ratio)


async def _episode(hub: Hub, run: CeilingRun, seed: int) -> bool:
    opened = await run.env.open(EpisodeSpec(task=run.task, seed=seed, stage=CEILING))
    try:
        binding = EnvBinding(
            url="inproc://ceiling", episode_id=opened.episode_id, token=opened.agent_token
        )
        spec = OpenEpisode(
            stage=CEILING, tier="P", env=binding, embodiment=run.embodiment, structured=True
        )
        session = await hub.open_episode(spec)
        calls = 0

        async def call(tool: str, **arguments: Any) -> CallOutcome:
            nonlocal calls
            calls += 1
            made = ToolCall(call_id=f"c{calls}", name=tool, arguments=arguments, origin="model")
            return await hub.call(session.episode_id, made)

        try:
            tools = tuple(t.name for t in session.tools if t.name != FINISH.partition(".")[2])
            truth = await run.env.gt_snapshot(opened.episode_id)
            await run.planner(CeilingEpisode(truth=truth, tools=tools, call=call))
        finally:  # a planner that raises still leaves the Hub with no episode open
            await hub.close_episode(session.episode_id)
        return (await run.env.judge(opened.episode_id)).success
    finally:
        await run.env.close(opened.episode_id)


async def validate(target: ValidationTarget, store: SkillSetStore) -> VerificationReport:
    """The four steps (see the module); each runs only if those before it passed. The frozen
    manifest carries the report of the first three; the returned report has all four."""
    plugin = load_skillset(target.ref)
    bindings = Bindings()
    for ref in dict.fromkeys([*target.also, target.ref]):
        bindings = bindings.merged(load_skillset(ref).manifest.bindings)
    steps: dict[str, StepResult] = {}
    interface = check_interface(plugin.manifest, target.profile, bindings.provided())
    steps["interface"] = _lines(interface)
    if steps["interface"].passed:
        steps["boundary"] = _lines(check_boundary(plugin, target.hosts, target.forbidden))
    if target.ceiling is not None and all(step.passed for step in steps.values()):
        steps["ceiling"] = await ceiling(plugin, target.ceiling)
    if len(steps) == 3 and all(step.passed for step in steps.values()):
        steps["freeze"] = await _freeze(plugin.manifest, VerificationReport(steps=steps), store)
    return VerificationReport(steps=steps)


async def _freeze(
    manifest: SkillSet, report: VerificationReport, store: SkillSetStore
) -> StepResult:
    verified = manifest.model_copy(update={"report": report})
    try:
        ref = await store.commit(verified)
        await store.freeze(ref, reason="validated")
    except ValueError as error:
        return StepResult(passed=False, detail=str(error))
    return StepResult(passed=True, detail=f"{ref.name} {ref.version} {ref.hash}")


def _lines(lines: Sequence[str]) -> StepResult:
    return StepResult(passed=not lines, detail="; ".join(lines))


def core_hosts() -> dict[str, type[Toolset]]:
    """The core's registered toolsets by name (`ValidationTarget.hosts`): a core skillset's
    implementation is among them."""
    names = entrypoints.names_in(TOOLSETS, "roboscript")
    return {name: entrypoints.load(TOOLSETS, name) for name in names}


async def validate_registered(
    skillset: str,
    embodiment: str,
    store: SkillSetStore,
    *,
    also: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    ceiling: CeilingRun | None = None,
) -> VerificationReport:
    """`validate` on `skillset` (a registered name or an import path) for the robot `embodiment`
    names (a registered name or a YAML file), the core's toolsets its hosts (`core_hosts`)."""
    target = ValidationTarget(
        ref=skillset,
        profile=load_embodiment(embodiment),
        also=also,
        hosts=core_hosts(),
        forbidden=forbidden,
        ceiling=ceiling,
    )
    return await validate(target, store)


def _report_lines(report: VerificationReport) -> list[str]:
    """One line per step that ran: `<step>: passed` or `<step>: FAILED`, its measure and detail."""
    lines = []
    for name, step in report.steps.items():
        line = f"{name}: {'passed' if step.passed else 'FAILED'}"
        if step.value is not None:
            line += f" ({step.value:g})"
        lines.append(f"{line}: {step.detail}" if step.detail else line)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m roboscript.testing.validation",
        description="Validate a registered skillset (docs/how-to/add-a-skillset.md).",
    )
    ap.add_argument("skillset", help="a name in roboscript.skillsets, or package.module:attr")
    ap.add_argument("--embodiment", required=True, help="a registered embodiment or its YAML")
    ap.add_argument("--also", action="append", default=[], help="a skillset mounted with it")
    ap.add_argument("--forbidden", action="append", default=[], help="an env's privileged module")
    ap.add_argument("--ceiling", help="package.module:factory returning the CeilingRun")
    ap.add_argument("--store", type=Path, help="the SkillSetStore directory to freeze it in")
    args = ap.parse_args(argv)
    if args.ceiling is not None and args.store is None:
        ap.error("--ceiling freezes the skillset once it passes: give --store")
    ceiling = None
    if args.ceiling is not None:
        ceiling = entrypoints.load_ref(args.ceiling)()
        if not isinstance(ceiling, CeilingRun):
            found = type(ceiling).__name__
            ap.error(f"--ceiling {args.ceiling} returned a {found}, not a CeilingRun")

    async def run(directory: Path) -> VerificationReport:
        return await validate_registered(
            args.skillset,
            args.embodiment,
            SkillSetStore(directory),
            also=args.also,
            forbidden=args.forbidden,
            ceiling=ceiling,
        )

    if args.store is not None:
        report = anyio.run(run, args.store)
    else:  # no ceiling: nothing is frozen, the store is never written
        with tempfile.TemporaryDirectory() as directory:
            report = anyio.run(run, Path(directory))
    for line in _report_lines(report):
        print(line)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
