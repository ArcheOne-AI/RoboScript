"""What a golden context snapshot is, for the RoboTwin snapshot test (`test_context_snapshots`).

What a model sees changes only on purpose. For every stage x tier x role (and Actor
variant) an episode is built from, a snapshot captures

- the tool list exactly as the Responses request carries it (`tools`: order, name, description,
  parameter schema, `strict`), with each tool's descriptor hash (the hash permits refer to), and
  the system-only tools the runtime may call in the episode (`system_tools`, no model sees them);
- each role prompt (the request's `instructions`) rendered for fixed tasks;
- `contexts.json`: for each production episode, the sha256 of the canonical `{instructions,
  tools}` pair, which is what `agent/recording.py` logs in a `context` line, so a real run's
  line can be compared with it.

Everything goes through the production paths: the packaged Hub catalog and `Hub.open_episode`
for the tools, `to_responses_tool` (the loop's export), the recipe's `Prompts` for the
instructions. Nothing here starts a simulator, uses a GPU or the network.

Goldens (`snapshots/<stack>/`) are keyed by stage, tier, role and task only: no paths, no module
names, no timestamps. A change to what a model sees fails with a diff; once it is intended,
regenerate with

    RS_UPDATE_SNAPSHOTS=1 uv run pytest tests/benchmarks/robotwin/test_context_snapshots.py

and commit the new files with the change. A normal run never writes them.

Moving code: only the stack builders in the test modules name where things live; the goldens must
stay byte for byte the same.
"""

import difflib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from roboscript.agent.recording import digest
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import Tier
from roboscript.hub import Hub
from roboscript.learning import LearningConfig, ProtocolRunner
from roboscript.learning.method import Declaration
from roboscript.memory import Editor, MemoryDoc, open_cache
from roboscript.settings import repo_root
from roboscript.tools import ToolProvider, to_responses_tool
from roboscript_methods.wicl.method import Wicl

UPDATE: Final = "RS_UPDATE_SNAPSHOTS"
RECIPE: Final = repo_root() / "recipes" / "wicl"
S2_BUDGET: Final = 4
S2_SEED: Final = 10004


@dataclass(frozen=True)
class SnapshotTask:
    """`name`: the benchmark's task id (the Memory header's); `shape` what kind of task it is
    (for the reader); `instruction`: the live instruction the Actor gets; `memory`: the Memory
    its prompts show."""

    name: str
    shape: str
    instruction: str
    memory: MemoryDoc


@dataclass(frozen=True)
class Variant:
    """An episode kind whose tool list is captured: `stage`, `tier`, `role`, its name
    (`actor_hinted` and `actor_unhinted` are the S1 Actor's two toolset profiles) and `skip`,
    toolsets its profile leaves out (None: all mounted)."""

    stage: str
    tier: Tier
    role: str
    name: str
    skip: tuple[str, ...] | None = None

    @property
    def key(self) -> str:
        return f"{self.stage}-{self.tier}-{self.name}"


@dataclass(frozen=True)
class Prompt:
    """A rendered `instructions` kind, run on `variant`'s tool list in the production episode
    (`stage`, `tier` as the protocols default them)."""

    name: str
    variant: str  # a Variant.name
    stage: str
    tier: Tier
    render: Callable[[SnapshotTask], str]


@dataclass(frozen=True)
class Stack:
    """What a snapshot is taken of: the mounted `providers` and the method's `phases`, the
    `variants` whose tools are captured, the `prompts` rendered for `tasks`."""

    name: str
    providers: Sequence[ToolProvider]
    phases: Sequence[PhaseSpec]
    blobs: LocalBlobStore
    variants: Sequence[Variant]
    prompts: Sequence[Prompt]
    tasks: Sequence[SnapshotTask]


# The episode kinds production opens: imitate.py (S1: the hinted and unhinted Actor, the Demo
# Reader, the Reviser for cleanup, revisions and the criteria), practice.py (S2: the Actor, the
# Reviser), agent_host.py (P3: the Actor). `dev` is for tests only; its Actor is captured too.
PIPELINE_VARIANTS: Final = (
    *(
        Variant("s1", tier, role, name, skip)
        for tier in ("P", "D")
        for role, name, skip in (
            ("actor", "actor_hinted", None),
            ("actor", "actor_unhinted", ("demo",)),
            ("demo_reader", "demo_reader", None),
            ("reviser", "reviser", None),
        )
    ),
    Variant("s2", "D", "actor", "actor"),
    Variant("s2", "D", "reviser", "reviser"),
    Variant("dev", "D", "actor", "actor"),
    Variant("dev", "P", "actor", "actor"),
)


def pipeline_prompts(runner: ProtocolRunner) -> list[Prompt]:
    """The prompts of the S1 and S2 episodes, rendered as `imitate.py` and `practice.py` do."""
    prompts, profile, recipe = runner.prompts, runner.profile, runner.memory

    def actor(rules: str) -> Callable[[SnapshotTask], str]:
        return lambda t: prompts.actor(
            instruction=t.instruction, memory=t.memory, rules=rules, profile=profile
        )

    def s1_editor(task: SnapshotTask) -> Editor:
        return Editor(task.memory, rules=recipe.rules("s1"), lint=recipe.lint)

    def s2_editor(task: SnapshotTask) -> Editor:
        return open_cache(
            task.memory,
            budget=S2_BUDGET,
            seed=S2_SEED,
            rules=recipe.rules("s2"),
            lint=recipe.lint,
        )

    def draft(task: SnapshotTask) -> MemoryDoc:
        return MemoryDoc(task=task.memory.task, instruction=task.instruction)

    return [
        Prompt("actor_s1_hinted", "actor_hinted", "s1", "P", actor("s1_hinted")),
        Prompt("actor_s1", "actor_unhinted", "s1", "P", actor("s1")),
        Prompt(
            "demo_reader",
            "demo_reader",
            "s1",
            "P",
            lambda t: prompts.demo_reader(
                instruction=t.instruction, memory=draft(t), profile=profile
            ),
        ),
        Prompt(
            "reviser_s1",
            "reviser",
            "s1",
            "P",
            lambda t: prompts.reviser(
                instruction=t.instruction, editor=s1_editor(t), profile=profile
            ),
        ),
        Prompt(
            "criteria",
            "reviser",
            "s1",
            "P",
            lambda t: prompts.criteria(
                instruction=t.instruction, editor=s1_editor(t), profile=profile
            ),
        ),
        Prompt(
            "cleanup",
            "reviser",
            "s1",
            "P",
            lambda t: prompts.cleanup(
                instruction=t.instruction, editor=s1_editor(t), profile=profile
            ),
        ),
        Prompt("actor_s2", "actor", "s2", "D", actor("s2")),
        Prompt(
            "reviser_s2",
            "reviser",
            "s2",
            "D",
            lambda t: prompts.reviser(
                instruction=t.instruction, editor=s2_editor(t), profile=profile
            ),
        ),
    ]


def _no_brain(episode: Any) -> Any:
    raise AssertionError("no episode runs here")


def build_runner(
    tmp: Path,
    providers: Sequence[ToolProvider],
    embodiment: str | Path,
    method: type[Declaration] = Wicl,
) -> tuple[ProtocolRunner, LocalBlobStore]:
    blobs = LocalBlobStore(tmp / "blobs")
    runner = ProtocolRunner(
        LearningConfig(run_dir=tmp / "run", method="wicl", recipe=RECIPE),
        providers=providers,
        blobs=blobs,
        controls={},
        embodiment=embodiment,
        brains={"actor": _no_brain},
        method=method,
    )
    return runner, blobs


def _json(value: Any) -> str:
    """Exact JSON, key order kept (the order the request has), one trailing newline."""
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


async def capture(stack: Stack) -> dict[str, str]:
    """The stack's golden files by path relative to its snapshot directory."""
    files: dict[str, str] = {}
    exported: dict[str, list[dict[str, Any]]] = {}
    async with Hub(stack.providers, blobs=stack.blobs, phases=stack.phases) as hub:
        for variant in stack.variants:
            mounted = [provider.name for provider in stack.providers]
            profile = (
                None if variant.skip is None else [n for n in mounted if n not in variant.skip]
            )
            session = await hub.open_episode(
                OpenEpisode(
                    stage=variant.stage, tier=variant.tier, role=variant.role, toolsets=profile
                )
            )
            await hub.close_episode(session.episode_id)
            tools = [to_responses_tool(descriptor) for descriptor in session.tools]
            exported[variant.key] = tools
            hashes = {descriptor.name: descriptor.hash for descriptor in session.tools}
            files[f"tools/{variant.key}.json"] = _json(
                {
                    "stage": variant.stage,
                    "tier": variant.tier,
                    "role": variant.role,
                    "variant": variant.name,
                    "tools_sha256": digest(tools),
                    "tool_hashes": hashes,
                    "tools": tools,
                    "system_tools": list(session.system_tools),
                }
            )
    contexts: dict[str, dict[str, str]] = {}
    for prompt in stack.prompts:
        variant = next(
            v
            for v in stack.variants
            if v.name == prompt.variant and v.stage == prompt.stage and v.tier == prompt.tier
        )
        tools = exported[variant.key]
        for task in stack.tasks:
            text = prompt.render(task)
            where = f"{prompt.stage}-{prompt.tier}-{prompt.name}__{task.name}"
            files[f"instructions/{where}.txt"] = text
            contexts[where] = {
                "tools": f"tools/{variant.key}.json",
                "tools_sha256": digest(tools),
                "instructions_sha256": digest(text),
                "context_sha256": digest({"instructions": text, "tools": tools}),
            }
    files["contexts.json"] = _json(contexts)
    return files


def _diff(name: str, want: str, got: str) -> str:
    lines = list(
        difflib.unified_diff(
            want.splitlines(), got.splitlines(), f"golden/{name}", f"now/{name}", lineterm="", n=1
        )
    )
    shown = lines[:40]
    more = len(lines) - len(shown)
    return "\n".join(shown + ([f"... {more} more diff lines"] if more > 0 else []))


def compare(snapshots: Path, stack: str, files: Mapping[str, str]) -> None:
    """The stack's files against `snapshots/<stack>/`; `RS_UPDATE_SNAPSHOTS=1` writes them there."""
    directory = snapshots / stack
    if os.environ.get(UPDATE) == "1":
        for stale in sorted(directory.rglob("*")) if directory.exists() else []:
            if stale.is_file() and stale.relative_to(directory).as_posix() not in files:
                stale.unlink()
        for name, text in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        return
    problems: list[str] = []
    for name, got in files.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing golden: {stack}/{name}")
        elif (want := path.read_bytes().decode("utf-8")) != got:
            problems.append(f"changed: {stack}/{name}\n{_diff(name, want, got)}")
    if directory.exists():
        for path in sorted(directory.rglob("*")):
            name = path.relative_to(directory).as_posix()
            if path.is_file() and name not in files:
                problems.append(f"stale golden (nothing builds it now): {stack}/{name}")
    if problems:
        shown = "\n\n".join(problems)
        raise AssertionError(
            f"what the model sees changed ({len(problems)} files). If intended, regenerate with "
            f"{UPDATE}=1 and commit the goldens:\n\n{shown}"
        )
