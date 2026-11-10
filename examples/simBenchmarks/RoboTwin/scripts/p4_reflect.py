"""Run one P4 reflection round offline on a copy of a pipeline run's P3 records
(docs/concepts/learning.md).

Per task (every task with `<task>/default/P3/<mode>/` under `--snapshot`, or `--tasks`), the first
`--episodes` official episodes of block `--block` (`roboscript_robotwin.eval.seed_of`: official
index and seed from the official clients' records), keyed `block<N>/<task>/ep<index>`; the view
holds their D parts and the official success. Several blocks: repeat `--block`, and `--snapshot`
once (every block from that run) or once per block (in the same order, block i from snapshot i);
the round reads them all and its document is for the block after the last one. The reflection
(`roboscript_methods.rsi.reflect`: reflectors per task, the curator, the reviewer; prompts and
rules from `--recipe`) runs through the model gateway (`RS_GATEWAY_URL` or configs/gateway.yaml,
`RS_GATEWAY_TOKEN`) with `--model` / `--effort`, or with scripted brains (`--scripted`: each role
lists, reads and finishes, for tests). It writes the round's directory `--out` (which must be new
or empty): inputs/, reflect/, edits.jsonl, G<round>.md, G<round>.json, reflection.json, gaps.md,
the store the document is frozen in (`rsi/`), the images the roles saw (`blobs/`) and the events.
Keyframe images come from each `<snapshot>/blobs`: the run stops before any model call when one is
missing or they lack a picked episode's frame observation, unless `--allow-no-images` (then the
roles see no keyframes of the episodes that lack them). `--demos <dir>` gives each task's
demonstration package `<dir>/<task>/package` (a missing one stops the run); the roles see only
its final keyframe's RGB images (`package.json` and its blobs, never its truth layer).
`--resume <dir>` makes the run a second reflection pass over the same round: it starts from the
document the first pass in `<dir>` wrote (entries keep their ids and provenance), the edits that
pass applied counting against the round's budgets (`rsi.reflect.resume_round`). With
`--consolidate-only` it runs only the round's consolidation step over that document (merges and
retirements, reviewed, free of the budget).

    uv run python examples/simBenchmarks/RoboTwin/scripts/p4_reflect.py \\
        --snapshot /path/to/block1-clean --mode clean --block 1 --round 1 --out /path/to/r1 \\
        [--demos /path/to/demos] [--tasks adjust_bottle,beat_block_hammer] \\
        (--model <model> [--context-window N] [--effort high] | --scripted)
"""

import argparse
import json
import sys
from pathlib import Path

import anyio

from roboscript.agent import ScriptedBrain, call
from roboscript.agent.session import Brain
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.ids import BlobId, TaskName
from roboscript.gateway import openai_client
from roboscript.hub import Hub
from roboscript.learning.config import BrainSpec
from roboscript.learning.demo import load_package
from roboscript.learning.eval_view import (
    EvalEpisode,
    EvalRecordView,
    episode_frames,
    read_eval_episode,
)
from roboscript.learning.runner import brain_maker
from roboscript.learning.seeds import SeedRange, SeedSplit
from roboscript.records import EventWriter
from roboscript.tools import LocalProvider
from roboscript_methods.rsi.doc import EpisodeKey
from roboscript_methods.rsi.lint import LintContext
from roboscript_methods.rsi.method import CURATOR, REVIEWER, Rsi, RsiConfig
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_methods.rsi.reflect import Reflection, reflect, resume_round
from roboscript_methods.rsi.store import FileRsiStore
from roboscript_methods.rsi.toolset import RsiTools
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import LINT_OVERLAYS
from roboscript_robotwin.eval.seed_of import P3_DIR, block_episodes


def picked(
    read: list[tuple[Path, int]], tasks: list[str], mode: str, episodes: int
) -> tuple[list[EvalEpisode], list[EpisodeKey]]:
    """The first `episodes` official episodes of each task's blocks (each from its snapshot),
    read as the view reads them, and their keys."""
    found, keys = [], []
    for task, (snapshot, block) in ((task, source) for task in tasks for source in read):
        for entry in block_episodes(snapshot, task, mode, block, episodes):
            episode = read_eval_episode(entry.directory, entry.seed)
            if episode is None:
                raise ValueError(f"{entry.directory}: picked, but lost to the infrastructure")
            found.append(episode)
            keys.append(EpisodeKey(block=f"block{block}", task=TaskName(task), episode=entry.index))
    return found, keys


class SourceBlobs(LocalBlobStore):
    """The snapshots' blob stores read as one: a blob is read from the first root that holds it
    (content-addressed, so any root holding it holds the same bytes). Read-only."""

    def __init__(self, roots: list[Path]) -> None:
        super().__init__(roots[0])
        self.roots = roots

    def path(self, ref: BlobRef | BlobId) -> Path:
        blob_id = ref.id if isinstance(ref, BlobRef) else ref
        paths = [root / blob_id[:2] / blob_id for root in self.roots]
        return next((path for path in paths if path.is_file()), paths[0])

    def _write(self, ref: BlobRef, data: bytes) -> None:
        raise TypeError("the snapshots' blobs are read-only")


def sources(snapshots: list[Path], blocks: list[int]) -> list[tuple[Path, int]]:
    """Each block with the snapshot it is read from; ValueError when they do not pair up."""
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"--block names a block twice: {blocks}")
    if len(snapshots) == 1:
        return [(snapshots[0], block) for block in blocks]
    if len(snapshots) != len(blocks):
        raise ValueError(
            f"{len(snapshots)} --snapshot for {len(blocks)} --block: give one, or one per block"
        )
    return list(zip(snapshots, blocks, strict=True))


def round_split(blocks: list[int]) -> tuple[SeedSplit, str]:
    """The split of a round reading the official `blocks`, and its evaluation block: the one
    after the last of them."""
    blocks = sorted(blocks)
    evaluated = blocks[-1] + 1
    split = SeedSplit(
        blocks={
            f"block{block}": SeedRange(start=start, count=count)
            for block in (*blocks, evaluated)
            for start, count in (ROBOTWIN.eval_seeds.block(block),)
        },
        reflects={f"block{evaluated}": tuple(f"block{block}" for block in blocks)},
    )
    return split, f"block{evaluated}"


def lacking_images(episodes: list[EvalEpisode], source: LocalBlobStore) -> list[str]:
    """Each picked episode whose frames' observations the store lacks (or that took no frames),
    with how many of its frames it lacks."""
    found = []
    for episode in episodes:
        assert episode.directory is not None  # read from a directory
        frames = episode_frames(episode.directory)
        lacked = sum(not source.exists(frame.observation) for frame in frames)
        if not frames or lacked:
            found.append(f"{episode.directory}: {lacked} of {len(frames)} frames")
    return found


def demo_packages(demos: Path, tasks: list[str]) -> dict[TaskName, Path]:
    """Each task's demonstration package directory under `demos`; ValueError for a task without
    one, or whose package is another task's."""
    found = {}
    for task in tasks:
        package = demos / task / "package"
        demo = load_package(package)
        if demo.task != task:
            raise ValueError(f"{package} holds a demonstration of {demo.task}, not {task}")
        found[TaskName(task)] = package
    return found


def scripted(role: str, label: str, keys: list[EpisodeKey]) -> Brain:
    """A role that lists its task's episodes (a reflector), reads one digest and finishes."""
    if role in (CURATOR, REVIEWER):
        return ScriptedBrain([[call("rsi_doc")], [call("finish", note=f"{role}: nothing to do")]])
    first = next(key for key in keys if key.task == label)
    return ScriptedBrain(
        [
            [call("rsi_episodes", task=label)],
            [call("rsi_episode", key=first.label(), turns="1-2")],
            [call("finish", note="scripted: nothing proposed. Gaps: none named")],
        ]
    )


async def main_async(args: argparse.Namespace) -> int:
    try:
        read = sources([s.resolve() for s in args.snapshot], args.block or [1])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    snapshots = list(dict.fromkeys(snapshot for snapshot, _ in read))
    if args.tasks:
        tasks = sorted(args.tasks.split(","))
    else:
        tasks = sorted(
            {p.name for s in snapshots for p in s.iterdir() if (p / P3_DIR / args.mode).is_dir()}
        )
    unknown = sorted(set(tasks) - set(ROBOTWIN.tasks))
    if unknown:
        print(f"not RoboTwin tasks: {unknown}", file=sys.stderr)
        return 2
    episodes, keys = await anyio.to_thread.run_sync(picked, read, tasks, args.mode, args.episodes)
    view = EvalRecordView.of(episodes)
    split, evaluated_block = round_split([block for _, block in read])
    recipe = load_rsi_recipe(args.recipe, args.lint_overlay)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty: a rerun goes to a new directory", file=sys.stderr)
        return 2
    roots = [s / "blobs" for s in snapshots if (s / "blobs").is_dir()]
    blobs_dir = ", ".join(str(s / "blobs") for s in snapshots)
    source = SourceBlobs(roots) if roots else None
    if len(roots) < len(snapshots) and not args.allow_no_images:
        print(
            f"not every one of {blobs_dir} is there: pass --allow-no-images to reflect without "
            "keyframes",
            file=sys.stderr,
        )
        return 2
    if source is not None:
        lacking = await anyio.to_thread.run_sync(lacking_images, episodes, source)
        if lacking and not args.allow_no_images:
            shown = "\n".join(lacking[:20]) + (
                f"\n... {len(lacking) - 20} more" if len(lacking) > 20 else ""
            )
            print(
                f"{blobs_dir} lacks the frames of {len(lacking)} episodes:\n{shown}",
                file=sys.stderr,
            )
            return 2
    demos = {}
    if args.demos is not None:
        try:
            demos = await anyio.to_thread.run_sync(demo_packages, args.demos.resolve(), tasks)
        except (OSError, ValueError) as error:
            print(f"--demos: {error}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    if args.scripted:

        def brain(role: str, label: str) -> Brain:
            return scripted(role, label, keys)
    else:
        config = {
            "model": args.model,
            "reasoning_effort": args.effort,
            "parallel_tool_calls": True,
            "history": {"tool_images": args.frames_shown},
        }
        if args.context_window is not None:
            config["context"] = {"windows": {args.model: args.context_window}}
        make = brain_maker(BrainSpec(name="responses", config=config))

        def brain(role: str, label: str) -> Brain:
            return make()

    start, spent = None, None
    if args.consolidate_only and args.resume is None:
        print("--consolidate-only consolidates a round's document: give --resume", file=sys.stderr)
        return 2
    if args.resume is not None:
        try:
            start, spent = resume_round(args.resume.resolve(), args.round)
        except (OSError, ValueError, KeyError) as error:
            print(f"--resume: {error}", file=sys.stderr)
            return 2
    tools = RsiTools()
    own = LocalBlobStore(out / "blobs")
    with EventWriter(out / "events.jsonl", "p4_reflect") as events:
        hub = Hub([LocalProvider(tools)], blobs=own, phases=Rsi.phases, events=events)
        reflection = Reflection(
            hub=hub,
            tools=tools,
            brain=brain,
            blobs=own,
            store=FileRsiStore(out / "rsi"),
            directory=out,
            recipe=recipe,
            context=LintContext(vocabulary=recipe.lint, tasks=sorted(ROBOTWIN.tasks)),
            source=source,
            demos=demos,
            models=None if args.scripted else openai_client,
            events=events,
            concurrency=args.concurrency,
        )
        async with hub:
            reflected = await reflect(
                reflection,
                RsiConfig(evaluation=evaluated_block),
                view,
                split,
                round=args.round,
                keys=keys,
                start=start,
                spent=spent,
                consolidate_only=args.consolidate_only,
            )
    record = json.loads((out / "reflection.json").read_text(encoding="utf-8"))
    shown = ("proposals", "consolidation", "usage", "skipped", "frames_viewed")
    summary = {k: record[k] for k in shown if k in record}
    print(json.dumps({"rsi": reflected.ref.model_dump(mode="json"), **summary}, indent=2))
    if source is None:
        print(f"no {blobs_dir}: the round had no keyframe images", file=sys.stderr)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--snapshot",
        type=Path,
        action="append",
        required=True,
        help="a pipeline run's copy; repeat it to pair one with each --block",
    )
    parser.add_argument("--mode", default="clean", choices=["clean", "randomized"])
    parser.add_argument(
        "--block", type=int, action="append", help="a block to read (repeatable; default 1)"
    )
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--recipe", type=Path, default=Path("recipes/rsi"))
    parser.add_argument("--lint-overlay", type=Path, default=Path(LINT_OVERLAYS["rsi"]))
    parser.add_argument("--model", help="the reflection roles' model (required unless --scripted)")
    parser.add_argument(
        "--context-window", type=int, help="the model's context window in tokens (default: 128k)"
    )
    parser.add_argument("--effort", default="high")
    parser.add_argument("--concurrency", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--frames-shown", type=int, default=12)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--demos", type=Path, default=None)
    parser.add_argument("--allow-no-images", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="a first reflection pass's run directory over the same round: start from its "
        "document, its applied edits counted against the round's budgets",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="with --resume: only the consolidation step over that document",
    )
    args = parser.parse_args()
    if args.model is None and not args.scripted:
        parser.error("--model is required unless --scripted")
    sys.exit(anyio.run(main_async, args))


if __name__ == "__main__":
    main()
