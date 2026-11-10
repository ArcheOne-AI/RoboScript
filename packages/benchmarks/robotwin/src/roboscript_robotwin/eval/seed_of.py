"""The episodes of an official P3 block as (block, seed, episode), for P4
(docs/concepts/learning.md):
what a reflection reads of a pipeline run's P3 records and which seed each episode ran.

A unit's mode keeps an official block's run as `<run>/<task>/default/P3/<mode>/block<N>` and
the runs set aside before it as `block<N>-aborted<k>` (`launch.P3Run.resume`). `block_episodes`
takes the episodes `block<N>` counts, as the launcher does: its finished episodes by official
index, seed and success (`launch.finished`: the official clients' records, `shards/shard<k>.jsonl`
or `logs/client.log`, and what its `carried.json` carried; official seeds rise from
100000 x (1 + block) and skip the seeds the expert could not solve; episodes infrastructure lost
are not finished). Set-aside runs count only through that `carried.json` chain. Each index's host
directory comes from `launch.host_indices` (the host numbers episodes per slot and never sees the
seed): in `block<N>`, or, for an episode carried before its records were brought, in the run it
was carried from. Records that disagree (the host says infrastructure lost an episode the client
finished, or a different success) are an error, never a guess.

`python -m roboscript_robotwin.eval.seed_of <run> --mode clean --block 1 --episodes 10 --out
picked.txt` writes the picked episodes' directories, relative to the run, one per line (for a
read-only snapshot's blob pass), and prints per task how many it found.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.agent.host import infra_lost
from roboscript.contracts.ids import TaskName
from roboscript_robotwin.eval.launch import finished, host_indices

P3_DIR = "default/P3"
"""A task's P3 blocks under a pipeline run: `<run>/<task>/default/P3/<mode>/`."""


class BlockEpisode(BaseModel):
    """One episode of a block: its official index and seed, the run whose host directory holds
    its records (`block<N>`, or a set-aside run it was carried from) and that directory, the
    official success, how it ended, whether infrastructure failed it (`infra`) and, when the host
    voided its end (an end after an unanswered infra error), the end it voided."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    block: NonNegativeInt
    index: NonNegativeInt
    seed: NonNegativeInt
    run: str
    directory: Path
    success: bool
    end: str | None
    infra: bool
    voided: str | None = None


def block_episodes(
    run: Path, task: str, mode: str, block: int, episodes: int | None = None
) -> list[BlockEpisode]:
    """The episodes of `task`'s official `block` in `mode` of the pipeline run `run`, by index
    (the first `episodes` indices; None: all), taken as the module says. ValueError where the
    host's records and the client's disagree."""
    path = run / task / P3_DIR / mode / f"block{block}"
    if not path.is_dir():
        return []
    official = {e["index"]: (e["seed"], bool(e["success"])) for e in finished(path)}
    directories = host_indices(path)
    got = []
    for index in sorted(official):
        if episodes is not None and index >= episodes:
            break
        seed, success = official[index]
        if index not in directories:
            raise ValueError(f"{path}: finished episode {index} has no host records")
        directory = directories[index]
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        host_success = bool((summary.get("official") or {}).get("success"))
        if infra_lost(summary):
            raise ValueError(
                f"{directory}: the host says infrastructure lost episode {index}, "
                "the client finished it"
            )
        if host_success != success:
            raise ValueError(
                f"{directory}: the host says success {host_success}, the client "
                f"{success} for episode {index}"
            )
        got.append(
            BlockEpisode(
                task=TaskName(task),
                block=block,
                index=index,
                seed=seed,
                run=directory.parent.parent.parent.name,
                directory=directory,
                success=success,
                end=(summary.get("end") or {}).get("reason"),
                infra=bool(summary.get("infra")),
                voided=summary.get("voided_end"),
            )
        )
    return got


def main() -> None:
    parser = argparse.ArgumentParser(description="the first episodes of each task's P3 block")
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", default="clean", choices=["clean", "randomized"])
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    lines, counts = [], {}
    for task_dir in sorted(p for p in run.iterdir() if (p / P3_DIR / args.mode).is_dir()):
        picked = block_episodes(run, task_dir.name, args.mode, args.block, args.episodes)
        counts[task_dir.name] = len(picked)
        lines += [str(e.directory.relative_to(run)) for e in picked]
    args.out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    short = {task: n for task, n in counts.items() if n < args.episodes}
    print(f"{len(counts)} tasks, {len(lines)} episodes, short: {short}", file=sys.stderr)


if __name__ == "__main__":
    main()
