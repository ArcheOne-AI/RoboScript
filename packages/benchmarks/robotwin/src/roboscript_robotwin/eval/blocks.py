"""P3 blocks as a scheduler sees them (`robotwin_pipeline`):
the settings of a unit's official evaluation (`P3Settings`: blocks, episodes, modes, shards,
profiles), whether its blocks are done (`block_done`, `p3_pending`), from each block's
`result.json` (`launch.P3Result`), and what a launch does with them: which are open
(`open_blocks`), what a dry run says of each (`block_plan`), closing them from set-aside runs
(`close_blocks`), the profile each runs with (`block_profile`), and discarding a mode's blocks to
run them again (`redo_p3`)."""

import contextlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.ids import TaskName
from roboscript.learning import set_aside
from roboscript.orchestrate.jobs import RunKey
from roboscript.settings import repo_root
from roboscript_robotwin.eval.launch import (
    PROFILE,
    Mode,
    P3Result,
    P3Run,
    closable,
    close,
    finished,
    resumable,
)

log = logging.getLogger(__name__)

SPARE: Final = "spare"
"""The lane of a unit's randomized P3 blocks under `p3.spare_parallel`."""
LANE_DIRS: Final = {SPARE: "P3/randomized"}
"""Where a lane keeps its records (`pipeline.json`, `worker.log`, ...) in its unit's directory:
beside the blocks it owns, so that its logs are its own."""
P3_PLACES: Final = 2
"""The simulator places a P3 block holds on its GPU per shard: RoboTwin's policy server and
client."""
MANIFEST: Final = "P3/manifest.json"
_BLOCK: Final = re.compile(r"block\d+")


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


class P3Settings(BaseModel):
    """`config`: the launcher's (`roboscript_robotwin.eval.launch.P3Config`, copy
    examples/simBenchmarks/RoboTwin/configs/p3.example.yaml), whose GPU, motion service, vision GPU
    and Agent host Memory store each job sets; `blocks`: the official seed blocks to evaluate (block
    n from seed 100000 * (n + 1); none: no P3); `episodes` per block, at
    most the official 100 (fewer stop the client early, and then the host's summaries count), and
    `episodes_randomized` per randomized block (None: `episodes`; `episodes_of`). A block cut on
    purpose counts every episode it finished, so n may exceed the count asked for; a later launch
    asking for fewer episodes than a set-aside run of the block finished closes the block from that
    run without running anything (`close_blocks`), and one asking for more resumes it from the
    run that finished the most; `randomized`: also evaluate the clean-learned Memory on
    randomized scenes (C2R; a later launch can add it with `--p3-mode randomized`, which leaves
    S1 and S2 as they are): `True` right after the variant's clean blocks, in the same job;
    `"spare"` as a unit of its own at the lowest priority, placed only while no other job waits
    for GPU room (`robotwin_pipeline.schedule`: every clean block goes first); `spare_parallel`
    (with `"spare"` only): that unit is a job of its own (`<variant>/P3/randomized/`,
    `roboscript.orchestrate.jobs.RunKey.lane`) that may run while the variant's clean blocks run,
    once its Memory froze, instead of after them; `shards`: server and client pairs per block at
    once (`roboscript_robotwin.eval.launch.P3Run.shards`: the same seeds and official rate, in
    about 1/`shards` of the time while the gateway keeps up), holding `P3_PLACES` places each;
    `strict_shards: false` merges shards whose expert checks disagree on the solvable seeds (a
    rough estimate) instead of failing the block; `steal_shards` (with `shards` > 1): the shards
    take the block's episodes as they come free instead of by index modulo `shards`
    (`P3Run.steal_shards`).

    `actor_brain_differs`: P3's Actor brain (the launcher's `host.brain`, in `config`) may
    differ from `learning.brains.actor`, which S2 practised with; off, a launch refuses that
    (`schedule.preflight`).

    Each block records the profile it runs with (`profile.json`); a resumed block keeps the one
    it started with."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config: Path = Path("examples/simBenchmarks/RoboTwin/configs/p3.yaml")
    blocks: list[NonNegativeInt] = [1]
    episodes: int = Field(default=100, ge=1, le=100)
    episodes_randomized: int | None = Field(default=None, ge=1, le=100)
    randomized: bool | Literal["spare"] = False
    spare_parallel: bool = False
    shards: PositiveInt = 1
    strict_shards: bool = True
    steal_shards: bool = False
    actor_brain_differs: bool = False

    @field_validator("config")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return _absolute(path)

    @model_validator(mode="after")
    def _spare_parallel(self) -> Self:
        if self.spare_parallel and self.randomized != "spare":
            raise ValueError("p3.spare_parallel needs p3.randomized: spare")
        return self

    @field_validator("blocks")
    @classmethod
    def _once(cls, blocks: list[int]) -> list[int]:
        if len(set(blocks)) < len(blocks):
            raise ValueError(f"blocks lists a block twice: {blocks}")
        return blocks

    @property
    def modes(self) -> list[Literal["clean", "randomized"]]:
        return ["clean", "randomized"] if self.randomized else ["clean"]

    def episodes_of(self, mode: str) -> int:
        """The episodes per block of `mode`."""
        if mode == "randomized" and self.episodes_randomized is not None:
            return self.episodes_randomized
        return self.episodes

    @property
    def spare(self) -> bool:
        """Whether randomized blocks wait for spare GPU room (`randomized: spare`)."""
        return self.randomized == "spare"

    @property
    def parallel(self) -> bool:
        """Whether the spare unit is a job of its own beside the clean blocks
        (`spare_parallel`)."""
        return self.spare and self.spare_parallel


def block_result(out: Path) -> P3Result | None:
    path = out / "result.json"
    return P3Result.model_validate_json(path.read_bytes()) if path.is_file() else None


def block_done(out: Path, episodes: int | None = None) -> bool:
    """Whether a P3 block's run counts: no configuration error, no episode whose last attempt
    infrastructure lost (`P3Result.outcomes`: a resumed run evaluates it again; nor
    outcomes it could not read, `outcomes_error`, once episodes ended), and its
    client either exited 0 with the official rate written (`_result.txt`) or was cut on purpose,
    after at least `episodes` ended episodes when that is given (`p3.episodes`: a launch that asks
    for more runs a block cut shorter again)."""
    result = block_result(out)
    if result is None or result.config_error is not None:
        return False
    if lost_outcomes(result):
        return False
    if result.outcomes_error is not None and any(
        e.get("official") is not None for e in result.episodes
    ):  # episodes ended, but which of them infrastructure lost is unknown
        return False
    if result.truncated:
        ended = sum(e.get("official") is not None for e in result.episodes)
        return episodes is None or ended >= episodes
    return result.client_exit == 0 and result.success_rate is not None


def lost_outcomes(result: P3Result) -> list[dict[str, Any]]:
    """The episodes of a block's run whose last attempt infrastructure lost (`outcomes`)."""
    return [outcome for outcome in result.outcomes if outcome.get("lost")]


def owned_modes(p3: P3Settings, key: RunKey) -> list[Mode] | None:
    """The P3 modes whose blocks a unit's job runs (None: every mode): under `spare_parallel`
    its spare lane's the randomized ones, the unit's own job the rest."""
    if not p3.parallel:
        return None
    return ["randomized"] if key.lane == SPARE else ["clean"]


def p3_pending(p3: P3Settings, run_dir: Path, modes: Sequence[Mode] | None = None) -> bool:
    """Whether a block of `modes` (the settings' by default) of the unit in `run_dir` is not done
    yet."""
    return any(
        not block_done(run_dir / "P3" / mode / f"block{block}", p3.episodes_of(mode))
        for mode in (modes or p3.modes)
        for block in p3.blocks
    )


def carry_source(out: Path, *, open_too: bool = False) -> tuple[Path, int] | None:
    """The set-aside run block `out` would resume from (`resumable`) and the episodes a block
    closed from it would count (`closable`); with `open_too`, `out` itself is a candidate when it
    is open (not done: a launch sets it aside first, `robotwin_pipeline.schedule.prune`), as a
    dry run sees it."""
    found = []
    if (source := resumable(out)) is not None:
        found.append((len(finished(source)), source))
    if open_too and out.is_dir() and not block_done(out):
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            found.append((len(finished(out)), out))
    if not found or (best := max(found, key=lambda f: f[0]))[0] == 0:
        return None
    return best[1], closable(best[1])


def block_plan(p3: P3Settings, run_dir: Path, modes: Sequence[Mode]) -> dict[str, str]:
    """What a launch would do with each P3 block of `modes` (read only, `--dry-run`): done
    (complete, or cut on purpose at its count), closed from a set-aside run that finished its
    count (`close_blocks`), resumed from one that did not, or started."""
    plan = {}
    for mode in modes:
        wanted = p3.episodes_of(mode)
        for block in p3.blocks:
            out = run_dir / "P3" / mode / f"block{block}"
            name = f"{mode}/block{block}"
            result = block_result(out)
            if block_done(out, wanted) and result is not None:
                ended = sum(e.get("official") is not None for e in result.episodes)
                plan[name] = f"done ({ended} ended)" if result.truncated else "done (complete)"
                continue
            found = carry_source(out, open_too=True)
            if found is None:
                plan[name] = f"start ({wanted} episodes)"
            elif found[1] >= wanted:
                plan[name] = f"close at {wanted}: {found[1]} finished in {found[0].name}"
            else:
                plan[name] = f"resume to {wanted}: {found[1]} finished in {found[0].name}"
    return plan


async def close_blocks(
    p3: P3Settings, run_dir: Path, task: TaskName, modes: Sequence[Mode]
) -> list[str]:
    """Close each block of `modes` that is not done at its count (`P3Settings.episodes_of`) and
    whose set-aside run to resume from finished at least that many episodes, from that run and
    without running anything (`p3.launch.close`): every episode it finished counts. Its profile
    is the run's. The blocks closed."""
    manifest = run_dir / MANIFEST
    closed = []
    for mode in modes:
        wanted = p3.episodes_of(mode)
        for block in p3.blocks:
            out = run_dir / "P3" / mode / f"block{block}"
            if block_done(out, wanted) or not manifest.is_file():
                continue
            found = carry_source(out)
            if found is None or found[1] < wanted:
                continue
            set_aside(out)  # an empty or open directory left behind
            profile = block_profile(found[0])
            run = P3Run(
                task=task,
                seed=block,
                mode=mode,
                manifest=manifest,
                out_dir=out,
                max_episodes=wanted,
                shards=p3.shards,
                strict_shards=p3.strict_shards,
                profile=profile,
            )
            close(run, found[0])
            closed.append(f"{mode}/block{block}")
    return closed


def open_blocks(run_dir: Path, modes: Sequence[Mode] | None = None) -> list[Path]:
    """The P3 blocks of a job (of `modes`; None: every mode) that are not done (cut short,
    failed, or running)."""
    blocks = run_dir.glob("P3/*/block*")
    return sorted(
        out
        for out in blocks
        if _BLOCK.fullmatch(out.name)
        and (modes is None or out.parent.name in modes)
        and not block_done(out)
    )


def block_profile(resume: Path | None) -> dict[str, Any]:
    """The profile a P3 block runs with: the one the run it resumes from (`resume`) recorded,
    else current."""
    if resume is not None and (resume / PROFILE).is_file():
        kept = json.loads((resume / PROFILE).read_bytes())
        return {**kept, "kept_from": resume.name}
    return {"profile": "current"}


REDONE: Final = "redone"
"""The suffix (`<name>-redone<n>`) of a P3 block directory `redo_p3` moved aside: no carry reads
it (`p3.launch.resumable` takes `<block>-aborted<n>` alone, the pipeline `block<n>`)."""


BLOCK_RUN: Final = re.compile(r"block\d+(-aborted\d+)?")


def redo_p3(directory: Path, task: TaskName, variant: str, mode: str) -> list[str]:
    """Discard a variant's P3 blocks of `mode` so they run again from scratch: each block
    directory and its set-aside runs (`block<n>`, `block<n>-aborted<k>`) is moved aside as
    `<name>-redone<m>` (kept), under a name no resume carries episodes from; a launch of P3
    (`--p3-mode`, or the variant's own P3) then finds the blocks pending. The names moved, as
    they were."""
    folder = directory / task / variant / "P3" / mode
    moved = []
    for path in sorted(folder.glob("block*")) if folder.is_dir() else []:
        if not BLOCK_RUN.fullmatch(path.name):
            continue
        n = 1
        while (aside := path.with_name(f"{path.name}-{REDONE}{n}")).exists():
            n += 1
        path.rename(aside)
        moved.append(path.name)
    if moved:
        log.warning("%s/%s: P3 %s blocks moved aside to run again: %s", task, variant, mode, moved)
    return moved


def redo_targets(found: Sequence[str]) -> list[tuple[TaskName, Mode]]:
    """`--redo-p3` values `TASK:MODE` as pairs (ValueError naming one that is not)."""
    pairs = []
    for given in found:
        task, sep, mode = given.partition(":")
        if not sep or mode not in ("clean", "randomized"):
            raise ValueError(f"--redo-p3 {given}: give TASK:clean or TASK:randomized")
        pairs.append((TaskName(task), cast(Mode, mode)))
    return pairs
