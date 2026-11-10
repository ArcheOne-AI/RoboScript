"""A P3 job's blocks (`robotwin_pipeline`): the launcher's
config for a job on its GPU (`p3_config`, under a profile's overrides: `with_profile`), and its
blocks of each mode evaluated in order, resumed and closed as their records say
(`evaluate_blocks`)."""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from roboscript.contracts.provenance import Provenance
from roboscript.kernel import compose
from roboscript.learning import set_aside
from roboscript.orchestrate.graph import one_line
from roboscript.orchestrate.jobs import RunKey
from roboscript_robotwin.eval.blocks import (
    P3Settings,
    block_done,
    block_profile,
    block_result,
    close_blocks,
    lost_outcomes,
)
from roboscript_robotwin.eval.launch import (
    Mode,
    P3Config,
    P3Run,
    evaluate,
    resumable,
    scene_names,
)

log = logging.getLogger(__name__)

P3_STARTS: Final = 3
"""Launches of a P3 block that fails before its official client starts (a service or the policy
server that did not come up)."""


class InfraFailure(RuntimeError):
    """Infrastructure failed a P3 block (its client or policy server), apart from a configuration
    error."""


def p3_config(
    base: P3Config,
    run_dir: Path,
    gpu: int,
    motion: str,
    vision: str | None,
    blobs: Path,
    demo: Mapping[str, Any] | None = None,
) -> P3Config:
    """The launcher's config for a job: its GPU (added to the GPUs lent); the Agent host
    reading the job's Memory store and showing the demonstration as `demo` sets (`demo_host`)
    and, when it mounts the manipulation or scene toolsets, planning with the GPU's motion
    service at `motion`; when it mounts the scene toolsets and the GPU has a vision service at
    `vision`, cross-checking with that one, in the pipeline's blob store `blobs` (else a vision
    service the base config starts goes on the job's GPU)."""
    values = base.model_dump(mode="json")
    store = {"name": "file", "options": {"root": str(run_dir)}}
    host = {**base.host, "memory": store, **(demo or {})}
    if demo and "episodes" in demo:  # the practice's episode settings over the base's
        host["episodes"] = {**base.host.get("episodes", {}), **demo["episodes"]}
    gpus = base.gpus if gpu in base.gpus else [*base.gpus, gpu]
    values |= {"gpu": gpu, "gpus": gpus, "host": host}
    mounted = {spec.name for spec in base.toolsets()}
    scene = scene_names(mounted)
    if "manipulation" in mounted or scene:
        values["motion"] = {"url": motion}
    if scene and vision is not None:
        values["vision"] = {"url": vision, "blob_dir": str(blobs)}
    elif base.vision is not None and base.vision.config is not None:
        values["vision_gpu"] = gpu
    return P3Config.model_validate(values)


def with_profile(launcher: P3Config, overrides: Mapping[str, Any]) -> P3Config:
    """`launcher` whose Agent host's toolsets have `overrides` merged into their configs
    (by toolset name, `scene` for the scene toolsets alike: a profile a resumed block kept), and
    `host` into the Agent host's own config (its episodes' rules, the identification's retries)."""
    mounted = {spec.name for spec in launcher.toolsets()}
    if unknown := sorted(set(overrides) - mounted - {"scene", HOST}):
        raise ValueError(f"the P3 profile names toolsets the Agent host does not mount: {unknown}")
    values = launcher.model_dump(mode="json")
    if HOST in overrides:
        kept = {"toolsets": values["host"].get("toolsets", [])}
        values["host"] = {**compose(values["host"], overrides[HOST]), **kept}
    toolsets = []
    scene = scene_names(mounted)
    for spec in values["host"].get("toolsets", []):
        patches = [overrides.get(spec["name"], {})]
        if spec["name"] in scene:
            patches.insert(0, overrides.get("scene", {}))
        toolsets.append({**spec, "config": compose(spec.get("config", {}), *patches)})
    values["host"] = {**values["host"], "toolsets": toolsets}
    return P3Config.model_validate(values)


HOST: Final = "host"
"""The P3 profile's key for the Agent host's own config (not a toolset's)."""


async def evaluate_blocks(
    p3: P3Settings,
    key: RunKey,
    run_dir: Path,
    manifest: Path,
    launcher: P3Config,
    modes: Sequence[Mode],
    provenance_of: Callable[[dict[str, Any]], Provenance | None],
    pins: Path | None = None,
) -> None:
    """P3 of the frozen Memory (`manifest`) through `launcher`, each of `modes`' blocks in order,
    each mode's up to its count (`P3Settings.episodes_of`); those done are skipped. A block an
    earlier launch left unfinished is set aside and runs on from the episodes it finished
    (`launch.resumable`): the same seeds, only the rest evaluated; one whose set-aside run
    finished its count already is closed from it, nothing run (`close_blocks`). Each block's
    episodes record the provenance `provenance_of` gives for its profile (`block_profile`).
    `pins`: a directory of verdicts files shared with other runs of the same blocks,
    `<task>/<mode>-block<N>.jsonl` (`P3Run.pins`; None: each block pins its own). The job's last
    block to run gives back each shard's room as its client ends (`P3Run.release_shards`); a
    rerun of its lost episodes after that would start shards in room given away, so the job
    fails as infrastructure instead and runs again on a placement of its own."""
    order = [(mode, block) for mode in modes for block in p3.blocks]
    for mode in modes:
        wanted = p3.episodes_of(mode)
        cut = None if wanted == 100 else wanted
        await close_blocks(p3, run_dir, key.task, [mode])  # their set-aside runs suffice
        for block in p3.blocks:
            out = run_dir / "P3" / mode / f"block{block}"
            if block_done(out, wanted):
                continue
            later = order[order.index((mode, block)) + 1 :]
            last = all(
                block_done(run_dir / "P3" / m / f"block{b}", p3.episodes_of(m)) for m, b in later
            )
            evaluation = P3Run(
                task=key.task,
                seed=block,
                mode=mode,
                manifest=manifest,
                out_dir=out,
                max_episodes=cut,
                shards=p3.shards,
                strict_shards=p3.strict_shards,
                steal_shards=p3.steal_shards,
                pins=None if pins is None else pins / key.task / f"{mode}-block{block}.jsonl",
                release_shards=last and p3.shards > 1,
            )
            start = 1
            while True:  # starts that fail, up to P3_STARTS; reruns of lost episodes, bounded
                set_aside(out)  # an interrupted run's finished episodes carry on (`resumable`)
                carried = resumable(out)
                profile = block_profile(carried)
                # its episodes' provenance: the job's, under the block's profile
                provenance = provenance_of(profile)
                update = {"resume": carried, "profile": profile, "provenance": provenance}
                resume = evaluation.model_copy(update=update)
                overrides = profile.get("overrides") or {}
                try:
                    await evaluate(
                        resume, with_profile(launcher, overrides) if overrides else launcher
                    )
                except (RuntimeError, TimeoutError) as error:
                    failed, started = block_result(out), _client_started(out)
                    set_aside(out)  # no open block, no stale pid record
                    if failed is not None and failed.config_error is not None:
                        raise
                    if not started and start < P3_STARTS:
                        log.warning("%s: P3 did not start (%s); again", key.name, one_line(error))
                        start += 1
                        continue
                    raise InfraFailure(f"P3 {mode} block {block}: {one_line(error)}") from error
                result = block_result(out)
                if result is not None and (lost := lost_outcomes(result)):
                    # no task outcome; the block resumes, evaluating only these again
                    # (`launch.resume_plan`, which refuses one lost `MAX_LOSSES` runs in a row)
                    if evaluation.release_shards:
                        set_aside(out)
                        raise InfraFailure(
                            f"P3 {mode} block {block} lost {lost} after its ended shards' GPU "
                            "room went back: runs again on a placement of its own"
                        )
                    log.warning("%s: P3 %s block %d lost %s; again", key.name, mode, block, lost)
                    continue
                if not block_done(out, wanted):
                    set_aside(out)  # a launch again resumes from its finished episodes
                    if result is not None and result.truncated:  # its clients stopped short
                        ended = sum(e.get("official") is not None for e in result.episodes)
                        raise InfraFailure(
                            f"P3 {mode} block {block}: the official clients ended with {ended} "
                            f"of {wanted} episodes"
                        )
                    raise InfraFailure(
                        f"P3 {mode} block {block}: the official client exited 0 without its "
                        "result (_result.txt)"
                    )
                break


def _client_started(out: Path) -> bool:
    """Whether a P3 launch got as far as an official client (`client`, `client.shard<k>` in its
    `pids.json`)."""
    pids = out / "pids.json"
    names = json.loads(pids.read_bytes()) if pids.is_file() else {}
    return any(name.split(".")[0] == "client" for name in names)
