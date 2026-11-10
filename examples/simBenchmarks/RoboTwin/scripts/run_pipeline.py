"""Run the S1 -> S2 -> P3 pipeline (robotwin_pipeline): each task's S1 imitate, then each
variant's S2 practice (which freezes) and P3's official evaluation, for every task x variant of
configs/pipeline.yaml (copy configs/pipeline.example.yaml; layered over its env binding,
RoboTwin's examples/simBenchmarks/RoboTwin/run/wicl/ by default), in parallel on the GPUs it
names, each job in a worker process of its own (this script with --worker). The model gateway is the
persistent one (configs/gateway.yaml, RS_GATEWAY_TOKEN); the toolsets are `run_s2.toolsets()`'s:
sensing, manipulation (configs/manipulation.yaml, planning with the job's GPU's motion service)
and the scene toolsets (configs/scene.yaml if present, with that GPU's motion and vision
services).

    mkdir -p <root> && PYTHONPATH=examples/simBenchmarks/RoboTwin nohup .venv/bin/python \\
        examples/simBenchmarks/RoboTwin/scripts/run_pipeline.py > <root>/launch.log 2>&1 & \\
        echo $! > <root>/pipeline.pid

    ... --resume <root>/<stamp> [--retry-failed]        continue a pipeline directory with its own
                                                        pipeline.yaml (--set may not change a
                                                        job that started): every job that has
                                                        not ended, afresh; failed jobs too
                                                        with --retry-failed (those that failed
                                                        on infrastructure are taken up anyway)
    ... --resume <root>/<stamp> --redo-p3 <task>:randomized --p3-mode randomized
                                                        discard a task's P3 blocks of a mode
                                                        (block<n> and block<n>-aborted<k> moved
                                                        aside as -redone<m>, never carried) and
                                                        run them again from scratch
    ... --resume <root>/<stamp> --s1-only               run S1s only: no variant's S2 or P3 starts
    ... --resume <root>/<stamp> --dry-run [...]         print what a launch with these options
                                                        would do, job by job; write nothing
    ... --resume <root>/<stamp> --p3-mode randomized
                                                        C2R: P3 of the variants that froze, in
                                                        randomized scenes (nothing else runs)
    ... --allow-dirty                                  launch code of a dirty tree (a source
                                                        label ending in -dirty: refused
                                                        otherwise); the launch records it
    ... --summary <root>/<stamp>                        write its summary now; start nothing
    ... --set max_runs=8 --set p3.episodes=10           change config values (dotted keys)
    ... --set p3.unconverged=stop                       a variant whose S2 stopped at the seed
                                                        cap without converging ends not_converged
                                                        (default best_memory: P3 on its best
                                                        first-try version, none: counted 0; last:
                                                        P3 on the cap's; a new directory only)

<root> is `root` of the pipeline config, and each launch writes <root>/<stamp>/. Run from the
repository root in the agent venv, with ROBOTWIN_ROOT and RS_ENV_TOKEN in the environment or .env.
It logs to <stamp>/pipeline.log as well. Under nohup a dropped ssh session leaves it running;
SIGTERM (`kill -TERM $(cat <root>/pipeline.pid)`; each launch's pid is
also in <stamp>/launches.jsonl) stops every process it started and leaves each job resumable.
Exit status 0 when every job ended done, not_passed or not_converged
(examples/simBenchmarks/RoboTwin/README.md, "The pipeline").
"""

import argparse
import functools
import logging
import sys
from pathlib import Path

import anyio
import run_s2
from robotwin_pipeline import (
    GatewayCheck,
    RunEnv,
    Wiring,
    compose_config,
    load_directory,
    new_directory,
    plan,
    plan_text,
    run_pipeline,
    worker,
    write_summary,
)
from robotwin_pipeline.run import check_dirty, check_source

from roboscript.gateway import openai_client
from roboscript.gateway.client import gateway_url
from roboscript.settings import EnvSettings, require_source_sha, runs_path
from roboscript.stack import Stopped
from roboscript.tools import ToolProvider
from roboscript_robotwin.eval.blocks import redo_targets

log = logging.getLogger("run_pipeline")


def wiring(root: Path | None = None) -> Wiring:
    """The jobs' toolsets as `run_s2.toolsets` mounts them, with the demo toolset (S1 jobs' Demo
    Reader and hinted attempts see it; the catalog hides it from every other stage), with
    `run_s2.manipulation_config` (configs under `root`, the repository's by default) planning
    with the job's GPU's motion service and the scene toolsets using that GPU's services; the
    gateway's model clients; this script as the worker."""

    def toolsets(env: RunEnv) -> list[ToolProvider]:
        manipulation = run_s2.manipulation_config(env.motion, root=root)
        return run_s2.toolsets(manipulation, env.services, demo=True, vlm_model=env.vlm_model)

    here = str(Path(__file__).resolve())
    return Wiring(
        toolsets=toolsets, worker=[sys.executable, here, "--worker"], models=openai_client
    )


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    if given[:1] == ["--worker"]:
        return worker.main(wiring(), given[1:])
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="pipeline", help="a config name or YAML file")
    ap.add_argument("--resume", type=Path, help="continue this pipeline directory")
    ap.add_argument("--retry-failed", action="store_true", help="with --resume: failed jobs too")
    ap.add_argument(
        "--p3-mode",
        action="append",
        choices=["clean", "randomized"],
        help="with --resume: only P3 of the variants that froze, in this mode",
    )
    ap.add_argument(
        "--redo-p3",
        action="append",
        default=[],
        metavar="TASK:MODE",
        help="with --resume: discard this task's P3 blocks of MODE (moved aside as -redone<n>, "
        "never carried) so they run again from scratch",
    )
    ap.add_argument("--s1-only", action="store_true", help="run S1s only (no S2 or P3)")
    ap.add_argument(
        "--dry-run", action="store_true", help="with --resume: print the launch's plan only"
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="launch code whose source label ends in -dirty (recorded in launches.jsonl)",
    )
    ap.add_argument("--summary", type=Path, help="write this pipeline directory's summary")
    ap.add_argument("--set", action="append", default=[], help="a config override key=value")
    args = ap.parse_args(given)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.summary is not None:
        found = write_summary(args.summary, load_directory(args.summary))
        print(f"{args.summary}/summary.md: {found['runs']}")
        return 0
    resuming = (args.retry_failed, args.p3_mode, args.redo_p3, args.dry_run)
    if any(resuming) and args.resume is None:
        ap.error(
            "--retry-failed, --redo-p3, --dry-run and --p3-mode continue a pipeline: add --resume"
        )
    if args.dry_run:
        directory = runs_path(args.resume).resolve()
        found = plan(
            load_directory(directory, args.set),
            directory,
            retry_failed=args.retry_failed,
            s1_only=args.s1_only,
            redo_p3=redo_targets(args.redo_p3),
            p3_modes=args.p3_mode,
        )
        print(plan_text(found))
        return 0
    try:  # every record names the code that ran, and that code is a commit's
        dirty = check_dirty(require_source_sha(), args.allow_dirty)
    except ValueError as error:
        ap.error(str(error))
    token = EnvSettings().gateway_token
    if token is None:
        ap.error("RS_GATEWAY_TOKEN is not set (the environment or .env)")
    if args.resume is not None:
        directory = runs_path(args.resume).resolve()
        try:  # before anything is written: run_pipeline checks again
            check_source(directory)
        except ValueError as error:
            ap.error(str(error))
        config = load_directory(directory, args.set)
    else:
        config = compose_config(args.config, args.set)
        directory = new_directory(config)
    # the S1 and S2 scene toolsets' VLM (run_s2.toolsets; the run's, `vlm_model`, if it names one)
    vlm = config.vlm_model or run_s2.scene_config().vlm.model
    gateway = GatewayCheck(url=gateway_url(), token=token, models=frozenset({vlm}))
    handler = logging.FileHandler(directory / "pipeline.log")
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(handler)
    log.info(
        "pipeline %s: %d tasks x %d variants", directory, len(config.tasks), len(config.variants)
    )
    launch = functools.partial(
        run_pipeline,
        config,
        wiring(),
        directory,
        retry_failed=args.retry_failed,
        gateway=gateway,
        note={"argv": given, **dirty},
        p3_modes=args.p3_mode,
        s1_only=args.s1_only,
        redo_p3=redo_targets(args.redo_p3),
    )
    try:
        found = anyio.run(launch)
    except Stopped as stopped:
        log.warning("%s: resume with --resume %s", stopped, directory)
        return 1
    log.info("done: %s; %s/summary.md", found["runs"], directory)
    ended = {"done", "not_passed", "not_converged"} | (
        {"queued", "blocked", "stopped"} if args.s1_only else set()
    )
    return 0 if set(found["runs"]) <= ended else 1


if __name__ == "__main__":
    raise SystemExit(main())
