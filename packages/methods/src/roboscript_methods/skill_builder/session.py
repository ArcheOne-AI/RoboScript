"""The builder's coding-agent sessions: the driver's `session` callable, built from the
deployment's config.

`SessionConfig` chooses the backend. `codex` runs the Codex CLI (`builder.codex.run_session`) with
the configured model, its requests going to the RoboScript gateway only (address from
`configs/gateway.yaml` or `RS_GATEWAY_URL`, token `RS_GATEWAY_TOKEN` from `.env`); never to a
provider directly. `fake` runs a scripted stand-in through the same session machinery (workspace,
`bin/case`, budget, `result.json`) with no model and no network, for tests and dry runs: it
copies the skills of `fake_skills` (if given) into the workspace, each renamed with the cluster's
name suffix, and runs `bin/case` on each gap its `cases.yaml` lists that the brief has.

    session = make_session(SessionConfig(backend="codex", model="<gateway model name>"),
                           read_roots=[...], hide=[...])
    Driver(run, bench, session=session, ...)
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from roboscript_methods.skill_builder.builder.brief import Brief
from roboscript_methods.skill_builder.builder.clusters import Cluster
from roboscript_methods.skill_builder.builder.codex import (
    Budget,
    Runtime,
    SessionResult,
    run_session,
)
from roboscript_methods.skill_builder.builder.driver import (
    CaseFn,
    Session,
    SessionContext,
    SessionOutcome,
)

Backend = Literal["codex", "fake"]
FAKE_GATEWAY: Final = "http://127.0.0.1:9/v1"
"""Written into a fake session's Codex config, never contacted (the fake starts no model)."""


class SessionConfig(BaseModel):
    """A deployment's choice of coding agent (validated where it is read: a config file or the
    command line)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend = "codex"
    model: str | None = None
    """The gateway's logical model name (`configs/gateway.yaml`, `models`); required by `codex`."""
    effort: str | None = None
    """Codex's reasoning effort (`model_reasoning_effort`); None: Codex's default."""
    command: list[str] = ["codex"]
    """How to start the Codex CLI."""
    state_root: Path | None = None
    """Where each session's `CODEX_HOME` and empty `HOME` live while it runs (removed after);
    None: `~/.cache/rs-builder`. Not under `/tmp`, where Codex refuses its sandbox helper."""
    fake_skills: Path | None = None
    """`fake` only: a directory of skill directories the fake session writes and runs."""
    sandbox_read_roots: list[Path] = []
    """Absolute directories readable in the sandboxes (the session's and every job's) besides
    the ones the benchmark's build script binds: where a benchmark's own files point outside its
    checkout (a planner config naming another checkout's assets)."""

    @model_validator(mode="after")
    def _consistent(self) -> "SessionConfig":
        if self.backend == "codex" and not self.model:
            raise ValueError("a codex session needs `model`: a model name the gateway serves")
        if self.backend != "fake" and self.fake_skills is not None:
            raise ValueError("`fake_skills` is for the fake backend only")
        if self.fake_skills is not None and not self.fake_skills.is_dir():
            raise ValueError(f"fake_skills {self.fake_skills} is not a directory")
        for root in self.sandbox_read_roots:
            if not root.is_absolute() or not root.is_dir():
                raise ValueError(f"sandbox_read_roots: {root} is not an absolute directory")
        return self


def session_runtime(
    config: SessionConfig,
    *,
    read_roots: Sequence[Path] = (),
    hide: Sequence[Path] = (),
    root: Path | None = None,
) -> Runtime:
    """The `codex.Runtime` sessions of `config` run with: `read_roots` readable and `hide` hidden
    in the Agent's sandbox (beside the repository, the interpreter and the accepted skills);
    `root`, the repository whose `configs/` and `.env` are read (default: this one). A `codex`
    backend reads the gateway's address and token now, so a missing one fails before any round."""
    knobs: dict[str, object] = {"read_roots": list(read_roots), "hide": list(hide)}
    if config.state_root is not None:
        knobs["codex_root"] = config.state_root
    if config.backend == "codex":
        assert config.model is not None  # the validator's
        return Runtime.from_gateway(
            config.model, config.effort, root=root, codex=list(config.command), **knobs
        )
    fake = [sys.executable, "-c", FAKE_AGENT]
    if config.fake_skills is not None:
        fake += ["--skills", str(config.fake_skills.resolve())]
    return Runtime(
        codex=fake,
        model=config.model or "fake",
        effort=config.effort,
        gateway_url=FAKE_GATEWAY,
        token=SecretStr("fake"),
        **knobs,  # type: ignore[arg-type]
    )


def make_session(
    config: SessionConfig,
    *,
    read_roots: Sequence[Path] = (),
    hide: Sequence[Path] = (),
    root: Path | None = None,
) -> Session:
    """The driver's `session` callable (`builder.driver.Session`) for `config`: each call renders
    the cluster's or task's brief (`build_brief`), runs one session in the workspace within the
    context's limits and returns its outcome. `read_roots`, `hide`, `root`: `session_runtime`'s."""
    runtime = session_runtime(config, read_roots=read_roots, hide=hide, root=root)

    async def session(
        found: Cluster, work: Path, case: CaseFn, accepted: list[Path], context: SessionContext
    ) -> SessionOutcome:
        brief = build_brief(found, accepted, context)
        limits = context.limits
        budget = Budget(case_runs=limits.case_runs, wall_s=limits.wall_s, tokens=limits.tokens)
        used = runtime
        if config.backend == "fake":  # the fake renames its skills as the brief asks
            used = runtime.model_copy(
                update={"codex": [*runtime.codex, "--suffix", brief.name_suffix]}
            )
        return outcome(await run_session(brief, work, budget, case, used))

    return session


def build_brief(found: Cluster, accepted: list[Path], context: SessionContext) -> Brief:
    """A session's brief: a per-task session's (`Brief.from_task`) when `context.task` is set,
    else the cluster's (`Brief.from_cluster`), with the context's relay notes and strategy."""
    if context.task is not None:
        task = context.task
        brief = Brief.from_task(found, accepted, context.reports, task.plan, task.condition,
                                name=task.task, center=task.center)  # fmt: skip
    else:
        brief = Brief.from_cluster(found, accepted, reports=context.reports,
                                   conditions=context.conditions,
                                   centers=context.centers)  # fmt: skip
    if context.relay or context.strategy:
        brief = brief.model_copy(update={"relay": context.relay, "strategy": context.strategy})
    return brief


def outcome(result: SessionResult) -> SessionOutcome:
    """The driver's view of a session's result."""
    return SessionOutcome(
        status=result.status,
        tokens=result.tokens,
        wall_s=result.wall_s,
        case_runs=result.case_runs,
        last_message=result.last_message,
        detail={
            "session_id": result.session_id,
            "transcript": str(result.transcript),
            "exit_code": result.exit_code,
            "usage_source": result.usage_source,
        },
    )


FAKE_AGENT: Final = r"""
import json, os, re, shutil, subprocess, sys
import yaml

args = sys.argv[1:]


def option(name, default=None):
    return args[args.index(name) + 1] if name in args else default


work, last = option("-C"), option("-o")
skills, suffix = option("--skills"), option("--suffix", "")
sys.stdin.read()  # the prompt: nothing reads it
os.chdir(work)


def emit(event):
    print(json.dumps(event), flush=True)


emit({"type": "thread.started", "thread_id": "fake"})
emit({"type": "turn.started"})
status = json.load(open(os.path.join(".case", "status.json"), encoding="utf-8"))
gaps, left = set(status["gaps"]), status["budget"]
notes = []
for found in sorted(os.listdir(skills)) if skills else []:
    source = os.path.join(skills, found)
    if not os.path.isfile(os.path.join(source, "skill.py")):
        continue
    name = found if found.endswith(suffix) else found + suffix
    target = os.path.join("skill", name)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    if name != found:  # NAME and SPEC's name follow the directory's
        quoted = re.compile("([\"'])" + re.escape(found) + "\\1")
        for file in ("skill.py", "fit.py"):
            path = os.path.join(target, file)
            with open(path, encoding="utf-8") as f:
                text = quoted.sub(lambda m: m.group(1) + name + m.group(1), f.read())
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
    with open(os.path.join(target, "cases.yaml"), encoding="utf-8") as f:
        listed = (yaml.safe_load(f) or {}).get("gaps") or []
    explained = 0
    for gap in [g for g in listed if g in gaps]:
        if left <= 0:
            break
        done = subprocess.run(["bin/case", name, gap], capture_output=True, text=True)
        left -= 1
        rows = done.stdout.split("\n", 1)[-1]
        try:
            explained += bool(json.loads(rows).get("row", {}).get("explained"))
        except ValueError:
            pass
    notes.append(f"{name}: explained {explained}")
text = "; ".join(notes) or "fake session: no skill written"
emit({"type": "item.completed", "item": {"id": "fake", "type": "agent_message", "text": text}})
emit({"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}})
with open(last, "w", encoding="utf-8") as f:
    f.write(text)
"""
"""The fake backend's stand-in for `codex exec`: standard library and PyYAML, the session's
interpreter."""
