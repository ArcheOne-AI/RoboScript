"""One builder session (README, "Pieces"): a Codex CLI run (`codex exec`) on one gap cluster, in a
sandbox workspace, through the model gateway, within a fixed budget, with no human in the loop.

The workspace (`workdir`, Codex's working root and its only writable directory):

    BRIEF.md        the cluster's gaps (`brief.render`) and the workspace's tools
    skill/          where the Agent writes its skill directory
    plan.json       a per-task session's task plan (`builder.plans`), its starting one
    bin/case        the Agent's only way to the simulator: a request file in `.case/`, then the
                    answer `run_session` writes after awaiting the driver's `case` callable;
                    `--submit` queues one and returns, `--wait` collects (`CASES_IN_FLIGHT`);
                    a per-task session's `--plan` (`--submit-plan`) asks for a plan trial:
                    `plan.json` and `skill/` as they are then (`<task>/#plan`)
    .case/ .tmp/    the case channel and the commands' temporary directory

Beside it, `<workdir>.codex/` (not readable from the sandbox): the transcript (`codex exec --json`
events), Codex's stderr, each case run's snapshot of the skill and its row (`cases/NN/`,
`cases.jsonl`), `pid`, `result.json`. `CODEX_HOME` and an empty `HOME` (no login can bypass the
gateway) are `Runtime.codex_root/<session id>/`, removed when the session ends.

Isolation: a Codex permissions profile (bubblewrap on Linux) lets commands write only the
workspace, read only the system, the repository, the interpreter and accepted skills (minus every
`.env*`, the repository's `configs`, `.dev`, `Runtime.hide` and the sessions' private directories:
`sandbox.masked`), and reach no network; a skill holding a symbolic link is not run. The model's
requests leave Codex itself, to the gateway only, with the key `<RS_GATEWAY_TOKEN>:<session id>`
in `CODEX_HOME/config.toml` (never in a command's environment; the file is removed when the
session ends). The model and effort are `Runtime.model` and `Runtime.effort` (the deployment's
config: `skill_builder.session.SessionConfig`); Codex's apps, browser, computer use, plugins,
hooks, image generation, web search, multi-agent and project docs are off. The commands'
`RS_ENV_FILE` is an empty file in the workspace (`.tmp/rs.env`): the repository's `.env` is
masked, unreadable (EACCES), and the settings loader reads that file instead.

Budget: case runs are counted here, each when it starts (requests past `Budget.case_runs` are
answered "budget spent"; a run the infrastructure broke, `infra` in its row, is not counted); at
most `CASES_IN_FLIGHT` of a session's runs go at once (the bench's GPU slots bound them all), the
skill snapshotted when its request is seen, so later edits do not change what runs;
the wall clock and the tokens are watched every `Runtime.poll_s`, the tokens as the larger of the
gateway's request log for the session id and Codex's own `turn.completed` usage (input, cached
included, plus output); past either, Codex and every process under it are stopped (its process
group id is in `<workdir>.codex/pid` while it runs). A stalled model stream ends as `stalled`.
A session cancelled by its caller (the driver's own wall clock) still writes `result.json`, status
`cancelled`, with the tokens and case runs it had used.
"""

import contextlib
import json
import os
import re
import shutil
import signal
import string
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final, Literal

import anyio
from anyio.abc import Process
from pydantic import BaseModel, SecretStr

from roboscript_methods.skill_builder.builder import plans
from roboscript_methods.skill_builder.builder.brief import Brief, render
from roboscript_methods.skill_builder.builder.sandbox import masked
from roboscript_methods.skill_builder.demo_coverage.fitting import RELATIVE_RULE

PROMPT: Final = Path(__file__).with_name("prompts") / "builder.md"
TASK_PROMPT: Final = Path(__file__).with_name("prompts") / "task.md"
"""A per-task session's prompt (`brief.task`)."""
TRIAL: Final = "#plan"
"""A plan trial's gap, after `<task>/` (`driver.TRIAL`)."""
DISABLED: Final = (
    "apps", "browser_use", "browser_use_external", "computer_use", "image_generation",
    "in_app_browser", "multi_agent", "plugins", "remote_plugin", "hooks", "realtime_conversation",
    "skill_mcp_dependency_install", "tool_suggest", "skill_search", "worktrees",
)  # fmt: skip
"""Codex features a builder has no use for (each a way out of the workspace or a distraction)."""
CASES_IN_FLIGHT: Final = 4
"""Case runs one session may have going at once (`bin/case --submit`, then `--wait`): a session
waits on the model most of the time, and its runs one by one leave the GPUs idle."""
ENV_FILE: Final = ".tmp/rs.env"
"""The commands' settings file, empty, in the workspace (`RS_ENV_FILE`)."""
SESSION: Final = re.compile(r"[^A-Za-z0-9._-]")
STREAM_IDLE_MS: Final = 120_000
STREAM_RETRIES: Final = 5
"""A gateway stream silent this long is dropped and retried (a stream can stall for minutes)."""
STALL_S: Final = 900.0
"""No event from Codex this long, outside case runs, is a stalled session (past Codex's own
stream retries): stopped, status `stalled`, for the driver to retry as infrastructure."""
STALLED: Final = re.compile(
    r"stream|disconnect|idle|timed? ?out|reconnect|connection|overloaded|\b(429|50[0234])\b", re.I
)
"""A Codex error that is the gateway's or the upstream's, not the session's."""


class Budget(BaseModel):
    """A session's budget (README, Budgets: the same for every cluster)."""

    case_runs: int = 12
    wall_s: float = 3 * 3600.0
    tokens: int = 5_000_000


class CaseRequest(BaseModel):
    """One case run the Agent asked for: its skill (a snapshot, so later edits do not change what
    ran) on one gap of the brief."""

    n: int
    """1-based, in the order served."""
    gap: str
    """`<task>/<segment>`."""
    task: str
    segment: str
    skill: str
    skill_dir: Path


class Tokens(BaseModel):
    input: int = 0
    cached: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output


class SessionResult(BaseModel):
    status: str
    """`done` (Codex exited 0), `timeout` (wall clock), `tokens` (token budget), `stalled` (the
    model stream stalled: Codex failed on a gateway or upstream error, or sent no event for
    `STALL_S` outside case runs; infrastructure, retry it), `error: ...`."""
    exit_code: int | None
    skills: list[Path]
    """The skill directories in `skill/` (a `skill.py` in each) when the session ended."""
    tokens: int
    usage: Tokens
    usage_source: Literal["gateway", "codex"]
    wall_s: float
    case_runs: int
    cases: list[dict[str, Any]]
    """Per request served: `n`, gap, skill, `explained`, `error`, seconds (rows in `cases/`)."""
    session_id: str
    transcript: Path
    last_message: str


class Runtime(BaseModel):
    """Deployment knobs of a session (`from_gateway` reads the gateway's address and token from the
    repository's `configs/gateway.yaml` and `.env`; `skill_builder.session` builds one from its
    config)."""

    codex: list[str] = ["codex"]
    """The command that starts Codex (`codex exec ...` is appended)."""
    model: str
    """The logical model name the gateway serves (`configs/gateway.yaml`)."""
    effort: str | None = None
    """Codex's `model_reasoning_effort`; None: Codex's default."""
    gateway_url: str
    token: SecretStr
    gateway_log: Path | None = None
    """The gateway's request log (`GatewayConfig.log_file`); None: count Codex's usage only."""
    python: Path = Path(sys.executable)
    read_roots: list[Path] = []
    """Readable beside the system, the repository (`brief.code_root`) and the accepted skills."""
    hide: list[Path] = []
    """Not readable even where inside a readable root (the runs root: other sessions' private
    directories, the gate's and the summaries' files); the session's workspace stays writable."""
    codex_root: Path = Path.home() / ".cache" / "rs-builder"
    """Where each session's `CODEX_HOME` and empty `HOME` live (`<codex_root>/<session id>/`,
    removed when it ends): outside the runs root, which the sandbox may hide, since Codex's
    sandbox helper is run from `CODEX_HOME`, and not under `/tmp`, where Codex refuses to put it."""
    poll_s: float = 1.0
    grace_s: float = 10.0

    @classmethod
    def from_gateway(
        cls, model: str, effort: str | None = None, root: Path | None = None, **knobs: Any
    ) -> "Runtime":
        """A runtime whose model requests go to the RoboScript gateway only: its address from
        `RS_GATEWAY_URL` or `configs/gateway.yaml` (under `root`, the repository's by default),
        its token `RS_GATEWAY_TOKEN` from `.env`, its request log for the token count; the base
        interpreter readable and the runs root hidden, besides what `knobs` add."""
        from roboscript.gateway.client import gateway_url
        from roboscript.gateway.config import GatewayConfig
        from roboscript.settings import EnvSettings, load_config

        settings = EnvSettings()
        if settings.gateway_token is None:
            raise ValueError("RS_GATEWAY_TOKEN is not set: add it to .env (see .env.example)")
        reads = [Path(sys.base_prefix), *knobs.pop("read_roots", [])]
        hide = [*[p for p in [settings.runs_root] if p is not None], *knobs.pop("hide", [])]
        return cls(
            model=model,
            effort=effort,
            gateway_url=gateway_url(root),
            token=settings.gateway_token,
            gateway_log=load_config("gateway", GatewayConfig, root=root).log_file,
            read_roots=reads,
            hide=hide,
            **knobs,
        )


async def run_session(
    brief: Brief,
    workdir: Path,
    budget: Budget,
    case: Callable[[CaseRequest], Awaitable[dict[str, Any]]],
    runtime: Runtime,
) -> SessionResult:
    """Run one Codex session on `brief` in `workdir` (created; must not exist or be empty, unless
    `brief.relay`: then the workspace a previous session left, continued) and serve its case
    requests with `case`, within `budget`, as `runtime` says."""
    workdir, private, session_id = await anyio.to_thread.run_sync(prepare, brief, workdir, runtime)
    state = runtime.codex_root / session_id

    channel = Channel(workdir, private, brief, budget, case)
    channel.status()
    usage = Usage(runtime.gateway_log, session_id)
    transcript = private / "transcript.jsonl"
    template = TASK_PROMPT if brief.task is not None else PROMPT
    prompt = string.Template(template.read_text(encoding="utf-8")).substitute(
        case_runs=budget.case_runs,
        at_once=CASES_IN_FLIGHT,
        wall_h=f"{budget.wall_s / 3600:g}",
        tokens_m=f"{budget.tokens / 1e6:g}",
        relative_rule=RELATIVE_RULE,
        suffix=brief.name_suffix,
        target=plans.TARGET_TEXT,
        repeats=(
            f"{plans.FIRST_REPEATS} (then {plans.CONFIRMED_REPEATS - plans.FIRST_REPEATS} "
            f"more where {plans.CONFIRM_FROM} of those latched)"
        ),
    )
    command = [*runtime.codex, "exec", "--json", "--skip-git-repo-check", "--ignore-rules",
               "-C", str(workdir), "-o", str(private / "last_message.txt"), "-"]  # fmt: skip
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(state / "home"),
        "CODEX_HOME": str(state / "codex_home"),
        "LANG": "C.UTF-8",
    }
    started = time.monotonic()
    status, exit_code = "done", None
    last = ""
    errors: list[str] = []
    heard = started

    def outcome(status: str, exit_code: int | None) -> SessionResult:
        """The session's result, written to `result.json`."""
        final = private / "last_message.txt"
        result = SessionResult(
            status=status,
            exit_code=exit_code,
            skills=sorted(p.parent for p in (workdir / "skill").glob("*/skill.py")),
            tokens=usage.tokens().total,
            usage=usage.tokens(),
            usage_source=usage.source,
            wall_s=round(time.monotonic() - started, 1),
            case_runs=channel.used,
            cases=channel.served,
            session_id=session_id,
            transcript=transcript,
            last_message=final.read_text(encoding="utf-8") if final.is_file() else last,
        )
        (private / "result.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        return result

    try:
        with (private / "codex.stderr").open("wb") as stderr:
            process = await anyio.open_process(
                command, stdin=-1, stdout=-1, stderr=stderr, cwd=workdir, env=env,
                start_new_session=True,
            )  # fmt: skip
            pid = private / "pid"  # Codex's process group: a resumed driver kills orphans by it
            pid.write_text(f"{process.pid}\n", encoding="utf-8")
            try:
                async with anyio.create_task_group() as group:
                    stop: list[str] = []

                    async def read() -> None:
                        nonlocal last, heard
                        assert process.stdout is not None
                        with transcript.open("a", encoding="utf-8") as log:
                            async for line in lines(process.stdout):
                                heard = time.monotonic()
                                log.write(line + "\n")
                                log.flush()
                                event = parse(line)
                                usage.codex_event(event)
                                if (text := message(event)) is not None:
                                    last = text
                                if (error := failure(event)) is not None:
                                    errors.append(error)

                    async def watch() -> None:
                        while True:
                            await anyio.sleep(runtime.poll_s)
                            await anyio.to_thread.run_sync(usage.poll)
                            now = time.monotonic()
                            elapsed = now - started
                            channel.status(elapsed, usage.tokens().total)
                            silent = now - max(heard, channel.quiet_since)
                            if not channel.inflight and silent > STALL_S:
                                stop.append("stalled")
                            elif elapsed > budget.wall_s:
                                stop.append("timeout")
                            elif usage.tokens().total > budget.tokens:
                                stop.append("tokens")
                            if stop:
                                await terminate(process, runtime.grace_s)
                                return

                    group.start_soon(channel.serve, runtime.poll_s)
                    group.start_soon(watch)
                    assert process.stdin is not None
                    await process.stdin.send(prompt.encode())
                    await process.stdin.aclose()
                    await read()
                    exit_code = await process.wait()
                    group.cancel_scope.cancel()
                failed = f"error: codex exited {exit_code}: {errors[-1] if errors else ''}"
                if stop:
                    status = stop[0]
                elif exit_code == 0:
                    status = "done"
                else:
                    status = "stalled" if any(STALLED.search(e) for e in errors) else failed
            finally:
                with anyio.CancelScope(shield=True):
                    if process.returncode is None:
                        await terminate(process, runtime.grace_s)
                pid.unlink(missing_ok=True)
    except anyio.get_cancelled_exc_class():  # the caller's clock: what was used, still kept
        usage.poll()
        outcome(status="cancelled", exit_code=None)
        raise
    finally:
        shutil.rmtree(state, ignore_errors=True)  # the gateway key does not outlive the session
    await anyio.to_thread.run_sync(usage.poll)
    return outcome(status=status, exit_code=exit_code)


def prepare(brief: Brief, workdir: Path, runtime: Runtime) -> tuple[Path, Path, str]:
    """The workspace, the private directory beside it and Codex's config: (workdir, private,
    session id)."""
    workdir = workdir.resolve()
    private = workdir.with_name(workdir.name + ".codex")
    if brief.relay:  # continues the session that left `workdir`: its files stay, a new channel
        n = 1
        while (private := workdir.with_name(f"{workdir.name}.codex-relay{n}")).exists():
            n += 1
        for d in (workdir / ".case", workdir / ".tmp"):
            shutil.rmtree(d, ignore_errors=True)
    elif workdir.exists() and any(workdir.iterdir()):
        raise FileExistsError(f"{workdir}: a session's workspace must be new")
    session_id = SESSION.sub("-", f"p1b-{brief.cluster_id}-{uuid.uuid4().hex[:8]}")[:128]
    state = runtime.codex_root / session_id
    for d in (workdir / "skill", workdir / "bin", workdir / ".case", workdir / ".tmp",
              private / "cases", state / "codex_home", state / "home"):  # fmt: skip
        d.mkdir(parents=True, exist_ok=True)
    (workdir / "BRIEF.md").write_text(render(brief) + workspace_notes(runtime, brief),
                                      encoding="utf-8")  # fmt: skip
    (workdir / ENV_FILE).write_text("", encoding="utf-8")
    if brief.task is not None and not brief.relay:  # a relay keeps the plan it was left
        write_json(workdir / "plan.json", brief.task.plan)
    client = workdir / "bin" / "case"
    client.write_text(
        CLIENT.replace("#!python", f"#!{runtime.python}").replace("$at_once", str(CASES_IN_FLIGHT)),
        encoding="utf-8",
    )
    client.chmod(0o755)
    config = state / "codex_home" / "config.toml"
    config.write_text(codex_config(brief, workdir, runtime, session_id), encoding="utf-8")
    config.chmod(0o600)
    return workdir, private, session_id


def workspace_notes(runtime: Runtime, brief: Brief | None = None) -> str:
    plan = brief is not None and brief.task is not None
    return "\n".join([
        "",
        "## Workspace",
        "",
        "- `skill/`: write your skill directory here.",
        *(["- `plan.json`: the task's plan; `bin/case --plan` runs it as it is now (with "
           "`skill/`), `bin/case --submit-plan` queues one."] if plan else []),
        "- `bin/case SKILL GAP`: one case run; `bin/case --submit SKILL GAP` queues one and "
        f"returns, `bin/case --wait ID ...` prints rows, at most {CASES_IN_FLIGHT} run at once "
        "(`bin/case --help`).",
        f"- Python: `{runtime.python}` (the packages `roboscript`, `roboscript_methods` and "
        "numpy are installed).",
        "",
    ])  # fmt: skip


def toml(value: Any) -> str:
    return json.dumps(str(value) if isinstance(value, Path) else value)


def codex_config(brief: Brief, workdir: Path, runtime: Runtime, session_id: str) -> str:
    """`CODEX_HOME/config.toml`: the gateway as the only provider, the permissions profile."""
    repo = brief.code_root.resolve()
    reads = [repo, runtime.python.parent.parent, *runtime.read_roots, *brief.accepted]
    hide = (*runtime.hide, *workdir.parent.glob(f"{workdir.name}.codex*"))  # relays' too
    key = f"{runtime.token.get_secret_value()}:{session_id}"
    lines = [
        f"model = {toml(runtime.model)}",
        'model_provider = "rs_gateway"',
        *([f"model_reasoning_effort = {toml(runtime.effort)}"] if runtime.effort else []),
        'approval_policy = "never"',
        'default_permissions = "builder"',
        'web_search = "disabled"',
        "project_doc_max_bytes = 0",
        "check_for_update_on_startup = false",
        "",
        "[analytics]",
        "enabled = false",
        "",
        "[features]",
        *[f"{name} = false" for name in DISABLED],
        "",
        "[shell_environment_policy]",
        'inherit = "core"',
        "set = { " + ", ".join(f"{k} = {toml(v)}" for k, v in {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": workdir / ".tmp",
            "RS_ENV_FILE": workdir / ENV_FILE,
        }.items()) + " }",
        "",
        "[permissions.builder.filesystem]",
        '":minimal" = "read"',
        *[f'{toml(p)} = "read"' for p in dict.fromkeys(p.resolve() for p in reads)],
        *[f'{toml(p)} = "none"' for p in masked(reads, repo, hide) if p != workdir],
        f'{toml(workdir)} = "write"',
        "",
        "[permissions.builder.network]",
        "enabled = false",
        "",
        "[model_providers.rs_gateway]",
        'name = "RoboScript gateway"',
        f"base_url = {toml(runtime.gateway_url)}",
        'wire_api = "responses"',
        f"stream_idle_timeout_ms = {STREAM_IDLE_MS}",
        f"stream_max_retries = {STREAM_RETRIES}",
        f"experimental_bearer_token = {toml(key)}",
        "",
    ]  # fmt: skip
    return "\n".join(lines)


class Channel:
    """The case channel: `bin/case` writes `.case/<id>.req.json`; this answers `<id>.res.json`."""

    def __init__(
        self,
        workdir: Path,
        private: Path,
        brief: Brief,
        budget: Budget,
        case: Callable[[CaseRequest], Awaitable[dict[str, Any]]],
    ) -> None:
        self.dir, self.skills, self.private = workdir / ".case", workdir / "skill", private
        self.gaps = brief.gap_ids + [g.id for g in brief.also_open if brief.task is not None]
        self.task = brief.task.task if brief.task is not None else None
        """A per-task session's task: it may ask for plan trials (`<task>/#plan`)."""
        self.taken = {p.name for p in brief.accepted}
        """The accepted skills' names: a plan trial's own skill may not take one."""
        self.suffix = brief.name_suffix
        self.budget, self.case = budget, case
        self.used = 0
        """Case runs counted against the budget (a run the infrastructure broke is not)."""
        self.asked = 0
        """Requests run, each numbered (its snapshot's directory)."""
        self.served: list[dict[str, Any]] = []
        self.inflight: set[str] = set()
        """The requests running: the session waits on the simulator, not on the model."""
        self.quiet_since = time.monotonic()
        self.left = {"minutes_left": round(budget.wall_s / 60), "tokens_left": budget.tokens}

    def status(self, elapsed_s: float | None = None, tokens: int | None = None) -> None:
        """`.case/status.json` (`bin/case --list`): case runs, the gaps, and when the watch gives
        them, the wall clock and the tokens left (the Agent has no clock of its own)."""
        if elapsed_s is not None and tokens is not None:
            self.left = {
                "minutes_left": round(max(0.0, self.budget.wall_s - elapsed_s) / 60),
                "tokens_left": max(0, self.budget.tokens - tokens),
            }
        status = {"used": self.used, "budget": self.budget.case_runs, "gaps": self.gaps,
                  "running": len(self.inflight), "at_once": CASES_IN_FLIGHT}  # fmt: skip
        write_json(self.dir / "status.json", status | self.left)

    async def serve(self, poll_s: float) -> None:
        """The requests in the order made: each read, checked and its skill snapshotted as soon
        as it is seen (refused at once, not counted), then run, at most `CASES_IN_FLIGHT` at
        once; the rest wait their turn."""
        waiting: list[tuple[Path, Queued]] = []
        async with anyio.create_task_group() as group:
            while True:
                for request in sorted(self.dir.glob("*.req.json"), key=lambda p: p.stat().st_mtime):
                    taken = request.with_name(request.name.replace(".req.", ".run."))
                    request.rename(taken)
                    found = self.take(taken)
                    if isinstance(found, Queued):
                        waiting.append((taken, found))
                    else:
                        self.reply(taken, found)
                while waiting and len(self.inflight) < CASES_IN_FLIGHT:
                    taken, found = waiting.pop(0)
                    self.inflight.add(taken.name)
                    group.start_soon(self.one, taken, found)
                await anyio.sleep(poll_s)

    def reply(self, taken: Path, answer: dict[str, Any]) -> None:
        write_json(taken.with_name(taken.name.replace(".run.", ".res.")), answer)
        taken.rename(taken.with_name(taken.name.replace(".run.", ".done.")))
        self.status()

    async def one(self, taken: Path, queued: "Queued") -> None:
        try:
            answer = await self.answer(queued)
        finally:
            self.inflight.discard(taken.name)
            if not self.inflight:
                self.quiet_since = time.monotonic()
        self.reply(taken, answer)

    def take(self, request: Path) -> "Queued | dict[str, Any]":
        """A request read and checked, its skill snapshotted (`<private>/queued/<id>/`) from the
        copy `bin/case` made when it was asked (`.case/<id>.skill/`; `skill/` without one): what
        runs is the skill as it was then; or the refusal."""
        try:
            asked = json.loads(request.read_text(encoding="utf-8"))
            if asked.get("plan"):
                copied = self.dir / f"{request.name.split('.')[0]}.plan"
                try:
                    return self.queue_plan(request, copied)
                finally:
                    shutil.rmtree(copied, ignore_errors=True)
            skill, gap = str(asked["skill"]), str(asked["gap"])
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return {"ok": False, "error": f"unreadable request: {error}"}
        name = Path(skill).name  # a directory of skill/, whatever path was given
        copied = self.dir / f"{request.name.split('.')[0]}.skill"
        directory = copied / name if (copied / name).is_dir() else self.skills / name
        try:
            return self.queue(request, name, gap, directory)
        finally:
            shutil.rmtree(copied, ignore_errors=True)

    def queue(
        self, request: Path, name: str, gap: str, directory: Path
    ) -> "Queued | dict[str, Any]":
        if name in ("", ".", "..") or not (directory / "skill.py").is_file():
            return {"ok": False, "error": f"no skill directory skill/{name} with a skill.py"}
        if gap not in self.gaps:
            return {"ok": False, "error": f"{gap!r} is not a gap of the brief: {self.gaps}"}
        if not name.endswith(self.suffix):
            return {"ok": False, "error": f"skill/{name}: its name must end in {self.suffix!r} "
                    "(BRIEF.md: another session writes a skill beside you)"}  # fmt: skip
        if self.used >= self.budget.case_runs:
            return {"ok": False, "error": f"case budget spent ({self.budget.case_runs} runs)"}
        snapshot = self.private / "queued" / request.name.split(".")[0] / directory.name
        if snapshot.parent.exists():
            shutil.rmtree(snapshot.parent)
        # links copied as links, never followed, then refused: checked on the copy, not before it
        shutil.copytree(directory, snapshot, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))  # fmt: skip
        links = [p for p in [snapshot, *walk(snapshot)] if p.is_symlink()]
        if links:  # a link would pull files from outside the workspace into the snapshot
            shutil.rmtree(snapshot.parent)
            found = ", ".join(str(p.relative_to(snapshot.parent)) for p in links)
            return {"ok": False, "error": f"skill/{name} holds symbolic links: {found}"}
        return Queued(name=name, gap=gap, snapshot=snapshot)

    def queue_plan(self, request: Path, copied: Path) -> "Queued | dict[str, Any]":
        """A plan trial's request (a per-task session's): `plan.json` and `skill/` as `bin/case`
        copied them when asked (`.case/<id>.plan/`), snapshotted, or the refusal."""
        if self.task is None:
            return {"ok": False, "error": "plan trials are a per-task session's"}
        if not (copied / "plan.json").is_file():
            return {"ok": False, "error": "no plan.json in the working directory"}
        try:  # what the trial's `plans.read` would refuse, refused here with the reason
            plan = plans.read(copied / "plan.json")
        except ValueError as error:
            return {"ok": False, "error": f"plan.json is not a task plan: {str(error)[:500]}"}
        if plan.task != self.task:
            return {"ok": False, "error": f"plan.json is task {plan.task!r}'s, not {self.task!r}"}
        names = sorted(p.parent.name for p in (copied / "skill").glob("*/skill.py"))
        wrong = [n for n in names if not n.endswith(self.suffix) or n in self.taken]
        if wrong:
            return {"ok": False, "error": f"skill/{wrong[0]}: a skill's name must end in "
                    f"{self.suffix!r} and not be an accepted skill's"}  # fmt: skip
        if self.used >= self.budget.case_runs:
            return {"ok": False, "error": f"case budget spent ({self.budget.case_runs} runs)"}
        snapshot = self.private / "queued" / request.name.split(".")[0] / "plan"
        if snapshot.parent.exists():
            shutil.rmtree(snapshot.parent)
        shutil.copytree(copied, snapshot, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))  # fmt: skip
        links = [p for p in [snapshot, *walk(snapshot)] if p.is_symlink()]
        if links:
            shutil.rmtree(snapshot.parent)
            return {"ok": False, "error": "plan.json or skill/ holds symbolic links"}
        return Queued(name="plan", gap=f"{self.task}/{TRIAL}", snapshot=snapshot)

    async def answer(self, queued: "Queued") -> dict[str, Any]:
        """The response to one queued request: refused when the budget is spent by the time it
        starts (not counted), or the case run's row."""
        if self.used >= self.budget.case_runs:
            shutil.rmtree(queued.snapshot.parent, ignore_errors=True)
            return {"ok": False, "error": f"case budget spent ({self.budget.case_runs} runs)"}
        n = self.asked + 1
        self.asked = n
        self.used += 1
        snapshot = self.private / "cases" / f"{n:02d}" / queued.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        queued.snapshot.rename(snapshot)
        queued.snapshot.parent.rmdir()
        gap, name = queued.gap, queued.name
        task, _, segment = gap.partition("/")
        started = time.monotonic()
        asked_for = CaseRequest(
            n=n, gap=gap, task=task, segment=segment, skill=name, skill_dir=snapshot
        )
        row = await self.case(asked_for)
        if row.get("infra"):  # the infrastructure's, not the session's: not counted
            self.used -= 1
        write_json(snapshot.parent / "row.json", row)
        record = {
            "n": n,
            "gap": gap,
            "skill": name,
            "explained": row.get("explained"),
            "infra": bool(row.get("infra")),
            "error": row.get("error"),
            "seconds": round(time.monotonic() - started, 1),
        }
        self.served.append(record)
        with (self.private / "cases.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        left = self.budget.case_runs - self.used
        return {"ok": True, "run": n, "runs_left": left, "counted": not row.get("infra"),
                "row": row}  # fmt: skip


class Queued(BaseModel):
    """A request checked and its skill snapshotted, waiting for its turn."""

    name: str
    gap: str
    snapshot: Path


class Usage:
    """Tokens of the session: the gateway's request log (lines of `session_id` appended since the
    session started) and Codex's `turn.completed` events; the larger total counts."""

    def __init__(self, log: Path | None, session_id: str) -> None:
        self.log, self.needle = log, f'"session":"{session_id}"'.encode()
        self.offset = log.stat().st_size if log is not None and log.is_file() else 0
        self.rest = b""
        self.gateway, self.codex = Tokens(), Tokens()

    @property
    def source(self) -> Literal["gateway", "codex"]:
        return "gateway" if self.gateway.total >= self.codex.total and self.log else "codex"

    def tokens(self) -> Tokens:
        return self.gateway if self.source == "gateway" else self.codex

    def poll(self) -> None:
        if self.log is None or not self.log.is_file():
            return
        with self.log.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        *complete, self.rest = (self.rest + data).split(b"\n")
        for line in complete:
            if self.needle not in line:
                continue
            used = (parse(line.decode("utf-8", "replace")).get("data") or {}).get("usage") or {}
            self.gateway = add(self.gateway, used)

    def codex_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "turn.completed":
            self.codex = add(self.codex, event.get("usage") or {})


def add(tokens: Tokens, usage: dict[str, Any]) -> Tokens:
    """`tokens` plus a Responses usage (`input_tokens_details.cached_tokens`) or Codex's
    (`cached_input_tokens`)."""
    details = usage.get("input_tokens_details") or {}
    cached = usage.get("cached_input_tokens", details.get("cached_tokens", 0))
    return Tokens(
        input=tokens.input + int(usage.get("input_tokens") or 0),
        cached=tokens.cached + int(cached or 0),
        output=tokens.output + int(usage.get("output_tokens") or 0),
    )


def parse(line: str) -> dict[str, Any]:
    try:
        found = json.loads(line)
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def failure(event: dict[str, Any]) -> str | None:
    """An `error` or `turn.failed` event's message."""
    if event.get("type") == "error":
        return str(event.get("message", ""))
    if event.get("type") == "turn.failed":
        return str((event.get("error") or {}).get("message", ""))
    return None


def message(event: dict[str, Any]) -> str | None:
    item = event.get("item") or {}
    if event.get("type") == "item.completed" and item.get("type") == "agent_message":
        return str(item.get("text", ""))
    return None


async def lines(stream: Any) -> Any:
    rest = b""
    async for chunk in stream:
        *complete, rest = (rest + chunk).split(b"\n")
        for line in complete:
            yield line.decode("utf-8", "replace")
    if rest:
        yield rest.decode("utf-8", "replace")


def descendants(pid: int) -> list[int]:
    """`pid`'s descendants on Linux (`/proc`), children first; [] elsewhere."""
    found: list[int] = []
    queue = [pid]
    while queue:
        parent = queue.pop()
        for children in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                kids = [int(k) for k in children.read_text(encoding="utf-8").split()]
            except OSError:
                continue
            found += kids
            queue += kids
    return found


def signal_tree(pid: int, sig: int) -> None:
    """`sig` to `pid`'s process group (Codex's session) and to every descendant (a command in a
    terminal of its own has a session of its own)."""
    for target in [*descendants(pid), pid]:
        with contextlib.suppress(ProcessLookupError):
            os.kill(target, sig)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


async def terminate(process: Process, grace_s: float) -> None:
    """SIGTERM to Codex and everything under it, SIGKILL after `grace_s`."""
    if process.returncode is not None:
        return
    tree = descendants(process.pid)
    signal_tree(process.pid, signal.SIGTERM)
    with anyio.move_on_after(grace_s):
        await process.wait()
    for pid in [*tree, *descendants(process.pid)]:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    if process.returncode is None:
        signal_tree(process.pid, signal.SIGKILL)
        await process.wait()


def walk(directory: Path) -> list[Path]:
    """Everything under `directory`, never descending into a link."""
    found = []
    for parent, dirs, files in os.walk(directory):
        found += [Path(parent) / n for n in dirs + files]
    return found


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


CLIENT: Final = """#!python
\"\"\"Run cases: your skill on gaps of the brief, in the demonstration's scene.

    bin/case SKILL GAP           one case run, waited for. SKILL: a directory under skill/;
                                 GAP: <task>/<segment> from BRIEF.md
    bin/case --submit SKILL GAP  queue one and return at once (its id is printed); the skill
                                 runs as it is now, later edits do not change it
    bin/case --wait ID [ID ...]  wait for requests already made and print their rows
    bin/case --list              the gaps you may run, the case runs, minutes and tokens left
    bin/case --plan              a per-task session's plan trial: plan.json and skill/ as they
                                 are now, waited for; --submit-plan queues one and returns

At most $at_once case runs go at once (the rest wait their turn); each counts when it starts.
Prints each row as JSON. Exit 0: every case ran; 3: a request was refused; 4: no answer within
WAIT_S seconds (the request stays queued: `bin/case --wait ID` again).
\"\"\"
import json, os, shutil, sys, time, uuid

CASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".case")
WAIT_S = 3600


def wait(rid, named=False):
    answer = os.path.join(CASE, rid + ".res.json")
    deadline = time.monotonic() + WAIT_S
    while not os.path.exists(answer):
        if time.monotonic() > deadline:
            print(f"request {rid}: no answer in {WAIT_S} s; it stays queued: "
                  f"bin/case --wait {rid}", flush=True)
            return 4
        time.sleep(1)
    with open(answer, encoding="utf-8") as f:
        found = json.load(f)
    if named:
        print(f"request {rid}:")
    print(json.dumps(found, indent=1), flush=True)
    return 0 if found.get("ok") else 3


def submit_plan():
    rid = uuid.uuid4().hex[:12]
    root = os.path.dirname(CASE)
    target = os.path.join(CASE, rid + ".plan")
    os.makedirs(target)
    if os.path.isfile(os.path.join(root, "plan.json")):
        shutil.copyfile(os.path.join(root, "plan.json"), os.path.join(target, "plan.json"))
    if os.path.isdir(os.path.join(root, "skill")):
        shutil.copytree(os.path.join(root, "skill"), os.path.join(target, "skill"), symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    tmp = os.path.join(CASE, rid + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"plan": True}, f)
    os.rename(tmp, os.path.join(CASE, rid + ".req.json"))
    return rid


def submit(skill, gap):
    rid = uuid.uuid4().hex[:12]
    name = os.path.basename(os.path.normpath(skill))
    source = os.path.join(os.path.dirname(CASE), "skill", name)
    if name not in ("", ".", "..") and os.path.isfile(os.path.join(source, "skill.py")):
        # the skill as it is now: what the run takes, whatever is edited before it starts
        shutil.copytree(source, os.path.join(CASE, rid + ".skill", name), symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    tmp = os.path.join(CASE, rid + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"skill": skill, "gap": gap}, f)
    os.rename(tmp, os.path.join(CASE, rid + ".req.json"))
    return rid


args = sys.argv[1:]
if args[:1] == ["--list"]:
    with open(os.path.join(CASE, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    print(f"case runs used {status['used']} of {status['budget']}; about "
          f"{status['minutes_left']} minutes and {status['tokens_left']} tokens left; "
          f"{status['running']} running (at most {status['at_once']} at once); gaps:")
    print("\\n".join(status["gaps"]))
elif args[:1] == ["--wait"] and len(args) >= 2:
    sys.exit(max([wait(rid, len(args) > 2) for rid in args[1:]]))
elif args == ["--plan"]:
    rid = submit_plan()
    print(f"request {rid}: queued; a plan trial takes minutes", flush=True)
    sys.exit(wait(rid))
elif args == ["--submit-plan"]:
    rid = submit_plan()
    print(f"request {rid}: queued; `bin/case --wait {rid}` prints its report", flush=True)
elif args[:1] == ["--submit"] and len(args) == 3:
    rid = submit(args[1], args[2])
    print(f"request {rid}: queued; `bin/case --wait {rid}` prints its row", flush=True)
elif len(args) == 2 and not args[0].startswith("-"):
    rid = submit(args[0], args[1])
    print(f"request {rid}: queued; a case run takes minutes", flush=True)
    sys.exit(wait(rid))
else:
    print(__doc__)
    sys.exit(0 if args[:1] in (["-h"], ["--help"]) else 2)
"""
"""`bin/case`: standard library only, its interpreter the session's Python."""
