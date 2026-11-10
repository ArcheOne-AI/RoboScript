"""A pipeline's status and summary, read from its jobs' records (never from a job's report of
itself): `status.json` and `status.md` (a compact table) while it runs, `summary.json` and
`summary.md` once every job has ended.

The core reads what every job has: its state (`pipeline.json`), and each episode's tokens (the
brain's and perception's, per stage and per role; a running episode's by the model calls its event
log shows so far) and time, voided infrastructure runs and set-aside stages included (directories
`<name>-infra<n>`, `<name>-aborted<n>`, `learning.episode`), and so do its infrastructure faults;
guard trips are the Agent host's runaway guards. The summary adds a failure taxonomy of the
episodes that count (neither voided nor set aside): the receipts' reasons, the hard failures by the
workflow stage their call named (`<reason>@<stage>`), the ends of the failed episodes, the
identification outcomes by role (fresh identifications only) and how episodes opened;
infrastructure ends apart. The scheduler's view (GPUs, alerts, memory pressure:
`launch.Launch.extra`) heads the status table.

Everything else is the parts' (`ReportPart`, in order: a method's, then a benchmark's
evaluation): which episodes a job has and how a stage's summaries read (`EpisodeReading`; the
core's two kinds: `LEARNING`, `learning.episode`'s, and `HOST`, `agent.host`'s), their fields of a
job's row and of the summary, their flags, their lines above the summary table, their columns of
both tables, the lanes beside a row and the rows a summary row stands on (`chain`)."""

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, NamedTuple, Protocol

from roboscript.agent.host import infra_lost
from roboscript.contracts.events import EventKind
from roboscript.contracts.receipt import HARD_FAILURES
from roboscript.orchestrate.jobs import RunKey, RunState
from roboscript.records import write_atomic
from roboscript.tools import IDENTIFIED

Row = dict[str, Any]
_HARD: Final = frozenset(str(reason) for reason in HARD_FAILURES)
_ASIDE: Final = re.compile(r".*-(aborted|infra)\d+")  # set aside or voided
TAXONOMY: Final = (
    "receipts",
    "hard_failures",
    "failed_ends",
    "identification",
    "openings",
    "infra_ends",
)
"""The core's fields of a job's failure taxonomy, by stage."""


class Episode(NamedTuple):
    """An episode of a job: its stage, its directory and whether it counts (neither voided nor in
    a stage set aside)."""

    stage: str
    path: Path
    counts: bool


class EpisodeReading(Protocol):
    """How a stage's episode summaries (`summary.json`) read: the role that ran it (`summary`
    None: running, from its `path`'s records), whether infrastructure ended it, its end's reason,
    whether it failed, how it opened (None: not recorded), whether its identifications are fresh
    and whether a runaway guard ended it."""

    def role(self, summary: Mapping[str, Any] | None, path: Path) -> str | None: ...

    def infra(self, summary: Mapping[str, Any]) -> bool: ...

    def end(self, summary: Mapping[str, Any]) -> str: ...

    def failed(self, summary: Mapping[str, Any]) -> bool: ...

    def opening(self, summary: Mapping[str, Any]) -> str | None: ...

    def fresh(self, summary: Mapping[str, Any]) -> bool: ...

    def guard_tripped(self, summary: Mapping[str, Any]) -> bool: ...


class _Learning:
    """`learning.episode.EpisodeSummary`: an S1/S2-style episode of a role."""

    def role(self, summary: Mapping[str, Any] | None, path: Path) -> str | None:
        if summary is None:
            return (read_json(path / "config.json") or {}).get("role")
        return summary["role"]

    def infra(self, summary: Mapping[str, Any]) -> bool:
        return summary["end_class"] == "infra"

    def end(self, summary: Mapping[str, Any]) -> str:
        return str(summary["end"].get("reason", "none"))

    def failed(self, summary: Mapping[str, Any]) -> bool:
        return summary["role"] == "actor" and not summary["success"]

    def opening(self, summary: Mapping[str, Any]) -> str | None:
        opening = summary.get("identification") or {}
        return opening["status"] if opening else None

    def fresh(self, summary: Mapping[str, Any]) -> bool:
        return (summary.get("identification") or {}).get("status") == "identified"

    def guard_tripped(self, summary: Mapping[str, Any]) -> bool:
        return False


class _Host:
    """`agent.host`'s episode summaries: the Actor's episodes of an official client, those
    infrastructure lost as `agent.host.infra_lost` reads them (a success stands)."""

    def role(self, summary: Mapping[str, Any] | None, path: Path) -> str | None:
        return "actor"

    def infra(self, summary: Mapping[str, Any]) -> bool:
        return infra_lost(summary)

    def end(self, summary: Mapping[str, Any]) -> str:
        return str((summary["end"] or {}).get("reason", "none"))

    def failed(self, summary: Mapping[str, Any]) -> bool:
        return not (summary["official"] or {}).get("success", False)

    def opening(self, summary: Mapping[str, Any]) -> str | None:
        return None

    def fresh(self, summary: Mapping[str, Any]) -> bool:
        return True

    def guard_tripped(self, summary: Mapping[str, Any]) -> bool:
        return bool(summary["guard_tripped"])


LEARNING: Final[EpisodeReading] = _Learning()
HOST: Final[EpisodeReading] = _Host()


@dataclass(frozen=True)
class JobView:
    """A job as the report reads it: the pipeline's directory, its key and state, and the time."""

    directory: Path
    key: RunKey
    state: RunState
    now: float

    @property
    def run_dir(self) -> Path:
        """Its unit's directory (a task's first job's too)."""
        return self.directory / self.key.task / self.key.variant


class Section(NamedTuple):
    """A part's fields of a job's row, in order, and of its failure taxonomy (a deep row's; None:
    none)."""

    fields: Row
    taxonomy: Row | None = None


@dataclass(frozen=True)
class Column:
    """A table column: its header and its cell of a row, given the whole status or summary."""

    header: str
    cell: Callable[[Row, Mapping[str, Any]], str]


class Lane(NamedTuple):
    """A job's lane beside a row: its state's record (`run`, `status`, `gpu`, `launches`,
    `respawns`, `running_s`, `failure`, `error`), the stage the status table shows for it, and
    whether the table shows it (its error shows either way)."""

    record: Mapping[str, Any]
    stage: str
    shown: bool


class ReportPart(Protocol):
    """A package's part of the status and summary (a method's learning, a benchmark's
    evaluation): the stages whose episodes it reads (`readings`, in order); a job's episode
    directories by stage (`episodes`) and its section of the job's row (`section`; None: not its
    job); the summary's fields over the rows (`summary`), its flags (`flags`), its lines above the
    summary table (`headline`: every part's first, then `notes`); its columns (`status_columns`,
    `summary_columns`), the labels of its taxonomy fields (`taxonomy_labels`), a row's lanes
    (`lanes`) and the rows a summary row stands on (`chain`: their infrastructure counts with it;
    None: the row is one of those, not a row of the summary table)."""

    readings: Mapping[str, EpisodeReading]
    status_columns: Sequence[Column]
    summary_columns: Sequence[Column]
    taxonomy_labels: Mapping[str, str]

    def episodes(self, job: JobView) -> Iterable[tuple[str, Path]]: ...

    def section(self, job: JobView, deep: bool) -> Section | None: ...

    def summary(self, rows: list[Row]) -> Row: ...

    def flags(self, rows: list[Row]) -> list[str]: ...

    def headline(self, found: Mapping[str, Any]) -> list[str]: ...

    def notes(self, found: Mapping[str, Any]) -> list[str]: ...

    def lanes(self, row: Mapping[str, Any]) -> list[Lane]: ...

    def chain(self, row: Mapping[str, Any], rows: list[Row]) -> list[Row] | None: ...


@dataclass(frozen=True)
class Report:
    """The status and summary of the parts (`parts`, in order)."""

    parts: Sequence[ReportPart]

    @property
    def readings(self) -> dict[str, EpisodeReading]:
        return {stage: r for part in self.parts for stage, r in part.readings.items()}

    def status(
        self,
        directory: Path,
        keys: Iterable[RunKey],
        now: float,
        extra: Mapping[str, Any] | None = None,
    ) -> Row:
        """Every job's row, the counts by status and what the scheduler adds (GPUs, alerts)."""
        rows = [self.job_row(directory, key, now) for key in keys]
        return {
            "directory": str(directory),
            "updated": utc(now),
            "runs": dict(Counter(row["status"] for row in rows)),
            **(extra or {}),
            "rows": rows,
        }

    def summary(self, directory: Path, keys: Iterable[RunKey], now: float) -> Row:
        """The status with each job's details and failure taxonomy, the tokens, the flags and
        the parts' fields."""
        rows = [self.job_row(directory, key, now, deep=True) for key in keys]
        spent: Counter[str] = Counter()
        roles: Counter[str] = Counter()
        for row in rows:
            spent.update(row["tokens"]["all"])
            roles.update(row["tokens"]["roles"])
        page: Row = {
            "directory": str(directory),
            "updated": utc(now),
            "runs": dict(Counter(row["status"] for row in rows)),
            "tokens": dict(spent),
            "role_tokens": dict(roles),
            "flags": [flag for part in self.parts for flag in part.flags(rows)],
        }
        for part in self.parts:
            page |= part.summary(rows)
        return page | {"rows": rows}

    def job_row(
        self,
        directory: Path,
        key: RunKey,
        now: float,
        *,
        deep: bool = False,
    ) -> Row:
        """What a job's records say (see the module docstring)."""
        state = RunState.load(key.unit_dir(directory), key)
        job = JobView(directory, key, state, now)
        episodes = [
            Episode(stage, path, _counted(path, job.run_dir))
            for part in self.parts
            for stage, path in part.episodes(job)
        ]
        row: Row = {
            "run": key.name,
            "status": state.status,
            "stage": state.stage,
            "gpu": state.gpu,
            "launches": state.launches,
            "respawns": state.respawns,
            "running_s": round(state.running_s(now), 1),
            "failure": state.failure,
            "error": state.error,
        }
        sections = [s for part in self.parts if (s := part.section(job, deep)) is not None]
        for section in sections:
            row |= section.fields
        row |= costs(episodes, self.readings)
        if deep:
            row["taxonomy"] = taxonomy(episodes, self.readings)
            for section in sections:
                row["taxonomy"] |= section.taxonomy or {}
        return row

    def write_status(
        self,
        directory: Path,
        keys: Iterable[RunKey],
        now: float,
        extra: Mapping[str, Any] | None = None,
    ) -> Row:
        found = self.status(directory, keys, now, extra)
        write(directory / "status.json", json.dumps(found, indent=1) + "\n")
        write(directory / "status.md", self.status_table(found))
        return found

    def write_summary(self, directory: Path, keys: Iterable[RunKey], now: float) -> Row:
        found = self.summary(directory, keys, now)
        write(directory / "summary.json", json.dumps(found, indent=1) + "\n")
        write(directory / "summary.md", self.summary_page(found))
        return found

    def status_table(self, found: Mapping[str, Any]) -> str:
        """The status as a compact Markdown table, with the GPUs' services and the alerts."""
        stages = list(self.readings)
        columns = [
            Column("job", lambda row, _: row["run"]),
            Column("status", lambda row, _: row["status"]),
            Column("stage", lambda row, _: row["stage"] or "-"),
            Column("GPU", lambda row, _: "-" if row["gpu"] is None else str(row["gpu"])),
            *(column for part in self.parts for column in part.status_columns),
            Column("tokens", lambda row, _: role_tokens(row["tokens"])),
            Column(
                f"infra {'/'.join(stages)}",
                lambda row, _: "/".join(str(row["infra"][stage]) for stage in stages),
            ),
            Column("respawns", lambda row, _: str(row["respawns"])),
            Column("time", lambda row, _: duration(row["running_s"])),
        ]
        tally = ", ".join(f"{n} {status}" for status, n in sorted(found["runs"].items()))
        lines = [
            f"# Pipeline {found['directory']}",
            "",
            f"Updated {found['updated']}: {tally}.",
            "",
            *_scheduler_lines(found),
            "",
            *_header(columns),
        ]
        lanes = []
        for row in found["rows"]:
            lines.append(_line(column.cell(row, found) for column in columns))
            for lane in (lane for part in self.parts for lane in part.lanes(row)):
                lanes.append(lane.record)
                if lane.shown:
                    record = lane.record
                    gpu = "-" if record["gpu"] is None else str(record["gpu"])
                    cells = [record["run"], record["status"], lane.stage, gpu]
                    cells += ["-"] * (len(columns) - 6)
                    cells += [str(record["respawns"]), duration(record["running_s"])]
                    lines.append(_line(cells))
        errors = [
            f"- {row['run']}: {row['error']}" for row in [*found["rows"], *lanes] if row["error"]
        ]
        if errors:
            lines += ["", "Errors:", *errors]
        return "\n".join(lines) + "\n"

    def summary_page(self, found: Mapping[str, Any]) -> str:
        """The summary in Markdown: the parts' lines, a table of the rows that do not stand
        under others (with those they stand on), then the taxonomy."""
        stages = list(self.readings)
        rows = found["rows"]

        def infra(row: Row, _: Mapping[str, Any]) -> str:
            chain = [up for part in self.parts for up in part.chain(row, rows) or ()]
            return "/".join(
                str(row["infra"][stage] + sum(up["infra"][stage] for up in chain))
                for stage in stages
            )

        columns = [
            Column("run", lambda row, _: row["run"]),
            Column("status", lambda row, _: row["status"]),
            *(column for part in self.parts for column in part.summary_columns),
            Column(f"infra {'/'.join(stages)}", infra),
            Column("guard trips", lambda row, _: str(row["guard_trips"])),
            Column("tokens", lambda row, _: tokens(row["tokens"]["all"])),
            Column("time", lambda row, _: duration(row["running_s"])),
        ]
        overview = [line for part in self.parts for line in part.headline(found)]
        overview += [line for part in self.parts for line in part.notes(found)]
        lines = [
            f"# Pipeline summary {found['directory']}",
            "",
            f"Updated {found['updated']}. Tokens: {tokens(found['tokens'])}.",
            *(f"**{flag}.**" for flag in found["flags"]),
            "",
            *([*overview, ""] if overview else []),
            *_header(columns),
        ]
        for row in rows:
            if any(part.chain(row, rows) is None for part in self.parts):
                continue
            lines.append(_line(column.cell(row, found) for column in columns))
        labels = {
            name: label for part in self.parts for name, label in part.taxonomy_labels.items()
        }
        lines += ["", "## Failure taxonomy", ""]
        for row in rows:
            lines.append(f"### {row['run']}" + (f" ({row['error']})" if row["error"] else ""))
            for name, by_stage in row["taxonomy"].items():
                if name in labels:
                    if by_stage:
                        lines.append(f"- {labels[name]}: {counts(by_stage)}")
                    continue
                for stage, found_counts in by_stage.items():
                    if found_counts:
                        lines.append(f"- {name.replace('_', ' ')} {stage}: {counts(found_counts)}")
            lines.append("")
        return "\n".join(lines)


def _counted(path: Path, run_dir: Path) -> bool:
    """Whether an episode counts: no directory on its way from the unit's is voided or set
    aside."""
    return not any(_ASIDE.fullmatch(part) for part in path.relative_to(run_dir).parts)


def _header(columns: Sequence[Column]) -> list[str]:
    return [_line(column.header for column in columns), "|" + "---|" * len(columns)]


def _line(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _scheduler_lines(found: Mapping[str, Any]) -> list[str]:
    """The alerts, the GPUs' table, the GPUs skipped and the jobs waiting for memory."""
    lines = [f"**ALERT: {alert}**" for alert in found.get("alerts", [])]
    if gpus := found.get("gpus"):
        lines += _gpu_table(gpus, found.get("gpu_cap_gb"))
    lines += [f"GPU {gpu}: skipped: {why}." for gpu, why in (found.get("skipped") or {}).items()]
    lines += [
        f"{name}: out of GPU memory, waiting for room ({p['retries']} retries, "
        f"{p['waited_s']} of {p['limit_s']:g} s)."
        for name, p in (found.get("pressure") or {}).items()
    ]
    return lines


def costs(episodes: list[Episode], readings: Mapping[str, EpisodeReading]) -> Row:
    """Tokens (the brain's and perception's) per stage and per role over every episode, a
    running one's by the model calls its records show so far; episode time and infrastructure
    episodes per stage over the ended ones; guard trips over those that count."""
    stages = list(readings)
    by_stage: dict[str, Counter[str]] = {stage: Counter() for stage in stages}
    roles: Counter[str] = Counter()
    seconds = dict.fromkeys(stages, 0.0)
    infra = dict.fromkeys(stages, 0)
    guards = 0
    for stage, path, counted in episodes:
        reading = readings[stage]
        episode = read_json(path / "summary.json")
        if episode is None:
            usage = live_usage(path)
            role = reading.role(None, path)
        else:
            usage = Counter(episode.get("usage") or {})
            usage.update(episode.get("perception_usage") or {})
            role = reading.role(episode, path)
            seconds[stage] += episode.get("wall_clock_s") or episode.get("elapsed_s") or 0.0
            infra[stage] += reading.infra(episode)
            guards += counted and reading.guard_tripped(episode)
        by_stage[stage].update(usage)
        roles[str(role)] += _total(usage)
    every: Counter[str] = Counter()
    for found in by_stage.values():
        every.update(found)
    return {
        "tokens": {
            "all": dict(every),
            **{stage: _total(c) for stage, c in by_stage.items()},
            "roles": dict(roles),
        },
        "episode_s": {stage: round(s, 1) for stage, s in seconds.items()},
        "infra": infra,
        "guard_trips": guards,
    }


def live_usage(path: Path) -> Counter[str]:
    """A running episode's tokens so far, from its complete event lines: each model call's as it
    ends (`model.response`) and perception's (a host writes an episode's events as it ends)."""
    usage: Counter[str] = Counter()
    kinds = (EventKind.MODEL_RESPONSE, EventKind.PERCEPTION_STEP)
    for event in json_lines(path / "events.jsonl", [f'"{kind}"'.encode() for kind in kinds]):
        if event["kind"] not in kinds:
            continue
        data = event.get("data") or {}
        asked = data.get("asked") if event["kind"] == EventKind.PERCEPTION_STEP else data
        got = (asked.get("usage") or {}) if isinstance(asked, dict) else {}
        usage.update({key: got.get(key) or 0 for key in ("input_tokens", "output_tokens")})
    return usage


def json_lines(path: Path, marks: list[bytes]) -> Iterator[dict[str, Any]]:
    """The complete JSON lines of a file being appended to that contain one of `marks`."""
    if not path.is_file():
        return
    with path.open("rb") as lines:
        for line in lines:
            if line.endswith(b"\n") and any(mark in line for mark in marks):
                yield json.loads(line)


def taxonomy(episodes: list[Episode], readings: Mapping[str, EpisodeReading]) -> Row:
    """Of the episodes that count: receipts, hard failures by workflow stage, the failed ends,
    identification outcomes by role and how episodes opened; infrastructure ends of every
    episode apart, one an unanswered infrastructure error voided with the end it would have had
    (e.g. `infra_error(finish)`)."""
    found: dict[str, dict[str, Counter[str]]] = {
        name: {stage: Counter() for stage in readings} for name in TAXONOMY
    }
    for stage, path, counted in episodes:
        reading = readings[stage]
        episode = read_json(path / "summary.json")
        if episode is None:
            continue
        end = reading.end(episode)
        if reading.infra(episode):
            # one lost to an unanswered infrastructure error, by the end it would have had
            # (`agent.session.unanswered_end`: `voided_end`)
            voided = episode.get("voided_end")
            found["infra_ends"][stage][f"{end}({voided})" if voided else end] += 1
            continue
        if not counted:
            continue
        failed = reading.failed(episode)
        if (opening := reading.opening(episode)) is not None:
            found["openings"][stage][opening] += 1
        fresh = reading.fresh(episode)
        if failed:
            found["failed_ends"][stage][end] += 1
        workflow: dict[str, str] = {}  # call id -> the workflow stage its rationale named
        for kind, data in _events(path / "events.jsonl"):
            if kind == EventKind.TOOL_CALL:
                rationale = (data.get("arguments") or {}).get("rationale")
                if isinstance(rationale, dict) and rationale.get("stage"):
                    workflow[data["call_id"]] = rationale["stage"]
            elif kind == EventKind.TOOL_RESULT:
                for receipt in _receipts(data.get("result")):
                    reason = receipt["reason"]
                    found["receipts"][stage][reason] += 1
                    if reason in _HARD:
                        at = workflow.get(data["call_id"], "?")
                        found["hard_failures"][stage][f"{reason}@{at}"] += 1
            elif fresh and data.get("name") == IDENTIFIED:
                for role in (data.get("value") or {}).get("roles", {}).values():
                    found["identification"][stage][role.get("reason", "?")] += 1
    return {
        name: {stage: dict(c.most_common()) for stage, c in by_stage.items()}
        for name, by_stage in found.items()
    }


def _events(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """The tool calls, tool results and scene facts of an episode's complete event lines."""
    wanted = (EventKind.TOOL_CALL, EventKind.TOOL_RESULT, EventKind.SCENE_FACT)
    for event in json_lines(path, [f'"{kind}"'.encode() for kind in wanted]):
        if event["kind"] in wanted:
            yield event["kind"], event.get("data") or {}


def _receipts(result: Any) -> Iterator[dict[str, Any]]:
    """The result if it is an action receipt, else the receipts in its fields (alone, listed or
    by arm), as the Hub finds them."""
    if not isinstance(result, dict):
        return
    if "reason" in result and "ok" in result:
        yield result
        return
    for value in result.values():
        items = (
            value.values()
            if isinstance(value, dict)
            else value
            if isinstance(value, list)
            else [value]
        )
        yield from (
            item for item in items if isinstance(item, dict) and "reason" in item and "ok" in item
        )


def _gpu_table(gpus: Mapping[str, Any], cap: float | None) -> list[str]:
    """Each GPU's memory as the scheduler counts it (`Slots.ledger`), in GB: in use (all, and
    what our jobs' processes show of it), reserved by us and not shown yet, claimed by others,
    room left under the cap less the margin; and its services."""
    lines = [
        f"GPU memory (GB, cap {cap:g}):" if cap is not None else "GPU memory (GB):",
        "",
        "| GPU | state | places | used | ours | reserved | others' claims | room | services |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for gpu, card in gpus.items():
        each = card["services"].items()
        shown = ", ".join(f"{n} {v['state']} ({v['restarts']} restarts)" for n, v in each)
        state = card["state"] + (", backing off" if card.get("backing_off") else "")
        if card["used_gb"] is None:
            memory = ["unread", "", f"{card['reserved_gb']:g}", f"{card['claimed_gb']:g}", ""]
        else:
            memory = [
                f"{card[k]:g}" for k in ("used_gb", "ours_gb", "reserved_gb", "claimed_gb")
            ] + [f"{card['room_gb']:g}"]
        lines.append(f"| {gpu} | {state} | {card['places']} | {' | '.join(memory)} | {shown} |")
    return [*lines, ""]


def _total(usage: Counter[str]) -> int:
    return usage["input_tokens"] + usage["output_tokens"]


def tokens(usage: Mapping[str, int]) -> str:
    """Input and output tokens, counted short."""
    return count(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))


def role_tokens(found: Mapping[str, Any]) -> str:
    """All of a job's tokens, and each role's when more than one role spent any."""
    roles = {role: n for role, n in found["roles"].items() if n}
    if len(roles) < 2:
        return tokens(found["all"])
    each = ", ".join(f"{role} {count(n)}" for role, n in sorted(roles.items()))
    return f"{tokens(found['all'])} ({each})"


def count(total: int) -> str:
    return f"{total / 1e6:.2f}M" if total >= 1e5 else f"{total / 1e3:.1f}k"


def rate(value: float | None, interval: list[float] | None) -> str:
    """A rate with its interval, in percent ("-": none)."""
    if value is None or interval is None:
        return "-"
    return f"{value:.0%} [{interval[0]:.0%}, {interval[1]:.0%}]"


def counts(found: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {n}" for name, n in found.items())


def duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return (
        f"{minutes // 60}h{minutes % 60:02d}m"
        if minutes >= 60
        else f"{minutes}m{int(seconds) % 60:02d}s"
    )


def read_json(path: Path) -> Any:
    """A JSON file's content (None: no such file)."""
    return json.loads(path.read_bytes()) if path.is_file() else None


def utc(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(path: Path, text: str) -> None:
    write_atomic(path, text.encode())
