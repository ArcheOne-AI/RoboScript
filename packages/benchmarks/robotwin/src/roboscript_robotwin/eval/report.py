"""RoboTwin P3's part of a pipeline's status and summary (`roboscript.orchestrate.report`),
read from a unit's blocks (`P3/<mode>/block<n>`) and its spare lane's records.

P3's rate counts the official result of a complete block (`result.json`'s success rate over
RoboTwin's `OFFICIAL_EPISODES`) and the Agent host's summaries of a block cut on purpose
(`p3.episodes`, `p3.episodes_randomized`: every episode it finished, so n may exceed the count;
one closed from a set-aside run shows `closed`), and shows both n beside the episodes the host
saw end; a block still running shows its progress. The summary gives each mode's rate over the
units (`p3_overall`): a unit stopped before P3 (the `stopped` and `zeroed` callbacks) counts 0 of
every episode, and the rate over all units also counts as 0 those not in yet or not counted. It
flags a block that counts with more than `INFRA_FLAG` of its episodes failed by infrastructure
(`agent.host.infra_lost`: a success stands; the official rate counts them as failures). Under
`p3.randomized: spare`, a unit's randomized blocks that neither ran to the end nor run now show
as waiting for spare room (`spare_waiting`); under `p3.spare_parallel`, its lane's job (`spare`:
its state, from its own `pipeline.json`), a line of its own in the table. A unit whose P3 the
caller does not count (`gated`) shows as not counted (`p3.counted`: false; `P3Labels.cell` before
its cells): the blocks it ran stay on disk. The caller names its reasons (`P3Labels`)."""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from roboscript.agent.host import infra_lost
from roboscript.contracts.ids import TaskName
from roboscript.learning import wilson
from roboscript.orchestrate.jobs import RunKey, RunState
from roboscript.orchestrate.report import (
    HOST,
    Column,
    JobView,
    Lane,
    Row,
    Section,
    rate,
    read_json,
)
from roboscript_robotwin.eval.blocks import LANE_DIRS, SPARE, P3Settings, block_done, block_result
from roboscript_robotwin.eval.jobs import Gated
from roboscript_robotwin.eval.launch import CLOSED, host_summaries

OFFICIAL_EPISODES: Final = 100
"""The episodes RoboTwin's client runs per block (`_result.txt` holds successes / 100)."""
INFRA_FLAG: Final = 0.05
_BLOCK: Final = re.compile(r"block\d+")
_MODE: Final = (
    "successes",
    "n",
    "n_official",
    "n_host",
    "host_ended",
    "running_ended",
    "running_successes",
)


@dataclass(frozen=True)
class P3Labels:
    """How the report names the units the caller sets apart: `zeroed` (in the headline: those
    counted 0), `not_counted` (in the headline) and `cell` (before the table cells of a unit whose
    P3 does not count)."""

    zeroed: str = "stopped before P3 counted 0"
    not_counted: str = "not counted"
    cell: str = "P3 (not counted)"


class P3Report:
    """RoboTwin P3's `ReportPart`, under `p3`: of the jobs that are units (`units`), with
    `gated` (a unit whose P3 does not count), `zeroed` (one stopped before P3 with nothing to
    evaluate: counted 0), `stopped` (a unit's row whose task stopped before it: counted 0) and
    the `labels` of those."""

    def __init__(
        self,
        p3: P3Settings,
        *,
        units: Callable[[RunKey], bool],
        gated: Gated,
        zeroed: Gated,
        stopped: Callable[[Mapping[str, Any], list[Row]], bool],
        labels: P3Labels | None = None,
    ) -> None:
        self.p3 = p3
        self.labels = labels = labels or P3Labels()
        self.units = units
        self.gated = gated
        self.zeroed = zeroed
        self.stopped = stopped
        self.readings = {"p3": HOST}
        self.taxonomy_labels: dict[str, str] = {}
        self.status_columns = (
            Column(
                "P3 ok/n",
                lambda row, _: "-" if row.get("p3") is None else p3_cell(row["p3"], labels.cell),
            ),
        )
        self.summary_columns = (
            Column(
                "P3 rate [95%] (n: official, host; host saw end)",
                lambda row, _: _p3_rate(row, labels.cell),
            ),
        )

    def episodes(self, job: JobView) -> Iterable[tuple[str, Path]]:
        if not self.units(job.key):
            return []
        found = sorted((job.run_dir / "P3").glob("*/*/host/*/ep*/config.json"))
        return [("p3", path.parent) for path in found]

    def section(self, job: JobView, deep: bool) -> Section | None:
        if not self.units(job.key):
            return None
        run_dir, task = job.run_dir, TaskName(job.key.task)
        p3 = _p3(run_dir / "P3", deep, self.p3.episodes_of)
        zeroed = self.zeroed(run_dir, task)
        p3["counted"] = zeroed or not self.gated(run_dir, task)
        if zeroed:  # counted 0
            p3["zeroed"] = True
        fields: Row = {"p3": p3}
        runs = job.state  # the job that runs the randomized blocks
        if self.p3.parallel:
            lane = RunKey(job.key.task, job.key.variant, SPARE)
            runs = RunState.load(run_dir / LANE_DIRS[SPARE], lane)
            fields["spare"] = {
                "run": lane.name,
                "status": runs.status,
                "gpu": runs.gpu,
                "launches": runs.launches,
                "respawns": runs.respawns,
                "running_s": round(runs.running_s(job.now), 1),
                "failure": runs.failure,
                "error": runs.error,
            }
        if self.p3.spare and not self.gated(run_dir, task):
            p3["spare_waiting"] = _spare_waiting(run_dir / "P3", self.p3, runs)
        return Section(fields)

    def summary(self, rows: list[Row]) -> Row:
        return {"p3_overall": self.overall(rows)}

    def flags(self, rows: list[Row]) -> list[str]:
        return _infra_flags(rows)

    def headline(self, found: Mapping[str, Any]) -> list[str]:
        """Each mode's rate over the units (`p3_overall`)."""
        lines = []
        for mode, m in (found.get("p3_overall") or {}).items():
            mean = "-" if m["mean_rate"] is None else f"{m['mean_rate']:.1%}"
            line = (
                f"P3 {mode}: {m['successes']}/{m['n']} over {m['variants']} variants "
                f"(mean {mean}; {len(m['zeroed'])} {self.labels.zeroed}"
            )
            if m["pending"]:
                line += f"; not yet in: {', '.join(m['pending'])}"
            if m["not_counted"]:
                line += f"; {self.labels.not_counted}: {', '.join(m['not_counted'])}"
            lines.append(f"- {line})")
            if m["pending"] or m["not_counted"]:
                every = m["all"]
                shown = "-" if every["rate"] is None else f"{every['rate']:.1%}"
                lines.append(
                    f"- P3 {mode} over all {every['variants']} variants, those not yet in or not "
                    f"counted as 0: {every['successes']}/{every['n']} ({shown})"
                )
        return lines

    def notes(self, found: Mapping[str, Any]) -> list[str]:
        return []

    def lanes(self, row: Mapping[str, Any]) -> list[Lane]:
        """The spare lane's job, shown once it was launched or left the queue."""
        if not (lane := row.get("spare")):
            return []
        shown = bool(lane["launches"] or lane["status"] != "queued")
        return [Lane(lane, "p3 randomized", shown)]

    def chain(self, row: Mapping[str, Any], rows: list[Row]) -> list[Row] | None:
        return []

    def overall(self, rows: list[Row]) -> dict[str, Any]:
        """Each P3 mode's rate over the pipeline's units (successes over episodes, pooled; and
        the mean of the units' rates): a unit whose task stopped before it (`stopped`) or that
        stopped before P3 (`zeroed`) counts 0 of every episode its blocks would run; one whose
        blocks of the mode are not all done (`pending`), or whose P3 does not count
        (`not_counted`: `gated`), is left out of `rate` and named, and `all` is the rate over
        every unit with those counted 0 of every episode too (the headline once nothing is
        pending)."""
        found: dict[str, Any] = {}
        for mode in self.p3.modes:
            k = n = 0
            rates: list[float] = []
            zeroed, pending_runs, not_counted = [], [], []
            full = self.p3.episodes_of(mode) * len(self.p3.blocks)
            for row in rows:
                if "p3" not in row:
                    continue
                if self.stopped(row, rows) or row["p3"].get("zeroed"):
                    n += full
                    rates.append(0.0)
                    zeroed.append(row["run"])
                    continue
                if not row["p3"]["counted"]:
                    not_counted.append(row["run"])
                    continue
                blocks = [
                    b for name, b in row["p3"]["blocks"].items() if name.startswith(f"{mode}/")
                ]
                stats = row["p3"]["modes"].get(mode)
                if len(blocks) < len(self.p3.blocks) or not all(b["done"] for b in blocks):
                    pending_runs.append(row["run"])
                    continue
                if not stats or not stats["n"]:
                    pending_runs.append(row["run"])
                    continue
                k += stats["successes"]
                n += stats["n"]
                rates.append(stats["successes"] / stats["n"])
            rest = full * (len(pending_runs) + len(not_counted))
            found[mode] = {
                "successes": k,
                "n": n,
                "rate": k / n if n else None,
                "all": {
                    "successes": k,
                    "n": n + rest,
                    "rate": k / (n + rest) if n + rest else None,
                    "variants": len(rates) + len(pending_runs) + len(not_counted),
                },
                "mean_rate": sum(rates) / len(rates) if rates else None,
                "variants": len(rates),
                "zeroed": zeroed,
                "pending": pending_runs,
                "not_counted": not_counted,
            }
        return found


def _infra_flags(rows: list[Row]) -> list[str]:
    """The P3 blocks that count with more than `INFRA_FLAG` of their episodes failed by
    infrastructure."""
    flags = []
    for row in rows:
        for name, block in (row.get("p3") or {"blocks": {}})["blocks"].items():
            seen, infra = block["host_seen"], block["infra"]
            if block["done"] and seen and infra / seen > INFRA_FLAG:
                flags.append(
                    f"P3 {row['run']} {name}: {infra} of {seen} episodes failed by infrastructure"
                )
    return flags


def _p3(folder: Path, deep: bool, wanted: Callable[[str], int]) -> dict[str, Any]:
    """Each block (set-aside ones left out) and each mode's rate: a complete block's official
    result, a block cut on purpose after its mode's count (`wanted`, `p3.episodes_of`) by the
    host's counts, every episode it finished (`closed`: closed from a set-aside run, nothing run;
    `closed.json`); a running one's progress (and a block cut shorter, which runs again)."""
    blocks: dict[str, dict[str, Any]] = {}
    for out in sorted(folder.glob("*/block*")):
        if not _BLOCK.fullmatch(out.name):
            continue
        result, done = block_result(out), block_done(out, wanted(out.parent.name))
        # the summaries that count: not those of attempts lost before (`launch.host_summaries`)
        episodes = [e for p in host_summaries(out) if (e := read_json(p))]
        ended = [e for e in episodes if e["official"] is not None]
        complete = done and result is not None and not result.truncated
        blocks[f"{out.parent.name}/{out.name}"] = {
            "done": done,
            "complete": complete,
            "official_rate": result.success_rate if complete and result is not None else None,
            "host_seen": len(episodes),
            "host_ended": len(ended),
            "host_successes": sum(bool(e["official"]["success"]) for e in ended),
            "infra": sum(infra_lost(e) for e in episodes),
            "client_exit": None if result is None else result.client_exit,
            "closed": (out / CLOSED).is_file(),
        }
    modes: dict[str, dict[str, Any]] = {}
    for name, block in blocks.items():
        mode = modes.setdefault(name.split("/")[0], dict.fromkeys(_MODE, 0))
        if block["complete"]:
            n = OFFICIAL_EPISODES
            mode["successes"] += round(block["official_rate"] * n)
            mode["n"] += n
            mode["n_official"] += n
            mode["host_ended"] += block["host_ended"]
        elif block["done"]:  # cut on purpose: the host's counts
            mode["successes"] += block["host_successes"]
            mode["n"] += block["host_ended"]
            mode["n_host"] += block["host_ended"]
            mode["host_ended"] += block["host_ended"]
        else:  # running, or cut short and to run again
            mode["running_ended"] += block["host_ended"]
            mode["running_successes"] += block["host_successes"]
    for mode in modes.values():
        n, k = mode["n"], mode["successes"]
        mode["rate"] = k / n if n else None
        if deep:
            mode["wilson95"] = list(wilson(k, n)) if n else None
    return {"blocks": blocks, "modes": modes}


def _spare_waiting(folder: Path, p3: P3Settings, state: RunState) -> list[str]:
    """The randomized blocks not done and not running now: under `p3.randomized: spare`, they
    wait for spare GPU room."""
    waiting = []
    for block in p3.blocks:
        out = folder / "randomized" / f"block{block}"
        running = state.status == "running" and out.is_dir()
        if not running and not block_done(out, p3.episodes_of("randomized")):
            waiting.append(f"randomized/block{block}")
    return waiting


def p3_cell(p3: Mapping[str, Any], not_counted: str = P3Labels.cell) -> str:
    """A unit's P3 in the status table: each mode's successes so far, and what waits (after
    `not_counted` when its P3 does not count)."""
    shown = []
    for mode, m in p3["modes"].items():
        cell = f"{mode} {m['successes']}/{m['n']}"
        if m["running_ended"]:
            cell += f" +{m['running_successes']}/{m['running_ended']} running"
        shown.append(cell)
    if waiting := p3.get("spare_waiting"):
        shown.append(f"{', '.join(waiting)} waiting for spare room")
    if not p3.get("counted", True):
        return f"{not_counted}: {'; '.join(shown) or '-'}"
    return "; ".join(shown) or "-"


def _p3_rate(row: Mapping[str, Any], not_counted: str) -> str:
    """A unit's P3 in the summary table: each mode's rate, its n and the episodes the host saw
    end ("-": not a unit)."""
    if row.get("p3") is None:
        return "-"
    p3 = "; ".join(
        f"{mode} {rate(m['rate'], m['wilson95'])} "
        f"(n {m['n_official']}, {m['n_host']}; {m['host_ended']})"
        for mode, m in row["p3"]["modes"].items()
    )
    if not row["p3"]["counted"]:
        p3 = f"{not_counted}: {p3 or '-'}"
    return p3 or "-"
