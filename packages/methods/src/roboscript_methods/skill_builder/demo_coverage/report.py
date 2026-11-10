"""`coverage.md` from the per-task reports a benchmark's glue writes (one JSON file per task): the
tasks covered, the gaps by cause, which skills explain which segments.

A task is covered when every segment of its demonstration is explained and the ceiling latched
the official success in every repeat. A gap is a segment no skill fits (`no skill fits`), one
whose call ran and ended off the segment's end (`fit out of tolerance`), one whose call was
refused or failed hard (`entry fails`), or a task whose ceiling did not latch success in every
repeat (`success not reached`); a segment whose replay or call the benchmark ended is
`not judged`.
"""

import collections
import json
from pathlib import Path
from typing import Any, Final

CAUSES: Final = (
    "no skill fits",
    "fit out of tolerance",
    "entry fails",
    "success not reached",
    "not judged",
)


def gaps(report: dict[str, Any]) -> list[dict[str, Any]]:
    """A task report's gaps: its segments' and its ceiling's."""
    found = []
    for call in report["calls"]:
        if call["skill"] is None:
            found.append({"segment": call["segment"], "cause": "no skill fits", "why": call["why"]})
    for row in report["explained"]:
        if row["explained"] or row.get("applicable") is False:  # a chain's segment not reached
            continue
        cause = row["cause"]
        if cause in ("replay ended", "the benchmark ended the episode"):
            cause = "not judged"
        found.append({"segment": row["segment"], "cause": cause, "why": why(row)})
    runs = report["ceiling"]
    if not runs or not all(run["success"] for run in runs):
        latched = sum(run["success"] for run in runs)
        found.append(
            {"segment": "the task", "cause": "success not reached", "why": f"{latched}/{len(runs)}"}
        )
    return found


def why(row: dict[str, Any]) -> str:
    """A failed explanation in a line: its receipts and what truth measured."""
    parts = []
    for key in ("grasp", "lift", "place", "release", "move", "home"):
        if key in row:
            parts.append(f"{key} {row[key]['reason']}: {row[key]['note'][:120]}")
    for key in ("rose_m", "off_m", "dz_m", "dropped_m", "pos_m", "axis_m", "deg"):
        if key in row:
            parts.append(f"{key} {row[key]}")
    if row.get("ended"):
        parts.append(f"ended: {row['ended']}")
    return "; ".join(parts)


def render(directory: Path) -> str:
    """`coverage.md` for the reports in `directory`."""
    reports = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("*.json"))
        if not p.name.endswith(".reuse.json")  # the reuse step's, beside them
    }
    lines = ["# Coverage (P1-3)", "", f"Reports: `{directory}`.", ""]
    lines += ["| task | segments explained | ceiling latched | gaps |", "|---|---|---|---|"]
    by_cause: dict[str, list[str]] = collections.defaultdict(list)
    by_skill: dict[str, list[str]] = collections.defaultdict(list)
    covered = []
    for task, report in reports.items():
        found = gaps(report)
        explained = [r for r in report["explained"] if r["explained"]]
        runs = report["ceiling"]
        latched = sum(run["success"] for run in runs)
        lines.append(
            f"| {task} | {len(explained)}/{len(report['calls'])} | {latched}/{len(runs)} "
            f"| {len(found)} |"
        )
        for gap in found:
            by_cause[gap["cause"]].append(f"{task}: {gap['segment']}: {gap['why']}")
        for row in explained:
            by_skill[row["skill"]].append(f"{task}: {row['segment']}")
        if not found:
            covered.append(task)
    lines += ["", f"**Covered:** {len(covered)} of {len(reports)}: {', '.join(covered) or 'none'}."]
    lines += ["", "## Gaps by cause", ""]
    for cause in CAUSES:
        entries = by_cause.get(cause, [])
        lines += [f"### {cause} ({len(entries)})", ""]
        lines += [f"- {entry}" for entry in entries] or ["- none"]
        lines.append("")
    lines += ["## Skills and the segments they explain", ""]
    for skill, entries in sorted(by_skill.items()):
        lines += [f"### {skill} ({len(entries)})", ""]
        lines += [f"- {entry}" for entry in entries]
        lines.append("")
    return "\n".join(lines)


def render_reuse(directory: Path) -> str:
    """The reuse step (`<task>.reuse.json` beside the coverage reports): per task the gaps
    tried, those an alternative closed and how, the ceiling with the closed ones; then every gap
    left open, which the next step (a new skill) takes."""
    reports = {
        p.name.removesuffix(".reuse.json"): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("*.reuse.json"))
    }
    lines = ["# Gap filling (P1-4): reuse", "", f"Reports: `{directory}`.", ""]
    lines += ["| task | gaps tried | closed | ceiling latched |", "|---|---|---|---|"]
    by_form: dict[str, list[str]] = collections.defaultdict(list)
    still: list[str] = []
    totals = [0, 0, 0]
    for task, report in reports.items():
        tried = sorted({t["segment"] for t in report["tries"]})
        closed = report["closed"]
        runs = report["ceiling"]
        latched = sum(run["success"] for run in runs)
        lines.append(f"| {task} | {len(tried)} | {len(closed)} | {latched}/{len(runs)} |")
        totals[0] += len(tried)
        totals[1] += len(closed)
        totals[2] += latched == len(runs) and bool(runs)
        for gap in closed:
            by_form[f"{gap['skill']} ({gap['form'].split()[0]})"].append(
                f"{task}: {gap['segment']}"
            )
        done = {gap["segment"] for gap in closed}
        for segment in tried:
            if segment not in done:
                last = [t for t in report["tries"] if t["segment"] == segment][-1]
                still.append(f"{task}: {segment}: last {last['form']}: {why(last)[:200]}")
    lines += [
        "",
        f"**{totals[1]} of {totals[0]} gaps tried closed by reuse; {totals[2]} of {len(reports)} "
        "tasks latch success in every repeat after it.**",
        "",
        "## Closed, by the entry and form that closed them",
        "",
    ]
    for form, entries in sorted(by_form.items()):
        lines += [f"### {form} ({len(entries)})", ""] + [f"- {e}" for e in entries] + [""]
    lines += [f"## Still open ({len(still)})", ""] + [f"- {e}" for e in still]
    return "\n".join(lines)
