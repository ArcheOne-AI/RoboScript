"""Diff two runs' comparison streams (`Tape.observe`): the baseline's and a replay's on other code.

Each scope (a pipeline job, `<task>/<dir>/<phase>`) is compared record by record, in order; the
first record that differs is reported with the records before it and a diff of the two, and
labeled with what it tells:

- `env_open`: the env opened the episode differently (its expert pre-run, the instruction): the
  simulator, before anything of ours ran;
- `env_answer`: the env answered the same request differently: the simulator;
- `env_request`: our code sent the env something else (joint waypoints, an observation request);
- `tool_result`: a tool returned something else to the model;
- `memory`: a Memory edit or version differs;
- `episode`: an episode opened with other tools, stage, tier or role;
- `length`: one stream ended before the other.

A replay's misses (`misses/`) and the recorded answers it left unserved (`summary.*.json`) are
reported too: either means the runs did not take the same path.

What an agent does after an action stopped at the success latch, in an episode of a phase whose
episodes an outside evaluation client ends (`client_phases`, the method's: wicl's P3), is not
compared (`after_latch`): the episode's outcome is latched, and the client's `end` races the
agent's next calls, so how many of them still ran depends on timing. Those records of the agent
plane and the tool results are cut until the next episode opens (the control plane's `judge` and
the Memory's records stay), and the recorded answers a replay left unserved in a scope that
had such calls are a note, not a problem.
"""

import json
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from roboscript.records.replay.tape import diff_lines, merged_lines


@dataclass(frozen=True)
class Divergence:
    scope: str
    index: int
    label: str
    report: str


@dataclass
class Comparison:
    """`divergences`: the first per scope that differs; `problems`: scopes on one side only,
    replay misses and unserved answers."""

    scopes: list[str] = field(default_factory=list)
    records: int = 0
    divergences: list[Divergence] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    kinds: Counter[str] = field(default_factory=Counter)
    """Records compared equal, by kind (before the first divergence of each scope)."""

    @property
    def equal(self) -> bool:
        return not self.divergences and not self.problems

    def text(self) -> str:
        head = f"{len(self.scopes)} scopes, {self.records} records compared: " + (
            "identical" if self.equal else "DIFFERENT"
        )
        kinds = ", ".join(f"{kind} {n}" for kind, n in sorted(self.kinds.items()))
        parts = [head, f"equal records by kind: {kinds}", *self.problems]
        parts += [d.report for d in self.divergences]
        parts += self.notes
        return "\n\n".join(parts)


def compare(baseline: Path, replay: Path, *, client_phases: Collection[str]) -> Comparison:
    """Compare the comparison streams of two tape directories (`RS_REPLAY_DIR`s); in episodes of
    `client_phases` the records after a success latch are cut (`after_latch`)."""
    found = Comparison()
    a, b = scopes(baseline), scopes(replay)
    for scope in sorted(a ^ b):
        side = "baseline" if scope in a else "replay"
        found.problems.append(f"scope {scope or '-'} is in the {side} only")
    latched: dict[str, int] = {}
    for scope in sorted(a & b):
        found.scopes.append(scope)
        left, cut_left = after_latch(merged_lines(baseline / scope, "compare"), client_phases)
        right, cut_right = after_latch(merged_lines(replay / scope, "compare"), client_phases)
        if cut_left or cut_right:
            latched[scope] = cut_left
            found.notes.append(
                f"scope {scope}: after the success latch, {cut_left} recorded and {cut_right} "
                "replayed records not compared"
            )
        found.records += min(len(left), len(right))
        divergence = first_divergence(scope, left, right)
        equal = len(left) if divergence is None else divergence.index
        found.kinds.update(record["kind"] for record in left[:equal])
        if divergence is not None:
            found.divergences.append(divergence)
    misses, unserved = replay_problems(replay)
    found.problems += misses
    for scope, text in unserved:
        (found.notes if latched.get(scope) else found.problems).append(text)
    return found


AGENT_PLANE: Final = ("env.agent.", "tool")


def after_latch(
    records: list[dict[str, Any]], client_phases: Collection[str]
) -> tuple[list[dict[str, Any]], int]:
    """`records` without the agent plane's and the tools' records that follow an `execute` which
    stopped at the success latch in an episode of `client_phases` (wicl's P3), until the next
    episode opens; and how many were cut. (Phases that end an episode at the latch themselves,
    wicl's S1 and S2, have nothing racing them.)"""
    kept, cut, latched, raced = [], 0, False, False
    for record in records:
        kind = record["kind"]
        if kind == "episode":
            latched, raced = False, record["data"].get("stage") in client_phases
        if latched and kind.startswith(AGENT_PLANE):
            cut += 1
            continue
        kept.append(record)
        ok = record["data"].get("ok") if isinstance(record["data"], dict) else None
        if raced and kind == "env.agent.execute" and isinstance(ok, dict):
            latched = ok.get("stopped") == "success"
    return kept, cut


def scopes(root: Path) -> set[str]:
    """The scopes under a tape directory that hold a comparison stream ('' for the root)."""
    return {
        "" if path.parent == root else str(path.parent.relative_to(root))
        for path in root.rglob("compare.*.jsonl")
    }


def first_divergence(
    scope: str, left: list[dict[str, Any]], right: list[dict[str, Any]], context: int = 3
) -> Divergence | None:
    """The first record where two streams differ, with the `context` records before it."""
    for index in range(max(len(left), len(right))):
        a = _content(left[index]) if index < len(left) else None
        b = _content(right[index]) if index < len(right) else None
        if a == b:
            continue
        label = "length" if a is None or b is None else _label(a, b)
        lines = [
            f"scope {scope or '-'}: first divergence at record {index} ({label})",
            f"  in {_episode(left, index)}",
        ]
        for before in range(max(0, index - context), index):
            lines.append(f"  [{before}] {_summary(_content(left[before]))}")
        if a is None or b is None:
            longer, side = (right, "replay") if a is None else (left, "baseline")
            lines.append(f"  the {side} goes on with {len(longer) - index} records, from:")
            lines.append(f"  [{index}] {_summary(_content(longer[index]))}")
        else:
            lines.append(f"  [{index}] {_summary(a)}  vs  {_summary(b)}")
            lines.extend("  " + line for line in diff_lines(a, b))
        return Divergence(scope, index, label, "\n".join(lines))
    return None


def replay_problems(replay: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """A replay's misses, and the recorded answers it left unserved by scope."""
    misses = []
    for miss in sorted(replay.rglob("misses/*.json")):
        report = json.loads(miss.read_text(encoding="utf-8"))
        misses.append(f"replay miss ({miss.relative_to(replay)}):\n{report.get('report', report)}")
    unserved = []
    for path in sorted(replay.rglob("summary.*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("unserved"):
            where = str(path.parent.relative_to(replay))
            text = f"scope {where}: recorded answers never asked for: {summary['unserved']}"
            unserved.append((where, text))
    return misses, unserved


def _content(record: dict[str, Any]) -> dict[str, Any]:
    return {"kind": record["kind"], "data": record["data"]}


def _label(a: dict[str, Any], b: dict[str, Any]) -> str:
    kind = a["kind"]
    if kind != b["kind"]:
        return "kind"
    if kind == "env.control.open":
        return "env_open"
    if kind.startswith(("env.agent.", "env.control.")):
        same = a["data"].get("request") == b["data"].get("request")
        return "env_answer" if same else "env_request"
    if kind == "tool":
        return "tool_result"
    if kind.startswith("memory."):
        return "memory"
    return str(kind)


def _episode(records: list[dict[str, Any]], index: int) -> str:
    """The episode the record at `index` belongs to: the last `episode` record before it."""
    for record in reversed(records[: index + 1]):
        if record["kind"] == "episode":
            data = record["data"]
            return f"episode {data.get('stage')}/{data.get('role')} (tier {data.get('tier')})"
    return "no episode"


def _summary(content: dict[str, Any] | None) -> str:
    if content is None:
        return "(nothing)"
    data = content["data"]
    name = data.get("name") if isinstance(data, dict) else None
    return str(content["kind"]) + (f" {name}" if name else "")
