"""Reading the reports `run_check` writes (one JSON file per task in a directory): every task's
metrics and all of them pooled, the gate's verdict on the pool (a benchmark's gate file), the
calls that failed by truth or were dishonest, runs set side by side (shares with their Wilson
intervals), and reports judged again by other criteria on their own measurements."""

import json
import math
from pathlib import Path
from typing import Any, Final

import yaml

from roboscript_methods.skill_builder.core_entry.check import Criteria, metrics
from roboscript_methods.skill_builder.core_entry.check import rejudge as judged_again

GROUPS: Final = ("move_to_pose", "displacement", "gripper")
COLUMNS: Final = (
    "poses_ok",
    "pose_pos_m",
    "displacement_m",
    "gripper_reached",
    "grasp_planned",
    "grasp_held",
    "grasp_success",
    "place_placed",
    "place_upright",
    "demo_picked",
    "return_home",
    "honest",
    "ended",
)


def reports(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))
    }


def pooled(found: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One report of every task's rows, as `metrics` reads them."""
    pool: dict[str, list[Any]] = {group: [] for group in (*GROUPS, "grasp")}
    for report in found.values():
        for group, rows in pool.items():
            rows += report[group] if isinstance(report[group], list) else []
    return pool


def share(text: str) -> tuple[int, int]:
    passed, of = text.split("/")
    return int(passed), int(of)


def wilson(passed: int, of: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of a share."""
    if not of:
        return (0.0, 1.0)
    p = passed / of
    centre = (p + z * z / (2 * of)) / (1 + z * z / of)
    half = z * math.sqrt(p * (1 - p) / of + z * z / (4 * of * of)) / (1 + z * z / of)
    return (max(0.0, centre - half), min(1.0, centre + half))


def value(found: dict[str, Any], metric: str) -> float | None:
    """A pooled metric's value: `name` a share's fraction, `name.median` / `name.max` an error's."""
    name, _, field = metric.partition(".")
    entry = found[name]
    if field:
        return None if entry is None else float(entry[field])
    passed, of = share(entry)
    return passed / of if of else None


def verdicts(found: dict[str, Any], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each gate on the pooled metrics: `pass`, `fail`, `unset` (no threshold) or `no data`."""
    out = []
    for gate in gates:
        got = value(found, gate["metric"])
        bound = gate.get("at_least", gate.get("at_most"))
        if bound is None:
            status = "unset"
        elif got is None:
            status = "no data"
        elif "at_least" in gate:
            status = "pass" if got >= bound - 1e-12 else "fail"
        else:
            status = "pass" if got <= bound + 1e-12 else "fail"
        out.append({**gate, "value": got, "status": status})
    return out


def load_gates(path: Path) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))["gates"]
    return gates


def summarize(directory: Path, gates: Path) -> None:
    """Every task's metrics (median/max for errors), all tasks' rows pooled, the gate's verdict
    on the pool (with the thresholds it applied), then every failed or dishonest call."""
    found = reports(directory)

    def cells(m: dict[str, Any]) -> list[Any]:
        return [
            f"{m[k]['median']}/{m[k]['max']}" if isinstance(m[k], dict) else m[k] for k in COLUMNS
        ]

    print("task", *COLUMNS, "errors", sep="\t")
    errors = 0
    for task, report in found.items():
        mine = metrics(report)  # from the rows: a report without a metric has it
        errors += len(mine["errors"])
        print(task, *cells(mine), len(mine["errors"]), sep="\t")
    pool = metrics(pooled(found))
    print("all", *cells(pool), errors, sep="\t")
    print(f"\ngate ({gates})")
    print("metric", "value", "threshold", "status", "95% interval", "source", sep="\t")
    for v in verdicts(pool, load_gates(gates)):
        name = v["metric"].partition(".")[0]
        interval = ""
        if isinstance(pool[name], str):
            low, high = wilson(*share(pool[name]))
            interval = f"[{low:.3f}, {high:.3f}]"
        bound = "at least" if "at_least" in v else "at most"
        threshold = v.get("at_least", v.get("at_most"))
        shown = "" if v["value"] is None else f"{v['value']:.4g}"
        cells = (v["metric"], shown, f"{bound} {threshold}", v["status"], interval, v["source"])
        print(*cells, sep="\t")
    print("\nfailed or dishonest (what goes back to P1-1, with the receipts)")
    for task, report in found.items():
        for line in failures(report):
            print(f"{task}\t{line}")


def failures(report: dict[str, Any]) -> list[str]:
    """A report's calls that failed by truth or were dishonest, one line each."""
    found = []
    for group in GROUPS:
        rows = report[group]
        if not isinstance(rows, list):
            found.append(f"{group} ERROR {rows['error'][:160]}")
            continue
        for row in rows:
            what = row.get("call", group)
            if "ended" in row:
                found.append(f"ENDED {what} {row.get('arm', '')}: {row['ended']}")
                continue
            if not row["honest"] or not row.get("within", True) or not row["ok"]:
                flag = "" if row["honest"] else "DISHONEST "
                off = row.get("pos_m", row.get("error_m", ""))
                arm = row.get("arm", "")
                found.append(f"{flag}{what} {arm} {row['reason']} {off} {row['note']}")
    for row in report["grasp"]:
        if "error" in row:
            found.append(f"grasp ERROR {row['error'][:160]}")
            continue
        label, size = row["label"], row.get("size_m")
        if "skipped" in row:
            found.append(f"grasp {label} skipped: {row['skipped']}")
            continue
        if "ended" in row:
            found.append(f"ENDED grasp {label}: {row['ended']}")
            continue
        if not row["planned"]:
            notes = [r["note"] for r in row.get("refused", [])]
            found.append(f"grasp {label} {size} unplanned: {notes[-1] if notes else ''}")
            continue
        if not row["held"] or not row["grasp_honest"]:
            flag = "" if row["grasp_honest"] else "DISHONEST "
            found.append(
                f"{flag}grasp {label} {size} {row['arm']}: grasp {row['grasp']['reason']}, lift "
                f"{row['lift']['reason']}, rose {row['rose_m']} m: {row['grasp']['note']}"
            )
        place = row.get("place")
        if place is not None and "ended" in place:
            found.append(f"ENDED place {label}: {place['ended']}")
            continue
        if place is not None and (not place["placed"] or not place["honest"]):
            flag = "" if place["honest"] else "DISHONEST "
            found.append(
                f"{flag}place {label} {place['reason']}: off {place['off_m']} m, dz "
                f"{place['dz_m']} m, tilt {place['tilt_deg']} deg, gripper {place['gripper']}: "
                f"{place['note']}"
            )
        origin = row.get("return_home")  # none where an end came first
        if origin is None:
            continue
        if "ended" in origin:
            found.append(f"ENDED return_home after {label}: {origin['ended']}")
        elif not origin["within"] or not origin["honest"]:
            found.append(f"return_home after {label} {origin['reason']}: {origin['note']}")
    return [line[:400] for line in found]


def rejudge(directory: Path, criteria: Criteria, picked: dict[str, list[str]], why: str) -> None:
    """The reports judged again by `criteria` on their own measurements, each task's objects
    the demonstration picked up (`picked`) marked; `rejudged` in a report says when and why."""
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report = judged_again(report, criteria, set(picked.get(report["task"], [])))
        report["rejudged"] = why
        path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def outcome(g: dict[str, Any]) -> str:
    """A grasp row in a word: `held` (truth: it rose) with `+P` / `+p-` placed or not (`+E`: the
    benchmark ended the episode in the place), `unpl` no grasp planned, `skip` not tried, `ENDED`
    ended by the benchmark, else the grasp and lift receipts' reasons."""
    if "error" in g:
        return "ERR"
    if "skipped" in g:
        return "skip"
    if "ended" in g:
        return "ENDED"
    if not g.get("planned"):
        return "unpl"
    found = "held" if g["held"] else f"{g['grasp']['reason'][:5]}/{g['lift']['reason'][:5]}"
    if "place" in g:
        found += "+E" if "ended" in g["place"] else "+P" if g["place"]["placed"] else "+p-"
    return found


def compare(runs: dict[str, Path]) -> None:
    """Runs side by side on the tasks they share: every object whose outcome differs between
    them or was not grasped and placed in the first, then each run's totals."""
    found = {name: reports(directory) for name, directory in runs.items()}
    tasks = sorted(set.intersection(*(set(r) for r in found.values())))
    first = next(iter(runs))
    print("task object".ljust(40), *(name.ljust(14) for name in runs))
    for task in tasks:
        for label in [g["label"] for g in found[first][task]["grasp"]]:
            cells = []
            for name in runs:
                g = next((x for x in found[name][task]["grasp"] if x["label"] == label), None)
                cells.append("-" if g is None else outcome(g))
            if len(set(cells)) > 1 or cells[0] != "held+P":
                print(f"{task} {label}"[:40].ljust(40), *(c.ljust(14) for c in cells))
    print(f"\n{len(tasks)} tasks; shares with their 95% Wilson intervals")
    for name, by_task in found.items():
        pool = metrics(pooled({t: by_task[t] for t in tasks}))
        keys = ("grasp_success", "demo_picked", "place_placed", "honest", "poses_ok")
        print(f"{name}: " + ", ".join(f"{k} {pool[k]} {interval(pool[k])}" for k in keys))


def interval(text: str) -> str:
    low, high = wilson(*share(text))
    return f"[{low:.3f}, {high:.3f}]"
