"""The builder driver (`skill_builder/builder`: `clusters.py`, `driver.py`) without a simulator
or a model: the paired regression blames the candidates a losing task runs, and a run's rounds
with a fake benchmark and a fake session gate what the session wrote, accept, rerun and stop."""

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roboscript_methods.skill_builder.builder.clusters import load
from roboscript_methods.skill_builder.builder.driver import (
    Cost,
    Driver,
    SessionContext,
    SessionOutcome,
    regress,
    state,
    written,
)

pytestmark = pytest.mark.unit


def receipt(reason: str) -> dict[str, Any]:
    return {"reason": reason, "ok": reason == "ok", "steps": 3, "note": f"{reason} note"}


def row(
    segment: str,
    explained: bool,
    cause: str | None = None,
    skill: str = "place_object",
    **receipts: str,
) -> dict[str, Any]:
    return {"segment": segment, "skill": skill, "explained": explained, "cause": cause} | {
        k: receipt(v) for k, v in receipts.items()
    }


def call(segment: str, kind: str, skill: str | None = "place_object") -> dict[str, Any]:
    start = int(segment.rsplit("[", 1)[1].split(",")[0]) if "[" in segment else 0
    return {
        "segment": segment,
        "arm": "left",
        "kind": kind,
        "frames": [start, start + 9],
        "obj": "thing",
        "target": None,
        "skill": skill,
        "form": "pose",
        "why": "w",
    }


def write_task(
    directory: Path,
    task: str,
    calls: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    ceiling: list[bool],
    tries: list[dict[str, Any]] | None = None,
    closed: list[dict[str, Any]] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    runs = [{"success": s} for s in ceiling]
    report = {"task": task, "calls": calls, "explained": rows, "ceiling": runs}
    reuse = {
        "task": task,
        "tries": tries or [],
        "closed": closed or [],
        "calls": calls,
        "ceiling": runs,
    }
    (directory / f"{task}.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / f"{task}.reuse.json").write_text(json.dumps(reuse), encoding="utf-8")


def test_regress_blames_the_candidates_a_losing_task_runs(tmp_path: Path) -> None:
    seg = "left place thing [0, 9]"
    write_task(tmp_path / "b", "t", [call(seg, "place")], [row(seg, True)], [True, True])
    before = state(load(tmp_path / "b"))
    write_task(
        tmp_path / "a",
        "t",
        [call(seg, "place", "new_place")],
        [row(seg, False, "entry fails", "new_place", place="lost_hold")],
        [True, True],
    )
    kinds = {"new_place": ["place"], "other": ["place"]}
    losses, dropped = regress(before, load(tmp_path / "a"), kinds)
    assert dropped == {"new_place"} and losses[0]["blamed"] == ["new_place"]
    _, none = regress(before, load(tmp_path / "b"), kinds)
    assert none == set()
    # a task missing from the rerun is not judged (infrastructure), unless its job was stopped
    # inside a candidate's act
    kinds2 = {"new_place": ["place"], "a_grasp": ["grasp"]}
    losses, dropped = regress(before, {}, kinds2)
    assert dropped == set() and losses[0]["not_judged"] and losses[0]["blamed"] == []
    losses, dropped = regress(before, {}, kinds2, {"t": {"stuck": "new_place"}})
    assert dropped == {"new_place"} and losses[0]["stuck"] == "new_place"
    # ceiling repeats compared over judged ones: one infra repeat does not hide a loss
    write_task(tmp_path / "c", "t", [call(seg, "place")], [row(seg, True)], [False, True])
    report = json.loads((tmp_path / "c" / "t.reuse.json").read_text(encoding="utf-8"))
    report["ceiling"][1]["infra"] = True
    (tmp_path / "c" / "t.reuse.json").write_text(json.dumps(report), encoding="utf-8")
    _, dropped = regress(before, load(tmp_path / "c"), {"place_object": ["place"]})
    assert dropped == {"place_object"}


SKILL = """
NAME = "{name}"
{extra}
"""


def write_skill(root: Path, name: str, cases: list[str], extra: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "skill.py").write_text(textwrap.dedent(SKILL.format(name=name, extra=extra)), "utf-8")
    (d / "fit.py").write_text(
        "def fit(segment, recording, scene, base, poses):\n    return {}\n", encoding="utf-8"
    )
    (d / "cases.yaml").write_text(json.dumps({"gaps": cases}), encoding="utf-8")
    return d


GAP_A = "task_a/left place thing [0, 9]"
SEG_A = "left place thing [0, 9]"
SEG_B = "left place thing [5, 14]"


def is_fixed(skill: Path) -> bool:
    return skill.is_dir() and "# fixed" in (skill / "skill.py").read_text(encoding="utf-8")


class FakeBench:
    """task_a has a place out of reach that `good_place` closes; task_b is explained and covered,
    and loses its segment when `bad_place` is in the sweep (its calls then run it). The validator
    fails a skill that names a task; `idle_place` passes its case and is used by no task."""

    case_slots = 4

    def __init__(self, root: Path):
        self.root, self.sweeps, self.cases_run, self.accepted = root, [], [], []
        self.validated: list[str] = []
        self.priors: list[tuple[str, str | None]] = []
        self.reused: list[tuple[str, bool]] = []
        self.earlier: list[list[str]] = []
        self.transfers: list[tuple[str, str]] = []
        self.off = [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] * 10

    def tasks(self) -> list[str]:
        return ["task_a", "task_b"]

    async def sweep(
        self,
        out: Path,
        skills: list[Path],
        tasks: list[str] | None = None,
        prior: Path | None = None,
        earlier: Any = (),
        plans: Path | None = None,
        reuse_before: bool = False,
    ) -> Cost:
        self.priors.append((out.name, prior and prior.name))
        self.reused.append((out.name, reuse_before))
        self.earlier.append([f"{p.parent.name}/{p.name}" for p in earlier])
        names = {p.name for d in skills for p in written(Path(d))}
        self.sweeps.append((out.name, sorted(names)))
        closer = next((n for n in ("good_place", "slow_place") if n in names), None)
        closed = [{"segment": SEG_A, "skill": closer, "form": "written"}] if closer else []
        if tasks is None or "task_a" in tasks:
            write_task(
                out,
                "task_a",
                [call(SEG_A, "place")],
                [row(SEG_A, False, "entry fails", place="unreachable") | {"off_m": 0.1}],
                [True, True] if closer else [False, False],
                closed=closed,
            )
        if tasks is not None and "task_b" not in tasks:
            pass
        elif "bad_place" in names and self.breaks(out):
            write_task(
                out,
                "task_b",
                [call(SEG_B, "place", "bad_place")],
                [row(SEG_B, False, "entry fails", "bad_place", place="lost_hold")],
                [True, True],
            )
        else:
            write_task(out, "task_b", [call(SEG_B, "place")], [row(SEG_B, True)], [True, True])
        return Cost(slot_s=3600.0, jobs=2)

    async def case(self, gap: str, skill: Path, out: Path) -> dict[str, Any]:
        self.cases_run.append((gap, skill.name))
        good = skill.name in ("good_place", "bad_place", "idle_place") or is_fixed(skill)
        return {"gap": gap, "explained": good and gap == GAP_A, "off_m": self.off.pop(0)}

    def breaks(self, out: Path) -> bool:
        """Whether `bad_place` breaks task_b in the sweep into `out`."""
        return True

    async def transfer(
        self, gap: str, skill: Path, out: Path, demos: list[int] | None = None
    ) -> list[dict[str, Any]]:
        self.transfers.append((gap, skill.name))
        replay = skill.name == "replay_place"  # explains its own scene only
        if replay:  # two other demonstrations, neither explained: bound to its scene
            return [{"demo": d, "explained": False, "infra": False} for d in (900001, 900002)]
        return [{"demo": 900001, "explained": True, "infra": skill.name == "flaky_place"}]

    async def core_transfer(self, gap: str, out: Path) -> list[dict[str, Any]]:
        """The core's own entry transfers to both other demonstrations."""
        self.transfers.append((gap, "core"))
        return [{"demo": d, "explained": True} for d in (900001, 900002)]

    async def validate(self, skill: Path) -> dict[str, Any]:
        self.validated.append(str(skill))
        names_a_task = "task_a" in (skill / "skill.py").read_text(encoding="utf-8")
        cases = json.loads((skill / "cases.yaml").read_text(encoding="utf-8"))["gaps"]
        findings = ["skill.py:3: [task] names the task 'task_a'"] if names_a_task else []
        return {"ok": not findings, "name": skill.name, "kinds": ["place"], "cases": cases,
                "findings": findings}  # fmt: skip

    def accept(self, skill: Path, provenance: dict[str, Any]) -> Path:
        self.accepted.append((skill.name, provenance))
        return self.root / "written" / skill.name

    def describe(self, out: Path, skills: list[Path]) -> list[str]:
        return [f"sweep {out}", f"case {out}"]


def request(n: int, gap: str, skill: Path) -> SimpleNamespace:
    return SimpleNamespace(n=n, gap=gap, skill_dir=skill)


async def fake_session(
    found: Any, work: Path, case: Any, accepted: list[Path], context: SessionContext
) -> SessionOutcome:
    """Writes four skills: one that closes the gap, one that also breaks task_b, one that names
    a task, one used by no task; asks for one case run."""
    assert found.id == "place-reach" and context.reports.name == "coverage"
    write_skill(work / "skill", "good_place", [GAP_A])
    write_skill(work / "skill", "bad_place", [GAP_A, "task_z/not an open gap [0, 1]"])
    write_skill(work / "skill", "named_place", [GAP_A], extra="# made for task_a")
    write_skill(work / "skill", "idle_place", [GAP_A])
    await case(request(1, GAP_A, work / "skill" / "good_place"))
    return SessionOutcome(status="done", tokens=1000, case_runs=1)


async def test_rounds_gate_accept_rerun_and_stop(tmp_path: Path) -> None:
    # both tasks latch both repeats once `good_place` is in: two runs are no confirmed ceiling,
    # so no task is at its target (`met`), and the stop rule's other half, nothing grew, is
    # what ends this run
    bench = FakeBench(tmp_path)
    run = tmp_path / "runs" / "r"
    driver = Driver(run, bench, session=fake_session)
    await driver.run_all()

    r0 = json.loads((run / "round-0" / "summary.json").read_text(encoding="utf-8"))
    assert r0["accepted"] == ["good_place", "idle_place"]  # one no task used: kept (coverage)
    reasons = {r["name"]: r["reason"] for r in r0["rejected"]}
    assert reasons == {
        "bad_place": "regression: net -1.0 (lost task_b; gained -)",
        "named_place": "static check",
    }
    # a segment explained, and the ceiling 0/2 -> 2/2: its rate weighs `FIRST_REPEATS` (4) runs
    assert r0["ledger"]["rerun"]["good_place"]["gains"] == {"task_a": 5.0}
    assert r0["state"] == "after"  # a candidate dropped: rerun with the accepted only
    after = r0["tasks"]["task_a"]["after"]
    assert (after["explained"], after["latched"], after["judged"]) == (1, 2, 2)
    assert r0["totals"]["ceiling_all_before"] == 1 and r0["totals"]["ceiling_all_after"] == 2
    assert r0["cost"]["tokens"] == 1000
    assert [c["id"] for c in r0["clusters"]] == ["place-reach"]
    rerun = ["bad_place", "good_place", "idle_place"]
    assert bench.sweeps == [
        ("coverage", []),
        ("rerun", rerun),
        ("rerun-confirm-before", []),  # task_b lost: both arms again, paired
        ("rerun-confirm-after", rerun),
        ("after", ["good_place", "idle_place"]),  # the kept ones' own sweep, compared first
    ]
    gate = json.loads((run / "round-0" / "gate.json").read_text(encoding="utf-8"))
    bad = next(v for v in gate.values() if v["name"] == "bad_place")
    assert [c["gap"] for c in bad["cases"]] == [GAP_A]  # the case that is no open gap: ignored
    # validated and run from the snapshot, never from the workspace
    assert all("/gate/skills/" in v for v in bench.validated)
    assert bench.transfers and all(gap == GAP_A for gap, _ in bench.transfers)
    good = next(v for v in gate.values() if v["name"] == "good_place")
    assert good["transfer"][0]["explained"] and len(good["sha256"]) == 16
    assert (run / "accepted" / "good_place" / "skill.py").is_file()
    assert [name for name, _ in bench.accepted] == ["good_place", "idle_place"]
    provenance = bench.accepted[0][1]
    assert provenance["cluster"] == "place-reach" and provenance["case_runs"] == 1
    assert r0["relays"] == [] and r0["clusters"][0]["sessions"][0]["decision"] == "candidate"

    r1 = json.loads((run / "round-1" / "summary.json").read_text(encoding="utf-8"))
    assert r1["clusters"] == [] and r1["accepted"] == []
    assert not (run / "round-2").exists()  # a round that accepts nothing ends the run

    sweeps, cases = len(bench.sweeps), len(bench.cases_run)
    await Driver(run, bench, session=fake_session).run_all()
    assert (len(bench.sweeps), len(bench.cases_run)) == (sweeps, cases)


class Relayed:
    """A first session that runs out of case runs with a skill that does not pass but gets nearer
    (off_m falls); its relay, in the same workspace, fixes the skill."""

    def __init__(self) -> None:
        self.contexts: list[SessionContext] = []

    async def __call__(
        self, found: Any, work: Path, case: Any, accepted: list[Path], context: SessionContext
    ) -> SessionOutcome:
        self.contexts.append(context)
        skill = work / "skill" / "slow_place"
        if not context.relay:
            write_skill(work / "skill", "slow_place", [GAP_A])
            for n in range(1, context.limits.case_runs + 1):
                await case(request(n, GAP_A, skill))
            return SessionOutcome(
                status="done", tokens=10, case_runs=context.limits.case_runs,
                last_message="nearly there",
            )  # fmt: skip
        assert skill.is_dir()  # the workspace the first session left
        (skill / "skill.py").write_text(
            (skill / "skill.py").read_text(encoding="utf-8") + "# fixed\n", encoding="utf-8"
        )
        return SessionOutcome(status="done", tokens=20)


async def writes(found: Any, work: Path, case: Any, accepted: list[Path], context: Any) -> Any:
    """Writes `good_place` in the first round only."""
    if context.reports.parent.name == "round-0":
        write_skill(work / "skill", "good_place", [GAP_A])
    return SessionOutcome(status="done", tokens=10)


async def nothing(found: Any, work: Path, case: Any, accepted: list[Path], context: Any) -> Any:
    return SessionOutcome(status="done")


# ---- the chained step: gaps, metrics, the stop and the re-baseline -------------------------------


S1, S2, S3 = "left grasp thing [0, 9]", "left place thing [10, 19]", "left place thing [20, 29]"


# ---- the relative rule's rewrite gaps, four repeats, smaller sessions -------------------------


# ---- rebaselining, earlier chains, split clusters and relays -------------------------------------


# ---- a round holding a speculative rerun's records resumes --------------------------------------
