"""S2 `practice` on MockEnv (seed 10001 unsolvable) with scripted brains that decide from
`Episode.extra`. Early stop and freeze, merge vs jitter vs discard, infrastructure re-runs and
unsolvable opens, void tries, instruction reuse, the start from S1's Memory, crash-safe resume,
the success-only ablation, the demo truth reaching only the Reviser and only when asked, and
hooks."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

import pytest
from pydantic import ValidationError

from roboscript.agent import ScriptedBrain, call, read_lines
from roboscript.contracts.errors import InfraError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import PlanRequest, PlanResult
from roboscript.envs.mock import TABLE_Z
from roboscript.learning import DemoTruth, Episode, EpisodeOutcome, Hooks
from roboscript.learning.demo import TRUTH_FILE, TruthFrame
from roboscript.memory import (
    EditRecord,
    FreezeRecord,
    MemoryDiff,
    MergeRecord,
    VersionMeta,
    VersionSource,
)
from roboscript.motion import MockPlanner
from roboscript.testing.protocols import (
    LIFT_RED,
    NO_OBSERVE,
    RATIONALE,
    Act,
    acting,
    color_of,
    editing,
    lift_memory,
)
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import LocalProvider, ToolsetRuntime
from roboscript.toolsets.manipulation import Manipulation
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.worldstate import WorldState
from roboscript_methods.wicl.ledger import Ledger, LedgerRow
from roboscript_methods.wicl.practice import (
    Practice,
    PracticeConfig,
    PracticeResult,
)
from roboscript_methods.wicl.toolsets.done import Done

Plan = Mapping[tuple[int, int], Act]


def by_try(plan: Plan, default: Act = "fail") -> Any:
    """An Actor doing `plan[(seed, attempt)]`, else `default`."""
    return acting(lambda e: plan.get((e.extra["seed"], e.extra["attempt"]), default))


def by_revision(edits: Mapping[tuple[int, int], list[dict[str, Any]]]) -> Any:
    """A Reviser making `edits[(seed, revision)]`."""
    return editing(lambda e: edits.get((e.extra["seed"], e.extra["revision"]), []))


def lift_to(value: float) -> dict[str, Any]:
    return {"op": "set_number", "param": "lift_height", "value": value, "why": "too low"}


def rows(setup: Any) -> list[LedgerRow]:
    return Ledger(setup.run_dir / LIFT_RED / "S2" / "ledger.jsonl").rows()


async def practice(
    setup: Any, brains: dict[str, Any], hooks: Hooks | None = None, **config: Any
) -> PracticeResult:
    config = {"start": 0, **config}
    # a demonstration package (shown or not) has its end gate finish (`tools.done_gate`): mount
    # check_done
    gated = "demo" in config or "done_goal" in config
    extra = (
        [
            LocalProvider(WorldState(), SceneConfig(pipeline=PERCEPTION)),
            LocalProvider(Done(), SceneConfig(pipeline=PERCEPTION)),
        ]
        if gated
        else []
    )
    async with setup.runner(brains, hooks=hooks, extra=extra) as runner:
        result = await runner.run(Practice(), LIFT_RED, config)
    assert isinstance(result, PracticeResult)
    return result


async def test_three_first_try_passes_stop_and_freeze(setup: Any) -> None:
    await setup.start(lift_memory(0.06))
    result = await practice(setup, {"actor": by_try({}, "pass")})
    assert (result.reason, result.k, result.version, result.frozen) == ("converged", 3, 0, True)
    ledger = rows(setup)
    assert [(r.seed, r.k, r.outcome, r.streak_after) for r in ledger] == [
        (10000, 1, "pass", 1),
        (10001, None, "unsolvable", 1),
        (10002, 2, "pass", 2),
        (10003, 3, "pass", 3),
    ]
    assert ledger[0].alpha_k == 1.0 and ledger[0].budget_k == 4 and ledger[0].revisions == []
    frozen = await setup.store.frozen(LIFT_RED)
    assert frozen is not None and (frozen.version, frozen.reason) == (0, "converged")
    s2 = setup.run_dir / LIFT_RED / "S2"
    started = json.loads((s2 / "start.json").read_text(encoding="utf-8"))
    assert (started["version"], started["done_criteria"]) == (0, 0)  # finishes not gated
    stop = PracticeResult.model_validate_json((s2 / "stop.json").read_bytes())
    assert stop == result and stop.summary["first_try"]["successes"] == 3


async def test_the_early_stop_is_passes_among_the_last_counted_seeds(setup: Any) -> None:
    """First tries pass, fail, pass, fail, pass (10001 is unsolvable, not counted): 3 of the last
    5 stop it at k = 5, which no 3 in a row would; a stricter window runs on to its cap."""
    await setup.start(lift_memory(0.06))
    plan: Plan = {(seed, 1): "pass" for seed in (10000, 10003, 10005)}
    result = await practice(setup, {"actor": by_try(plan), "reviser": by_revision({})})
    assert (result.reason, result.k) == ("converged", 5)
    assert [r.first_try_pass for r in rows(setup) if r.k is not None] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert max(r.streak_after for r in rows(setup)) == 1
    with pytest.raises(ValidationError, match="cannot pass within"):
        PracticeConfig(stop_passes=4, stop_window=3)


async def test_merge_jitter_and_discard(setup: Any) -> None:
    """Seed 10000: try 2 passes with an edited cache (merge at alpha 1); 10002: try 2 passes with
    the cache unchanged (a jitter); 10003: try 3 passes after the second revision (merge at
    alpha 10/12, the decay's at k = 3); 10004: every try fails (discarded); then the cap of 4
    counted seeds."""
    await setup.start(lift_memory(0.02))
    plan: Plan = {(10000, 2): "pass", (10002, 2): "pass", (10003, 3): "pass"}
    edits = {(10000, 1): [lift_to(0.06)], (10003, 2): [lift_to(0.1)]}
    brains = {"actor": by_try(plan), "reviser": by_revision(edits)}
    result = await practice(setup, brains, max_seeds=4)
    assert (result.reason, result.k, result.version) == ("not converged", 4, 2)
    ledger = [r for r in rows(setup) if r.k is not None]
    assert [(r.seed, r.outcome, len(r.tries), len(r.revisions)) for r in ledger] == [
        (10000, "merge", 2, 1),
        (10002, "jitter", 2, 1),
        (10003, "merge", 3, 2),
        (10004, "skip", 3, 2),
    ]
    alphas = [(r.alpha_k, r.budget_k) for r in ledger]
    assert alphas == pytest.approx([(1.0, 4), (10 / 11, 4), (10 / 12, 3), (10 / 13, 3)])
    assert [(r.memory_before, r.memory_after) for r in ledger] == [(0, 1), (1, 1), (1, 2), (2, 2)]
    assert [t.memory for t in ledger[2].tries] == ["v001", "c1", "c2"]
    assert [(r.revision, r.end, r.accepted) for r in ledger[2].revisions] == [
        (1, "finish", 0),
        (2, "finish", 1),
    ]
    merged = await setup.store.load(LIFT_RED, 2)
    expected = 0.06 + 10 / 12 * (0.1 - 0.06)
    assert merged.doc.params["lift_height"].value == pytest.approx(expected, abs=1e-6)  # 6 places
    assert merged.meta.source == VersionSource(
        kind="merge", seed=10003, attempt=3, ref="S2/10003/merge.json"
    )
    s2 = setup.run_dir / LIFT_RED / "S2"
    verdicts = [
        json.loads((s2 / str(seed) / "merge.json").read_text(encoding="utf-8"))["verdict"]
        for seed in (10000, 10002, 10003, 10004)
    ]
    assert verdicts == ["merge", "jitter", "merge", "skip"]
    cache = s2 / "10003" / "cache"
    assert {p.name for p in cache.iterdir()} == {
        "c1",
        "c2",
        "diff_c1.json",
        "diff_c2.json",
        "reviser",
    }
    assert "## SCRATCH" in (cache / "c2" / "memory.md").read_text(encoding="utf-8")
    assert (
        json.loads((cache / "diff_c2.json").read_text(encoding="utf-8"))["used"] == 0
    )  # numbers cost nothing
    assert ledger[3].failure_labels == ["finished without success"] * 3


async def test_infrastructure_reruns_and_unsolvable_opens(setup: Any) -> None:
    """A try voided by a gateway failure is re-run without using a try (reopened without the
    env's setup check); a seed whose opens keep faulting is unsolvable, as in the official loop."""
    await setup.start(lift_memory(0.06))
    setup.env.open_faults = {10002}
    failed: set[int] = set()

    def plan(episode: Episode) -> Act:
        if episode.extra["seed"] == 10000 and 10000 not in failed:
            failed.add(10000)
            return "gateway"
        return "pass"

    result = await practice(setup, {"actor": acting(plan)}, max_seeds=2)
    assert result.reason == "not converged"
    ledger = rows(setup)
    assert [(r.seed, r.outcome) for r in ledger] == [
        (10000, "pass"),
        (10001, "unsolvable"),
        (10002, "unsolvable"),
        (10003, "pass"),
    ]
    assert ledger[0].tries[0].infra_reruns == 1 and "3 runs in a row" in ledger[2].note
    assert (setup.run_dir / LIFT_RED / "S2" / "10000" / "try1-infra1").is_dir()
    first, again = [spec for spec in setup.env.opened if spec.seed == 10000]
    assert (first.repeat, again.repeat) == (False, True)
    assert again.instruction == "lift the red box"


class FailingPlanner(MockPlanner):
    """MockPlanner whose first `failing` plans fail as infrastructure, as cuRobo's service fails
    a request whose graph interpolation raised, or refuses calls while it restarts."""

    def __init__(self, failing: int) -> None:
        super().__init__()
        self.failing = failing

    async def plan(self, request: PlanRequest) -> PlanResult:
        if self.failing:
            self.failing -= 1
            raise InfraError("not_ready", "the planner failed this request (a test)")
        return await super().plan(request)


class OnPlanner(Manipulation):
    """The manipulation toolset on a given planner."""

    def __init__(self, planner: MockPlanner) -> None:
        super().__init__()
        self.given = planner

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.planner = self.given


def _reviewed(revised: list[int]) -> Any:
    """A Reviser making no edit, noting the seed of each revision it runs."""
    review = editing(lambda _: [])

    def make(episode: Episode) -> ScriptedBrain:
        revised.append(episode.extra["seed"])
        return review(episode)

    return make


FREE_MOVE: Final = [-0.1, -0.1, 1.0, 0.7071068, 0.0, 0.7071068, 0.0]


async def test_a_try_that_ends_after_an_unanswered_infrastructure_error_is_void(
    setup: Any,
) -> None:
    """The planner fails the Actor's first move as infrastructure and the Actor then finishes, no
    call answered after that error: the try ends as infrastructure's, as
    two such errors in a row end one, is not counted and reaches no Reviser; it runs again on its
    seed, passes, and counts as a first-try pass."""
    await setup.start(lift_memory(0.06))
    manipulation = OnPlanner(FailingPlanner(1))
    config = {"planner": {"mock": True}, "table_top_z": TABLE_Z}
    provider = LocalProvider(manipulation, config, ["mock"])
    runs: list[int] = []
    revised: list[int] = []

    def actor(episode: Episode) -> ScriptedBrain:
        runs.append(episode.extra["seed"])
        first, acted = len(runs) == 1, False

        async def script(request: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal acted
            if acted:
                return [call("finish", note="done")]
            acted = True
            if first:  # a free move, which the planner fails; then the finish above
                return [call("move_to_pose", rationale=RATIONALE, arm="left", pose=FREE_MOVE)]
            return [call("lift", rationale=RATIONALE, color=color_of(request), height_m=0.06)]

        return ScriptedBrain(script, NO_OBSERVE)

    brains = {"actor": actor, "reviser": _reviewed(revised)}
    async with setup.runner(brains, extra=[provider]) as runner:
        await runner.run(Practice(), LIFT_RED, {"start": 0, "max_seeds": 1})
    [row] = rows(setup)
    assert (row.seed, row.outcome, row.first_try_pass) == (10000, "pass", True)
    assert row.tries[0].infra_reruns == 1 and runs == [10000, 10000] and revised == []
    voided = setup.run_dir / LIFT_RED / "S2" / "10000" / "try1-infra1"
    summary = EpisodeOutcome.model_validate_json((voided / "summary.json").read_bytes())
    assert (summary.end.reason, summary.end_class) == ("infra_error", "infra")
    assert summary.end.note.startswith("it ended (finish) after a call failed")
    assert summary.end.note.endswith("; its note: done")  # the end it had, its note kept
    assert summary.voided_end == "finish"
    assert (
        summary.voided_by
        == "move_to_pose: The tool is unavailable right now (not_ready); try again."
    )
    calls = read_lines(voided / "tool_calls.jsonl")
    assert (calls[0]["name"], calls[0]["code"]) == ("move_to_pose", "infra_error")
    assert all(c["code"] != "infra_error" for c in calls[1:])  # one error, not two in a row


async def test_tries_reuse_the_instruction_without_the_prerun(setup: Any) -> None:
    await setup.start(lift_memory(0.06))
    await practice(setup, {"actor": by_try({}), "reviser": by_revision({})}, max_seeds=1)
    specs = [spec for spec in setup.env.opened if spec.seed == 10000]
    assert [(s.repeat, s.prerun_attempts) for s in specs] == [
        (False, 3),
        (True, 3),
        (True, 3),
    ]
    assert {s.instruction for s in specs[1:]} == {"lift the red box"} and specs[
        0
    ].instruction is None
    assert all(s.stage == "s2" and not s.options for s in specs)


async def test_s2_starts_where_s1_passed(setup: Any) -> None:
    """Without `start`, S2 needs exactly one passed S1 exit and starts from its version."""
    await setup.start(lift_memory(0.02))
    v1 = await setup.store.commit(
        lift_memory(0.06), status="s1_draft", source=VersionSource(kind="s1"), parent=0
    )
    brains = {"actor": by_try({}, "pass")}
    with pytest.raises(ValueError, match="holds 0 passed runs"):
        await practice(setup, brains, start=None)
    s1 = setup.run_dir / LIFT_RED / "S1"
    for seed, passed in ((7, False), (8, True)):
        (s1 / str(seed)).mkdir(parents=True)
        exit = {"passed": passed, "version": v1.meta.version, "hash": v1.meta.hash}
        (s1 / str(seed) / "exit.json").write_text(json.dumps(exit), encoding="utf-8")
    result = await practice(setup, brains, start=None, max_seeds=1)
    assert rows(setup)[0].memory_before == 1 and result.version == 1


class CrashAtMerge(Hooks):
    """Stops the practice right after a merge was committed, before its ledger row."""

    async def on_merge(
        self, task: str, seed: int, record: MergeRecord, version: VersionMeta | None
    ) -> None:
        if version is not None and seed == 10002:
            raise RuntimeError("crashed after the merge")


async def test_resume_after_a_crash_between_merge_and_row(setup: Any) -> None:
    """The merge of seed 10002 lands in the store but its row does not: the resumed practice
    continues from the last row's Memory, ignores that orphan merge and runs the seed again;
    the rejected edits of the interrupted seed are not kept. The early stop's window spans the
    resume: 10000, 10003 and 10004 are 3 first-try passes among 4 counted seeds, so it stops at
    10004, where 3 in a row would have run 10005 too."""
    await setup.start(lift_memory(0.02))
    plan: Plan = {(10000, 1): "pass", (10002, 2): "pass"}
    plan |= {(seed, 1): "pass" for seed in (10003, 10004, 10005)}
    bad = {"op": "set_number", "param": "lift_height", "value": 0.5, "why": "far too far"}
    edits = {(10002, 1): [bad, lift_to(0.06)]}
    brains = {"actor": by_try(plan), "reviser": by_revision(edits)}
    with pytest.raises(RuntimeError, match="crashed after the merge"):
        await practice(setup, brains, hooks=CrashAtMerge())
    assert [r.seed for r in rows(setup)] == [10000, 10001]
    [orphan] = [m for m in await setup.store.versions(LIFT_RED) if m.status == "merged"]
    assert (orphan.version, orphan.source.seed) == (1, 10002)
    assert not (setup.run_dir / LIFT_RED / "S2" / "rejected.jsonl").exists()
    with pytest.raises(ValueError, match="has started"):
        await practice(setup, brains)
    result = await practice(setup, brains, resume=True)
    assert (result.reason, result.k, result.version) == ("converged", 4, 2)
    ledger = rows(setup)
    assert [(r.seed, r.outcome, r.memory_before, r.memory_after) for r in ledger[2:]] == [
        (10002, "merge", 0, 2),  # merged from v000 again, not from the orphan v001
        (10003, "pass", 2, 2),
        (10004, "pass", 2, 2),
    ]
    rejected = read_lines(setup.run_dir / LIFT_RED / "S2" / "rejected.jsonl")
    assert len(rejected) == 1  # the re-run seed's one rejection, written with its row
    assert (setup.run_dir / LIFT_RED / "S2" / "10002-aborted1").is_dir()


def actor_counting(plan: Plan, counts: dict[tuple[int, int], int]) -> Any:
    """`by_try(plan)`, counting the Actor episodes by (seed, attempt)."""
    acted = by_try(plan)

    def make(episode: Episode) -> Any:
        key = (episode.extra["seed"], episode.extra["attempt"])
        counts[key] = counts.get(key, 0) + 1
        return acted(episode)

    return make


async def test_a_seed_cut_after_a_passed_try_1_is_not_run_again(
    setup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped after seed 10002's try 1 passed, before its row: the resumed practice writes the
    row from the recorded try 1 without opening the seed again (a rerun would fail)."""
    from roboscript_methods.wicl import practice as module

    await setup.start(lift_memory(0.06))
    counts: dict[tuple[int, int], int] = {}
    commit = module._Practice.commit

    def crash(self: Any, row: LedgerRow) -> None:
        if row.seed == 10002:
            raise RuntimeError("stopped before the row")
        commit(self, row)

    monkeypatch.setattr(module._Practice, "commit", crash)
    plan: Plan = {(seed, 1): "pass" for seed in (10000, 10002, 10003)}
    with pytest.raises(RuntimeError, match="before the row"):
        await practice(setup, {"actor": actor_counting(plan, counts)})
    monkeypatch.setattr(module._Practice, "commit", commit)
    assert [r.seed for r in rows(setup)] == [10000, 10001]
    opened = len(setup.env.opened)
    rerun = actor_counting({(10000, 1): "pass", (10003, 1): "pass"}, counts)
    result = await practice(setup, {"actor": rerun}, resume=True)
    assert (result.reason, result.k) == ("converged", 3)
    assert counts[10002, 1] == 1
    assert [spec.seed for spec in setup.env.opened[opened:]] == [10003]
    ledger = rows(setup)
    assert [(r.seed, r.k, r.first_try_pass) for r in ledger[2:]] == [
        (10002, 2, True),
        (10003, 3, True),
    ]
    assert not (setup.run_dir / LIFT_RED / "S2" / "10002-aborted1").exists()


async def test_the_success_only_ablation_hides_the_diagnosis(setup: Any) -> None:
    await setup.start(lift_memory(0.02))
    briefs: list[str] = []

    def reviser(episode: Episode) -> ScriptedBrain:
        async def script(request: dict[str, Any]) -> list[dict[str, Any]]:
            briefs.append(json.dumps(request["input"]))
            return [call("finish", note="ok")]

        return ScriptedBrain(script, NO_OBSERVE)

    brains = {"actor": by_try({}), "reviser": reviser}
    await practice(setup, brains, max_seeds=1, diagnosis=False)
    assert briefs and all("Ground truth" not in brief for brief in briefs)
    s2 = setup.run_dir / LIFT_RED / "S2"
    assert (s2 / "10000" / "try1" / "referee" / "diagnostics.json").is_file()  # still recorded


def demo_layer(setup: Any, task: str = LIFT_RED, seed: int = 7, s1_seed: int = 7) -> Any:
    """A package directory with only a ground-truth layer (of `task` at `seed`), and S1's
    `exit.json` at `s1_seed`: the demonstration's path."""
    package = setup.tmp / f"package-{task}-{seed}"
    package.mkdir()
    red = Pose.from_list([0.1, 0.2, 0.76, 1.0, 0.0, 0.0, 0.0])
    truth = DemoTruth(
        task=task,
        seed=seed,
        objects={"red": "box"},
        frames=[TruthFrame(frame=0, what=["start"], poses={"red": red})],
    )
    (package / TRUTH_FILE).write_text(truth.model_dump_json(), encoding="utf-8")
    exit = setup.run_dir / LIFT_RED / "S1" / str(s1_seed) / "exit.json"
    exit.parent.mkdir(parents=True, exist_ok=True)
    exit.write_text(
        json.dumps({"seed": s1_seed, "passed": True, "version": 0, "hash": "-"}), encoding="utf-8"
    )
    return package


async def test_the_demo_truth_reaches_only_the_reviser_and_only_when_asked(setup: Any) -> None:
    package = demo_layer(setup)
    await setup.start(lift_memory(0.02))
    brains = {"actor": by_try({}), "reviser": by_revision({})}
    await practice(setup, brains, max_seeds=1, demo_truth=str(package))
    seed = setup.run_dir / LIFT_RED / "S2" / "10000"
    reviser = (seed / "cache" / "reviser" / "r1" / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "The demonstration's ground truth" in reviser and "(0.100, 0.200, 0.760)" in reviser
    for attempt in ("try1", "try2", "try3"):
        assert "ground truth" not in (seed / attempt / "llm_calls.jsonl").read_text(
            encoding="utf-8"
        )
    digest = hashlib.sha256((package / TRUTH_FILE).read_bytes()).hexdigest()
    label = f"on for the Reviser: {package} (sha256 {digest})"
    config = json.loads(
        (seed / "cache" / "reviser" / "r1" / "config.json").read_text(encoding="utf-8")
    )
    assert config["protocol"]["demo_truth"] == label
    s2 = setup.run_dir / LIFT_RED / "S2"
    assert json.loads((s2 / "start.json").read_text(encoding="utf-8"))["demo_truth"] == label
    assert (
        json.loads((s2 / "stop.json").read_text(encoding="utf-8"))["summary"]["demo_truth"] == label
    )


async def test_the_demo_truth_is_off_by_default(setup: Any) -> None:
    await setup.start(lift_memory(0.02))
    await practice(setup, {"actor": by_try({}), "reviser": by_revision({})}, max_seeds=1)
    reviser = setup.run_dir / LIFT_RED / "S2" / "10000" / "cache" / "reviser" / "r1"
    assert "The demonstration's ground truth" not in (reviser / "llm_calls.jsonl").read_text(
        encoding="utf-8"
    )
    assert (
        "demo_truth"
        not in json.loads((reviser / "config.json").read_text(encoding="utf-8"))["protocol"]
    )
    assert (
        "demo_truth"
        not in json.loads((reviser.parents[3] / "stop.json").read_text(encoding="utf-8"))["summary"]
    )


class Recorder(Hooks):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def on_seed_start(self, task: str, seed: int, memory: VersionMeta) -> None:
        self.calls.append(("seed_start", seed, memory.version))

    async def on_try_end(self, task: str, seed: int, attempt: int, outcome: EpisodeOutcome) -> None:
        self.calls.append(("try_end", seed, attempt, outcome.success))

    async def on_revise(
        self, task: str, seed: int, revision: int, records: list[EditRecord], diff: MemoryDiff
    ) -> None:
        self.calls.append(("revise", seed, revision, len(records), diff.empty))

    async def on_merge(
        self, task: str, seed: int, record: MergeRecord, version: VersionMeta | None
    ) -> None:
        self.calls.append(("merge", seed, None if version is None else version.version))

    async def on_skip(self, task: str, seed: int, reason: str) -> None:
        self.calls.append(("skip", seed, reason))

    async def on_stop(self, task: str, reason: str, frozen: FreezeRecord | None) -> None:
        self.calls.append(("stop", reason, None if frozen is None else frozen.version))


async def test_hooks_see_every_step(setup: Any) -> None:
    await setup.start(lift_memory(0.02))
    hooks = Recorder()
    plan: Plan = {(10000, 2): "pass", (10002, 1): "pass"}
    brains = {"actor": by_try(plan), "reviser": by_revision({(10000, 1): [lift_to(0.06)]})}
    await practice(setup, brains, hooks=hooks, max_seeds=2)
    assert hooks.calls == [
        ("seed_start", 10000, 0),
        ("try_end", 10000, 1, False),
        ("revise", 10000, 1, 1, False),
        ("try_end", 10000, 2, True),
        ("merge", 10000, 1),
        ("seed_start", 10001, 1),
        ("skip", 10001, "unsolvable"),
        ("seed_start", 10002, 1),
        ("try_end", 10002, 1, True),
        ("stop", "not converged", 1),
    ]
    lines = read_lines(setup.run_dir / LIFT_RED / "events.jsonl")
    steps = [line["data"]["step"] for line in lines if line["kind"] == "protocol.step"]
    assert steps[0] == "start" and steps[-1] == "end" and "merge" in steps
    assert any(line["kind"] == "memory.merge" for line in lines)
