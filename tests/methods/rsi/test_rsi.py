"""P4's pieces (`roboscript_methods.rsi`, docs/concepts/learning.md): typed edits (add, modify,
merge and retire with stable ids, GLOBAL's evidence rule, evidence naming an episode the round
read) and lint, what an Actor reads of the RSI document, and its declaration, store and toolset
through the core's conformance suites."""

import itertools
from pathlib import Path
from typing import Any

import pytest

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.ids import TaskName
from roboscript.testing.artifacts import ArtifactStoreConformance, ArtifactStoreTarget
from roboscript.testing.memory import sample_memory
from roboscript.testing.methods import MethodConformance
from roboscript.testing.phases import TestMethod
from roboscript.testing.toolsets import ToolsetConformance, ToolsetTarget
from roboscript_methods.rsi.doc import (
    CONTEXT_NOTES,
    EpisodeKey,
    Evidence,
    RsiDoc,
    render,
)
from roboscript_methods.rsi.edits import (
    AddEntry,
    Edit,
    MergeEntries,
    ModifyEntry,
    RetireEntry,
    RsiEditor,
    VoteEntry,
)
from roboscript_methods.rsi.lint import LintContext
from roboscript_methods.rsi.method import (
    Rsi,
    attach,
)
from roboscript_methods.rsi.recipe import RsiRules, load_rsi_recipe
from roboscript_methods.rsi.store import FileRsiStore
from roboscript_methods.wicl.method import Wicl

TASK = TaskName("carry_tool")
OTHER = TaskName("lift_red")
THIRD = TaskName("stack_two")
RECIPE = load_rsi_recipe(Path("recipes/rsi"), Path(__file__).with_name("lint_overlay.yaml"))
RULES = RECIPE.rules
CONTEXT = LintContext(
    vocabulary=RECIPE.lint,
    tasks=[TASK, OTHER, THIRD],
    objects=["tool_red"],
    memories={TASK: sample_memory(TASK)},
)


def _ev(task: TaskName = TASK, episode: int = 0, step: int | None = 12) -> Evidence:
    return Evidence(block="block1", task=task, episode=episode, step=step)


def _key(task: TaskName = TASK, episode: int = 0, block: str = "block1") -> EpisodeKey:
    return EpisodeKey(block=block, task=task, episode=episode)


EPISODES = [_key(task, i) for task in (TASK, OTHER, THIRD) for i in range(30)]


SEEN = [_ev()]
SPREAD = [_ev(TASK), _ev(OTHER), _ev(THIRD)]


def _editor(
    round: int = 1,
    rules: RsiRules = RULES,
    doc: RsiDoc | None = None,
    episodes: list[EpisodeKey] = EPISODES,
) -> RsiEditor:
    return RsiEditor(doc or RsiDoc(), rules, CONTEXT, round, episodes)


def _add(
    task: TaskName | None, when: str, do: str, check: str = "the gripper holds it"
) -> AddEntry:
    return AddEntry(
        task=task, when=when, do=do, check=check, evidence=SPREAD if task is None else SEEN
    )


def _accepted(editor: RsiEditor, *ops: Edit) -> list[str]:
    records = [editor.edit(op) for op in ops]
    assert all(r.accepted for r in records), [r.reason for r in records]
    return [r.entry or "" for r in records]


# The recipe


# The document and what an Actor reads of it


def test_typed_edits_add_modify_merge_and_retire_with_stable_ids() -> None:
    editor = _editor()
    ids = _accepted(
        editor,
        _add(None, "a grasp closes on nothing", "look again before regrasping"),
        _add(None, "an object slides out of the fingers", "grasp nearer its middle"),
        _add(TASK, "the head points away", "turn it before the strike"),
    )
    assert ids == ["G-01", "G-02", f"T-{TASK}-01"]
    more = _ev(episode=1, step=None)
    _accepted(editor, ModifyEntry(id=f"T-{TASK}-01", do="turn it first", evidence=[more]))
    (merged,) = _accepted(
        editor,
        MergeEntries(ids=["G-01", "G-02"], when="a grasp fails", do="regrasp", check="it holds"),
    )
    assert merged == "G-03"
    _accepted(editor, RetireEntry(id=f"T-{TASK}-01", reason="no gain measured"))
    doc = editor.doc
    assert [(e.id, e.status) for e in doc.global_entries] == [
        ("G-01", "retired"),
        ("G-02", "retired"),
        ("G-03", "probation"),
    ]
    (task_entry,) = doc.task_entries[TASK]
    assert (task_entry.do, task_entry.status, task_entry.modified) == (
        "turn it first",
        "retired",
        (1,),
    )
    assert task_entry.evidence == (*SEEN, more)
    assert [c.op for c in task_entry.history] == ["add", "modify", "retire"]
    assert doc.global_entries[2].evidence == tuple(SPREAD)  # merged evidence, once each
    assert [c.op for c in doc.global_entries[0].history] == ["add", "merged_into"]
    # the next id is never a retired one's
    assert _accepted(editor, _add(TASK, "the block tips", "press it down")) == [f"T-{TASK}-02"]
    refused = editor.edit(RetireEntry(id=f"T-{TASK}-01", reason="again"))
    assert (refused.accepted, refused.reason) == (False, f"entry T-{TASK}-01 is retired")
    assert not editor.edit(MergeEntries(ids=["G-03", "G-03"], when="w", do="p", check="c")).accepted
    text = render(doc)
    assert "# EXPERIENCE: GLOBAL" in text and f"# EXPERIENCE: TASK {TASK}" in text
    assert "(retired;" in text and f"block1/{TASK}/ep000 step 12" in text


def test_the_actor_reads_readable_entries_without_evidence_counts_or_status() -> None:
    editor = _editor()
    _accepted(
        editor,
        _add(None, "a grasp closes on nothing", "look again before regrasping"),
        _add(TASK, "the head points away", "turn it before the strike"),
    )
    doc = editor.doc
    assert doc.context(None) == "\n".join(
        [
            "# EXPERIENCE: GLOBAL",
            *CONTEXT_NOTES,
            "- [G-01] When: a grasp closes on nothing \u2192 look again before regrasping "
            "Check: the gripper holds it",
        ]
    )
    # the notes once, above the first readable part
    assert doc.context(TASK) == (
        f"# EXPERIENCE: TASK {TASK}\n- [T-{TASK}-01] When: the head points away \u2192 turn "
        "it before the strike Check: the gripper holds it"
    )
    assert doc.context(OTHER) == ""
    task_only = RsiDoc(task_entries={TASK: doc.task_entries[TASK]})
    assert task_only.context(None) == ""
    assert task_only.context(TASK).splitlines()[1:3] == list(CONTEXT_NOTES)
    for text in (doc.context(None), doc.context(TASK)):
        assert "block1" not in text and "helpful" not in text and "probation" not in text
    # the same entries in another order read the same (by id)
    shuffled = doc.model_copy(update={"global_entries": doc.global_entries[::-1]})
    assert shuffled.context(None) == doc.context(None)


# Edit checks


def test_global_needs_evidence_from_three_tasks_and_merge_promotes() -> None:
    editor = _editor()
    narrow = editor.edit(
        AddEntry(
            task=None,
            when="a grasp slips",
            do="grasp lower",
            check="it holds",
            evidence=[_ev(TASK), _ev(OTHER)],
        )
    )
    assert (narrow.accepted, narrow.reason) == (
        False,
        "GLOBAL needs evidence from 3 distinct tasks among this round's episodes, this has 2",
    )
    a, b = _accepted(
        editor,
        AddEntry(
            task=TASK, when="a grasp slips", do="grasp lower", check="c", evidence=[_ev(TASK)]
        ),
        AddEntry(
            task=OTHER, when="the grasp slides", do="grasp lower", check="c", evidence=[_ev(OTHER)]
        ),
    )
    two = editor.edit(MergeEntries(ids=[a, b], into_global=True, when="w", do="d", check="c"))
    assert two.reason.endswith("3 distinct tasks among this round's episodes, this has 2")
    assert "one part" in editor.edit(MergeEntries(ids=[a, b], when="w", do="d", check="c")).reason
    (c,) = _accepted(
        editor,
        AddEntry(
            task=THIRD, when="the grip slips", do="grasp lower", check="c", evidence=[_ev(THIRD)]
        ),
    )
    _accepted(editor, VoteEntry(id=a, vote="helpful", episode=_key(TASK, 4)))
    (promoted,) = _accepted(
        editor,
        MergeEntries(
            ids=[a, b, c],
            into_global=True,
            when="a grasp slips",
            do="grasp lower",
            check="it holds",
        ),
    )
    assert promoted == "G-01"
    _, entry = editor.doc.find(promoted)
    assert (entry.status, entry.helpful, {e.task for e in entry.evidence}) == (
        "probation",
        0,
        {TASK, OTHER, THIRD},
    )
    assert entry.votes_helpful == (_key(TASK, 4),)  # votes carried, apart from paired counts
    again = editor.edit(VoteEntry(id=promoted, vote="harmful", episode=_key(TASK, 4)))
    assert again.reason == f"episode block1/{TASK}/ep004 has voted on G-01 already"
    assert all(editor.doc.find(i)[1].status == "retired" for i in (a, b, c))
    assert editor.left(None) == 7  # the promotion is GLOBAL's edit


def test_evidence_must_name_an_episode_the_round_read() -> None:
    editor = _editor()
    stray = AddEntry(
        task=TASK,
        when="w",
        do="d",
        check="c",
        evidence=[_ev(), Evidence(block="b9", task=TASK, episode=0)],
    )
    assert editor.edit(stray).reason == f"no episode this round read is b9/{TASK}/ep000"
    (entry_id,) = _accepted(editor, _add(TASK, "the head points away", "turn it"))
    late = ModifyEntry(id=entry_id, evidence=[_ev(episode=77)])
    assert editor.edit(late).reason == f"no episode this round read is block1/{TASK}/ep077"
    # a task outside the run, or one an id cannot hold, is refused, not a validation error
    for name in ("no_such_task", "bad-name"):
        record = editor.edit(
            AddEntry(task=TaskName(name), when="w", do="d", check="c", evidence=SEEN)
        )
        assert record.reason == f"{name!r} is no task of this run"


# Lint


@pytest.mark.parametrize(
    ("task", "when", "do", "rule"),
    [
        (None, "beating the block", "see carry_tool", "global_task"),
        (None, "the task is lift red", "go slowly", "global_task"),
        (None, "the tool red one", "grasp it", "global_object"),
        (None, "obj_3 is near", "go around it", "global_object"),
        (None, "a grasp slips twice in 2 tries", "grasp again", "global_number"),
        (None, "simx shows a glitch", "wait", "simulator"),
        (TASK, "the head is off", "lower by 2 cm", "do_number"),
        (TASK, "on seed 100012 the head is off", "turn it", "seed"),
        (TASK, "the head sits at x=0.31", "turn it", "coordinate"),
        (TASK, "the head is at (0.1, -0.2)", "turn it", "coordinate"),
        (TASK, "success_probe stays false", "turn it", "evaluator"),
        (TASK, "the referee reads no lift", "turn it", "evaluator"),
        (TASK, "the ground truth pose is off", "turn it", "evaluator"),
        (TASK, "as in ep003 the head is off", "turn it", "episode_ref"),
        (TASK, "episode 12 shows the head off", "turn it", "episode_ref"),
        (TASK, "the head is off in block 2", "turn it", "episode_ref"),
        (TASK, "see block1/carry_tool/ep004", "turn it", "episode_ref"),
    ],
)
def test_lint_refuses(task: TaskName | None, when: str, do: str, rule: str) -> None:
    record = _editor().edit(_add(task, when, do))
    assert not record.accepted and record.reason == "the lint refuses it"
    assert rule in {f.rule for f in record.findings}


# Counters and status


# Store by round


# The declaration


@pytest.fixture(params=["rsi", "test+rsi", "wicl+rsi"])
def method_target(request: pytest.FixtureRequest) -> Any:
    return {"rsi": Rsi, "test+rsi": attach(TestMethod), "wicl+rsi": attach(Wicl)}[request.param]


class TestRsiConforms(MethodConformance):
    pass


# The store and the toolset


@pytest.fixture
def artifact_store_target(tmp_path: Path) -> ArtifactStoreTarget:
    store = FileRsiStore(tmp_path / "rsi")
    count = itertools.count()

    async def commit(name: str) -> ArtifactRef:
        entry = _add(TASK, f"case {next(count)} of a slip", "grasp nearer the middle")
        editor = _editor()
        _accepted(editor, entry)
        return await store.commit(name, editor.doc)

    return ArtifactStoreTarget(store=store, commit=commit)


class TestRsiStoreConforms(ArtifactStoreConformance):
    pass


@pytest.fixture
def toolset_target() -> ToolsetTarget:
    return ToolsetTarget.registered("rsi")


class TestRsiToolsetConforms(ToolsetConformance):
    pass
