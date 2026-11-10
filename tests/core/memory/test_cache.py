"""Per-seed caches, diffs and merges: a cache is a copy with scratch and a budget, a merge
interpolates numbers and takes the rest, and an unchanged cache is a jitter."""

from typing import Any

from roboscript.memory import (
    Change,
    Editor,
    GraspSpec,
    Lesson,
    MemoryDoc,
    MemoryRecipe,
    ParamMerge,
    content_hash,
    diff,
    lint,
    merge,
    open_cache,
)

WHY = {"why": "the grasp stopped short of the handle"}


def cache_of(memory: MemoryDoc, recipe: MemoryRecipe, budget: int = 4) -> Editor:
    return open_cache(memory, budget=budget, seed=10004, rules=recipe.rules("s2"), lint=recipe.lint)


def ok(editor: Editor, **op: Any) -> None:
    record = editor.edit({**op, **WHY} if op["op"] != "scratch" else op)
    assert record.accepted, record.reason


def test_a_cache_is_a_copy_with_scratch_and_a_budget(
    memory: MemoryDoc, recipe: MemoryRecipe
) -> None:
    m = memory.replace(version=3, created="2026-09-28T12:00:00Z")
    cache = cache_of(m, recipe)
    assert (cache.doc.scratch, cache.doc.version, cache.doc.source) == ([], None, "cache of v003")
    assert (cache.budget, cache.left, cache.seed, cache.rules) == (4, 4, 10004, recipe.rules("s2"))
    assert diff(m, cache.doc).empty and cache.base == cache.doc


def test_merge_interpolates_numbers_and_takes_the_rest(
    memory: MemoryDoc, recipe: MemoryRecipe
) -> None:
    m = memory.replace(version=3)
    cache = cache_of(m, recipe)
    ok(cache, op="set_number", param="pre_grasp", value=0.12)
    ok(cache, op="set_number", param="grasp_handle", field="offset_m", value=[-0.01, 0, -0.01])
    ok(cache, op="set_category", param="place_on_base", field="frame", value="target")
    ok(cache, op="set_number", param="place_on_base", field="offset_m", value=[0, 0, 0.06])
    ok(cache, op="add_lesson", when="the tool lies diagonal", do="grasp nearer the head")
    ok(cache, op="scratch", text="the handle looked short")
    merged = merge(m, cache.doc, 0.5, seed=10004, attempt=2)
    doc = merged.doc
    assert doc is not None
    assert doc.params["pre_grasp"].value == 0.11
    assert doc.params["grasp_handle"].value.offset_m == (-0.02, 0.0, -0.01)
    # a spec whose category changed is the cache's, whole: its numbers were tuned for it
    assert doc.params["place_on_base"].value == cache.doc.params["place_on_base"].value
    assert (doc.version, doc.stage, doc.scratch) == (None, "S2", None)
    assert doc.source == "merge into v003 from seed 10004 try 2"
    assert doc.lessons[-1] == Lesson(
        id="L2", when="the tool lies diagonal", do="grasp nearer the head", seed=10004, merges=1
    )
    assert doc.lessons[0] == m.lessons[0]
    assert doc.params["pre_grasp"].source == "S1 demo reader"
    assert doc.params["pre_grasp"].history == [
        Change(how="merge", before=0.1, after=0.11, seed=10004, alpha=0.5)
    ]
    ours, theirs = (
        spec.params["place_on_base"].value.model_dump(mode="json") for spec in (m, cache.doc)
    )
    assert doc.params["place_on_base"].history == [
        Change(how="merge", before=ours, after=theirs, seed=10004)
    ]
    record = merged.record
    assert (record.seed, record.attempt, record.alpha, record.jitter) == (10004, 2, 0.5, False)
    assert (record.base, record.cache) == (content_hash(m), content_hash(cache.doc))
    assert record.params == [
        ParamMerge(param="pre_grasp", how="interpolated", before=0.1, cache=0.12, after=0.11),
        ParamMerge(
            param="grasp_handle",
            field="offset_m",
            how="interpolated",
            before=[-0.03, 0.0, -0.01],
            cache=[-0.01, 0.0, -0.01],
            after=[-0.02, 0.0, -0.01],
        ),
        ParamMerge(param="place_on_base", how="replaced", before=ours, cache=theirs, after=theirs),
    ]
    assert [(t.where, t.how) for t in record.text] == [("lessons.L2", "added")]
    assert lint(doc, recipe.lint) == []
    assert merge(m, cache.doc, 1.0).doc.params["pre_grasp"].value == 0.12  # type: ignore[union-attr]
    assert merge(m, cache.doc, 0.2).doc.params["pre_grasp"].value == 0.104  # type: ignore[union-attr]


POSE = GraspSpec(
    anchor="pose",
    approach_axis=(0.0, 0.0, -1.0),
    closing_axis=(1.0, 0.0, 0.0),
    offset_m=(0.0, 0.0, 0.02),
    reference_q=(1.0, 0.0, 0.0, 0.0),
)


def test_an_unchanged_cache_is_a_jitter(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    cache = cache_of(memory, recipe)
    ok(cache, op="scratch", text="nothing to change; the planner was unlucky")
    merged = merge(memory, cache.doc, 0.75, seed=10009, attempt=2)
    assert merged.doc is None
    assert merged.record.jitter and merged.record.params == [] and merged.record.text == []
    assert merged.record.base == content_hash(memory)
