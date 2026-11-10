"""The Memory document: memory.md and params.json round trips, the content hash, and what models
read."""

import json
import re
from pathlib import Path

import pytest

from roboscript.memory import (
    GraspSpec,
    MemoryDoc,
    MemoryRecipe,
    PlaceSpec,
    PlaceTarget,
    content_hash,
    dump_params,
    load_params,
    open_cache,
    parse_memory,
    read_doc,
    render_memory,
    view,
    write_doc,
)
from roboscript.testing.memory import sample_memory

MERGED = {
    "version": 3,
    "stage": "S2",
    "created": "2026-09-28T12:00:00Z",
    "source": "merge into v002 from seed 10004 try 2",
}


def cache_of(doc: MemoryDoc) -> MemoryDoc:
    return doc.replace(scratch=["suspect the handle is estimated short", "try the head side"])


def test_render_and_parse_are_inverse(memory: MemoryDoc) -> None:
    for doc in (
        memory,
        memory.replace(**MERGED),
        memory.replace(source="-"),
        cache_of(memory),
        MemoryDoc(task="t"),
    ):
        text = render_memory(doc)
        assert parse_memory(text, doc.params) == doc
        assert render_memory(parse_memory(text, doc.params)) == text


HEADER = "\n# TASK MEMORY: t   (draft, S1, -)\n"


def test_params_json_round_trip(memory: MemoryDoc, tmp_path: Path) -> None:
    text = dump_params(memory)
    data = json.loads(text)
    assert data["schema"] == "rs.memory.params/1" and list(data["params"]) == list(memory.params)
    assert data["params"]["grasp_handle"]["value"]["offset_m"] == [-0.03, 0.0, -0.01]
    assert load_params(text) == memory.params
    write_doc(tmp_path, memory)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["memory.md", "params.json"]
    assert read_doc(tmp_path) == memory
    with pytest.raises(ValueError, match=re.escape("is not 'rs.memory.params/1'")):
        load_params(json.dumps({"schema": "x", "params": {}}))
    with pytest.raises(ValueError, match="keys 'schema' and 'params'"):
        load_params("[]")
    (tmp_path / "memory.md").write_text("junk", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path}: line 1")):
        read_doc(tmp_path)


def test_the_content_hash_covers_content_not_scratch(memory: MemoryDoc) -> None:
    digest = content_hash(memory)
    assert len(digest) == 64 and content_hash(sample_memory()) == digest
    assert content_hash(cache_of(memory)) == digest
    assert content_hash(memory.replace(notes="Measure twice.")) != digest
    assert content_hash(memory.replace(version=1)) != digest


def test_the_actor_reads_no_provenance(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    m = memory.replace(**MERGED, statement="Beat the base with the tool.")
    actor = view(m, detail=False)
    assert actor.startswith("# TASK MEMORY: lift_red\n\n## TASK  [stable]\nBeat the base")
    titles = ["TASK", "OBJECTS", "WORKFLOW", "PARAMETERS", "LESSONS", "NOTES"]
    at = [actor.index(f"\n## {title}") for title in titles]
    assert at == sorted(at)  # reading order: the parameters after the workflow naming them
    assert "\n## PARAMETERS\n- pre_grasp = 0.1 m\n" in actor
    assert "- L1\n  when: the handle points away from the robot\n" in actor
    assert "INSTRUCTION" not in actor and memory.instruction not in actor
    assert "INSTRUCTION" not in view(m) and memory.instruction not in view(m)
    for hidden in ("seed", "merge", "v003", "2026", "bounds", "trust", "from S1", "SCRATCH"):
        assert hidden not in actor, hidden
    assert (
        "- grasp_handle = grasp_spec(part=handle, approach=top, anchor=top_center, "
        "x_toward=head, offset_m=[-0.03, 0, -0.01], closing=minor, yaw_deg=0)\n"
    ) in actor
    cache = open_cache(m, budget=4, seed=10007, rules=recipe.rules("s2"), lint=recipe.lint)
    assert cache.edit({"op": "scratch", "text": "the planner took a wide arc"}).accepted
    assert view(cache.doc, detail=False) == actor  # a jitter measures the planner, nothing else


POSE_GRASP = GraspSpec(
    anchor="pose",
    approach_axis=(0.0018, -0.8547, -0.5191),
    closing_axis=(0.0035, 0.5191, -0.8547),
    reference_q=(0.144, 0.519, -0.811, -0.228),
    x_toward="far_end",
    offset_m=(0.01, -0.02, 0.003),
    purpose="pull",
    arm="left",
)
FULL_GRASP = GraspSpec(
    part="handle",
    approach="side",
    anchor="apex",
    x_toward="head",
    offset_m=(-0.03, 0.0, -0.01),
    offset_frac=(0.25, -0.1, 0.0),
    closing="major",
    yaw_deg=15.0,
    tilt_deg=-10.0,
    lean_deg=5.0,
    purpose="press",
    reference_q=(0.7071, 0.0, 0.7071, 0.0),
    arm="right",
)
FULL_PLACE = PlaceSpec(
    target=PlaceTarget(kind="object", obj="rack", part="pin", anchor="edge:front"),
    frame="world_axes",
    offset_m=(0.0, 0.02, 0.05),
    orient="relative",
    yaw_deg=30.0,
    rpy_deg=(0.0, 90.0, -45.0),
    placement_point="part:loop",
    placement_point_at=(0.5, -0.25, 1.0),
    x_toward="handle",
    approach=(0.0, 0.0, -1.0),
    free_turn_deg=45.0,
    region_m=0.03,
    receiver=POSE_GRASP,
    arm="left",
    mirror=True,
)
SPECS = {
    "pose_grasp": POSE_GRASP,
    "full_grasp": FULL_GRASP,
    "plain_grasp": GraspSpec(),
    "full_place": FULL_PLACE,
    "landmark_place": PlaceSpec(target=PlaceTarget(kind="landmark", landmark="front_edge_mid")),
    "robot_place": PlaceSpec(target=PlaceTarget(kind="robot_frame"), offset_m=(0.3, 0.0, 0.1)),
}
