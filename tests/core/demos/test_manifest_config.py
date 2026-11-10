"""The manifest of chosen demonstrations."""

import hashlib
import json
from pathlib import Path

from roboscript.demos import DemoManifest, TaskDemo
from roboscript.demos.manifest import Candidate, file_ref


def test_manifest_round_trip(tmp_path: Path) -> None:
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "keypoints.json").write_text("{}", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-source.hdf5"
    outside.write_bytes(b"hdf5")
    kps = file_ref(tmp_path / "task" / "keypoints.json", tmp_path)
    assert kps.path == "task/keypoints.json"
    assert kps.sha256 == hashlib.sha256(b"{}").hexdigest()
    assert file_ref(outside, tmp_path).path == outside.as_posix()
    choice = Candidate(episode=7, seed=9, rank=1, score=0.1, reasons=["closes left 1, right 0"])
    entry = TaskDemo(
        source="reviewed",
        status="overridden",
        episode=7,
        auto_episode=3,
        seed=9,
        category="C2",
        instruction="Lift the red box.",
        score=0.1,
        reasons=choice.reasons,
        margin=0.02,
        notes=["episode 7 (rank 2) costs 0.30: path 1.40x the task median"],
        keypoint_counts={"left": 5},
        config_hash="c" * 64,
        recording=file_ref(outside, tmp_path),
        demo=[kps],
        keypoints=kps,
        candidates=[choice],
    )
    manifest = DemoManifest(
        created="2026-09-28T12:00:00+08:00",
        source_root="/data",
        categories="roboscript.envs.mock:MOCK",
        tasks={"t": entry},
    )
    text = manifest.model_dump_json()
    assert json.loads(text)["schema"] == "rs.demos/1"
    assert DemoManifest.model_validate_json(text) == manifest
