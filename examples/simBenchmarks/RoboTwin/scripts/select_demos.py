"""Choose one expert demonstration per RoboTwin task, offline (docs/concepts/learning.md).

Every episode of the tasks under the expert data root is read, its keypoints extracted and its
quality scored (`roboscript.demos`); the best one is picked and the top `selection.keep` are shown
for review. The keypoints serve this choice only: the official recordings have no success frame,
so the keypoints S1 judges by come from re-recording the chosen episode. Written to --out:

    manifest.json                     the choice per task, why, and the files with their hashes
    index.html                        all tasks; export a human's picks as reviewed.json
    <task>/keypoints.json             the chosen episode's keypoints
    <task>/demo/                      the chosen episode as a Demo (demo.json + arrays.npz)
    <task>/scores.json                every candidate's rank, score and criteria
    <task>/review.html                the shown candidates side by side
    <task>/candidates/episode<i>/     their video, keypoints.json and frames with keypoints drawn

A run with --task updates those tasks and keeps the others in the manifest. Picks from the review
pages come back through --reviewed (JSON {task: episode}) and win (`source` `reviewed`): the
automatic pick's episode confirms it, another overrides it.

    python examples/simBenchmarks/RoboTwin/scripts/select_demos.py --out /path/to/data/demos \\
        [--task beat_block_hammer ...] [--expert-root DIR] [--reviewed reviewed.json] \\
        [--set keypoints.tolerance.xy_m=0.04 ...]

Needs h5py and Pillow (the simulator env has both) and packages/benchmarks/robotwin/src on
PYTHONPATH; CPU only. Settings come from configs/demos.yaml (copy
configs/demos.example.yaml); task categories, which set the orientation tolerance, and the
embodiment from `roboscript_robotwin.benchmark:ROBOTWIN`. The expert data root defaults to
RS_EXPERT_DEMO_ROOT.
"""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Final

from pydantic import NonNegativeInt, TypeAdapter

from roboscript.contracts.geometry import Vec3
from roboscript.contracts.ids import TaskName
from roboscript.demos import (
    DemosConfig,
    KeypointSet,
    demo_keypoints,
    episode_features,
    rank_episodes,
    save_demo,
)
from roboscript.demos.manifest import Candidate, DemoManifest, TaskDemo, file_ref
from roboscript.demos.selection import margin, review_notes
from roboscript.embodiments import EmbodimentProfile
from roboscript.settings import config_hash, load_config
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import expert_demo_root
from roboscript_robotwin.demos.expert import SETTING, load_episode, seeds, video_path
from roboscript_robotwin.demos.review import Shown, index_page, render_candidate, task_page

CATEGORIES: Final = "rs_robotwin.benchmark:ROBOTWIN"
# aloha_agilex's two arms mirror each other across the robot's midline, the plane x = 0.
MIRROR: Final[Vec3] = (1.0, 0.0, 0.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", action="append", default=[], help="a task to process (default: all)")
    ap.add_argument("--expert-root", type=Path)
    ap.add_argument("--reviewed", type=Path, help="the review pages' picks (reviewed.json)")
    ap.add_argument("--set", action="append", default=[], help="a config override key=value")
    args = ap.parse_args(argv)

    config = load_config("demos", DemosConfig, overrides=args.set)
    root = args.expert_root or expert_demo_root()
    if root is None:
        ap.error("set RS_EXPERT_DEMO_ROOT (.env) or pass --expert-root")
    tasks = args.task or sorted(p.name for p in root.iterdir() if (p / SETTING).is_dir())
    if missing := [t for t in tasks if not (root / t / SETTING).is_dir()]:
        ap.error(f"no expert episodes under {root} for {missing}")
    if unknown := sorted(set(tasks) - set(ROBOTWIN.tasks)):
        ap.error(f"{CATEGORIES} does not know {unknown}")
    reviewed: dict[str, int] = {}
    if args.reviewed:
        text = args.reviewed.read_bytes()
        reviewed = TypeAdapter(dict[str, NonNegativeInt]).validate_json(text, strict=True)
    for task, episode in reviewed.items():
        if task not in tasks or episode >= len(seeds(root, task)):
            ap.error(f"{args.reviewed}: {task} has no episode {episode} among the tasks processed")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[TaskName, TaskDemo] = {}
    if (out / "manifest.json").is_file():
        earlier = DemoManifest.model_validate_json((out / "manifest.json").read_text("utf-8"))
        if earlier.source_root != str(root):
            ap.error(f"{out} holds choices from {earlier.source_root}, not {root}")
        entries = dict(earlier.tasks)

    profile = ROBOTWIN.profile()
    for task in tasks:
        e = choose(task, root, out, config, profile, reviewed.get(task))
        entries[TaskName(task)] = e
        counts = " ".join(f"{a}:{n}" for a, n in e.keypoint_counts.items())
        gap = "-" if e.margin is None else f"{e.margin:.2f}"
        print(
            f"{task:28s} episode {e.episode:2d} seed {e.seed:3d} score {e.score:5.2f} margin {gap}"
            f" {e.status} keypoints {counts}" + "".join(f"\n    note: {n}" for n in e.notes),
            flush=True,
        )
    manifest = DemoManifest(
        created=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        source_root=str(root),
        categories=CATEGORIES,
        tasks=dict(sorted(entries.items())),
    )
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", "utf-8")
    (out / "index.html").write_text(index_page(manifest), "utf-8")
    return 0


def choose(
    task: str,
    root: Path,
    out: Path,
    config: DemosConfig,
    profile: EmbodimentProfile,
    reviewed: int | None,
) -> TaskDemo:
    category = ROBOTWIN.tasks[TaskName(task)].category
    tolerance = config.keypoints.tolerance_for(category)
    demos, keysets, features = {}, {}, []
    for episode in range(len(seeds(root, task))):
        demo = load_episode(root, task, episode, profile)
        keysets[episode] = demo_keypoints(demo, config.keypoints, tolerance)
        features.append(episode_features(demo, keysets[episode]))
        demos[episode] = demo
    ranking = rank_episodes(features, config.selection, MIRROR)
    auto = ranking[0]
    pick = next(r for r in ranking if r.episode == (auto.episode if reviewed is None else reviewed))
    shown = ranking[: config.selection.keep]
    if pick not in shown:  # best first: the chosen episode ranks lower than the ones shown
        shown = [*shown, pick]

    task_out = out / task
    if (task_out / "candidates").is_dir():  # an earlier run's, maybe of other episodes
        shutil.rmtree(task_out / "candidates")
    views = []
    for r in shown:
        directory = task_out / "candidates" / f"episode{r.episode}"
        frames = render_candidate(demos[r.episode], keysets[r.episode], directory)
        shutil.copyfile(video_path(root, task, r.episode), directory / "video.mp4")
        _write_json(directory / "keypoints.json", keysets[r.episode])
        views.append(
            Shown(r, keysets[r.episode], directory.relative_to(task_out).as_posix(), frames)
        )
    demo = demos[pick.episode]
    save_demo(demo, task_out / "demo")
    _write_json(task_out / "keypoints.json", keysets[pick.episode])
    (task_out / "scores.json").write_text(
        json.dumps([r.model_dump() for r in ranking], indent=1) + "\n", "utf-8"
    )
    status = "auto" if reviewed is None else "overridden"
    if reviewed == auto.episode:
        status = "confirmed"
    entry = TaskDemo(
        source="auto" if reviewed is None else "reviewed",
        status=status,
        episode=pick.episode,
        auto_episode=auto.episode,
        seed=pick.seed,
        category=category,
        instruction=demo.meta.instructions[0],
        score=pick.score,
        reasons=pick.reasons,
        margin=margin(ranking),
        notes=review_notes(ranking, features, config.selection, pick.episode),
        keypoint_counts={arm: len(kps) for arm, kps in keysets[pick.episode].arms.items()},
        config_hash=config_hash(config),
        recording=file_ref(demo.meta.provenance.path, out),
        demo=[file_ref(task_out / "demo" / name, out) for name in ("demo.json", "arrays.npz")],
        keypoints=file_ref(task_out / "keypoints.json", out),
        candidates=[
            Candidate(episode=r.episode, seed=r.seed, rank=r.rank, score=r.score, reasons=r.reasons)
            for r in shown
        ],
    )
    (task_out / "review.html").write_text(task_page(task, entry, views, ranking), "utf-8")
    return entry


def _write_json(path: Path, keypoints: KeypointSet) -> None:
    path.write_text(keypoints.model_dump_json(indent=1) + "\n", "utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
