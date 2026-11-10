"""Each task's demonstration as P1-3 reads it, from the re-recorded demonstrations
(`scripts/record_demos.py`'s `<recorded>/<task>/`; with `--transfer`, its
`<recorded>/transfer/<task>/<seed>/`): the core's demonstration format (`roboscript.demos.
save_demo`: the arm tracks, the ee pose measured, the joints and gripper as commanded; without
the frames' observations) and, beside it, `truth.npz` (each object's recorded pose per frame,
`object.<label>`, and the joint-position commands a replay sends, `commands`: `commands[t]` is
what frame t + 1 was commanded) and `source.json` (the recording, its seed and episode, each
object's category). `--transfer` also writes a task's transfer demonstrations, by seed, to
`<task>/transfer/<k>-<seed>/`: the gate's transfer cases (`cover_demos.py --transfer`; never a
P2 input). Only accepted recordings are exported (recorded, their validation passed); a transfer
seed must lie in the demonstration block and not be the chosen one's (`record_demos.DEMO_SEEDS`).

    python demo_export.py --recorded <demos>/recorded --out <recordings> [--tasks a,b] [--transfer]

`recorded(directory)` reads an export back (`cover_demos.py`, `entry_check.py`).
"""

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np

from roboscript.contracts.ids import TaskName
from roboscript.demos import load_demo, load_truth, save_demo
from roboscript.demos.demo import Demo
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning.seeds import SeedRange
from roboscript_methods.skill_builder.demo_coverage.segments import Recording
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.demos.records import read
from roboscript_robotwin.demos.validate import VALIDATION, Validation

DEMO_SEEDS: Final = SeedRange(start=0, count=68)
"""The demonstration block of the seed split (`record_demos.DEMO_SEEDS`): a transfer
demonstration's seed lies in it, so it never touches a learning or evaluation seed."""
TRUTH: Final = "truth.npz"
SOURCE: Final = "source.json"


class NotAccepted(ValueError):
    """A recording that is not an accepted one: not recorded, or its validation did not pass."""


def commands(demo: Demo, profile: EmbodimentProfile) -> np.ndarray:
    """The joint-position step sent after each frame but the last, `(frames - 1, action_dim)`:
    what each arm's joints and gripper were commanded at the next frame (a re-recording sends one
    action per frame, `roboscript_robotwin.sim.record`)."""
    found = np.zeros((demo.meta.frames - 1, profile.action_dim))
    for arm, spec in profile.arms.items():
        track = demo.arms[arm]
        if track.joints is None:
            raise ValueError(f"{demo.meta.task}: the recording has no joint commands for {arm}")
        found[:, spec.action[0] : spec.action[1]] = track.joints[1:]
        found[:, spec.gripper_index] = track.gripper[1:]
    return found


def accepted(directory: Path) -> int:
    """The seed of the accepted recording in `directory` (`record_demos.py`'s task layout)."""
    record = read(directory)
    if record is None or not record.recorded:
        raise NotAccepted(f"{directory}: no recording")
    path = directory / "referee" / VALIDATION
    checked = Validation.model_validate_json(path.read_text("utf-8")) if path.is_file() else None
    if checked is None or not checked.passed:
        raise NotAccepted(f"{directory}: its validation did not pass")
    return record.seed


def export(directory: Path, out: Path, profile: EmbodimentProfile) -> None:
    """The accepted recording in `directory` written to `out` as P1-3 reads it."""
    seed = accepted(directory)
    record = read(directory)
    assert record is not None
    demo = load_demo(directory / "recording")
    truth = load_truth(directory / "referee")
    if len(truth.poses) != demo.meta.frames:
        raise ValueError(f"{directory}: {len(truth.poses)} truth frames, {demo.meta.frames} demo")
    meta = demo.meta.model_copy(update={"observations": False})
    save_demo(Demo(meta, demo.arms, demo.calib, demo.source_frames, demo.physics_steps), out)
    arrays = {f"object.{name}": truth.poses[:, k] for k, name in enumerate(truth.objects)}
    np.savez(out / TRUTH, commands=commands(demo, profile), **arrays)
    source = {
        "recording": str(directory),
        "seed": seed,
        "episode": record.episode,
        "categories": truth.categories,
        "articulated": sorted(truth.joints),
    }
    (out / SOURCE).write_text(json.dumps(source, indent=1) + "\n", encoding="utf-8")


def recorded(found: Path) -> Recording:
    """A demonstration exported to `found`: the core's `Demo`, each object's recorded pose per
    frame, the commands a replay sends, where it came from."""
    with np.load(found / TRUTH, allow_pickle=False) as truth:
        objects = {k.split(".", 1)[1]: truth[k] for k in truth.files if k.startswith("object.")}
        sent = truth["commands"]
    source = json.loads((found / SOURCE).read_text(encoding="utf-8"))
    return Recording(load_demo(found), objects, sent, source)


def transfer_dirs(recorded_dir: Path, task: str) -> list[Path]:
    """A task's transfer recordings (`<recorded>/transfer/<task>/<seed>/`), by seed."""
    found = [d for d in (recorded_dir / "transfer" / task).glob("*") if d.name.isdigit()]
    return sorted(found, key=lambda d: int(d.name))


def export_task(recorded_dir: Path, out: Path, task: str, transfer: bool) -> list[int]:
    """The task's chosen demonstration and (`transfer`) its accepted transfer demonstrations:
    the transfer seeds written."""
    profile = ROBOTWIN.profile()
    export(recorded_dir / task, out / task, profile)
    chosen = accepted(recorded_dir / task)
    written: list[int] = []
    for directory in transfer_dirs(recorded_dir, task) if transfer else []:
        try:
            seed = accepted(directory)
        except NotAccepted:  # a candidate record_demos tried and rejected
            continue
        if not DEMO_SEEDS.holds(seed) or seed == chosen:
            raise ValueError(f"{directory}: seed {seed} is not a transfer seed (demonstration "
                             f"block {DEMO_SEEDS.start}-{DEMO_SEEDS.start + DEMO_SEEDS.count - 1},"
                             f" chosen {chosen})")  # fmt: skip
        export(directory, out / task / "transfer" / f"{len(written):02d}-{seed}", profile)
        written.append(seed)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recorded", type=Path, required=True, help="record_demos.py's --out")
    parser.add_argument("--out", type=Path, required=True, help="the exported recordings")
    parser.add_argument("--tasks", help="comma-separated; default: every recorded task")
    parser.add_argument("--transfer", action="store_true", help="also the transfer recordings")
    args = parser.parse_args(argv)
    source, out = args.recorded.resolve(), args.out.resolve()
    tasks = (
        args.tasks.split(",")
        if args.tasks
        else sorted(d.name for d in source.iterdir() if d.name in ROBOTWIN.tasks)
    )
    for task in tasks:
        seeds = export_task(source, out, TaskName(task), args.transfer)
        print(f"{task}: written" + (f", transfer seeds {seeds}" if args.transfer else ""))


if __name__ == "__main__":
    main()
