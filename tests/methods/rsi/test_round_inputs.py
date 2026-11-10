"""What a round reads besides the digests: the demonstration's final keyframe, RGB only, never the
package's truth layer."""

from pathlib import Path

import numpy as np

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.camera import CameraFrame
from roboscript.contracts.ids import FrameId, TaskName
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.demos import BLOBS_DIR, KeypointSet
from roboscript.learning.demo import PACKAGE_FILE, TRUTH_FILE, DemoPackage, save_package
from roboscript.testing.scenes import robot_state
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_methods.rsi.toolset import DEMO_FINAL, _demo_final
from tests.methods.rsi.records import _camera

RULES = load_rsi_recipe(Path("recipes/rsi")).reflect
TASK = TaskName("lift_red")


def _package(directory: Path) -> tuple[DemoPackage, dict[str, object]]:
    """A package of three frames whose success frame is the last; a truth layer beside it."""
    blobs = LocalBlobStore(directory / BLOBS_DIR)
    frames, last = [], {}
    for n in range(3):
        cameras = {}
        for camera in ("head", "wrist"):
            rgb = np.full((2, 2, 3), 10 * n, dtype=np.uint8)
            rgb[0, 0] = (n, 7, 1 if camera == "head" else 2)
            cameras[camera] = CameraFrame(camera=_camera(camera), rgb=blobs.put_image(rgb))
        frame_id = FrameId(f"demo-{n}")
        seen = Observation(frame_id=frame_id, step=15 * n, cameras=cameras, robot=robot_state())
        ref = blobs.put_bytes(seen.model_dump_json().encode(), "application/json")
        frames.append(
            FrameEntry(frame_id=frame_id, alias=f"frame_{n + 1}", step=15 * n, observation=ref)
        )
        last = {name: frame.rgb for name, frame in cameras.items()}
    keypoints = KeypointSet(
        task=TASK, seed=7, frame_steps=15, success_frame=2, config_hash="x", pointing={}, arms={}
    )
    package = DemoPackage(
        task=TASK, seed=7, instruction="raise it", keypoints=keypoints, frames=frames
    )
    save_package(directory, package)
    (directory / TRUTH_FILE).write_text("not json: never read", encoding="utf-8")
    return package, last


def test_the_demonstrations_final_keyframe_is_its_rgb_images_only(tmp_path: Path) -> None:
    directory = tmp_path / "package"
    _, last = _package(directory)
    assert (directory / PACKAGE_FILE).is_file()
    found = _demo_final(directory, ())
    assert [(i.frame, i.camera, i.step, i.frame_id) for i in found] == [
        (DEMO_FINAL, "head", 30, "demo-2"),
        (DEMO_FINAL, "wrist", 30, "demo-2"),
    ]
    assert [i.image for i in found] == [last["head"], last["wrist"]]
    assert all(i.image.media_type == "image/png" for i in found)
    assert [i.camera for i in _demo_final(directory, ["wrist"])] == ["wrist"]
