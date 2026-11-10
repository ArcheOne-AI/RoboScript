"""roboscript.demos.demo: the TCP from the ee pose, and the demo.json + arrays.npz round trip
(nothing pickled)."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from roboscript.demos import (
    CameraCalib,
    CameraTrack,
    Demo,
    FrameSource,
    load_demo,
    save_demo,
    tcp_from_ee,
)
from roboscript.spatial.rotation import quat_from_axis_angle

DOWN = quat_from_axis_angle([0.0, 1.0, 0.0], np.pi / 2)


def test_tcp_is_the_ee_moved_along_its_pointing_axis() -> None:
    ee = np.r_[0.1, 0.2, 0.9, DOWN]
    tcp = tcp_from_ee(ee, (0.12, 0.0, 0.0))
    np.testing.assert_allclose(tcp[:3], [0.1, 0.2, 0.78], atol=1e-12)  # pointing down: 12 cm lower
    np.testing.assert_allclose(tcp[3:], DOWN, atol=1e-12)


def with_camera(demo: Demo) -> Demo:
    n = demo.meta.frames
    source = FrameSource(path=Path("/data/episode0.hdf5"), dataset="obs/head/rgb", codec="jpeg")
    meta = demo.meta.model_copy(
        update={"cameras": {"head": CameraTrack(width=32, height=24, images=source)}}
    )
    calib = {
        "head": CameraCalib(
            K=np.tile(np.eye(3), (n, 1, 1)), cam2world=np.tile(np.eye(4), (n, 1, 1))
        )
    }
    return Demo(meta, demo.arms, calib)


def test_round_trip_through_json_and_npz(
    motion: type, make_demo: Callable[..., Demo], tmp_path: Path
) -> None:
    demo = with_camera(make_demo({"left": motion().rest(5).move((0.1, 0.0, 0.0), 20)}, 5))
    save_demo(demo, tmp_path / "demo")
    assert sorted(p.name for p in (tmp_path / "demo").iterdir()) == ["arrays.npz", "demo.json"]
    assert (
        json.loads((tmp_path / "demo" / "demo.json").read_text(encoding="utf-8"))["schema"]
        == "rs.demo/1"
    )
    back = load_demo(tmp_path / "demo")  # np.load(allow_pickle=False) inside
    assert back.meta == demo.meta
    for arm, track in demo.arms.items():
        for field in ("ee", "tcp", "gripper"):
            np.testing.assert_array_equal(getattr(back.arms[arm], field), getattr(track, field))
        assert back.arms[arm].joints is None
    np.testing.assert_array_equal(back.calib["head"].cam2world, demo.calib["head"].cam2world)
