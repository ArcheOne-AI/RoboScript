"""Embodiment profiles: refusal of bad profiles, ee and TCP conversion both ways, and the first
joint beyond the stops."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from roboscript.contracts.geometry import Pose
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment


def mock_data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(MOCK_BIMANUAL.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["arms"]["left"].update(action=[0, 5]), "does not fit"),
        (lambda d: d["arms"]["right"].update(gripper_index=6), "once each"),
        (lambda d: d.update(action_dim=15), "once each"),
        (lambda d: d["cameras"]["wrist_left"].pop("arm"), "names the arm carrying it"),
        (lambda d: d["cameras"]["overhead"].update(arm="left"), "names the arm carrying it"),
        (lambda d: d["arms"]["left"].update(ee_link=None), "ee_link"),
        (lambda d: d["arms"]["left"].update(home=[0.0] * 5), "home has 5 joints, the arm 6"),
        (lambda d: d["arms"]["left"].update(stops=[[-1, 1]] * 5), "stops has 5 joints, the arm 6"),
        (lambda d: d["arms"]["left"].update(stops=[[1, -1]] * 6), "low < high"),
        (lambda d: d.update(world_axes=" "), "world_axes"),
        (lambda d: d.update(extra=1), "Extra inputs"),
        (
            lambda d: d["arms"]["left"]["gripper"].update(holding_threshold=0.0),
            "holding_threshold 0.0 must lie between the empty reading 0.0 and open 1.0",
        ),
    ],
)
def test_bad_profiles_are_refused(tmp_path: Path, change: Any, message: str) -> None:
    data = mock_data()
    change(data)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message) as caught:
        load_embodiment(path)
    assert str(path) in str(caught.value)


def test_ee_and_tcp_convert_both_ways() -> None:
    profile = load_embodiment(MOCK_BIMANUAL)
    down = Pose(p=(0.1, -0.2, 0.9), q=(0.7071068, 0.0, 0.7071068, 0.0))  # x axis pointing down
    tcp = profile.tcp_pose("left", down)
    assert np.allclose(tcp.p, (0.1, -0.2, 0.8)) and np.allclose(tcp.q, down.q)
    back = profile.ee_pose("left", tcp)
    assert np.allclose(back.as_list(), down.as_list())


def test_past_stops_names_the_first_joint_beyond_them() -> None:
    arm = load_embodiment(MOCK_BIMANUAL).arms["left"]
    assert arm.past_stops([[5.0] * 6]) is None  # no stops: the planner's limits hold
    arm = arm.model_copy(update={"stops": [(-1.0, 1.0)] * 6})
    assert arm.past_stops([[0.0] * 6, [0.5] * 6]) is None
    assert arm.past_stops([[0.0, 0.0, -1.5, 0.0, 2.0, 0.0]]) == (
        "joint 3 would turn to -1.50 rad, past its stop at -1.00"
    )
    # a joint at its stop as the float32 planner returns it is at it, not past it
    wrist = arm.model_copy(update={"stops": [(-10.0, 10.0)] * 3 + [(-1.85, 2.47)] * 3})
    at_stop = [0.0, 0.0, 0.0, float(np.float32(-1.85)), float(np.float32(2.47)), 0.0]
    assert at_stop[3] < -1.85 and wrist.past_stops([at_stop]) is None
    assert wrist.past_stops([[0.0, 0.0, 0.0, -1.851, 0.0, 0.0]]) is not None
