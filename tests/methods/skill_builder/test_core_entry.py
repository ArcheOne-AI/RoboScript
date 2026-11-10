"""The shared entry check (`skill_builder/core_entry`): its metrics count unplanned and untried
objects as failures."""

from roboscript_methods.skill_builder.core_entry.check import (
    metrics,
)


def report() -> dict[str, object]:
    """Two poses (one deviated), a displacement, a close, three objects: one the demonstration
    picked held (rose 4 cm) and placed, one no arm planned, one the demonstration picked that has
    no box (a cloth)."""
    origin = {"call": "return_home", "ok": True, "within": True, "honest": True}
    place = {"ok": True, "off_m": 0.01, "dz_m": 0.001, "tilt_deg": 40.0}
    place |= {"placed": True, "upright": False, "honest": True}
    grasp = {"ok": True}
    return {
        "task": "t",
        "move_to_pose": [
            {"pos_m": 0.002, "deg": 1.0, "ok": True, "honest": True},
            {"pos_m": 0.03, "deg": 5.0, "ok": False, "honest": True},
            origin,
        ],
        "displacement": [{"error_m": 0.003, "ok": True, "honest": True}, origin],
        "gripper": [
            {"close": False, "reading": 0.95, "reached": True, "ok": True, "honest": True},
        ],
        "grasp": [
            {
                "label": "cup",
                "picked": True,
                "planned": True,
                "rose_m": 0.04,
                "held": False,
                "grasp_honest": True,
                "grasp": grasp,
                "lift": grasp,
                "place": place,
                "return_home": origin,
            },
            {"label": "box", "planned": False, "held": False, "return_home": origin},
            {"label": "cloth", "picked": True, "skipped": "no box"},
        ],
    }


def test_metrics_count_unplanned_and_untried_objects_as_failures() -> None:
    found = metrics(report())
    assert found["poses_ok"] == "1/2"
    assert found["grasp_success"] == "0/2"  # as judged: the cup's 4 cm was not held at 5 cm
    assert found["demo_picked"] == "0/2"  # the cloth could not be tried: a failure too
    assert found["return_home"] == "4/4"
