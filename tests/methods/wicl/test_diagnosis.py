"""learning.diagnosis: how the Reviser reads a try's ground truth (`roles.diagnosis_text`): the
benchmark's success checks never reach it."""

from roboscript.contracts.env import Diagnosis, ObjectTruth
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.ids import EpisodeId, FrameId
from roboscript.perception.registry import SceneRegistry
from roboscript.testing.scenes import TABLE_Z, Box, surface
from roboscript_methods.wicl.diagnosis import explain
from roboscript_methods.wicl.roles import diagnosis_text

F1 = FrameId("frame-1")
SIZE = 0.04
UP = (1.0, 0.0, 0.0, 0.0)
MID = TABLE_Z + SIZE / 2  # a block's center height
ALIASES = {"o1": "obj_1", "o2": "obj_2", "o3": "obj_3"}


def registry(**roles: tuple[float, float]) -> SceneRegistry:
    """Blocks registered for `roles`, on the table at (x, y)."""
    found = SceneRegistry()
    for role, (x, y) in roles.items():
        points = surface(Box((x, y, MID), (SIZE, SIZE, SIZE)))
        found.register(role, points, frame=F1, step=0, table_z=TABLE_Z)
    return found


def truth(name: str, start: Vec3, end: Vec3 | None = None, **more: object) -> ObjectTruth:
    """A task object whose pose sits at its center; the referee's centers stand apart only in
    `test_without_centers_the_poses_stand_in`."""
    end = start if end is None else end
    return ObjectTruth.model_validate(
        {
            "name": name,
            "initial": Pose(p=start, q=UP),
            "final": Pose(p=end, q=UP),
            "displacement_m": 0.0,
            "initial_center": start,
            "final_center": end,
            **more,
        }
    )


def diagnosis(*objects: ObjectTruth) -> Diagnosis:
    return Diagnosis(episode_id=EpisodeId("env-1"), success=False, objects=list(objects), notes=[])


def boxes(**roles: Box) -> SceneRegistry:
    found = SceneRegistry()
    for role, box in roles.items():
        found.register(role, surface(box), frame=F1, step=0, table_z=TABLE_Z)
    return found


def test_the_benchmarks_success_checks_never_reach_the_reviser() -> None:
    """`Diagnosis.conditions` is P1's build only: the checks name the
    simulator's objects and their margins are the official tolerances; the Reviser's reading
    (`explain`, `diagnosis_text`) carries none of it."""
    from roboscript.contracts.env import ConditionMargin, ConditionTruth

    registered = registry(tool=(0.10, 0.0))
    ground = diagnosis(truth("hammer", (0.10, 0.0, MID))).model_copy(
        update={
            "conditions": [
                ConditionTruth(
                    stage="pending 1/1",
                    check="is_A_in_B(label_A=hammer, label_B=crate_tol)",
                    holds=False,
                    margins=[ConditionMargin(arg="dis_threshold", value=0.0123, flips_at=0.0456)],
                )
            ]
        }
    )
    explained = explain(ground, registered, ALIASES.__getitem__, robot=None)
    shown = explained.model_dump_json() + "\n".join(diagnosis_text(explained))
    for leaked in ("is_A_in_B", "crate_tol", "dis_threshold", "0.0123", "0.0456", "pending"):
        assert leaked not in shown
