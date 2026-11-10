"""The OBJECTS section's role lines and their optional `x toward:` lines."""

from roboscript.memory import (
    MemoryDoc,
    Order,
    by_order,
    object_orders,
    ranks,
)
from roboscript.memory.objects import directed_arm, pointed_arm, side_rule


def test_order_lines_bind_look_alikes_by_where_they_lie() -> None:
    text = (
        "- base_bowl: stacked on; a bowl\n"
        "  order: 1 among bowls by distance to the robot\n"
        "- top_bowl: stacked last; a bowl\n"
        "  x toward: the rim\n"
        "  order: 2 among bowls by distance to the robot\n"
    )
    doc = MemoryDoc(task="t", objects=text)
    orders = object_orders(doc)
    assert orders == {
        "base_bowl": Order(1, "bowls", "distance to the robot"),
        "top_bowl": Order(2, "bowls", "distance to the robot"),
    }
    starts = {"o1": (0.0, 0.1, 0.76), "o2": (0.05, -0.2, 0.76), "o3": (0.3, 0.3, 0.76)}
    base = (0.0, -0.6, 0.0)
    # the identification bound them the wrong way round; the order swaps them
    bound = {"base_bowl": "o1", "top_bowl": "o2", "other": "o3"}
    assert by_order(orders, bound, starts, (-1.0, 0.0, 0.0), base) == {
        "base_bowl": "o2",
        "top_bowl": "o1",
        "other": "o3",
    }
    sizes = {"o1": (0.12, 0.12, 0.05), "o2": (0.125, 0.12, 0.05), "o3": (0.2, 0.1, 0.05)}
    assert ranks(starts, sizes, "o2", (-1.0, 0.0, 0.0), base) == {
        "distance to the robot": (1, 2),
        "the robot's left to right": (2, 2),  # left is -x here: o1 lies further left
    }
    assert ranks(starts, sizes, "o3", (-1.0, 0.0, 0.0), base) is None


def test_an_arm_side_gives_the_arm_by_where_x_points() -> None:
    left = (0.0, 1.0, 0.0)  # the robot's left in the world
    assert pointed_arm((0.0, 1.0, 0.0), left) == "left"
    assert pointed_arm((0.3, -0.9, 0.0), left) == "right"
    assert pointed_arm((1.0, 0.1, 0.0), left) is None  # straight ahead: no side
    assert pointed_arm((0.0, 0.0, 1.0), left) is None  # up
    assert directed_arm("toward its x", (0.2, -1.0, 0.0), left) == "right"
    assert directed_arm("away from its x", (0.2, -1.0, 0.0), left) == "left"
    assert directed_arm("either", (0.2, -1.0, 0.0), left) is None
    # The rule the demonstration's arm follows, and back: mirrored with the object.
    assert side_rule("left", (0.0, 1.0, 0.0), left) == "toward its x"
    assert side_rule("left", (0.0, -1.0, 0.0), left) == "away from its x"
    for x in ((0.1, 1.0, 0.0), (0.1, -1.0, 0.0)):
        for arm in ("left", "right"):
            rule = side_rule(arm, x, left)
            assert rule is not None and directed_arm(rule, x, left) == arm
            mirrored = (x[0], -x[1], x[2])
            other = "right" if arm == "left" else "left"
            assert directed_arm(rule, mirrored, left) == other
