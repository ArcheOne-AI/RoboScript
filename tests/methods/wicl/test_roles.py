"""Role prompts and briefs: what each role reads of the Memory, and the demonstration brief
without coordinates."""

from pathlib import Path

import pytest

from roboscript.contracts.geometry import Pose
from roboscript.contracts.tools import TextPart
from roboscript.demos import Keypoint, KeypointSet, Tolerance
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MOCK
from roboscript.learning import DemoPackage
from roboscript.memory import MemoryRecipe, open_cache
from roboscript.testing.protocols import LIFT_RED, lift_memory
from roboscript.tools.skillsets import guides
from roboscript_methods.wicl.roles import (
    Prompts,
    demo_brief,
)

PROMPTS = Path(__file__).resolve().parents[3] / "recipes" / "wicl" / "prompts"
PROFILE = load_embodiment(MOCK_BIMANUAL)
TCP = Pose(p=(0.1234, -0.0567, 0.8901), q=(0.707, 0.0, 0.707, 0.0))


def test_each_role_reads_the_memory_its_way(recipe: MemoryRecipe) -> None:
    prompts = Prompts(PROMPTS, guides=guides(MOCK.skillsets))  # the Mock benchmark's guide
    memory = lift_memory(0.02)
    actor = prompts.actor(instruction="Lift it.", memory=memory, rules="s2", profile=PROFILE)
    assert "Lift it." in actor and "{{" not in actor
    assert "- lift_height = 0.02 m" in actor and "trust" not in actor and "(draft" not in actor
    assert "0.1 m ahead of the ee" in actor  # the mock robot's TCP offset
    assert f"World frame: {PROFILE.world_axes}." in actor  # and its words for the axes
    reader = prompts.demo_reader(instruction="Lift it.", memory=memory, profile=PROFILE)
    assert "trust ±0.05" in reader
    cache = open_cache(memory, budget=3, seed=10000, rules=recipe.rules("s2"), lint=recipe.lint)
    reviser = prompts.reviser(instruction="Lift it.", editor=cache, profile=PROFILE)
    assert "## SCRATCH" in reviser and "3 of 3 budget units are left" in reviser
    assert len(prompts.digest) == 64 and prompts.stage_rules["s1_hinted"]
    other = Prompts(PROMPTS, guides=["World frame: {{world_axes}}."])
    assert other.digest != prompts.digest  # the guides are part of the version


def test_the_demonstration_brief_has_no_coordinates() -> None:
    kps = [
        Keypoint(
            index=1,
            frame=1,
            tcp=TCP,
            gripper="open",
            cue="stop",
            phase="approach",
            tolerance=Tolerance(),
        ),
        Keypoint(
            index=2,
            frame=3,
            tcp=TCP,
            gripper="closed",
            cue="close",
            phase="grasp",
            tolerance=Tolerance(angle_deg=15),
        ),
    ]
    keypoints = KeypointSet(
        task=LIFT_RED,
        seed=7,
        frame_steps=1,
        success_frame=3,
        config_hash="x",
        pointing={"right": (1.0, 0.0, 0.0)},
        arms={"right": kps},
    )
    package = DemoPackage(
        task=LIFT_RED, seed=7, instruction="lift the red box", keypoints=keypoints
    )
    [text] = demo_brief(package)
    assert isinstance(text, TextPart)
    assert "Instruction: lift the red box" in text.text
    assert "K1 approach: the arm comes to rest, gripper open after it" in text.text
    assert "K2 grasp: the gripper closes" in text.text
    assert "15 deg" not in text.text and "within" not in text.text  # tolerances: the referee's
    assert "0.123" not in text.text and "0.89" not in text.text and "seed" not in text.text
    with pytest.raises(ValueError, match="not lift_red seed 8"):
        DemoPackage(task=LIFT_RED, seed=8, instruction="x", keypoints=keypoints)
    with pytest.raises(ValueError, match="no success frame"):
        bare = keypoints.model_copy(update={"success_frame": None})
        DemoPackage(task=LIFT_RED, seed=7, instruction="x", keypoints=bare)
