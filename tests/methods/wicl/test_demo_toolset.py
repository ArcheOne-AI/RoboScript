"""The demo toolset over the Hub: `demo_relation` on an episode seeded as S1 seeds the Demo
Reader's (the recording's frames and the `DEMO` fact) with objects registered in it, as the model
reads it (aliases, specs, what they do not reproduce), what it refuses, and who sees it."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.geometry import Pose
from roboscript.contracts.hub import CallOutcome, EpisodeSeed, OpenEpisode
from roboscript.contracts.ids import EpisodeId, FrameId, TaskName
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.contracts.tools import TextPart, ToolCall
from roboscript.demos import Keypoint, KeypointSet, Tolerance, frame_alias
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.manipulation import GraspSpec, ObjectProxy, grasp_candidates
from roboscript.perception.registry import SceneRegistry
from roboscript.records import EventWriter
from roboscript.testing.phases import TEST_PHASES
from roboscript.testing.scenes import TABLE_Z, Box, robot_state, surface, table_model
from roboscript.tools import DEMO, DEMO_FITS, TABLE, LocalProvider
from roboscript.toolsets.privileged import Privileged
from roboscript_methods.wicl.toolsets.demo import Demo, DemoRelation

Model = Callable[..., ToolCall]
TO_GRASP = np.array([[0.0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
GRASP = GraspSpec(approach="top", anchor="top_center", closing="minor")


def text_of(outcome: CallOutcome) -> str:
    return "\n".join(part.text for part in outcome.content if isinstance(part, TextPart))


def error_text(outcome: CallOutcome) -> str:
    assert outcome.is_error, outcome
    return text_of(outcome)


async def demonstration(blobs: LocalBlobStore) -> tuple[EpisodeSeed, SceneRegistry]:
    """A left-arm pick of a long box (`tool`) set down on a flat one (`goal`), 5 cm to the right
    of its middle: the seed (the frames at its keypoints and the `DEMO` fact) and the registry
    of the scene it started from."""
    registry = SceneRegistry()
    for name, box in (
        ("tool", Box((0.1, 0.0, TABLE_Z + 0.02), (0.12, 0.05, 0.04), yaw_deg=20)),
        ("goal", Box((-0.15, 0.1, TABLE_Z + 0.01), (0.2, 0.12, 0.02))),
    ):
        points = surface(box)
        registry.register(
            name,
            points[points[:, 2] > TABLE_Z + 1e-6],
            frame=FrameId("f0"),
            step=0,
            table_z=TABLE_Z,
        )
    tool = registry.named("tool")
    assert tool is not None
    proxy = ObjectProxy.from_geometry(registry.geometry(tool.id))
    grasp = proxy.pose @ grasp_candidates(GRASP, proxy, "left")[0].matrix @ TO_GRASP
    carry = grasp.copy()
    carry[:3, 3] += (-0.2, 0.1, 0.02)  # onto the goal, 5 cm to the right of its middle
    moments = [(10, "stop", "approach", grasp, False), (20, "close", "grasp", grasp, True)]
    moments += [(30, "stop", "place", carry, True), (40, "open", "release", carry, False)]
    keypoints = KeypointSet(
        task="lift_red",
        seed=0,
        frame_steps=1,
        success_frame=40,
        config_hash="test",
        pointing={"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)},
        arms={
            "left": [
                Keypoint(
                    index=i,
                    frame=frame,
                    tcp=Pose.from_matrix(tcp),
                    gripper="closed" if closed else "open",
                    cue=cue,  # type: ignore[arg-type]
                    phase=phase,  # type: ignore[arg-type]
                    tolerance=Tolerance(),
                )
                for i, (frame, cue, phase, tcp, closed) in enumerate(moments, 1)
            ]
        },
    )
    entries = []
    for frame, _, _, tcp, closed in moments:
        state = robot_state(step=frame, tcp={"left": tcp[:3, 3]})
        left = state.arms["left"].model_copy(
            update={"tcp": Pose.from_matrix(tcp), "gripper": 0.5 if closed else 1.0}
        )
        seen = Observation(
            frame_id=FrameId(f"f{frame}"),
            step=frame,
            cameras={},
            robot=state.model_copy(update={"arms": {**state.arms, "left": left}}),
        )
        blob = await blobs.aput_bytes(seen.model_dump_json().encode(), "application/json")
        entries.append(
            FrameEntry(
                frame_id=seen.frame_id, alias=frame_alias(frame), step=frame, observation=blob
            )
        )
    seed = EpisodeSeed(frames=entries, facts={DEMO: keypoints.model_dump(mode="json")})
    return seed, registry


async def opened(
    hub: Hub, seed: EpisodeSeed, registry: SceneRegistry
) -> tuple[EpisodeId, dict[str, str]]:
    spec = OpenEpisode(
        stage="s1", role="demo_reader", embodiment=str(MOCK_BIMANUAL), structured=True, seed=seed
    )
    ep = (await hub.open_episode(spec)).episode_id
    state = hub.store.get(ep)
    async with state.frames.edit_registry() as registered:
        registered.objects.update(registry.objects)
        registered.counter = registry.counter
    state.frames.set_fact(TABLE, table_model())
    aliases = {o.name: state.aliases.alias("obj", o.id) for o in registry.objects.values()}
    return ep, aliases


async def test_demo_relation_gives_the_specs_of_the_recorded_gripper(
    tmp_path: Path, events: EventWriter, model: Model
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    seed, registry = await demonstration(blobs)
    async with Hub([LocalProvider(Demo())], blobs=blobs, phases=TEST_PHASES, events=events) as hub:
        ep, obj = await opened(hub, seed, registry)
        grasp = await hub.call(ep, model("demo_relation", keypoint="left K2", obj=obj["tool"]))
        assert not grasp.is_error, text_of(grasp)
        found = DemoRelation.model_validate(grasp.structured)
        assert (found.relation, found.obj, found.held, found.phase) == (
            "grasp",
            obj["tool"],
            None,
            "grasp",
        )
        posed, fit = found.fits
        assert posed.form == "pose" and posed.spec.anchor == "pose"
        assert (posed.position_error_m, posed.angle_error_deg) == pytest.approx((0, 0), abs=1e-3)
        assert fit.form == "grasp" and fit.variant == "top/0"
        assert fit.spec.reference_q is not None
        spec = fit.spec.model_copy(update={"reference_q": None})
        assert spec == GRASP.model_copy(update={"arm": "left"})  # rounded: exactly
        assert text_of(grasp) == (
            f"left K2 (grasp): the gripper relative to {obj['tool']}, as a grasp spec (what each "
            "does not reproduce in brackets; pose first):\n"
            '- pose: {"anchor":"pose","offset_m":[0.0,0.0,0.02],"reference_q":[0.123,0.123,'
            '-0.696,-0.696],"arm":"left","approach_axis":[0.0,0.0,-1.0],"closing_axis":[0.0,1.0,'
            "0.0]} (0.0 mm, 0.0 deg, candidate pose/0)\n"
            '- grasp: {"approach":"top","anchor":"top_center","reference_q":[0.123,0.123,-0.696,'
            '-0.696],"arm":"left"} (0.0 mm, 0.0 deg, candidate top/0)\n'
            "- distances (m, number parameters for the action): approach_m 0.000, "
            "grasp_offset_m 0.000"
        )
        assert found.distances == {"approach_m": 0.0, "grasp_offset_m": 0.0}  # K1 stopped there
        place = await hub.call(ep, model("demo_relation", keypoint="left K4", obj=obj["goal"]))
        found = DemoRelation.model_validate(place.structured)
        assert (found.relation, found.obj, found.held) == ("place", obj["goal"], obj["tool"])
        assert found.distances == {
            "approach_m": 0.0,
            "stop_back_m": 0.0,
        }  # released where K3 stopped
        forms = {fit.form: fit for fit in found.fits}
        assert list(forms) == ["target", "world_axes", "plain", "robot_frame"]
        assert [fit.consistent for fit in found.fits] == [True, True, True, False]  # on the goal
        plain = forms["plain"]  # over the goal: its inside bottom, the 5 cm this scene put it off
        assert (plain.spec.target.anchor, plain.spec.offset_m) == ("inside_bottom", (0, 0, 0))
        assert plain.position_error_m == pytest.approx(0.05, abs=0.03)
        target = forms["target"].spec
        assert target.target.model_dump(exclude_none=True) == {
            "kind": "object",
            "obj": "goal",  # the target's role, as a place spec names it
            "anchor": "top_center",
        }
        assert target.offset_m[1:] == pytest.approx((0.0, 0.0), abs=1e-3)  # along the goal's x
        assert abs(target.offset_m[0]) == pytest.approx(0.05, abs=1e-3)
        assert forms["world_axes"].spec.offset_m[:2] == pytest.approx((0.05, 0.0), abs=1e-3)
        head = text_of(place).splitlines()[0]
        assert head.startswith(
            f"left K4 (release): {obj['tool']}, which the arm holds, placed at {obj['goal']}, as "
            "place specs; the placed object rests on the target"
        )
        assert head.endswith(
            "the forms it bears out are marked consistent (what each does not reproduce in "
            "brackets; consistent first):"
        )
        # K3 holds the tool on its way to the open at K4 (and moved it nowhere since its close,
        # so no held path either): no placement there, refused naming the open
        carried = await hub.call(ep, model("demo_relation", keypoint="left K3", obj=obj["goal"]))
        assert error_text(carried) == (
            "left K3 is on the way to the release: a placement is fitted where the gripper "
            "opens, left K4 (its approach_m is the straight move onto it); ask demo_relation there"
        )
        by_role = await hub.call(ep, model("demo_relation", keypoint="left K2", obj="tool"))
        assert DemoRelation.model_validate(by_role.structured).obj == obj["tool"]
        fits = hub.store.get(ep).frames.fact(DEMO_FITS)
    # What the memory toolset checks a fitted spec's definition against, by keypoint.
    assert fits is not None and sorted(fits) == ["left K2", "left K4"]
    fitted = {**fits["left K2"][1], "reference_q": None}  # its fit-time orientation: a preference
    assert fitted == GRASP.model_copy(update={"arm": "left"}).model_dump(mode="json")
    assert fits["left K2"][0]["anchor"] == "pose"
    assert len(fits["left K4"]) == 4 and fits["left K4"][0] == target.model_dump(mode="json")


async def test_only_s1_roles_that_study_the_demonstration_see_it(
    tmp_path: Path, events: EventWriter
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    seen = {}
    async with Hub([LocalProvider(Demo())], blobs=blobs, phases=TEST_PHASES, events=events) as hub:
        for stage, role in (
            ("s1", "demo_reader"),
            ("s1", "actor"),
            ("s1", "reviser"),
            ("s2", "actor"),
            ("p3", "actor"),
        ):
            ep = (await hub.open_episode(OpenEpisode(stage=stage, role=role))).episode_id
            tools = await hub.tools(ep)
            seen[stage, role] = [tool.name for tool in tools if tool.meta.toolset == "demo"]
    demo = ["demo_relation"]
    assert seen == {
        ("s1", "demo_reader"): demo,
        ("s1", "actor"): demo,
        ("s1", "reviser"): [],
        ("s2", "actor"): [],
        ("p3", "actor"): [],
    }


async def test_fitting_on_the_truth_needs_a_privileged_episode(
    tmp_path: Path, events: EventWriter, model: Model
) -> None:
    """The ground-truth fit (a flag) reads the truth toolset, tier P: a tier-D
    episode is refused, and without the toolset mounted it says so."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    seed, registry = await demonstration(blobs)
    control = MockEnv(
        MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs
    )
    for providers, says in (
        ([LocalProvider(Demo(), {"fit": "truth"}), LocalProvider(Privileged(control))], "tier-P"),
        ([LocalProvider(Demo(), {"fit": "truth"})], "the truth toolset mounted"),
    ):
        async with Hub(providers, blobs=blobs, phases=TEST_PHASES, events=events) as hub:
            ep, obj = await opened(hub, seed, registry)
            fit = await hub.call(ep, model("demo_relation", keypoint="left K2", obj=obj["tool"]))
        assert fit.is_error and says in text_of(fit)
