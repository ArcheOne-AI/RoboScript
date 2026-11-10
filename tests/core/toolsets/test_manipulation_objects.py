"""The manipulation toolset's actions on registered objects, over the Hub, MockEnv (task
stack_red_on_blue) and MockPlanner, with a synthetic registry (MockEnv's boxes registered from
surface samples at their true poses): a spec the scene cannot make is a refusal, not a crash, and
a spec named by its Memory parameter runs with every stored field."""

import math
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.geometry import Pose
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import EpisodeId, FrameId, TaskName
from roboscript.contracts.receipt import ActionReceipt
from roboscript.contracts.tools import EnvBinding, TextPart, ToolCall
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import TABLE_Z, MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.manipulation import (
    ObjectProxy,
    Reach,
)
from roboscript.perception.registry import SceneRegistry
from roboscript.records import EventWriter, read_events
from roboscript.spatial.pose import pose_compose, pose_inverse
from roboscript.spatial.rotation import quat_to_matrix
from roboscript.testing.phases import TEST_PHASES
from roboscript.testing.scenes import Box, surface, table_model
from roboscript.tools import (
    MEMORY_SPECS,
    TABLE,
    LocalProvider,
    ToolContext,
)
from roboscript.toolsets.manipulation import Manipulation

Model = Callable[..., ToolCall]
CONFIG = {"planner": {"mock": True}, "table_top_z": TABLE_Z}
TOP = {"approach": "top", "closing": "major"}
WHY = {"stage": "stack", "scene": "boxes", "progress": "testing", "plan": "act"}
ON_BLUE = {"target": {"kind": "object", "obj": "blue", "anchor": "top_center"}}


@dataclass
class Scene:
    """A Hub episode on MockEnv whose boxes are registered under their colors (`obj`: color ->
    alias; `ids`: color -> registry id)."""

    env: MockEnv
    hub: Hub
    model: Model
    ep: EpisodeId
    env_ep: EpisodeId
    obj: dict[str, str]
    ids: dict[str, str]

    async def call(self, tool: str, **arguments: Any) -> CallOutcome:
        if tool != "check_reach":
            arguments = {"rationale": WHY, **arguments}
        return await self.hub.call(self.ep, self.model(tool, **arguments))

    async def act(self, tool: str, **arguments: Any) -> ActionReceipt:
        outcome = await self.call(tool, **arguments)
        assert not outcome.is_error, text_of(outcome)
        return ActionReceipt.model_validate(outcome.structured)

    async def registry(self) -> SceneRegistry:
        return await self.hub.store.get(self.ep).frames.registry()

    async def tcp(self, arm: str) -> np.ndarray:
        return np.array((await self.env.robot(self.env_ep)).arms[arm].tcp.p)

    async def truth(self, color: str) -> np.ndarray:
        return np.array((await self.env.gt_snapshot(self.env_ep)).objects[color].p)

    async def side(self, color: str) -> str:
        return "left" if (await self.truth(color))[0] < 0 else "right"


def text_of(outcome: CallOutcome) -> str:
    return "\n".join(part.text for part in outcome.content if isinstance(part, TextPart))


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[tuple[MockEnv, LocalBlobStore]]:
    blobs = LocalBlobStore(tmp_path / "blobs")
    config = MockConfig(task=TaskName("stack_red_on_blue"), blob_dir=tmp_path / "blobs")
    yield MockEnv(config, blobs=blobs), blobs


async def scene(
    env: tuple[MockEnv, LocalBlobStore],
    events: EventWriter,
    model: Model,
    toolset: Manipulation | None = None,
    stage: str = "s1",
    config: dict[str, Any] = CONFIG,
    skillsets: Sequence[str] = ("mock",),
) -> tuple[Hub, Callable[[], Any]]:
    mock, blobs = env
    provider = LocalProvider(toolset or Manipulation(), config, list(skillsets))
    hub = Hub([provider], blobs=blobs, phases=TEST_PHASES, events=events, env_client=lambda _: mock)

    async def opened() -> Scene:
        spec = EpisodeSpec(task=TaskName("stack_red_on_blue"), seed=3)
        env_ep = await mock.open(spec)
        binding = EnvBinding(
            url="inproc://mock", episode_id=env_ep.episode_id, token=env_ep.agent_token
        )
        episode = OpenEpisode(
            stage=stage, role="actor", env=binding, embodiment=str(MOCK_BIMANUAL), structured=True
        )
        session = await hub.open_episode(episode)
        state = hub.store.get(session.episode_id)
        ids, aliases = {}, {}
        async with state.frames.edit_registry() as registry:
            for color, pose in (await mock.gt_snapshot(env_ep.episode_id)).objects.items():
                x = quat_to_matrix(pose.q)[:, 0]
                box = Box(pose.p, (0.05, 0.05, 0.05), math.degrees(math.atan2(x[1], x[0])))
                points = surface(box)
                record = registry.register(
                    color,
                    points[points[:, 2] > TABLE_Z + 1e-6],
                    frame=FrameId("f0"),
                    step=0,
                    table_z=TABLE_Z,
                )
                ids[color] = record.id
                aliases[color] = state.aliases.alias("obj", record.id)
        state.frames.set_fact(TABLE, table_model())
        return Scene(mock, hub, model, session.episode_id, env_ep.episode_id, aliases, ids)

    return hub, opened


async def _proxy(s: Scene, color: str) -> ObjectProxy:
    geometry = (await s.registry()).geometry(s.ids[color])
    return ObjectProxy.from_geometry(geometry)


def _radius(p: np.ndarray, point: np.ndarray, direction: np.ndarray) -> float:
    r = p - point
    return float(np.linalg.norm(r - (r @ direction) * direction))


def _angle(a: np.ndarray, b: np.ndarray, point: np.ndarray, direction: np.ndarray) -> float:
    u, v = a - point, b - point
    u, v = u - (u @ direction) * direction, v - (v @ direction) * direction
    return math.degrees(math.atan2(float(direction @ np.cross(u, v)), float(u @ v)))


async def test_a_spec_the_scene_cannot_make_is_a_refusal_not_a_crash(
    env: tuple[MockEnv, LocalBlobStore],
    events: EventWriter,
    model: Model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SpecError from synthesis (the Agent's spec, e.g. an aim with nothing to aim at) comes
    back as a `tool_error` the Agent reads; the episode goes on and nothing moved."""
    import roboscript.toolsets.manipulation as toolset
    from roboscript.manipulation import SpecError

    async def unmakeable(*args: Any, **kwargs: Any) -> Any:
        raise SpecError("x_toward other_arm aims at the other gripper, whose TCP is not known")

    hub, opened = await scene(env, events, model)
    async with hub:
        s = await opened()
        arm, red = await s.side("red"), s.obj["red"]
        steps = (await env[0].robot(s.env_ep)).step
        monkeypatch.setattr(toolset, "_built", unmakeable)
        refused = await s.call("grasp_object", arm=arm, obj=red, grasp=TOP)
        monkeypatch.undo()
        assert refused.error_code == "tool_error" and not refused.episode_ended
        assert text_of(refused) == (
            "the spec cannot be made here: x_toward other_arm aims at the other gripper, whose "
            "TCP is not known"
        )
        assert (await env[0].robot(s.env_ep)).step == steps
        assert (await s.act("grasp_object", arm=arm, obj=red, grasp=TOP)).reason == "ok"


class Measure:
    """Stands in for the scene toolsets' measurements: `poses` by object id, sure as given;
    `keep` stores a pose (a held object's grasp following it), `look_pose` is 5 cm above where
    the arm is, `hinge` estimates the given side, sure as given, and `turn` sees nothing."""

    def __init__(
        self, poses: dict[str, Any], confidence: float = 0.9, hinge: tuple[str, float] | None = None
    ) -> None:
        self.poses, self.confidence, self.hinged = poses, confidence, hinge
        self.relocated: set[str] = set()  # reported `relocated` (gone from there, found nearby)
        self.unsure: dict[str, float] = {}  # reported `low_confidence`, moved this far (m)
        self.refined: list[tuple[str, str]] = []
        self.remeasured: list[str] = []
        self.looked: list[tuple[str, str]] = []

    async def refine(self, ctx: ToolContext, target: str, arm: str, part: str | None = None) -> Any:
        self.refined.append((target, arm))
        if target not in self.poses:
            return None
        pose, sure = self.poses[target], self.confidence
        reason = "relocated" if target in self.relocated else "ok"
        if target in self.unsure:
            reason = "low_confidence"
        return SimpleNamespace(
            obj=target, frame="f", cameras=[], pose=pose, confidence=sure, reason=reason,
            shift_m=self.unsure.get(target, 0.0),
        )  # fmt: skip

    async def keep(self, ctx: ToolContext, measured: Any) -> None:
        robot = await ctx.env.robot(ctx.env_episode)
        async with ctx.frames.edit_registry() as registry:
            record = registry.objects[measured.obj]
            tcp = {a: robot.arms[a].tcp.as_list() for a in record.grasps}
            at = measured.pose.as_list()
            grasps = {a: pose_compose(pose_inverse(t), at) for a, t in tcp.items()}
            update = {"pose": measured.pose, "grasps": grasps}
            registry.objects[measured.obj] = record.model_copy(update=update)

    async def remeasure(self, ctx: ToolContext, obj: str) -> Any:
        self.remeasured.append(obj)
        reason = "relocated" if obj in self.relocated else "ok"
        return SimpleNamespace(
            pose=self.poses.get(obj), confidence=1.0, reason=reason, stored=True, shift_m=0.12
        )

    async def look_pose(self, ctx: ToolContext, obj: str, arm: str) -> Pose | None:
        self.looked.append((obj, arm))
        ee = (await ctx.env.robot(ctx.env_episode)).arms[arm].ee
        return Pose(p=(ee.p[0], ee.p[1], ee.p[2] + 0.05), q=ee.q)

    async def hinge(self, ctx: ToolContext, obj: str, part: str) -> Any:
        if self.hinged is None:
            return None
        side, sure = self.hinged
        return SimpleNamespace(part=part, side=side, at="upright", confidence=sure)


async def test_a_spec_named_by_its_memory_parameter_runs_with_every_stored_field(
    env: tuple[MockEnv, LocalBlobStore], events: EventWriter, model: Model
) -> None:
    """A call may name the Memory's spec (`{name}`) instead of writing it out: the tool runs the
    stored spec, every field (here the grasp's major closing, which a transcription could drop),
    and logs the expansion; the same for a placement, in `together` and in `check_reach`; a name
    the Memory has no such spec under is a refusal listing the ones it has, and naming the tool
    that takes it when it is a parameter of another kind."""
    from roboscript.memory import GraspSpec, GraspSpecParam, MemoryDoc, PlaceSpec, PlaceSpecParam
    from roboscript.tools.memory_facts import spec_seed

    hub, opened = await scene(env, events, model, stage="dev")
    doc = MemoryDoc(
        task="stack_red_on_blue",
        params={
            "grasp_box": GraspSpecParam(value=GraspSpec.model_validate(TOP)),
            "on_blue": PlaceSpecParam(value=PlaceSpec.model_validate(ON_BLUE)),
        },
    )
    seeded = spec_seed(doc, None)
    assert seeded is not None and set(seeded.facts[MEMORY_SPECS]["specs"]) == {
        "grasp_box",
        "on_blue",
    }
    async with hub:
        s = await opened()
        s.hub.store.get(s.ep).frames.set_fact(MEMORY_SPECS, seeded.facts[MEMORY_SPECS])
        arm, red = await s.side("red"), s.obj["red"]
        check = await s.call(
            "check_reach", **{arm: {"action": "grasp_object", "obj": red, "grasp": "grasp_box"}}
        )
        assert Reach.model_validate(check.structured).reachable, text_of(check)
        grasped = await s.act("grasp_object", arm=arm, obj=red, grasp="{grasp_box}")
        assert grasped.reason == "ok" and grasped.grasp is not None
        assert grasped.grasp.n_candidates == 4  # as the written-out spec gives
        await s.act("move_by", arm=arm, z=0.1)
        other = "right" if arm == "left" else "left"
        placed = await s.act(
            "together",
            moves=[
                {
                    "arm": arm,
                    "action": {
                        "action": "place_object",
                        "obj": red,
                        "place": "{on_blue}",
                        "stop_back_m": 0.005,
                    },
                },
                {"arm": other, "action": {"action": "open_gripper"}},
            ],
        )
        assert placed.reason in ("ok", "stopped_short"), placed.note
        wrong = await s.call("grasp_object", arm=other, obj=s.obj["blue"], grasp="on_blue")
        assert wrong.is_error and "no grasp_spec parameter 'on_blue'" in text_of(wrong)
        assert "grasp_spec parameters: grasp_box" in text_of(wrong)
        assert "'on_blue' is a place_spec parameter, which place_object takes" in text_of(wrong)
        missing = await s.call("grasp_object", arm=other, obj=s.obj["blue"], grasp="nowhere")
        assert missing.is_error and "name one of them or write the spec out" in text_of(missing)
    expanded = [e.data for e in read_events(events.path) if e.kind == "manipulation.spec_expanded"]
    assert [(d["name"], d["kind"]) for d in expanded] == [
        ("grasp_box", "grasp_spec"),
        ("grasp_box", "grasp_spec"),
        ("on_blue", "place_spec"),
    ]
    assert expanded[0]["spec"]["closing"] == "major"


async def test_plan_arms_gives_holds_open_at_once_different_arms_and_a_receiver_the_other_one(
    env: tuple[MockEnv, LocalBlobStore], events: EventWriter, model: Model
) -> None:
    """`plan_arms` over the Memory's holds: two objects held at the same time take different
    arms, a hand-over's receiving hold the arm that does not give, and the plan is the episode's
    `ARM_PLAN` fact; an override is the Agent's, with its reason; `together` refuses two moves
    for one arm."""
    from roboscript.memory import (
        GraspSpec,
        GraspSpecParam,
        MemoryDoc,
        PlaceSpec,
        PlaceSpecParam,
        WorkflowStage,
    )
    from roboscript.tools import ARM_PLAN, IDENTIFIED
    from roboscript.tools.memory_facts import spec_seed

    def stage(name: str, call: str) -> WorkflowStage:
        return WorkflowStage(name=name, call=call)

    doc = MemoryDoc(
        task="stack_red_on_blue",
        workflow=[
            stage("take_a", "grasp_object(arm=<arm>, obj=a, grasp={top})"),
            stage("take_b", "grasp_object(arm=<arm>, obj=b, grasp={top})"),
            stage("put_a", "place_object(arm=<arm of a>, obj=a, place={top_of_b})"),
            stage("pass_b", "grasp_object(arm=<arm>, obj=b, grasp={top})"),
            stage("let_go", "open_gripper(arm=<arm>)"),
        ],
        params={
            "top": GraspSpecParam(value=GraspSpec.model_validate(TOP)),
            "top_of_b": PlaceSpecParam(value=PlaceSpec.model_validate(ON_BLUE)),
        },
    )
    seeded = spec_seed(doc, None)
    assert seeded is not None
    hub, opened = await scene(env, events, model, stage="dev")
    async with hub:
        s = await opened()
        frames = s.hub.store.get(s.ep).frames
        frames.set_fact(MEMORY_SPECS, seeded.facts[MEMORY_SPECS])
        roles = {"a": {"object": s.ids["red"]}, "b": {"object": s.ids["blue"]}}
        frames.set_fact(IDENTIFIED, {"roles": roles})
        planned = await s.call("plan_arms")
        assert not planned.is_error, text_of(planned)
        holds = planned.structured["holds"]
        assert [h["role"] for h in holds] == ["a", "b", "b"]
        assert holds[0]["arm"] != holds[1]["arm"]  # open at the same time
        assert holds[2]["arm"] != holds[1]["arm"] and holds[2]["why"] == "receiver"
        assert holds[0]["stages"] == ["take_a", "put_a"]
        fact = frames.fact(ARM_PLAN)
        assert fact is not None and fact["roles"]["b"] == [holds[1]["arm"], holds[2]["arm"]]
        assert fact["stages"]["take_b"] == [holds[1]["arm"]]
        chosen = await s.call(
            "plan_arms",
            assign=[{"stage": "take_a", "arm": holds[1]["arm"]}],
            rationale={**WHY, "plan": "the a object is easier from the other side"},
        )
        assert not chosen.is_error, text_of(chosen)
        assert chosen.structured["holds"][0]["why"] == "agent"
        assert "easier from the other side" in (chosen.structured["override"] or "")
        same = await s.call(
            "together",
            moves=[
                {"arm": "left", "action": {"action": "open_gripper"}},
                {"arm": "left", "action": {"action": "open_gripper"}},
            ],
        )
        assert same.is_error and "both moves name the left arm" in text_of(same)
