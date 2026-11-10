"""The perception, geometry and worldstate toolsets over the Hub and MockEnv, with a scripted model
and the vision service's stand-in: identify once (system-only), grasp relations after pick and
place, what an Actor may change (corrections within its role, the initial frame only), and who
sees which tool."""

from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
import pytest
from pydantic import Field

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import ObserveRequest
from roboscript.contracts.hub import CallOutcome
from roboscript.contracts.ids import BatchId, EpisodeId, new_id
from roboscript.contracts.receipt import ActionReceipt, Press, ReceiptReason
from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ImagePart, TextPart, ToolCall
from roboscript.contracts.vision import Detection
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import TABLE_Z
from roboscript.hub import Hub
from roboscript.manipulation.receipts import grip_events
from roboscript.perception.registry import SceneRegistry
from roboscript.records import read_events
from roboscript.spatial.camera import project
from roboscript.spatial.pose import pose_compose
from roboscript.testing.perception import (
    FakeVision,
    ScriptedModel,
    asked_for,
    box_query,
    instruction_of,
    query_roles,
    som_cards,
    som_queries,
)
from roboscript.testing.phases import ROLES as PHASE_ROLES
from roboscript.testing.phases import TEST_PHASES, extended_phases
from roboscript.testing.protocols import RATIONALE
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import (
    IDENTIFIED,
    REGISTRY,
    TABLE,
    LocalProvider,
    Rationale,
    ToolContext,
    Toolset,
    tool,
)
from roboscript.toolsets.geometry import Geometry
from roboscript.toolsets.manipulation import Manipulation
from roboscript.toolsets.perception import Perception, identify_objects
from roboscript.toolsets.scene import SceneConfig, after_action
from roboscript.toolsets.sensing import Sensing
from roboscript.toolsets.worldstate import WorldState

Model = Callable[..., ToolCall]
HEAD = "overhead"
ROLES = {
    "lifted": "the block the robot lifts; a small block",
    "kept": "the block that stays where it is; a small block",
}
INSTRUCTION = "Lift the red block and leave the blue block where it is."
PHRASES = {"lifted": "red block", "kept": "blue block"}
Box4 = tuple[float, float, float, float]
PROFILE = load_embodiment(MOCK_BIMANUAL)
DOWN = (0.0, np.pi / 2, 0.0)  # MockEnv's joints: the ee pose; pitched down, the TCP 0.1 m below


def _box(box: Any) -> dict[str, Any]:
    return {"found": box is not None, "box": None if box is None else list(box), "point": None}


def by_color(
    boxes: dict[str, Any] | None = None,
    lie: dict[str, str] | None = None,
    phrases: dict[str, str] = PHRASES,
    part: int | None = 1,
) -> Any:
    """Set-of-Mark picks the mark whose card color the role's words name (`phrases` for
    abstract roles, else the description; `lie` maps a color to the one it picks instead); the
    words call gives each role's phrase and the color in it; the look-alike call answers
    nothing; box requests get `boxes[words]` or nothing; a part's call, mark `part`."""

    def words(body: dict[str, Any], key: str, text: str) -> str:
        return phrases.get(query_roles(body).get(key, ""), text)

    def spoken(phrase: str) -> dict[str, Any]:
        color = next((c for c in ("red", "green", "blue") if c in phrase), None)
        return {"phrase": phrase, "noun": "block", "color": color, "size": None, "place": None}

    def answer(body: dict[str, Any]) -> Any:
        kind = asked_for(body)
        if kind == "role_words":
            return {key: spoken(phrases.get(role, role)) for key, role in query_roles(body).items()}
        if kind == "compare":
            return dict.fromkeys(query_roles(body))
        if kind == "part_mark":
            return {"mark": part}
        if kind == "axis":
            return {"arrow": "B"}
        if kind == "set_of_mark":
            cards = som_cards(body)
            out: dict[str, Any] = {}
            for key, text in som_queries(body).items():
                said = words(body, key, text)
                color = next((c for c in cards.values() if c in said), None)
                color = (lie or {}).get(color or "", color)
                mark = next((m for m, c in cards.items() if c == color), None)
                out[key] = {"phrase": said, "mark": mark} if query_roles(body) else mark
            return out
        if kind == "boxes":
            return {
                key: _box((boxes or {}).get(words(body, key, text)))
                for key, text in som_queries(body).items()
            }
        return _box((boxes or {}).get(box_query(body)))

    return answer


class Hands(Toolset):
    """Pick and place on MockEnv at world points (a test stand-in for the manipulation toolset),
    each action followed by the grasp-relation hook."""

    name = "hands"

    @tool(kind="execution", tier="D")
    async def pick(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        x: Annotated[float, Field(description="World x of the block's center.")],
        y: Annotated[float, Field(description="World y of the block's center.")],
    ) -> ActionReceipt:
        """Grasp the block at x, y from above and lift it 10 cm."""
        arm: Arm = "left" if x < 0 else "right"
        z = TABLE_Z + 0.025
        path = [(z + 0.1, 1.0, False), (z, 1.0, False), (z, 0.0, True), (z + 0.1, 0.0, False)]
        return await _move(ctx, arm, [(x, y, *point) for point in path])

    @tool(kind="execution", tier="D")
    async def place(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: Arm,
        x: Annotated[float, Field(description="World x to put it at.")],
        y: Annotated[float, Field(description="World y to put it at.")],
    ) -> ActionReceipt:
        """Carry what the arm holds to x, y, set it down and let go."""
        z = TABLE_Z + 0.026
        path = [(z + 0.1, 0.0, False), (z, 0.0, False), (z, 1.0, True), (z + 0.1, 1.0, False)]
        return await _move(ctx, arm, [(x, y, *point) for point in path])

    @tool(kind="execution", tier="D")
    async def push(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        at: Annotated[list[float], Field(description="The TCP where the push was stopped.")],
        closed: Annotated[bool, Field(description="The fingers were closed on nothing.")],
        press_grasp: Annotated[bool, Field(description="The action was a press grasp.")],
    ) -> ActionReceipt:
        """A push the executor saw stopped (no motion here: the receipt alone)."""
        x, y, z = at
        press = Press(arm="left", step=0, tcp=(x, y, z), short_m=0.02, closed=closed)
        receipt = ActionReceipt(reason=ReceiptReason.ok, arm="left", presses=[press])
        await after_action(ctx, receipt, presses={"left"} if press_grasp else ())
        return receipt


async def _move(
    ctx: ToolContext, arm: Arm, points: list[tuple[float, float, float, float, bool]]
) -> ActionReceipt:
    """Steps to TCP points (x, y, z, gripper, until stable), then the hook."""
    spec = PROFILE.arms[arm]
    before = await ctx.env.robot(ctx.env_episode)
    qpos = list(before.qpos_cmd)
    steps = []
    for x, y, z, grip, settle in points:
        q = list(qpos)
        q[spec.action[0] : spec.action[1]] = [x, y, z + 0.1, *DOWN]
        q[spec.gripper_index] = grip
        steps.append(ActionStep(qpos=q, until_gripper_stable=settle))
    batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps, stop_on_success=False)
    result = await ctx.env.execute(ctx.env_episode, batch)
    receipt = ActionReceipt(
        reason=ReceiptReason.ok,
        arm=arm,
        steps_used=result.executed,
        tcp_path={arm: [state.robot.arms[arm].tcp for state in result.states]},
        grips=grip_events(before, result.states, PROFILE),
    )
    await after_action(ctx, receipt)
    return receipt


HANDS = extended_phases(
    TEST_PHASES,
    {role: [f"hands.{name}" for name in ("pick", "place", "push")] for role in PHASE_ROLES},
    after="worldstate",
)


class Rig:
    """The toolsets behind a Hub in front of MockEnv (task lift_red: red, green and blue
    blocks), a scripted model and the vision service's stand-in."""

    def __init__(self, rig: Any, answer: Any, config: SceneConfig | None = None) -> None:
        self.env, self.blobs, self.events, self.base = rig.env, rig.blobs, rig.events, rig
        self.model = ScriptedModel(answer)
        self.vision = FakeVision(rig.blobs)
        self.config = SceneConfig(pipeline=PERCEPTION) if config is None else config
        self.extra: list[LocalProvider] = []  # a method's toolsets, mounted after the core's

    def hub(self, hands: bool = False, manipulation: bool = False, skillset: str = "mock") -> Hub:
        providers = [
            LocalProvider(Sensing()),
            LocalProvider(Perception(vision=self.vision), self.config, [skillset]),
            LocalProvider(Geometry(), self.config),
            LocalProvider(WorldState(), self.config),
            *self.extra,
        ]
        if hands:
            providers.append(LocalProvider(Hands()))
        if manipulation:
            config = {"planner": {"mock": True}, "table_top_z": TABLE_Z}
            providers.append(LocalProvider(Manipulation(), config, ["mock"]))
        return Hub(
            providers,
            blobs=self.blobs,
            phases=HANDS,
            events=self.events,
            env_client=lambda _: self.env,
            models=self.model.client,
        )

    async def episode(self, hub: Hub, stage: str = "s2", role: str = "actor") -> EpisodeId:
        ep, env_ep = await self.base.episode(hub, stage=stage, role=role)
        self.env_episode = env_ep
        self.truth = {
            color: np.asarray(pose.p)
            for color, pose in (await self.env.gt_snapshot(env_ep)).objects.items()
        }
        self.boxes = await head_boxes(self.env, env_ep)
        self.vision.boxes = lambda query: (
            [Detection(box=self.boxes[query], score=0.4, label=query)]
            if query in self.boxes
            else []
        )
        return ep


async def head_boxes(env: Any, episode: EpisodeId) -> dict[str, Box4]:
    """Each block's pixel box in the head camera, from the ground truth (the stand-ins' answers)."""
    camera = (await env.observe(episode, ObserveRequest(cameras=[HEAD]))).cameras[HEAD].camera
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * 0.025
    boxes = {}
    for color, pose in (await env.gt_snapshot(episode)).objects.items():
        points = [pose_compose(pose.as_list(), [*c, 1, 0, 0, 0])[:3] for c in corners]
        uv, _, _ = project(np.array(points), np.array(camera.K), np.array(camera.cam2world))
        u0, v0 = uv.min(axis=0)
        u1, v1 = uv.max(axis=0) + 1
        boxes[f"{color} block"] = (float(u0), float(v0), float(u1), float(v1))
    return boxes


def text_of(outcome: CallOutcome) -> str:
    return "\n".join(part.text for part in outcome.content if isinstance(part, TextPart))


def ok(outcome: CallOutcome) -> dict[str, Any]:
    assert not outcome.is_error, text_of(outcome)
    assert outcome.structured is not None
    return outcome.structured


def free_spot(t: Rig, arm: Arm) -> np.ndarray:
    """A place on `arm`'s side of the table that no block other than the red one is near."""
    free = [np.array(p) for p in ((-0.3, 0.2), (0.3, 0.2), (-0.3, -0.15), (0.3, -0.15))]
    others = [t.truth[c][:2] for c in t.truth if c != "red"]
    return next(
        p
        for p in free
        if (p[0] < 0) == (arm == "left") and all(np.hypot(*(p - o)) > 0.12 for o in others)
    )


async def registry_of(hub: Hub, ep: EpisodeId, blobs: LocalBlobStore) -> SceneRegistry:
    """The registry as the `REGISTRY` fact keeps it (what the log and a seeded episode see)."""
    fact = hub.store.get(ep).frames.fact(REGISTRY)
    assert fact is not None
    data = await blobs.aget_bytes(BlobRef.model_validate(fact["blob"]))
    return SceneRegistry.model_validate_json(data)


async def test_identify_once_on_the_system_channel(rig: Any, model: Model) -> None:
    t = Rig(rig, by_color())
    async with t.hub() as hub:
        ep = await t.episode(hub)
        visible = {tool.name for tool in await hub.tools(ep)}
        outcome = await identify_objects(hub, ep, INSTRUCTION, ROLES)
        found = ok(outcome)
        again = ok(await identify_objects(hub, ep, INSTRUCTION, ROLES))
        by_model = await hub.call(ep, model("identify", instruction=INSTRUCTION))
        observed = await hub.call(ep, model("observe"))
        store = hub.store.get(ep).frames
    assert "identify" not in visible  # system-only: no model sees it
    assert by_model.error_code == "unknown_tool"  # nor calls it
    assert found["frame"] == "frame_1" and found["camera"] == HEAD  # the initial scene
    assert [(r["role"], r["obj"], r["reason"], r["phrase"]) for r in found["roles"]] == [
        ("lifted", "obj_1", "ok", "red block"),
        ("kept", "obj_2", "ok", "blue block"),
    ]
    assert [r["reason"] for r in again["roles"]] == ["registered", "registered"]
    # one Set-of-Mark call for both roles, their words, the look-alike call; then none
    asked = {asked_for(r): r for r in t.model.requests}
    assert sorted(asked) == ["compare", "role_words", "set_of_mark"] and len(t.model.requests) == 3
    request = asked["set_of_mark"]
    assert instruction_of(request) == INSTRUCTION
    assert query_roles(request) == {"r1": "lifted", "r2": "kept"}
    # the kind of object the roles name (a merged proposal to split: none), then the cross-check
    # on the model's phrases
    assert t.vision.calls == [("detect", ["small block"]), ("detect", ["red block", "blue block"])]
    objects = ok(observed)["objects"]
    assert [(o["obj"], o["name"], o["confidence"]) for o in objects] == [
        ("obj_1", "lifted", 1.0),
        ("obj_2", "kept", 1.0),
    ]
    for o, color in zip(objects, ("red", "blue"), strict=True):
        assert np.linalg.norm(np.subtract(o["center"], t.truth[color])) < 0.003
        assert o["size"] == pytest.approx([0.05, 0.05, 0.05], abs=0.003)
    lines = text_of(observed).splitlines()
    assert "Registered objects:" in lines
    assert any(line.startswith("- obj_1 lifted: center (") for line in lines)
    assert text_of(outcome).startswith('Objects in frame_1:\n- lifted ("red block"): obj_1 at (')
    assert [p.label for p in outcome.content if isinstance(p, ImagePart)] == [HEAD]
    events = list(read_events(t.events.path))
    facts = [e.data["name"] for e in events if e.kind == "scene.fact"]
    assert facts[:3] == [TABLE, REGISTRY, IDENTIFIED] and store.fact(TABLE) is not None
    identified = store.fact(IDENTIFIED)
    assert identified is not None and set(identified["roles"]) == {"lifted", "kept"}
    steps = [e.data["step"] for e in events if e.kind == "perception.step"]
    assert steps[:6] == ["table", "proposals", "som", "words", "detect", "compare"]
    assert steps.count("register") == 2
    assert sorted((await store.registry()).objects) == ["o1", "o2"]


async def test_the_grasp_relations_follow_pick_and_place(rig: Any, model: Model) -> None:
    """After a pick the REGISTRY fact holds the block by the arm and every observation shows it
    at the TCP; after a place it is where it was set down, settled on the table."""
    t = Rig(rig, by_color())
    async with t.hub(hands=True) as hub:
        ep = await t.episode(hub)
        await identify_objects(hub, ep, INSTRUCTION, ROLES)
        red = t.truth["red"]
        arm: Arm = "left" if red[0] < 0 else "right"
        picked = await hub.call(ep, model("pick", rationale=RATIONALE, x=red[0], y=red[1]))
        held = (await registry_of(hub, ep, t.blobs)).get("o1")
        looked = await hub.call(ep, model("observe"))
        observed = ok(looked)
        spot = free_spot(t, arm)
        placed = await hub.call(
            ep, model("place", rationale=RATIONALE, arm=arm, x=spot[0], y=spot[1])
        )
        settled = (await registry_of(hub, ep, t.blobs)).get("o1")
        truth = np.asarray((await t.env.gt_snapshot(t.env_episode)).objects["red"].p)
        after = ok(await hub.call(ep, model("observe")))
    assert not picked.is_error and not placed.is_error
    assert list(held.grasps) == [arm]  # attached in the fact, by the arm that closed on it
    [lifted] = [o for o in observed["objects"] if o["obj"] == "obj_1"]
    tcp = np.asarray(observed["arms"][arm]["tcp"]["p"])
    assert lifted["held_by"] == [arm] and np.linalg.norm(np.subtract(lifted["center"], tcp)) < 0.01
    assert f"held by {arm}" in text_of(looked)
    assert not settled.grasps
    [down] = [o for o in after["objects"] if o["obj"] == "obj_1"]
    assert np.linalg.norm(np.subtract(down["center"], truth)) < 0.01  # where it was set down
    assert down["center"][2] == pytest.approx(TABLE_Z + 0.025, abs=0.005)
    steps = [e.data["step"] for e in read_events(t.events.path) if e.kind == "hands.step"]
    assert steps == ["attach", "detach", "settle"]  # the hook logs as the acting toolset


async def test_an_actor_s_corrections_stay_within_its_role(rig: Any, model: Model) -> None:
    """Set-of-Mark takes the blue block for the red one (the lifted role), GroundingDINO shows
    the red one, and the box finds nothing: the blue block goes to the kept role (confirmed),
    and the lifted role, whose mark shows the blue block again, falls back to GroundingDINO's
    red block (unconfirmed). An Actor may not rebind the kept role to a region only the lifted
    role showed, nor the lifted role to the kept role's block."""
    t = Rig(rig, by_color(lie={"red": "blue"}))
    async with t.hub() as hub:
        ep = await t.episode(hub)
        found = ok(await identify_objects(hub, ep, INSTRUCTION, ROLES))
        lifted, kept = found["roles"]
        red, blue = (c["region"] for c in lifted["candidates"])
        rebound = await hub.call(
            ep, model("register", name="kept", region=red, replaces=kept["obj"])
        )
        twice = await hub.call(
            ep, model("register", name="lifted", region=blue, replaces=lifted["obj"])
        )
        observed = ok(await hub.call(ep, model("observe")))
    assert (lifted["reason"], lifted["confidence"], lifted["same_as"]) == ("unconfirmed", 0.5, None)
    assert "registered for another role" in lifted["note"]
    assert _near(lifted["center"], t.truth["red"]) and _near(kept["center"], t.truth["blue"])
    assert kept["reason"] == "ok"
    assert [c["backend"] for c in lifted["candidates"]] == ["grounding_dino", "som"]
    assert rebound.is_error and "a region it showed for that role" in text_of(rebound)
    assert twice.is_error and "shows the object registered for 'kept'" in text_of(twice)
    names = [
        (o["name"], _near(o["center"], t.truth[c]))
        for o, c in zip(observed["objects"], ("blue", "red"), strict=True)
    ]
    assert names == [("kept", True), ("lifted", True)]  # one record per block


async def test_an_actor_looks_at_the_initial_frame_only(rig: Any, model: Model) -> None:
    t = Rig(rig, by_color())
    async with t.hub() as hub:
        ep = await t.episode(hub)
        await identify_objects(hub, ep, INSTRUCTION, ROLES)
        await hub.call(ep, model("observe"))
        initial = ok(await hub.call(ep, model("ground", query="red block")))
        await hub.call(ep, model("observe"))
        later = await hub.call(ep, model("ground", query="red block", frame="frame_2"))
        newest = await hub.call(ep, model("ground", query="red block", frame="frame_3"))
        part = await hub.call(ep, model("locate_part", obj="obj_1", part="corner", frame="frame_3"))
        hidden = [await hub.call(ep, model(name, obj="obj_1")) for name in ("refresh", "locate")]
    assert initial["frame"] == "frame_1"
    assert ok(newest)["frame"] == "frame_3"  # the newest, to find an untaken object again
    for refused in (later, part):
        assert refused.is_error and "only on the initial scene" in text_of(refused)
    assert [h.error_code for h in hidden] == ["unknown_tool", "unknown_tool"]


async def test_who_sees_which_tool(rig: Any) -> None:
    t = Rig(rig, by_color())
    seen, system = {}, {}
    async with t.hub() as hub:
        for role in ("actor", "demo_reader", "reviser"):
            ep = await t.episode(hub, stage="p3", role=role)
            seen[role] = {tool.name for tool in await hub.tools(ep)}
            system[role] = hub.store.get(ep).system_tools
    measuring = {"lift", "fit", "table", "project", "measure_pixel", "relation", "check"}
    actor = {"table", "check", "ground", "locate_part", "register", "register_part"}
    # The Actor's actions measure for themselves (the measuring tools are service calls then).
    remeasuring = {"remeasure", "refine", "look_pose"}
    beside = measuring - actor | remeasuring | {"segment", "identify", "locate", "refresh"}
    assert actor <= seen["actor"] and not beside & seen["actor"]
    assert measuring | actor | remeasuring | {"segment", "locate", "refresh"} <= seen["demo_reader"]
    assert "identify" not in seen["demo_reader"]
    assert measuring | {"ground", "segment"} <= seen["reviser"]  # it reads, on logged frames
    writes = {"identify", "locate", "locate_part", "register", "register_part", "refresh"}
    assert not writes & seen["reviser"]
    assert system == {"actor": ("identify",), "demo_reader": ("identify",), "reviser": ()}


def _near(center: Any, truth: Any, tolerance: float = 0.003) -> bool:
    return bool(np.linalg.norm(np.subtract(center, truth)) < tolerance)


def arrows(*labels: str) -> Any:
    """`by_color`, with the arrow calls answered `labels` in turn."""
    answer, given = by_color(), iter(labels)

    def answering(body: dict[str, Any]) -> Any:
        return {"arrow": next(given)} if asked_for(body) == "axis" else answer(body)

    return answering
