"""Written skills (`skill_builder/builder`) without a simulator: the loader reads a skill's
directory, the gate's static checks find breaches (the sandbox escapes among them, `fit`
included) and pass a clean skill, the validator imports a skill only in a child process and only
once those pass, an `act` that exits, hangs or floods is an entry that fails, and a fit holding
an absolute position is refused."""

import contextlib
import json
import os
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.skills import SkillSpec
from roboscript.demos.keypoints import KeypointConfig
from roboscript.manipulation import Built
from roboscript.manipulation.specs import GraspSpec
from roboscript_methods.skill_builder.builder.plugins import (
    WrittenSkill,
    load_skills,
    validate,
)
from roboscript_methods.skill_builder.builder.static_check import Vocabulary, check
from roboscript_methods.skill_builder.core_entry.check import Criteria
from roboscript_methods.skill_builder.demo_coverage import run
from roboscript_methods.skill_builder.demo_coverage.fitting import (
    Call,
    fit_error,
    fit_written,
)
from roboscript_methods.skill_builder.demo_coverage.run import explain
from roboscript_methods.skill_builder.demo_coverage.segments import CutConfig, Segment

pytestmark = pytest.mark.unit


VOCAB = Vocabulary.of(["stack_two", "push_T"], ["block_2", "cup0", "t"])

SKILL = '''
"""Grasps with a nudge."""

from roboscript.contracts.skills import SkillSpec

NAME = "{name}"
KINDS = ("grasp",)
SPEC = SkillSpec(name=NAME, description="a grasp", parameters={{"type": "object"}},
                 chain=["parameters", "planning"])


async def act(api, arm, obj, target, params):
    return []
'''
FIT = """
def fit(segment, recording, scene, base, poses):
    return {"offset": 0.01}
"""


def write_skill(root: Path, name: str, skill: str = SKILL, fit: str = FIT,
                cases: str = "gaps: ['task_a/left grasp cup [0, 9]']\n") -> Path:  # fmt: skip
    d = root / name
    d.mkdir(parents=True)
    (d / "skill.py").write_text(textwrap.dedent(skill.format(name=name)), encoding="utf-8")
    (d / "fit.py").write_text(textwrap.dedent(fit), encoding="utf-8")
    (d / "cases.yaml").write_text(cases, encoding="utf-8")
    return d


def test_a_written_skill_loads_from_its_directory_or_one_holding_skills(tmp_path: Path) -> None:
    write_skill(tmp_path / "accepted", "nudge_grasp")
    write_skill(tmp_path / "accepted", "side_grasp")
    one, two = load_skills([tmp_path / "accepted"], VOCAB)
    assert (one.name, two.name) == ("nudge_grasp", "side_grasp")
    assert one.kinds == ("grasp",) and one.cases == ("task_a/left grasp cup [0, 9]",)
    assert one.fit(None, None, None, None, None) == {"offset": 0.01}
    side = load_skills([tmp_path / "accepted" / "side_grasp"], VOCAB)
    assert [k.name for k in side] == ["side_grasp"]


CLEAN = '''
"""A press from above: lowers onto the block until it touches (labels in prose are fine)."""

import math

import numpy as np
from roboscript.contracts.skills import SkillSpec
from roboscript.manipulation import Built

NAME = "press_down"
KINDS = ("touch",)
SPEC = SkillSpec(name=NAME, description="d", parameters={}, chain=["parameters"])
CLEARANCE = 0.02


async def act(api, arm, obj, target, params):
    depth = params["depth"] * 0.5 + CLEARANCE
    turn = math.radians(90.0) * 2
    return [await api.grasp_object(arm, obj, params["spec"], depth, turn, type(obj).__name__)]
'''


def test_a_clean_skill_passes_the_static_checks(tmp_path: Path) -> None:
    d = write_skill(tmp_path, "press_down", skill=CLEAN.replace("{", "{{").replace("}", "}}"))
    (d / "fit.py").write_text(CLEAN_FIT, encoding="utf-8")  # tier P: positions allowed
    assert check(d, VOCAB) == []


CLEAN_FIT = """
import json

import numpy as np
from roboscript_methods.skill_builder.demo_coverage.fitting import supporting
from roboscript.manipulation.fit import fit_grasp_pose


def fit(segment, recording, scene, base, poses):
    depth = float(np.linalg.norm(recording.demo.arms[segment.arm].tcp[segment.end, :3]))
    return json.loads(json.dumps({"depth": depth * 0.3, "spec": {"anchor": "pose"}}))
"""


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("from roboscript.demos import gt_scene", "import"),
        ("import os", "import"),
        ("from . import helper", "import"),
        ("from some_adapter import sim", "import"),
        ("X = open('truth.json')", "call"),
        ("X = getattr(object, 'x')", "call"),
        ("X = object.__subclasses__()", "dunder"),
        ("X = api.run", "runs"),
        ("HEIGHT = 0.75", "position"),
        ("OFFSET = (0.0, -0.12, 0.0)", "position"),
        ("# made for stack two", "task"),
        ("MODE = 'Push-T'", "task"),
        ("WHICH = 'block_2'", "label"),
        ("WHICH = 'the cup on the left'", "label"),
        # sandbox escapes
        ("import traceback\nX = traceback.walk_stack(None)", "import"),
        ("X = lambda frame: frame.f_back.f_locals", "attribute"),
        ("X = lambda error: error.__traceback__.tb_frame", "dunder"),
        ("X = lambda coro: coro.cr_frame", "attribute"),
        ("import asyncio\nX = asyncio.create_subprocess_exec", "import"),
        ("X = np.asyncio", "attribute"),
        ("import posix\nX = posix.system('id')", "import"),
        ("import operator\nX = operator.attrgetter('executor')", "import"),
        ("X = '{0.__class__}'.format(1)", "dunder"),
        ("X = '{0.x}'.format(1)", "attribute"),
        ("X = format(1, 'x')", "call"),
        ("X = np.load('truth.npz')", "attribute"),
        ("from numpy import load", "import"),
        ("import numpy.lib.npyio", "import"),
        ("X = np.lib", "attribute"),
        ("X = np.zeros(3).tofile", "attribute"),
        ("import typing\nX = typing.sys", "attribute"),
        ("import typing\nX = typing.get_type_hints", "attribute"),
        ("from roboscript.manipulation.workspace import load_workspace", "import"),
        ("from roboscript.manipulation.executor import Executor", "runs"),
        ("X = lambda api: api._released", "attribute"),
        ("X = __builtins__", "dunder"),
        ("X = input()", "call"),
        ("X = vars()", "call"),
        ("def reset(api):\n    api.deadline = None", "mutates"),
        ("np.pi = 3", "mutates"),
        ("class Bomb:\n    def __reduce__(self):\n        return ()", "dunder"),
        # match patterns read attributes no `.x` shows
        (
            "def peek(api):\n    match api:\n        case object(executor=e):\n            pass",
            "attribute",
        ),
        (
            "def peek(x):\n    match x:\n        case object(__class__=c):\n            return c",
            "attribute",
        ),
        ("from dataclasses import asdict", "import"),
        ("import dataclasses\nX = lambda api: dataclasses.fields(api)", "attribute"),
        ("X = print", "call"),
        ("X = SystemExit(0)", "call"),
        ("X = exit", "call"),
        ("class Out(BaseException):\n    pass", "call"),
        # positions and labels spelled as arithmetic
        ("HEIGHT = 37 * 0.01", "position"),
        ("AT = np.array([42, -18]) / 100", "position"),
        ("AT = -(3 + 4) / 10", "position"),
        ("WHICH = 'blo' + 'ck'", "label"),
        ("WHICH = ''.join(['cu', 'p0'])", "label"),
    ],
)
def test_the_static_checks_find_each_breach(tmp_path: Path, line: str, rule: str) -> None:
    skill = (CLEAN + "\n" + line + "\n").replace("{", "{{").replace("}", "}}")
    d = write_skill(tmp_path, "press_down", skill=skill)
    assert rule in {f.rule for f in check(d, VOCAB)}


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("import os", "import"),
        ("import pathlib", "import"),
        ("import subprocess", "import"),
        ("import socket", "import"),
        ("import importlib", "import"),
        ("import pickle", "import"),
        ("from roboscript.demos import gt_scene", "import"),
        ("X = open('/data/x')", "call"),
        ("X = json.load", "attribute"),
        ("X = np.fromfile", "attribute"),
        ("X = lambda frame: frame.f_globals", "attribute"),
        ("X = '__class__'", "dunder"),
        ("np.linalg.norm = abs", "mutates"),
        ("def spoil(scene):\n    scene.boxes['x'] = None", "mutates"),
    ],
)
def test_fit_is_checked_for_escapes_too(tmp_path: Path, line: str, rule: str) -> None:
    d = write_skill(tmp_path, "nudge_grasp", fit=CLEAN_FIT + "\n" + line + "\n")
    found = check(d, VOCAB)
    assert rule in {f.rule for f in found} and {f.file for f in found} == {"fit.py"}


# ---- the coverage run -----------------------------------------------------------------------


def at(z: float) -> Pose:
    return Pose(p=(0.3, 0.0, z), q=(1.0, 0.0, 0.0, 0.0))


def segment(kind: str = "grasp") -> Segment:
    return Segment(arm="left", kind=kind, start=0, end=9, tcp=at(0.8),  # type: ignore[arg-type]
                   obj="cube", rest=9)  # fmt: skip


def skill(
    name: str = "nudge_grasp",
    kinds: tuple[str, ...] = ("grasp",),
    act: Any = None,
    fit: Any = None,
) -> WrittenSkill:
    async def built(api: Any, arm: Any, obj: Any, target: Any, params: Any) -> list[Built]:
        api.seen.append((obj, target, params))
        return [Built(arm=arm, call="approach"), Built(arm=arm, call="grasp")]

    spec = SkillSpec(name=name, description="d", parameters={}, chain=["parameters"])
    fitted = fit or (lambda segment, recording, scene, base, poses: {"offset": (0.0, 0.01)})
    return WrittenSkill(name, kinds, spec, act or built, fitted, (), Path("."), VOCAB)


def child_env() -> dict[str, str]:
    return dict(os.environ)


async def test_the_validator_imports_a_skill_in_a_child_and_only_once_it_passes(
    tmp_path: Path,
) -> None:
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"tasks": ["stack_two"], "labels": ["block_2"]}), "utf-8")
    marker = tmp_path / "imported"
    breach = SKILL + f"\nimport os\nos.mkdir({str(marker)!r})\n"
    bad = write_skill(tmp_path, "nudge_grasp", skill=breach)
    good = write_skill(tmp_path, "side_grasp")
    broken = write_skill(tmp_path, "slow_grasp", skill=SKILL.replace("async def act", "def act"))
    rows = await validate([bad, good, broken], vocab, [sys.executable], env=child_env())
    assert [r["ok"] for r in rows] == [False, True, False]
    assert any("imports os" in f for f in rows[0]["static"]) and rows[0]["contract"] == []
    assert not marker.exists()  # never imported
    assert rows[1]["kinds"] == ["grasp"] and rows[1]["cases"] == [
        "task_a/left grasp cup [0, 9]"]  # fmt: skip
    assert rows[2]["static"] == [] and rows[2]["contract"] == ["act is not an async function"]
    failed = await validate([good], vocab, [sys.executable, "-c", "raise SystemExit(3)"])
    assert not failed[0]["ok"] and "exited 3" in failed[0]["contract"][0]
    # a row printed by anything but the validator (it lacks the nonce) passes nothing
    row = json.dumps({"dir": str(good), "name": "side_grasp", "ok": True, "static": [],
                      "contract": []})  # fmt: skip
    forged = [sys.executable, "-c", f"print({row!r}); print('x ' + {row!r})"]
    assert not (await validate([good], vocab, forged))[0]["ok"]


class Scene:
    def __init__(self, z: float) -> None:
        self.poses = {"cube": at(z)}

    def proxy(self, name: str) -> str:
        return f"proxy {name}"


class Api:
    def __init__(self) -> None:
        self.seen: list[Any] = []
        self.executor = SimpleNamespace(env=SimpleNamespace())

    async def grasp_object(self, arm: str, obj: Any, spec: Any) -> Built:
        self.seen.append(("core grasp", obj))
        return Built(arm=arm, call="grasp")  # type: ignore[arg-type]

    async def move_by(self, arm: str, z: float) -> Built:
        return Built(arm=arm, call="lift")  # type: ignore[arg-type]

    async def return_home(self, arm: str) -> Built:
        return Built(arm=arm, call="home")  # type: ignore[arg-type]


async def no_conditions(_: str) -> Any:  # an env whose backend states no success conditions
    return SimpleNamespace(conditions=[])


class FakeCheck:
    """`core_entry.check.Check` as `explain` and `ceiling` use it: an object held by a run named
    `grasp` rises 10 cm with a run named `lift`."""

    def __init__(self, started: int = 0) -> None:
        self.api = Api()
        self.ran: list[str] = []
        self.z = 0.8
        self.started = started  # actions the episode counted before the first call

    @contextlib.asynccontextmanager
    async def episode(self) -> AsyncIterator[Any]:
        async def execute(_: str, batch: Any) -> Any:  # a hold: the robot stands still
            self.ran.append(f"wait {len(batch.steps)}")
            return SimpleNamespace(stopped="completed")

        yield SimpleNamespace(
            opened=SimpleNamespace(episode_id="e"),
            api=self.api,
            ended=None,
            first=SimpleNamespace(end_rules=[]),
            agent=SimpleNamespace(execute=execute),
            start=SimpleNamespace(step=self.started),
        )

    async def robot(self, ep: Any) -> Any:
        return SimpleNamespace(qpos_cmd=[0.0] * 14)

    @property
    def control(self) -> Any:
        async def judge(_: str) -> Any:
            return SimpleNamespace(ended=None, success=False, success_step=None, steps_used=0)

        async def gt_snapshot(_: str) -> Any:  # an env that tracks no stages
            return SimpleNamespace(progress=None)

        return SimpleNamespace(judge=judge, gt_snapshot=gt_snapshot, diagnose=no_conditions)

    async def truth(self, ep: Any) -> Scene:
        return Scene(self.z)

    async def run(self, ep: Any, *built: Built) -> ActionReceipt:
        self.ran.append("+".join(b.call for b in built))
        if built[0].call == "lift" and "grasp" in self.ran:
            self.z += 0.1
        return ActionReceipt(reason=ReceiptReason.ok)


RECORDING = SimpleNamespace(commands=np.zeros((1, 14)))


async def test_an_act_that_exits_hangs_or_floods_is_an_entry_that_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    async def exits(*_: Any) -> list[Built]:
        raise SystemExit(0)

    async def hangs(*_: Any) -> list[Built]:
        await asyncio.sleep(60)
        return []

    async def floods(api: Any, arm: Any, *_: Any) -> list[Built]:
        return [Built(arm=arm, call="move")] * (run.MOTIONS + 1)

    monkeypatch.setattr(run, "ACT_S", 0.1)
    call = Call(segment(), "nudge_grasp", params={})
    errors = []
    for act in (exits, hangs, floods):
        row = await explain(FakeCheck(), RECORDING, call, Criteria(), (0.15, 20.0),  # type: ignore[arg-type]
                            [skill(act=act)])  # fmt: skip
        assert row["explained"] is False and row["cause"] == "entry fails"
        errors.append(row["act_error"])
    assert errors[0] == "SystemExit: 0" and "within 0 s" in errors[1] and "more than" in errors[2]
    marks = [x for x in capsys.readouterr().err.splitlines() if x.startswith("[written act]")]
    assert marks == ["[written act] begin nudge_grasp", "[written act] end"] * 3


CUT = CutConfig(keypoints=KeypointConfig(), home_m=0.15, home_deg=20.0)


# ---- the ceiling's correctness (A) and the chained explanation (B) -----------------------------


def call_of(kind: str, arm: str, start: int, end: int, skill: str | None = "core") -> Call:
    s = Segment(
        arm=arm,
        kind=kind,
        start=start,
        end=end,
        tcp=at(0.8),  # type: ignore[arg-type]
        obj="cube" if kind in ("grasp", "place") else None,
        rest=end,
    )
    entry = {"grasp": "grasp_object", "home": "return_home"}.get(kind, kind)
    return Call(
        s,
        entry if skill == "core" else skill,
        grasp=GraspSpec(arm=arm),  # type: ignore[arg-type]
        why="" if skill else "no entry fits it",
    )


# ---- the chained reuse and a chained case --------------------------------------------------------


class Chain:
    """A check whose calls `perform` judges by name (`fake_perform`): `bad`/`weak` ones as below;
    the benchmark's stages left drop by one per call that is not `weak`, and success latches at
    none left. Each episode is a rollout from the reset (`episodes`: the calls each ran)."""

    def __init__(self, stages: int = 3) -> None:
        self.stages, self.left = stages, stages
        self.episodes: list[list[str]] = []

    @contextlib.asynccontextmanager
    async def episode(self) -> AsyncIterator[Any]:
        self.left = self.stages
        self.episodes.append([])
        yield SimpleNamespace(
            opened=SimpleNamespace(episode_id="e"), ended=None, start=SimpleNamespace(step=0)
        )

    @property
    def control(self) -> Any:
        from roboscript.contracts.env import Progress

        async def judge(_: str) -> Any:
            return SimpleNamespace(ended=None, success=self.left == 0)

        async def gt_snapshot(_: str) -> Any:
            return SimpleNamespace(progress=Progress(stages=self.stages, left=self.left))

        return SimpleNamespace(judge=judge, gt_snapshot=gt_snapshot, diagnose=no_conditions)


def fake_perform(check: Chain, ep: Any, recording: Any, call: Call, *_: Any, **__: Any) -> Any:
    async def done() -> dict[str, Any]:
        check.episodes[-1].append(str(call.skill))
        if str(call.skill).startswith("bad"):
            return {"explained": False, "cause": "entry fails",
                    "motions": [{"ok": False, "reason": "unreachable"}]}  # fmt: skip
        if call.skill != "weak":
            check.left = max(check.left - 1, 0)
        return {"explained": True, "cause": None}

    return done()


def three(middle: str) -> list[Call]:
    return [
        call_of("grasp", "left", 0, 9, skill="grasp_object"),
        call_of("grasp", "left", 10, 20, skill=middle),
        call_of("grasp", "left", 21, 30, skill="grasp_object"),
    ]


# ---- the relative rule -------------------------------------------------------------------------


def demonstration() -> tuple[Segment, Any, Any, Pose]:
    """A left arm that moves 10 frames from above a cube to beside it; a base off the origin."""
    from roboscript.contracts.ids import TaskName
    from roboscript.demos import gt_scene
    from roboscript.demos.demo import ArmTrack, Demo, DemoMeta, Provenance
    from roboscript_methods.skill_builder.demo_coverage.segments import Recording

    n = 10
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    tcp = np.hstack([np.linspace((0.3, 0.0, 1.0), (0.35, 0.05, 0.85), n), q])
    meta = DemoMeta(task=TaskName("t"), seed=0, instructions=["x"], frames=n, frame_steps=10,
                    ee_to_tcp={"left": (0.1, 0.0, 0.0)},
                    provenance=Provenance(source="test", path="x.h5", sha256="0" * 64))  # fmt: skip
    cube = np.hstack([np.tile([0.4, 0.05, 0.8], (n, 1)), q])
    arm = ArmTrack(ee=tcp, tcp=tcp, gripper=np.ones(n), joints=np.linspace(0.0, 0.5, 7 * n)
                   .reshape(n, 7))  # fmt: skip
    rec = Recording(Demo(meta, {"left": arm}),  # type: ignore[dict-item]
                    {"cube": cube}, np.zeros((n - 1, 14)))  # fmt: skip
    scene = gt_scene.Scene(boxes={}, parts={}, poses={"cube": Pose.from_list(cube[0])}, links={})
    seg = Segment(arm="left", kind="move", start=0, end=n - 1,  # type: ignore[arg-type]
                  tcp=Pose.from_list(tcp[-1]))  # fmt: skip
    return seg, rec, scene, Pose(p=(-0.2, 0.1, 0.7), q=(1.0, 0.0, 0.0, 0.0))


def tcp_at(recording: Any, frame: int) -> np.ndarray:
    return np.asarray(recording.demo.arms["left"].tcp[frame][:3])


FITS = {
    # the goal in the world, in the robot's frame (`base` inverted), a recorded object copied
    "world": lambda s, r, sc, b, p: {"goal": list(s.tcp.p), "clearance": 0.04},
    "robot": lambda s, r, sc, b, p: {"goal": (np.asarray(s.tcp.p) - b.p).tolist()},
    "object": lambda s, r, sc, b, p: {"at": list(sc.poses["cube"].p)},
    "scene_bound": lambda s, r, sc, b, p: {"k": 1} if s.tcp.p[0] < 0.36 else None,
    # relative: to the gripper's start, to the object, a height above the table (fixed: z kept)
    "gripper": lambda s, r, sc, b, p: {"goal": (np.asarray(s.tcp.p) - tcp_at(r, s.start)).tolist()},
    "cube": lambda s, r, sc, b, p: {"goal": (np.asarray(s.tcp.p) - sc.poses["cube"].p).tolist(),
                                    "height": s.tcp.p[2], "clearance": 0.04, "turn": 90},
    # a rest pose the robot returns to whatever the scene: in joint space
    "joints": lambda s, r, sc, b, p: {"rest_q": r.demo.arms["left"].joints[0].tolist()},
}  # fmt: skip


@pytest.mark.parametrize("name", ["world", "robot", "object", "scene_bound"])
def test_a_fit_with_absolute_positions_is_refused(name: str) -> None:
    from roboscript_methods.skill_builder.demo_coverage.fitting import RELATIVE_RULE, AbsoluteParams

    seg, rec, scene, base = demonstration()
    with pytest.raises(AbsoluteParams) as found:
        fit_written(skill(name, kinds=("move",), fit=FITS[name]), seg, rec, scene, base, None)  # type: ignore[arg-type]
    assert found.value.args[0].startswith(f"{name}: ")
    said = str(found.value)
    assert "relative to an object, a part or the gripper" in said or name == "scene_bound"
    assert len(fit_error(found.value)) <= 220 and "perceived object" in RELATIVE_RULE
