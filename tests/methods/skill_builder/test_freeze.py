"""Freezing a builder run's skills (`skill_builder/freeze.py`): a run without the sandbox (or one
that does not say) is refused, a skill the static checks refuse or that does not load is refused,
a skillset that does not fit the robot is refused, and a sandboxed run's skills are frozen in the
store with their specs (each read in a child process, never in this one)."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from roboscript.embodiments import load_embodiment
from roboscript.skills.store import SkillSetStore
from roboscript_methods.skill_builder.builder.sandbox import record_mode
from roboscript_methods.skill_builder.builder.static_check import Vocabulary
from roboscript_methods.skill_builder.freeze import FreezeRefused, freeze_run, read_specs

pytestmark = pytest.mark.unit

VOCAB = Vocabulary.of(["stack_two"], ["block_2"])
SKILL = """
from roboscript.contracts.skills import SkillNeeds, SkillSpec

NAME = "{name}"
KINDS = ("grasp",)
SPEC = SkillSpec(name=NAME, description="a grasp{extra}", parameters={{"type": "object"}},
                 needs=SkillNeeds(min_arms={arms}), chain=["parameters", "planning"])


async def act(api, arm, obj, target, params):
    return []
"""


def write_skill(root: Path, name: str, extra: str = "", arms: int = 1) -> Path:
    d = root / name
    d.mkdir(parents=True)
    text = SKILL.format(name=name, extra=extra, arms=arms)
    (d / "skill.py").write_text(textwrap.dedent(text), encoding="utf-8")
    (d / "fit.py").write_text("def fit(segment, recording, scene, base, poses):\n"
                              "    return {}\n", encoding="utf-8")  # fmt: skip
    (d / "cases.yaml").write_text("gaps: ['t/left grasp x [0, 9]']\n", encoding="utf-8")
    return d


def make_run(tmp_path: Path, mode: str | None = "bwrap") -> Path:
    run = tmp_path / "run"
    (run / "accepted").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"tasks": ["t"]}), encoding="utf-8")
    if mode is not None:
        record_mode(run, mode)  # type: ignore[arg-type]
    return run


async def freeze(run: Path, store: SkillSetStore, **knobs: object) -> object:
    return await freeze_run(
        run, name="built_test", version="v1", vocabulary=VOCAB,
        profile=load_embodiment("mock_bimanual"), store=store, argv=[sys.executable],
        **knobs,  # type: ignore[arg-type]
    )  # fmt: skip


async def test_a_sandboxed_runs_skills_are_frozen(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    write_skill(run / "accepted", "nudge_grasp")
    write_skill(run / "accepted", "side_grasp")
    store = SkillSetStore(tmp_path / "store")
    ref = await freeze(run, store, envs=["mock"])
    assert await store.frozen("built_test") == ref
    manifest = await store.load(ref)  # type: ignore[arg-type]
    assert [s.name for s in manifest.skills] == ["nudge_grasp", "side_grasp"]
    assert manifest.layer == "built" and manifest.envs == ["mock"]
    assert manifest.report is not None and manifest.report.passed
    assert await freeze(run, store, envs=["mock"]) == ref  # the same content again: its ref


@pytest.mark.parametrize("mode", ["none", None])
async def test_a_run_without_the_sandbox_is_never_frozen(tmp_path: Path, mode: str | None) -> None:
    run = make_run(tmp_path, mode)
    write_skill(run / "accepted", "nudge_grasp")
    store = SkillSetStore(tmp_path / "store")
    with pytest.raises(FreezeRefused, match="without the sandbox is never frozen"):
        await freeze(run, store)
    assert not (tmp_path / "store").exists()


async def test_a_summary_marked_unsandboxed_refuses_the_freeze(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    write_skill(run / "accepted", "nudge_grasp")
    (run / "round-0").mkdir()
    (run / "round-0" / "summary.md").write_text("UNSANDBOXED: ...\n", encoding="utf-8")
    with pytest.raises(FreezeRefused, match="marked UNSANDBOXED"):
        await freeze(run, SkillSetStore(tmp_path / "store"))


async def test_static_findings_and_misfits_are_refused(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    write_skill(run / "accepted", "nudge_grasp", extra=" for stack_two")
    with pytest.raises(FreezeRefused, match="static checks"):
        await freeze(run, SkillSetStore(tmp_path / "store"))
    run = make_run(tmp_path / "b")
    write_skill(run / "accepted", "three_grasp", arms=3)
    with pytest.raises(FreezeRefused, match="needs 3 arms"):
        await freeze(run, SkillSetStore(tmp_path / "store"))
    run = make_run(tmp_path / "c")
    with pytest.raises(FreezeRefused, match="holds no skill"):
        await freeze(run, SkillSetStore(tmp_path / "store"))


async def test_specs_are_read_in_a_child_and_a_broken_skill_is_named(tmp_path: Path) -> None:
    good = write_skill(tmp_path, "nudge_grasp")
    bad = write_skill(tmp_path, "bad_grasp")
    (bad / "skill.py").write_text("import sys\nprint('forged row')\nNAME = 1\n", encoding="utf-8")
    with pytest.raises(FreezeRefused, match="bad_grasp: NAME 1"):
        await read_specs([good.resolve(), bad.resolve()], [sys.executable])
    assert [s.name for s in await read_specs([good.resolve()], [sys.executable])] == ["nudge_grasp"]
    assert not [m for m in sys.modules if m.startswith("_written_skill_")]  # never imported here
