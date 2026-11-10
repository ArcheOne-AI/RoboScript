"""P3 on RoboTwin without the simulator: the policy package (its imports, the task object behind
the shim, `eval()` turns on a fake task object), shards that evaluate the seeds of the official
loop, resumed blocks, episodes that infrastructure lost, the relay (against an Agent host, and
its deadlines), the install, and the launcher around stand-ins for RoboTwin's official scripts."""

import contextlib
import functools
import hashlib
import io
import itertools
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import official_fakes
import pytest
import yaml
from official_fakes import server_overrides
from pydantic import SecretStr, ValidationError
from robotwin_fakes import FakeTaskEnv
from task_whitelist import task_uses, to_shim

from roboscript import rpc
from roboscript.contracts.action import ExecuteResult
from roboscript.contracts.errors import InfraError
from roboscript.contracts.host import AgentHost, OfficialOutcome, Poll
from roboscript.contracts.ids import BatchId, ManifestId, TaskName
from roboscript.contracts.observation import Observation
from roboscript.kernel import compose, entrypoints
from roboscript.learning.agent_host import AgentHostConfig, serve_host
from roboscript.memory import (
    FileMemoryStore,
    MemoryManifest,
    VersionSource,
    build_manifest,
    write_manifest,
)
from roboscript.rpc import read_ready
from roboscript.settings import write_secret
from roboscript.stack import Interpreters
from roboscript.testing.memory import sample_memory
from roboscript_robotwin.benchmark import HERE, ROBOTWIN
from roboscript_robotwin.config import with_perception_place
from roboscript_robotwin.eval import install, launch, wire
from roboscript_robotwin.eval.launch import P3Config, P3Run, evaluate
from roboscript_robotwin.eval.policy import deploy_policy
from roboscript_robotwin.eval.relay import Relay, RelayConfig
from roboscript_robotwin.sim import shim

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "examples" / "simBenchmarks" / "RoboTwin"
SRC = REPO / "packages" / "benchmarks" / "robotwin" / "src"
TASK = TaskName("beat_block_hammer")
CLEAN = "rs_agentic_clean_p3"
KEY = f"{TASK}.{CLEAN}.seed0"
TOKEN = SecretStr("a-host-token-for-the-tests")
RATIONALE = {"stage": "wait", "scene": "a hammer", "progress": "at rest", "plan": "stay"}
HOLD_THEN_FINISH = [
    [{"name": "hold", "arguments": {"rationale": RATIONALE, "steps": 2}}],
    [{"name": "finish", "arguments": {"note": "done"}}],
]
HOLD = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
agent_venv = pytest.mark.skipif(
    "scripted" not in entrypoints.names("roboscript.brains"),
    reason="the Agent host runs in the agent venv (roboscript installed, with entry points)",
)


async def frozen(directory: Path) -> tuple[Path, MemoryManifest]:
    """A store with beat_block_hammer's frozen Memory, and the manifest's file."""
    store = FileMemoryStore(directory / "memory")
    await store.commit(sample_memory(TASK), status="s1_draft", source=VersionSource(kind="s1"))
    await store.freeze(TASK, 0, reason="for the test")
    manifest = await build_manifest(store, [TASK])
    write_manifest(directory / "manifest.json", manifest)
    return directory / "manifest.json", manifest


def host_config_of(directory: Path, bridge: dict[str, float] | None = None) -> dict[str, Any]:
    """The Agent host's config, without what each run sets."""
    return {
        "memory": {"name": "file", "options": {"root": str(directory / "memory")}},
        "recipe": str(REPO / "recipes" / "wicl"),
        "episodes": {
            "embodiment": str(HERE / "embodiments" / "aloha_agilex.yaml"),
            "bridge": bridge or {"observe_timeout_s": 10, "report_timeout_s": 30},
            "latch_finish": False,  # the script finishes once
        },
        "method": "roboscript.testing.phases:TestMethod",  # its Actor holds in p3
        "memory_access": "bound",  # the Memory of the task id
        "toolsets": [{"name": "sensing"}, {"name": "hold"}],
        "brain": {"name": "scripted", "config": {"replies": HOLD_THEN_FINISH}},
    }


def answer(status: str, hold: int = 4, **fields: Any) -> dict[str, Any]:
    return {"status": status, "batch": None, "reason": None, "hold": hold, "infra": False, **fields}


class Model:
    """RoboTwin's ModelClient in front of a scripted relay: records what `eval()` sends."""

    def __init__(self, answers: dict[str, list[dict[str, Any]]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, Any]] = []

    def call(self, func_name: str, obs: Any = None) -> Any:
        self.calls.append((func_name, obs))
        if func_name == "end" or (func_name == "report" and not self.answers.get("report")):
            return answer("ok")
        return self.answers[func_name].pop(0)


def test_the_policy_package_imports_nothing_heavy(tmp_path: Path) -> None:
    """Loaded as RoboTwin loads it (`policy/RoboScript` on `./policy`), it imports only the
    standard library until a function runs."""
    (tmp_path / "policy").mkdir()
    (tmp_path / "policy" / install.POLICY).symlink_to(HERE / "eval" / "policy")
    code = (
        "import importlib, sys\n"
        "sys.path.append(sys.argv[1])\n"
        "policy = importlib.import_module('RoboScript')\n"
        "assert all(callable(getattr(policy, n)) for n in ('get_model', 'eval', 'reset_model'))\n"
        "heavy = ('roboscript', 'roboscript_robotwin', 'numpy', 'pydantic', 'anyio', 'httpx',\n"
        "         'yaml')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    done = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "policy")],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert done.stdout.strip() == "[]"


def test_eval_touches_the_task_object_only_through_the_shim() -> None:
    used, stray = task_uses(Path(deploy_policy.__file__), "TASK_ENV", to_shim)
    assert (used, stray) == (set(), [])


class _Recorder:
    """What `shard_policy` asks the model (only `reset_model` reaches it in the fake)."""

    def call(self, func_name: str, obs: Any = None) -> dict[str, Any]:
        return {"status": "ok"}


def _official_loop(
    monkeypatch: pytest.MonkeyPatch,
    test_num: int,
    shard: tuple[int, int] | None,
    carry: dict[int, Any] | None = None,
    verdicts: Any = None,
    rerun: dict[int, int] | None = None,
    lose: frozenset[int] = frozenset(),
    latch: frozenset[int] = frozenset(),
) -> tuple[list[int], list[dict[str, Any]], int]:
    """The fake official loop over seed block 0 unsharded (`shard` None) or as `shard_client`'s
    shard k of K (carrying `carry`'s episodes, evaluating `rerun`'s again; infrastructure loses
    the episodes on the seeds in `lose`, and the relay says so for those in `latch` too, whose
    success latches after all): the seeds it set up for evaluation, its record and its
    successes."""
    from roboscript_robotwin.eval import shard_client

    acts = functools.partial(_never_acts, lose=lose, latch=latch) if lose or latch else _never_acts
    monkeypatch.setattr(official_fakes, "eval_function_decorator", lambda *a, **k: acts)
    task_env = official_fakes.FakeOfficialEnv()
    task_env.step_lim = 1
    args = {"task_name": TASK, "policy_name": "p", "task_config": CLEAN, "ckpt_setting": "mf"}
    args.update(clear_cache_freq=5, render_freq=0)
    record = io.StringIO()
    if shard is None:
        _, successes = official_fakes.eval_policy(
            TASK, task_env, args, _Recorder(), 100000, test_num, instruction_type="unseen"
        )
    else:
        _, successes = shard_client.shard_policy(
            official_fakes, *shard, None, record, TASK, task_env, args, _Recorder(), 100000,
            test_num=test_num, instruction_type="unseen", carry=carry, verdicts=verdicts,
            rerun=rerun,
        )  # fmt: skip
    lines = [json.loads(line) for line in record.getvalue().splitlines()]
    evaluated = [seed for seed, again in itertools.pairwise(task_env.setups) if seed == again]
    return evaluated, lines, successes


def _never_acts(
    task_env: Any,
    model: Any,
    observation: Any,
    lose: frozenset[int] = frozenset(),
    latch: frozenset[int] = frozenset(),
) -> None:
    task_env.take_action([0.0] * 14)
    seed = task_env.seed
    task_env.eval_success = seed in latch or (seed % 2 == 0 and seed not in lose)  # even: success
    if seed in lose | latch:  # as `deploy_policy.eval` notes a relay's infra answer
        from roboscript_robotwin.sim import shim

        shim.mark_infra(task_env)


def test_shards_evaluate_the_seeds_of_the_official_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every shard checks every seed as the official loop does (the unsolvable and unstable ones
    skipped), so the shards' episodes together are the unsharded loop's, by index and seed, each
    evaluated once, and their successes add up to its."""
    serial, _, successes = _official_loop(monkeypatch, 10, None)
    assert serial[:4] == [100000, 100002, 100004, 100005]  # 100001 unsolvable, 100003 unstable
    episodes: dict[int, tuple[int, int]] = {}
    total = 0
    for k in range(3):
        evaluated, lines, own = _official_loop(monkeypatch, 10, (k, 3))
        assert lines[0] == {"shard": k, "of": 3, "test_num": 10, "episodes": 10}
        solvable = [(line["index"], line["seed"]) for line in lines[1:] if "index" in line]
        assert solvable == list(enumerate(serial))  # each shard finds the official seeds
        for line in lines[1:]:
            if line["evaluated"]:
                assert line["index"] % 3 == k and line["index"] not in episodes
                episodes[line["index"]] = (line["seed"], line["success"])
        assert evaluated == [seed for index, seed in enumerate(serial) if index % 3 == k]
        total += own
    assert [episodes[index][0] for index in range(10)] == serial
    assert total == successes == sum(seed % 2 == 0 for seed in serial)


def _episodes(lines: list[dict[str, Any]]) -> list[tuple[int, int, bool]]:
    """A shard record's evaluated episodes (carried ones included): index, seed, success."""
    found = [(e["index"], e["seed"], e["success"]) for e in lines[1:] if e["evaluated"]]
    return sorted(found)


@pytest.mark.parametrize("shards", [1, 2])
def test_a_resumed_block_equals_the_uninterrupted_one(
    monkeypatch: pytest.MonkeyPatch, shards: int
) -> None:
    """An unsharded block interrupted after 4 finished episodes (the 5th in flight, lost) and
    resumed from them, unsharded or as 2 shards: every seed checked again in the official order,
    only episodes 4.. evaluated, and the block's episodes and successes are the uninterrupted
    loop's; the carried ones are in the new records (a second interruption carries them on)."""
    from roboscript_robotwin.eval import shard_client

    serial, _, successes = _official_loop(monkeypatch, 10, None)
    _, whole, _ = _official_loop(monkeypatch, 10, (0, 1))
    assert _episodes(whole) == [(i, seed, seed % 2 == 0) for i, seed in enumerate(serial)]
    carry = {i: shard_client.Carried(seed, ok) for i, seed, ok in _episodes(whole)[:4]}
    resumed: list[tuple[int, int, bool]] = []
    evaluated: list[int] = []
    total = 0
    for k in range(shards):
        seeds, lines, own = _official_loop(monkeypatch, 10, (k, shards), carry)
        solvable = [(line["index"], line["seed"]) for line in lines[1:] if "index" in line]
        assert solvable == list(enumerate(serial))  # the expert check gives the same seeds
        assert all(
            line.get("carried")
            for line in lines[1:]
            if line.get("index", 99) < 4 and line["evaluated"]
        )
        resumed += _episodes(lines)
        evaluated += seeds
        total += own
    assert sorted(resumed) == _episodes(whole)
    assert sorted(evaluated) == serial[4:]  # nothing carried runs again
    assert total == successes


def _claiming_shards(
    monkeypatch: pytest.MonkeyPatch, block: Path, of: int, slow: frozenset[int], test_num: int
) -> dict[int, tuple[list[dict[str, Any]], int, list[int]]]:
    """`of` shards of the fake official loop at once (threads), taking the episodes as they come
    free (`shard_client.Claims`, sharing their verdicts), an episode on a seed in `slow` taking
    half a second: by shard, its record, its successes and the seeds it set up."""
    from roboscript_robotwin.eval import shard_client

    monkeypatch.setattr(shard_client, "POLL_S", 0.005)

    def acts(task_env: Any, model: Any, observation: Any) -> None:
        if task_env.seed in slow:
            time.sleep(0.5)
        _never_acts(task_env, model, observation)

    monkeypatch.setattr(official_fakes, "eval_function_decorator", lambda *a, **k: acts)
    verdicts = shard_client.Verdicts(block / "shards" / launch.VERDICTS)
    claims = shard_client.Claims(block / "shards" / launch.CLAIMS)
    found: dict[int, tuple[list[dict[str, Any]], int, list[int]]] = {}
    errors: list[BaseException] = []

    def shard(k: int) -> None:
        task_env = official_fakes.FakeOfficialEnv()
        task_env.step_lim = 1
        args = {"task_name": TASK, "policy_name": "p", "task_config": CLEAN, "ckpt_setting": "mf"}
        args.update(clear_cache_freq=5, render_freq=0)
        record = io.StringIO()
        try:
            _, own = shard_client.shard_policy(
                official_fakes, k, of, None, record, TASK, task_env, args, _Recorder(), 100000,
                test_num=test_num, instruction_type="unseen", verdicts=verdicts, claims=claims,
            )  # fmt: skip
        except BaseException as error:  # reported by the test
            errors.append(error)
            return
        lines = [json.loads(line) for line in record.getvalue().splitlines()]
        found[k] = (lines, own, task_env.setups)

    threads = [threading.Thread(target=shard, args=(k,)) for k in range(of)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors and len(found) == of
    return found


def test_a_seed_gets_one_instruction_whichever_shard_runs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instruction is drawn from the seed: shards taking the episodes in another order (a
    different slow episode) give every evaluated seed the same instruction, so two arms evaluating
    a block on the same seeds compare the same instructions. The fake generator shuffles and fills
    its phrasings with Python's global `random`, as the official one does."""

    def generate(task: str, episodes: list[Any], n: int = 0) -> list[dict[str, list[str]]]:
        phrasings = [f"lift the {random.choice(['cup', 'mug', 'can'])} {k}" for k in range(8)]
        random.shuffle(phrasings)
        return [{"seen": phrasings, "unseen": phrasings} for _ in episodes]

    monkeypatch.setattr(official_fakes, "generate_episode_descriptions", generate)
    given: list[dict[int, str]] = []
    set_instruction = official_fakes.FakeOfficialEnv.set_instruction

    def records(self: official_fakes.FakeOfficialEnv, instruction: str) -> None:
        given[-1][self.seed] = instruction
        set_instruction(self, instruction)

    serial, _, _ = _official_loop(monkeypatch, 10, None)
    monkeypatch.setattr(official_fakes.FakeOfficialEnv, "set_instruction", records)
    for slow, run in ((serial[0], "a"), (serial[-1], "b")):
        given.append({})
        (tmp_path / run).mkdir()
        _claiming_shards(monkeypatch, tmp_path / run, 3, frozenset({slow}), 10)
    assert given[0] == given[1] and set(given[0]) == set(serial)
    assert len(set(given[0].values())) > 1


def _save_records(block: Path, records: list[list[dict[str, Any]]]) -> None:
    (block / "shards").mkdir(parents=True, exist_ok=True)
    for k, lines in enumerate(records):
        text = "".join(json.dumps(line) + "\n" for line in lines)
        (block / "shards" / f"shard{k}.jsonl").write_text(text, encoding="utf-8")


def test_an_episode_infrastructure_lost_is_not_finished(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An episode the relay said infrastructure lost has no task outcome, so it is not carried as a
    failure: the resumed run evaluates it alone again and carries the episodes after it."""
    block = tmp_path / "block1-aborted1"
    (block / "shards").mkdir(parents=True)
    rows = {  # (seed offset, index, evaluated, success); shard 0's episode 2 lost
        0: [(0, 0, True, False), (1, None, False, None), (2, 1, False, None), (3, 2, True, False),
            (4, 3, False, None)],
        1: [(0, 0, False, None), (1, None, False, None), (2, 1, True, True), (3, 2, False, None),
            (4, 3, True, False)],
    }  # fmt: skip
    for k, lines in rows.items():
        text = json.dumps({"shard": k, "of": 2, "test_num": 100, "episodes": 100}) + "\n"
        for offset, index, evaluated, success in lines:
            line: dict[str, Any] = {"seed": 200000 + offset, "evaluated": evaluated}
            if index is not None:
                line["index"] = index
            if evaluated:
                line["success"] = success
            if (k, index) == (0, 2):
                line["infra"] = True
            text += json.dumps(line) + "\n"
        (block / "shards" / f"shard{k}.jsonl").write_text(text, encoding="utf-8")
    found = [(e["index"], e["seed"]) for e in launch.finished(block)]
    assert found == [(0, 200000), (1, 200002), (3, 200004)]
    assert launch.lost(block) == [{"index": 2, "seed": 200003, "losses": 1}]
    assert launch._finished_per_client(block) == {0: 2, 1: 2}  # the lost one ended its slot


def _resumed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: Path,
    name: str,
    shards: int = 1,
    lose: frozenset[int] = frozenset(),
) -> tuple[Path, list[int]]:
    """A run of the block resumed from `source` as `launch.evaluate` starts one (its
    `carried.json` from `resume_plan`), its shards' records written: its directory and the seeds
    it set up for evaluation."""
    from roboscript_robotwin.eval import shard_client

    carried, rerun = launch.resume_plan(source)
    out = tmp_path / name
    out.mkdir()
    record = {"source": str(source), "episodes": carried, "rerun": rerun}
    (out / launch.CARRIED).write_text(json.dumps(record), encoding="utf-8")
    records, evaluated = [], []
    for k in range(shards):
        carry = shard_client.read_carry(out / launch.CARRIED)
        again = shard_client.read_rerun(out / launch.CARRIED)
        seeds, lines, _ = _official_loop(monkeypatch, 10, (k, shards), carry, None, again, lose)
        records.append(lines)
        evaluated += seeds
    _save_records(out, records)
    return out, sorted(evaluated)


def _records(block: Path) -> list[list[dict[str, Any]]]:
    return [launch._shard_record(path) for path in sorted((block / "shards").glob("*.jsonl"))]


def test_an_episode_lost_again_on_its_rerun_is_rerun_until_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lost again on its rerun, an episode is evaluated again on the next resume, its losses
    counted on; after `MAX_LOSSES` runs in a row that lost it a resume refuses, naming it,
    instead of looping. A resumed run cut before it reached the rerun keeps it to rerun, its
    losses as they were, and its carried episodes finished."""
    serial, _, _ = _official_loop(monkeypatch, 10, None)
    lose = frozenset({serial[3]})
    _, lines, _ = _official_loop(monkeypatch, 10, (0, 1), lose=lose)
    source = tmp_path / "block1-aborted1"
    _save_records(source, [lines])
    out, evaluated = _resumed(monkeypatch, tmp_path, source, "block1-aborted2", lose=lose)
    assert evaluated == [serial[3]]
    assert launch.lost(out) == [{"index": 3, "seed": serial[3], "losses": 2}]
    assert len(launch.finished(out)) == 9
    cut = tmp_path / "block1-aborted3"  # resumed, cut before any line
    cut.mkdir()
    carried, rerun = launch.resume_plan(out)
    record = {"source": str(out), "episodes": carried, "rerun": rerun}
    (cut / launch.CARRIED).write_text(json.dumps(record), encoding="utf-8")
    assert len(launch.finished(cut)) == 9
    assert launch.lost(cut) == [{"index": 3, "seed": serial[3], "losses": 2}]
    again, evaluated = _resumed(monkeypatch, tmp_path, cut, "block1-aborted4", lose=lose)
    assert launch.lost(again)[0]["losses"] == launch.MAX_LOSSES
    with pytest.raises(launch.LostRepeatedly, match=r"runs in a row"):
        launch.resume_plan(again)


def test_the_official_client_s_log_marks_an_episode_infrastructure_lost(tmp_path: Path) -> None:
    """A first run through the official client: its log counts a lost episode as a failure, and
    our policy's `INFRA_LOST` line before that episode's rate line marks it lost instead."""
    lines = []
    for index in range(4):
        if index == 2:
            lines.append(wire.INFRA_LOST)
        wins = sum(i % 2 == 0 and i != 2 for i in range(index + 1))
        lines.append(f"Success rate: {wins}/{index + 1} => 0.5%, current seed: {200000 + index}")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "client.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert [e["index"] for e in launch.finished(tmp_path)] == [0, 1, 3]
    assert launch.lost(tmp_path) == [{"index": 2, "seed": 200002, "losses": 1}]
    assert launch._finished_per_client(tmp_path) == {0: 4}


def relay_args(url: str, manifest: ManifestId, token_file: Path, **more: Any) -> dict[str, Any]:
    """A policy server's `usr_args` for the relay: `deploy_policy.yml` and the overrides."""
    official = {"policy_name": "RoboScript", "instruction_type": "unseen", "port": 1}
    return {
        **official,
        **server_overrides(["--task_name", TASK, "--task_config", CLEAN, "--seed", "0"]),
        "ckpt_setting": manifest,
        "host_url": url,
        "host_token_file": str(token_file),
        **more,
    }


@contextlib.contextmanager
def served(app: Any, directory: Path, name: str) -> Iterator[str]:
    """An ASGI app served from a thread (the relay's calls block); its URL once `/readyz` is 200."""
    stop, ready_file = threading.Event(), directory / f"{name}-ready.json"

    async def main() -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(app)
            await anyio.to_thread.run_sync(stop.wait)
            tg.cancel_scope.cancel()

    thread = threading.Thread(target=anyio.run, args=(main,))
    thread.start()
    try:
        for _ in range(300):
            ready = read_ready(ready_file)
            if (
                ready is not None
                and httpx.get(f"{ready.url}/readyz", trust_env=False).status_code == 200
            ):
                break
            assert thread.is_alive(), f"{name} stopped"
            stop.wait(0.1)
        assert ready is not None
        yield ready.url
    finally:
        stop.set()
        thread.join(30)


@agent_venv
async def test_episodes_through_the_relay_and_an_agent_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RoboTwin's loop as `script/eval_policy.py` runs it (policy and task object in one
    process), with the relay's `call` in place of the socket client."""
    manifest_path, manifest = await frozen(tmp_path)
    run = {"manifest": str(manifest_path), "blob_dir": str(tmp_path / "blobs")}
    per_run = {**run, "episodes": {"records": str(tmp_path / "host")}}
    config = AgentHostConfig.model_validate(compose(host_config_of(tmp_path), per_run))
    monkeypatch.setattr(deploy_policy, "POLL_S", 0.05)
    write_secret(tmp_path / "host.token", TOKEN)
    host = functools.partial(serve_host, config, TOKEN, tmp_path / "host-ready.json")
    with served(host, tmp_path, "host") as url:
        args = relay_args(url, manifest.id, tmp_path / "host.token", blob_dir=run["blob_dir"])
        relay = deploy_policy.get_model(args)
        assert isinstance(relay, Relay)
        for _ in range(2):
            task_env = FakeTaskEnv(step_lim=12)
            await anyio.to_thread.run_sync(deploy_policy.reset_model, relay)
            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                await anyio.to_thread.run_sync(deploy_policy.eval, task_env, relay, observation)
            assert task_env.actions[:2] == [HOLD, HOLD] and len(task_env.actions) == 12
    gone = await anyio.to_thread.run_sync(relay.reset_model)  # the host is gone
    assert gone["status"] == "failed" and "ConnectError" in gone["reason"]
    assert (await anyio.to_thread.run_sync(relay.poll))["status"] == "failed"
    for index in (0, 1):
        episode = tmp_path / "host" / KEY / f"ep{index:03d}"
        summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
        assert summary["end"]["reason"] == "finish" and summary["turns"] == 2
        official = {"success": False, "steps": 12, "step_limit": 12, "failed_calls": 0}
        assert summary["official"] == official and summary["infra"] is False
        assert summary["instruction"] == "lift it" and summary["manifest"] == manifest.id


class FakeHost:
    """An `agent.host` plane that answers as told, after `delays` (seconds per method) and with
    the method in `failing` raising; records the calls it got."""

    def __init__(self) -> None:
        self.delays: dict[str, float] = {}
        self.failing: set[str] = set()
        self.polls: list[Poll] = []
        self.calls: list[tuple[str, Any]] = []

    async def _call(self, method: str, value: Any = None) -> None:
        self.calls.append((method, value))
        await anyio.sleep(self.delays.get(method, 0.0))
        if method in self.failing:
            raise InfraError("not_ready", f"{method} fails for the test")

    async def reset(
        self, key: str, task: TaskName, manifest: ManifestId, instruction: str | None = None
    ) -> None:
        await self._call("reset", manifest)

    async def submit(self, key: str, observation: Observation) -> None:
        await self._call("submit", observation.step)

    async def poll(self, key: str) -> Poll:
        await self._call("poll")
        return self.polls.pop(0) if self.polls else Poll(status="pending")

    async def report(self, key: str, result: ExecuteResult) -> None:
        await self._call("report", result.executed)

    async def end(self, key: str, outcome: OfficialOutcome) -> None:
        await self._call("end", outcome)


@contextlib.contextmanager
def fake_host(directory: Path, **more: Any) -> Iterator[tuple[FakeHost, Relay]]:
    """A `FakeHost` over RPC, and a relay to it (`more`: relay settings)."""
    host = FakeHost()
    app = rpc.make_app(rpc.routes(AgentHost, host, token=TOKEN))
    serve = functools.partial(rpc.serve, app, ready_file=directory / "fake-ready.json")
    write_secret(directory / "host.token", TOKEN)
    manifest = ManifestId("mf-0123456789ab")
    with served(serve, directory, "fake") as url:
        args = relay_args(url, manifest, directory / "host.token", **more)
        yield host, Relay.from_args({**args, "blob_dir": str(directory / "blobs")})


def packed(step: int = 0) -> dict[str, Any]:
    """What `eval()` submits."""
    task_env, profile = FakeTaskEnv(step_lim=10), ROBOTWIN.profile()
    frames = shim.frames_of(task_env.get_obs(), list(profile.cameras))
    robot = shim.robot_state(task_env, profile).model_copy(update={"step": step})
    return wire.pack(frames, robot, "lift it")


def reported() -> dict[str, Any]:
    result = ExecuteResult(batch_id=BatchId("b1"), executed=0, stopped="completed", states=[])
    return result.model_dump(mode="json")


def test_every_relay_call_answers_within_its_deadline(tmp_path: Path) -> None:
    """A slow host cannot keep a relay call past `host_timeout_s`, far below the client's 30 s
    socket timeout; a submit (a submit and a poll) that takes longer answers `pending` and the
    next poll waits for it again, up to its own `submit_timeout_s`."""
    with fake_host(tmp_path, host_timeout_s=1.0, submit_timeout_s=3.0) as (host, relay):
        host.delays = {"submit": 0.6, "poll": 0.6}  # each within the deadline, not both
        assert relay.reset_model()["status"] == "ok"
        started = time.monotonic()
        assert relay.submit(packed())["status"] == "pending"
        assert time.monotonic() - started < 1.5
        host.polls = [Poll(status="done", reason="finish")]
        assert relay.poll()["status"] == "done"  # the submit's own poll, once through
        assert [method for method, _ in host.calls[-2:]] == ["submit", "poll"]
        assert relay.end({"success": False, "steps": 1, "step_limit": 10})["status"] == "ok"
        calls: dict[str, Callable[[], dict[str, Any]]] = {
            "reset": relay.reset_model,
            "poll": relay.poll,
            "report": lambda: relay.report(reported()),
            "end": lambda: relay.end({"success": False, "steps": 1, "step_limit": 10}),
        }
        for method, call in calls.items():
            host.delays = {}
            assert relay.reset_model()["status"] == "ok"
            host.delays = {method: 5.0}
            started = time.monotonic()
            assert call()["status"] == "failed", method
            assert time.monotonic() - started < 1.5, method
    args = relay_args("http://h", ManifestId("mf-0123456789ab"), tmp_path, host_timeout_s=25)
    with pytest.raises(ValidationError, match="less than or equal to 20"):
        RelayConfig.model_validate(args)


def fake_robotwin(directory: Path) -> tuple[Path, install.Expected]:
    """A RoboTwin checkout whose official scripts are the stand-ins of `official_fakes`."""
    root = directory / "RoboTwin"
    for name in ("script", "policy", "task_config", "envs"):
        (root / name).mkdir(parents=True)
    scripts = {
        "policy_model_server.py": "from official_fakes import server_main\n\nserver_main()\n",
        "eval_policy_client.py": official_fakes.CLIENT_SCRIPT,
    }
    for name, text in scripts.items():
        (root / "script" / name).write_text(text, encoding="utf-8")
    (root / "script" / "test_render.py").write_text(official_fakes.TEST_RENDER, encoding="utf-8")
    (root / "envs" / "_base_task.py").write_text("# the task base class\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", "script", "envs")
    git(root, "commit", "-qm", "stand-ins")
    files = {
        f"script/{name}": hashlib.sha256((root / "script" / name).read_bytes()).hexdigest()
        for name in scripts
    }
    return root, install.Expected(commit=git(root, "rev-parse", "HEAD"), files=files)


def git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def test_install_is_reversible(tmp_path: Path) -> None:
    root, expected = fake_robotwin(tmp_path)
    record = tmp_path / "install.json"
    with pytest.raises(ValueError, match="not RoboTwin"):
        install.install(root, record=record)  # the real commit is expected by default
    assert not install.installed(root)
    done = install.install(root, expected=expected, record=record)
    assert install.installed(root) and len(done.links) == 3 and done.files == expected.files
    assert (root / "policy" / "RoboScript" / "deploy_policy.yml").is_file()
    assert (root / "task_config" / f"{CLEAN}.yml").resolve().parent == HERE / "task_configs"
    assert install.install(root, expected=expected, record=record).links == done.links  # again
    assert install.check_installed(root, record=record).root == root.resolve()
    assert git(root, "status", "--porcelain", "--untracked-files=no") == ""  # links: untracked
    other, _ = fake_robotwin(tmp_path / "other")
    with pytest.raises(FileExistsError, match="P3 is installed in"):
        install.install(other, expected=expected, record=record)  # one checkout at a time
    assert sorted(install.uninstall(root, record=record)) == sorted(
        link.path for link in done.links
    )
    assert not install.installed(root) and not record.exists()
    assert install.uninstall(root, record=record) == []  # nothing installed: nothing to do
    with pytest.raises(RuntimeError, match="not installed"):
        install.check_installed(root, record=record)

    (root / "policy" / "RoboScript").mkdir()  # someone else's policy of that name
    with pytest.raises(FileExistsError, match="not our links"):
        install.install(root, expected=expected, record=record)
    assert not (root / "task_config" / f"{CLEAN}.yml").exists()  # nothing added
    (root / "policy" / "RoboScript").rmdir()
    install.install(root, expected=expected, record=record)
    (root / "envs" / "_base_task.py").write_text("# edited after the install\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted changes to tracked files"):
        install.check_installed(root, record=record)  # checked again before each evaluation
    git(root, "commit", "-qam", "an edit")
    with pytest.raises(ValueError, match="not RoboTwin"):
        install.check_installed(root, record=record)  # HEAD moved since the install
    (root / "script" / "eval_policy_client.py").write_text("# changed\n", encoding="utf-8")
    now = install.Expected(commit=git(root, "rev-parse", "HEAD"), files=expected.files)
    with pytest.raises(ValueError, match=r"eval_policy_client\.py changed"):
        install.check_robotwin(root, now)
    link = root / "task_config" / f"{CLEAN}.yml"
    link.unlink()
    link.symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="missing or changed"):
        install.check_installed(root, record=record)
    with pytest.raises(ValueError, match="no longer our links"):
        install.uninstall(root, record=record)


@contextlib.contextmanager
def fake_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fake RoboTwin checkout with P3 installed, as `ROBOTWIN_ROOT`, the install recorded in
    `tmp_path`, and a home there too (the launcher's locks and records live under it)."""
    root, expected = fake_robotwin(tmp_path)
    monkeypatch.setattr(install, "record_path", lambda: tmp_path / "install.json")
    install.install(root, expected=expected)
    monkeypatch.setenv("ROBOTWIN_ROOT", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    yield root


def p3_config(tmp_path: Path, host: dict[str, Any] | None = None, **more: Any) -> P3Config:
    return P3Config(
        interpreters=Interpreters(agent=Path(sys.executable), sim=Path(sys.executable)),
        sim_pythonpath=[REPO / "packages" / "core" / "src", SRC, Path(__file__).resolve().parent],
        host=host or host_config_of(tmp_path),
        server_ready_timeout_s=60,
        stop_grace_s=5,
        **more,
    )


def assert_gone(groups: Iterable[int]) -> None:
    for group in groups:  # every process group is gone
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)


def left_in(root: Path) -> list[str]:
    """What is left in the checkout's `eval_result/`, and whether the task's record of made
    directories is: `eval_result` alone (every task's clients share it), after runs that found no
    `eval_result/`."""
    found = sorted(str(p.relative_to(root)) for p in (root / "eval_result").rglob("*"))
    if (root / "eval_result").exists():
        found.insert(0, "eval_result")
    if launch._made_path(root, TASK).exists():
        found.append("the record of made directories")
    return found


@agent_venv
async def test_the_launcher_around_the_official_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "MOTION_MODULE", "fake_motion")  # MockPlanner, no cuRobo
    with fake_checkout(tmp_path, monkeypatch) as root:
        manifest_path, manifest = await frozen(tmp_path)
        host = host_config_of(tmp_path)
        host["toolsets"].append({"name": "manipulation"})
        config = p3_config(tmp_path, host, motion={"config": "motion"})
        run = P3Run(task=TASK, seed=0, manifest=manifest_path, out_dir=tmp_path / "full")
        with anyio.fail_after(120):
            result = await evaluate(run, config)
        out = tmp_path / "full"
        assert (result.success_rate, result.truncated, result.client_exit) == (0.0, False, 0)
        assert result.task_config == CLEAN and result.manifest == manifest.id
        assert result.official_dir == out / "official" and result.infra_episodes == 0
        official = (out / "official" / "_result.txt").read_text(encoding="utf-8")
        assert "Instruction Type: unseen" in official and official.endswith("0.0")
        assert left_in(root) == ["eval_result"]  # moved out, and what the client made removed
        assert [episode["official"]["steps"] for episode in result.episodes] == [12, 12]
        assert {episode["end"]["reason"] for episode in result.episodes} == {"finish"}
        assert json.loads((out / "result.json").read_text(encoding="utf-8"))["success_rate"] == 0.0
        assert json.loads((out / "pids.json").read_text(encoding="utf-8")) == result.pids
        assert list(result.pids)[:2] == ["motion", "agent-host"]  # the planner first
        mounted = yaml.safe_load((out / "agent_host.yaml").read_text(encoding="utf-8"))["toolsets"]
        planner = {"ready_file": str(out / ".roboscript" / "ready" / "motion.json")}
        place = json.loads(json.dumps(with_perception_place({})))  # as the YAML holds it
        expected = {"table_perception": place, "planner": planner, "table_top_z": 0.74}  # clean
        assert mounted[-1] == {
            "name": "manipulation",
            "config": expected,
            "skillsets": ["robotwin"],
        }
        assert_gone(result.pids.values())
        server_log = (out / "logs" / "policy_server.log").read_text(encoding="utf-8")
        assert "relay calls of an episode" in server_log and "; 0 failed" in server_log
        client_command = (out / "logs" / "client.log").read_text(encoding="utf-8").splitlines()[0]
        assert "--host_token_file" in server_log.splitlines()[0]  # the path, for the server only
        assert "--host_token_file" not in client_command and "--host_url" not in client_command
        assert not (out / ".roboscript" / "host.token").exists()  # the run's token, gone

        with pytest.raises(FileExistsError, match="is not empty"):
            await evaluate(run, config)  # a run directory of its own, always

        (root / "eval_result" / TASK / "Other_Policy").mkdir(parents=True)  # not ours: it stays
        smoke = run.model_copy(update={"out_dir": tmp_path / "smoke", "max_episodes": 1})
        monkeypatch.setenv("RS_FAKE_EPISODES", "5")  # more than the launcher lets it run
        with anyio.fail_after(120):
            cut = await evaluate(smoke, config)
        assert (cut.success_rate, cut.truncated, cut.client_exit) == (None, True, 0)
        assert len(cut.episodes) == 1  # the client stops after exactly its episodes
        assert cut.official_dir == tmp_path / "smoke" / "official"
        assert left_in(root) == [
            "eval_result",
            f"eval_result/{TASK}",
            f"eval_result/{TASK}/Other_Policy",
        ]
        assert cut.episodes[0]["official"]["steps"] == 12
        assert_gone(cut.pids.values())


async def test_p3_identifies_as_s2_did(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A task S2 practised with the vision service's cross-check runs only with it."""
    with fake_checkout(tmp_path, monkeypatch):
        store = FileMemoryStore(tmp_path / "memory")
        await store.commit(sample_memory(TASK), status="s1_draft", source=VersionSource(kind="s1"))
        await store.freeze(TASK, 0, reason="for the test", vision=True)
        write_manifest(tmp_path / "manifest.json", await build_manifest(store, [TASK]))
        out = tmp_path / "run"
        run = P3Run(task=TASK, seed=0, manifest=tmp_path / "manifest.json", out_dir=out)
        with pytest.raises(ValueError, match="with the vision service's cross-check"):
            await evaluate(run, p3_config(tmp_path))
    assert not out.exists()  # refused before anything started


"""A process that runs `evaluate` as a user would, with the install recorded in its directory
and the stand-in motion service."""


async def listening(ready_file: Path) -> None:
    """Wait until a service has written its ready file (it listens, warm or not)."""
    with anyio.fail_after(60):
        while True:
            if await anyio.Path(ready_file).is_file():
                return
            await anyio.sleep(0.1)
