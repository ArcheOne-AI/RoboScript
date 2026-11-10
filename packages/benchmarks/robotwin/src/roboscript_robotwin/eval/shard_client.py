"""One shard of a RoboTwin-compatible sharded evaluation (`roboscript_robotwin.eval.launch`,
`P3Run.shards` > 1): RoboTwin's own `script/eval_policy_client.py`, its `main` as it is (the task
config, the embodiment, the result directory, the video size, the `ModelClient`, `_result.txt`),
with `shard_policy` in place of its `eval_policy`.

`shard_policy` follows the official loop and differs from it in four ways. It evaluates only the
episodes whose index in the block is the shard's modulo the shard count. It draws each episode's
instruction with generators seeded by that episode's seed. `--carry` replays an interrupted run's
checks and evaluates only what that run did not finish. `--verdicts` and `--claims` change which
shard checks a seed and which takes an episode.

Without those options it runs the expert check on every seed in the official order, as the
unsharded client does, and the others' seeds it only checks. So every shard finds the solvable
seeds the unsharded client would, in its order (the check is seeded by the seed alone,
`_init_task_env_`), and the shards together evaluate the block's episodes, each as the official
loop runs it: the expert's `play_once` on its seed right before (some tasks' success checks read
what it chose), set up again, the instruction from that episode's info, the video named by the
episode's index (`episode<index>.mp4`), `eval()` until the step limit or the latch. The sim cache
is cleared every `clear_cache_freq` of the shard's own episodes, as the official loop clears it
every so many of its own. `_result.txt` holds the shard's successes over the official 100; the
launcher adds the shards' up.

Simulator env, from `ROBOTWIN_ROOT` (nothing of RoboTwin's changes):

    python -m roboscript_robotwin.eval.shard_client --shard K --of N --record FILE [--episodes M]
    -- \\
        --port P --config policy/RoboScript/deploy_policy.yml --overrides ...

`--record`: JSON lines, a header (`shard`, `of`, `test_num`, `episodes`) and then one line per
seed checked (`seed`, `index` when solvable, `evaluated`, `success` when evaluated, `infra` when
the relay said infrastructure lost the episode and it did not succeed: a success stands), written
as it goes, from which the launcher merges the shards and checks that they found the same seeds.
`--episodes`: only the block's first M episodes (short runs and demos).
`--carry`: the episodes an interrupted run of the block finished
(`roboscript_robotwin.eval.launch`'s `carried.json`: index, seed, success each) and those it lost
to infrastructure (`rerun`: index, seed each), episodes 0..m-1 together without a hole. A lost
episode is evaluated again on its seed and index, its expert check run right before it as the
official loop does (checked again up to `PINNED_RECHECKS` times; failing, the shard stops), by the
shard whose index it is, and recorded with `rerun`: true. The interrupted run's expert checks are
replayed, not run again (on some tasks the check does not repeat for a seed): up to the last
carried or rerun seed, a carried seed is that episode (its index's shard counts its success and
records it, `carried`: true, so that a run interrupted again carries it on) and any other seed was
not solvable (recorded with `replayed`: true); nothing of them is set up. The expert check runs
from the seed after the last carried one on. What can be checked without the simulator stops the
shard before it begins: the task, the task config and the block's first seed named in the file
(when it names them), and carried indices with a hole or seeds that do not rise with them or lie
outside the block.
`--verdicts`: the block's pinned expert verdicts (`Verdicts`), shared by its shards and carried to
a resumed run. On some tasks the expert check of a seed does not repeat (it depends on what the
client ran before), so shards that each check every seed can find different solvable seeds. With
the file, the first shard to check a seed pins its verdict and every shard follows it: a seed
pinned unsolvable is skipped, one pinned solvable is an episode index, and only the shard that
evaluates it runs the expert on it (right before its episode, as the official loop does); another
shard's seeds it does not check at all.
`--claims` (with `--verdicts`): the shards take the block's episodes as they come free instead of
by index modulo the shard count (`Claims`, a file of the run's own: the arms of a P4 block share
the verdicts, never the claims). A shard that reaches a seed nobody pinned takes its check, and
when the seed is solvable takes its episode before it pins the verdict, then evaluates it right
after its own check, as the official loop does; the other shards wait for the verdict and skip the
episode. So within a run no seed is checked twice and no episode runs on another client's check
(on some tasks the check does not repeat: a second client's check of a seed pinned solvable can
fail). A seed pinned solvable before (by the other arm of a P4 block) goes to the first shard to
reach it. A shard whose episodes end early so goes on with the next one instead of idling while
the slowest shard finishes its own. Carried and lost episodes of a resumed run keep the modulo
split.
"""

import argparse
import contextlib
import fcntl
import importlib
import json
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import IO, Any, Final, NamedTuple

import numpy as np

OFFICIAL_CLIENT = "eval_policy_client"  # script/eval_policy_client.py, imported as a module
POLL_S: Final = 2.0
"""How often a shard waiting for another's verdict on a seed reads the verdicts (`--claims`)."""
PINNED_RECHECKS: Final = 2
"""How often a shard checks the seed of its own episode again when another shard (or a check
before it in its client) pinned the seed solvable and its own check failed: on some tasks the
check does not repeat."""


class Carried(NamedTuple):
    """An episode an interrupted run of the block finished: its seed and official success."""

    seed: int
    success: bool


class Verdicts:
    """The expert's verdicts on a block's seeds, one JSON line each (`seed`, `solvable`, `shard`)
    in a file its shards share: the first line of a seed is its verdict (first writer wins, under
    an exclusive lock), so every shard indexes the block alike, also across a resumed run that
    starts from a copy (`roboscript_robotwin.eval.launch`)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def get(self, seed: int) -> bool | None:
        """The pinned verdict on `seed`, None while no shard checked it."""
        with self._locked() as file:
            return _first(file, seed)

    def pin(self, seed: int, solvable: bool, shard: int) -> bool:
        """Pin this shard's verdict on `seed` unless another shard did first; the pinned one."""
        with self._locked() as file:
            pinned = _first(file, seed)
            if pinned is None:
                file.write(json.dumps({"seed": seed, "solvable": solvable, "shard": shard}) + "\n")
                file.flush()
                return solvable
            return pinned

    @contextlib.contextmanager
    def _locked(self) -> Iterator[IO[str]]:
        with self.path.open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                yield file
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)


class Claims:
    """Which shard takes what of a block's run, one JSON line each (`key`, `shard`) in a file its
    shards share: the first line of a key is its claim (first writer wins, under an exclusive
    lock). Keys: `check:<seed>` (the expert check of a seed nobody pinned) and `episode:<index>`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def take(self, key: str, shard: int) -> bool:
        """Claim `key` for `shard` unless another shard did first; whether `shard` holds it."""
        with self.path.open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                file.seek(0)
                for line in file:
                    if line.strip() and (claim := json.loads(line))["key"] == key:
                        return bool(claim["shard"] == shard)
                file.write(json.dumps({"key": key, "shard": shard}) + "\n")
                file.flush()
                return True
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)


def _first(file: IO[str], seed: int) -> bool | None:
    file.seek(0)
    for line in file:
        if line.strip() and (verdict := json.loads(line))["seed"] == seed:
            return bool(verdict["solvable"])
    return None


def read_carry(path: Path) -> dict[int, Carried]:
    """The carried episodes of a `carried.json`, by index."""
    episodes = json.loads(path.read_text(encoding="utf-8"))["episodes"]
    return {int(e["index"]): Carried(int(e["seed"]), bool(e["success"])) for e in episodes}


def check_carry(path: Path, task: str | None, task_config: str | None, st_seed: int) -> None:
    """Stop (`SystemExit`) unless `carried.json` is of this block: its task, task config and
    first seed when it names them."""
    record = json.loads(path.read_text(encoding="utf-8"))
    ours = {"task": task, "task_config": task_config, "st_seed": st_seed}
    for key, value in ours.items():
        if value is not None and key in record and record[key] != value:
            raise SystemExit(f"{path} carries {key} {record[key]!r}, this block is {value!r}")


def read_rerun(path: Path) -> dict[int, int]:
    """The episodes of a `carried.json` that infrastructure lost, to evaluate again: index ->
    seed."""
    episodes = json.loads(path.read_text(encoding="utf-8")).get("rerun", [])
    return {int(e["index"]): int(e["seed"]) for e in episodes}


def _check_carried(
    carry: Mapping[int, Carried],
    st_seed: int,
    test_num: int,
    rerun: Mapping[int, int] | None = None,
) -> None:
    """The carried episodes and those to evaluate again are 0..m-1 together, apart, on seeds that
    rise with them, within the block."""
    rerun = rerun or {}
    if both := sorted(set(carry) & set(rerun)):
        raise RuntimeError(f"episodes {both} are both carried and to be evaluated again")
    seeds_of = {index: c.seed for index, c in carry.items()} | dict(rerun)
    if sorted(seeds_of) != list(range(len(seeds_of))):
        shown = sorted(seeds_of)
        raise RuntimeError(f"the carried and rerun episodes {shown} are not 0..m-1 without a hole")
    seeds = [seeds_of[index] for index in range(len(seeds_of))]
    if seeds != sorted(set(seeds)) or (seeds and seeds[0] < st_seed):
        raise RuntimeError(f"the carried seeds {seeds} do not rise from the block's {st_seed}")
    if len(seeds_of) > test_num:
        raise RuntimeError(f"{len(seeds_of)} carried episodes, the block has {test_num}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--carry", type=Path)
    ap.add_argument("--verdicts", type=Path)
    ap.add_argument("--claims", type=Path)
    ap.add_argument("official", nargs=argparse.REMAINDER, help="the official client's arguments")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if not 0 <= args.shard < args.of:
        raise SystemExit(f"--shard {args.shard} is not one of {args.of} shards")
    if args.episodes is not None and args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.claims is not None and args.verdicts is None:
        raise SystemExit("--claims needs --verdicts: the shards follow each other's verdicts")
    official = args.official[1:] if args.official[:1] == ["--"] else args.official
    carry = {} if args.carry is None else read_carry(args.carry)
    rerun = {} if args.carry is None else read_rerun(args.carry)
    verdicts = None if args.verdicts is None else Verdicts(args.verdicts)
    claims = None if args.claims is None else Claims(args.claims)
    sys.path.insert(0, str(Path("script").resolve()))
    client = importlib.import_module(OFFICIAL_CLIENT)
    home = sys.modules[client.main.__module__]  # where `main` looks `eval_policy` up
    from test_render import Sapien_TEST  # as the official client's __main__, in this order

    Sapien_TEST()
    sys.argv = ["script/eval_policy_client.py", *official]
    usr_args = client.parse_args_and_config()
    if args.carry is not None:
        st_seed = 100000 * (1 + int(usr_args.get("seed", 0)))  # as the official `main` sets it
        check_carry(args.carry, usr_args.get("task_name"), usr_args.get("task_config"), st_seed)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("w", encoding="utf-8") as record:

        def eval_policy(*positional: Any, **named: Any) -> tuple[int, int]:
            return shard_policy(
                client,
                args.shard,
                args.of,
                args.episodes,
                record,
                *positional,
                carry=carry,
                verdicts=verdicts,
                rerun=rerun,
                claims=claims,
                **named,
            )

        home.eval_policy = eval_policy
        client.main(usr_args)


def shard_policy(
    client: ModuleType,
    shard: int,
    of: int,
    episodes: int | None,
    record: IO[str],
    task_name: str,
    TASK_ENV: Any,
    args: dict[str, Any],
    model: Any,
    st_seed: int,
    test_num: int = 100,
    video_size: str | None = None,
    instruction_type: str | None = None,
    policy_conda_env: str | None = None,
    carry: Mapping[int, Carried] | None = None,
    verdicts: Verdicts | None = None,
    rerun: Mapping[int, int] | None = None,
    claims: Claims | None = None,
) -> tuple[int, int]:
    """The official `eval_policy` for one shard (see the module docstring), line by line where
    it can be; the episodes in `carry` (by index) are counted, not evaluated again, those in
    `rerun` (index -> seed: infrastructure lost them) are evaluated again on their seed and
    index, and the expert checks up to the last of either are replayed, not run (but for the
    reruns' own); with `verdicts`, every shard follows the verdict the first one to check a seed
    pinned; with `claims` (and `verdicts`), the shards take the episodes after those as they come
    free (see the module docstring)."""
    from roboscript_robotwin.sim import shim

    carry, rerun = carry or {}, rerun or {}
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    print(f"shard {shard} of {of}")
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    succ_seed = 0
    ran = 0  # this shard's episodes
    eval_func = client.eval_function_decorator(
        args["policy_name"], "eval", conda_env=policy_conda_env
    )
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True
    count = test_num if episodes is None else min(episodes, test_num)
    _check_carried(carry, st_seed, test_num, rerun)
    by_seed = {carried.seed: index for index, carried in carry.items()}
    by_seed |= {seed: index for index, seed in rerun.items()}
    window = len(carry) + len(rerun)
    header = {"shard": shard, "of": of, "test_num": test_num, "episodes": count}
    _write(record, header if claims is None else {**header, "claims": True})
    if claims is not None and verdicts is None:
        raise ValueError("claims need verdicts: the shards follow each other's verdicts")

    def ours(index: int) -> bool:
        """Whether this shard evaluates episode `index` (past the replayed window)."""
        if claims is None:
            return index % of == shard
        return claims.take(f"episode:{index}", shard)

    while window and succ_seed < min(count, window):  # the interrupted run's checks, replayed
        index = by_seed.get(now_seed)
        if index is None:  # not solvable then
            _write(record, {"seed": now_seed, "evaluated": False, "replayed": True})
        elif index in rerun and index % of == shard:  # lost to infrastructure: again, alone
            episode_info, error, _ = _expert_check(client, TASK_ENV, args, index, now_seed)
            for _ in range(PINNED_RECHECKS):
                if error is None:
                    break
                print(f"seed {now_seed}: its expert check failed ({error}) before its rerun")
                episode_info, error, _ = _expert_check(client, TASK_ENV, args, index, now_seed)
            if error is not None:
                line = {"seed": now_seed, "index": index, "evaluated": False, "error": error}
                _write(record, {**line, "rerun": True})
                raise SystemExit(
                    f"seed {now_seed} (episode {index}) is to be evaluated again, but its expert "
                    f"check failed {1 + PINNED_RECHECKS} times ({error}): no episode info to "
                    "evaluate it with"
                )
            succ = _episode(
                client, TASK_ENV, args, model, eval_func, episode_info, index, now_seed,
                test_num, video_size, instruction_type,
            )  # fmt: skip
            TASK_ENV.suc += succ
            ran += 1
            TASK_ENV.close_env(clear_cache=((ran + 1) % clear_cache_freq == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            line = {"seed": now_seed, "index": index, "evaluated": True, "success": succ}
            line["rerun"] = True
            _write(record, {**line, "infra": True} if shim.infra(TASK_ENV) and not succ else line)
            print(
                f"{task_name} | {args['policy_name']} | {args['task_config']} | "
                f"{args['ckpt_setting']} | shard {shard} of {of}\n"
                f"Success rate: {TASK_ENV.suc}/{ran}, episode {index} (again), current seed: "
                f"{now_seed}\n"
            )
        elif index in rerun:
            _write(record, {"seed": now_seed, "index": index, "evaluated": False, "rerun": True})
        elif index % of == shard:
            carried = carry[index]
            TASK_ENV.suc += carried.success
            line = {"index": index, "evaluated": True, "success": carried.success}
            _write(record, {"seed": now_seed, **line, "carried": True})
        else:
            _write(record, {"seed": now_seed, "index": index, "evaluated": False, "carried": True})
        if index is not None:
            succ_seed += 1
        now_seed += 1

    while succ_seed < count:
        pinned = None if verdicts is None else verdicts.get(now_seed)
        if (
            verdicts is not None
            and claims is not None
            and pinned is None
            and not claims.take(f"check:{now_seed}", shard)
        ):  # another shard checks it
            pinned = _await_verdict(verdicts, now_seed)
        if pinned is False or (pinned and not ours(succ_seed)):  # followed, not checked
            line = {"seed": now_seed, "evaluated": False, "pinned": True}
            if pinned:
                line["index"] = succ_seed
                succ_seed += 1
            _write(record, line)
            now_seed += 1
            continue
        episode_info, error, solvable = _expert_check(client, TASK_ENV, args, succ_seed, now_seed)
        for _ in range(PINNED_RECHECKS if pinned else 0):  # its episode, pinned solvable
            if error is None:
                break
            print(f"seed {now_seed}: pinned solvable, the expert check failed ({error}); again")
            episode_info, error, solvable = _expert_check(
                client, TASK_ENV, args, succ_seed, now_seed
            )
        if verdicts is not None:
            if pinned is None:
                if claims is not None and solvable:  # its episode runs on this check
                    claims.take(f"episode:{succ_seed}", shard)
                pinned = verdicts.pin(now_seed, solvable, shard)
            if pinned != solvable:  # another shard's verdict stands
                print(f"seed {now_seed}: this expert check says {solvable}, pinned {pinned}")
            if pinned and error is not None:  # no episode info to evaluate it with
                line = {"seed": now_seed, "index": succ_seed, "evaluated": False, "error": error}
                _write(record, {**line, "pinned": True})
                # stopping, not skipping: a skipped index is a hole no resume fills, so the
                # block would end short of its episodes on every run; the launcher fails the
                # run and a resumed one checks this seed afresh (`launch.evaluate`)
                raise SystemExit(
                    f"seed {now_seed} (episode {succ_seed}) is pinned solvable, but its expert "
                    f"check failed {1 + PINNED_RECHECKS} times ({error}): no episode info to "
                    "evaluate it with"
                )
            solvable = pinned
        if not solvable:
            line = {"seed": now_seed, "evaluated": False}
            _write(record, line if error is None else {**line, "error": error})
            now_seed += 1
            continue
        index = succ_seed
        succ_seed += 1
        if not ours(index):  # another shard's episode
            _write(record, {"seed": now_seed, "index": index, "evaluated": False})
            now_seed += 1
            continue

        succ = _episode(
            client, TASK_ENV, args, model, eval_func, episode_info, index, now_seed, test_num,
            video_size, instruction_type,
        )  # fmt: skip

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        ran += 1
        TASK_ENV.close_env(clear_cache=((ran + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        line = {"seed": now_seed, "index": index, "evaluated": True, "success": succ}
        # lost to infrastructure: no task outcome, so a resumed run evaluates it again (a
        # success that latched while the client held still after the agent's end is one)
        _write(record, {**line, "infra": True} if shim.infra(TASK_ENV) and not succ else line)
        print(
            f"{task_name} | {args['policy_name']} | {args['task_config']} | "
            f"{args['ckpt_setting']} | shard {shard} of {of}\n"
            f"Success rate: {TASK_ENV.suc}/{ran}, episode {index}, current seed: {now_seed}\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def _await_verdict(verdicts: Verdicts, seed: int) -> bool:
    """The verdict another shard pins on `seed` (it took its check)."""
    while (pinned := verdicts.get(seed)) is None:
        time.sleep(POLL_S)
    return pinned


def _episode(
    client: ModuleType,
    TASK_ENV: Any,
    args: dict[str, Any],
    model: Any,
    eval_func: Any,
    episode_info: Any,
    index: int,
    seed: int,
    test_num: int,
    video_size: str | None,
    instruction_type: str | None,
) -> bool:
    """One episode as the official loop runs it (its expert check just ran): set up on `seed` as
    episode `index`, the instruction drawn by the seed from its info's, the video named by its
    index, `eval()` until the step limit or the latch; whether it succeeded."""
    from roboscript_robotwin.sim import shim

    TASK_ENV.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
    # the official generator shuffles and fills its phrasings with Python's global `random`, and
    # the pick below would follow numpy's: both are seeded from the episode's seed, not left to the
    # process's state (which follows the shard's episode order), so a seed gets one instruction in
    # every shard and run and paired arms compare alike; the process's state is put back after
    state = random.getstate()
    random.seed(seed)
    try:
        results = client.generate_episode_descriptions(
            args["task_name"], [episode_info["info"]], test_num
        )
    finally:
        random.setstate(state)
    instruction = np.random.RandomState(seed).choice(results[0][instruction_type])
    TASK_ENV.set_instruction(instruction=instruction)
    TASK_ENV.test_num = index  # names the video as the unsharded client would

    if TASK_ENV.eval_video_path is not None:
        ffmpeg = subprocess.Popen(
            [
                *("ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo"),
                *("-pixel_format", "rgb24", "-video_size", str(video_size)),
                *("-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p"),
                *("-vcodec", "libx264", "-crf", "23"),
                f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
            ],
            stdin=subprocess.PIPE,
        )
        TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

    succ = False
    shim.clear_infra(TASK_ENV)
    model.call(func_name="reset_model")
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        observation = TASK_ENV.get_obs()
        eval_func(TASK_ENV, model, observation)
        if TASK_ENV.eval_success:
            succ = True
            break
    if TASK_ENV.eval_video_path is not None:
        TASK_ENV._del_eval_video_ffmpeg()
    return succ


def _expert_check(
    client: ModuleType, TASK_ENV: Any, args: dict[str, Any], ep_num: int, seed: int
) -> tuple[Any, str | None, bool]:
    """The official loop's expert check of `seed`: the episode info, the error (`unstable`,
    `exception`) and whether the seed is solvable."""
    render_freq = args["render_freq"]
    args["render_freq"] = 0
    error = None
    episode_info = None
    try:
        TASK_ENV.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
        episode_info = TASK_ENV.play_once()
        TASK_ENV.close_env()
    except client.UnStableError as e:
        print(" -------------")
        print("Error: ", e)
        print(" -------------")
        TASK_ENV.close_env()
        error = "unstable"
    except Exception:
        print(" -------------")
        print("Error: ", traceback.format_exc())
        print(" -------------")
        TASK_ENV.close_env()
        print("error occurs !")
        error = "exception"
    args["render_freq"] = render_freq
    solvable = error is None and bool(TASK_ENV.plan_success and TASK_ENV.check_success())
    return episode_info, error, solvable


def _write(record: IO[str], line: dict[str, Any]) -> None:
    record.write(json.dumps(line) + "\n")
    record.flush()


if __name__ == "__main__":
    main()
