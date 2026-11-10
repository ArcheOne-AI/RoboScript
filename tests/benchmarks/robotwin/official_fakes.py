"""Stand-ins for RoboTwin's `script/policy_model_server.py` and `script/eval_policy_client.py`:
the same command line, overrides parsed as each script parses them (the server
`eval`s numbers only, the client every value), frames (4-byte length, then JSON with numpy arrays
as base64), policy loading from `./policy`, a server that binds its port in a thread and prints
"Model server started" once it listens (a failed bind leaves it running with a traceback),
dispatch by method name, and the client's `main` and episode loop (`eval_policy`: the expert check
on each seed, `unsolvable` and `unstable` ones skipped), with `FakeOfficialEnv` as the simulator
and `TEST_NUM` episodes instead of 100 (`RS_FAKE_EPISODES`, default 2); a client that exits 0 once
without `_result.txt` when `RS_TEST_CLIENT_NO_RESULT` names a marker file that does not exist yet.
A test writes a fake checkout whose server script calls `server_main()` and whose client script
is `CLIENT_SCRIPT`, beside `TEST_RENDER`."""

import argparse
import base64
import contextlib
import importlib
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import yaml
from robotwin_fakes import FakeTaskEnv

TEST_NUM = int(os.environ.get("RS_FAKE_EPISODES", "2"))
STEP_LIMIT = 12


class _Numpy(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            data = base64.b64encode(obj.tobytes()).decode("ascii")
            return {
                "__numpy_array__": True,
                "data": data,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer | np.floating | np.bool_):
            return obj.item()
        return super().default(obj)


def _array(item: dict[str, Any]) -> Any:
    if "__numpy_array__" in item:
        raw = base64.b64decode(item["data"])
        return np.frombuffer(raw, dtype=item["dtype"]).reshape(item["shape"])
    return item


def _send(sock: socket.socket, data: Any) -> None:
    payload = json.dumps(data, cls=_Numpy).encode()
    sock.sendall(len(payload).to_bytes(4, "big") + payload)


def _receive(sock: socket.socket) -> Any:
    size = int.from_bytes(_exactly(sock, 4), "big")
    return json.loads(_exactly(sock, size).decode(), object_hook=_array)


def _exactly(sock: socket.socket, size: int) -> bytes:
    chunks, left = [], size
    while left:
        chunk = sock.recv(min(left, 1 << 16))
        if not chunk:
            raise ConnectionError("the peer closed the connection")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def _config(parse: Any) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["port"] = args.port
    config.update(parse(args.overrides or []))
    return config


def server_overrides(pairs: list[str]) -> dict[str, Any]:
    """As `policy_model_server.py` parses `--overrides`: numbers `eval`ed, the rest as it is."""
    parsed, items = {}, iter(pairs)
    for key in items:
        value = next(items)
        parsed[key.lstrip("-")] = eval(value) if value.isnumeric() else value
    return parsed


def client_overrides(pairs: list[str]) -> dict[str, Any]:
    """As `eval_policy_client.py` parses `--overrides`: every value `eval`ed, kept as it is if
    that fails."""
    parsed = {}
    for index in range(0, len(pairs), 2):
        key, value = pairs[index].lstrip("-"), pairs[index + 1]
        with contextlib.suppress(Exception):  # the client's bare `except: pass`
            value = eval(value)
        parsed[key] = value
    return parsed


def _policy(config: dict[str, Any]) -> ModuleType:
    sys.path.append("./")
    sys.path.append("./policy")
    return importlib.import_module(config["policy_name"])


def server_main() -> None:
    config = _config(server_overrides)
    model = _policy(config).get_model(config)
    threading.Thread(target=_listen, args=(config["port"], model), daemon=True).start()
    while True:  # as RoboTwin's: the main thread waits, whatever became of the server's
        time.sleep(1)


def _listen(port: int, model: Any) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", port))
    server.listen(5)
    print(f"🚀 Model server started on localhost:{port}", flush=True)
    while True:
        connection, _ = server.accept()
        threading.Thread(target=_serve, args=(connection, model), daemon=True).start()


def _serve(connection: socket.socket, model: Any) -> None:
    with connection:
        while True:
            try:
                data = _receive(connection)
            except ConnectionError:
                return
            method, obs = getattr(model, data["cmd"]), data.get("obs")
            _send(connection, {"res": method(obs) if obs is not None else method()})


class ModelClient:
    def __init__(self, port: int) -> None:
        for _ in range(100):
            try:
                self.sock = socket.create_connection(("localhost", port), timeout=30)
                return
            except OSError:
                time.sleep(0.1)
        raise ConnectionError(f"no model server on port {port}")

    def call(self, func_name: str, obs: Any = None) -> Any:
        _send(self.sock, {"cmd": func_name, "obs": obs})
        return _receive(self.sock)["res"]


class UnStableError(Exception):
    """`envs.utils.create_actor.UnStableError`: a seed whose scene does not settle."""


def unsolvable(seed: int) -> bool:
    """The seeds the fake expert fails (10k + 1), as RoboTwin's fails some."""
    return seed % 10 == 1


def unstable(seed: int) -> bool:
    """The seeds whose fake scene does not settle (10k + 3)."""
    return seed % 10 == 3


class FakeOfficialEnv(FakeTaskEnv):
    """The task env as the official loop drives it: `setup_demo` on a seed, the expert's
    `play_once`, `close_env`; the counts start over at each setup."""

    def __init__(self) -> None:
        super().__init__(step_lim=STEP_LIMIT)
        self.seed, self.plan_success = -1, True
        self.eval_video_path, self.render_freq = None, 0
        self.setups: list[int] = []

    def setup_demo(self, now_ep_num: int, seed: int, is_test: bool, **args: Any) -> None:
        self.seed = seed
        self.setups.append(seed)
        self.take_action_cnt, self.eval_success, self.actions = 0, False, []

    def play_once(self) -> dict[str, Any]:
        if unstable(self.seed):
            raise UnStableError(f"seed {self.seed} did not settle")
        self.plan_success = not unsolvable(self.seed)
        return {"info": {"{A}": "the block"}}

    def check_success(self) -> bool:
        return True

    def close_env(self, clear_cache: bool = False) -> None:
        pass

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction


def parse_args_and_config() -> dict[str, Any]:
    return _config(client_overrides)


def eval_function_decorator(policy_name: str, model_name: str, conda_env: Any = None) -> Any:
    return getattr(importlib.import_module(policy_name), model_name)


def generate_episode_descriptions(
    task_name: str, episodes: list[dict[str, str]], max_descriptions: int = 1000000
) -> list[dict[str, list[str]]]:
    return [{"seen": ["lift it"], "unseen": ["lift it"]} for _ in episodes]


def eval_policy(
    task_name: str,
    TASK_ENV: Any,
    args: dict[str, Any],
    model: Any,
    st_seed: int,
    test_num: int = 100,
    video_size: str | None = None,
    instruction_type: str | None = None,
    policy_conda_env: str | None = None,
) -> tuple[int, int]:
    """The official loop (`script/eval_policy_client.py`), without the video and the viewer."""
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = succ_seed = 0
    now_seed = st_seed
    eval_func = eval_function_decorator(args["policy_name"], "eval")
    args["eval_mode"] = True
    while succ_seed < test_num:
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
        except UnStableError:
            TASK_ENV.close_env()
            now_seed += 1
            continue
        if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
            now_seed += 1
            continue
        succ_seed += 1
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        results = generate_episode_descriptions(task_name, [episode_info["info"]], test_num)
        TASK_ENV.set_instruction(instruction=np.random.choice(results[0][instruction_type]))
        succ = False
        model.call(func_name="reset_model")
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            eval_func(TASK_ENV, model, TASK_ENV.get_obs())
            if TASK_ENV.eval_success:
                succ = True
                break
        TASK_ENV.suc += succ
        print("Success!" if succ else "Fail!", flush=True)
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % args["clear_cache_freq"] == 0))
        TASK_ENV.test_num += 1
        print(f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num}, current seed: {now_seed}")
        now_seed += 1
    return now_seed, TASK_ENV.suc


def main(usr_args: dict[str, Any]) -> None:
    """The official `main`: the result directory, the seed block's first seed, the module's
    `eval_policy` (looked up when called, as the official one is) on `TEST_NUM` episodes, and
    `_result.txt`."""
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = ("task_name", "policy_name", "task_config", "ckpt_setting")
    save_dir = Path("eval_result", *(str(usr_args[key]) for key in parts), started)
    save_dir.mkdir(parents=True, exist_ok=True)
    _policy(usr_args)  # `./policy` on the path, as the official client's top level puts it
    args = {key: usr_args[key] for key in (*parts, "instruction_type")}  # the task config's, as
    args.update(clear_cache_freq=5, render_freq=0)  # the official `main` reads it, and the run's
    model = ModelClient(usr_args["port"])
    st_seed = 100000 * (1 + usr_args["seed"])
    kind = usr_args["instruction_type"]
    _, successes = eval_policy(
        usr_args["task_name"],
        FakeOfficialEnv(),
        args,
        model,
        st_seed,
        test_num=TEST_NUM,
        video_size="320x240",
        instruction_type=kind,
    )
    marker = os.environ.get("RS_TEST_CLIENT_NO_RESULT")
    if marker and not Path(marker).exists():  # once: exit 0 without the official result
        Path(marker).write_text("no result once", encoding="utf-8")
        return
    result = save_dir / "_result.txt"
    result.write_text(
        f"Timestamp: {started}\n\nInstruction Type: {kind}\n\n{successes / TEST_NUM}",
        encoding="utf-8",
    )
    print(f"Data has been saved to {result}", flush=True)


def client_main() -> None:
    main(parse_args_and_config())


CLIENT_SCRIPT: Final = """from official_fakes import (
    UnStableError, eval_function_decorator, generate_episode_descriptions, main,
    parse_args_and_config,
)

if __name__ == "__main__":
    main(parse_args_and_config())
"""
"""`script/eval_policy_client.py`'s stand-in: importable, as
`roboscript_robotwin.eval.shard_client` imports the official one."""
TEST_RENDER: Final = "def Sapien_TEST():\n    pass\n"
"""`script/test_render.py`'s stand-in."""
