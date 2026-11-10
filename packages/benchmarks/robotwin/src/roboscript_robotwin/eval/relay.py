"""The model object of RoboTwin's policy server in P3 (`get_model` returns it): it relays one
official client's calls to the Agent host's `agent.host` plane (typed RPC, bearer token from the
run's token file), keeping the client's episode slot as its key.

RoboTwin's server calls `reset_model()`, `poll()`, `submit(packed)`, `report(result)` and
`end(outcome)` by name, one at a time, from its connection thread. None of them raises: the server
answers an exception with a frame the client cannot read, which ends the evaluation. Each call
answers within one deadline (`host_timeout_s`), well below the client's 30 s socket timeout.

A failure comes back as a `failed` answer, and from then on `poll`, `submit` and `report` answer
`failed` without calling the host until a `reset_model` succeeds: the episode is lost to
infrastructure, `eval()` holds still to the step limit at once (`LOST_HOLD`: the official loop
ends only there or at the success latch, and a chunk per `eval()` would render the cameras again
for every chunk), and no batch runs twice.

A `submit` writes the observation's frames to the blob store, which shares a disk that can stall
for tens of seconds. So a submit runs on a thread of its own with a deadline of its own,
`submit_timeout_s` (put off by the time the host stood still meanwhile, `roboscript.stack.stall`):
when it is not through within `host_timeout_s`, `submit` answers `pending`, `eval()` polls, and
each `poll` waits for it as long again until it is through (then the host is polled) or has
failed. `end` still reaches the host when the episode's reset did, with the number of failures
(each once, not the answers repeating it), so that such episodes are counted apart from the
agent's failures. Once the host says the agent is done, `poll` and `submit` say so without calling
it. Every answer carries `hold`, the chunk of actions `eval()` holds still for when the agent acts
no more. At each episode's end the relay logs how long its calls took.

Nothing on a call's way waits for the disk or for a call that overran: each host call runs on a
thread of its own, which the server's thread waits for at most `host_timeout_s` and `GUARD_S` (a
call still running then is left behind, its episode lost), and the process logs on a thread of its
own (`roboscript.records.log_in_background`): the server's log file may sit on a file system that
stalls for minutes.
"""

import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Final

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, SecretStr

from roboscript import rpc
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ExecuteResult
from roboscript.contracts.host import INFRA_ENDS, AgentHost, OfficialOutcome
from roboscript.contracts.ids import ManifestId, TaskName
from roboscript.records import log_in_background
from roboscript.settings import EnvSettings, read_secret
from roboscript.stack import stall
from roboscript_robotwin.eval import wire

log = logging.getLogger(__name__)

CLIENT_TIMEOUT_S: Final = 30.0
"""The socket timeout of RoboTwin's `ModelClient` (`script/eval_policy_client.py`), hard-coded."""
LOST_HOLD: Final = 1_000_000
"""The `hold` of a `failed` answer: the episode is lost, so `eval()` holds still to its limit."""
GUARD_S: Final = 2.0
"""Past a call's own deadline (`host_timeout_s`), how long the server's thread waits for it before
it answers `failed` all the same: whatever holds the call (a stalled disk, a connection that
will not close), the answer comes within 22 s, below `CLIENT_TIMEOUT_S`."""


class RelayConfig(BaseModel):
    """The fields of `deploy_policy.yml` (with the launcher's overrides) that the relay reads; the
    official scripts add keys of their own, which it ignores. `host_timeout_s` bounds each call
    as a whole, at most two thirds of the client's socket timeout."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    task_name: TaskName
    task_config: str
    ckpt_setting: ManifestId
    seed: NonNegativeInt
    host_url: str
    host_token_file: Path | None = None  # None: RS_HOST_TOKEN (the environment or .env)
    blob_dir: Path | None = None  # None: RS_BLOB_DIR, as the launcher sets it
    host_timeout_s: float = Field(default=20.0, gt=0, le=20)  # 2/3 of CLIENT_TIMEOUT_S
    # a submit as a whole, across polls: below the Agent host's `bridge.observe_timeout_s`
    submit_timeout_s: float = Field(default=40.0, gt=0)
    hold_actions: PositiveInt = 10
    shard: NonNegativeInt | None = None  # a shard's server (`P3Run.shards`): a slot of its own


Relayed = Callable[..., Awaitable[wire.Answer]]


class Relay:
    """Relays for one official client; its slot key names the task, config and seed block (and
    the shard, for a shard's)."""

    def __init__(self, config: RelayConfig, token: SecretStr) -> None:
        self.config = config
        self.token = token
        self.key = f"{config.task_name}.{config.task_config}.seed{config.seed}"
        if config.shard is not None:
            self.key += f".shard{config.shard}"
        blob_dir = config.blob_dir or EnvSettings().blob_dir
        if blob_dir is None:
            raise ValueError("set blob_dir in deploy_policy.yml or RS_BLOB_DIR")
        self.blobs = LocalBlobStore(blob_dir)
        self._durations: defaultdict[str, list[float]] = defaultdict(list)
        self._reset = False  # the episode's reset reached the host
        self._lost: str | None = None  # why the episode is lost, after a failed answer
        self._failed = 0  # calls that failed in the episode (each once)
        self._done: str | None = None  # why the agent is done, once the host said so
        self._submitting: _Submit | None = None  # a submit still running, across polls
        ready = httpx.get(
            f"{config.host_url}/readyz", timeout=config.host_timeout_s, trust_env=False
        )
        if ready.status_code != 200:
            raise RuntimeError(
                f"the Agent host at {config.host_url} is not ready ({ready.status_code})"
            )

    @classmethod
    def from_args(cls, usr_args: Mapping[str, Any]) -> "Relay":
        log_in_background(logging.INFO, "%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)  # a line per poll otherwise
        config = RelayConfig.model_validate(dict(usr_args))
        if config.host_token_file is not None:
            token = read_secret(config.host_token_file)
        elif (token := EnvSettings().host_token) is None:
            raise ValueError("set host_token_file in deploy_policy.yml, or RS_HOST_TOKEN")
        return cls(config, token)

    # Called by RoboTwin's policy server by name.

    def reset_model(self) -> wire.Answer:
        self._reset, self._lost, self._failed, self._done = False, None, 0, None
        self._submitting = None  # a lost episode's: left to end on its own
        answer = self._call("reset", self._reset_host)
        self._reset = answer["status"] == "ok"
        return answer

    def poll(self) -> wire.Answer:
        if self._submitting is not None and self._lost is None:
            return self._await_submit()
        return self._relay("poll", self._poll)

    def submit(self, packed: wire.Packed) -> wire.Answer:
        if self._lost is not None or self._done is not None:
            return self._relay("submit", self._submit, packed)  # answered without the host
        timeout = self.config.submit_timeout_s
        run = lambda: anyio.run(self._within, self._submit, (packed,), timeout)  # noqa: E731
        self._submitting = _Submit(run, timeout)
        return self._await_submit()

    def report(self, result: dict[str, Any]) -> wire.Answer:
        return self._relay("report", self._report, result)

    def end(self, outcome: dict[str, Any]) -> wire.Answer:
        if self._reset:  # the host's slot holds this episode, lost or not
            answer = self._call("end", self._end, outcome)
        else:  # the slot holds an earlier episode, whose outcome an end would overwrite
            answer = self._answer("failed", reason=self._lost or "the reset failed")
        calls = "; ".join(
            f"{method} x{len(took)} max {max(took):.3f} s"
            for method, took in sorted(self._durations.items())
        )
        log.info("relay calls of an episode of %s: %s; %d failed", self.key, calls, self._failed)
        self._durations.clear()
        return answer

    def call(self, func_name: str, obs: Any = None) -> Any:
        """`ModelClient.call`, for `script/eval_policy.py` (policy and simulator in one process)."""
        method = getattr(self, func_name)
        return method(obs) if obs is not None else method()

    # Internals

    async def _reset_host(self, host: AgentHost) -> wire.Answer:
        await host.reset(self.key, self.config.task_name, self.config.ckpt_setting)
        return self._answer("ok")

    async def _poll(self, host: AgentHost) -> wire.Answer:
        poll = await host.poll(self.key)
        if poll.status == "done":
            self._done = poll.reason or "done"
            log.info("%s: the agent is done (%s); eval() holds still", self.key, self._done)
        return self._answer(**poll.model_dump(mode="json"))

    async def _submit(self, host: AgentHost, packed: wire.Packed) -> wire.Answer:
        started = time.monotonic()
        observation = await wire.observation(packed, self.blobs)
        self._durations["encode"].append(time.monotonic() - started)  # the frames to the disk
        await host.submit(self.key, observation)
        return await self._poll(host)

    async def _report(self, host: AgentHost, result: dict[str, Any]) -> wire.Answer:
        await host.report(self.key, ExecuteResult.model_validate(result))
        return self._answer("ok")

    async def _end(self, host: AgentHost, outcome: dict[str, Any]) -> wire.Answer:
        official = OfficialOutcome.model_validate({**outcome, "failed_calls": self._failed})
        await host.end(self.key, official)
        return self._answer("ok")

    def _relay(self, method: str, relayed: Relayed, *args: Any) -> wire.Answer:
        """A call of the agent's turn: none once the episode is lost or the agent done."""
        if self._lost is not None:  # the failure is counted once, where it happened
            return self._answer("failed", reason=self._lost)
        if self._done is not None and method in ("poll", "submit"):
            return self._answer("done", reason=self._done)
        return self._call(method, relayed, *args)

    def _call(self, method: str, relayed: Relayed, *args: Any) -> wire.Answer:
        """One call of the host's plane, all of it within `host_timeout_s` (on a thread of its
        own, given up `GUARD_S` later if it is still running)."""
        started = time.monotonic()
        try:
            within = self.config.host_timeout_s + GUARD_S
            timeout = self.config.host_timeout_s
            return _bounded(lambda: anyio.run(self._within, relayed, args, timeout), within)
        except Exception as error:  # the server would end the evaluation instead
            return self._lose(method, error, self.config.host_timeout_s)
        finally:
            self._durations[method].append(time.monotonic() - started)

    def _await_submit(self) -> wire.Answer:
        """The submit in flight, waited for up to `host_timeout_s`: its answer (it polls the
        host after submitting), `pending` while it runs within its own deadline, `failed` once
        it failed or ran `GUARD_S` past that deadline (put off as its own is)."""
        submitting = self._submitting
        assert submitting is not None
        try:
            answer = submitting.wait(self.config.host_timeout_s)
        except Exception as error:  # the server would end the evaluation instead
            self._submitting = None
            self._durations["submit"].append(submitting.took())
            return self._lose("submit", error, self.config.submit_timeout_s)
        if answer is None:
            return self._answer("pending")
        self._submitting = None
        self._durations["submit"].append(submitting.took())
        return answer

    def _lose(self, method: str, error: Exception, timeout: float) -> wire.Answer:
        """The episode is lost to `error` (counted once)."""
        log.error("relaying %s of %s to the Agent host failed", method, self.key, exc_info=error)
        if isinstance(error, TimeoutError):
            error = TimeoutError(f"the Agent host did not answer {method} within {timeout:g} s")
        self._lost = self._lost or f"{type(error).__name__}: {error}"
        self._failed += 1
        return self._answer("failed", reason=self._lost)

    async def _within(self, relayed: Relayed, args: tuple[Any, ...], seconds: float) -> wire.Answer:
        host = rpc.client(
            AgentHost,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            self.config.host_url,
            token=self.token,
            timeout=seconds,
        )
        try:
            with stall.stretched(seconds):
                return await relayed(host, *args)
            raise TimeoutError(f"the call did not return within {seconds:g} s")
        finally:
            await rpc.aclose(host)

    def _answer(self, status: str, **fields: Any) -> wire.Answer:
        """`infra`: the episode is lost to infrastructure (a failed call, or the host ended it on
        such a reason), so the client records it as no task outcome (`shard_client`)."""
        hold = LOST_HOLD if status == "failed" else self.config.hold_actions
        infra = status == "failed" or (status == "done" and fields.get("reason") in INFRA_ENDS)
        return wire.answer(status, hold, infra=infra, **fields)


class _Submit:
    """A submit on a daemon thread of its own (`run()`), with its deadline: `timeout` from its
    start, put off by the time the host stood still since (at most `timeout` more), and
    `GUARD_S` past that for the thread to give up on its own."""

    def __init__(self, run: Callable[[], wire.Answer], timeout: float) -> None:
        self.timeout = timeout
        self.started = time.monotonic()
        self.stalls = stall.stall_totals()
        self.box: queue.SimpleQueue[tuple[bool, Any]] = queue.SimpleQueue()

        def target() -> None:
            try:
                self.box.put((True, run()))
            except BaseException as error:  # handed to the waiting thread
                self.box.put((False, error))

        threading.Thread(target=target, name="relay-submit", daemon=True).start()

    def wait(self, seconds: float) -> wire.Answer | None:
        """Its answer if it ends within `seconds`; None while it runs within its deadline;
        raises its error, or `TimeoutError` once it is past the deadline."""
        try:
            ok, value = self.box.get(timeout=max(0.0, min(seconds, self._left())))
        except queue.Empty:
            if self._left() > 0:
                return None
            raise TimeoutError(f"the submit did not return within {self.timeout:g} s") from None
        if ok:
            return value  # type: ignore[no-any-return]
        raise value

    def took(self) -> float:
        return time.monotonic() - self.started

    def _left(self) -> float:
        stalled = stall.stalled_since(self.stalls)
        return self.timeout + min(stalled, self.timeout) + GUARD_S - self.took()


def _bounded(call: Callable[[], wire.Answer], seconds: float) -> wire.Answer:
    """`call()` on a daemon thread, its answer or exception if it ends within `seconds`, else
    `TimeoutError` (the thread is left to finish on its own)."""
    box: queue.SimpleQueue[tuple[bool, Any]] = queue.SimpleQueue()

    def run() -> None:
        try:
            box.put((True, call()))
        except BaseException as error:  # handed to the waiting thread
            box.put((False, error))

    threading.Thread(target=run, name="relay-call", daemon=True).start()
    try:
        ok, value = box.get(timeout=seconds)
    except queue.Empty:
        raise TimeoutError(f"the call did not return within {seconds:g} s") from None
    if ok:
        return value  # type: ignore[no-any-return]
    raise value
