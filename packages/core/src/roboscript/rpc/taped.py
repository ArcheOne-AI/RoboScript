"""The RPC edge of the replay tape (`roboscript.records.replay`): an interface's
implementation (an RPC proxy or the object in-process) with its calls taped.

`taped` wraps the motion and vision services (recorded, and served from the tape in a replay:
`REPLAYED`) and the env's control plane (recorded for comparison only, `OBSERVED_PLANES`);
`rpc.client` applies it to every client it makes. `observed` wraps the env's agent plane where
the Hub makes it (an RPC proxy, an in-process env, P3's bridge). Without a tape both return their
target as it is.
"""

import builtins
import json
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypeVar, cast

from roboscript.contracts.errors import Problem, RSError, error_from_problem
from roboscript.records.replay import OBSERVED_PLANES, REPLAYED, Tape, active
from roboscript.rpc.interface import Method, describe

T = TypeVar("T")

OPEN: Final = "env.control.open"
REOPENS: Final = 3
"""A replay opens a seed this many times more when the recording's pre-run (an env's optional
expert pre-run) found it solvable and the replay's did not (each open resets the seed's scene)."""


def taped(interface: type[T], target: T) -> T:
    """`target` with the calls the tape covers taped, or `target` itself without a tape or when
    the tape covers none of the interface's methods."""
    plane = describe(interface).contract.plane
    names = describe(interface).methods
    replayed = {name for name in names if f"{plane}.{name}" in REPLAYED}
    if not replayed and plane not in OBSERVED_PLANES:
        return target
    tape = active()
    if tape is None:
        return target
    return cast(T, Taped(interface, target, tape, observe_only=not replayed))


def observed(interface: type[T], target: T) -> T:
    """`target` with every call written to the comparison stream (never served), or `target`
    itself without a tape."""
    tape = active()
    if tape is None:
        return target
    return cast(T, Taped(interface, target, tape, observe_only=True))


def untaped(proxy: Any) -> Any:
    """What a `Taped` wraps (anything else as it is)."""
    return proxy.target if isinstance(proxy, Taped) else proxy


class Taped:
    """An interface's implementation whose methods go through the tape: covered ones recorded
    (record) or served (replay), the others called as always; with `observe_only`, every call
    goes to the comparison stream as `<plane>.<method>` with its request and answer. Other
    attributes are the target's."""

    def __init__(self, interface: type, target: Any, tape: Tape, *, observe_only: bool) -> None:
        spec = describe(interface)
        plane = spec.contract.plane
        self.target = target
        self.tape = tape
        calls = {}
        for name, method in spec.methods.items():
            stream = f"{plane}.{name}"
            if observe_only:
                calls[name] = _observed(tape, target, method, stream)
            elif stream in REPLAYED:
                calls[name] = _taped(tape, target, method, stream)
        self.__dict__.update(calls)

    def __getattr__(self, name: str) -> Any:  # only for what __init__ did not set
        return getattr(self.target, name)

    def __repr__(self) -> str:
        return f"<taped {self.target!r}>"


def _request(method: Method, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The request body as the wire has it, secrets masked."""
    arguments = method.signature.bind(*args, **kwargs).arguments
    body = method.params.dump_json(method.params.validate_python(arguments), by_alias=True)
    return json.loads(body)


def _dumped(method: Method, result: Any) -> Any:
    return json.loads(method.result.dump_json(result, by_alias=True))


def _failure(error: Exception) -> dict[str, Any]:
    if isinstance(error, RSError):
        return {"error": error.to_problem().model_dump(mode="json")}
    return {"exception": {"type": type(error).__name__, "message": str(error)}}


def _raise(recorded: dict[str, Any]) -> None:
    if "error" in recorded:
        raise error_from_problem(Problem.model_validate(recorded["error"]))
    raised = recorded["exception"]
    kind = getattr(builtins, raised["type"], None)
    kind = kind if isinstance(kind, type) and issubclass(kind, Exception) else RuntimeError
    raise kind(raised["message"])


def _taped(tape: Tape, target: Any, method: Method, stream: str) -> Callable[..., Awaitable[Any]]:
    async def call(*args: Any, **kwargs: Any) -> Any:
        request = _request(method, args, kwargs)
        if tape.replays(stream):
            recorded = tape.serve(stream, request)
            if "ok" not in recorded:
                _raise(recorded)
            return method.result.validate_json(json.dumps(recorded["ok"]))  # as from the wire
        try:
            result = await getattr(target, method.name)(*args, **kwargs)
        except Exception as error:
            tape.record(stream, request, _failure(error))
            raise
        tape.record(stream, request, {"ok": _dumped(method, result)})
        return result

    call.__name__ = call.__qualname__ = method.name
    return call


def _observed(
    tape: Tape, target: Any, method: Method, stream: str
) -> Callable[..., Awaitable[Any]]:
    async def call(*args: Any, **kwargs: Any) -> Any:
        request = _request(method, args, kwargs)
        if stream == OPEN and tape.mode == "replay":
            return await _open_as_recorded(tape, target, method, request, args, kwargs)
        try:
            result = await getattr(target, method.name)(*args, **kwargs)
        except Exception as error:
            tape.observe(stream, {"request": request, **_failure(error)})
            raise
        tape.observe(stream, {"request": request, "ok": _dumped(method, result)})
        return result

    call.__name__ = call.__qualname__ = method.name
    return call


async def _open_as_recorded(
    tape: Tape,
    target: Any,
    method: Method,
    request: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """`open` in a replay, which follows the recording where the env decides by chance: the
    verdict of its pre-run, where the env declares one (nondeterministic: `prerun_attempts`),
    and the first try's instruction, for an env whose instruction draw is not seeded. A seed the
    recording found unsolvable is answered as recorded without the simulator; one it found solvable
    is opened with the recorded instruction (unless the runner passed one), again up to `REOPENS`
    times while the pre-run fails. The comparison stream gets the runner's request as it made it."""
    recorded = tape.next_observed(OPEN) or {}
    ok = recorded.get("ok")
    if ok is not None and not ok["solvable"]:
        result = method.result.validate_python(ok)
    else:
        arguments = method.signature.bind(*args, **kwargs).arguments
        spec = arguments["spec"]
        if ok is not None and spec.instruction is None and ok.get("instruction") is not None:
            spec = spec.model_copy(update={"instruction": ok["instruction"]})
        open_ = getattr(target, method.name)
        try:
            result = await open_(spec)
            for _ in range(REOPENS if ok is not None else 0):
                if result.solvable:
                    break
                result = await open_(spec)
        except Exception as error:
            tape.observe(OPEN, {"request": request, **_failure(error)})
            raise
    tape.observe(OPEN, {"request": request, "ok": _dumped(method, result)})
    return result
