"""The model edge of the tape: the transport under an OpenAI client of the gateway
(`gateway.client.openai_client`), so that every model call (a brain's, a VLM's in perception, a
recipe's) is taped at one place, whatever the caller.

The stream of a call is `model:<path>` (`model:/v1/responses`), its request the JSON body. What
is recorded is what the client read: the status, the body and the gateway's `x-rs-*` headers
(the deployment that answered and its queue), or the transport error raised instead. A replay
answers from the record without a connection.

`tape_client` puts the tape under a client's own transport (record: the very transport the
client made, so its connection limits stay; replay: nothing under it). It is duck-typed on the
HTTP library: the OpenAI SDK runs on `httpx2`, a fork with httpx's API, which only the MCP
integration may import; requests, responses and errors are made from the request's own module.
"""

import base64
import json
import sys
from types import ModuleType
from typing import Any, Final, TypeVar

from roboscript.records.replay.tape import Tape

_KEPT_HEADERS: Final = ("content-type",)
"""Response headers a replay sends back besides the gateway's own (`x-rs-*`)."""

C = TypeVar("C")


def tape_client(client: C, tape: Tape) -> C:
    """`client` (an httpx or httpx2 `AsyncClient` without mounts) with `tape` under its
    transport."""
    inner = None if tape.replays("model:") else client._transport  # type: ignore[attr-defined]
    client._transport = TapeTransport(tape, inner)  # type: ignore[attr-defined]
    return client


REPLAY_URL: Final = "http://replay.invalid/v1"
"""The base URL of a replayed model client: the tape answers, so nothing listens there."""


def replay_client(session_id: str, tape: Tape | None = None, timeout: float = 900.0) -> Any:
    """An `openai.AsyncOpenAI` whose calls `tape` (the process's by default) answers: no
    gateway, token or network; a call the tape has no answer for ends the run (`ReplayMiss`)."""
    import openai  # only a model client needs it

    from roboscript.records.replay.tape import active

    tape = active() if tape is None else tape
    if tape is None or not tape.replays("model:"):
        raise ValueError("a replayed model client needs a replay tape: set RS_REPLAY=replay")
    http = tape_client(openai.DefaultAsyncHttpxClient(trust_env=False), tape)
    return openai.AsyncOpenAI(
        api_key=f"replay:{session_id}",
        base_url=REPLAY_URL,
        max_retries=0,
        timeout=timeout,
        http_client=http,
    )


class TapeTransport:
    """Records the calls `inner` answers (record mode) or answers them from the tape (replay;
    `inner` is then None)."""

    def __init__(self, tape: Tape, inner: Any | None) -> None:
        self.tape = tape
        self.inner = inner

    async def handle_async_request(self, request: Any) -> Any:
        http = _library(request)
        body = await request.aread()
        stream = f"model:{request.url.path}"
        payload = _payload(body)
        if self.tape.replays(stream):
            return _replayed(http, self.tape.serve(stream, payload), request)
        if self.inner is None:
            raise RuntimeError("a recording model transport needs the transport it records")
        try:
            response = await self.inner.handle_async_request(request)
            content = await response.aread()
        except http.HTTPError as error:
            raised = {"type": type(error).__name__, "message": str(error)}
            self.tape.record(stream, payload, {"raise": raised})
            raise
        self.tape.record(stream, payload, _recorded(response, content))
        return response  # read already: the client gets the content it would have read

    async def aclose(self) -> None:
        if self.inner is not None:
            await self.inner.aclose()


def _library(request: Any) -> ModuleType:
    """The HTTP library a request is of (`httpx` or `httpx2`)."""
    return sys.modules[type(request).__module__.split(".")[0]]


def _payload(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return {"bytes_b64": base64.b64encode(body).decode("ascii")}


def _recorded(response: Any, content: bytes) -> dict[str, Any]:
    headers = {
        name: value
        for name, value in response.headers.items()
        if name.lower() in _KEPT_HEADERS or name.lower().startswith("x-rs-")
    }
    try:
        body: dict[str, str] = {"text": content.decode("utf-8")}
    except UnicodeDecodeError:
        body = {"b64": base64.b64encode(content).decode("ascii")}
    return {"status": response.status_code, "headers": headers, **body}


def _replayed(http: ModuleType, recorded: dict[str, Any], request: Any) -> Any:
    if "raise" in recorded:
        raised = recorded["raise"]
        kind = getattr(http, raised["type"], None)
        if not (isinstance(kind, type) and issubclass(kind, http.HTTPError)):
            kind = http.TransportError
        if issubclass(kind, http.RequestError):
            raise kind(raised["message"], request=request)
        raise kind(raised["message"])
    content = (
        recorded["text"].encode("utf-8")
        if "text" in recorded
        else base64.b64decode(recorded["b64"])
    )
    return http.Response(
        recorded["status"], headers=recorded["headers"], content=content, request=request
    )
