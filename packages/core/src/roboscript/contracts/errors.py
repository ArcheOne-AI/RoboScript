"""Errors that cross a process boundary, carried as RFC 9457 `application/problem+json`.

`ProtocolError`: the request is wrong (malformed, version, token, plane, unknown resource).
`ToolError`: a tool refuses; the model sees an error result. `EpisodeEnded`: the episode is over.
`InfraError`: retryable (busy, not ready, timeout, simulator fault). Business outcomes such as an
unreachable target are 200 receipts, not errors. `internal` marks an unexpected server failure;
its detail is sanitized and the traceback stays in the server log.
"""

from typing import ClassVar, Final, Literal, cast, get_args

from pydantic import BaseModel

MEDIA_TYPE: Final = "application/problem+json"

ProtocolCode = Literal[
    "invalid_request", "unsupported_version", "unauthorized", "forbidden_plane", "not_found"
]
InfraCode = Literal["busy", "not_ready", "timeout", "sim_fault"]
ErrorCode = Literal[ProtocolCode, InfraCode, "episode_ended", "tool_failed", "internal"]

# code -> (HTTP status, title)
_CODES: Final[dict[str, tuple[int, str]]] = {
    "invalid_request": (400, "Invalid request"),
    "unsupported_version": (400, "Unsupported contract version"),
    "unauthorized": (401, "Unauthorized"),
    "forbidden_plane": (403, "Forbidden plane"),
    "not_found": (404, "Not found"),
    "episode_ended": (409, "Episode ended"),
    "tool_failed": (422, "Tool failed"),
    "busy": (503, "Busy"),
    "not_ready": (503, "Not ready"),
    "timeout": (503, "Timed out"),
    "sim_fault": (503, "Simulator fault"),
    "internal": (500, "Internal error"),
}
_PROTOCOL_CODES: Final = frozenset(get_args(ProtocolCode))
_INFRA_CODES: Final = frozenset(get_args(InfraCode))


class Problem(BaseModel):
    """RFC 9457 problem details with the stable `code` and `retryable` extension members."""

    type: str
    title: str
    status: int
    code: str
    detail: str | None = None
    retryable: bool = False


class RSError(Exception):
    """Base of the errors above; used directly only for `internal` and unknown codes."""

    retryable: ClassVar[bool] = False

    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail

    @property
    def status(self) -> int:
        return _CODES.get(self.code, _CODES["internal"])[0]

    def to_problem(self) -> Problem:
        status, title = _CODES.get(self.code, _CODES["internal"])
        return Problem(
            type=f"urn:roboscript:problem:{self.code}",
            title=title,
            status=status,
            code=self.code,
            detail=self.detail,
            retryable=self.retryable,
        )


class ProtocolError(RSError):
    def __init__(self, code: ProtocolCode, detail: str | None = None) -> None:
        super().__init__(code, detail)


class ToolError(RSError):
    def __init__(self, detail: str | None = None) -> None:
        super().__init__("tool_failed", detail)


class EpisodeEnded(RSError):
    def __init__(self, detail: str | None = None) -> None:
        super().__init__("episode_ended", detail)


class InfraError(RSError):
    retryable = True

    def __init__(self, code: InfraCode, detail: str | None = None) -> None:
        super().__init__(code, detail)


def error_from_problem(problem: Problem) -> RSError:
    """The exception a `Problem` stands for, the inverse of `RSError.to_problem`.

    A code this version does not know (from a newer server) becomes an `InfraError` when the
    problem says it is retryable, else a plain `RSError`.
    """
    code, detail = problem.code, problem.detail
    if code in _PROTOCOL_CODES:
        return ProtocolError(cast(ProtocolCode, code), detail)
    if code in _INFRA_CODES or (code not in _CODES and problem.retryable):
        return InfraError(cast(InfraCode, code), detail)
    if code == "tool_failed":
        return ToolError(detail)
    if code == "episode_ended":
        return EpisodeEnded(detail)
    return RSError(code, detail)
