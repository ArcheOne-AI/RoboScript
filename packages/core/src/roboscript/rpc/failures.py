"""Telling a service failing from a call going wrong (docs/concepts/learning.md).

A step that calls services on the episode's behalf and whose failure should void the episode
rather than stop the run (the open-time identification: vision, motion, a model API) wraps the
calls in `service_failures_as_infra`: a server error of one of ours (`RSError("internal")`, what
an HTTP 5xx becomes, e.g. a vision service out of CUDA memory), an HTTP status of 500 or more
from a model API's own client, or a connection that failed becomes an `InfraError`. Anything
else (an invalid request, a tool's own error such as a model refusing, a bug) goes on as raised.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from roboscript.contracts.errors import InfraError, RSError


def service_failure(error: BaseException) -> bool:
    """Whether `error`, or an error it was raised from, is a service failing (see the module)."""
    seen: BaseException | None = error
    while seen is not None:
        if type(seen) is RSError and seen.code == "internal":
            return True
        if isinstance(seen, ConnectionError | httpx.TransportError):
            return True
        status = getattr(seen, "status_code", None)
        if isinstance(status, int) and status >= 500:
            return True
        seen = seen.__cause__
    return False


@contextmanager
def service_failures_as_infra(what: str) -> Iterator[None]:
    """Raise a service failure inside as an `InfraError` (`not_ready`) naming `what` failed."""
    try:
        yield
    except InfraError:
        raise
    except Exception as error:
        if not service_failure(error):
            raise
        detail = f"{what}: a service failed ({type(error).__name__}: {error})"
        raise InfraError("not_ready", detail) from error
