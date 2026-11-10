"""Typed JSON-over-HTTP RPC between services (docs/architecture.md).

One interface class, a `typing.Protocol` with `PLANE` and `VERSION`, yields both server routes
(`routes`) and a client proxy (`client`), so the same code runs in-process (the object) or
remote (the proxy). Pydantic models travel as JSON; errors as problem+json; each request carries
a bearer token, `RS-Contract` and `traceparent`. `serve` runs an app and reports its URL
(`serve_while` also warms the service up once it listens, and stops one that failed for good:
`ServiceFault`); a service's ready file (`write_ready`
/ `read_ready`) tells the supervisor where it listens. `Worker` runs a service's calls one at a
time on a thread of its own (e.g. the one that owns its CUDA state).
"""

from roboscript.rpc.client import Transport, aclose, client, error_from_response
from roboscript.rpc.failures import service_failure, service_failures_as_infra
from roboscript.rpc.interface import describe
from roboscript.rpc.server import (
    Allow,
    Ready,
    ServiceFault,
    authorize,
    make_app,
    problem_response,
    read_ready,
    require_contract,
    routes,
    serve,
    serve_while,
    write_ready,
)
from roboscript.rpc.taped import observed, taped
from roboscript.rpc.worker import Worker

__all__ = [
    "Allow",
    "Ready",
    "ServiceFault",
    "Transport",
    "Worker",
    "aclose",
    "authorize",
    "client",
    "describe",
    "error_from_response",
    "make_app",
    "observed",
    "problem_response",
    "read_ready",
    "require_contract",
    "routes",
    "serve",
    "serve_while",
    "service_failure",
    "service_failures_as_infra",
    "taped",
    "write_ready",
]
