"""Contract versions per plane and the `RS-Contract` header
(docs/architecture.md).

A plane's version is `MAJOR.MINOR`: minors only add (optional fields, methods, codes), a major
break is refused. Clients send `RS-Contract: <plane>/<major.minor>` on every request.
"""

import re
from types import MappingProxyType
from typing import Final, NamedTuple

HEADER: Final = "RS-Contract"

VERSIONS: Final = MappingProxyType(
    {
        "env.agent": "1.2",
        "env.control": "2.0",
        "env.public": "1.2",
        "robot.control": "1.0",  # a real-robot server's control face (contracts.robot_server)
        "tool": "1.4",
        "event": "1",  # the event log schema is `rs.event/1`: majors only
        "hub.control": "2.6",
        "hub.permits": "1.0",
        "agent.host": "1.0",
        "gateway.admin": "1.0",
        "blobs": "1.0",
        "motion": "1.2",
        "vision": "1.0",
    }
)

_CONTRACT = re.compile(r"([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)/(\d+)(?:\.(\d+))?")


class Contract(NamedTuple):
    plane: str
    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.plane}/{self.major}.{self.minor}"


def parse(text: str) -> Contract:
    """Parse `<plane>/<major>[.<minor>]`, e.g. `env.agent/1.1`; a missing minor is 0."""
    match = _CONTRACT.fullmatch(text)
    if match is None:
        raise ValueError(f"not a contract version like 'env.agent/1.0': {text!r}")
    return Contract(match[1], int(match[2]), int(match[3] or 0))


def compatible(client: Contract, server: Contract) -> bool:
    """Whether a server can serve a client: same plane and major, client minor <= server minor."""
    return (
        client.plane == server.plane
        and client.major == server.major
        and client.minor <= server.minor
    )
