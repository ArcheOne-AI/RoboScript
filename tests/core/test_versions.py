import pytest

from roboscript.contracts.versions import HEADER, VERSIONS, Contract, compatible, parse


def test_known_planes_parse() -> None:
    assert HEADER == "RS-Contract"
    for plane, version in VERSIONS.items():
        contract = parse(f"{plane}/{version}")
        assert contract.plane == plane and contract.major >= 1
    assert parse("event/1") == Contract("event", 1, 0)
    assert parse("env.agent/1.12") == Contract("env.agent", 1, 12)
    assert str(parse("hub.control/2.3")) == "hub.control/2.3"


@pytest.mark.parametrize(
    "text", ["env.agent", "env.agent/", "env.agent/1.x", "Env/1.0", "/1.0", ""]
)
def test_malformed(text: str) -> None:
    with pytest.raises(ValueError, match="not a contract version"):
        parse(text)


@pytest.mark.parametrize(
    ("client", "server", "ok"),
    [
        ("env.agent/1.0", "env.agent/1.0", True),
        ("env.agent/1.0", "env.agent/1.3", True),  # an older client on a newer server
        ("env.agent/1.4", "env.agent/1.3", False),  # needs what the server lacks
        ("env.agent/2.0", "env.agent/1.9", False),
        ("env.agent/1.0", "env.agent/2.0", False),
        ("env.control/1.0", "env.agent/1.0", False),
    ],
)
def test_compatible(client: str, server: str, ok: bool) -> None:
    assert compatible(parse(client), parse(server)) is ok
