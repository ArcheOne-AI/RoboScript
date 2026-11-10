"""roboscript.gateway.openai_client and gateway_url."""

from pydantic import SecretStr

from roboscript.gateway import openai_client

CONFIG = {
    "schema": "rs-gateway/v1",
    "endpoints": {"a": {"url_env": "A_URL", "key_env": "A_KEY", "protocols": ["responses"]}},
    "models": {"m": {"deployments": [{"endpoint": "a", "upstream": "m"}]}},
}


def test_the_client_carries_its_session_in_the_key_and_never_retries() -> None:
    client = openai_client("ep-123", base_url="http://127.0.0.1:1/v1", token=SecretStr("tok"))
    assert client.api_key == "tok:ep-123"
    assert client.max_retries == 0
    assert str(client.base_url) == "http://127.0.0.1:1/v1/"
    assert client.timeout == 900.0
