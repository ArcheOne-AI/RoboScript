"""The model gateway (docs/concepts/agent-runtime.md): the only exit for LLM calls.

One Starlette + httpx process (`python -m roboscript.gateway`) in front of OpenAI-compatible
upstreams: sessions stick to a deployment, failures before the first byte fail over, and every
request is logged. Callers use `openai_client(session_id)`; the server lives in `app`.
"""

from roboscript.gateway.client import gateway_url, openai_client

__all__ = ["gateway_url", "openai_client"]
