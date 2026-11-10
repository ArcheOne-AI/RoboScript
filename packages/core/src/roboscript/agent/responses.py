"""`ResponsesBrain`, the reference brain (docs/concepts/agent-runtime.md): the own loop on the
OpenAI Responses API through the gateway, with `store=false` and the encrypted reasoning replayed
within a turn, strict function tools exported from the episode's visible tools, parallel calls only
when configured (a role without physical actions, whose edits come several at a time), and the
episode id as the prompt-cache key. It runs only calls the model proposed, by construction.
"""

from functools import partial
from typing import Any, Final, Literal

import openai
from pydantic import PositiveInt

from roboscript.agent.loop import OVERFLOW, LoopConfig, ModelError, Reply, run_loop
from roboscript.agent.session import AgentSession, BrainResult
from roboscript.contracts.trace import get_trace

_TRANSIENT: Final = frozenset({408, 409, 429})
"""4xx statuses that are the gateway's or the upstream's condition, not the request's fault."""
_POLICY: Final = frozenset({"invalid_prompt", "content_filter", "content_policy_violation"})
"""Upstream error codes caused by what the episode sent rather than by the configuration."""
_OVERFLOW: Final = (
    "context_length_exceeded",
    "context window",
    "prompt is too long",
    "maximum context length",
)
"""How upstreams say a prompt is too long: the code, or (from those that send none, as an
`invalid_request_error`) the message."""
_RETRY: Final = frozenset({502, 503, 504})
"""Gateway statuses of infrastructure failures a call may be sent again after."""
_UPSTREAM: Final = (
    "stream_error",
    "service_unavailable",
    "server_is_overloaded",
    "server_error",
    "internal_error",
    "overloaded",
    "rate_limit_exceeded",
    "gateway_concurrency_limit",
)
"""Codes of a failed upstream response that are the upstream's fault (the gateway's own
classification): a 502 `upstream_response_failed` naming one may be sent again; one naming none
is the request's."""


class ResponsesConfig(LoopConfig):
    """`model`: a gateway model or alias. `tool_choice`: `required` makes every reply a tool
    call, `auto` lets the model answer in text, which the loop then nudges. `reasoning_summary`:
    ask for readable summaries of the reasoning (None: none). `prompt_cache`: send the episode id
    as `prompt_cache_key`. `max_output_tokens`: None leaves the limit to the API.
    `parallel_tool_calls`: let a reply hold several calls, run in order (for roles without
    physical actions, e.g. one that makes several Memory edits in one reply)."""

    model: str = "brain-default"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "low"
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = "auto"
    tool_choice: Literal["auto", "required"] = "required"
    prompt_cache: bool = True
    max_output_tokens: PositiveInt | None = None
    parallel_tool_calls: bool = False


class ResponsesBrain:
    """The own loop with a Responses model; `AgentSession.models` supplies the gateway client."""

    Config = ResponsesConfig

    def __init__(self, config: ResponsesConfig | None = None) -> None:
        self.config = ResponsesConfig() if config is None else config

    async def run(self, session: AgentSession) -> BrainResult:
        if session.models is None:
            raise ValueError("ResponsesBrain needs AgentSession.models, a gateway client factory")
        options = self.options(session.episode_id)
        async with session.models(session.episode_id) as client:
            return await run_loop(session, partial(respond, client), options, self.config)

    def options(self, episode_id: str) -> dict[str, Any]:
        """The fixed fields of every request of an episode."""
        config = self.config
        reasoning: dict[str, Any] = {"effort": config.reasoning_effort}
        if config.reasoning_summary is not None:
            reasoning["summary"] = config.reasoning_summary
        options: dict[str, Any] = {
            "model": config.model,
            "reasoning": reasoning,
            "tool_choice": config.tool_choice,
            "parallel_tool_calls": config.parallel_tool_calls,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if config.prompt_cache:
            options["prompt_cache_key"] = episode_id
        if config.max_output_tokens is not None:
            options["max_output_tokens"] = config.max_output_tokens
        return options


async def respond(client: openai.AsyncOpenAI, request: dict[str, Any]) -> Reply:
    """One Responses call. A failure is a ModelError: `context_overflow` when the prompt was too
    long for the model (the loop shrinks it and sends it again once), `model_error:<config|
    policy>:<code>` when the request was otherwise refused (a 4xx; the class says whose fault it
    is), else `gateway_error` (no
    deployment answered, a timeout or a broken connection: infrastructure), to be sent again
    (`retry`) when it is a 502, 503 or 504 of the upstream's making or a connection error."""
    trace = get_trace()
    headers = {} if trace is None else {"traceparent": trace.to_traceparent()}
    try:
        raw = await client.responses.with_raw_response.create(**request, extra_headers=headers)
    except openai.APIStatusError as error:
        status, detail = error.status_code, f"HTTP {error.status_code}: {error.message}"
        if 400 <= status < 500 and status not in _TRANSIENT:
            code = error.code or error.type or str(status)
            if code == _OVERFLOW[0] or any(words in error.message.lower() for words in _OVERFLOW):
                raise ModelError(OVERFLOW, detail) from None
            kind = "policy" if code in _POLICY else "config"
            raise ModelError(f"model_error:{kind}:{code}", detail) from None
        raise ModelError("gateway_error", detail, retry=_retries(error)) from None
    except openai.APIConnectionError as error:  # also a timeout
        detail = f"{type(error).__name__}: {error.message}"
        raise ModelError("gateway_error", detail, retry=True) from None
    except openai.APIError as error:
        raise ModelError("gateway_error", f"{type(error).__name__}: {error.message}") from None
    response = raw.parse()
    dump = response.model_dump(mode="json", exclude_none=True)
    meta = {
        "id": response.id,
        "status": response.status,
        "deployment": raw.headers.get("x-rs-deployment"),
        "affinity_break": raw.headers.get("x-rs-affinity-break") == "1",
    }
    if queued := float(raw.headers.get("x-rs-queued-s") or 0):
        meta["queued_s"] = queued
    return Reply(output=dump.get("output", []), usage=dump.get("usage"), meta=meta)


def _retries(error: openai.APIStatusError) -> bool:
    """Whether a gateway error is infrastructure that the same request may get past: a 502,
    503 or 504, but not a failed response of the request's own nor a Hub's permit refusal."""
    if error.status_code not in _RETRY:
        return False
    if error.code == "upstream_response_failed":
        return any(code in error.message for code in _UPSTREAM)
    return error.code != "permit_failed"
