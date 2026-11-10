"""Episode bookkeeping that every env server shares: ids and agent-plane tokens (`Episodes`),
`execute` batches that are idempotent and refused while another one runs (`Batches`), and a
benchmark's reset options read from a spec (`reset_options`)."""

import asyncio
import hmac
import secrets
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import EpisodeOpened, EpisodeSpec
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError
from roboscript.contracts.ids import BatchId, EpisodeId, new_id


class Episodes:
    """One open episode at a time, each with its own agent-plane token.

    Opening an episode closes the previous one. Tokens of closed episodes stay known, so their
    holders get `EpisodeEnded` rather than `unauthorized`. An env ends the open episode before it
    resets its scene and checks the episode again where the scene is used, so a call that raced a
    reset fails instead of acting on the new scene.
    """

    def __init__(self) -> None:
        self.current: EpisodeId | None = None
        self._tokens: dict[EpisodeId, SecretStr] = {}

    def open(self) -> tuple[EpisodeId, SecretStr]:
        episode_id = EpisodeId(new_id("ep"))
        self._tokens[episode_id] = SecretStr(secrets.token_urlsafe(32))
        self.current = episode_id
        return episode_id, self._tokens[episode_id]

    def close(self, episode_id: EpisodeId) -> None:
        self.check(episode_id)
        self.end()

    def end(self) -> None:
        """Close the open episode, if any (before resetting the scene for another)."""
        self.current = None

    def check(self, episode_id: EpisodeId) -> None:
        """Raise unless `episode_id` is the open episode."""
        if episode_id == self.current:
            return
        if episode_id in self._tokens:
            raise EpisodeEnded(f"episode {episode_id} is closed")
        raise ProtocolError("not_found", f"no episode {episode_id}")

    def allow(self, bearer: SecretStr | None, method: str, args: dict[str, Any]) -> bool:
        """The agent plane's `rpc` authorization: the bearer is the token of the called episode."""
        token = self._tokens.get(args["episode_id"])
        return (
            bearer is not None
            and token is not None
            and hmac.compare_digest(
                bearer.get_secret_value().encode(), token.get_secret_value().encode()
            )
        )


OptionsT = TypeVar("OptionsT", bound=BaseModel)


def reset_options(declared: type[OptionsT], spec: EpisodeSpec) -> OptionsT:
    """`spec.options` as the benchmark declares its reset options (`declared`, a model whose
    fields default to the benchmark's own values); an option it does not declare, or a value it
    refuses, is `ProtocolError("invalid_request", ...)`."""
    try:
        return declared.model_validate(spec.options)
    except ValidationError as error:
        detail = f"reset options {spec.options!r}: {error.errors(include_url=False)}"
        raise ProtocolError("invalid_request", detail) from None


def unsolvable(embodiment: str, prerun_attempts: int) -> EpisodeOpened:
    """What `open` answers for a seed that does not count (the expert pre-run failed each time):
    no episode is open, so the id is unknown to every plane and there is no URL or token."""
    return EpisodeOpened(
        episode_id=EpisodeId(new_id("ep")),
        agent_url="",
        agent_token=SecretStr(""),
        instruction=None,
        step_limit=None,
        embodiment=embodiment,
        cameras=[],
        solvable=False,
        prerun_attempts=prerun_attempts,
    )


class Batches:
    """`execute` batches of one episode: a batch id seen before gets its first outcome again (for
    the same steps), its result or its error, and a call while a batch runs is `busy`.

    A batch runs to its end even when the caller goes away, so a retry never executes its steps
    a second time.
    """

    def __init__(self) -> None:
        self._done: dict[BatchId, tuple[ActionBatch, asyncio.Future[ExecuteResult]]] = {}
        self._running: asyncio.Future[ExecuteResult] | None = None

    async def run(
        self, batch: ActionBatch, execute: Callable[[], Awaitable[ExecuteResult]]
    ) -> ExecuteResult:
        if batch.batch_id in self._done:
            steps, outcome = self._done[batch.batch_id]
            if steps != batch:
                raise ProtocolError("invalid_request", f"batch {batch.batch_id} ran other steps")
            return outcome.result()  # or raises its error again
        if self._running is not None:
            raise InfraError("busy", "another batch is executing")
        running = asyncio.ensure_future(execute())
        self._running = running

        def finish(done: asyncio.Future[ExecuteResult]) -> None:
            self._running = None
            if not done.cancelled():
                done.exception()  # retrieved here, so asyncio never reports it as unhandled
                self._done[batch.batch_id] = (batch, done)

        running.add_done_callback(finish)
        return await asyncio.shield(running)
