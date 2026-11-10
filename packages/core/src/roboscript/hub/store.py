"""What the Hub keeps per episode (the EpisodeStore)."""

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from roboscript.contracts.env import EnvAgent
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import Budgets, DoneReport, EpisodeEnd, SeedAlias
from roboscript.contracts.ids import EpisodeId, PhaseName, RoleName, ToolName
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.refs import ROLE
from roboscript.contracts.tools import ContentPart, EnvBinding, Tier
from roboscript.contracts.trace import TraceContext
from roboscript.hub.config import RenderConfig
from roboscript.tools import DEMO_ALIAS, LOG_ALIAS, FrameStore

FinishGuard = Callable[[str], Awaitable[str | None]]
"""Installed by the runner: called with a `finish` call's note, it returns a message to refuse
that finish (at most once per episode), or None."""

ActionNotes = Callable[[], Awaitable[list[ContentPart]]]
"""Installed by the runner: called after each physical action, it returns what the model is shown
next to the action's result (hinted S1: the keypoint progress)."""


class Aliases:
    """The episode's alias table: each (kind, id) a model is shown becomes `<kind>_<n>`;
    `new(kind, value, alias)`, if given, hears of each alias made (the Hub logs it)."""

    def __init__(self, new: Callable[[str, str, str], None] | None = None) -> None:
        self._ids: dict[str, tuple[str, str]] = {}  # alias -> (kind, id)
        self._aliases: dict[tuple[str, str], str] = {}
        self._counts: Counter[str] = Counter()
        self._new = new

    def alias(self, kind: str, value: str) -> str:
        if kind == ROLE:  # a role is no id: shown as it is
            return value
        key = (kind, value)
        if key not in self._aliases:
            self._counts[kind] += 1
            alias = f"{kind}_{self._counts[kind]}"
            self._aliases[key] = alias
            self._ids[alias] = key
            if self._new is not None:
                self._new(kind, value, alias)
        return self._aliases[key]

    def items(self, exclude: tuple[str, ...] = ()) -> list[SeedAlias]:
        """Every alias, in the order made, but those of the kinds in `exclude`."""
        return [
            SeedAlias(kind=kind, value=value, alias=alias)
            for alias, (kind, value) in self._ids.items()
            if kind not in exclude
        ]

    def seed(self, kind: str, value: str, alias: str) -> None:
        """Keep an earlier episode's `alias` for (kind, value): `<kind>_<n>`, after which new
        aliases of `kind` continue, or for frames a try-qualified `log:<try>:<turn>[.<n>]` or a
        demonstration's `demo:<n>`. ValueError if `alias` is none of these or is taken."""
        prefix, _, number = alias.rpartition("_")
        logged = kind == "frame" and any(
            form.fullmatch(alias) is not None for form in (LOG_ALIAS, DEMO_ALIAS)
        )
        if not logged and (prefix != kind or not number.isdigit()):
            raise ValueError(f"{alias!r} is not an alias like {kind}_1")
        if self._ids.get(alias, (kind, value)) != (kind, value):
            raise ValueError(f"{alias!r} is given twice")
        self._aliases[(kind, value)] = alias
        self._ids[alias] = (kind, value)
        if not logged:
            self._counts[kind] = max(self._counts[kind], int(number))

    def resolve(self, kind: str, alias: str) -> str:
        """The id behind `alias`; ValueError (a validation message) if it is not one of `kind`."""
        found = self._ids.get(alias)
        if found is None:
            raise ValueError(f"unknown alias {alias!r}" + self._tries_loaded(alias))
        if found[0] != kind:
            raise ValueError(f"expected an alias like {kind}_1, got {alias!r}")
        return found[1]

    def _tries_loaded(self, alias: str) -> str:
        """For an unknown `log:` frame alias: whose frames there are, so that the model does not
        guess at a try whose frames were never loaded."""
        if not alias.startswith("log:"):
            return ""
        turns: dict[str, list[int]] = {}
        for given in self._ids:
            if LOG_ALIAS.fullmatch(given) is not None:
                attempt, turn = given.split(":")[1:]
                turns.setdefault(attempt, []).append(int(turn.split(".")[0]))
        if not turns:
            return "; no earlier try's frames are loaded here"
        loaded = "; ".join(
            f"log:{attempt}:{min(seen)} to log:{attempt}:{max(seen)}"
            for attempt, seen in turns.items()
        )
        return f"; only these tries' frames are loaded: {loaded} (other tries have none here)"


@dataclass
class Tally:
    """Mutable counters behind `contracts.hub.Counters`."""

    turns: int = 0
    info_calls: int = 0
    other_calls: int = 0
    system_calls: int = 0
    rejected: int = 0
    errors: int = 0
    steps: int = 0


@dataclass
class RetryState:
    """What `PhaseRules.retry_guard` remembers: `epoch` counts the information calls and the
    actions without a hard failure (a change since a failure); `measured`, the epoch of the last
    perception call; per workflow stage (None: calls naming none), the failed calls' keys with
    the epoch each last failed at, the hard failures and the refusals so far."""

    epoch: int = 0
    measured: int = -1
    failed: dict[str | None, dict[str, int]] = field(default_factory=dict)
    failures: Counter[str | None] = field(default_factory=Counter)
    refused: Counter[str | None] = field(default_factory=Counter)


@dataclass
class FailureRun:
    """What `PhaseRules.give_up_failures` remembers: the model's physical calls in a row that
    failed hard with the same `key` (workflow stage, reason) since a gripper last took or let go
    of something (`holding`, per arm, from the receipts), and `gave_up`, set once they are that
    many (reset by the next take or release)."""

    key: tuple[str | None, str] | None = None
    count: int = 0
    holding: dict[str, bool] = field(default_factory=dict)
    gave_up: bool = False


@dataclass(eq=False)
class EpisodeState:
    """One episode. `tools` (the visible ones, in order) never changes, nor do `system_tools`
    (those only system calls reach); `action` is the call holding the action lock; `frames` is
    the episode's frame store, `render` its render limits; `actions`: the actions admitted so
    far; `done` is what the last call of the rules' `done_check` found (None: none yet), and
    `done_actions` the actions admitted before it (stale once more are)."""

    id: EpisodeId
    stage: PhaseName
    tier: Tier
    role: RoleName
    tools: tuple[ToolName, ...]
    system_tools: tuple[ToolName, ...]
    budgets: Budgets
    rules: PhaseRules
    started: float  # on the Hub's clock
    trace: TraceContext
    aliases: Aliases
    frames: FrameStore
    render: RenderConfig
    env: EnvBinding | None = None
    env_client: EnvAgent | None = None
    embodiment: str | None = None
    structured: bool = False
    counters: Tally = field(default_factory=Tally)
    tool_calls: Counter[str] = field(default_factory=Counter)  # admitted model calls per tool
    ended: EpisodeEnd | None = None
    action: object | None = None
    finish_guard: FinishGuard | None = None
    finish_refusals: int = 0  # how often the finish guard refused
    finish_guard_times: int = 1  # how often it may
    actions: int = 0
    done: DoneReport | None = None
    done_actions: int = 0
    done_refused: int = 0  # finishes the done check refused (`PhaseRules.done_refusals`)
    action_notes: ActionNotes | None = None
    unmoved: int = 0  # rejections that moved nothing and went on (`PhaseRules.continue_unmoved`)
    unmoved_call: str | None = None  # the last of them, until an action runs after it
    unplanned: set[tuple[str | None, str]] = field(default_factory=set)
    """(stage, tool) of calls rejected unreachable or plan_failed before anything moved: their
    `PhaseRules.freed_arguments` are free from then on."""
    unanswered_infra: str | None = None
    """What the last model call failed with for infrastructure reasons (`infra_error`), until a
    later model call other than `finish` is forwarded to its tool and answered without an error,
    a hard-failure receipt included, not a tool error nor a refusal (`Hub.unanswered_infra`);
    "later" by dispatch (`dispatched`), so that calls run in parallel give one result whatever
    order they end in: `unanswered_seq` is the dispatch number of the call that set or cleared
    it."""
    unanswered_seq: int = 0
    dispatched: int = 0  # model calls dispatched so far (`Call.seq`)
    retry: RetryState = field(default_factory=RetryState)  # `PhaseRules.retry_guard`
    failures: FailureRun = field(default_factory=FailureRun)  # `PhaseRules.give_up_failures`
    toolset_state: dict[str, dict[str, Any]] = field(default_factory=dict)


class EpisodeStore:
    """Open episodes by id, and the ids of closed ones (whose calls still get the ended error)."""

    def __init__(self) -> None:
        self._open: dict[EpisodeId, EpisodeState] = {}
        self._closed: set[EpisodeId] = set()

    def add(self, episode: EpisodeState) -> None:
        self._open[episode.id] = episode

    def find(self, episode_id: EpisodeId) -> EpisodeState | None:
        return self._open.get(episode_id)

    def get(self, episode_id: EpisodeId) -> EpisodeState:
        episode = self._open.get(episode_id)
        if episode is None:
            closed = episode_id in self._closed
            raise ProtocolError(
                "not_found", f"episode {episode_id} is {'closed' if closed else 'unknown'}"
            )
        return episode

    def is_closed(self, episode_id: EpisodeId) -> bool:
        return episode_id in self._closed

    def close(self, episode_id: EpisodeId) -> EpisodeState:
        episode = self.get(episode_id)
        del self._open[episode_id]
        self._closed.add(episode_id)
        return episode
