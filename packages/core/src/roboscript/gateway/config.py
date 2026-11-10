"""The gateway's inputs: `configs/gateway.yaml` (endpoints, models, routing knobs) and `.env.api`
(upstream URLs and keys), which no other module reads."""

import os
import re
from pathlib import Path
from typing import Final, Literal, NamedTuple

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.settings import load_config, load_config_file, repo_root

Wire = Literal["responses", "chat"]
Tier = Literal["primary", "fallback"]
TIERS: Final[tuple[Tier, ...]] = ("primary", "fallback")  # in the order they are tried

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_STATE_DIR: Final = "~/.local/state/roboscript"  # the user's local state (XDG), on a local disk
_VARIABLE = r"^[A-Za-z_][A-Za-z0-9_]*$"


class Endpoint(BaseModel):
    """An upstream, named by the `.env.api` variables holding its base URL (ending in `/v1`) and
    key. `stream_only`: an upstream that serves only streamed Responses calls with a list `input`;
    the gateway streams every Responses call anyway, and for these it also wraps a string `input`.
    `tool_output_images: false`: an upstream that mishandles images inside a tool's output (a
    Responses `*_call_output` whose `output` holds an `input_image`), e.g. bills them as base64
    text; requests carrying such images are routed to the other deployments. `pool`: endpoints
    on one upstream account, whose concurrency limit their deployments share (one adaptive cap,
    between their largest `min_inflight` and the sum of their `max_inflight`). Pool endpoints
    only when that limit counts our calls on all of them: a shared cap that one key's overload
    cuts throttles the other keys too."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url_env: str = Field(pattern=_VARIABLE)
    key_env: str = Field(pattern=_VARIABLE)
    protocols: list[Wire] = Field(min_length=1)
    stream_only: bool = False
    tool_output_images: bool = True
    pool: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(set(self.protocols)) < len(self.protocols):
            raise ValueError(f"protocols {self.protocols} name one twice")
        if self.stream_only and self.protocols != ["responses"]:
            raise ValueError(
                "a stream_only endpoint speaks only responses (chat is not aggregated)"
            )
        return self


class Deployment(BaseModel):
    """A logical model on one endpoint, under the name that endpoint knows it by. A `fallback`
    deployment serves only when no `primary` one of the model can (a degradation, logged).
    `max_inflight`: calls it takes at most at once; further calls go elsewhere or wait for a slot.
    The cap in force adapts between `min_inflight` and it: halved when at least
    `aimd_cut_failure_share` of its recent attempts failed (`aimd_cut`), raised by one after
    `aimd_increase_after` answers in a row and multiplied by `aimd_recover_factor` after each
    `aimd_recover_s` without a cut or an overload refusal. `min_inflight` keeps a deployment
    usable whose overload signals do not come from our own load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str
    upstream: str = Field(min_length=1)
    tier: Tier = "primary"
    max_inflight: PositiveInt = 8
    min_inflight: PositiveInt = 1

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_inflight > self.max_inflight:
            raise ValueError(
                f"{self.endpoint}: min_inflight {self.min_inflight} is above max_inflight "
                f"{self.max_inflight}"
            )
        return self


class ModelSpec(BaseModel):
    """`default_reasoning_effort`: sent for a caller that names none (Responses
    `reasoning.effort`, Chat `reasoning_effort`), so that no call leaves without one; a caller's
    own effort always goes as it is."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deployments: list[Deployment] = Field(min_length=1)
    default_reasoning_effort: str | None = Field(default=None, pattern=r"^[a-z]+$")


class GatewayConfig(BaseModel):
    """`configs/gateway.yaml`; relative paths are taken from the working directory, `~` is the
    user's home. `state_file` and `log_file` default to the user's local state directory: the
    gateway appends a long line per request, so the log must be on a fast local disk,
    not on a busy data volume or a network file system."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["rs-gateway/v1"] = Field(alias="schema")
    listen: str
    endpoints: dict[str, Endpoint] = Field(min_length=1)
    models: dict[str, ModelSpec] = Field(min_length=1)
    aliases: dict[str, str] = {}
    policy: str = "default"  # a name in the `roboscript.gateway_policies` entry-point group
    cooldown_s: PositiveFloat = 30
    retry_after_cap_s: PositiveFloat = 120
    first_byte_timeout_s: PositiveFloat = 180
    connect_timeout_s: PositiveFloat = 30
    # Hedging an aggregated Responses call (a non-streaming client): once its attempt has run
    # this quantile of the model's recent answered attempts (over stats_window_s, all of its
    # deployments; at least hedge_min_s), a second attempt starts on another deployment with a
    # free slot, unless calls queue for the model; the first answer wins, the other is cancelled.
    # None: no hedging.
    hedge_quantile: float | None = Field(default=None, gt=0, lt=1)
    hedge_min_s: PositiveFloat = 60
    queue_timeout_s: NonNegativeFloat = 300  # a call waits this long for a free slot, then 503
    pin_wait_s: NonNegativeFloat = 30  # a session waits this long for a slot on its own deployment
    stats_window_s: PositiveFloat = 300  # the recent window of /admin/stats percentiles
    max_pins: PositiveInt = 10_000  # sessions whose deployment is remembered (least recent go)
    aimd_increase_after: PositiveInt = 2  # answers in a row that raise an adaptive cap by one
    aimd_recover_s: PositiveFloat = 30  # each interval this long without an overload signal ...
    aimd_recover_factor: float = Field(default=2, ge=1)  # ... multiplies it by this (+1 at least)
    # failure_share: halve an adaptive cap once at least aimd_cut_failure_share of its attempts
    # in the last aimd_cut_window_s failed (any failure), with at least aimd_cut_min_calls there;
    # signal: halve on every overload signal, timeout or broken stream
    aimd_cut: Literal["failure_share", "signal"] = "failure_share"
    aimd_cut_failure_share: float = Field(default=0.334, gt=0, le=1)
    aimd_cut_min_calls: PositiveInt = 6
    aimd_cut_window_s: PositiveFloat = 60
    health_min_calls: PositiveInt = 10  # attempts in the window before the failure rate counts
    health_failure_rate: float = Field(default=0.5, gt=0, le=1)  # takes a deployment out
    health_cooldown_max_s: PositiveFloat = 600  # the longest time out (2, 4, 8 ... cooldown_s)
    permit_timeout_s: PositiveFloat = 10  # to post an answer's permits to a Hub and get its ack
    state_file: Path = Field(Path(f"{_STATE_DIR}/gateway-state.json"), validate_default=True)
    log_max_queued: PositiveInt = 10_000  # log lines waiting for the disk; more are dropped
    log_file: Path = Field(Path(f"{_STATE_DIR}/gateway.jsonl"), validate_default=True)

    @field_validator("state_file", "log_file")
    @classmethod
    def _home(cls, path: Path) -> Path:
        return path.expanduser()

    @field_validator("listen")
    @classmethod
    def _listen(cls, listen: str) -> str:
        _address(listen)
        return listen

    @model_validator(mode="after")
    def _check(self) -> Self:
        if bad := [
            n for n in (*self.endpoints, *self.models, *self.aliases) if not _NAME.fullmatch(n)
        ]:
            raise ValueError(f"names must be letters, digits, '.', '_' or '-': {bad}")
        for name, model in self.models.items():
            used = [deployment.endpoint for deployment in model.deployments]
            if unknown := sorted(set(used) - self.endpoints.keys()):
                raise ValueError(f"model {name!r} names unknown endpoints {unknown}")
            if len(set(used)) < len(used):
                raise ValueError(f"model {name!r} lists an endpoint twice")
        for alias, target in self.aliases.items():
            if alias in self.models:
                raise ValueError(f"alias {alias!r} is also a model")
            if target not in self.models:
                raise ValueError(f"alias {alias!r} names unknown model {target!r}")
        return self

    @property
    def address(self) -> tuple[str, int]:
        return _address(self.listen)

    def resolve(self, name: str) -> str | None:
        """The logical model a requested name stands for: itself, an alias target, or None."""
        return name if name in self.models else self.aliases.get(name)


def _address(listen: str) -> tuple[str, int]:
    host, _, port = listen.rpartition(":")
    host = host.removeprefix("[").removesuffix("]")
    if not host or not port.isdigit() or int(port) > 65535:
        raise ValueError(f"listen must look like 127.0.0.1:18790, got {listen!r}")
    return host, int(port)


def load(source: str | Path) -> GatewayConfig:
    """A config file (a `Path`, or a string ending in `.yaml` or `.yml`) or a name under
    `configs/` such as `gateway`."""
    if isinstance(source, Path) or source.endswith((".yaml", ".yml")):
        return load_config_file(Path(source), GatewayConfig)
    return load_config(source, GatewayConfig)


class Upstream(NamedTuple):
    url: str  # base URL without a trailing slash, e.g. https://api.example.com/v1
    key: SecretStr


def env_api_file() -> Path:
    """`RS_ENV_API_FILE` if set, else `.env.api` at the repository root."""
    named = os.environ.get("RS_ENV_API_FILE")
    return Path(named) if named else repo_root() / ".env.api"


def read_upstreams(config: GatewayConfig, path: Path) -> dict[str, Upstream]:
    """Each endpoint's URL and key from `path`. Errors name variables, never values."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: copy .env.api.example and fill it in")
    values = {
        name: value for name, value in dotenv_values(path, interpolate=False).items() if value
    }
    needed = sorted({v for e in config.endpoints.values() for v in (e.url_env, e.key_env)})
    if missing := [name for name in needed if name not in values]:
        raise ValueError(f"{path}: set {', '.join(missing)}")
    upstreams = {}
    for name, endpoint in config.endpoints.items():
        url = values[endpoint.url_env].rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{path}: {endpoint.url_env} must be an http(s) URL")
        upstreams[name] = Upstream(url, SecretStr(values[endpoint.key_env]))
    return upstreams
