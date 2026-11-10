"""The repository root, the local `.env` (paths and tokens; never `.env.api`), and secret files
(a token handed to one run's processes, readable by its owner only)."""

import functools
import os
import re
from pathlib import Path

from pydantic import Field, SecretStr, create_model, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from roboscript.contracts.provenance import SOURCE_ENV, SOURCE_FILE

_WORKSPACE_TABLE = re.compile(r"^\s*\[tool\.uv\.workspace\]\s*$", re.MULTILINE)


def source_sha() -> str | None:
    """The code's revision: `RS_SOURCE_SHA`, else the one recorded under the
    repository root (`.roboscript/source.sha`); None when neither names one (no repository
    root found counts as none: an installed package outside a checkout)."""
    if sha := os.environ.get(SOURCE_ENV):
        return sha
    try:
        root = repo_root()
    except FileNotFoundError:
        return None
    path = root / SOURCE_FILE
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def require_source_sha() -> str:
    """The code's revision (`source_sha`), which every result a run writes names;
    ValueError when nothing names it, so that no run starts whose results could not say which
    code made them. The one check both a launcher and its jobs make."""
    if (sha := source_sha()) is None:
        raise ValueError(
            f"no code revision: set {SOURCE_ENV}, or run from a checkout that has a {SOURCE_FILE} "
            "file"
        )
    return sha


def same_source(a: str, b: str) -> bool:
    """Whether two revision names name one code version: equal, or one a prefix (of at least 7
    characters) of the other where neither is of a dirty tree (`-dirty`); a name may hold a full
    SHA or a short one."""
    if a == b:
        return True
    if a.endswith("-dirty") or b.endswith("-dirty"):
        return False
    return min(len(a), len(b)) >= 7 and (a.startswith(b) or b.startswith(a))


def repo_root() -> Path:
    """`RS_ROOT`, else the nearest directory at or above the working directory whose
    `pyproject.toml` has a `[tool.uv.workspace]` table."""
    if named := os.environ.get("RS_ROOT"):
        root = Path(named)
        if not root.is_dir():
            raise FileNotFoundError(f"RS_ROOT={named} is not a directory")
        return root.resolve()
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and _WORKSPACE_TABLE.search(pyproject.read_text(encoding="utf-8")):
            return directory
    raise FileNotFoundError(
        f"no pyproject.toml with [tool.uv.workspace] at or above {cwd}: "
        "run inside the repository or set RS_ROOT"
    )


def runs_path(path: Path) -> Path:
    """`path` with its leading `runs/` placed under `RS_RUNS_ROOT` when that is set, else
    relative to the working directory; absolute paths are kept. Records keep the
    relative spelling, so a run directory moves with `rsync` and the setting alone."""
    if path.is_absolute() or path.parts[:1] != ("runs",):
        return path
    root = EnvSettings().runs_root
    return path if root is None else root.joinpath(*path.parts[1:])


def _env_file() -> Path:
    named = os.environ.get("RS_ENV_FILE")
    if not named:
        return repo_root() / ".env"
    path = Path(named)
    if path.name == ".env.api":
        raise ValueError(f"RS_ENV_FILE={named}: only the model gateway reads .env.api")
    if not path.is_file():
        raise FileNotFoundError(f"RS_ENV_FILE={named} does not exist")
    return path


class EnvSettings(BaseSettings):
    """Local paths and tokens: the process environment first, then the file named by
    `RS_ENV_FILE`, else `.env` at the repository root (which may be missing)."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    blob_dir: Path | None = Field(default=None, validation_alias="RS_BLOB_DIR")
    """The blob store directory a stack shares among its services (`roboscript.stack` sets it)."""
    blob_roots: Path | None = Field(default=None, validation_alias="RS_BLOB_ROOTS")
    """A directory naming more blob stores a service reads its callers' blobs from, a file per
    store (`roboscript.blobs.BlobRoots`): a shared stack's callers' (`roboscript.stack.shared`
    sets it)."""
    runs_root: Path | None = Field(default=None, validation_alias="RS_RUNS_ROOT")
    """Where a relative `runs/...` path of a run or output directory lives (`runs_path`): a
    faster disk while the data is hot, a larger volume after `rsync`ing it back."""
    gateway_url: str | None = Field(
        default=None, pattern=r"^https?://\S+$", validation_alias="RS_GATEWAY_URL"
    )
    """The model gateway's base URL (`.../v1`) where it is not this machine's
    `configs/gateway.yaml` `listen`: a machine that calls another's gateway
    (`roboscript.gateway.client.gateway_url`)."""
    gateway_token: SecretStr | None = Field(default=None, validation_alias="RS_GATEWAY_TOKEN")
    gateway_admin_token: SecretStr | None = Field(
        default=None, validation_alias="RS_GATEWAY_ADMIN_TOKEN"
    )
    hub_secret: SecretStr | None = Field(default=None, validation_alias="RS_HUB_SECRET")
    hub_service_key: SecretStr | None = Field(default=None, validation_alias="RS_HUB_SERVICE_KEY")
    env_token: SecretStr | None = Field(default=None, validation_alias="RS_ENV_TOKEN")
    host_token: SecretStr | None = Field(default=None, validation_alias="RS_HOST_TOKEN")

    @field_validator("blob_dir", "blob_roots", "runs_root")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        return _absolute(path)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        dotenv = DotEnvSettingsSource(settings_cls, env_file=_env_file(), env_file_encoding="utf-8")
        return init_settings, env_settings, dotenv


def _absolute(path: Path | None) -> Path | None:
    # Services run from different working directories (a simulator may need its own).
    if path is not None and not path.is_absolute():
        raise ValueError(f"must be an absolute path, got {path}")
    return path


class _PathSettings(EnvSettings):
    """One path variable, read as `EnvSettings` reads its own fields (`path_variable`)."""

    path: Path | None = None

    @field_validator("path")
    @classmethod
    def _absolute_path(cls, path: Path | None) -> Path | None:
        return _absolute(path)


@functools.cache
def _path_settings(name: str) -> type[_PathSettings]:
    return create_model(  # type: ignore[no-any-return,call-overload]
        "PathVariable",
        __base__=_PathSettings,
        path=(Path | None, Field(default=None, validation_alias=name)),
    )


_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def path_variable(name: str) -> Path | None:
    """The directory the variable `name` names, read as `EnvSettings` reads its own paths (the
    process environment, then the env file) and as absolute as they must be; None where it is
    unset. A benchmark package declares the variables it needs and reads them here;
    `expand_path` reads a template's."""
    if not _VARIABLE.fullmatch(name):
        raise ValueError(f"{name!r} is no environment variable name")
    return _path_settings(name)().path


def write_secret(path: Path, secret: SecretStr) -> None:
    """Write `secret` to a new file that only its owner can read (0600); an existing file is an
    error, since another run may still use it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(secret.get_secret_value())


def read_secret(path: Path) -> SecretStr:
    """The secret in `path`, a file that only its owner may read (`write_secret`)."""
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"{path} is open to other users (mode {mode:o}): chmod 600 it")
    return SecretStr(path.read_text(encoding="utf-8").strip())


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_path(template: str) -> Path:
    """`template` with each `${NAME}` replaced by the path that variable names (`path_variable`:
    the environment, then `.env`), e.g. an embodiment's asset directory under a root
    variable its benchmark package declares; an unset one is an error."""
    if not _PLACEHOLDER.search(template):
        return Path(template)

    def value(match: re.Match[str]) -> str:
        name = match[1]
        path = path_variable(name)
        if path is None:
            raise ValueError(f"{template}: set {name} in the environment or in .env")
        return str(path)

    return Path(_PLACEHOLDER.sub(value, template))
