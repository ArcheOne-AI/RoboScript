"""`stack.yaml`: the services of a stack and the interpreters that run them."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class Interpreters(BaseModel):
    """Python executables: the agent venv (uv, 3.11) and, for stacks with such services, the
    simulator env (conda, 3.10) and the vision env (a venv with torch; see
    packages/vision/README.md)."""

    model_config = ConfigDict(extra="forbid")

    agent: Path
    sim: Path | None = None
    vision: Path | None = None


class ServiceSpec(BaseModel):
    """One service process: `<python> -m <module> <args>` or `<python> <argv> <args>`.

    The supervisor appends `--ready-file <path>`; the service writes it once listening (see
    `roboscript.rpc.serve`). A path `interpreter` is used as is, without the sim or vision
    environment.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    module: str | None = None
    argv: list[str] | None = None
    interpreter: Literal["agent", "sim", "vision"] | Path = "agent"
    args: list[str] = []
    env: dict[str, str] = {}
    gpu: int | None = Field(default=None, ge=0)  # sets CUDA_VISIBLE_DEVICES
    depends_on: list[str] = []
    ready_timeout_s: float = Field(default=120, gt=0)
    stop_grace_s: float = Field(default=10, ge=0)  # SIGTERM to SIGKILL

    @model_validator(mode="after")
    def _one_command(self) -> Self:
        if (self.module is None) == (self.argv is None):
            raise ValueError(f"service {self.name!r}: set exactly one of module and argv")
        return self


class StackConfig(BaseModel):
    """Paths are used as given, relative ones against the working directory."""

    model_config = ConfigDict(extra="forbid")

    interpreters: Interpreters
    sim_pythonpath: list[Path] = []  # PYTHONPATH of sim services: the packages' src dirs
    vision_pythonpath: list[Path] = []  # PYTHONPATH of vision services: core and vision src dirs
    run_dir: Path
    blob_dir: Path | None = None
    """The blob store directory the stack's services share (default `<run_dir>/blobs`); every
    service gets it as `RS_BLOB_DIR`, its default blob directory."""
    services: list[ServiceSpec]

    @model_validator(mode="after")
    def _check_services(self) -> Self:
        names = [s.name for s in self.services]
        if duplicates := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"duplicate service names: {duplicates}")
        for service in self.services:
            if unknown := sorted(set(service.depends_on) - set(names)):
                raise ValueError(f"service {service.name!r} depends on unknown {unknown}")
            named = service.interpreter
            if named in ("sim", "vision") and getattr(self.interpreters, str(named)) is None:
                raise ValueError(f"service {service.name!r} needs interpreters.{named}")
        self.start_order()
        return self

    def shared_blob_dir(self) -> Path:
        return (self.blob_dir or self.run_dir / "blobs").absolute()

    def start_order(self) -> list[ServiceSpec]:
        """Services in dependency order, config order among independent ones."""
        order: list[ServiceSpec] = []
        started: set[str] = set()
        pending = list(self.services)
        while pending:
            batch = [s for s in pending if started.issuperset(s.depends_on)]
            if not batch:
                raise ValueError(f"dependency cycle among services {[s.name for s in pending]}")
            order += batch
            started.update(s.name for s in batch)
            pending = [s for s in pending if s.name not in started]
        return order
