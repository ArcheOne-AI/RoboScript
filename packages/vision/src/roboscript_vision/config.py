"""Configuration of the vision service (`configs/vision.example.yaml`)."""

import os
from pathlib import Path
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, HttpUrl, PositiveInt

from roboscript.settings import EnvSettings

GROUNDING_DINO_BASE_COMMIT: Final = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
"""The hub commit of IDEA-Research/grounding-dino-base that the service loads by default."""


def _absolute(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"must be an absolute path, got {path}")
    return path


Absolute = Annotated[Path, AfterValidator(_absolute)]


class GroundingDinoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "IDEA-Research/grounding-dino-base"
    """A hub repository (`IDEA-Research/grounding-dino-base` or `-tiny`) or a directory holding
    its files."""
    revision: str | None = GROUNDING_DINO_BASE_COMMIT
    """The hub commit to load; another model needs its own (grounding-dino-tiny's is
    a2bb814dd30d776dcf7e30523b00659f4f141c71). None takes the latest; a directory ignores it."""


class Sam3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "facebook/sam3"
    """The hub repository (gated) or a directory holding its transformers-format files."""
    revision: str | None = None
    """The hub commit to load; None takes the latest; a directory ignores it."""


class VisionConfig(BaseModel):
    """The models of one vision service, on the first GPU that `CUDA_VISIBLE_DEVICES` names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blob_dir: Absolute | None = None
    """The blob store directory shared with the callers (images in, masks out); None takes
    `RS_BLOB_DIR`, which a stack sets for its services."""
    blob_roots: Absolute | None = None
    """A directory naming the callers' blob stores, a file per store (`BlobRoots` in
    `roboscript.blobs`): an image is read from the store that holds it and its masks written
    there. None takes `RS_BLOB_ROOTS`, which a shared stack sets (`roboscript.stack.shared`);
    unset, only `blob_dir`."""
    cache_dir: Absolute
    """`HF_HOME`: hub downloads go to `<cache_dir>/hub`."""
    hf_endpoint: HttpUrl | None = None
    """`HF_ENDPOINT`, a hub mirror; None keeps the default."""
    offline: bool = True
    """Load weights only from the cache (`HF_HUB_OFFLINE`); false lets missing files download
    through `hf_endpoint`."""
    cpu_threads: PositiveInt = 4
    """torch's CPU threads (`OMP_NUM_THREADS`), for image pre- and post-processing. torch's own
    default, one thread per core, contends with other processes on the same machine."""
    max_pending: PositiveInt = 8
    """Calls that may wait for or run on the GPU at once; more fail as `busy` (retryable)."""
    grounding_dino: GroundingDinoConfig | None = GroundingDinoConfig()
    """The detector behind `detect`; None serves no detection."""
    sam3: Sam3Config | None = Sam3Config()
    """The segmenter behind `segment`; None serves no segmentation."""
    warmup: bool = True
    """Run every model once on a blank image before `/readyz` turns 200, since a model's first
    call is slower than the rest."""

    def shared_blob_dir(self) -> Path:
        """`blob_dir`, else `RS_BLOB_DIR` (from the environment or `.env`)."""
        directory = self.blob_dir or EnvSettings().blob_dir
        if directory is None:
            raise ValueError("set blob_dir in the vision config, or RS_BLOB_DIR (a stack sets it)")
        return directory

    def caller_blob_roots(self) -> Path | None:
        """`blob_roots`, else `RS_BLOB_ROOTS` (from the environment or `.env`), else None."""
        return self.blob_roots or EnvSettings().blob_roots

    def apply_env(self) -> None:
        """Set what torch and the Hugging Face libraries read when they are imported: CPU
        threads, cache and mirror. Call it before they are."""
        os.environ["OMP_NUM_THREADS"] = str(self.cpu_threads)
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_HUB_OFFLINE"] = "1" if self.offline else "0"
        if self.hf_endpoint is not None:
            os.environ["HF_ENDPOINT"] = str(self.hf_endpoint).rstrip("/")
