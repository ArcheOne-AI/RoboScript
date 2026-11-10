"""Opening artifact stores by name (docs/concepts/agent-runtime.md): the
`roboscript.artifact_stores` extension point. The store protocol, refs and stacks are contracts
(`contracts.artifact`); the built-in store is Task Memory's (`memory.artifact.MemoryArtifacts`,
registered as `task_memory`)."""

from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from roboscript.contracts.artifact import Artifact, ArtifactRef, ArtifactStore
from roboscript.kernel import entrypoints

GROUP: Final = "roboscript.artifact_stores"


class ArtifactStoreSpec(BaseModel):
    """A store registered in `roboscript.artifact_stores` by `name`, and its options (the store's
    `Config`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    options: dict[str, Any] = {}


def open_artifact_store(name: str, options: Mapping[str, Any] | None = None) -> ArtifactStore:
    """The store registered as `name` in `roboscript.artifact_stores`, its options validated by
    its `Config` model (ValueError naming what is wrong)."""
    factory = entrypoints.load(GROUP, name)
    config = factory.Config.model_validate(dict(options or {}))
    store: ArtifactStore = factory(**dict(config))
    return store


async def load_frozen_ref(store: ArtifactStore, ref: ArtifactRef) -> Artifact:
    """The version `ref` names, which must still be its name's frozen version (an evaluation's
    refs name frozen versions; one frozen again since is a stale manifest)."""
    frozen = await store.frozen(ref.name)
    if frozen != ref:
        found = "nothing" if frozen is None else frozen.version
        raise ValueError(
            f"{ref.kind} {ref.name} {ref.version} is not frozen: the store has {found} frozen; "
            "build the manifest again"
        )
    return await store.load(ref)
