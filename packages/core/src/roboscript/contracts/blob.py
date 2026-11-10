"""References to bulk data in the content-addressed blob store
(docs/architecture.md)."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.ids import BlobId


class BlobRef(BaseModel):
    """Bulk data by reference: the sha256 of the stored bytes and how to read them."""

    model_config = ConfigDict(frozen=True)

    id: Annotated[BlobId, Field(pattern=r"^[0-9a-f]{64}$")]
    size: Annotated[int, Field(ge=0)]
    media_type: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
