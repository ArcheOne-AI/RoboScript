"""The Hub's configuration (`configs/hub.example.yaml`)."""

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from typing_extensions import Self

from roboscript.contracts.hub import RenderLimits
from roboscript.contracts.ids import RoleName

_ROLE_RENDER: dict[RoleName, RenderLimits] = {
    "demo_reader": RenderLimits(max_text_chars=16000),
    "reviser": RenderLimits(max_text_chars=16000),
}


class RenderConfig(BaseModel):
    """How results become model content: images are downscaled to `image_max_side` pixels on
    their longest side, at most `max_images` go out per result, and text, error text included,
    is cut after `max_text_chars` characters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_max_side: PositiveInt = 1024
    max_images: PositiveInt = 6
    max_text_chars: PositiveInt = 4000

    def merged(self, *limits: RenderLimits | None) -> Self:
        """These limits with the values each of `limits` sets, later ones winning."""
        update: dict[str, int] = {}
        for given in limits:
            if given is not None:
                update |= given.model_dump(exclude_none=True)
        return self.model_copy(update=update)


class HubConfig(BaseModel):
    """`render` is how results reach models and `roles` changes it by role (for roles that
    read long material); an episode may change it again (`OpenEpisode.render`).
    `env_timeout_s` bounds each request to an episode's env."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    render: RenderConfig = RenderConfig()
    roles: dict[RoleName, RenderLimits] = Field(default_factory=lambda: dict(_ROLE_RENDER))
    env_timeout_s: PositiveFloat = 300.0
