"""What an env shows the agent at one step, and frames as an episode keeps them."""

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame
from roboscript.contracts.ids import FrameId
from roboscript.contracts.robot import RobotState


class Observation(BaseModel):
    """Whitelisted camera frames, the robot state and the episode's instruction; frames are
    immutable and addressed by `frame_id`."""

    model_config = ConfigDict(frozen=True)

    frame_id: FrameId
    step: NonNegativeInt
    cameras: dict[str, CameraFrame]
    robot: RobotState
    instruction: str | None = None


class FrameEntry(BaseModel):
    """A frame in an episode's frame store: its id, the alias models know it by (`frame_3`), the
    step it was taken at and the `Observation`, stored as a JSON blob."""

    model_config = ConfigDict(frozen=True)

    frame_id: FrameId
    alias: str
    step: NonNegativeInt
    observation: BlobRef
