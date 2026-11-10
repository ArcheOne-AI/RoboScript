"""The vision service's config, its example files, and imports without torch."""

import os
import subprocess
import sys
from pathlib import Path

import yaml

import roboscript
import roboscript_vision
from roboscript.settings import load_config_file
from roboscript.stack import StackConfig
from roboscript_vision.backends import GroundingDino, Sam3, from_config
from roboscript_vision.config import GROUNDING_DINO_BASE_COMMIT, VisionConfig

CONFIGS = Path(__file__).parents[2] / "configs"


def test_the_example_is_valid() -> None:
    config = load_config_file(CONFIGS / "vision.example.yaml", VisionConfig)
    assert config.grounding_dino is not None and config.sam3 is not None
    assert config.grounding_dino.model == "IDEA-Research/grounding-dino-base"
    assert config.grounding_dino.revision == GROUNDING_DINO_BASE_COMMIT  # the pinned files
    assert config.blob_dir is None  # the stack's (RS_BLOB_DIR)
    detector, segmenter = from_config(config)
    assert isinstance(detector, GroundingDino) and detector.name == "grounding-dino-base"
    assert isinstance(segmenter, Sam3) and segmenter.name == "sam3"
    assert from_config(config.model_copy(update={"grounding_dino": None, "sam3": None})) == (
        None,
        None,
    )


def test_the_stack_example_runs_the_service_in_the_vision_env() -> None:
    data = yaml.safe_load((CONFIGS / "stacks" / "vision.example.yaml").read_text(encoding="utf-8"))
    stack = StackConfig.model_validate(data)
    (service,) = stack.services
    assert service.module == "roboscript_vision" and service.interpreter == "vision"
    assert stack.vision_pythonpath == [Path("packages/core/src"), Path("packages/vision/src")]
    assert stack.interpreters.sim is None  # no simulator service here
    assert stack.shared_blob_dir() == Path("runs/blobs").absolute()


def test_importing_the_service_imports_no_model_library() -> None:
    code = (
        "import sys, roboscript_vision.service, roboscript_vision.backends; "
        "loaded = {'torch', 'transformers', 'huggingface_hub'} & set(sys.modules); "
        "assert not loaded, loaded"
    )
    paths = [Path(roboscript.__file__).parents[1], Path(roboscript_vision.__file__).parents[1]]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
