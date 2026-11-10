"""perception.config: where the table is and the anchor camera have no defaults."""

from pathlib import Path

import pytest

from roboscript.perception.config import PerceptionConfig

CONFIGS = Path(__file__).parents[3] / "configs"


def test_where_the_table_is_and_the_anchor_camera_have_no_defaults() -> None:
    config = PerceptionConfig()
    assert config.anchor is None and config.table.xy_box is None and config.table.z_range is None
    with pytest.raises(ValueError, match=r"perception\.anchor is unset"):
        _ = config.anchor_camera
    with pytest.raises(ValueError, match=r"perception\.table\.z_range is unset"):
        _ = config.table.heights
