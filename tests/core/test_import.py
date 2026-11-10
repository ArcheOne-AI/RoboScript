import roboscript
import roboscript_vision


def test_packages_share_one_version() -> None:
    assert roboscript.__version__ == roboscript_vision.__version__
