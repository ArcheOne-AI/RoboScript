"""roboscript.kernel.entrypoints, with fake distributions installed on sys.path."""

import sys
from pathlib import Path

import pytest

from roboscript.kernel import KernelError, entrypoints

GROUP = "roboscript.test_only"


def install_fake_distribution(root: Path, name: str, entries: dict[str, str]) -> None:
    info = root / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    lines = "".join(f"{key} = {value}\n" for key, value in entries.items())
    (info / "entry_points.txt").write_text(f"[{GROUP}]\n{lines}", encoding="utf-8")


def test_names_import_nothing_and_load_imports_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    module = "rs_fake_plugin_module"
    (tmp_path / f"{module}.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    install_fake_distribution(tmp_path, "roboscript-fake", {"thing": f"{module}:Thing"})
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: sys.modules.pop(module, None))

    assert entrypoints.names(GROUP) == ["thing"]
    assert module not in sys.modules
    thing = entrypoints.load(GROUP, "thing")
    assert thing.__name__ == "Thing" and thing.__module__ == module
    with pytest.raises(LookupError, match=r"nothing named 'missing'.*\['thing'\]"):
        entrypoints.load(GROUP, "missing")


def test_a_name_registered_twice_names_both_distributions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_distribution(tmp_path, "roboscript-one", {"dup": "one:Plugin"})
    install_fake_distribution(tmp_path, "roboscript-two", {"dup": "two:Plugin"})
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(KernelError, match="'dup'") as error:
        entrypoints.names(GROUP)
    assert "roboscript-one" in str(error.value) and "roboscript-two" in str(error.value)


def test_load_ref_imports_a_module_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    module = "rs_fake_ref_module"
    (tmp_path / f"{module}.py").write_text(
        "class Benchmark:\n    TASKS = ('a', 'b')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: sys.modules.pop(module, None))
    assert entrypoints.load_ref(f"{module}:Benchmark").__name__ == "Benchmark"
    assert entrypoints.load_ref(f"{module}:Benchmark.TASKS") == ("a", "b")
    with pytest.raises(AttributeError, match="MISSING"):
        entrypoints.load_ref(f"{module}:MISSING")
    with pytest.raises(ModuleNotFoundError):
        entrypoints.load_ref("rs_no_such_module:Thing")
