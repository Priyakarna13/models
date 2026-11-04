from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


def test_compile_reads_the_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    parameters = recipe._variant(recipe._metadata(), "w16a16")["parameters"]

    assert parameters == {"precision": "w16a16", "input_name": "Input3", "input_shape": [1, 1, 28, 28]}


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="exactly one build profile named 'w8a8'"):
        recipe._variant(recipe._metadata(), "w8a8")


@pytest.mark.parametrize(("field", "value", "message"), [
    ("precision", "w8a8", "only the declared Modalix BF16 variant"),
    ("input_shape", [1, 1, 32, 32], "unsupported MNIST input shape"),
])
def test_compile_rejects_an_undeclared_precision_or_shape_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    field: str,
    value: object,
    message: str,
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["components"][0]["parameters"][field] = value
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match=message):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile["id"])
    assert not (tmp_path / "model").exists()


def test_compile_main_rejects_an_invocation_named_otherwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(sys, "argv", [
        "compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"),
        "--build-profile", "w16a16", "--compile-invocation", "model",
    ])

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err
