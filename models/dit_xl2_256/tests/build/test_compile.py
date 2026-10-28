"""Checks of compile.py's build parameters, and of the compiled build when the compile stage supplies it."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    parameters = compile_module._component_parameters("w16a16")
    assert list(parameters) == ["transformer_0", "transformer_1", "transformer_2", "transformer_3", "vae_decoder"]
    # Batch 1: each classifier-free-guidance slot is its own pass.
    assert all(item == {"precision": "w16a16"} for item in parameters.values())
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._component_parameters("w8a8")


def test_compiled_build_holds_every_component(compiled: Path) -> None:
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [item["component_id"] for item in outputs["outputs"]] == [
        "transformer_0", "transformer_1", "transformer_2", "transformer_3", "vae_decoder"]
    for item in outputs["outputs"]:
        assert (compiled / item["mpk_path"]).is_file()
