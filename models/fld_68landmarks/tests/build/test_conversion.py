"""Checks of convert.py's source and export checks, and of the converted ONNX when the conversion stage supplies it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def convert(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_2dfan2_alt_weights(convert: ModuleType) -> None:
    uri = "hf://models/public-data/ibug-face-alignment@8aef0e928499d334364f4c98e33dc6c4ffc8111a/fan/2dfan2_alt.pth"
    convert.validate_source({"type": "file", "uri": uri})
    with pytest.raises(ValueError, match="must pin exactly fan/2dfan2_alt.pth"):
        convert.validate_source({"type": "file", "uri": uri.replace("2dfan2_alt", "2dfan4")})


def test_network_has_ibug_2dfan2_alt_layout(convert: ModuleType) -> None:
    pytest.importorskip("torch")
    network = convert.build_network()
    names = set(network.state_dict())
    # Two hourglass modules of depth 4, each followed by its heatmap head; only the first feeds the second.
    assert {"conv1.weight", "m0.b2_plus_1.conv1.weight", "m1.b1_4.bn1.running_mean", "l1.weight"} <= names
    assert {"bl0.weight", "al0.weight"} <= names and "bl1.weight" not in names
    assert round(sum(p.numel() for p in network.parameters()) / 1e6, 2) == 12.11


def test_converted_model_has_the_declared_interface(converted: Path, convert: ModuleType) -> None:
    convert.validate_export(converted)
