from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["text_encoder", "image_model"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_image_model_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    edges = performance["components"]["image_model"]["execution_plan"]["edges"]
    assert [1, 720, 1280, 3] in [tensor["shape"] for edge in edges for tensor in edge["tensors"]]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id


def test_the_cpu_step_packs_the_prompt_as_calibration_does(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script, calibration = load_script("performance"), load_script("calibration")
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: array)
    features = np.random.default_rng(0).random((1, 16, 256), dtype=np.float32)
    prompt = calibration.tokens(script.PROMPT)

    np.testing.assert_array_equal(script.prompt_tensor(), calibration.one_hot(prompt))
    packed = script.pack_text(SimpleNamespace(to_numpy=lambda copy: features))
    np.testing.assert_array_equal(packed, calibration.packed(features, prompt))
