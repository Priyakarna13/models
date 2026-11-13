from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    (frame,) = component["execution_plan"]["edges"][0]["tensors"]
    assert frame["shape"] == [1, 720, 1280, 3]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"


def test_application_decode_keeps_confident_instances_with_input_size_masks(
    load_script: Callable[[str], ModuleType],
) -> None:
    performance_script = load_script("performance")
    shapes = (
        [(1, size, size, 80) for size in performance_script.LEVELS]
        + [(1, size, size, 4) for size in performance_script.LEVELS]
        + [(1, size, size, performance_script.KERNEL_PARAMETERS) for size in performance_script.LEVELS]
        + [(1, 80, 80, 8)]
    )
    outputs = [np.zeros(shape, np.float32) for shape in shapes]
    for scores in outputs[:3]:
        scores[:] = -20.0
    outputs[0][0, 10, 10, 2] = 3.0
    outputs[0][0, 10, 11, 2] = 2.0  # same class, overlapping: suppressed
    outputs[2][0, 5, 5, 7] = 1.0
    outputs[3][0, 10, 10] = outputs[3][0, 10, 11] = [20, 20, 20, 20]
    outputs[5][0, 5, 5] = [10, 10, 10, 10]
    outputs[6][0, 10, 10, -1] = outputs[8][0, 5, 5, -1] = 1.0

    boxes, scores, labels, masks = performance_script.decode(outputs)

    assert labels.tolist() == [2, 7]
    np.testing.assert_allclose(scores, 1 / (1 + np.exp(-np.array([3.0, 1.0]))), rtol=1e-6)
    assert masks.shape == (2, 640, 640) and masks.all()
