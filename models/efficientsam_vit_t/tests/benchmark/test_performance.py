from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.array.copy() if copy else self.array


def test_mpk_arguments_name_exactly_both_components(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")

    mpks = script.parse_mpks(["image_encoder=/data/encoder_mpk.tar.gz", "prompt_decoder=/data/decoder_mpk.tar.gz"])
    assert list(mpks) == ["image_encoder", "prompt_decoder"]
    with pytest.raises(SystemExit, match="exactly image_encoder=<path> and prompt_decoder=<path>"):
        script.parse_mpks(["image_encoder=/data/encoder_mpk.tar.gz"])


def test_the_cpu_step_upsamples_the_mask_with_the_best_score(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")
    # Neat returns the masks with height, width and channel axes, one channel per candidate.
    masks = np.full((1, 256, 256, 3), -5.0, np.float32)
    masks[0, :128, :, 1] = 5.0  # the best mask covers the top half
    masks[0, :, :, 0] = 5.0  # a worse-scored mask covers everything

    mask = script.frame_mask([_Tensor(np.asarray([[[0.2, 0.9, 0.1]]], np.float32)), _Tensor(masks)])

    assert mask.shape == (script.FRAME_HEIGHT, script.FRAME_WIDTH)
    assert mask[: script.FRAME_HEIGHT // 2 - 4].all() and not mask[script.FRAME_HEIGHT // 2 + 4 :].any()
    with pytest.raises(ValueError):  # masks in any other layout are refused, not misread
        script.frame_mask([_Tensor(np.asarray([[[0.2, 0.9, 0.1]]], np.float32)), _Tensor(masks.transpose(0, 3, 1, 2))])


def test_the_benchmark_frame_is_fixed_and_its_object_fills_the_prompt_box(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")
    first, second = script.synthetic_frame(), script.synthetic_frame()

    assert (first.shape, first.dtype) == ((script.FRAME_HEIGHT, script.FRAME_WIDTH, 3), np.uint8)
    np.testing.assert_array_equal(first, second)
    x0, y0, x1, y1 = (int(value) for value in script.OBJECT_BOX)
    centre = first[(y0 + y1) // 2, (x0 + x1) // 2]
    assert abs(int(centre[0]) - 200) < 20 and abs(int(centre[2]) - 40) < 20


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert sorted(performance["components"]) == ["image_encoder", "prompt_decoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id
