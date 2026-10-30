from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_a_box_prompt_is_its_two_corners_then_padding(accuracy_script: ModuleType) -> None:
    prompt = accuracy_script.box_prompt([100.0, 50.0, 200.0, 100.0], (480, 640))

    # 640 wide scales to 1024: corners (100, 50) and (300, 150), shifted to pixel centres and normalized.
    coords = prompt["point_coords"][0, :, 0, :2].T
    np.testing.assert_allclose(coords, (np.array([[100, 50], [300, 150]]) * 1.6 + 0.5) / 1024 * 2 - 1, rtol=1e-6)
    types = prompt["point_types"][0, :, 0, :].argmax(axis=0)
    assert types.tolist() == [3, 4, 0, 0, 0]


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_mask_layout_reads_both_layouts(accuracy_script: ModuleType, layout: str) -> None:
    masks = np.random.default_rng(0).normal(size=(4, 256, 256)).astype(np.float32)
    board = masks[None] if layout == "nchw" else masks.transpose(1, 2, 0)[None]

    read, scores = accuracy_script.mask_layout(board, np.arange(4, dtype=np.float32)[None])

    np.testing.assert_array_equal(read, masks)
    assert scores.tolist() == [0, 1, 2, 3]


def test_image_mask_drops_the_padding_and_fits_the_image(accuracy_script: ModuleType) -> None:
    # A 1024x512 resized image fills the top half of the padded input; its mask covers that half.
    logits = np.full((256, 256), -5.0, np.float32)
    logits[:128] = 5.0

    mask = accuracy_script.image_mask(logits, (300, 600))

    assert mask.shape == (300, 600)
    assert mask.all()


def test_a_result_scores_the_first_mask_by_its_predicted_iou(
    accuracy_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    masks = np.full((4, 256, 256), -5.0, np.float32)
    masks[0, 64:128, 64:128] = 5.0
    encoded = []
    monkeypatch.setattr(
        accuracy_script.mask_util, "encode", lambda array: encoded.append(array) or [{"counts": b"x"}]
    )

    item = accuracy_script.result(
        {"image_id": 7, "category_id": 3}, masks, np.array([0.9, 0.1, 0.2, 0.3], np.float32), (1024, 1024)
    )

    (array,) = encoded
    assert array.shape == (1024, 1024, 1) and array.sum() == 256 * 256
    assert item["segmentation"] == {"counts": "x"}
    assert (item["image_id"], item["category_id"], item["score"]) == (7, 3, pytest.approx(0.9))


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
