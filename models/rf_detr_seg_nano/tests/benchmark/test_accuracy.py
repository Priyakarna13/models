from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    module = load_script("accuracy")
    # Uncompressed column-major RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "encode", _encode)
    return module


def _encode(binary: np.ndarray) -> list[dict]:
    rles = []
    for index in range(binary.shape[2]):
        flat = binary[:, :, index].ravel(order="F").astype(np.int64)
        changes = np.flatnonzero(np.diff(np.concatenate([[0], flat, [1 - flat[-1]]])))
        rles.append({"size": list(binary.shape[:2]), "counts": np.diff(np.concatenate([[0], changes])).tolist()})
    return rles


def _decoder_outputs(accuracy: ModuleType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dets = np.full((1, accuracy.QUERY_COUNT, 4), 0.5, np.float32)
    labels = np.full((1, accuracy.QUERY_COUNT, accuracy.CLASSES), -20.0, np.float32)
    masks = np.full((1, accuracy.MASK_SIZE, accuracy.MASK_SIZE, accuracy.QUERY_COUNT), -1.0, np.float32)
    return dets, labels, masks


def test_selection_takes_the_best_pairs_and_skips_unused_classes(accuracy_script: ModuleType) -> None:
    _, labels, _ = _decoder_outputs(accuracy_script)
    labels[0, 3, 1] = 5.0
    labels[0, 7, 12] = 6.0  # COCO slot 12 is unused.
    labels[0, 9, 18] = 2.0
    valid = np.ones(accuracy_script.CLASSES, bool)
    valid[12] = False

    order, scores = accuracy_script.select(labels, valid)

    assert [(pair // accuracy_script.CLASSES, pair % accuracy_script.CLASSES) for pair in order] == [(3, 1), (9, 18)]
    np.testing.assert_allclose(scores, 1 / (1 + np.exp(-np.array([5.0, 2.0]))), rtol=1e-6)


def test_boxes_map_normalized_centres_to_clipped_image_pixels(accuracy_script: ModuleType) -> None:
    dets = np.array([[[0.5, 0.5, 0.2, 0.4], [0.05, 0.5, 0.2, 0.2]]], np.float32)

    boxes = accuracy_script.image_boxes(dets, np.array([0, 1]), 200, 100)

    np.testing.assert_allclose(boxes, [[80, 30, 40, 40], [0, 40, 30, 20]], atol=1e-4)


def test_masks_resize_to_the_image_and_threshold_at_zero(accuracy_script: ModuleType) -> None:
    logits = np.full((1, 4, 4), -1.0, np.float32)
    logits[0, :, 2:] = 1.0

    (rle,) = accuracy_script.encode_masks(logits, 8, 4)

    image = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).reshape(8, 4).T.astype(bool)
    assert image[:, :4].sum() == 0 and image[:, 5:].all()


def test_decode_returns_one_result_per_selected_pair(accuracy_script: ModuleType) -> None:
    dets, labels, masks = _decoder_outputs(accuracy_script)
    labels[0, 4, 1] = 3.0
    masks[0, 10:20, 10:20, 4] = 2.0

    (result,) = accuracy_script.decode(dets, labels, masks, 42, 64, 48, np.ones(accuracy_script.CLASSES, bool))

    assert (result["image_id"], result["category_id"]) == (42, 1)
    assert result["segmentation"]["size"] == [48, 64]
    with pytest.raises(ValueError, match="non-finite"):
        labels[0, 0, 0] = np.nan
        accuracy_script.decode(dets, labels, masks, 42, 64, 48, np.ones(accuracy_script.CLASSES, bool))


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
