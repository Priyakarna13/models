from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    """All-zero box encodings and class logits low enough that no anchor scores."""
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[len(accuracy.GRIDS) :]:
        logits[:] = -30.0
    return outputs


def test_anchors_follow_the_tensorflow_ssd_generator(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    anchors = accuracy.ANCHORS

    assert anchors.shape == (1917, 4)
    # The first anchor is the 0.1 square box centred on the first 19x19 cell.
    centre = 0.5 / 19
    np.testing.assert_allclose(anchors[0], [centre - 0.05, centre - 0.05, centre + 0.05, centre + 0.05], atol=1e-6)
    # The last anchor is the 1x1 cell's interpolated square box: sqrt(0.95 * 1.0) on each side.
    side = np.sqrt(0.95)
    np.testing.assert_allclose(anchors[-1], [0.5 - side / 2, 0.5 - side / 2, 0.5 + side / 2, 0.5 + side / 2], atol=1e-6)


def test_zero_encodings_decode_to_the_anchors_clipped_to_the_image(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    boxes = accuracy.decode_boxes(np.zeros((1917, 4), np.float32))

    np.testing.assert_allclose(boxes, accuracy.ANCHORS.clip(0, 1), atol=1e-6)


def test_decode_scales_boxes_to_the_image_and_keeps_label_map_ids(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # The 1x1 level's square box, scored for class index 18, which is COCO category 18 (dog).
    outputs[11][0, 0, 0, 5 * 91 + 18] = 4.0

    rows = accuracy.decode(outputs, 7, 640, 480)

    (row,) = rows
    top, left, bottom, right = accuracy.ANCHORS[-1].clip(0, 1) * [480, 640, 480, 640]
    np.testing.assert_allclose(row, [7, left, top, right - left, bottom - top, 1 / (1 + np.exp(-4.0)), 18], rtol=1e-5)


def _per_class_nms(accuracy: ModuleType, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
    """TensorFlow's form: greedy NMS within each class, then the best MAX_DETECTIONS over all classes."""
    found = []
    for label in range(scores.shape[1]):
        order = np.argsort(-scores[:, label], kind="stable")
        order = order[scores[order, label] > accuracy.SCORE_THRESHOLD]
        kept: list[int] = []
        for anchor in order:
            if kept and (accuracy._iou(boxes[anchor], boxes[kept]) > accuracy.IOU_THRESHOLD).any():
                continue
            kept.append(int(anchor))
        found += [(scores[anchor, label], anchor, label) for anchor in kept[:100]]
    found.sort(key=lambda item: -item[0])
    return [(anchor, label) for _, anchor, label in found[: accuracy.MAX_DETECTIONS]]


@pytest.mark.parametrize("seed", range(3))
def test_select_matches_per_class_nms(seed: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(seed)
    corners = rng.uniform(0, 0.8, (300, 2))
    boxes = np.hstack([corners, corners + rng.uniform(0.05, 0.3, (300, 2))]).astype(np.float32)
    scores = rng.uniform(0, 1, (300, 6)).astype(np.float32) ** 4

    anchors, labels = accuracy.select(boxes, scores)

    assert list(zip(anchors.tolist(), labels.tolist())) == _per_class_nms(accuracy, boxes, scores)


@pytest.mark.parametrize("output", [0, 6])
def test_decode_rejects_non_finite_outputs(output: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[output][0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, 640, 480)


def test_decode_rejects_another_output_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, 640, 480)


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.parse_mpks(["model=/data/ssd_mpk.tar.gz"]) == Path("/data/ssd_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        accuracy.parse_mpks(["backbone=/data/ssd_mpk.tar.gz"])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
