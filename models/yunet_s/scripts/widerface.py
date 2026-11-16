"""WIDER FACE validation AP on the easy, medium and hard subsets, scored as the official evaluation scores it.

Ported from OpenCV Zoo's tools/eval/datasets/widerface.py (opencv/opencv_zoo at
48b10fa505aeb51fbd127fc60a5c660040e823b7, Apache-2.0), a Python port of the official MATLAB evaluation. It keeps that
code's conventions, so OpenCV Zoo's published YuNet scores reproduce:
- every score is rescaled to [0, 1] by the minimum and maximum over the whole set;
- an image without detections contributes one placeholder detection, [10, 10, 20, 20] with score 0.002;
- in each image, every detection in descending score order takes the ground-truth face it overlaps most; at IoU 0.5 or
  more it recalls that face, or is ignored when the face is outside the subset scored;
- precision and recall are sampled at 1000 score thresholds, and AP is the area under the VOC precision envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

SETTINGS = ("easy", "medium", "hard")
IOU_THRESHOLD = 0.5
THRESHOLDS = 1000
PLACEHOLDER = np.array([[10.0, 10.0, 20.0, 20.0, 0.002]])


def load_ground_truth(directory: Path) -> list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]]:
    """Each validation image as (event, image name, face boxes [x, y, w, h], 1-based indices each subset scores)."""
    from scipy.io import loadmat

    faces = loadmat(str(directory / "wider_face_val.mat"))
    subsets = {setting: loadmat(str(directory / f"wider_{setting}_val.mat"))["gt_list"] for setting in SETTINGS}
    images = []
    for index, event in enumerate(faces["event_list"]):
        boxes = faces["face_bbx_list"][index][0]
        for position, item in enumerate(faces["file_list"][index][0]):
            keep = {setting: subsets[setting][index][0][position][0].astype(int).ravel() for setting in SETTINGS}
            images.append((str(event[0][0]), str(item[0][0]), boxes[position][0].astype(float), keep))
    return images


def _overlaps(boxes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """IoU of [x1, y1, x2, y2] boxes against faces, counting pixels inclusively as the official evaluation does."""
    top_left = np.maximum(boxes[:, np.newaxis, 0:2], faces[:, 0:2])
    bottom_right = np.minimum(boxes[:, np.newaxis, 2:4], faces[:, 2:4])
    inner = np.prod(bottom_right - top_left + 1, axis=2) * (top_left < bottom_right).all(axis=2)
    box_area = np.prod(boxes[:, 2:4] - boxes[:, 0:2] + 1, axis=1)
    face_area = np.prod(faces[:, 2:4] - faces[:, 0:2] + 1, axis=1)
    return inner / (box_area[:, np.newaxis] + face_area - inner)


def _match(detections: np.ndarray, faces: np.ndarray, ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per detection, the faces recalled so far, and whether it is a counted proposal (1) or ignored (-1)."""
    boxes, targets = detections[:, :4].copy(), faces.copy()
    boxes[:, 2:4] += boxes[:, 0:2]
    targets[:, 2:4] += targets[:, 0:2]
    overlaps = _overlaps(boxes, targets)
    recalled_so_far = np.zeros(len(detections))
    recalled = np.zeros(len(faces))
    proposals = np.ones(len(detections))
    for row in range(len(detections)):
        best = overlaps[row].argmax()
        if overlaps[row, best] >= IOU_THRESHOLD:
            if ignore[best] == 0:
                recalled[best] = -1
                proposals[row] = -1
            elif recalled[best] == 0:
                recalled[best] = 1
        recalled_so_far[row] = np.count_nonzero(recalled == 1)
    return recalled_so_far, proposals


def _image_curve(detections: np.ndarray, proposals: np.ndarray, recalled_so_far: np.ndarray) -> np.ndarray:
    """Per score threshold: the counted proposals and the recalled faces among detections at or above it."""
    curve = np.zeros((THRESHOLDS, 2))
    for step in range(THRESHOLDS):
        above = np.where(detections[:, 4] >= 1 - (step + 1) / THRESHOLDS)[0]
        if len(above):
            last = above[-1]
            curve[step] = np.count_nonzero(proposals[: last + 1] == 1), recalled_so_far[last]
    return curve


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def evaluate(
    detections: Mapping[tuple[str, str], np.ndarray],
    ground_truth: list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]],
) -> dict[str, float]:
    """AP per subset for detections keyed by (event, image name), each an array of [x, y, w, h, score] rows."""
    rows = {}
    for event, name, _, _ in ground_truth:
        found = np.asarray(detections.get((event, name), np.zeros((0, 5))), dtype=float).reshape(-1, 5)
        found = found if len(found) else PLACEHOLDER.copy()
        rows[event, name] = found[np.argsort(-found[:, 4], kind="stable")]
    scores = np.concatenate([found[:, 4] for found in rows.values()])
    low, spread = scores.min(), scores.max() - scores.min()
    for found in rows.values():
        found[:, 4] = (found[:, 4] - low) / spread
    aps = {}
    for setting in SETTINGS:
        curve, faces_scored = np.zeros((THRESHOLDS, 2)), 0
        for event, name, faces, keep in ground_truth:
            faces_scored += len(keep[setting])
            found = rows[event, name]
            if not len(faces):
                continue
            ignore = np.zeros(len(faces))
            if len(keep[setting]):
                ignore[keep[setting] - 1] = 1
            recalled_so_far, proposals = _match(found, faces, ignore)
            curve += _image_curve(found, proposals, recalled_so_far)
        aps[setting] = _voc_ap(curve[:, 1] / faces_scored, curve[:, 1] / curve[:, 0])
    return aps
