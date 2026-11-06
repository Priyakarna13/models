"""FAN's CPU steps and the AFW faces, shared by compile.py and accuracy.py.

The crop and the decode follow ibug's FANPredictor (MIT), in NumPy and OpenCV rather than PyTorch.

Crop: the face box's centre and mean side, the side enlarged by 1 / 0.55, a square around the centre, zero-padded
where it leaves the image, resized to 256x256 RGB in [0, 1].
Decode: each heatmap's peak; a disc of radius 0.1 * sqrt(64 * 64) around it; the centroid of the clamped heatmap in
that disc, at pixel centres (index + 0.5); then mapped back from the 64x64 heatmap to the image through the crop.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

CROP_RATIO = 0.55
INPUT_SIZE = 256
HEATMAP_SIZE = 64
LANDMARKS = 68
RADIUS = 0.1


def crop_box(face_box: np.ndarray) -> np.ndarray:
    """[left, top, right, bottom] of the square crop around a [left, top, right, bottom] face box, in whole pixels."""
    left, top, right, bottom = (float(value) for value in face_box[:4])
    centre = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = ((bottom - top) + (right - left)) / 2.0 / CROP_RATIO
    start = np.round(centre - size / 2.0)
    return np.concatenate([start, np.round(start + size) + 1]).astype(int)


def crop(image_rgb: np.ndarray, face_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The model input [256, 256, 3] float32 in [0, 1] for one face, and its crop box in the image."""
    box = crop_box(face_box)
    left, top, right, bottom = box
    pad = [max(0, -top), max(0, bottom - image_rgb.shape[0]), max(0, -left), max(0, right - image_rgb.shape[1])]
    padded = np.pad(image_rgb, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0))) if any(pad) else image_rgb
    patch = padded[top + pad[0]: bottom + pad[0], left + pad[2]: right + pad[2]]
    resized = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))
    return resized.astype(np.float32) / np.float32(255), box


def decode(heatmaps: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[68, 64, 64] heatmaps of one face -> its [68, 2] landmarks in the image and [68] peak scores."""
    heatmaps = heatmaps.astype(np.float32).reshape(LANDMARKS, HEATMAP_SIZE, HEATMAP_SIZE)
    scores = heatmaps.max(axis=(1, 2))
    flat = heatmaps.reshape(LANDMARKS, -1).argmax(axis=1)
    peaks = np.stack([flat // HEATMAP_SIZE, flat % HEATMAP_SIZE], axis=1).astype(np.float32)
    rows, columns = np.meshgrid(np.arange(HEATMAP_SIZE, dtype=np.float32), np.arange(HEATMAP_SIZE, dtype=np.float32),
                                indexing="ij")
    distance = np.hypot(rows[None] - peaks[:, 0, None, None], columns[None] - peaks[:, 1, None, None])
    masked = np.where(distance <= RADIUS * HEATMAP_SIZE, heatmaps, 0).clip(min=0)
    total = np.maximum(masked.sum(axis=(1, 2)), np.finfo(np.float32).eps)
    centres = np.arange(0.5, HEATMAP_SIZE, dtype=np.float32)
    xs = (masked.sum(axis=1) * centres).sum(axis=1) / total
    ys = (masked.sum(axis=2) * centres).sum(axis=1) / total
    left, top, right, bottom = box
    landmarks = np.stack([xs * (right - left) / HEATMAP_SIZE + left, ys * (bottom - top) / HEATMAP_SIZE + top], axis=1)
    return landmarks, scores


# The Dataset Registry's afw-faces-default archive: cal_data/ holds face crops for calibration; test_data/ holds face
# crops, each with an iBUG .pts file of its 68 annotated points. Each crop's bounds are its face box.
CALIBRATION_SPLIT, TEST_SPLIT = "cal_data", "test_data"


def parse_pts(text: str) -> np.ndarray:
    """An iBUG .pts file's 68 (x, y) points."""
    points = np.array(text.split("{")[1].split("}")[0].split(), dtype=np.float64)
    if points.size != 2 * LANDMARKS:
        raise ValueError(f"expected {LANDMARKS} points, found {points.size / 2:g}")
    return points.reshape(LANDMARKS, 2)


def _files(dataset: Path, split: str) -> dict[str, bytes]:
    """A split's files by name, from an extracted directory or the archive; macOS metadata files are skipped."""
    def wanted(name: str) -> bool:
        parts = PurePosixPath(name).parts
        return len(parts) >= 2 and parts[-2] == split and not parts[-1].startswith(".")

    if dataset.is_dir():
        return {path.name: path.read_bytes() for path in sorted((dataset / split).glob("*"))
                if path.is_file() and not path.name.startswith(".")}
    found = {}
    with tarfile.open(dataset, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            if member.isfile() and wanted(name):
                stream = archive.extractfile(member)
                if stream is not None:
                    found[PurePosixPath(name).name] = stream.read()
    return dict(sorted(found.items()))


def decode_rgb(content: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def whole_image_box(image: np.ndarray) -> np.ndarray:
    """A face crop's own bounds as its face box."""
    return np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float64)


def test_faces(dataset: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Each annotated test face: its name, RGB image and 68 annotated points."""
    files = _files(dataset, TEST_SPLIT)
    faces = [(name, decode_rgb(content), parse_pts(files[name[:-4] + ".pts"].decode("utf-8")))
             for name, content in files.items() if name.lower().endswith(".jpg") and name[:-4] + ".pts" in files]
    if not faces:
        raise ValueError(f"{dataset} has no annotated faces under {TEST_SPLIT}/")
    return faces


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """The calibration faces, cropped as the model takes them: [count, 256, 256, 3] float32 RGB in [0, 1]."""
    images = [decode_rgb(content) for name, content in _files(dataset, CALIBRATION_SPLIT).items()
              if name.lower().endswith(".jpg")]
    if len(images) != count:
        raise ValueError(f"calibration requires {count} faces, but {dataset} has {len(images)}")
    return np.stack([crop(image, whole_image_box(image))[0] for image in images])


def nme(landmarks: np.ndarray, truth: np.ndarray) -> float:
    """Mean point error over the inter-ocular distance: the outer eye corners, points 36 and 45."""
    return float(np.linalg.norm(landmarks - truth, axis=1).mean() / np.linalg.norm(truth[36] - truth[45]))
