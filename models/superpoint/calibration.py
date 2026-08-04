#!/usr/bin/env python3
"""Validate the immutable 128-image SuperPoint calibration snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_NPZ_SHA256 = "7361a0089fa8391436f2cfc4f1febb69c248ca332e8d8a63957e25b4bb6e65f2"
CALIBRATION_ARRAY_SHA256 = "260d9b7b7af937d7ae2d2edb6774f4e4f60af9438dc01781444fb901f0670f58"
CALIBRATION_MANIFEST_SHA256 = "9c865cca0fcbeb7a82efd979aabfb4b8f8eeb852a1cc2822b57cd272d27372d8"
SOURCE_COUNTS = {"coco": 80, "hpatches": 32, "tum": 16}

WIDTH = 640
HEIGHT = 480
SAMPLE_COUNT = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bgr_to_gray(frame: np.ndarray) -> np.ndarray:
    """Apply the deployment BT.601 fixed-point grayscale conversion."""

    if frame.dtype != np.uint8 or frame.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"expected one {HEIGHT}x{WIDTH} uint8 BGR frame, found {frame.shape}/{frame.dtype}")
    bgr = frame.astype(np.uint32)
    return (bgr[:, :, 2] * 4899 + bgr[:, :, 1] * 9617 + bgr[:, :, 0] * 1868 + 8192) >> 14


def validate_calibration(path: Path) -> np.ndarray:
    if sha256_file(path) != CALIBRATION_NPZ_SHA256:
        raise ValueError(f"calibration archive SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["image"]:
            raise ValueError(f"calibration archive must contain only 'image', found {archive.files}")
        calibration = np.asarray(archive["image"])
    expected_shape = (SAMPLE_COUNT, 1, 1, HEIGHT, WIDTH)
    if calibration.shape != expected_shape or calibration.dtype != np.float32:
        raise ValueError(f"unexpected calibration tensor: {calibration.shape}/{calibration.dtype}")
    if not np.isfinite(calibration).all() or calibration.min() < 0.0 or calibration.max() > 1.0:
        raise ValueError("calibration values must be finite FP32 values in [0, 1]")
    digest = hashlib.sha256(calibration.tobytes(order="C")).hexdigest()
    if digest != CALIBRATION_ARRAY_SHA256:
        raise ValueError(f"calibration array SHA-256 mismatch: {digest}")
    return calibration


def validate_manifest(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CALIBRATION_MANIFEST_SHA256:
        raise ValueError(f"calibration manifest SHA-256 mismatch: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("candidate_npz_sha256") != CALIBRATION_NPZ_SHA256:
        raise ValueError("calibration manifest does not identify the pinned NPZ")
    if manifest.get("candidate_sha256") != CALIBRATION_ARRAY_SHA256:
        raise ValueError("calibration manifest does not identify the pinned tensor")
    if manifest.get("sources") != SOURCE_COUNTS or len(manifest.get("records", [])) != SAMPLE_COUNT:
        raise ValueError("calibration manifest has an unexpected source composition")
    return manifest
