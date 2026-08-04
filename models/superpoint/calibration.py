#!/usr/bin/env python3
"""Reproduce and validate the TUM RGB-D calibration split for SuperPoint."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

import certifi
import numpy as np

DATASET_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk-rgb.avi"
DATASET_SHA256 = "3faa0fa667fa95d28ecf0d231c3383ddef8695f391f07d48f484af39de43a854"
DECODED_VIDEO_SHA256 = "8398d4611395efdbea5c9e7086aa86e5012abe58e17627a56b8d664c2f1e0437"
CALIBRATION_NPZ_SHA256 = "c55acf48a9f85f093eaf21f695bf9176c8c3d5555e85f09ce9171e7de0480142"
CALIBRATION_ARRAY_SHA256 = "3f9968092295f95e98d2429611f5b87216e7fbbf0fc29678272462228ca4ec40"
VALIDATION_ARRAY_SHA256 = "64c4ed7a99c1f47a8e09fe4a94674999a170b59b7aa5ce60e2413a01127eab41"

WIDTH = 640
HEIGHT = 480
FRAME_COUNT = 613
SPLIT_COUNT = 192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_indices() -> tuple[np.ndarray, np.ndarray]:
    selected = np.rint(np.linspace(0, FRAME_COUNT - 1, SPLIT_COUNT)).astype(np.int64)
    return selected[np.arange(SPLIT_COUNT) % 3 != 1], selected[np.arange(SPLIT_COUNT) % 3 == 1]


def bgr_to_gray(frame: np.ndarray) -> np.ndarray:
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
    if calibration.shape != (128, 1, 1, HEIGHT, WIDTH) or calibration.dtype != np.float32:
        raise ValueError(f"unexpected calibration tensor: {calibration.shape}/{calibration.dtype}")
    digest = hashlib.sha256(calibration.tobytes(order="C")).hexdigest()
    if digest != CALIBRATION_ARRAY_SHA256:
        raise ValueError(f"calibration array SHA-256 mismatch: {digest}")
    return calibration


def download_dataset(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / "rgbd_dataset_freiburg1_desk-rgb.avi"
    if not destination.is_file():
        context = ssl.create_default_context(cafile=certifi.where())
        request = Request(DATASET_URL, headers={"User-Agent": "sima-neat-model-registry/0.1"})
        with urlopen(request, timeout=120, context=context) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    digest = sha256_file(destination)
    if digest != DATASET_SHA256:
        raise ValueError(f"TUM source SHA-256 mismatch: {digest}")
    return destination


def transcode_dataset(source: Path, work_dir: Path, ffmpeg: str = "ffmpeg") -> Path:
    destination = work_dir / "freiburg1-desk-crf22-slow.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )
    return destination


def decode_split(video: Path, ffmpeg: str = "ffmpeg") -> tuple[np.ndarray, np.ndarray]:
    calibration_indices, validation_indices = split_indices()
    calibration_slots = {int(frame): slot for slot, frame in enumerate(calibration_indices)}
    validation_slots = {int(frame): slot for slot, frame in enumerate(validation_indices)}
    calibration = np.empty((128, 1, 1, HEIGHT, WIDTH), dtype=np.float32)
    validation = np.empty((64, 1, 1, HEIGHT, WIDTH), dtype=np.float32)

    process = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not create stdout")
    digest = hashlib.sha256()
    frame_size = HEIGHT * WIDTH * 3
    frame_index = 0
    while raw := process.stdout.read(frame_size):
        if len(raw) != frame_size:
            raise RuntimeError(f"short decoded frame {frame_index}: {len(raw)} bytes")
        digest.update(raw)
        target = calibration_slots.get(frame_index)
        validation_target = validation_slots.get(frame_index)
        if target is not None or validation_target is not None:
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            gray = bgr_to_gray(bgr).astype(np.float32) * np.float32(1.0 / 255.0)
            if target is not None:
                calibration[target, 0, 0] = gray
            else:
                validation[validation_target, 0, 0] = gray
        frame_index += 1

    if process.wait() != 0 or frame_index != FRAME_COUNT:
        raise RuntimeError(f"ffmpeg decoded {frame_index} frames; expected {FRAME_COUNT}")
    if digest.hexdigest() != DECODED_VIDEO_SHA256:
        raise RuntimeError("decoded TUM video does not match the pinned FFmpeg 6.1/libx264 stream")

    expected = ((calibration, CALIBRATION_ARRAY_SHA256), (validation, VALIDATION_ARRAY_SHA256))
    for array, expected_digest in expected:
        actual = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if actual != expected_digest:
            raise RuntimeError(f"SuperPoint calibration split SHA-256 mismatch: {actual}")
    return calibration, validation


def regenerate(work_dir: Path, ffmpeg: str = "ffmpeg") -> tuple[Path, Path]:
    source = download_dataset(work_dir)
    video = transcode_dataset(source, work_dir, ffmpeg)
    calibration, validation = decode_split(video, ffmpeg)
    calibration_path = work_dir / "superpoint_tum_freiburg1_desk_calibration_128.npz"
    validation_path = work_dir / "superpoint_tum_freiburg1_desk_validation_64.npz"
    np.savez_compressed(calibration_path, image=calibration)
    np.savez_compressed(validation_path, image=validation, frame_index=split_indices()[1])
    return calibration_path, validation_path
