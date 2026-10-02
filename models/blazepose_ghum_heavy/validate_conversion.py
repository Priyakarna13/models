#!/usr/bin/env python3
"""Validate the pinned tf2onnx conversion against the official TFLite model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

TFLITE_SHA256 = "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe"
ONNX_SHA256 = "efcc9e62707d596c86bfc962a26a673401d8438702216f82fc15ead54c6f3a83"
INPUT_NAME = "input_1"
INPUT_SHAPE = (1, 256, 256, 3)
OUTPUT_NAMES = ("Identity", "Identity_1", "Identity_4")
MAX_ABSOLUTE_ERROR = 0.005
MAX_MEAN_ABSOLUTE_ERROR = 0.0005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_input() -> np.ndarray:
    values = np.arange(np.prod(INPUT_SHAPE), dtype=np.float32).reshape(INPUT_SHAPE)
    return np.remainder(values, np.float32(251.0)) / np.float32(250.0)


def compare_outputs(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    if set(candidate) != set(OUTPUT_NAMES) or not set(OUTPUT_NAMES).issubset(reference):
        raise ValueError(
            f"conversion output names mismatch: reference={sorted(reference)}, candidate={sorted(candidate)}"
        )
    report: dict[str, Any] = {"outputs": {}}
    for name in OUTPUT_NAMES:
        expected = np.asarray(reference[name], dtype=np.float32)
        actual = np.asarray(candidate[name], dtype=np.float32)
        if actual.shape != expected.shape:
            raise ValueError(f"conversion output {name} shape mismatch: {actual.shape} != {expected.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"conversion output {name} contains non-finite values")
        delta = np.abs(actual - expected)
        maximum = float(delta.max())
        mean = float(delta.mean())
        if maximum > MAX_ABSOLUTE_ERROR or mean > MAX_MEAN_ABSOLUTE_ERROR:
            raise ValueError(f"conversion output {name} exceeds tolerance: max={maximum}, mean={mean}")
        report["outputs"][name] = {
            "maximum_absolute_error": maximum,
            "mean_absolute_error": mean,
            "shape": list(actual.shape),
        }
    report["passed"] = True
    return report


def validate_conversion(tflite_path: Path, onnx_path: Path) -> dict[str, Any]:
    tflite_path = tflite_path.expanduser().resolve()
    onnx_path = onnx_path.expanduser().resolve()
    if sha256_file(tflite_path) != TFLITE_SHA256:
        raise ValueError(f"BlazePose TFLite SHA-256 mismatch: {tflite_path}")
    if sha256_file(onnx_path) != ONNX_SHA256:
        raise ValueError(f"BlazePose ONNX SHA-256 mismatch: {onnx_path}")

    import onnxruntime as ort
    import tensorflow as tf

    sample = deterministic_input()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    if len(input_details) != 1 or tuple(input_details[0]["shape"]) != INPUT_SHAPE:
        raise ValueError(f"unexpected BlazePose TFLite input contract: {input_details}")
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    reference = {detail["name"]: interpreter.get_tensor(detail["index"]) for detail in interpreter.get_output_details()}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or session.get_inputs()[0].name != INPUT_NAME:
        raise ValueError("unexpected BlazePose ONNX input contract")
    values = session.run(list(OUTPUT_NAMES), {INPUT_NAME: sample})
    candidate = dict(zip(OUTPUT_NAMES, values, strict=True))
    report = compare_outputs(reference, candidate)
    report.update(
        {
            "onnx_sha256": ONNX_SHA256,
            "sample": "deterministic_modulo_251_rgb",
            "tflite_sha256": TFLITE_SHA256,
        }
    )
    return report
