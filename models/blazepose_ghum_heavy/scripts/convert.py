"""Convert the official BlazePose GHUM Heavy TFLite to ONNX and check ONNX against TFLite."""

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from surgery import INPUT_NAME, INPUT_SHAPE, OUTPUTS, validate_source_model

MODEL_DIR = Path(__file__).resolve().parents[1]

TFLITE = "pose_landmark_heavy.tflite"
OPSET = 16
OUTPUT_NAMES = tuple(name for name, _ in OUTPUTS)
MAX_ABSOLUTE_ERROR = 0.005
MAX_MEAN_ABSOLUTE_ERROR = 0.0005


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", TFLITE):
        raise ValueError(f"Expected the TFLite file {TFLITE}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    validate_source_model(model)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def deterministic_input() -> np.ndarray:
    values = np.arange(np.prod(INPUT_SHAPE), dtype=np.float32).reshape(INPUT_SHAPE)
    return np.remainder(values, np.float32(251.0)) / np.float32(250.0)


def compare_outputs(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    if set(candidate) != set(OUTPUT_NAMES) or not set(OUTPUT_NAMES).issubset(reference):
        raise ValueError(
            f"conversion output names mismatch: reference={sorted(reference)}, candidate={sorted(candidate)}"
        )
    report: dict[str, Any] = {}
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
        report[name] = {"maximum_absolute_error": maximum, "mean_absolute_error": mean}
    return report


def check_against_tflite(tflite: Path, onnx_path: Path) -> dict[str, Any]:
    import onnxruntime as ort
    import tensorflow as tf

    sample = deterministic_input()
    interpreter = tf.lite.Interpreter(model_path=str(tflite))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    if len(input_details) != 1 or tuple(input_details[0]["shape"]) != INPUT_SHAPE:
        raise ValueError(f"unexpected BlazePose TFLite input contract: {input_details}")
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    reference = {detail["name"]: interpreter.get_tensor(detail["index"]) for detail in interpreter.get_output_details()}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    values = session.run(list(OUTPUT_NAMES), {INPUT_NAME: sample})
    return compare_outputs(reference, dict(zip(OUTPUT_NAMES, values, strict=True)))


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(source), "--output", str(output),
         "--opset", str(OPSET)],
        check=True,
    )
    validate_export(output)
    report = check_against_tflite(source, output)
    print(f"Deterministic-input TFLite/ONNX check passed: {report}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
