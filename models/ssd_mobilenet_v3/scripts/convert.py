#!/usr/bin/env python3
"""Export the complete TorchVision SSDLite320 forward pass from pinned COCO weights.

Preprocessing, heads, box decoding and NMS stay in this source graph. The separate
surgery owns the raw-head cut and BatchNorm folding required by Model Compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from model_registry.source import SourceSpec
from surgery import HEADS, INPUT_SHAPE, SOURCE_INPUT
from surgery import audit_source as audit

WEIGHTS_URL = "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
WEIGHTS_SIZE = 14069355
WEIGHTS_SOURCE = {"uri": WEIGHTS_URL, "sha256": WEIGHTS_SHA256, "size_bytes": WEIGHTS_SIZE}
# Optional, unapproved numerical experiment only; not a source-acceptance gate.
EXPECTED_VERSIONS = {
    "torch": "2.8.0+cpu",
    "torchvision": "0.23.0+cpu",
    "onnx": "1.22.0",
    "onnxruntime": "1.22.1",
    "numpy": "2.5.1",
}
EXPERIMENTAL_MAX_ABSOLUTE_ERROR = 5e-4
EXPERIMENTAL_MAX_NORMALIZED_RMSE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_weights(path: Path) -> Path:
    if sha256(path) != WEIGHTS_SHA256 or path.stat().st_size != WEIGHTS_SIZE:
        raise ValueError("official TorchVision COCO weight digest mismatch")
    return path


def load_detector(path: Path):
    import torch
    from torch import nn
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    verified_weights(path)
    # Both parameters must be None: the original COCO factory uses the reduced
    # tail architecture and no separately downloaded ImageNet backbone weights.
    detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
    detector.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if detector.transform.image_mean != [0.5] * 3 or detector.transform.image_std != [0.5] * 3:
        raise ValueError("TorchVision factory input normalization changed")
    if detector.anchor_generator.num_anchors_per_location() != [6] * 6:
        raise ValueError("TorchVision anchor contract changed")
    if sum(isinstance(module, nn.Conv2d) for module in detector.modules()) != 98:
        raise ValueError("TorchVision convolution contract changed")
    return detector


def flatten_heads(heads: list[np.ndarray], columns: int) -> np.ndarray:
    flattened = []
    for head in heads:
        batch, _channels, height, width = head.shape
        flattened.append(
            head.reshape(batch, 6, columns, height, width).transpose(0, 3, 4, 1, 2).reshape(batch, -1, columns)
        )
    return np.concatenate(flattened, axis=1)


def parity(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite output or changed output shape")
    error = actual.astype(np.float64) - reference.astype(np.float64)
    maximum = float(np.max(np.abs(error)))
    rms_reference = float(np.sqrt(np.mean(np.square(reference.astype(np.float64)))))
    nrmse = float(np.sqrt(np.mean(np.square(error)))) / max(rms_reference, 1e-12)
    denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    cosine_denominator = float(
        np.linalg.norm(reference.astype(np.float64).ravel()) * np.linalg.norm(actual.astype(np.float64).ravel())
    )
    cosine = (
        (
            float(
                np.dot(
                    reference.astype(np.float64).ravel(),
                    actual.astype(np.float64).ravel(),
                )
            )
            / cosine_denominator
        )
        if cosine_denominator
        else float(np.array_equal(reference, actual))
    )
    return {
        "max_abs": maximum,
        "normalized_rmse": nrmse,
        "max_relative_abs_floor_1e-6": float(np.max(np.abs(error) / denominator)),
        "cosine_similarity": cosine,
        "within_experimental_limits": maximum <= EXPERIMENTAL_MAX_ABSOLUTE_ERROR
        and nrmse <= EXPERIMENTAL_MAX_NORMALIZED_RMSE,
    }


def compare_heads(expected: dict, actual: list[np.ndarray]) -> dict:
    """Compare the kept tensors, without treating near-tied NMS order as a head regression."""
    if len(actual) != 12:
        raise ValueError("expected twelve raw-head probes")
    results = {}
    for key, columns, section in (("bbox_regression", 4, slice(0, 6)), ("cls_logits", 91, slice(6, 12))):
        result = parity(expected[key], flatten_heads(actual[section], columns))
        if result["max_abs"] > EXPERIMENTAL_MAX_ABSOLUTE_ERROR:
            raise ValueError(f"source head parity exceeds the existing absolute limit: {key}: {result}")
        results[key] = result
    return results


def expected_versions(machine: str) -> dict[str, str]:
    if machine not in {"x86_64", "aarch64"}:
        raise ValueError(f"unsupported conversion architecture: {machine}")
    return {**EXPECTED_VERSIONS, "torchvision": "0.23.0" if machine == "aarch64" else "0.23.0+cpu"}


def resolve_weights(path: Path | None) -> Path:
    """Use the canonical verified source cache; an explicit local file is verified too."""
    if path is None:
        model = load_and_validate_model(Path(__file__).resolve().parents[1])
        source = model["conversion"]["default"]["source"]
        expected = {"type": "file", "sha256": WEIGHTS_SHA256, "size_bytes": WEIGHTS_SIZE}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("model.yaml conversion source differs from the pinned TorchVision weights")
        spec = SourceSpec.from_mapping(source)
        if spec.uri != WEIGHTS_URL:
            # SourceSpec enforces immutable, full-hash artifact/HF paths. A
            # mirror changes transport only, never the original checkpoint.
            approved = spec.uri.startswith("hf://models/") or (
                urlsplit(spec.uri).hostname == "artifacts.stg.neat.sima.ai"
            )
            if not approved or model.get("upstream", {}).get("derived_from") != WEIGHTS_URL:
                raise ValueError("checkpoint mirror requires an approved immutable source and original provenance")
        path, _cache_hit = Fetcher().fetch(spec)
    return verified_weights(path)


def export(path: Path, weights: Path) -> dict:
    import onnxruntime as ort
    import torch

    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != expected_versions(platform.machine()):
        raise ValueError(f"export environment differs from pinned source recipe: {versions}")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    detector = load_detector(weights)

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = detector

        def forward(self, image):
            result = self.detector([image[0]])[0]
            return result["boxes"], result["scores"], result["labels"]

    source = Source().eval()
    example = torch.linspace(0, 1, int(np.prod(INPUT_SHAPE))).reshape(INPUT_SHAPE)
    with torch.inference_mode():
        torch.onnx.export(
            source,
            example,
            str(path),
            opset_version=17,
            dynamo=False,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.PRESERVE,
            input_names=[SOURCE_INPUT],
            output_names=["boxes", "scores", "labels"],
        )
    graph = onnx.load(path)
    # The number surviving score filtering/NMS varies with image content.
    for value in graph.graph.output:
        value.type.tensor_type.shape.dim[0].dim_param = "detections"
    onnx.helper.set_model_props(
        graph,
        {
            "sima.architecture": "torchvision.ssdlite320_mobilenet_v3_large",
            "sima.precision.source": "fp32_pretrained_not_qat",
            "sima.input.normalization": "rgb_float32_0_to_1; normalization inside graph",
            "sima.torchvision.weights_url": WEIGHTS_URL,
            "sima.torchvision.weights_sha256": WEIGHTS_SHA256,
            "sima.export.versions": json.dumps(versions, sort_keys=True),
        },
    )
    onnx.save(graph, path)
    artifact = audit(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # Probe the kept internal tensors in memory; never publish a cut conversion.
    probe = onnx.load(path)
    produced = {name for node in probe.graph.node for name in node.output}
    if not set(HEADS) <= produced:
        raise ValueError("full source is missing raw-head probes")
    probe.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in HEADS)
    session = ort.InferenceSession(probe.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    trials = []
    for name, image in (
        ("black_320", torch.zeros(INPUT_SHAPE)),
        ("white_320", torch.ones(INPUT_SHAPE)),
        ("ramp_320", example),
        ("seed0_320", torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(0))),
    ):
        with torch.inference_mode():
            reference = [value.numpy() for value in source(image)]
            transformed, _ = detector.transform([image[0]], None)
            heads = detector.head(list(detector.backbone(transformed.tensors).values()))
        actual = session.run(None, {SOURCE_INPUT: image.numpy()})
        trial = {
            "input": name,
            "outputs": {},
            "raw_heads": compare_heads({key: value.numpy() for key, value in heads.items()}, actual[3:]),
        }
        for key, expected, observed in zip(("boxes", "scores", "labels"), reference, actual[:3], strict=True):
            trial["outputs"][key] = parity(expected, observed)
        trials.append(trial)
    return {
        "schema_version": 1,
        "artifact": artifact,
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "versions": versions,
        "export_script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "parity": trials,
        "source_contract": "complete upstream FP32 forward; no cut, BN fold or QAT",
        "experimental_parity_limits": {
            "max_abs": EXPERIMENTAL_MAX_ABSOLUTE_ERROR,
            "normalized_rmse": EXPERIMENTAL_MAX_NORMALIZED_RMSE,
        },
        "qualification": {
            "structural_export": "pass",
            "numeric_comparison": "measured",
            "kept_heads_absolute_parity": "pass",
            "source_acceptance": "pending_coco",
            "optional_strict_parity_experiment": "pass"
            if all(item["within_experimental_limits"] for trial in trials for item in trial["outputs"].values())
            else "fail",
            "compiled": False,
            "coco_accuracy": False,
            "board_performance": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", choices=["default"])
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--strict-parity-experiment",
        action="store_true",
        help="Apply unapproved diagnostic limits, not the source-acceptance contract",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = resolve_weights(args.weights)
    path = args.output_dir / "model.onnx"
    report = export(path, verified_weights(weights))
    (args.output_dir / "export-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(path), **report["artifact"]}, indent=2))
    return (
        2
        if args.strict_parity_experiment and report["qualification"]["optional_strict_parity_experiment"] == "fail"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
