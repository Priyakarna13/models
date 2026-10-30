#!/usr/bin/env python3
"""Export the original TorchVision COCO weights, without QAT or hidden input affine.

This is a new FP32 source artifact. It is not the historical QAT source bundle.
The same FP32 source may later seed an explicitly labelled PTQ build; this exporter
neither quantizes nor claims detector accuracy or device performance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from model_registry.source import SourceSpec
from surgery import FEATURE_SIZES, INPUT_NAME, INPUT_SHAPE, OUTPUTS, audit

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


def build_raw(detector):
    """Construct raw heads without requiring Torch in host contract tests."""
    from torch import nn

    class RawSSDLite(nn.Module):
        """Only the original backbone and heads; input is already normalized."""

        def __init__(self, detector: nn.Module):
            super().__init__()
            self.backbone = detector.backbone
            self.regression_heads = detector.head.regression_head.module_list
            self.classification_heads = detector.head.classification_head.module_list

        def forward(self, images):
            features = list(self.backbone(images).values())
            locations = [head(feature) for head, feature in zip(self.regression_heads, features)]
            confidences = [head(feature) for head, feature in zip(self.classification_heads, features)]
            return tuple(locations + confidences)

    return RawSSDLite(detector)


def fold_batch_normalization(module) -> int:
    from torch import nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    module.eval()
    folded = sum(fold_batch_normalization(child) for child in module.children())
    if isinstance(module, nn.Sequential):
        names = list(module._modules)
        for first, second in pairwise(names):
            convolution, batch_norm = module._modules[first], module._modules[second]
            if isinstance(convolution, nn.Conv2d) and isinstance(batch_norm, nn.BatchNorm2d):
                module._modules[first] = fuse_conv_bn_eval(convolution, batch_norm)
                module._modules[second] = nn.Identity()
                folded += 1
    return folded


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


def expected_versions(machine: str) -> dict[str, str]:
    if machine not in {"x86_64", "aarch64"}:
        raise ValueError(f"unsupported conversion architecture: {machine}")
    return {**EXPECTED_VERSIONS, "torchvision": "0.23.0" if machine == "aarch64" else "0.23.0+cpu"}


def resolve_weights(path: Path | None) -> Path:
    """Use the canonical verified source cache; an explicit local file is verified too."""
    if path is None:
        source = load_and_validate_model(Path(__file__).resolve().parents[1])["conversion"]["default"]["source"]
        expected = {**WEIGHTS_SOURCE, "type": "file"}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("model.yaml conversion source differs from the pinned TorchVision weights")
        path, _cache_hit = Fetcher().fetch(SourceSpec.from_mapping(source))
    return verified_weights(path)


def export(path: Path, weights: Path) -> dict:
    import onnxruntime as ort
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(0)
    detector = load_detector(weights)
    fused = copy.deepcopy(detector)
    folds = fold_batch_normalization(fused)
    if folds != 70 or any(isinstance(module, nn.BatchNorm2d) for module in fused.modules()):
        raise ValueError("expected exactly 70 BatchNorm folds and none remaining")
    raw = build_raw(fused).eval()
    example = torch.linspace(-1, 1, int(np.prod(INPUT_SHAPE))).reshape(INPUT_SHAPE)
    with torch.inference_mode():
        torch.onnx.export(
            raw,
            example,
            str(path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[name for name, _shape in OUTPUTS],
            dynamo=False,
        )
    graph = onnx.load(path)
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != expected_versions(platform.machine()):
        raise ValueError(f"export environment differs from pinned source recipe: {versions}")
    properties = {
        "sima.architecture": "torchvision.ssdlite320_mobilenet_v3_large",
        "sima.precision.source": "fp32_pretrained_not_qat",
        "sima.input.normalization": "rgb_uint8/127.5-1; external to graph",
        "sima.output.order": "loc[0..5],confidence[0..5]; NCHW",
        "sima.torchvision.weights_url": WEIGHTS_URL,
        "sima.torchvision.weights_sha256": WEIGHTS_SHA256,
        "sima.export.versions": json.dumps(versions, sort_keys=True),
    }
    onnx.helper.set_model_props(graph, properties)
    onnx.save(graph, path)
    audit_result = audit(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    trials = []
    # The original detector's independent head flattening is the reference;
    # this catches permutations that a raw-wrapper-only comparison cannot.
    for name, rgb in (
        ("black_320", torch.zeros(3, 320, 320)),
        ("white_320", torch.ones(3, 320, 320)),
        ("ramp_720p", torch.linspace(0, 1, 3 * 720 * 1280).reshape(3, 720, 1280)),
        (
            "seed0_720p",
            torch.rand(3, 720, 1280, generator=torch.Generator().manual_seed(0)),
        ),
    ):
        with torch.inference_mode():
            images, _ = detector.transform([rgb], None)
            source = detector.head(list(detector.backbone(images.tensors).values()))
            raw_outputs = [value.numpy() for value in raw(images.tensors)]
        actual = session.run(None, {INPUT_NAME: images.tensors.numpy()})
        if [tuple(value.shape) for value in actual] != [shape for _name, shape in OUTPUTS]:
            raise ValueError("runtime public output shape mismatch")
        trial = {
            "input": name,
            "source_vs_raw": {},
            "source_vs_onnx": {},
            "raw_vs_onnx": [],
            "source_vs_raw_heads": [],
            "source_vs_onnx_heads": [],
        }
        for key, columns, section in (
            ("bbox_regression", 4, slice(0, 6)),
            ("cls_logits", 91, slice(6, 12)),
        ):
            reference = source[key].numpy()
            trial["source_vs_raw"][key] = parity(reference, flatten_heads(raw_outputs[section], columns))
            trial["source_vs_onnx"][key] = parity(reference, flatten_heads(actual[section], columns))
        for key, columns, start in (("bbox_regression", 4, 0), ("cls_logits", 91, 6)):
            offset = 0
            reference = source[key].numpy()
            for level, size in enumerate(FEATURE_SIZES):
                count = size * size * 6
                head_reference = reference[:, offset : offset + count]
                for label, values in (
                    ("source_vs_raw_heads", raw_outputs),
                    ("source_vs_onnx_heads", actual),
                ):
                    trial[label].append(
                        {
                            "head": OUTPUTS[start + level][0],
                            **parity(
                                head_reference,
                                flatten_heads([values[start + level]], columns),
                            ),
                        }
                    )
                offset += count
        trial["raw_vs_onnx"] = [parity(reference, result) for reference, result in zip(raw_outputs, actual)]
        trials.append(trial)
    return {
        "schema_version": 1,
        "artifact": audit_result,
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "versions": versions,
        "export_script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "normalization": {
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
            "source_range": [0, 1],
            "external_to_graph": True,
        },
        "resize": {
            "size": [320, 320],
            "interpolation": "bilinear",
            "align_corners": False,
            "antialias": False,
        },
        "input": {
            "name": INPUT_NAME,
            "shape": INPUT_SHAPE,
            "dtype": "float32",
            "layout": "NCHW",
        },
        "outputs": [{"name": name, "shape": shape, "dtype": "float32"} for name, shape in OUTPUTS],
        "anchors": 3234,
        "convolutions": 98,
        "batch_normalization_folds": folds,
        "experimental_parity_limits": {
            "max_abs": EXPERIMENTAL_MAX_ABSOLUTE_ERROR,
            "normalized_rmse": EXPERIMENTAL_MAX_NORMALIZED_RMSE,
        },
        "parity": trials,
        "quantization": "none; future w8a8 PTQ is a new baseline, not recovered QAT",
        "qualification": {
            "structural_export": "pass",
            "numeric_comparison": "measured",
            "source_acceptance": "pending_coco",
            "optional_strict_parity_experiment": "pass"
            if all(
                item["within_experimental_limits"]
                for trial in trials
                for item in [
                    *trial["source_vs_raw"].values(),
                    *trial["source_vs_onnx"].values(),
                    *trial["raw_vs_onnx"],
                    *trial["source_vs_raw_heads"],
                    *trial["source_vs_onnx_heads"],
                ]
            )
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
