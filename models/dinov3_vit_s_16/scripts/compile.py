#!/usr/bin/env python3
"""Compile the pinned DINOv3 ViT-S/16 ONNX into a Modalix MPK and measure its numerical fidelity."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    SkipCalibration,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import load_images, select_names
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import INPUT, INPUT_SHAPE, OUTPUT_SHAPES, SOURCE_OUTPUTS, embeddings, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "dinov3_vit_s_16"
COMPONENT_ID = "model"
PRECISIONS = ("w16a16", "w8a16", "w8a8")


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _variant(metadata: dict[str, Any], variant_id: str) -> dict[str, Any]:
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one build profile named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == COMPONENT_ID)
    return {**profile, "parameters": binding["parameters"]}


def _quantization_config(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    if precision == "w8a8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w8a16":
        activation = bfloat16_scheme()
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w16a16":
        activation = weights = bfloat16_scheme()
    else:
        raise ValueError("this recipe supports only the declared Modalix BF16 and INT8 variants")
    # Only INT8 activations take their ranges from calibration images; BF16 builds skip calibration.
    calibration = HistogramMSEMethod(int(parameters["calibration_bins"])) if precision == "w8a8" else SkipCalibration()
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(calibration)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        # with_activation_quantization turns per-token quantization on for BF16. W8A16 uses it to scale each token's
        # activations on its own before an INT8-weight operation; W16A16 does not.
        .with_per_token_quant(precision == "w8a16")
    )


def _float32_bytes(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape)) * 4


def audit_mpk(archive: Path) -> dict[str, Any]:
    """Require one MLA plugin, no CPU plugins and the model's input."""
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0]}")
        manifest = json.load(source)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = manifest.get("plugins", [])
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if public_inputs != {INPUT: _float32_bytes(INPUT_SHAPE)}:
        raise RuntimeError(f"MPK input interface mismatch: {public_inputs}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins)}


def compile_model(source: Path, component_dir: Path, variant_id: str, dataset: Path | None) -> Any:
    """Quantize (INT8 on the declared calibration images), compile and audit the MPK; returns the quantized model."""
    metadata = _metadata()
    variant = _variant(metadata, variant_id)
    parameters = variant["parameters"]
    if variant["target"] != "modalix" or parameters.get("precision") not in PRECISIONS:
        raise ValueError("this recipe supports only the declared Modalix BF16 and INT8 variants")
    data = None
    if parameters["precision"] == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build calibrates; pass --dataset")
        selection = metadata["numerical_fidelity"]["selection"]
        samples = int(parameters["calibration_samples"])
        if int(selection["offset"]) < samples:
            raise ValueError("the fidelity images must follow the calibration images")
        print(f"Calibrating on {samples} imagenet-2012 cal images", flush=True)
        calibration = load_images(dataset, select_names(dataset, int(selection["seed"]), 0, samples))
        data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): calibration}))

    component_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(source), shape_dict={INPUT: INPUT_SHAPE}, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    quantized = loaded.quantize(
        calibration_data=data,
        quantization_config=_quantization_config(parameters),
        model_name=f"{MODEL_ID}_{variant_id}",
        log_level=logging.INFO,
    )
    quantized.compile(
        output_path=str(component_dir),
        batch_size=1,
        compress=True,
        log_level=logging.INFO,
        tessellate_parameters=None,
        l2_caching_mode=L2CachingMode.NONE,
        preserve=True,
        deployable=True,
        retained_temporary_directory_name=None,
        use_power_limits=False,
        max_mla_power=None,
        # The option applies to every layer norm, and INT8 ones reject it; BF16 activations keep layer norms in BF16.
        layer_norm_use_fp32_intermediates=parameters["precision"] != "w8a8",
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    print(f"Compiled MPK: {archives[0]}, {audit_mpk(archives[0])}", flush=True)
    return quantized


@dataclass
class _OutputError:
    """Running sums of one output's error against the original ONNX, over every fidelity sample."""

    name: str
    count: int = 0
    dot: float = 0.0
    expected_squared: float = 0.0
    actual_squared: float = 0.0
    error_squared: float = 0.0
    error_absolute: float = 0.0
    max_absolute_error: float = 0.0

    def add(self, expected: np.ndarray, actual: np.ndarray) -> None:
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{self.name} shape mismatch: {expected.shape} != {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{self.name} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        self.count += expected.size
        self.dot += float(np.dot(expected.ravel(), actual.ravel()))
        self.expected_squared += float(np.sum(np.square(expected)))
        self.actual_squared += float(np.sum(np.square(actual)))
        self.error_squared += float(np.sum(np.square(error)))
        self.error_absolute += float(np.sum(absolute))
        self.max_absolute_error = max(self.max_absolute_error, float(np.max(absolute)))

    def summary(self) -> dict[str, Any]:
        expected_rms = max(np.sqrt(self.expected_squared / self.count), np.finfo(np.float64).eps)
        norm = np.sqrt(self.expected_squared * self.actual_squared)
        cosine = self.dot / norm if norm else float(self.expected_squared == self.actual_squared)
        return {
            "name": self.name,
            "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
            "normalized_rmse": float(np.sqrt(self.error_squared / self.count) / expected_rms),
            "mean_absolute_error": float(self.error_absolute / self.count),
            "max_absolute_error": self.max_absolute_error,
        }


def measure_fidelity(source: Path, candidate: Any, dataset: Path, config: dict[str, Any], variant_id: str) -> dict[str, Any]:
    """Compare the quantized graph's outputs with the converted ONNX's on the declared calibration images."""
    selection = config["selection"]
    names = select_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    errors = [_OutputError(name) for name in config["outputs"]]
    for image in load_images(dataset, names):
        expected = embeddings(*reference.run(list(SOURCE_OUTPUTS), {INPUT: image.transpose(2, 0, 1)[None]}))
        # The outputs' shapes differ, so each quantized output is matched to its name by its shape.
        executed = candidate.execute(inputs={InputName(INPUT): image[None]}, use_jax=True)
        outputs = {np.asarray(output).shape: np.asarray(output) for output in executed}
        for error in errors:
            shape = OUTPUT_SHAPES[error.name]
            if shape not in outputs:
                raise RuntimeError(f"the quantized model returned no {error.name} of shape {shape}: {list(outputs)}")
            error.add(expected[error.name], outputs[shape])
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": variant_id,
        "status": config["status"],
        "reference": "Original ONNX (ONNX Runtime)",
        "candidate": f"AFE {variant_id.upper()} graph (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": [error.summary() for error in errors],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    if args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    output_dir = args.output_dir.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    fidelity = _metadata()["numerical_fidelity"]
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        prepared = prepare_model(source, Path(temporary))
        candidate = compile_model(prepared, output_dir / COMPONENT_ID, args.build_profile, dataset)
        print(f"Measuring numerical fidelity on {fidelity['selection']['samples']} images", flush=True)
        report = measure_fidelity(source, candidate, args.fidelity_dataset.expanduser().resolve(), fidelity,
                                  args.build_profile)
    (output_dir / "fidelity.json").write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    for metric in report["metrics"]:
        print(f"Fidelity {metric['name']}: cosine {metric['cosine_similarity']:.5f}, "
              f"normalized RMSE {metric['normalized_rmse']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
