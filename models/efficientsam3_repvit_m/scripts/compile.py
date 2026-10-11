#!/usr/bin/env python3
"""Split the verified EfficientSAM3 ONNX and compile its text encoder and image model into BF16 MPKs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import CALIBRATION_SEED, calibration_inputs, one_hot, packed, prompts, select_images, tokens
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    IMAGE_INPUTS,
    IMAGE_OUTPUTS,
    INTERFACE,
    MASK_SIZE,
    QUERIES,
    TEXT_INPUT,
    TEXT_OUTPUT,
    graph_paths,
    prepare_model,
    validate_float_split,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "efficientsam3_repvit_m"
INVOCATION_ID = "efficientsam3"
PROFILE_ID = "modalix_bf16"
COMPONENTS = ("text_encoder", "image_model")
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]


def _float32_bytes(name: str) -> int:
    return math.prod(INTERFACE[name]) * 4


# MPK manifests record each public tensor's size in bytes; the MPKs keep the ONNX float32 interface.
COMPONENT_INTERFACES = {
    "text_encoder": {"inputs": {TEXT_INPUT: _float32_bytes(TEXT_INPUT)}, "outputs": [_float32_bytes(TEXT_OUTPUT)]},
    "image_model": {
        "inputs": {name: _float32_bytes(name) for name in IMAGE_INPUTS},
        "outputs": [_float32_bytes(name) for name in IMAGE_OUTPUTS],
    },
}


def _model() -> dict[str, Any]:
    return yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))


def _quantization(precision: str, calibration_bins: int) -> Any:
    if precision != "bf16":
        raise ValueError(f"unsupported precision: {precision}")
    return (
        default_quantization.with_activation_quantization(bfloat16_scheme())
        .with_weight_quantization(bfloat16_scheme())
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(HistogramMSEMethod(calibration_bins))
        .with_bias_correction(BiasCorrectionType.NONE)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        .with_prefer_int8_udf(True)
        # with_activation_quantization turns per-token quantization on for BF16; this recipe keeps it off.
        .with_per_token_quant(False)
    )


def _quantize_component(
    graph: Path,
    temporary_dir: Path,
    component_id: str,
    precision: str,
    calibration_bins: int,
    inputs: dict[str, np.ndarray],
) -> tuple[Any, Path]:
    name = f"{MODEL_ID}_{component_id}_{PROFILE_ID}"
    build_dir = temporary_dir / name
    build_dir.mkdir(parents=True, exist_ok=False)
    shapes = {key: tuple(INTERFACE[key]) for key in inputs}
    importer = onnx_source(str(graph), shape_dict=shapes, layout="NCHW", flexible_batch_size=False)
    loaded = load_model(importer, target=gen2_target)
    data = convert_data_generator_to_iterable(DataGenerator({InputName(key): value for key, value in inputs.items()}))
    quantized = loaded.quantize(
        calibration_data=data,
        quantization_config=_quantization(precision, calibration_bins),
        model_name=name,
        log_level=logging.INFO,
    )
    return quantized, build_dir


def _read_mpk_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
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
        return names, json.load(source)


def audit_mpk(archive: Path, component_id: str) -> dict[str, Any]:
    contract = COMPONENT_INTERFACES[component_id]
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid {component_id} MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, "
            f"host={len(host_plugins)}, shared_objects={shared_objects}"
        )
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"{component_id} MPK compiler mismatch: expected {COMPILER_VERSION}, "
            f"found {manifest.get('model_sdk_version')}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if public_inputs != contract["inputs"]:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {public_inputs}")
    # A single-output MPK ends in the float32 cast; a multi-output one gathers its casts in a pass-through.
    output_sizes = [node.get("size") for node in plugins[-1].get("output_nodes", [])] if plugins else []
    if output_sizes != contract["outputs"]:
        raise RuntimeError(f"{component_id} MPK output interface mismatch: {output_sizes}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins), "output_sizes": output_sizes}


def _compile_component(quantized: Any, build_dir: Path, component_id: str) -> Path:
    quantized.compile(
        output_path=str(build_dir),
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
        layer_norm_use_fp32_intermediates=True,
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    audit_mpk(archives[0], component_id)
    return archives[0]


def image_outputs(outputs: list[Any]) -> list[np.ndarray]:
    """Return the image model's detections and NCHW mask logits from an AFE execution."""
    detections, masks = (np.asarray(output, dtype=np.float32) for output in outputs)
    if masks.shape == (1, MASK_SIZE, MASK_SIZE, QUERIES):
        masks = masks.transpose(0, 3, 1, 2)
    shapes = [list(detections.shape), list(masks.shape)]
    if shapes != [INTERFACE[name] for name in IMAGE_OUTPUTS]:
        raise RuntimeError(f"unexpected image model outputs: {shapes}")
    return [detections, masks]


def _record_metrics(
    totals: dict[str, dict[str, float]],
    reference: list[np.ndarray],
    candidate: list[np.ndarray],
    names: list[str],
) -> None:
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{name} shape mismatch: {expected.shape} != {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{name} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        total = totals.setdefault(
            name,
            {
                "count": 0.0,
                "dot": 0.0,
                "expected_squared": 0.0,
                "actual_squared": 0.0,
                "error_squared": 0.0,
                "error_absolute": 0.0,
                "max_absolute_error": 0.0,
            },
        )
        total["count"] += expected.size
        total["dot"] += float(np.dot(expected.ravel(), actual.ravel()))
        total["expected_squared"] += float(np.sum(np.square(expected)))
        total["actual_squared"] += float(np.sum(np.square(actual)))
        total["error_squared"] += float(np.sum(np.square(error)))
        total["error_absolute"] += float(np.sum(absolute))
        total["max_absolute_error"] = max(total["max_absolute_error"], float(np.max(absolute)))


def _finish_metrics(totals: dict[str, dict[str, float]], names: list[str]) -> list[dict[str, Any]]:
    results = []
    for name in names:
        total = totals[name]
        count = total["count"]
        expected_rms = max(np.sqrt(total["expected_squared"] / count), np.finfo(np.float64).eps)
        norm = np.sqrt(total["expected_squared"] * total["actual_squared"])
        cosine = total["dot"] / norm if norm else float(total["expected_squared"] == total["actual_squared"])
        results.append(
            {
                "name": name,
                "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
                "normalized_rmse": float(np.sqrt(total["error_squared"] / count) / expected_rms),
                "mean_absolute_error": float(total["error_absolute"] / count),
                "max_absolute_error": total["max_absolute_error"],
            }
        )
    return results


def _write_fidelity(output_dir: Path, config: dict[str, Any], metrics: list[dict[str, Any]]) -> Path:
    selection = config["selection"]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": PROFILE_ID,
        "component_id": config["result_component"],
        "scope": "full_pipeline",
        "status": config["status"],
        "reference": "Original full ONNX (ONNX Runtime)",
        "candidate": "AFE BF16 text encoder → packed text → AFE BF16 image model (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": selection["samples"],
        "metrics": metrics,
    }
    path = output_dir / "fidelity.json"
    path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prepare_graphs_and_reference(
    source: Path,
    dataset: Path,
    fidelity_names: list[str],
    temporary_dir: Path,
) -> None:
    text_graph, image_graph = prepare_model(source, temporary_dir / "graphs")
    prompts, images = calibration_inputs(dataset, fidelity_names)
    report, reference = validate_float_split(source, text_graph, image_graph, images, prompts)
    logging.info("Float split matches the converted graph: %s", json.dumps(report))
    np.savez(temporary_dir / "reference_outputs.npz", **reference)


def _compile_text_encoder(
    graph: Path,
    temporary_dir: Path,
    calibration_prompts: list[str],
    fidelity_prompts: list[str],
    precision: str,
    calibration_bins: int,
) -> None:
    calibration = np.concatenate([one_hot(tokens(prompt)) for prompt in calibration_prompts])
    encoder, build_dir = _quantize_component(
        graph, temporary_dir, "text_encoder", precision, calibration_bins, {TEXT_INPUT: calibration}
    )
    _compile_component(encoder, build_dir, "text_encoder")

    # The image model calibrates and is checked on the quantized encoder's features, as the board runs.
    def packed_text(prompt: str) -> np.ndarray:
        (features,) = encoder.execute(inputs={InputName(TEXT_INPUT): one_hot(tokens(prompt))}, use_jax=True)
        return packed(np.asarray(features), tokens(prompt))

    np.savez(
        temporary_dir / "text_outputs.npz",
        calibration=np.concatenate([packed_text(prompt) for prompt in calibration_prompts]),
        fidelity=np.concatenate([packed_text(prompt) for prompt in fidelity_prompts]),
    )


def _compile_image_model(
    graph: Path,
    dataset: Path,
    calibration_names: list[str],
    fidelity_names: list[str],
    temporary_dir: Path,
    precision: str,
    calibration_bins: int,
    fidelity_config: dict[str, Any],
) -> None:
    with np.load(temporary_dir / "text_outputs.npz") as text:
        calibration_text, fidelity_text = text["calibration"], text["fidelity"]
    _, calibration_images = calibration_inputs(dataset, calibration_names)
    model, build_dir = _quantize_component(
        graph,
        temporary_dir,
        "image_model",
        precision,
        calibration_bins,
        {"pixel_values": calibration_images, "text": calibration_text},
    )
    del calibration_images
    _compile_component(model, build_dir, "image_model")

    _, fidelity_images = calibration_inputs(dataset, fidelity_names)
    totals: dict[str, dict[str, float]] = {}
    with np.load(temporary_dir / "reference_outputs.npz") as reference:
        for index, image in enumerate(fidelity_images):
            inputs = {InputName("pixel_values"): image[None], InputName("text"): fidelity_text[index : index + 1]}
            actual = image_outputs(list(model.execute(inputs=inputs, use_jax=True)))
            expected = [reference[name][index][None] for name in IMAGE_OUTPUTS]
            _record_metrics(totals, expected, actual, IMAGE_OUTPUTS)
    _write_fidelity(temporary_dir, fidelity_config, _finish_metrics(totals, IMAGE_OUTPUTS))


def _execute_phase(target: Callable[..., None], args: tuple[Any, ...]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    target(*args)


def _run_phase(name: str, target: Callable[..., None], *args: Any) -> None:
    logging.info("Starting %s", name)
    process = multiprocessing.get_context("spawn").Process(name=name, target=_execute_phase, args=(target, args))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{name} failed with exit code {process.exitcode}")
    logging.info("Completed %s", name)


def _component_archive(temporary_dir: Path, component_id: str) -> Path:
    archives = sorted((temporary_dir / f"{MODEL_ID}_{component_id}_{PROFILE_ID}").glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    return archives[0]


def compile_model(source: Path, calibration_dataset: Path, fidelity_dataset: Path, output_dir: Path) -> dict[str, Path]:
    metadata = _model()
    profile = next(profile for profile in metadata["build_profiles"] if profile["id"] == PROFILE_ID)
    bindings = {binding["id"]: binding["parameters"] for binding in profile["components"]}
    calibration_samples = int(bindings["text_encoder"]["calibration_samples"])
    fidelity_config = next(entry for entry in metadata["fidelity"] if entry["id"] == "full_pipeline_fidelity")
    selection = fidelity_config["selection"]
    if int(bindings["image_model"]["calibration_samples"]) != calibration_samples:
        raise ValueError("text encoder and image model must calibrate on the same samples")
    if int(selection["offset"]) != calibration_samples:
        raise ValueError("fidelity selection must begin after the calibration partition")
    if int(selection["seed"]) != CALIBRATION_SEED:
        raise ValueError(f"fidelity selection seed must be the calibration seed {CALIBRATION_SEED}")
    if calibration_dataset.resolve() != fidelity_dataset.resolve():
        raise ValueError("EfficientSAM3 calibration and fidelity must use the same verified dataset")
    calibration_names, fidelity_names = select_images(
        calibration_dataset, calibration_samples, int(selection["samples"])
    )
    calibration_prompts, fidelity_prompts = prompts(len(calibration_names)), prompts(len(fidelity_names))
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        text_graph, image_graph = graph_paths(temporary_dir / "graphs")
        _run_phase(
            "EfficientSAM3 graph preparation",
            _prepare_graphs_and_reference,
            source,
            fidelity_dataset,
            fidelity_names,
            temporary_dir,
        )
        graph_hashes = {"text_encoder": sha256_file(text_graph)[0], "image_model": sha256_file(image_graph)[0]}
        _run_phase(
            "EfficientSAM3 text encoder compilation",
            _compile_text_encoder,
            text_graph,
            temporary_dir,
            calibration_prompts,
            fidelity_prompts,
            str(bindings["text_encoder"]["precision"]),
            int(bindings["text_encoder"]["calibration_bins"]),
        )
        _run_phase(
            "EfficientSAM3 image model compilation and fidelity",
            _compile_image_model,
            image_graph,
            calibration_dataset,
            calibration_names,
            fidelity_names,
            temporary_dir,
            str(bindings["image_model"]["precision"]),
            int(bindings["image_model"]["calibration_bins"]),
            fidelity_config,
        )

        output_dir.mkdir(parents=True)
        archives = {}
        for component_id in COMPONENTS:
            source_archive = _component_archive(temporary_dir, component_id)
            (output_dir / component_id).mkdir()
            archives[component_id] = output_dir / component_id / source_archive.name
            shutil.copyfile(source_archive, archives[component_id])

        fidelity_component = fidelity_config["result_component"]
        fidelity_path = output_dir / fidelity_component / "fidelity.json"
        shutil.copyfile(temporary_dir / "fidelity.json", fidelity_path)
        manifest = {
            "model_id": MODEL_ID,
            "build_profile_id": PROFILE_ID,
            "compile_invocation_id": INVOCATION_ID,
            "outputs": [
                {
                    "component_id": component_id,
                    "mpk_path": f"{component_id}/{archives[component_id].name}",
                    "mpk_sha256": sha256_file(archives[component_id])[0],
                }
                for component_id in COMPONENTS
            ],
            "derived_graphs": [
                {"component_id": component_id, "sha256": graph_hashes[component_id]} for component_id in COMPONENTS
            ],
            "correctness": {
                "component_id": fidelity_component,
                "path": f"{fidelity_component}/fidelity.json",
                "sha256": sha256_file(fidelity_path)[0],
            },
        }
        (output_dir / "compile-outputs.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fidelity-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.build_profile != PROFILE_ID:
        parser.error(f"unsupported build profile: {args.build_profile}")
    if args.compile_invocation != INVOCATION_ID:
        parser.error(f"unsupported compile invocation: {args.compile_invocation}")
    source = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    fidelity_dataset = args.fidelity_dataset.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the manual conversion workflow")
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if not fidelity_dataset.exists():
        parser.error(f"fidelity dataset does not exist: {fidelity_dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs = compile_model(source, dataset, fidelity_dataset, args.output_dir.expanduser().resolve())
    for component_id, archive in outputs.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
