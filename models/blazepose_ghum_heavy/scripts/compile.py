#!/usr/bin/env python3
"""Cut the converted BlazePose GHUM Heavy ONNX to its landmark outputs and compile it into one Modalix MPK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import InputName, QuantizationParams, SkipCalibration, bfloat16_scheme, gen2_target
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.load.importers.general_importer import onnx_source
from model_registry.model import load_model as load_metadata
from surgery import INPUT, INPUT_SHAPE, OUTPUT_SHAPES, OUTPUTS, SEEDS, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "blazepose_ghum_heavy"
COMPONENT_ID = "model"
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(OUTPUT_SHAPES[name]) * 4 for name in OUTPUTS]
# Mean absolute error the BF16 graph may have from the converted ONNX on any seeded input, about twice the
# worst measured: 5.7 px, 1.6e-5 and 0.034 m. A broken build misses by tens of pixels.
MAX_MEAN_ERROR = {"Identity": 12.0, "Identity_1": 1e-3, "Identity_4": 0.07}


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _quantization(precision: str) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. BF16 activations need no calibration."""
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


def check_quantized_outputs(source: Path, quantized: Any) -> dict[str, float]:
    """Require each quantized output to stay within MAX_MEAN_ERROR of the converted ONNX on every seeded input."""
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    errors = dict.fromkeys(OUTPUTS, 0.0)
    for seed in SEEDS:
        sample = np.random.RandomState(seed).uniform(0, 1, INPUT_SHAPE).astype(np.float32)
        expected = reference.run(OUTPUTS, {INPUT: sample})
        actual = quantized.execute(inputs={InputName(INPUT): sample}, use_jax=True)
        for name, expected_value, actual_value in zip(OUTPUTS, expected, actual, strict=True):
            actual_value = np.asarray(actual_value, np.float32).reshape(expected_value.shape)
            if not np.isfinite(actual_value).all():
                raise RuntimeError(f"quantized output {name} holds non-finite values (seed {seed})")
            errors[name] = max(errors[name], float(np.mean(np.abs(actual_value - expected_value))))
    logging.info("Largest mean absolute error of the quantized outputs against the converted ONNX: %s", errors)
    exceeded = {name: error for name, error in errors.items() if error > MAX_MEAN_ERROR[name]}
    if exceeded:
        raise RuntimeError(f"quantized outputs exceed {MAX_MEAN_ERROR}: {exceeded}")
    return errors


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


def audit_mpk(archive: Path) -> None:
    """Require the whole network on one MLA segment, with no host code, and the declared interface."""
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != INPUT_BYTES:
        raise RuntimeError(f"MPK input interface mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel != "pass_through" or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str) -> Path:
    quantization = _quantization(str(_parameters(profile_id)["precision"]))
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph = prepare_model(source, build_dir / "graphs")
        importer = onnx_source(
            str(graph), shape_dict={INPUT: tuple(INPUT_SHAPE)}, layout="NHWC", flexible_batch_size=True
        )
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=None,
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        check_quantized_outputs(source, quantized)
        quantized.compile(
            output_path=str(build_dir / "mpk"),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            deployable=True,
        )
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        audit_mpk(archives[0])
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the conversion workflow")
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    # Importing the Model Compiler configures the root logger, so the recipe's INFO lines need force.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile)
    logging.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
