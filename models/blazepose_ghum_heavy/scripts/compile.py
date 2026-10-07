#!/usr/bin/env python3
"""Compile the pinned BlazePose GHUM Heavy source into its Modalix BF16 MPK."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from model_registry.dependencies import resolve_model_compiler
from surgery import INPUT_NAME, INPUT_SHAPE, OUTPUTS, prepare_model
from validate_conversion import deterministic_input, validate_conversion

LOG = logging.getLogger("blazepose_ghum_heavy.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
PUBLIC_INPUT_SIZE = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
PUBLIC_OUTPUT_SIZES = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _, shape in OUTPUTS]


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one BlazePose build profile named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == "model")
    variant = {**profile, "parameters": binding["parameters"]}
    parameters = variant["parameters"]
    declared_outputs = tuple(parameters.get("output_names", []))
    if (
        variant["target"] != "modalix"
        or parameters.get("precision") != "bf16"
        or parameters.get("input_name") != INPUT_NAME
        or tuple(parameters.get("input_shape", [])) != INPUT_SHAPE
        or declared_outputs != tuple(name for name, _ in OUTPUTS)
    ):
        raise ValueError("BlazePose build profile does not match the qualified Modalix BF16 contract")
    return variant


def audit_mpk(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe BlazePose MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one BlazePose MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read BlazePose MPK manifest: {manifests[0]}")
        manifest = json.load(io.TextIOWrapper(source, encoding="utf-8"))
        preprocessors = [name for name in names if name.endswith("_preproc.json")]
        if len(preprocessors) > 1:
            raise RuntimeError(f"expected at most one BlazePose MPK preprocessor, found {preprocessors}")
        preprocessor = None
        if preprocessors:
            source = stream.extractfile(preprocessors[0])
            if source is None:
                raise RuntimeError(f"cannot read BlazePose MPK preprocessor: {preprocessors[0]}")
            preprocessor = json.load(io.TextIOWrapper(source, encoding="utf-8"))

    compiler = str(manifest.get("model_sdk_version"))
    if compiler != COMPILER_VERSION:
        raise RuntimeError(f"BlazePose MPK compiler mismatch: expected {COMPILER_VERSION}, found {compiler}")
    input_nodes = manifest.get("input_nodes", [])
    if (
        len(input_nodes) != 1
        or input_nodes[0].get("name") != INPUT_NAME
        or input_nodes[0].get("size") != PUBLIC_INPUT_SIZE
        or input_nodes[0].get("input_range") != [0.0, 1.0]
    ):
        raise RuntimeError(f"BlazePose MPK public input ABI mismatch: {input_nodes}")
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    a65_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "A65"]
    terminal = plugins[-1] if plugins else {}
    terminal_kernel = terminal.get("config_params", {}).get("kernel")
    output_sizes = [node.get("size") for node in terminal.get("output_nodes", [])]
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    if len(mla_plugins) != 1 or a65_plugins or len(elfs) != 1 or shared_objects:
        raise RuntimeError(
            f"invalid BlazePose MPK placement: MLA={len(mla_plugins)}, A65={len(a65_plugins)}, "
            f"ELF={elfs}, SO={shared_objects}"
        )
    if terminal_kernel != "pass_through" or output_sizes != PUBLIC_OUTPUT_SIZES:
        raise RuntimeError(f"BlazePose MPK public output ABI mismatch: {output_sizes}")
    if preprocessor is not None:
        preprocessor_contract = {
            "input_channels": preprocessor.get("input_channels"),
            "input_img_type": preprocessor.get("input_img_type"),
            "normalize": preprocessor.get("normalize"),
            "output_channels": preprocessor.get("output_channels"),
            "output_height": preprocessor.get("output_height"),
            "output_img_type": preprocessor.get("output_img_type"),
            "output_width": preprocessor.get("output_width"),
            "source": "preprocessor_file",
        }
        expected_preprocessor = {
            "input_channels": 3,
            "input_img_type": "RGB",
            "normalize": True,
            "output_channels": 3,
            "output_height": 256,
            "output_img_type": "RGB",
            "output_width": 256,
            "source": "preprocessor_file",
        }
        if preprocessor_contract != expected_preprocessor:
            raise RuntimeError(f"BlazePose MPK preprocessor ABI mismatch: {preprocessor_contract}")
    else:
        boundary = plugins[0] if plugins else {}
        boundary_params = boundary.get("config_params", {}).get("params", {})
        input_shapes = boundary_params.get("input_shapes")
        output_shapes = boundary_params.get("output_shapes")
        input_dtype = str(boundary_params.get("in_dtype", "")).lower()
        output_dtype = str(boundary_params.get("out_dtype", "")).lower()
        expected_shapes = [list(INPUT_SHAPE)]
        if (
            boundary.get("config_params", {}).get("kernel") != "cast_transform"
            or input_dtype != "float32"
            or output_dtype not in {"bf16", "bfloat16"}
            or input_shapes != expected_shapes
            or output_shapes != expected_shapes
        ):
            raise RuntimeError(f"BlazePose MPK input boundary ABI mismatch: {boundary}")
        preprocessor_contract = {
            "input_dtype": input_dtype,
            "input_shape": expected_shapes[0],
            "normalize": True,
            "output_dtype": output_dtype,
            "output_shape": expected_shapes[0],
            "source": "manifest_boundary",
        }
    return {
        "compiler_version": compiler,
        "elf": elfs[0],
        "mla_plugin_count": len(mla_plugins),
        "public_input": {"dtype": "float32", "name": INPUT_NAME, "shape": list(INPUT_SHAPE)},
        "public_output_dtype": "float32",
        "public_output_sizes": output_sizes,
        "preprocessor": preprocessor_contract,
        "shared_object_count": len(shared_objects),
    }


def compile_model(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant_config(variant_id)
    from afe.apis.defines import (
        CalibrationMethod,
        InputName,
        RequantizationMode,
        bfloat16_scheme,
        default_quantization,
        gen2_target,
    )
    from afe.apis.loaded_net import load_model
    from afe.apis.release_v1 import get_model_sdk_version
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat
    from sima_utils.data.data_generator import DataGenerator

    actual_compiler_version = get_model_sdk_version()
    if actual_compiler_version != COMPILER_VERSION:
        raise RuntimeError(
            f"BlazePose requires repository-pinned Model Compiler {COMPILER_VERSION}, found {actual_compiler_version}"
        )
    tflite = source.with_name("pose_landmark_heavy.tflite")
    conversion_report = validate_conversion(tflite, source)

    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="blazepose-ghum-heavy-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        prepared = prepare_model(source, Path(temporary) / "blazepose_ghum_heavy.onnx")
        importer = ImporterParams(
            format=ModelFormat.onnx,
            file_paths=[str(prepared)],
            input_names=[INPUT_NAME],
            input_shapes=[INPUT_SHAPE],
            input_types=[ScalarType.float32],
            layout="NHWC",
            output_names=[name for name, _ in OUTPUTS],
        )
        loaded = load_model(importer, target=gen2_target)
        calibration_data = convert_data_generator_to_iterable(
            DataGenerator({InputName(INPUT_NAME): deterministic_input()})
        )
        scheme = bfloat16_scheme()
        quantization = (
            default_quantization.with_activation_quantization(scheme)
            .with_weight_quantization(scheme)
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(CalibrationMethod.from_str("mse"))
        )
        model_name = f"blazepose_ghum_heavy_{variant_id}"
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            model_name=model_name,
            log_level=logging.INFO,
        )
        quantized.save(model_name=model_name, output_directory=str(output_dir))
        quantized.compile(output_path=str(output_dir), batch_size=1, log_level=logging.INFO)

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one BlazePose MPK, found {archives}")
    report = {
        "conversion": conversion_report,
        "model_id": "blazepose_ghum_heavy",
        "mpk": audit_mpk(archives[0]),
        "output_contract": [{"name": name, "shape": list(shape)} for name, shape in OUTPUTS],
        "variant_id": variant_id,
    }
    (output_dir / "blazepose-ghum-heavy-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--component", required=True)
    args = parser.parse_args()
    if args.component != "model":
        parser.error(f"unsupported component: {args.component}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile)
    LOG.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
