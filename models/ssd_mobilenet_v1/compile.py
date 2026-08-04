#!/usr/bin/env python3
"""Quantize and compile one pinned SSD-MobileNet V1 Modalix variant."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from calibration import (
    ARCHIVE_SHA256,
    CALIBRATION_ARRAY_SHA256,
    CALIBRATION_IMAGE_IDS,
    VALIDATION_ARRAY_SHA256,
    sha256_file,
    validate_dataset,
)
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec
from surgery import INPUT_NAME, INPUT_SHAPE, OUTPUTS, prepare_model

LOG = logging.getLogger("ssd_mobilenet_v1.compile")
MODEL_DIR = Path(__file__).resolve().parent
COMPILER_VERSION = "2.1.3"
PUBLIC_INPUT_SIZE = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
PUBLIC_OUTPUT_SIZES = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _, shape in OUTPUTS]


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["compile_variants"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one SSD-MobileNet V1 variant named {variant_id!r}")
    variant = matches[0]
    parameters = variant["parameters"]
    if variant["target"] != "modalix" or parameters.get("precision") not in {"int8", "bf16"}:
        raise ValueError("SSD-MobileNet V1 supports only declared Modalix INT8 and BF16 variants")
    if not isinstance(parameters.get("tessellation_mla"), bool):
        raise ValueError("tessellation_mla must be a boolean")
    expected_quantization = (
        ("asymmetric_per_tensor_8bit", "symmetric_per_channel_8bit")
        if parameters["precision"] == "int8"
        else ("bfloat16", "bfloat16")
    )
    declared_quantization = (parameters.get("activation_quantization"), parameters.get("weight_quantization"))
    if declared_quantization != expected_quantization:
        raise ValueError(f"SSD quantization declaration mismatch: {declared_quantization}")
    if parameters.get("calibration_method") != "mse" or parameters.get("requantization") != "sima":
        raise ValueError("SSD variants require MSE calibration and SiMa requantization")
    if parameters.get("calibration_samples") != len(CALIBRATION_IMAGE_IDS):
        raise ValueError(f"SSD variants require exactly {len(CALIBRATION_IMAGE_IDS)} calibration samples")
    return variant


def calibration_source() -> SourceSpec:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    calibration = metadata["calibration"]
    if calibration.get("status") != "ready":
        raise ValueError("SSD-MobileNet V1 requires pinned representative calibration data")
    return SourceSpec.from_mapping(calibration["source"])


def _tessellation_parameters(
    model: Any,
    *,
    TensorDRAMLayout: Any,
    TensorTessellateParameters: Any,
    node_is_tuple: Any,
) -> dict[str, Any]:
    if "MLA_0" not in model._net.nodes:
        raise RuntimeError("quantized SSD graph does not contain MLA_0")
    mla = model._net.nodes["MLA_0"]
    input_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC,
    )
    output_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC16,
    )
    parameters = {name: input_layout for name in mla.input_names}
    output_node = mla.ir.nodes[mla.ir.output_node_name]
    output_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    parameters.update({f"{name}_output": output_layout for name in output_names})
    if len(output_names) != len(OUTPUTS):
        raise RuntimeError(f"expected {len(OUTPUTS)} MLA outputs, found {len(output_names)}")
    return parameters


def _safe_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
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
    return names, manifest


def _kernel(plugin: dict[str, Any]) -> str:
    return str(plugin.get("config_params", {}).get("kernel", ""))


def _params(plugin: dict[str, Any]) -> dict[str, Any]:
    value = plugin.get("config_params", {}).get("params", {})
    return value if isinstance(value, dict) else {}


def audit_quantized_model(path: Path, precision: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"quantized SSD serialization has no layer topology: {path}") from exc
    convolutions = [layer for layer in layers if str(layer.get("class_name", "")).startswith("MLA_0/ConvAddActivation")]
    if len(convolutions) != 47:
        raise RuntimeError(f"expected 47 quantized SSD MLA convolutions, found {len(convolutions)}")
    expected_scalar_type = 6 if precision == "int8" else 13
    scalar_types = {layer.get("config", {}).get("conv_attrs.input_type", {}).get("value") for layer in convolutions}
    statuses = {layer.get("config", {}).get("_status") for layer in convolutions}
    if scalar_types != {expected_scalar_type} or statuses != {"SIMA_QUANTIZED"}:
        raise RuntimeError(
            f"{precision.upper()} SSD compute precision audit failed: types={scalar_types}, statuses={statuses}"
        )
    if precision == "int8":
        per_channel = {layer.get("config", {}).get("per_channel") for layer in convolutions}
        has_weight_quantization = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
        if per_channel != {True} or has_weight_quantization != {True}:
            raise RuntimeError("INT8 SSD weights are not represented as per-channel quantized convolution weights")
    else:
        has_bfloat_weights = {"weights_attrs.data" in layer.get("config", {}) for layer in convolutions}
        has_integer_weights = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
        if has_bfloat_weights != {True} or has_integer_weights != {False}:
            raise RuntimeError("BF16 SSD convolution weights do not match the serialized BF16 topology")
    return {
        "compute_convolution_count": len(convolutions),
        "compute_scalar_type": expected_scalar_type,
        "internal_precision": precision,
        "quantized_model_json_sha256": sha256_file(path),
        "weight_quantization": "per_channel_int8" if precision == "int8" else "bfloat16",
    }


def audit_mpk(archive: Path, variant_id: str) -> dict[str, Any]:
    variant = variant_config(variant_id)
    precision = str(variant["parameters"]["precision"])
    tessellation_mla = bool(variant["parameters"]["tessellation_mla"])
    names, manifest = _safe_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    a65_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "A65"]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects:
        raise RuntimeError(f"invalid SSD MPK packaging: MLA={len(mla_plugins)} ELF={len(elfs)} SO={shared_objects}")
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"SSD MPK compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}"
        )

    input_nodes = manifest.get("input_nodes", [])
    if (
        len(input_nodes) != 1
        or input_nodes[0].get("name") != INPUT_NAME
        or input_nodes[0].get("size") != PUBLIC_INPUT_SIZE
    ):
        raise RuntimeError(f"SSD MPK public input ABI mismatch: {input_nodes}")
    terminal = plugins[-1] if plugins else {}
    output_sizes = [node.get("size") for node in terminal.get("output_nodes", [])]
    if _kernel(terminal) != "pass_through" or output_sizes != PUBLIC_OUTPUT_SIZES:
        raise RuntimeError(f"SSD MPK public output ABI mismatch: {output_sizes}")

    unpack = [plugin for plugin in plugins if _kernel(plugin) == "unpack_transform"]
    if len(unpack) != 1:
        raise RuntimeError(f"expected one SSD output unpack transform, found {len(unpack)}")
    internal_types = [str(value).lower() for value in _params(unpack[0]).get("tensor_types", [])]
    aliases = {"int8"} if precision == "int8" or not tessellation_mla else {"bf16", "bfloat16"}
    if len(internal_types) != len(OUTPUTS) or any(value not in aliases for value in internal_types):
        raise RuntimeError(f"{precision.upper()} SSD output transport audit failed: {internal_types}")

    if precision == "int8":
        quantize = [plugin for plugin in plugins if _kernel(plugin) == "quantization_transform"]
        dequantize = [plugin for plugin in plugins if _kernel(plugin) == "dequantization_transform"]
        quantized_types = [str(_params(plugin).get("output_data_type", "")).lower() for plugin in quantize]
        dequantized_types = [str(_params(plugin).get("input_data_type", "")).lower() for plugin in dequantize]
        if quantized_types != ["int8"] or dequantized_types != ["int8"] * len(OUTPUTS):
            raise RuntimeError(
                f"INT8 SSD boundary transforms are inconsistent: quantize={quantized_types}, "
                f"dequantize={dequantized_types}"
            )

    external_tessellation = any(PurePosixPath(name).name == "0_tessellate.json" for name in names)
    if external_tessellation == tessellation_mla or a65_plugins:
        placement = "in MLA" if tessellation_mla else "on EV74 outside MLA"
        raise RuntimeError(
            f"SSD tessellation placement audit failed for {placement}: "
            f"external_tessellation={external_tessellation}, A65={len(a65_plugins)}"
        )

    return {
        "archive": archive.name,
        "calibration_archive_sha256": ARCHIVE_SHA256,
        "calibration_tensor_sha256": CALIBRATION_ARRAY_SHA256,
        "compiler_version": COMPILER_VERSION,
        "elf": elfs[0],
        "output_transport_tensor_count": len(internal_types),
        "output_transport_types": internal_types,
        "mla_plugin_count": len(mla_plugins),
        "public_input": {"dtype": "float32", "name": INPUT_NAME, "shape": list(INPUT_SHAPE)},
        "public_output_dtype": "float32",
        "public_output_sizes": output_sizes,
        "shared_object_count": len(shared_objects),
        "tessellation": "mla" if tessellation_mla else "ev74",
        "validation_tensor_sha256": VALIDATION_ARRAY_SHA256,
    }


def compile_model(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    precision = str(variant["parameters"]["precision"])
    tessellation_mla = bool(variant["parameters"]["tessellation_mla"])

    # These imports intentionally stay inside the Vulcan-only path. Registry,
    # surgery, provenance, and MPK-audit tests do not require an SDK install.
    from afe.apis.defines import (
        CalibrationMethod,
        InputName,
        RequantizationMode,
        TensorDRAMLayout,
        TensorTessellateParameters,
        bfloat16_scheme,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.ir.node import node_is_tuple
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat
    from sima_utils.data.data_generator import DataGenerator

    calibration_path, _ = Fetcher().fetch(calibration_source())
    calibration, _validation = validate_dataset(calibration_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ssd-mobilenet-v1-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        prepared = prepare_model(source, Path(temporary) / "ssd_mobilenet_v1_heads.onnx")
        importer = ImporterParams(
            format=ModelFormat.onnx,
            file_paths=[str(prepared)],
            input_names=[INPUT_NAME],
            input_shapes=[INPUT_SHAPE],
            input_types=[ScalarType.float32],
            layout="NCHW",
            output_names=[name for name, _ in OUTPUTS],
        )
        loaded = load_model(importer, target=gen2_target)
        calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT_NAME): calibration}))
        if precision == "int8":
            activation_scheme = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            weight_scheme = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
        else:
            activation_scheme = weight_scheme = bfloat16_scheme()
        quantization = (
            default_quantization.with_activation_quantization(activation_scheme)
            .with_weight_quantization(weight_scheme)
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(CalibrationMethod.from_str("mse"))
        )
        model_name = f"ssd_mobilenet_v1_{variant_id}"
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            any_shape_on_mla=False,
            automatic_layout_conversion=False,
            model_name=model_name,
            log_level=logging.INFO,
        )
        quantized.save(model_name=model_name, output_directory=str(output_dir))
        quantized_report = audit_quantized_model(output_dir / f"{model_name}.sima.json", precision)
        tessellation = None
        if tessellation_mla:
            tessellation = _tessellation_parameters(
                quantized,
                TensorDRAMLayout=TensorDRAMLayout,
                TensorTessellateParameters=TensorTessellateParameters,
                node_is_tuple=node_is_tuple,
            )
        quantized.compile(
            output_path=str(output_dir),
            batch_size=1,
            log_level=logging.INFO,
            tessellate_parameters=tessellation,
        )

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one SSD MPK, found {len(archives)} in {output_dir}")
    report = audit_mpk(archives[0], variant_id)
    report.update(quantized_report)
    (output_dir / "ssd-mobilenet-v1-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.variant)
    LOG.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
