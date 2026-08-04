#!/usr/bin/env python3
"""Compile the pinned SuperPoint source into one Modalix MPK variant."""

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
from calibration import validate_calibration
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec
from surgery import HEADS, INPUT_NAME, INPUT_SHAPE, prepare_model

LOG = logging.getLogger("superpoint.compile")
MODEL_DIR = Path(__file__).resolve().parent


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["compile_variants"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one SuperPoint variant named {variant_id!r}")
    return matches[0]


def calibration_source() -> SourceSpec:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    calibration = metadata["calibration"]
    if calibration.get("status") != "ready":
        raise ValueError("SuperPoint requires pinned calibration data")
    return SourceSpec.from_mapping(calibration["source"])


def _calibration_data(path: Path, input_name: Any) -> list[dict[Any, np.ndarray]]:
    calibration = validate_calibration(path)
    return [{input_name: np.transpose(calibration[index], (0, 2, 3, 1))} for index in range(calibration.shape[0])]


def _channel_extent(shape: tuple[int, ...]) -> int | None:
    matches = [extent for extent in shape if extent in {65, 256}]
    return matches[0] if len(matches) == 1 else None


def _tessellation_parameters(
    model: Any,
    *,
    input_enable_mla: bool,
    output_enable_mla: bool,
    TensorDRAMLayout: Any,
    TensorTessellateParameters: Any,
    node_is_tuple: Any,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, node in model._net.nodes.items():
        if not str(name).startswith("MLA") or not hasattr(node, "ir") or not hasattr(node.ir, "nodes"):
            continue
        for input_name in node.input_names:
            rank = len(tuple(node.ir.nodes[input_name].get_type().output.value.shape))
            parameters[input_name] = TensorTessellateParameters(
                tile_shape=(0,) * rank,
                enable_mla=input_enable_mla,
                dram_layout=TensorDRAMLayout.HWC if rank >= 4 else None,
                dram_shape=None,
            )

        output_node = node.ir.nodes[node.ir.output_node_name]
        output_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
        for output_name in output_names:
            shape = tuple(node.ir.nodes[output_name].get_type().output.value.shape)
            channel = _channel_extent(shape)
            layout = None
            if output_enable_mla and len(shape) >= 4:
                layout = TensorDRAMLayout.HWC16 if channel and channel % 16 else TensorDRAMLayout.HWC
            parameters[f"{output_name}_output"] = TensorTessellateParameters(
                tile_shape=(0,) * len(shape),
                enable_mla=output_enable_mla,
                dram_layout=layout,
                dram_shape=None,
            )
    if not parameters:
        raise RuntimeError("quantized SuperPoint graph does not contain an MLA stage")
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


def audit_mpk(archive: Path, variant_id: str) -> dict[str, Any]:
    names, manifest = _safe_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects:
        raise RuntimeError(
            f"invalid SuperPoint MPK packaging: MLA={len(mla_plugins)} ELF={len(elfs)} SO={shared_objects}"
        )

    first_inputs = plugins[0].get("input_nodes", []) if plugins else []
    input_dtype = str(first_inputs[0].get("dtype", "")).lower() if len(first_inputs) == 1 else ""
    mla_outputs = mla_plugins[0].get("output_nodes", [])
    mla_layout = str(mla_outputs[0].get("layout", "")).lower() if len(mla_outputs) == 1 else ""
    kernels = [str(plugin.get("config_params", {}).get("kernel", "")) for plugin in plugins]
    frame_types = [
        str(plugin.get("config_params", {}).get("params", {}).get("frame_type", "")).lower() for plugin in plugins
    ]
    unpack_types = [
        str(dtype).lower()
        for plugin in plugins
        for dtype in plugin.get("config_params", {}).get("params", {}).get("tensor_types", [])
    ]

    if variant_id == "modalix_int8_packed":
        if input_dtype != "float32" or mla_layout != "tessellated" or frame_types.count("int8") != 2:
            raise RuntimeError("INT8 packed SuperPoint boundary does not match the qualified ABI")
        expected_boundary = "FP32 input; INT8 packed MLA heads; EV74 detessellation/dequantization"
    elif variant_id == "modalix_bf16_packed":
        if input_dtype != "bfloat16" or mla_layout != "tessellated" or frame_types.count("bfloat16") != 2:
            raise RuntimeError("BF16 packed SuperPoint boundary does not match the qualified ABI")
        expected_boundary = "BF16 input and detessellated heads from packed MLA output"
    elif variant_id == "modalix_fp32_dense":
        if input_dtype != "bfloat16" or mla_layout != "normal" or unpack_types.count("float32") != 2:
            raise RuntimeError("FP32 dense SuperPoint boundary does not match the qualified ABI")
        expected_boundary = "BF16 input; dense FP32 heads"
    else:
        raise ValueError(f"unsupported SuperPoint variant: {variant_id}")

    if "pass_through" not in kernels and variant_id != "modalix_bf16_packed":
        raise RuntimeError("SuperPoint multi-output publication is missing the terminal pass-through stage")
    return {
        "archive": archive.name,
        "boundary": expected_boundary,
        "elf": elfs[0],
        "plugin_count": len(plugins),
        "mla_plugin_count": len(mla_plugins),
        "shared_object_count": len(shared_objects),
    }


def compile_model(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    parameters = variant["parameters"]
    if variant["target"] != "modalix" or parameters.get("profile") != "w8a8-vision":
        raise ValueError("SuperPoint variants require Modalix W8A8 compilation")

    from afe.apis.defines import (
        BiasCorrectionType,
        CalibrationMethod,
        InputName,
        RequantizationMode,
        TensorDRAMLayout,
        TensorTessellateParameters,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.ir.defines import NodeName
    from afe.ir.node import node_is_tuple
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat

    calibration_path, _ = Fetcher().fetch(calibration_source())
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="superpoint-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        derived = prepare_model(source, Path(temporary) / "superpoint_dense_core_480x640.onnx")
        output_names = [name for name, _ in HEADS.values()]

        public_input_mode = str(parameters["public_input_dtype"])
        public_output_mode = str(parameters["public_output_dtype"])
        float_types = {"float32": ScalarType.float32, "bfloat16": ScalarType.bfloat16}
        quantization_mla = bool(parameters["quantization_mla"])
        public_inputs = {InputName(INPUT_NAME): float_types[public_input_mode]} if quantization_mla else {}
        public_outputs = (
            {NodeName(name): float_types[public_output_mode] for name in output_names} if quantization_mla else None
        )
        importer = ImporterParams(
            format=ModelFormat.onnx,
            file_paths=[str(derived)],
            input_names=[INPUT_NAME],
            input_shapes=[INPUT_SHAPE],
            input_types=[ScalarType.float32],
            layout="NCHW",
            output_names=output_names,
            public_input_types=public_inputs,
            public_output_types=public_outputs,
        )
        loaded = load_model(importer, target=gen2_target)
        calibration = _calibration_data(calibration_path, InputName(INPUT_NAME))
        quantization = (
            default_quantization.with_activation_quantization(
                quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            )
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_requantization_mode(RequantizationMode.sima)
            .with_quantization_mla(quantization_mla)
            .with_tesselation_mla(bool(parameters["tesselation_mla"]))
            .with_calibration(CalibrationMethod.from_str("mse"))
            .with_bias_correction(BiasCorrectionType.REGULAR)
            .with_neat_quantization(str(parameters["profile"]))
        )
        model_name = f"superpoint_{variant_id}"
        quantized = loaded.quantize(
            calibration_data=calibration,
            quantization_config=quantization,
            any_shape_on_mla=True,
            automatic_layout_conversion=True,
            model_name=model_name,
            log_level=logging.ERROR,
        )
        tessellation = _tessellation_parameters(
            quantized,
            input_enable_mla=bool(parameters["input_tesselation_mla"]),
            output_enable_mla=bool(parameters["output_tesselation_mla"]),
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
        raise RuntimeError(f"expected one SuperPoint MPK, found {len(archives)} in {output_dir}")
    report = audit_mpk(archives[0], variant_id)
    (output_dir / "superpoint-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
