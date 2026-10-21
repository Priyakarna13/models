"""Deployment preprocessing and qualification for the tiny-drone model."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from contract import INPUT_NAME, DeploymentContract
from onnx_audit import sha256_file
from PIL import Image

MODEL_NAME = "yolo26n_p2_tiny_drone_int8_qat_b1"
EXPECTED_ATTENTION_MATMULS = 8
EXPECTED_ATTENTION_SOFTMAXES = 4
SERIALIZED_INT8_SCALAR_TYPE = 1
EXPECTED_MODEL_SDK_VERSION = json.loads(
    (Path(__file__).resolve().parents[3] / "deps" / "manifest.json").read_text(encoding="utf-8")
)["model-compiler"]["version"]


def preprocess_rgb_image(image: Image.Image, contract: DeploymentContract) -> np.ndarray:
    """Apply the deployment letterbox to one decoded RGB image."""
    width, height = image.size
    scale = min(contract.model_size / width, contract.model_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (contract.model_size - resized_width) // 2
    top = (contract.model_size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (contract.model_size, contract.model_size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float32) / np.float32(255.0)


def audit_quantized_model(path: Path, expected_convolutions: int) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"serialized model has no layer topology: {path}") from exc
    convolutions = [layer for layer in layers if str(layer.get("class_name", "")).startswith("MLA_0/ConvAddActivation")]
    if len(convolutions) != expected_convolutions:
        raise RuntimeError(f"expected {expected_convolutions} MLA convolutions, found {len(convolutions)}")

    def serialized_scalar_types(value: Any) -> tuple[int, ...]:
        values: list[int] = []
        if isinstance(value, dict):
            if value.get("__decode_key__") == "ScalarType" and isinstance(value.get("value"), int):
                values.append(value["value"])
            else:
                for child in value.values():
                    values.extend(serialized_scalar_types(child))
        elif isinstance(value, list):
            for child in value:
                values.extend(serialized_scalar_types(child))
        return tuple(values)

    def io_types(layer: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        config = layer.get("config", {})
        return (
            serialized_scalar_types(config.get("_input_types")),
            serialized_scalar_types(config.get("_output_types")),
        )

    convolution_io = {io_types(layer) for layer in convolutions}
    statuses = {layer.get("config", {}).get("_status") for layer in convolutions}
    per_channel = {layer.get("config", {}).get("per_channel") for layer in convolutions}
    integer_weights = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
    if (
        convolution_io != {((SERIALIZED_INT8_SCALAR_TYPE,), (SERIALIZED_INT8_SCALAR_TYPE,))}
        or statuses != {"SIMA_QUANTIZED"}
        or per_channel != {True}
        or integer_weights != {True}
    ):
        raise RuntimeError(
            "compiled compute graph is not fully per-channel INT8: "
            f"io_types={convolution_io}, statuses={statuses}, per_channel={per_channel}, weights={integer_weights}"
        )
    attention_kinds = {
        "BatchMatmul": EXPECTED_ATTENTION_MATMULS,
        "Softmax": EXPECTED_ATTENTION_SOFTMAXES,
    }
    attention_report = {}
    for kind, expected_count in attention_kinds.items():
        attention_layers = [layer for layer in layers if str(layer.get("class_name", "")).startswith(f"MLA_0/{kind}")]
        if len(attention_layers) != expected_count:
            raise RuntimeError(f"expected {expected_count} MLA {kind} layers, found {len(attention_layers)}")
        expected_inputs = (SERIALIZED_INT8_SCALAR_TYPE,) * (2 if kind == "BatchMatmul" else 1)
        expected_io = (expected_inputs, (SERIALIZED_INT8_SCALAR_TYPE,))
        attention_types = {io_types(layer) for layer in attention_layers}
        attention_statuses = {layer.get("config", {}).get("_status") for layer in attention_layers}
        if attention_types != {expected_io} or attention_statuses != {"SIMA_QUANTIZED"}:
            raise RuntimeError(
                f"compiled {kind} layers are not fully INT8: types={attention_types}, statuses={attention_statuses}"
            )
        attention_report[kind] = {
            "count": len(attention_layers),
            "input_scalar_type": "int8",
            "output_scalar_type": "int8",
            "status": "SIMA_QUANTIZED",
        }
    return {
        "compute_convolution_count": len(convolutions),
        "compute_scalar_type": "int8",
        "internal_precision": "int8",
        "attention": attention_report,
        "quantized_model_json_sha256": sha256_file(path),
        "weight_quantization": "per_channel_int8",
    }


def _safe_mpk(archive: Path) -> tuple[list[str], dict[str, Any]]:
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
            raise RuntimeError("cannot read MPK manifest")
        manifest = json.load(source)
    return names, manifest


def _kernel(plugin: dict[str, Any]) -> str:
    return str(plugin.get("config_params", {}).get("kernel", ""))


def audit_mpk(archive: Path, contract: DeploymentContract) -> dict[str, Any]:
    names, manifest = _safe_mpk(archive)
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    a65 = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "A65"]
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    if len(mla) != 1 or len(elfs) != 1 or a65 or shared_objects:
        raise RuntimeError(
            f"invalid MPK placement: MLA={len(mla)}, ELF={len(elfs)}, A65={len(a65)}, SO={shared_objects}"
        )
    inputs = manifest.get("input_nodes", [])
    public_input_size = int(np.prod(contract.input_shape)) * np.dtype(np.float32).itemsize
    public_output_sizes = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _name, shape in contract.outputs]
    if len(inputs) != 1 or inputs[0].get("name") != INPUT_NAME or inputs[0].get("size") != public_input_size:
        raise RuntimeError(f"MPK public input ABI mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    terminal_inputs = terminal.get("input_nodes", [])
    output_nodes = terminal.get("output_nodes", [])
    role_input_abi = [(str(node.get("name", "")).rsplit("/", 1)[-1], node.get("size")) for node in terminal_inputs]
    output_sizes = [node.get("size") for node in output_nodes]
    expected_output_abi = list(zip(contract.output_names(), public_output_sizes, strict=True))
    if (
        _kernel(terminal) != "pass_through"
        or role_input_abi != expected_output_abi
        or output_sizes != public_output_sizes
    ):
        raise RuntimeError(
            "MPK public output ABI mismatch: "
            f"expected {expected_output_abi}, got roles={role_input_abi}, output_sizes={output_sizes}"
        )
    mla_outputs = mla[0].get("output_nodes", [])
    unpack = [plugin for plugin in plugins if _kernel(plugin) == "unpack_transform"]
    if unpack:
        if (
            len(mla_outputs) != 1
            or len(unpack) != 1
            or unpack[0].get("config_params", {}).get("params", {}).get("tensor_types")
            != ["int8"] * len(contract.outputs)
        ):
            raise RuntimeError("MPK packed INT8 output transport mismatch")
    elif len(mla_outputs) != len(contract.outputs):
        raise RuntimeError(f"expected {len(contract.outputs)} MLA outputs, found {len(mla_outputs)}")
    quantize = [plugin for plugin in plugins if _kernel(plugin) == "quantization_transform"]
    dequantize = [plugin for plugin in plugins if _kernel(plugin) == "dequantization_transform"]
    external_tessellate = [plugin for plugin in plugins if _kernel(plugin) == "tessellation_transform"]
    external_detessellate = [plugin for plugin in plugins if _kernel(plugin) == "detessellation_transform"]
    quantized_types = [plugin.get("config_params", {}).get("params", {}).get("output_data_type") for plugin in quantize]
    dequantized_types = [
        plugin.get("config_params", {}).get("params", {}).get("input_data_type") for plugin in dequantize
    ]
    if quantized_types != ["int8"] or dequantized_types != ["int8"] * len(contract.outputs):
        raise RuntimeError(f"MPK INT8 transport mismatch: quantize={quantized_types}, dequantize={dequantized_types}")
    if external_tessellate or external_detessellate:
        raise RuntimeError(
            "MPK does not implement tensor layout on MLA: "
            f"external_tessellation={[plugin.get('name') for plugin in external_tessellate]}, "
            f"external_detessellation={[plugin.get('name') for plugin in external_detessellate]}"
        )
    quantized_outputs = [node.get("name") for node in quantize[0].get("output_nodes", [])]
    mla_inputs = [node.get("name") for node in mla[0].get("input_nodes", [])]
    if len(quantized_outputs) != 1 or mla_inputs != quantized_outputs:
        raise RuntimeError(
            f"MPK quantizer does not feed MLA directly: quantized_outputs={quantized_outputs}, mla_inputs={mla_inputs}"
        )
    if str(manifest.get("model_sdk_version")) != EXPECTED_MODEL_SDK_VERSION:
        raise RuntimeError(
            "MPK compiler version mismatch: "
            f"expected {EXPECTED_MODEL_SDK_VERSION}, found {manifest.get('model_sdk_version')}"
        )
    return {
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "compiler_version": str(manifest.get("model_sdk_version")),
        "elf": elfs[0],
        "mla_plugin_count": len(mla),
        "a65_plugin_count": len(a65),
        "ev74_plugin_count": sum(str(plugin.get("processor", "")).upper() == "EV74" for plugin in plugins),
        "public_input": {
            "dtype": "float32",
            "name": INPUT_NAME,
            "shape": list(contract.input_shape),
        },
        "public_output_dtype": "float32",
        "public_output_names": [name for name, _size in role_input_abi],
        "public_output_sizes": output_sizes,
        "public_output_abi": role_input_abi,
        "output_transport_types": dequantized_types,
        "mla_output_buffer_count": len(mla_outputs),
        "packed_outputs": bool(unpack),
        "quant_to_mla_direct": True,
        "external_tessellation_plugin_count": len(external_tessellate),
        "external_detessellation_plugin_count": len(external_detessellate),
        "shared_object_count": len(shared_objects),
        "tessellation": "mla",
    }
