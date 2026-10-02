#!/usr/bin/env python3
"""Retain the BlazePose application ABI and replace channel-padding operators."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx.utils import Extractor

SOURCE_SHA256 = "efcc9e62707d596c86bfc962a26a673401d8438702216f82fc15ead54c6f3a83"
INPUT_NAME = "input_1"
INPUT_SHAPE = (1, 256, 256, 3)
SOURCE_OUTPUTS = (
    ("Identity", (1, 195)),
    ("Identity_1", (1, 1)),
    ("Identity_2", (1, 256, 256, 1)),
    ("Identity_3", (1, 64, 64, 39)),
    ("Identity_4", (1, 117)),
)
OUTPUTS = (SOURCE_OUTPUTS[0], SOURCE_OUTPUTS[1], SOURCE_OUTPUTS[4])
CHANNEL_PADDING_COUNT = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_contract(values: list[onnx.ValueInfoProto]) -> list[tuple[str, tuple[int, ...]]]:
    return [(value.name, tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)) for value in values]


def validate_source_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    inputs = _tensor_contract(list(model.graph.input))
    outputs = _tensor_contract(list(model.graph.output))
    if inputs != [(INPUT_NAME, INPUT_SHAPE)]:
        raise ValueError(f"BlazePose source input mismatch: {inputs}")
    if outputs != list(SOURCE_OUTPUTS):
        raise ValueError(f"BlazePose source output mismatch: {outputs}")
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("") != 16:
        raise ValueError(f"BlazePose source ONNX opset mismatch: {opsets}")


def _replace_channel_padding(model: onnx.ModelProto) -> list[str]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    initializers = {item.name: numpy_helper.to_array(item) for item in inferred.graph.initializer}
    shapes = {
        item.name: tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim)
        for item in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }
    replaced: list[str] = []
    for index, node in enumerate(inferred.graph.node):
        if node.op_type != "Pad" or "/channel_padding" not in node.name:
            continue
        if len(node.input) != 2 or node.attribute:
            raise ValueError(f"unsupported BlazePose channel Pad form: {node.name}")
        input_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        if input_shape is None or output_shape is None or len(input_shape) != 4 or len(output_shape) != 4:
            raise ValueError(f"missing static BlazePose channel Pad shape: {node.name}")
        input_channels = input_shape[1]
        output_channels = output_shape[1]
        pads = initializers[node.input[1]].tolist()
        expected_pads = [0, 0, 0, 0, 0, output_channels - input_channels, 0, 0]
        if input_shape[0] != output_shape[0] or input_shape[2:] != output_shape[2:] or pads != expected_pads:
            raise ValueError(
                f"unexpected BlazePose channel Pad contract: {node.name}, {input_shape} -> {output_shape}, {pads}"
            )
        weights = np.zeros((output_channels, input_channels, 1, 1), dtype=np.float32)
        diagonal = np.arange(input_channels)
        weights[diagonal, diagonal, 0, 0] = 1.0
        weight_name = f"{node.name}/identity_conv_weight"
        inferred.graph.initializer.append(numpy_helper.from_array(weights, weight_name))
        inferred.graph.node[index].CopyFrom(
            helper.make_node(
                "Conv",
                [node.input[0], weight_name],
                list(node.output),
                name=f"{node.name}/identity_conv",
                kernel_shape=[1, 1],
                strides=[1, 1],
            )
        )
        replaced.append(node.name)
    if len(replaced) != CHANNEL_PADDING_COUNT:
        raise ValueError(f"expected {CHANNEL_PADDING_COUNT} BlazePose channel Pads, found {replaced}")

    used_initializers = {name for node in inferred.graph.node for name in node.input}
    for initializer in list(inferred.graph.initializer):
        if initializer.name not in used_initializers:
            inferred.graph.initializer.remove(initializer)
    model.CopyFrom(inferred)
    return replaced


def validate_prepared_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    inputs = _tensor_contract(list(model.graph.input))
    outputs = _tensor_contract(list(model.graph.output))
    if inputs != [(INPUT_NAME, INPUT_SHAPE)]:
        raise ValueError(f"prepared BlazePose input mismatch: {inputs}")
    if outputs != list(OUTPUTS):
        raise ValueError(f"prepared BlazePose output mismatch: {outputs}")
    remaining = [node.name for node in model.graph.node if "/channel_padding" in node.name and node.op_type == "Pad"]
    if remaining:
        raise ValueError(f"prepared BlazePose graph retains channel Pads: {remaining}")


def prepare_model(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(f"BlazePose source SHA-256 mismatch: {source}")
    model = onnx.load(source)
    validate_source_model(model)
    extracted = Extractor(shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)).extract_model(
        [INPUT_NAME], [name for name, _ in OUTPUTS]
    )
    replaced = _replace_channel_padding(extracted)
    extracted.graph.name = "blazepose_ghum_heavy_3d"
    extracted.ir_version = 8
    extracted.producer_name = "sima-neat-blazepose-surgery"
    extracted.producer_version = "1"
    del extracted.metadata_props[:]
    helper.set_model_props(
        extracted,
        {
            "sima.output_order": ",".join(name for name, _ in OUTPUTS),
            "sima.preprocess": "RGB float32 NHWC [0,1], resize 256x256",
            "sima.replaced_channel_padding": str(len(replaced)),
            "source.sha256": SOURCE_SHA256,
        },
    )
    extracted = shape_inference.infer_shapes(extracted, strict_mode=True, data_prop=True)
    validate_prepared_model(extracted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(extracted, destination)
    return destination


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_model(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
