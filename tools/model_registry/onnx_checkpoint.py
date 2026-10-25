"""Export large ONNX graphs with lossless views into their pinned safetensors source.

The converted graph is the only generated source. Its original FP32 tensors remain in
an independently pinned checkpoint member of the source file set, including exact
transposed views. Derived constants stay inline. This avoids ONNX's 2 GiB protobuf
limit without rounding weights or publishing a second copy of the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from pathlib import Path
from typing import Any


def checkpoint_index(path: Path) -> dict[tuple[str, int], tuple[int, int, list[int], bool]]:
    import numpy as np

    index = {}
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        base = 8 + length
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for value in header.values():
                if not isinstance(value, dict) or value.get("dtype") != "F32":
                    continue
                start, stop = value["data_offsets"]
                shape = value["shape"]
                content = data[base + start : base + stop]
                index.setdefault(
                    (hashlib.sha256(content).hexdigest(), len(content)), (base + start, len(content), shape, False)
                )
                if len(shape) == 2 and min(shape) > 1:
                    transposed = np.frombuffer(content, dtype="<f4").reshape(shape).T.tobytes()
                    index.setdefault(
                        (hashlib.sha256(transposed).hexdigest(), len(transposed)),
                        (base + start, len(content), shape, True),
                    )
    return index


def bind_checkpoint(model: Any, checkpoint: Path, *, index: dict | None = None) -> dict[str, int]:
    import onnx
    from onnx import external_data_helper, helper

    index = checkpoint_index(checkpoint) if index is None else index
    original = list(model.graph.initializer)
    model.graph.ClearField("initializer")
    prefix_nodes = []
    matched = inline = 0
    names = {value.name for value in original}
    names.update(output for node in model.graph.node for output in node.output)
    for tensor in original:
        raw = tensor.raw_data
        match = (
            index.get((hashlib.sha256(raw).hexdigest(), len(raw)))
            if raw and tensor.data_type == onnx.TensorProto.FLOAT
            else None
        )
        if match is None:
            model.graph.initializer.append(tensor)
            inline += len(raw)
            continue
        offset, length, source_shape, transpose = match
        desired_name, desired_shape = tensor.name, list(tensor.dims)
        if transpose:
            stem = desired_name + "__checkpoint"
            while stem in names:
                stem += "_"
            names.add(stem)
            tensor.name = stem
            tensor.ClearField("dims")
            tensor.dims.extend(source_shape)
            intermediate = desired_name if desired_shape == list(reversed(source_shape)) else stem + "_transposed"
            prefix_nodes.append(helper.make_node("Transpose", [stem], [intermediate], perm=[1, 0]))
            if intermediate != desired_name:
                shape_name = stem + "_shape"
                model.graph.initializer.append(
                    helper.make_tensor(shape_name, onnx.TensorProto.INT64, [len(desired_shape)], desired_shape)
                )
                prefix_nodes.append(helper.make_node("Reshape", [intermediate, shape_name], [desired_name]))
        external_data_helper.set_external_data(tensor, checkpoint.name, offset, length)
        tensor.ClearField("raw_data")
        model.graph.initializer.append(tensor)
        matched += length
    nodes = list(model.graph.node)
    model.graph.ClearField("node")
    model.graph.node.extend([*prefix_nodes, *nodes])
    return {"checkpoint_bytes": matched, "inline_bytes": inline, "constant_view_nodes": len(prefix_nodes)}


def append_graph(destination: Any, graph: Any, *, prefix: str, replacements: dict[str, str] | None = None) -> Any:
    from onnx import compose

    part = compose.add_prefix(graph, prefix)
    mapping = replacements or {}
    for node in part.graph.node:
        for ports in (node.input, node.output):
            for index, key in enumerate(ports):
                ports[index] = mapping.get(key, key)
    for values in (part.graph.input, part.graph.output, part.graph.value_info):
        for value in values:
            value.name = mapping.get(value.name, value.name)
    destination.graph.node.extend(part.graph.node)
    destination.graph.initializer.extend(part.graph.initializer)
    destination.graph.value_info.extend(part.graph.value_info)
    return part.graph


def partition(model: Any, output_names: list[str]) -> Any:
    """Select a static graph region without loading or copying its external weight bytes."""
    from copy import deepcopy

    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    needed = set(output_names)
    selected = set()
    todo = list(output_names)
    while todo:
        key = todo.pop()
        node = producers.get(key)
        if node is None or id(node) in selected:
            continue
        selected.add(id(node))
        for name in node.input:
            if name and name not in needed:
                needed.add(name)
                todo.append(name)
    result = deepcopy(model)
    for field, values in (
        ("node", [node for node in model.graph.node if id(node) in selected]),
        ("initializer", [value for name, value in initializers.items() if name in needed]),
        ("input", [value for value in model.graph.input if value.name in needed]),
        ("output", [value for value in model.graph.output if value.name in output_names]),
        ("value_info", [value for value in model.graph.value_info if value.name in needed]),
    ):
        result.graph.ClearField(field)
        getattr(result.graph, field).extend(values)
    if {value.name for value in result.graph.output} != set(output_names):
        raise ValueError("Partition outputs must be declared graph outputs")
    return result


def export_part(module: Any, arguments: tuple, inputs: list[str], outputs: list[str], path: Path) -> Any:
    """Export one bounded part before composing a model larger than 2 GiB."""
    import onnx
    import torch
    from onnxsim import simplify

    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            arguments,
            str(path),
            input_names=inputs,
            output_names=outputs,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        expected = module(*arguments)
    graph = onnx.load(path)
    values = (expected,) if isinstance(expected, torch.Tensor) else expected
    for output, value in zip(graph.graph.output, values, strict=True):
        output.type.tensor_type.shape.ClearField("dim")
        for size in value.shape:
            output.type.tensor_type.shape.dim.add().dim_value = int(size)
    graph.ir_version = 8
    graph, valid = simplify(graph)
    if not valid:
        raise ValueError("ONNX simplification did not preserve the exported part")
    return graph


def frontend_metadata(processor: Any) -> dict:
    import base64

    import numpy as np
    import torch

    filters = np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32)
    window = torch.hann_window(400, periodic=False).numpy().astype(np.float32)
    return {
        name: {"shape": list(value.shape), "data": base64.b64encode(value.tobytes()).decode("ascii")}
        for name, value in (("mel_filters", filters), ("window", window))
    }
