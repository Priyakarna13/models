#!/usr/bin/env python3
"""Create the static SuperPoint dense-head graph consumed by MLA."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference

INPUT_NAME = "image"
INPUT_SHAPE = (1, 1, 480, 640)
HEADS = {
    "/convPb/Conv": ("/convPb/Conv_output_0", (1, 65, 60, 80)),
    "/convDb/Conv": ("/convDb/Conv_output_0", (1, 256, 60, 80)),
}


def _set_shape(value: onnx.ValueInfoProto, shape: Sequence[int]) -> None:
    del value.type.tensor_type.shape.dim[:]
    for extent in shape:
        value.type.tensor_type.shape.dim.add().dim_value = int(extent)


def _rename_tensor(model: onnx.ModelProto, old: str, new: str) -> None:
    if old == new:
        return
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name == old:
            value.name = new
    for initializer in model.graph.initializer:
        if initializer.name == old:
            initializer.name = new
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new


def _infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        return shape_inference.infer_shapes(model, strict_mode=False)


def _extract_ancestors(model: onnx.ModelProto, outputs: Sequence[str]) -> onnx.ModelProto:
    producers = {
        tensor: (index, node) for index, node in enumerate(model.graph.node) for tensor in node.output if tensor
    }
    roots = {value.name for value in model.graph.input}
    roots.update(value.name for value in model.graph.initializer)

    required_nodes: set[int] = set()
    pending = list(outputs)
    while pending:
        tensor = pending.pop()
        if tensor in roots:
            continue
        if tensor not in producers:
            raise ValueError(f"SuperPoint output ancestor has no producer: {tensor}")
        index, node = producers[tensor]
        if index in required_nodes:
            continue
        required_nodes.add(index)
        pending.extend(name for name in node.input if name)

    nodes = [copy.deepcopy(node) for index, node in enumerate(model.graph.node) if index in required_nodes]
    node_inputs = {name for node in nodes for name in node.input if name}
    inputs = [copy.deepcopy(value) for value in model.graph.input if value.name in node_inputs]
    initializers = [copy.deepcopy(value) for value in model.graph.initializer if value.name in node_inputs]
    public_outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in HEADS.values()]

    present = {name for node in nodes for name in (*node.input, *node.output) if name}
    public_names = {value.name for value in (*inputs, *public_outputs)}
    value_info = [
        copy.deepcopy(value)
        for value in model.graph.value_info
        if value.name in present and value.name not in public_names
    ]
    graph = helper.make_graph(
        nodes,
        "superpoint_dense_core_480x640",
        inputs,
        public_outputs,
        initializer=initializers,
        value_info=value_info,
    )
    result = helper.make_model(
        graph,
        opset_imports=copy.deepcopy(model.opset_import),
        producer_name="sima_superpoint_surgery",
    )
    result.ir_version = model.ir_version
    # The dense core uses only operators supported by opset 16. Pinning it keeps
    # the compiler input aligned with the qualified engineering graph.
    for opset in result.opset_import:
        if not opset.domain:
            opset.version = min(opset.version, 16)
    return _infer_shapes(result)


def validate_dense_core(model: onnx.ModelProto) -> None:
    inputs = list(model.graph.input)
    if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
        raise ValueError(f"expected one input named {INPUT_NAME!r}")
    if inputs[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("SuperPoint source input must be float32")
    input_shape = tuple(dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim)
    if input_shape != INPUT_SHAPE:
        raise ValueError(f"unexpected SuperPoint input shape: {input_shape}")

    output_contract = [(name, shape) for name, shape in HEADS.values()]
    outputs = []
    for value in model.graph.output:
        shape = tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        outputs.append((value.name, shape))
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError(f"SuperPoint output {value.name!r} must be float32")
    if outputs != output_contract:
        raise ValueError(f"unexpected SuperPoint outputs: {outputs}")

    if len(model.graph.node) != 25:
        raise ValueError(f"expected 25 dense-core nodes, found {len(model.graph.node)}")
    if {node.op_type for node in model.graph.node} - {"Conv", "MaxPool", "Relu"}:
        raise ValueError("SuperPoint surgery left unsupported postprocessing operators in the graph")
    onnx.checker.check_model(model)


def prepare_model(source: Path, output: Path) -> Path:
    model = onnx.load(str(source), load_external_data=False)
    graph_inputs = [value for value in model.graph.input if value.name not in {i.name for i in model.graph.initializer}]
    if len(graph_inputs) != 1:
        raise ValueError(f"expected exactly one SuperPoint graph input, found {len(graph_inputs)}")
    _rename_tensor(model, graph_inputs[0].name, INPUT_NAME)
    _set_shape(model.graph.input[0], INPUT_SHAPE)

    nodes_by_name = {node.name: node for node in model.graph.node}
    for node_name, (output_name, _) in HEADS.items():
        node = nodes_by_name.get(node_name)
        if node is None or node.op_type != "Conv" or len(node.output) != 1:
            raise ValueError(f"required SuperPoint head was not found: {node_name}")
        _rename_tensor(model, node.output[0], output_name)

    result = _extract_ancestors(_infer_shapes(model), [name for name, _ in HEADS.values()])
    validate_dense_core(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, str(output))
    return output
