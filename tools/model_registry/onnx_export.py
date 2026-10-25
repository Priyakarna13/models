"""Bounded PyTorch exports and standard ONNX graph composition/extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    """Extract a region with ONNX, retaining only its used public inputs."""
    from onnx.utils import Extractor

    if not set(output_names) <= {value.name for value in model.graph.output}:
        raise ValueError("Partition outputs must be declared graph outputs")
    result = Extractor(model).extract_model([value.name for value in model.graph.input], output_names)
    used = {name for node in result.graph.node for name in node.input} | set(output_names)
    inputs = [value for value in result.graph.input if value.name in used]
    result.graph.ClearField("input")
    result.graph.input.extend(inputs)
    result.metadata_props.extend(model.metadata_props)
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
    graph = onnx.load(path)
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
