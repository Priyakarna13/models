"""Extract MobileSAM's source components and the compiled decoder's CPU Fourier boundary.

The existing source already contains the 4D attention rewrite. Retaining that immutable source avoids moving
Torch/segment-anything dependencies into the Model Compiler environment without a supported dependency contract.
"""

from pathlib import Path

import onnx
from pipeline import (
    COMPILED_INPUT_SHAPES,
    COMPONENTS,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
    PROMPT_ENCODING_METADATA,
    PromptEncoding,
    tensor_name,
)


def split_components(source: Path, directory: Path) -> dict[str, Path]:
    """Extract each component's graph from the side-by-side ONNX."""
    directory.mkdir(parents=True)
    graphs = {}
    for component in COMPONENTS:
        graphs[component] = directory / f"{component}.onnx"
        onnx.utils.extract_model(
            str(source),
            str(graphs[component]),
            [tensor_name(component, name) for name in INPUT_SHAPES[component]],
            [tensor_name(component, name) for name in OUTPUT_SHAPES[component]],
        )
    return graphs


def compile_decoder(source: Path, output: Path, source_sha256: str) -> dict:
    """Cut only source Fourier encoding; retain learned prompt types, masking and attention on MLA.

    The exact FP32 table travels in a checksum-bound package supporting file, not a runtime checkpoint download.
    The source decoder remains untouched for reference inference; the compiled decoder has a different input.
    """
    import hashlib

    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(source)
    encoded = "/prompt_decoder/Concat_output_0"
    prefix = onnx.utils.Extractor(model).extract_model(["prompt_decoder_point_coords"], [encoded])
    nodes = list(prefix.graph.node)
    if [node.op_type for node in nodes] != ["Gather", "Gather", "Transpose", "MatMul", "Mul", "Sin", "Cos", "Concat"]:
        raise ValueError("unsupported source Fourier encoding graph")
    constants = {item.name: numpy_helper.to_array(item) for item in prefix.graph.initializer}
    attributes = lambda node: {item.name: helper.get_attribute_value(item) for item in node.attribute}  # noqa: E731
    expected_inputs = [
        ["prompt_decoder_point_coords", nodes[0].input[1]],
        [nodes[0].output[0], nodes[1].input[1]],
        [nodes[1].output[0]],
        [nodes[2].output[0], nodes[3].input[1]],
        [nodes[3].output[0], nodes[4].input[1]],
        [nodes[4].output[0]],
        [nodes[4].output[0]],
        [nodes[5].output[0], nodes[6].output[0]],
    ]
    if (
        [list(node.input) for node in nodes] != expected_inputs
        or [attributes(node) for node in nodes]
        != [{"axis": 0}, {"axis": 1}, {"perm": [1, 0]}, {}, {}, {}, {}, {"axis": -1}]
        or any(
            constants[nodes[index].input[1]].shape != () or constants[nodes[index].input[1]] != 0 for index in (0, 1)
        )
    ):
        raise ValueError("unsupported source Fourier encoding operation order")
    gaussian, scale = constants[nodes[3].input[1]], constants[nodes[4].input[1]]
    if gaussian.dtype != np.float32 or scale.dtype != np.float32:
        raise ValueError("source Fourier coefficients must be float32")
    metadata = {
        PROMPT_ENCODING_METADATA: {
            "version": 1,
            "source_sha256": source_sha256,
            "gaussian": gaussian.tolist(),
            "gaussian_sha256": hashlib.sha256(gaussian.astype("<f4").tobytes()).hexdigest(),
            "phase_scale": float(scale),
        }
    }
    import json

    PromptEncoding.from_metadata({PROMPT_ENCODING_METADATA: json.dumps(metadata[PROMPT_ENCODING_METADATA])})
    cut = onnx.utils.Extractor(model).extract_model(
        ["prompt_decoder_image_embeddings", encoded, "prompt_decoder_point_types"],
        [tensor_name("prompt_decoder", name) for name in OUTPUT_SHAPES["prompt_decoder"]],
    )
    shape = COMPILED_INPUT_SHAPES["prompt_decoder"]["point_encoding"]
    cut.graph.input[1].CopyFrom(
        helper.make_tensor_value_info("prompt_decoder_point_encoding", TensorProto.FLOAT, shape)
    )
    shape_name, transposed = "mobile_sam_encoding_shape", "mobile_sam_encoding_transposed"
    occupied = {name for node in cut.graph.node for name in [*node.input, *node.output]}
    if {shape_name, transposed} & occupied:
        raise ValueError("compiled Fourier boundary tensor name collision")
    cut.graph.initializer.append(numpy_helper.from_array(np.array([shape[3], shape[1]], np.int64), shape_name))
    boundary = [
        helper.make_node("Transpose", ["prompt_decoder_point_encoding"], [transposed], perm=[0, 3, 2, 1]),
        helper.make_node("Reshape", [transposed, shape_name], [encoded]),
    ]
    original = list(cut.graph.node)
    del cut.graph.node[:]
    cut.graph.node.extend(boundary + original)
    if any(node.op_type in ("Sin", "Cos") for node in cut.graph.node):
        raise ValueError("compiled decoder still contains Fourier lookup operations")
    onnx.checker.check_model(cut)
    onnx.save(cut, output)
    return metadata
