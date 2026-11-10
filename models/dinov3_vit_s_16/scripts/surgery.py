"""Prepare the converted DINOv3 ViT-S/16 graph for the Model Compiler, and check that it computes the same embeddings.

The conversion is onnx-community's opset-20 export with its input fixed at 1x3x224x224. For the compiler, constant
folding turns the RoPE cos/sin computation into tables for the 14x14 patch grid. Each layer still splits the five
prefix tokens (class and registers) from the 196 patch tokens with SplitToSequence and SequenceAt; those become one
Split with the same constant sizes, after which the reshape targets fold too. Each exact Gelu becomes its erf form, so
the graph needs only opset 17. The outputs become the pooled class embedding and the patch tokens; the four register
tokens, which DINOv3 uses only inside the encoder, are dropped. Each step runs in its own process, so the family's
largest graph, ViT-L/16, folds within 9 GiB. ONNX Runtime then checks the prepared graph against the converted one.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MODEL_ID = "dinov3_vit_s_16"
INPUT = "pixel_values"
INPUT_SHAPE = (1, 3, 224, 224)
HIDDEN_SIZE = 384
PATCHES = 196
# The class token and the four register tokens come before the patch tokens.
PREFIX = 5
SOURCE_OUTPUTS = ("pooler_output", "last_hidden_state")
OUTPUT_SHAPES = {"class_embedding": (1, HIDDEN_SIZE), "patch_embeddings": (1, PATCHES, HIDDEN_SIZE)}
OUTPUTS = tuple(OUTPUT_SHAPES)
OPSET = 17
SEEDS = range(123, 127)
# Operators the prepared graph must not keep: dynamic shapes, control flow, sequences, and opset-20 Gelu.
DYNAMIC_OPS = ("If", "Loop", "Shape", "Range", "SplitToSequence", "SequenceAt", "Gelu")


def fold_constants(source: Path, output: Path) -> None:
    """Replace each tensor the input's values cannot change with an initializer, and drop the nodes that computed it.

    The input's shape is fixed, so every Shape output, and each tensor computed only from shapes and initializers, is
    a constant. ONNX Runtime computes them once on a zero input with its graph optimizations off, so the folding does
    not depend on which onnxsim an environment ships; the compiler's ships one too old to fold this graph.
    """
    import onnxruntime as ort

    model = onnx.load(str(source))
    graph = model.graph
    constants = {tensor.name for tensor in graph.initializer}
    folded, kept = [], []
    for node in graph.node:
        if node.op_type in ("If", "Loop", "Scan"):
            raise ValueError(f"{node.name} is control flow, which the folding does not evaluate")
        if node.op_type == "Shape" or all(name in constants for name in node.input if name):
            constants.update(name for name in node.output if name)
            folded.append(node)
        else:
            kept.append(node)
    read = {name for node in kept for name in node.input} | {value.name for value in graph.output}
    targets = [name for node in folded for name in node.output if name in read]
    if targets:
        graph.output.extend(onnx.ValueInfoProto(name=name) for name in targets)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        values = session.run(targets, {INPUT: np.zeros(INPUT_SHAPE, np.float32)})
        del session, graph.output[-len(targets):]
        graph.initializer.extend(
            numpy_helper.from_array(value, name) for name, value in zip(targets, values, strict=True)
        )
    graph.ClearField("node")
    graph.node.extend(kept)
    used = {name for node in kept for name in node.input} | {value.name for value in graph.output}
    # Deleted in place: copying the kept weights would double ViT-L/16's 1.2 GB.
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]
    graph.ClearField("value_info")
    onnx.save(model, str(output))


def _constant(graph: onnx.GraphProto, name: str) -> np.ndarray:
    for tensor in graph.initializer:
        if tensor.name == name:
            return numpy_helper.to_array(tensor)
    raise ValueError(f"{name} is not a constant; fold the graph first")


def _rename_uses(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        node.input[:] = [new if name == old else name for name in node.input]


def split_prefix_tokens(graph: onnx.GraphProto) -> None:
    """Replace each SplitToSequence and the SequenceAt nodes that read it with one Split of the same sizes."""
    parts, renamed = {}, {}
    for node in graph.node:
        if node.op_type == "SplitToSequence":
            sizes = _constant(graph, node.input[1]).reshape(-1)
            parts[node.output[0]] = [f"{node.output[0]}_part{index}" for index in range(len(sizes))]
        elif node.op_type == "SequenceAt":
            renamed[node.output[0]] = parts[node.input[0]][int(_constant(graph, node.input[1]))]
        elif any(name in parts for name in node.input):
            raise ValueError(f"{node.name} reads a token sequence; only SequenceAt is rewritten")
    nodes = []
    for node in graph.node:
        if node.op_type == "SequenceAt":
            continue
        if node.op_type == "SplitToSequence":
            axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
            node = helper.make_node(
                "Split", node.input[:2], parts[node.output[0]], name=f"{node.name}_split", axis=axis
            )
        node.input[:] = [renamed.get(name, name) for name in node.input]
        nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(nodes)


def gelu_as_erf(graph: onnx.GraphProto) -> None:
    """Rewrite each exact Gelu as 0.5 * x * (1 + erf(x / sqrt(2))), which opset 17 provides."""
    graph.initializer.extend([
        numpy_helper.from_array(np.array(np.sqrt(2.0), dtype=np.float32), "gelu_sqrt2"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "gelu_one"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "gelu_half"),
    ])
    nodes = []
    for node in graph.node:
        if node.op_type != "Gelu":
            nodes.append(node)
            continue
        if any(a.name == "approximate" and a.s not in (b"", b"none") for a in node.attribute):
            raise ValueError(f"{node.name} is a tanh-approximated Gelu; only the exact form is rewritten")
        x, y, stem = node.input[0], node.output[0], node.name or node.output[0]
        nodes += [
            helper.make_node("Div", [x, "gelu_sqrt2"], [f"{stem}_scaled"], name=f"{stem}_div"),
            helper.make_node("Erf", [f"{stem}_scaled"], [f"{stem}_erf"], name=f"{stem}_erf"),
            helper.make_node("Add", [f"{stem}_erf", "gelu_one"], [f"{stem}_shifted"], name=f"{stem}_add"),
            helper.make_node("Mul", [x, f"{stem}_shifted"], [f"{stem}_gated"], name=f"{stem}_mul"),
            helper.make_node("Mul", [f"{stem}_gated", "gelu_half"], [y], name=f"{stem}_half"),
        ]
    graph.ClearField("node")
    graph.node.extend(nodes)


def select_outputs(graph: onnx.GraphProto) -> None:
    """Expose the pooled class embedding and the patch tokens, without the class and register tokens."""
    pooled, hidden = SOURCE_OUTPUTS
    for node in graph.node:
        node.output[:] = [OUTPUTS[0] if name == pooled else name for name in node.output]
    _rename_uses(graph, pooled, OUTPUTS[0])
    graph.initializer.extend([
        numpy_helper.from_array(np.array([PREFIX], dtype=np.int64), "patch_start"),
        numpy_helper.from_array(np.array([PREFIX + PATCHES], dtype=np.int64), "patch_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "patch_axis"),
    ])
    graph.node.append(helper.make_node("Slice", [hidden, "patch_start", "patch_end", "patch_axis"], [OUTPUTS[1]],
                                       name="patch_tokens"))
    graph.ClearField("output")
    graph.output.extend(
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in OUTPUT_SHAPES.items()
    )


def rewrite(path: Path) -> None:
    """Make the folded graph static at opset 17 with the class and patch embeddings as its outputs."""
    model = onnx.load(str(path))
    split_prefix_tokens(model.graph)
    gelu_as_erf(model.graph)
    select_outputs(model.graph)
    model.ClearField("opset_import")
    model.opset_import.extend([helper.make_opsetid("", OPSET)])
    onnx.save(model, str(path))


def validate_prepared(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"The prepared DINOv3 graph requires opset {OPSET}")
    names = ([value.name for value in model.graph.input], [value.name for value in model.graph.output])
    if names != ([INPUT], list(OUTPUTS)):
        raise ValueError(f"Expected input {INPUT} and outputs {OUTPUTS} in that order")
    shapes = {INPUT: INPUT_SHAPE, **OUTPUT_SHAPES}
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = tuple(d.dim_value for d in tensor.shape.dim)
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != shapes[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    left = sorted({node.op_type for node in model.graph.node if node.op_type in DYNAMIC_OPS})
    if left:
        raise ValueError(f"The prepared graph must be static at opset {OPSET}; it still has {', '.join(left)}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("The prepared graph must be one self-contained ONNX file")


def embeddings(pooled: np.ndarray, hidden: np.ndarray) -> dict[str, np.ndarray]:
    """The class and patch embeddings in the converted graph's pooler_output and last_hidden_state."""
    return {OUTPUTS[0]: pooled, OUTPUTS[1]: hidden[:, PREFIX:]}


def _synthetic(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal(INPUT_SHAPE).astype(np.float32)


def check_equivalent(source: Path, prepared: Path) -> None:
    """Require the prepared graph's embeddings to match the converted graph's on seeded inputs."""
    import onnxruntime as ort

    converted = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    expected = [embeddings(*converted.run(list(SOURCE_OUTPUTS), {INPUT: _synthetic(seed)})) for seed in SEEDS]
    del converted
    session = ort.InferenceSession(str(prepared), providers=["CPUExecutionProvider"])
    for seed, want in zip(SEEDS, expected, strict=True):
        got = dict(zip(OUTPUTS, session.run(list(OUTPUTS), {INPUT: _synthetic(seed)}), strict=True))
        for name in OUTPUTS:
            np.testing.assert_allclose(got[name], want[name], rtol=1e-3, atol=1e-3, err_msg=name)
    print(f"Converted/prepared ONNX check passed on {len(SEEDS)} synthetic inputs: {prepared}", flush=True)


def _in_own_process(function, *args) -> None:
    process = multiprocessing.get_context("spawn").Process(target=function, args=args)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{function.__name__} failed with exit code {process.exitcode}")


def prepare_model(source: Path, output_dir: Path) -> Path:
    """Write the compiler's graph for the converted ONNX into output_dir and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = output_dir / f"{MODEL_ID}_prepared.onnx"
    _in_own_process(fold_constants, source, prepared)
    _in_own_process(rewrite, prepared)
    _in_own_process(fold_constants, prepared, prepared)
    validate_prepared(prepared)
    _in_own_process(check_equivalent, source, prepared)
    return prepared
