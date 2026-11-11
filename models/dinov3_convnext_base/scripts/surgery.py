"""Prepare the converted DINOv3 ConvNeXt Base graph for the Model Compiler, and check it computes the same embeddings.

The conversion is timm's network as it is, with each exact GELU exported as x * 0.5 * (1 + erf(x / sqrt(2))). The
compiler builds each erf from an INT8 lookup table, so every GELU becomes its tanh form,
x * sigmoid(x * (a + b * x^2)), which stays within 1e-4 of the exact GELU and compiles to native BF16 operations.
Nothing else changes. ONNX Runtime then checks the prepared graph against the converted one.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT = "pixel_values"
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUTS = ("class_embedding", "patch_embeddings")
OPSET = 17
SEEDS = range(123, 127)
# The tanh GELU stays within 1e-4 of the exact one, so the embeddings keep their direction.
MIN_COSINE = 0.9999
# tanh(sqrt(2/pi) * (x + 0.044715 x^3)) = 2 * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)) - 1.
GELU_LINEAR = 2.0 * math.sqrt(2.0 / math.pi)
GELU_CUBIC = GELU_LINEAR * 0.044715


def _constants(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    values = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    return values


def _scalar(constants: dict[str, np.ndarray], name: str) -> float | None:
    value = constants.get(name)
    return float(value) if value is not None and value.size == 1 else None


def _other(node: onnx.NodeProto, name: str) -> str:
    """The input of a two-input node that is not name."""
    if list(node.input).count(name) != 1 or len(node.input) != 2:
        raise ValueError(f"{node.name} does not read {name} once beside one other input")
    return node.input[1] if node.input[0] == name else node.input[0]


def gelu_as_tanh(graph: onnx.GraphProto) -> int:
    """Rewrite each exact GELU, Div(x, sqrt 2) -> Erf -> Add 1 -> Mul x -> Mul 0.5, as x * sigmoid(x * (a + b x^2)).

    Returns how many GELUs it rewrote. Any Erf outside that pattern stops the rewrite.
    """
    constants = _constants(graph)
    producers = {output: node for node in graph.node for output in node.output}
    readers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node)

    def only_reader(name: str, op_type: str) -> onnx.NodeProto:
        nodes = readers.get(name, [])
        if len(nodes) != 1 or nodes[0].op_type != op_type:
            raise ValueError(f"{name} must feed exactly one {op_type}, not {[node.op_type for node in nodes]}")
        return nodes[0]

    removed, added, rewritten = set(), {}, 0
    for erf in [node for node in graph.node if node.op_type == "Erf"]:
        divide = producers.get(erf.input[0])
        if (
            divide is None
            or divide.op_type != "Div"
            or not math.isclose(_scalar(constants, divide.input[1]) or 0.0, math.sqrt(2.0), rel_tol=1e-6)
        ):
            raise ValueError(f"{erf.name} is not the erf of an exact GELU: it does not read x / sqrt(2)")
        x = divide.input[0]
        add = only_reader(erf.output[0], "Add")
        if not math.isclose(_scalar(constants, _other(add, erf.output[0])) or 0.0, 1.0):
            raise ValueError(f"{add.name} must add 1 to the erf")
        times_x = only_reader(add.output[0], "Mul")
        if _other(times_x, add.output[0]) != x:
            raise ValueError(f"{times_x.name} must multiply the GELU's own input")
        half = only_reader(times_x.output[0], "Mul")
        if not math.isclose(_scalar(constants, _other(half, times_x.output[0])) or 0.0, 0.5):
            raise ValueError(f"{half.name} must halve the product")
        if len(readers.get(divide.output[0], [])) != 1:
            raise ValueError(f"{divide.name} feeds more than the erf")

        prefix = f"{erf.name}_tanh_gelu"
        parts = ("square", "cubic", "inner", "scaled", "gate")
        square, cubic, inner, scaled, gate = (f"{prefix}/{part}" for part in parts)
        nodes = [
            helper.make_node("Mul", [x, x], [square], name=f"{prefix}/Square"),
            helper.make_node("Mul", [square, f"{prefix}/b"], [cubic], name=f"{prefix}/Cubic"),
            helper.make_node("Add", [cubic, f"{prefix}/a"], [inner], name=f"{prefix}/Inner"),
            helper.make_node("Mul", [x, inner], [scaled], name=f"{prefix}/Scaled"),
            helper.make_node("Sigmoid", [scaled], [gate], name=f"{prefix}/Gate"),
            helper.make_node("Mul", [x, gate], [half.output[0]], name=f"{prefix}/Output"),
        ]
        graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(GELU_CUBIC, np.float32), f"{prefix}/b"),
                numpy_helper.from_array(np.asarray(GELU_LINEAR, np.float32), f"{prefix}/a"),
            ]
        )
        removed.update(id(node) for node in (divide, erf, add, times_x, half))
        # The replacement takes the place of the GELU's last node, so the graph stays topologically sorted.
        added[id(half)] = nodes
        rewritten += 1

    kept = []
    for node in graph.node:
        if id(node) in added:
            kept.extend(added[id(node)])
        elif id(node) not in removed:
            kept.append(node)
    used = {name for node in kept for name in node.input} | {value.name for value in graph.output}
    # The removed GELUs leave their sqrt(2), 1 and 0.5 constants unread.
    kept = [node for node in kept if node.op_type != "Constant" or node.output[0] in used]
    graph.ClearField("node")
    graph.node.extend(kept)
    used = {name for node in graph.node for name in node.input}
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]
    return rewritten


def validate_prepared(path: Path) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"The prepared graph requires opset {OPSET}")
    if [value.name for value in model.graph.input] != [INPUT] or [value.name for value in model.graph.output] != list(
        OUTPUTS
    ):
        raise ValueError(f"Expected input {INPUT} and outputs {OUTPUTS} in that order")
    if any(node.op_type == "Erf" for node in model.graph.node):
        raise ValueError("The prepared graph still holds an exact GELU")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_equivalent(source: Path, prepared: Path) -> None:
    """Require the prepared graph's embeddings to match the converted graph's on seeded inputs."""
    import onnxruntime as ort

    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (source, prepared)]
    for seed in SEEDS:
        image = np.random.RandomState(seed).standard_normal(INPUT_SHAPE).astype(np.float32)
        expected, actual = (session.run(list(OUTPUTS), {INPUT: image}) for session in sessions)
        if not all(np.isfinite(value).all() for value in actual):
            raise ValueError(f"The prepared graph returns non-finite values for seed {seed}")
        worst = min(_cosine(e, a) for e, a in zip(expected, actual, strict=True))
        print(f"seed {seed}: prepared vs converted cosine {worst:.8f}", flush=True)
        if worst < MIN_COSINE:
            raise ValueError(f"Seed {seed}: the prepared graph's cosine {worst:.6f} is below {MIN_COSINE}")


def prepare_model(source: Path, output_dir: Path) -> Path:
    """Write the compiler's graph for the converted ONNX into output_dir and return its path."""
    model = onnx.load(str(source))
    rewritten = gelu_as_tanh(model.graph)
    if rewritten == 0:
        raise ValueError("The converted graph has no exact GELU to rewrite")
    prepared = output_dir / "prepared.onnx"
    onnx.save(model, str(prepared))
    validate_prepared(prepared)
    check_equivalent(source, prepared)
    print(f"Prepared {prepared}: {rewritten} GELUs in tanh form", flush=True)
    return prepared
