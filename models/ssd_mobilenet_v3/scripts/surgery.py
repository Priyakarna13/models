"""Validate the published raw FP32 source; no graph surgery or input affine."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

import onnx

SOURCE_SHA256 = "bdd8f4ef575bdc0cb02d33e571992d3c45c63f0bc00aeca5f51583c50e811bd8"
INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, 320, 320)
FEATURE_SIZES = (20, 10, 5, 3, 2, 1)
OUTPUTS = tuple(
    (f"ssd_{role}_{level}", (1, 6 * columns, size, size))
    for role, columns in (("loc", 4), ("confidence", 91))
    for level, size in enumerate(FEATURE_SIZES)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path: Path) -> dict:
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    operators = Counter(node.op_type for node in graph.graph.node)
    if (
        operators["Conv"] != 98
        or operators["BatchNormalization"]
        or operators["QuantizeLinear"]
        or operators["DequantizeLinear"]
    ):
        raise ValueError(f"unexpected FP32 operator contract: {operators}")
    if [(item.domain, item.version) for item in graph.opset_import] != [("", 17)]:
        raise ValueError("unexpected ONNX opset")

    def contract(values):
        return [
            (
                item.name,
                tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim),
                item.type.tensor_type.elem_type,
            )
            for item in values
        ]

    if contract(graph.graph.input) != [(INPUT_NAME, INPUT_SHAPE, onnx.TensorProto.FLOAT)]:
        raise ValueError("unexpected public input contract")
    if contract(graph.graph.output) != [(name, shape, onnx.TensorProto.FLOAT) for name, shape in OUTPUTS]:
        raise ValueError("unexpected ordered public output contract")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("source must be one self-contained ONNX file")
    return {
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "operators": dict(sorted(operators.items())),
    }


def prepare_model(source: Path, output: Path) -> Path:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("published V3 source SHA-256 mismatch")
    audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return output
