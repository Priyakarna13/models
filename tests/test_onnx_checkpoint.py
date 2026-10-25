"""Lossless checkpoint views and independent CPU/encoder graph partitioning."""

import json
import struct

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from model_registry.onnx_checkpoint import bind_checkpoint, partition
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def test_checkpoint_views_preserve_values_and_partition_dependencies(tmp_path):
    weights = np.arange(12, dtype=np.float32).reshape(3, 4)
    content = weights.tobytes()
    header = json.dumps({"weights": dict(dtype="F32", shape=[3, 4], data_offsets=[0, len(content)])}).encode()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + content)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["encoder_input", "direct"], ["encoder_output"]),
                helper.make_node("MatMul", ["predictor_input", "transposed"], ["predictor_output"]),
                helper.make_node("Identity", ["reshaped"], ["reshape_output"]),
            ],
            "parts",
            [
                helper.make_tensor_value_info("encoder_input", TensorProto.FLOAT, [1, 3]),
                helper.make_tensor_value_info("predictor_input", TensorProto.FLOAT, [1, 4]),
            ],
            [
                helper.make_tensor_value_info("encoder_output", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("predictor_output", TensorProto.FLOAT, [1, 3]),
                helper.make_tensor_value_info("reshape_output", TensorProto.FLOAT, [2, 6]),
            ],
            initializer=[
                numpy_helper.from_array(value, name)
                for name, value in (
                    ("direct", weights),
                    ("transposed", weights.T.copy()),
                    ("reshaped", weights.T.copy().reshape(2, 6)),
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    expected = ReferenceEvaluator(model).run(
        None, {"encoder_input": np.ones((1, 3), np.float32), "predictor_input": np.ones((1, 4), np.float32)}
    )
    stats = bind_checkpoint(model, checkpoint)
    assert stats["checkpoint_bytes"] == len(content) * 3
    assert stats["inline_bytes"] == 0
    assert stats["constant_view_nodes"] == 3
    encoder = partition(model, ["encoder_output"])
    assert [v.name for v in encoder.graph.input] == ["encoder_input"]
    assert [n.op_type for n in encoder.graph.node] == ["MatMul"]
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path))
    actual = ReferenceEvaluator(onnx.load(path)).run(
        None, {"encoder_input": np.ones((1, 3), np.float32), "predictor_input": np.ones((1, 4), np.float32)}
    )
    for reference, output in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, output)
    with pytest.raises(ValueError, match="declared"):
        partition(model, ["unknown"])
