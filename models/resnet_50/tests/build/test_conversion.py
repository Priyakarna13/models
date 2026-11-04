"""Check the real exported interface and the converter's PyTorch agreement evidence."""

import json

import numpy as np
import onnx


def test_exported_resnet_interface(converted):
    model = onnx.load(str(converted))
    onnx.checker.check_model(model)
    assert [value.name for value in model.graph.input] == ["pixel_values"]
    assert [value.name for value in model.graph.output] == ["logits"]
    for value, shape in zip([*model.graph.input, *model.graph.output], [[1, 3, 224, 224], [1, 1000]], strict=True):
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert [dim.dim_value for dim in value.type.tensor_type.shape.dim] == shape
    report = json.loads((converted.parent / "conversion-validation.json").read_text())
    assert report["status"] == "passed"
    assert [item["seed"] for item in report["comparisons"]] == [0, 1, 2]
    assert all(np.isfinite(item["max_absolute_error"]) for item in report["comparisons"])
