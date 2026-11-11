"""Small checks for conversion failures that a synthetic smoke input can miss."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.mark.parametrize("case", ["valid", "missing", "shape", "nan", "world_landmarks"])
def test_output_agreement(case, converter: ModuleType):
    rng = np.random.RandomState(0)
    expected = {
        name: rng.uniform(-1, 1, shape).astype(np.float32) for name, shape in list(converter.INTERFACE.items())[1:]
    }
    actual = {name: value + 1e-6 for name, value in expected.items()}
    if case == "missing":
        del actual["Identity_4"]
    elif case == "shape":
        actual["Identity"] = actual["Identity"][:, :-1]
    elif case == "nan":
        actual["Identity_1"][0, 0] = np.nan
    elif case == "world_landmarks":
        # One millimetre off in one world coordinate.
        actual["Identity_4"][0, 0] += 1e-3
    else:
        assert max(converter.compare_outputs(expected, actual).values()) < 2e-6
        return
    with pytest.raises(ValueError):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    nodes = [
        helper.make_node("Constant", [], [value.name], value=onnx.numpy_helper.from_array(np.zeros(shape, np.float32)))
        for value, shape in zip(values[1:], list(converter.INTERFACE.values())[1:])
    ]
    if case == "order":
        values[1], values[2] = values[2], values[1]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[0].type.tensor_type.shape.dim[3].dim_value = 4
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13 if case == "opset" else 16)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_tflite(model_dir: Path, converter: ModuleType):
    converter.validate_source(load_metadata(model_dir)["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="pose_landmark_heavy.tflite"):
        converter.validate_source(
            {
                "type": "file",
                "uri": "hf://models/Pictorial/mediapipe@main/modules/pose_landmark/pose_landmark_full.tflite",
            }
        )
