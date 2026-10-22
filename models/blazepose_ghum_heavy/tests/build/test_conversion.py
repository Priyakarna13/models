"""Small checks for conversion failures that one deterministic comparison input can miss."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def _outputs(value: float) -> dict[str, np.ndarray]:
    return {
        name: np.full(shape, value, dtype=np.float32)
        for name, shape in (("Identity", (1, 195)), ("Identity_1", (1, 1)), ("Identity_4", (1, 117)))
    }


def test_conversion_gate_accepts_qualified_deltas_and_rejects_regression(converter: ModuleType) -> None:
    reference = _outputs(1.0)
    candidate = {name: value + np.float32(0.0001) for name, value in reference.items()}
    assert set(converter.compare_outputs(reference, candidate)) == {"Identity", "Identity_1", "Identity_4"}
    candidate["Identity"][0, 0] = 2.0
    with pytest.raises(ValueError, match="exceeds tolerance"):
        converter.compare_outputs(reference, candidate)


@pytest.mark.parametrize("case", ["missing", "shape", "nan"])
def test_conversion_gate_rejects_a_changed_interface(case: str, converter: ModuleType) -> None:
    reference = _outputs(1.0)
    candidate = _outputs(1.0)
    if case == "missing":
        del candidate["Identity_4"]
        match = "output names mismatch"
    elif case == "shape":
        candidate["Identity"] = candidate["Identity"][:, :-1]
        match = "shape mismatch"
    else:
        candidate["Identity_1"][0, 0] = np.nan
        match = "non-finite"
    with pytest.raises(ValueError, match=match):
        converter.compare_outputs(reference, candidate)


def _source_interface_model(converter: ModuleType, opset: int) -> onnx.ModelProto:
    helper = onnx.helper
    surgery_outputs = converter.validate_source_model.__globals__["SOURCE_OUTPUTS"]
    inputs = [helper.make_tensor_value_info(converter.INPUT_NAME, onnx.TensorProto.FLOAT, converter.INPUT_SHAPE)]
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in surgery_outputs]
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name, onnx.TensorProto.FLOAT, shape, np.zeros(int(np.prod(shape)))),
        )
        for name, shape in surgery_outputs
    ]
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("opset", [16, 13])
def test_export_must_keep_the_tf2onnx_interface(tmp_path: Path, opset: int, converter: ModuleType) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_source_interface_model(converter, opset), path)
    if opset == converter.OPSET:
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError, match="opset"):
            converter.validate_export(path)


def test_comparison_input_is_deterministic_rgb_in_unit_range(converter: ModuleType) -> None:
    sample = converter.deterministic_input()
    assert sample.shape == converter.INPUT_SHAPE and sample.dtype == np.float32
    assert 0.0 <= float(sample.min()) and float(sample.max()) <= 1.0
    np.testing.assert_array_equal(sample, converter.deterministic_input())


def test_source_must_be_the_pinned_tflite(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/blazepose_ghum_heavy")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="pose_landmark_heavy.tflite"):
        converter.validate_source({"type": "file", "uri": "hf://models/Pictorial/mediapipe@main/pose_landmark_full.tflite"})
