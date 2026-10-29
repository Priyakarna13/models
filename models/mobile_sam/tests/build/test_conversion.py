"""Small checks for conversion failures that a synthetic smoke input can miss."""

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


def _interface(converter, case):
    helper = onnx.helper
    inputs = [
        helper.make_tensor_value_info(converter.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in converter.COMPONENTS
        for n, s in converter.INPUT_SHAPES[c].items()
    ]
    outputs = [
        helper.make_tensor_value_info(converter.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in converter.COMPONENTS
        for n, s in converter.OUTPUT_SHAPES[c].items()
    ]
    if case == "dtype":
        inputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        inputs[2].type.tensor_type.shape.dim[3].dim_value = 6
    elif case == "missing":
        outputs.pop()
    nodes = []
    for value in outputs:
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor = helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(int(np.prod(shape))))
        nodes.append(helper.make_node("Constant", [], [value.name], value=tensor))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    opset = converter.OPSET - 1 if case == "opset" else converter.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "dtype", "shape", "missing"])
def test_side_by_side_onnx_interface(tmp_path, case, converter: ModuleType):
    path = tmp_path / "model.onnx"
    onnx.save(_interface(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "drift"])
def test_onnx_outputs_must_match_pytorch(case, converter: ModuleType):
    expected = np.random.RandomState(0).normal(0, 4, (1, 4, 256, 256)).astype(np.float32)
    actual = expected + 1e-5
    error, match = ValueError, "Unexpected or non-finite"
    if case == "shape":
        actual = actual[:, :3]
    elif case == "nan":
        actual[0, 0, 0, 0] = np.nan
    elif case == "drift":
        actual = expected + 0.01
        error, match = AssertionError, "masks"
    else:
        converter.compare(expected, actual, "masks")
        return
    with pytest.raises(error, match=match):
        converter.compare(expected, actual, "masks")


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType):
    model = load_metadata(ROOT / "models/mobile_sam")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="mobile_sam.pt"):
        converter.validate_source({"type": "file", "uri": "hf://models/dhkim2810/MobileSAM@main/sam_vit_b.pth"})


def test_check_prompts_fit_the_decoder_and_cover_every_point_type(
    converter: ModuleType, load_script: Callable[[str], ModuleType]
):
    kinds = {kind for _, types in converter.PROMPTS for kind in types}

    assert kinds == set(load_script("pipeline").POINT_TYPES[1:])
    assert max(len(points) for points, _ in converter.PROMPTS) == converter.POINTS
