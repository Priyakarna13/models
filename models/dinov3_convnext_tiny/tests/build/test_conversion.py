"""Checks of the export interface, its source and the converted ONNX."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[2]
# timm's ConvNeXt-T stages, one GELU per block.
DEPTHS = (3, 3, 9, 3)


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def interface_fixture(converter: ModuleType, case: str) -> onnx.ModelProto:
    """A graph with the declared interface, or one broken in the named way; each output is a constant."""
    helper = onnx.helper
    shapes = {name: list(shape) for name, shape in converter.INTERFACE.items()}
    names = list(shapes)
    if case == "names":
        names[2] = "last_hidden_state"
    elif case == "shape":
        shapes["patch_embeddings"][1] = converter.PATCHES + 1
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in zip(names, shapes.values())
    ]
    if case == "dtype":
        values[1].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [value.name],
            value=helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(shape).ravel()),
        )
        for value, shape in zip(values[1:], list(shapes.values())[1:])
    ]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])


def test_interface_is_the_class_and_patch_embeddings(converter: ModuleType) -> None:
    assert converter.INTERFACE == {
        "pixel_values": [1, 3, 224, 224],
        "class_embedding": [1, converter.HIDDEN_SIZE],
        "patch_embeddings": [1, 49, converter.HIDDEN_SIZE],
    }


@pytest.mark.parametrize("case", ["valid", "opset", "names", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, converter: ModuleType) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(interface_fixture(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType) -> None:
    model = load_metadata(MODEL_DIR)
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source({"type": "file", "uri": "https://example.com/model.pth"})


def test_cosine_compares_directions_not_scale(converter: ModuleType) -> None:
    vector = np.asarray([1.0, 2.0, 3.0])

    assert converter.cosine(vector, 4 * vector) == pytest.approx(1.0)
    assert converter.cosine(vector, -vector) == pytest.approx(-1.0)
    assert converter.MIN_COSINE >= 0.99999


def test_converted_model_returns_finite_class_and_patch_embeddings(converted: Path, converter: ModuleType) -> None:
    import onnxruntime as ort

    converter.validate_export(converted)
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).normal(size=converter.INTERFACE["pixel_values"]).astype(np.float32)
    class_embedding, patch_embeddings = session.run(list(converter.OUTPUTS), {"pixel_values": image})
    assert class_embedding.shape == (1, converter.HIDDEN_SIZE)
    assert patch_embeddings.shape == (1, converter.PATCHES, converter.HIDDEN_SIZE)
    assert np.isfinite(class_embedding).all() and np.isfinite(patch_embeddings).all()
    # Different patches of a varied image get different embeddings.
    assert np.std(patch_embeddings[0], axis=0).mean() > 0
    # The conversion is timm's network as it is: each GELU keeps its exact erf form for surgery.py to rewrite.
    assert sum(node.op_type == "Erf" for node in onnx.load(converted).graph.node) == sum(DEPTHS)
