"""Checks of the export interface, the source, the pipeline's components and the converted ONNX."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.conversion import conversion_outputs
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.fixture
def pipeline(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("pipeline")


def interface_fixture(tmp_path: Path, converter: ModuleType, case: str) -> Path:
    """A graph with the declared interface and its weight in model.onnx.data, or one broken in the named way."""
    helper = onnx.helper
    shapes = {name: list(shape) for name, shape in converter.INTERFACE.items()}
    names = list(shapes)
    if case == "names":
        names[1] = "last_hidden_state"
    elif case == "shape":
        shapes[names[-1]][1] += 1
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in zip(names, shapes.values())
    ]
    if case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [value.name],
            value=helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(shape).ravel()),
        )
        for value, shape in zip(values[1:], list(shapes.values())[1:])
    ]
    weights = [onnx.numpy_helper.from_array(np.zeros(1024, np.float32), "model.norm.weight")]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    if case == "embedded":
        onnx.save(model, path)
    else:
        location = "weights.bin" if case == "location" else converter.DATA_FILE
        onnx.save(
            model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=location, size_threshold=0
        )
    return path


def test_the_components_cover_every_block_once_and_chain_their_tokens(pipeline: ModuleType) -> None:
    blocks = [pipeline.BLOCKS[component] for component in pipeline.COMPONENTS]
    assert blocks[0][0] == 0 and blocks[-1][1] == pipeline.DEPTH == 32
    assert all(previous[1] == following[0] for previous, following in zip(blocks, blocks[1:]))
    assert (pipeline.HIDDEN_SIZE, pipeline.PREFIX, pipeline.PATCHES, pipeline.TOKENS) == (1280, 5, 196, 201)
    interfaces = [pipeline.INTERFACES[component] for component in pipeline.COMPONENTS]
    assert interfaces[0] == {"pixel_values": [1, 3, 224, 224], "tokens": [1, 1280, 1, 201]}
    for previous, following in zip(interfaces, interfaces[1:]):
        assert previous["tokens"] == following["input_tokens"]
    assert list(interfaces[-1])[1:] == list(pipeline.OUTPUTS) == ["class_embedding", "patch_embeddings"]
    assert interfaces[-1]["class_embedding"] == [1, 1280]
    assert interfaces[-1]["patch_embeddings"] == [1, 196, 1280]


@pytest.mark.parametrize("case", ["valid", "opset", "names", "dtype", "shape", "embedded", "location"])
def test_fixed_onnx_interface_with_external_weights(tmp_path: Path, case: str, converter: ModuleType) -> None:
    path = interface_fixture(tmp_path, converter, case)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_the_conversion_publishes_one_model_for_the_pipelines_components(
    converter: ModuleType, pipeline: ModuleType
) -> None:
    converter.validate_outputs(conversion_outputs(MODEL_DIR))
    with pytest.raises(ValueError, match="model.onnx"):
        converter.validate_outputs([f"{component}.onnx" for component in pipeline.COMPONENTS])
    assert [component["id"] for component in load_metadata(MODEL_DIR)["components"]] == list(pipeline.COMPONENTS)


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


def test_converted_model_returns_finite_class_and_patch_embeddings(
    converted: Path, converter: ModuleType, load_script: Callable[[str], ModuleType]
) -> None:
    import onnxruntime as ort

    converter.validate_export(converted)
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).normal(size=converter.INTERFACE["pixel_values"]).astype(np.float32)
    class_embedding, patch_embeddings = session.run(list(converter.OUTPUTS), {"pixel_values": image})
    del session
    assert class_embedding.shape == (1, 1280)
    assert patch_embeddings.shape == (1, 196, 1280)
    assert np.isfinite(class_embedding).all() and np.isfinite(patch_embeddings).all()
    # Different patches of a varied image get different embeddings.
    assert np.std(patch_embeddings[0], axis=0).mean() > 0
    # It holds every weight surgery.py rebuilds the components from.
    assert len(load_script("surgery").load_weights(converted)) == 486
