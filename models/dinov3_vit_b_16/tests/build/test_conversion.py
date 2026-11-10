"""Checks of the conversion's fixed interface, its source and checkpoint configuration, and the converted ONNX."""

import json
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


def _export_model(converter: ModuleType, case: str = "valid") -> onnx.ModelProto:
    """A graph with the export's interface: the image in, last_hidden_state and pooler_output out."""
    helper = onnx.helper
    image = helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, ["batch", 3, "height", "width"])
    shapes = {
        "last_hidden_state": ["batch", "tokens", converter.HIDDEN_SIZE],
        "pooler_output": ["batch", converter.HIDDEN_SIZE],
    }
    outputs = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shapes[name]) for name in converter.OUTPUT_SHAPES
    ]
    if case == "order":
        outputs.reverse()
    elif case == "dtype":
        outputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    nodes = [
        helper.make_node("Relu", ["pixel_values"], ["last_hidden_state"]),
        helper.make_node("Sigmoid", ["pixel_values"], ["pooler_output"]),
    ]
    graph = helper.make_graph(nodes, "export", [image], outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17 if case == "opset" else converter.OPSET)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "dynamic"])
def test_the_converted_interface(tmp_path, case, converter: ModuleType):
    source, path = tmp_path / "source.onnx", tmp_path / "model.onnx"
    onnx.save(_export_model(converter, case), source)
    if case == "dynamic":
        onnx.save(onnx.load(source), path)
    else:
        converter.fix_input_shape(source, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_the_conversion_fixes_only_the_input_size(tmp_path, converter: ModuleType):
    # ONNX's own reference evaluator, so the check runs without ONNX Runtime.
    from onnx.reference import ReferenceEvaluator

    source, path = tmp_path / "source.onnx", tmp_path / "model.onnx"
    onnx.save(_export_model(converter), source)

    converter.fix_input_shape(source, path)

    original, converted = onnx.load(source), onnx.load(path)
    assert [d.dim_value for d in converted.graph.input[0].type.tensor_type.shape.dim] == converter.INPUT_SHAPE
    assert converted.graph.node == original.graph.node and converted.graph.output == original.graph.output
    image = np.random.default_rng(0).normal(size=converter.INPUT_SHAPE).astype(np.float32)
    expected, actual = (ReferenceEvaluator(model).run(None, {"pixel_values": image}) for model in (original, converted))
    for want, got in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(got, want)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType):
    model = load_metadata(ROOT / "models/dinov3_vit_b_16")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source({"type": "file", "uri": "https://example.com/checkpoint.pth"})


def _config(converter: ModuleType, **changes) -> dict:
    config = {"model_type": converter.MODEL_TYPE, "image_size": 224, "patch_size": 16,
              "hidden_size": converter.HIDDEN_SIZE, "num_register_tokens": 4}
    return {**config, **changes}


@pytest.mark.parametrize("changes", [{}, {"model_type": "dinov2"}, {"hidden_size": 1536}, {"num_register_tokens": 0}])
def test_config_must_be_this_checkpoint(tmp_path, changes, converter: ModuleType):
    (tmp_path / "config.json").write_text(json.dumps(_config(converter, **changes)), encoding="utf-8")
    if not changes:
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="config.json declares"):
            converter.validate_config(tmp_path)


@pytest.mark.parametrize(
    "changes",
    [{}, {"do_center_crop": True}, {"size": {"shortest_edge": 256}}, {"resample": 3}, {"image_mean": [0.5, 0.5, 0.5]}],
)
def test_the_checkpoints_preprocessing_must_be_what_the_scripts_apply(tmp_path, changes, converter: ModuleType):
    declared = {**converter.PREPROCESSING, "image_processor_type": "DINOv3ViTImageProcessorFast", **changes}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(declared), encoding="utf-8")
    if not changes:
        converter.validate_preprocessor(tmp_path)
    else:
        with pytest.raises(ValueError, match="preprocessor_config.json"):
            converter.validate_preprocessor(tmp_path)


def test_the_conversion_checks_the_preprocessing_compile_applies(converter: ModuleType, load_script):
    calibration = load_script("calibration")
    assert converter.PREPROCESSING["size"] == {"height": calibration.SIZE, "width": calibration.SIZE}
    np.testing.assert_allclose(converter.PREPROCESSING["image_mean"], calibration.MEAN)
    np.testing.assert_allclose(converter.PREPROCESSING["image_std"], calibration.STD)
    assert converter.PREPROCESSING["rescale_factor"] == pytest.approx(calibration.RESCALE)


def test_converted_model_returns_finite_token_and_pooled_embeddings(converted: Path, converter: ModuleType):
    import onnxruntime as ort

    converter.validate_export(converted)
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).normal(size=converter.INPUT_SHAPE).astype(np.float32)
    hidden, pooled = session.run(list(converter.OUTPUT_SHAPES), {"pixel_values": image})
    assert hidden.shape == (1, converter.PREFIX + converter.PATCHES, converter.HIDDEN_SIZE)
    assert pooled.shape == (1, converter.HIDDEN_SIZE)
    assert np.isfinite(hidden).all() and np.isfinite(pooled).all()
    # Different patches of a varied image get different embeddings.
    assert np.std(hidden[0, converter.PREFIX :], axis=0).mean() > 0
