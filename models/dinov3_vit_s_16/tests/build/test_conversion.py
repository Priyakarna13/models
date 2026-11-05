"""Checks of the conversion's fixed interface, its source and checkpoint configuration, and the converted ONNX."""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def _interface_model(converter: ModuleType, case: str) -> onnx.ModelProto:
    helper = onnx.helper
    inputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, converter.INTERFACE[name]) for name in converter.INPUTS]
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, converter.INTERFACE[name]) for name in converter.OUTPUTS]
    if case == "order":
        outputs.reverse()
    elif case == "dtype":
        outputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        outputs[1].type.tensor_type.shape.dim[1].dim_value = 201
    nodes = []
    for output in outputs:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        dtype = np.float64 if output.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE else np.float32
        nodes.append(helper.make_node("Constant", [], [output.name], value=numpy_helper.from_array(np.zeros(shape, dtype))))
    if case == "dynamic":
        nodes.append(helper.make_node("Shape", ["pixel_values"], ["shape"]))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape", "dynamic"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    path = tmp_path / "model.onnx"
    onnx.save(_interface_model(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def _export_like_model(converter: ModuleType) -> onnx.ModelProto:
    """An opset-20 graph with the export's patterns: a SplitToSequence of the prefix and patch tokens read by
    SequenceAt, an exact Gelu, and the pooled and full token outputs."""
    helper = onnx.helper
    tokens = [1, converter.PREFIX + converter.PATCHES, converter.HIDDEN_SIZE]
    constants = {
        "sizes": np.array([converter.PREFIX, converter.PATCHES], dtype=np.int64),
        "first": np.array(0, dtype=np.int64),
        "second": np.array(1, dtype=np.int64),
    }
    nodes = [
        helper.make_node("SplitToSequence", ["tokens", "sizes"], ["sequence"], axis=1),
        helper.make_node("SequenceAt", ["sequence", "first"], ["prefix"]),
        helper.make_node("SequenceAt", ["sequence", "second"], ["patches"]),
        helper.make_node("Gelu", ["patches"], ["activated"]),
        helper.make_node("Concat", ["prefix", "activated"], ["last_hidden_state"], axis=1),
        helper.make_node("Gather", ["last_hidden_state", "first"], ["pooler_output"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "export",
        [helper.make_tensor_value_info("tokens", onnx.TensorProto.FLOAT, tokens)],
        [helper.make_tensor_value_info("last_hidden_state", onnx.TensorProto.FLOAT, tokens),
         helper.make_tensor_value_info("pooler_output", onnx.TensorProto.FLOAT, [1, converter.HIDDEN_SIZE])],
        [numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def test_the_rewrites_keep_the_exports_outputs(tmp_path, converter: ModuleType):
    # ONNX's own reference evaluator, because the registry's package tests run without ONNX Runtime.
    from onnx.reference import ReferenceEvaluator

    path = tmp_path / "model.onnx"
    onnx.save(_export_like_model(converter), path)
    tokens = np.random.default_rng(0).normal(size=(1, converter.PREFIX + converter.PATCHES, converter.HIDDEN_SIZE))
    feed = {"tokens": tokens.astype(np.float32)}
    hidden, pooled = ReferenceEvaluator(str(path)).run(None, feed)

    converter.make_static(path)

    model = onnx.load(path)
    assert [item.version for item in model.opset_import if not item.domain] == [17]
    assert not {node.op_type for node in model.graph.node} & set(converter.DYNAMIC_OPS)
    class_embedding, patch_embeddings = ReferenceEvaluator(model).run(list(converter.OUTPUTS), feed)
    np.testing.assert_allclose(class_embedding, pooled, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(patch_embeddings, hidden[:, converter.PREFIX :], rtol=1e-5, atol=1e-6)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType):
    model = load_metadata(ROOT / "models/dinov3_vit_s_16")
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


def test_converted_model_returns_finite_class_and_patch_embeddings(converted: Path, converter: ModuleType):
    import onnxruntime as ort

    converter.validate_export(converted)
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).normal(size=converter.INTERFACE["pixel_values"]).astype(np.float32)
    class_embedding, patch_embeddings = session.run(None, {"pixel_values": image})
    assert class_embedding.shape == (1, converter.HIDDEN_SIZE)
    assert patch_embeddings.shape == (1, converter.PATCHES, converter.HIDDEN_SIZE)
    assert np.isfinite(class_embedding).all() and np.isfinite(patch_embeddings).all()
    # Different patches of a varied image get different embeddings.
    assert np.std(patch_embeddings[0], axis=0).mean() > 0
