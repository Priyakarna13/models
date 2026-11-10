from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

MODEL_ID = "efficientsam3_repvit_m"


@pytest.mark.parametrize(("op", "reference"), [("Sin", np.sin), ("Cos", np.cos), ("Log", np.log)])
def test_polynomial_replacements_stay_within_their_bound_over_the_whole_domain(
    load_script: Callable[[str], ModuleType], op: str, reference: Callable[[np.ndarray], np.ndarray]
) -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(op, ["x"], ["y"])],
        op,
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    surgery.replace_functions(model, {op: 1})
    x = np.linspace(*surgery.DOMAINS[op], 1_000_001, dtype=np.float32)

    (y,) = ReferenceEvaluator(model).run(None, {"x": x})

    assert op not in {node.op_type for node in model.graph.node}
    assert y.dtype == np.float32
    assert np.abs(y.astype(np.float64) - reference(x.astype(np.float64))).max() <= surgery.FUNCTION_BOUND


def test_prompts_pack_into_the_image_models_text_input(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    prompt = calibration.tokens("traffic light")

    encoded = calibration.one_hot(prompt)
    features = np.arange(16 * 256, dtype=np.float32).reshape(1, 16, 256)
    text = calibration.packed(features, prompt)

    assert prompt == [49406, 3399, 1395, 49407]
    assert encoded.shape == (1, 16, 49408) and encoded.dtype == np.float32
    assert encoded[0, :4].argmax(axis=1).tolist() == prompt
    assert encoded[0, :4].sum() == 4 and not encoded[0, 4:].any()
    assert text.shape == (1, 17, 256)
    np.testing.assert_array_equal(text[0, :16], features[0])
    assert text[0, 16].tolist() == [1.0] * 4 + [0.0] * 252
    # Every category fits the context with its start and end tokens; calibration spreads its prompts over them.
    assert len(calibration.COCO_TOKENS) == 80
    assert max(len(calibration.tokens(name)) for name in calibration.COCO_TOKENS) <= calibration.CONTEXT
    assert calibration.prompts(4) == ["person", "elephant", "wine glass", "dining table"]


def _folded_source(surgery: ModuleType) -> onnx.ModelProto:
    """A folded graph with the converted interface and the decoder's Sin, Cos and Log counts."""
    helper = onnx.helper
    values = {
        name: helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.INTERFACE.items()
    }
    shapes = [
        helper.make_tensor(f"{name}_shape", onnx.TensorProto.INT64, [len(shape)], shape)
        for name, shape in surgery.INTERFACE.items()
    ]
    nodes = [
        helper.make_node("ReduceMax", ["token_one_hot"], ["token"], axes=[0, 1, 2], keepdims=0),
        helper.make_node("Expand", ["token", "text_features_shape"], ["text_features"]),
        helper.make_node("ReduceMax", ["pixel_values"], ["pixel"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("ReduceMax", ["text"], ["prompt"], axes=[0, 1, 2], keepdims=0),
        helper.make_node("Add", ["pixel", "prompt"], ["value_0"]),
    ]
    ops = [op for op, count in surgery.EXPECTED.items() for _ in range(count)]
    nodes += [helper.make_node(op, [f"value_{index}"], [f"value_{index + 1}"]) for index, op in enumerate(ops)]
    nodes += [
        helper.make_node("Expand", [f"value_{len(ops)}", f"{name}_shape"], [name])
        for name in ("detections", "mask_logits")
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [values[name] for name in ("token_one_hot", "pixel_values", "text")],
        [values[name] for name in ("text_features", "detections", "mask_logits")],
        initializer=shapes,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_surgery_cuts_the_text_encoder_and_the_image_model(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "model.onnx"
    onnx.save(_folded_source(surgery), source)

    text, image = (onnx.load(path) for path in surgery.prepare_model(source, tmp_path / "graphs"))

    def interface(values) -> list[tuple[str, list[int]]]:
        return [(value.name, [d.dim_value for d in value.type.tensor_type.shape.dim]) for value in values]

    assert interface(text.graph.input) == [("token_one_hot", [1, 16, 49408])]
    assert interface(text.graph.output) == [("text_features", [1, 16, 256])]
    # The board feeds the image model the image and the packed text in this order.
    assert interface(image.graph.input) == [("pixel_values", [1, 3, 1008, 1008]), ("text", [1, 17, 256])]
    assert interface(image.graph.output) == [("detections", [1, 200, 6]), ("mask_logits", [1, 200, 192, 192])]
    assert {node.op_type for node in text.graph.node} == {"ReduceMax", "Expand"}
    assert not {node.op_type for graph in (text, image) for node in graph.graph.node} & {"Sin", "Cos", "Log"}
    assert text.ir_version == image.ir_version == 8


def test_fidelity_measures_images_after_the_calibration_images(
    tmp_path: Path, model_dir: Path, load_script: Callable[[str], ModuleType]
) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_script("calibration")
    compile_module = load_script("compile")
    bgr = np.zeros((48, 64, 3), np.uint8)
    bgr[..., 0] = 255
    _, content = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, 101])
    dataset = tmp_path / "coco-2017-cal.tar"
    with tarfile.open(dataset, "w") as archive:
        for index in range(24):
            member = tarfile.TarInfo(f"images/{index:012d}.webp")
            member.size = content.size
            archive.addfile(member, io.BytesIO(content.tobytes()))
    metadata = load_metadata(model_dir)
    selection = metadata["numerical_fidelity"]["selection"]

    calibration_names, fidelity_names = compile_module._image_selections(
        compile_module._component_parameters(metadata, "w16a16"), dataset, dataset, selection
    )

    assert len(calibration_names) == 16 and len(fidelity_names) == selection["samples"]
    assert calibration.select_images(dataset, selection["seed"], 0, 20) == calibration_names + fidelity_names
    with pytest.raises(ValueError, match="begin after the calibration images"):
        compile_module._image_selections(
            compile_module._component_parameters(metadata, "w16a16"), dataset, dataset, {**selection, "offset": 15}
        )
    # Images are RGB, stretched to the input size and scaled to [-1, 1].
    images = calibration.load_images(dataset, fidelity_names[:1])
    assert images.shape == (1, 1008, 1008, 3) and images.dtype == np.float32
    np.testing.assert_allclose(images[0, 500, 500], [-1.0, -1.0, 1.0])


def _mpk(path: Path, manifest: dict) -> Path:
    files = {"model.elf": b"", "model_mpk.json": json.dumps(manifest).encode()}
    with tarfile.open(path, "w:gz") as stream:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize("component_id", ["text_encoder", "image_model"])
def test_mpk_audit_requires_one_mla_segment_and_the_float32_interface(
    tmp_path: Path, load_script: Callable[[str], ModuleType], component_id: str
) -> None:
    compile_module = load_script("compile")
    contract = compile_module.COMPONENT_INTERFACES[component_id]
    plugins = [
        {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
        {"sequence": 1, "processor": "MLA", "config_params": {}},
        {"sequence": 2, "processor": "EV74", "output_nodes": [{"size": size} for size in contract["outputs"]]},
    ]
    manifest = {"input_nodes": [{"name": name, "size": size} for name, size in contract["inputs"].items()]}

    assert compile_module.audit_mpk(_mpk(tmp_path / "valid.tar.gz", {**manifest, "plugins": plugins}), component_id)[
        "output_sizes"
    ] == contract["outputs"]
    host = [*plugins[:2], {"sequence": 1, "processor": "A65"}, plugins[2]]
    with pytest.raises(RuntimeError, match="host=1"):
        compile_module.audit_mpk(_mpk(tmp_path / "host.tar.gz", {**manifest, "plugins": host}), component_id)


def test_build_holds_both_components_and_the_pipeline_fidelity(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))

    assert [item["component_id"] for item in outputs["outputs"]] == ["text_encoder", "image_model"]
    for item in outputs["outputs"]:
        archive = compiled / item["mpk_path"]
        assert archive.name == f"{MODEL_ID}_{build_profile}_{item['component_id']}_mpk.tar.gz"
        compile_module.audit_mpk(archive, item["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    assert (fidelity["model_id"], fidelity["build_profile_id"]) == (MODEL_ID, build_profile)
    metrics = {metric.pop("name"): metric for metric in fidelity["metrics"]}
    assert list(metrics) == ["detections", "mask_logits"]
    assert all(np.isfinite(value) for metric in metrics.values() for value in metric.values()), metrics
