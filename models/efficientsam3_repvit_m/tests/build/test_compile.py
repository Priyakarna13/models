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


@pytest.mark.parametrize(
    ("case", "fuse_qkv"),
    [
        ("text", True),
        ("image", False),
        ("decoder", False),
        ("broadcast", False),
        ("geometry", False),
    ],
)
def test_attention_preserves_heads_positions_and_masks(
    tmp_path: Path, load_script: Callable[[str], ModuleType], case: str, fuse_qkv: bool
) -> None:
    from onnx.reference import ReferenceEvaluator

    torch = pytest.importorskip("torch")
    surgery = load_script("surgery")
    generator = torch.Generator().manual_seed(232)
    width, heads = 32, 4
    queries, keys = (
        (16, 16) if case == "text" else (9, 9) if case == "image" else (1, 9) if case == "geometry" else (5, 9)
    )

    def random(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    weight, bias = random(3 * width, width), random(3 * width)
    output_weight, output_bias = random(width, width), random(width)
    query = random(1, 1, queries, width)
    key, value = random(1, 1, keys, width), random(1, 1, keys, width)
    mask = random(1, heads, queries, keys)
    if case == "text":
        key = value = query
        mask = torch.full((1, 1, queries, keys), -1e4).triu(1)
    elif case == "image":
        value = query
        query = key = query + random(1, 1, queries, width)
        mask = None
    elif case == "broadcast":
        mask = torch.zeros(1, 1, 1, keys)
        mask[..., -3:] = -1e4

    projected = [
        torch.nn.functional.linear(x, w, b).reshape(1, -1, heads, width // heads).transpose(1, 2)
        for x, w, b in zip((query, key, value), weight.chunk(3), bias.chunk(3), strict=True)
    ]
    context = torch.nn.functional.scaled_dot_product_attention(*projected, attn_mask=mask)
    expected = torch.nn.functional.linear(
        context.transpose(1, 2).reshape(1, 1, queries, width), output_weight, output_bias
    )
    module = surgery.build_attention(weight, bias, output_weight, output_bias, heads, fuse_qkv=fuse_qkv)
    arguments = (query,) if fuse_qkv else (query, key, value)
    with torch.no_grad():
        actual = module(*arguments, bias=mask)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=3e-6)
    assert actual.shape == (1, 1, queries, width)
    if fuse_qkv:
        with pytest.raises(ValueError, match="one shared Q/K/V input"):
            module(query, key, value)

    path = tmp_path / "attention.onnx"
    inputs = {"query": query}
    if not fuse_qkv:
        inputs.update(key=key, value=value)
    if mask is not None:
        inputs["bias"] = mask
    torch.onnx.export(
        module,
        arguments,
        str(path),
        kwargs={"bias": mask},
        input_names=list(inputs),
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    (result,) = ReferenceEvaluator(model).run(None, {name: tensor.numpy() for name, tensor in inputs.items()})
    np.testing.assert_allclose(result, expected.numpy(), rtol=3e-5, atol=5e-6)


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


@pytest.mark.parametrize("script_name", ["accuracy", "performance"])
def test_packaged_tokenizer_prepares_free_text_prompts(
    load_script: Callable[[str], ModuleType], model_dir: Path, script_name: str
) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("ftfy")
    script = load_script(script_name)
    tokenizer = script.load_tokenizer(model_dir / "tests" / "fixtures")
    calibration = load_script("calibration")

    for name in ("person", "traffic light", "dog"):
        assert script.tokens(name, tokenizer) == calibration.tokens(name)
    assert script.tokens("  DOG\t person\n", tokenizer) == [49406, 1929, 2533, 49407]
    assert script.tokens("dog &amp;amp; person", tokenizer) == [49406, 1929, 261, 2533, 49407]
    assert script.tokens("", tokenizer) == [49406, 49407]
    assert len(script.tokens("dog " * 14, tokenizer)) == 16
    with pytest.raises(ValueError, match="16-token context"):
        script.tokens("dog " * 15, tokenizer)


@pytest.mark.parametrize("script_name", ["accuracy", "performance"])
def test_benchmarks_require_the_packaged_tokenizer(
    load_script: Callable[[str], ModuleType], tmp_path: Path, script_name: str
) -> None:
    pytest.importorskip("tokenizers")
    with pytest.raises(Exception, match="No such file|os error 2"):
        load_script(script_name).load_tokenizer(tmp_path)


def _upstream_outputs(surgery: ModuleType) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    scores = np.zeros(surgery.QUERIES)
    scores[0] = 0.9
    return {
        "text_features": rng.standard_normal(surgery.INTERFACE["text_features"][1:]),
        "scores": scores,
        "boxes_px": rng.uniform(0, 1008, (surgery.QUERIES, 4)),
        "mask_logits": rng.standard_normal((surgery.QUERIES, 8, 8)),
    }


@pytest.mark.parametrize("case", ["valid", "padding", "confident_box", "score", "shape", "nan"])
def test_export_must_agree_with_the_upstream_outputs(case: str, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    expected = _upstream_outputs(surgery)
    actual = {name: value.copy() for name, value in expected.items()}
    if case == "valid":
        actual["boxes_px"][0] += 0.05
    elif case == "padding":
        # Padding positions carry no prompt and are masked out of the image model.
        actual["text_features"][4:] += 1
    elif case == "confident_box":
        actual["boxes_px"][0, 0] += 1
    elif case == "score":
        actual["scores"][3] += 0.01
    elif case == "shape":
        actual["scores"] = actual["scores"][:-1]
    else:
        actual["mask_logits"][0, 0, 0] = np.nan
    if case in {"valid", "padding"}:
        surgery.compare_outputs(expected, actual, valid=4)
    else:
        with pytest.raises(ValueError):
            surgery.compare_outputs(expected, actual, valid=4)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_export_has_the_fixed_onnx_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    helper = onnx.helper
    values = {
        name: helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.INTERFACE.items()
    }
    inputs = [values[name] for name in surgery.EXPORT_INPUTS]
    outputs = [values[name] for name in surgery.EXPORT_OUTPUTS]
    if case == "order":
        inputs.reverse()
    elif case == "dtype":
        outputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        inputs[0].type.tensor_type.shape.dim[1].dim_value = 32
    nodes = [helper.make_node("Constant", [], [value.name], value_float=0.0) for value in outputs]
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        surgery.validate_export(path)
    else:
        with pytest.raises(ValueError):
            surgery.validate_export(path)


def test_resize_sizes_fold_to_constants(load_script: Callable[[str], ModuleType]) -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    helper = onnx.helper

    def resize(source: str, target: str, name: str) -> list[onnx.NodeProto]:
        return [
            helper.make_node("Shape", [source], [f"{name}/shape"], name=f"{name}/Shape"),
            helper.make_node("Slice", [f"{name}/shape", "zero", "two", "zero"], [f"{name}/lead"], name=f"{name}/Slice"),
            helper.make_node("Concat", [f"{name}/lead", f"{name}/target"], [f"{name}/sizes"], axis=0),
            helper.make_node("Resize", [source, "", "", f"{name}/sizes"], [target], mode="nearest"),
        ]

    def integers(name: str, *values: int) -> onnx.TensorProto:
        return onnx.numpy_helper.from_array(np.array(values, np.int64), name)

    graph = helper.make_graph(
        [*resize("x", "middle", "first"), *resize("middle", "y", "second")],
        "fixture",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        initializer=[
            integers("zero", 0),
            integers("two", 2),
            integers("first/target", 4, 4),
            integers("second/target", 8, 8),
            integers("unused", 7),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    x = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    (expected,) = ReferenceEvaluator(model).run(None, {"x": x})

    folded = surgery.fold_resize_sizes(model)

    assert [node.op_type for node in folded.graph.node] == ["Resize", "Resize"]
    sizes = {tensor.name: onnx.numpy_helper.to_array(tensor).tolist() for tensor in folded.graph.initializer}
    # The second Resize's input has a known shape only once the first has its constant size.
    assert sizes == {"first/sizes": [1, 3, 4, 4], "second/sizes": [1, 3, 8, 8]}
    np.testing.assert_array_equal(ReferenceEvaluator(folded).run(None, {"x": x})[0], expected)


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

    assert (
        compile_module.audit_mpk(_mpk(tmp_path / "valid.tar.gz", {**manifest, "plugins": plugins}), component_id)[
            "output_sizes"
        ]
        == contract["outputs"]
    )
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
