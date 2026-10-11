from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "efficientsam3_repvit_m"
SCRIPTS_DIR = MODEL_DIR / "scripts"
# Compiler and board modules that compile.py and accuracy.py import but these tests never call.
COMPILER_MODULES = [
    "afe",
    "afe.apis",
    "afe.apis.defines",
    "afe.apis.loaded_net",
    "afe.apis.model",
    "afe.core",
    "afe.core.utils",
    "afe.load",
    "afe.load.importers",
    "afe.load.importers.general_importer",
    "sima_utils",
    "sima_utils.data",
    "sima_utils.data.data_generator",
]
BOARD_MODULES = ["pyneat", "faster_coco_eval", "faster_coco_eval.core"]


def load_module(name: str, monkeypatch: pytest.MonkeyPatch | None = None, stubs: list[str] | None = None) -> ModuleType:
    for stub in stubs or []:
        monkeypatch.setitem(sys.modules, stub, MagicMock())
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"efficientsam3_repvit_m_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Sibling scripts are imported by name, so another model's calibration or surgery must not stand in for ours.
    shadowed = {dependency: sys.modules.pop(dependency, None) for dependency in ("calibration", "surgery")}
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        for dependency, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(dependency, None)
            else:
                sys.modules[dependency] = previous
    return module


def test_prompt_tokens_one_hot_and_packed_text() -> None:
    calibration = load_module("calibration")
    prompt = calibration.tokens("traffic light")

    encoded = calibration.one_hot(prompt)
    features = np.arange(16 * 256, dtype=np.float32).reshape(1, 16, 256)
    text = calibration.packed(features, prompt)

    assert prompt == [49406, 3399, 1395, 49407]
    assert encoded.shape == (1, 16, 49408)
    assert encoded.dtype == np.float32
    assert encoded[0, :4].argmax(axis=1).tolist() == prompt
    assert encoded[0, :4].sum() == 4 and not encoded[0, 4:].any()
    assert text.shape == (1, 17, 256)
    np.testing.assert_array_equal(text[0, :16], features[0])
    assert text[0, 16].tolist() == [1.0] * 4 + [0.0] * 252


def test_coco_prompts_fit_the_context_and_spread_over_categories() -> None:
    calibration = load_module("calibration")
    names = list(calibration.COCO_TOKENS)

    assert len(names) == 80
    assert max(len(calibration.tokens(name)) for name in names) <= calibration.CONTEXT
    assert calibration.prompts(4) == ["person", "elephant", "wine glass", "dining table"]
    assert calibration.prompts(80) == names


def test_prompt_fills_at_most_the_context_with_its_start_and_end_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = load_module("calibration")
    monkeypatch.setitem(calibration.COCO_TOKENS, "longest", tuple(range(1, 15)))
    monkeypatch.setitem(calibration.COCO_TOKENS, "too long", tuple(range(1, 16)))

    assert len(calibration.tokens("longest")) == calibration.CONTEXT == 16
    with pytest.raises(ValueError, match="has 15 tokens; the 16-position context holds at most 14"):
        calibration.tokens("too long")


def test_preprocessing_stretches_rgb_to_minus_one_one() -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    bgr = np.zeros((20, 30, 3), np.uint8)
    bgr[..., 0] = 255
    _, content = cv2.imencode(".png", bgr)

    image = calibration.preprocess_image(content.tobytes())

    assert image.shape == (1008, 1008, 3)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[500, 500], [-1.0, -1.0, 1.0])


@pytest.mark.parametrize("size", [(640, 480), (1280, 1100), (1008, 1008)])
def test_resize_without_mixed_directions_is_one_cv2_call(size: tuple[int, int]) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    width, height = size
    image = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    interpolation = cv2.INTER_AREA if width > 1008 else cv2.INTER_LINEAR

    np.testing.assert_array_equal(
        calibration.resize(image), cv2.resize(image, (1008, 1008), interpolation=interpolation)
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_resize_area_averages_a_shrinking_axis_and_interpolates_an_enlarging_one(transpose: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    rng = np.random.default_rng(0)
    # A 1280x720 image, or its 720x1280 transpose: one axis shrinks to 1008 and the other enlarges.
    shrinking = rng.integers(0, 256, (1, 1280, 3), dtype=np.uint8)
    enlarging = rng.integers(0, 256, (720, 1, 3), dtype=np.uint8)
    area = cv2.resize(shrinking, (1008, 1), interpolation=cv2.INTER_AREA)
    linear = cv2.resize(enlarging, (1, 1008), interpolation=cv2.INTER_LINEAR)

    def resized(image: np.ndarray) -> np.ndarray:
        if not transpose:
            return calibration.resize(image)
        return calibration.resize(np.ascontiguousarray(image.transpose(1, 0, 2))).transpose(1, 0, 2)

    # An image that varies along one axis only resizes as that axis's one-dimensional resize, up to one level
    # of OpenCV fixed-point rounding. Area or bilinear resizing on both axes misses by more than 30 levels.
    for image, expected in ((np.repeat(shrinking, 720, 0), area), (np.repeat(enlarging, 1280, 1), linear)):
        np.testing.assert_allclose(resized(image).astype(int), np.broadcast_to(expected, (1008, 1008, 3)), atol=1)


def test_image_selection_is_seeded_disjoint_and_prefix_consistent(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    for split in ("cal2017", "val2017"):
        (tmp_path / split).mkdir()
        for index in range(30):
            (tmp_path / split / f"{index:012d}.webp").write_bytes(b"")

    calibration_names, fidelity_names = calibration.select_images(tmp_path, 16, 4)
    longer, _ = calibration.select_images(tmp_path, 20, 4)

    assert all(name.startswith("cal2017/") for name in calibration_names + fidelity_names)
    assert set(calibration_names).isdisjoint(fidelity_names)
    # The image model calibrates on the same images whatever the fidelity sample size.
    assert longer[:16] == calibration_names
    assert (calibration_names, fidelity_names) == calibration.select_images(tmp_path, 16, 4)
    with pytest.raises(ValueError, match="cannot select"):
        calibration.select_images(tmp_path, 28, 4)


def _function_model(op: str) -> onnx.ModelProto:
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node(op, ["x"], ["y"])],
        op,
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [64])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [64])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize(("op", "reference"), [("Sin", np.sin), ("Cos", np.cos), ("Log", np.log)])
def test_surgery_replaces_functions_within_their_domain(op: str, reference) -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_module("surgery")
    model = _function_model(op)
    surgery.replace_functions(model, {op: 1})
    x = np.linspace(*surgery.DOMAINS[op], 64, dtype=np.float32)

    (y,) = ReferenceEvaluator(model).run(None, {"x": x})

    assert op not in {node.op_type for node in model.graph.node}
    np.testing.assert_allclose(y, reference(x.astype(np.float64)), rtol=0, atol=surgery.FUNCTION_BOUND)


def test_surgery_rejects_unexpected_function_counts() -> None:
    surgery = load_module("surgery")

    with pytest.raises(ValueError, match="Unsupported EfficientSAM3 graph"):
        surgery.replace_functions(_function_model("Sin"), {"Sin": 2})


def test_surgery_requires_a_folded_source(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    helper = onnx.helper
    values = {
        name: helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.INTERFACE.items()
    }
    nodes = [helper.make_node("Shape", ["token_one_hot"], ["shape"])]
    nodes += [
        helper.make_node("Constant", [], [name], value=helper.make_tensor(name, onnx.TensorProto.FLOAT, [1], [0.0]))
        for name in ("text_features", "detections", "mask_logits")
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [values[name] for name in ("token_one_hot", "pixel_values", "text")],
        [values[name] for name in ("text_features", "detections", "mask_logits")],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    with pytest.raises(ValueError, match="fold it with onnxsim"):
        surgery.prepare_model(path, tmp_path / "out")


def _folded_source(surgery: ModuleType) -> onnx.ModelProto:
    """Return a folded graph with the converted interface and the decoder's Sin, Cos and Log counts."""
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
    ops = ["Sin"] * 5 + ["Cos"] * 5 + ["Log"] * 20
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


def test_surgery_cuts_the_text_encoder_and_the_image_model(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "model.onnx"
    onnx.save(_folded_source(surgery), source)

    text, image = (onnx.load(path) for path in surgery.prepare_model(source, tmp_path / "out"))

    def interface(values) -> list[tuple[str, list[int]]]:
        return [(value.name, [d.dim_value for d in value.type.tensor_type.shape.dim]) for value in values]

    assert interface(text.graph.input) == [("token_one_hot", [1, 16, 49408])]
    assert interface(text.graph.output) == [("text_features", [1, 16, 256])]
    # The board feeds the image model's inputs in this order.
    assert interface(image.graph.input) == [("pixel_values", [1, 3, 1008, 1008]), ("text", [1, 17, 256])]
    assert interface(image.graph.output) == [("detections", [1, 200, 6]), ("mask_logits", [1, 200, 192, 192])]
    assert not {node.op_type for graph in (text, image) for node in graph.graph.node} & {"Sin", "Cos", "Log"}
    assert {node.op_type for node in text.graph.node} == {"ReduceMax", "Expand"}
    assert text.ir_version == image.ir_version == 8


def test_accuracy_reads_nchw_and_hwc_image_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    detections = np.random.default_rng(0).random((1, 200, 6), dtype=np.float32)
    masks = np.random.default_rng(1).random((1, 200, 192, 192), dtype=np.float32)

    np.testing.assert_array_equal(accuracy.detection_rows(detections), detections[0])
    for layout in (masks, masks.transpose(0, 2, 3, 1).copy()):
        np.testing.assert_array_equal(accuracy.query_masks(layout), masks[0])
    with pytest.raises(ValueError, match="non-finite"):
        accuracy.detection_rows(detections * np.nan)
    kept = (np.zeros((0, 6), np.float32), np.zeros(0, np.int64), np.zeros((0, 192, 192), np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        accuracy.best_queries(kept, detections[0], lambda: masks[0] * np.nan, 1)


def test_accuracy_keeps_the_best_queries_of_all_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    monkeypatch.setattr(accuracy, "TOP_K", 3)
    kept = (np.zeros((0, 6), np.float32), np.zeros(0, np.int64), np.zeros((0, 2, 2), np.float32))
    reads = []

    for category, scores in ((1, [0.1, 0.9, 0.3]), (7, [0.5, 0.05, 0.95]), (9, [0.5, 0.2, 0.1])):
        detections = np.zeros((3, 6), np.float32)
        detections[:, 4] = scores
        masks = np.stack([np.full((2, 2), score, np.float32) for score in scores])
        kept = accuracy.best_queries(kept, detections, lambda masks=masks: reads.append(1) or masks, category)

    assert kept[0][:, 4].tolist() == pytest.approx([0.95, 0.9, 0.5])
    # The earlier prompt keeps a tied score, and a prompt with no better query never reads its masks.
    assert kept[1].tolist() == [7, 1, 7]
    assert kept[2][:, 0, 0].tolist() == pytest.approx([0.95, 0.9, 0.5])
    assert len(reads) == 2


def test_accuracy_maps_boxes_and_masks_to_the_original_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    encoded = []
    monkeypatch.setattr(
        accuracy.mask_util,
        "encode",
        lambda masks: encoded.append(masks) or [{"mask": index} for index in range(masks.shape[-1])],
    )
    detections = np.array([[-5.04, 504, 1008, 1058.4, 0.8, 0]], np.float32)
    masks = np.full((1, 192, 192), -1, np.float32)
    masks[0, :, 96:] = 1

    boxes, segments = accuracy.coco_results(42, 200, 100, (detections, np.array([18]), masks))

    assert boxes == [
        {"image_id": 42, "category_id": 18, "score": pytest.approx(0.8), "bbox": pytest.approx([-1, 50, 201, 55])}
    ]
    assert segments == [{"image_id": 42, "category_id": 18, "score": pytest.approx(0.8), "segmentation": {"mask": 0}}]
    (binary,) = encoded
    assert binary.shape == (100, 200, 1) and binary.flags.f_contiguous
    assert not binary[:, :99].any() and binary[:, 101:].all()


def test_accuracy_metrics_match_the_model_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))

    assert list(accuracy.METRICS) == model["accuracy"][0]["metrics"]
    assert accuracy.METRICS["coco_mask_map"] == ("segm", "AP_all")
    assert accuracy.parse_mpks(["image_model=b.tar.gz", "text_encoder=a.tar.gz"]) == {
        "text_encoder": Path("a.tar.gz"),
        "image_model": Path("b.tar.gz"),
    }
    with pytest.raises(SystemExit):
        accuracy.parse_mpks(["model=a.tar.gz"])


class _Output:
    def __init__(self, value: np.ndarray):
        self.value, self.shape = value, value.shape

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.value


def test_board_inputs_are_passed_as_prepared_hwc_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    pyneat = accuracy.pyneat
    pyneat.Tensor.from_numpy.side_effect = lambda array, **options: (array, options)
    models, runs = [], []

    class Model:
        def __init__(self, path: str, options) -> None:
            self.text = path == "text.tar.gz"
            models.append(options)

        def input_specs(self) -> list[SimpleNamespace]:
            shapes = [[16, 49408]] if self.text else [[17, 256], [1008, 1008, 3]]
            return [SimpleNamespace(shape=shape) for shape in shapes]

        def run(self, inputs: list, timeout_ms: int) -> list[_Output]:
            runs.append(inputs)
            if self.text:
                return [_Output(np.zeros((1, 16, 256), np.float32))]
            return [_Output(np.zeros((192, 192, 200), np.float32)), _Output(np.zeros((1, 200, 6), np.float32))]

    monkeypatch.setattr(pyneat, "Model", Model)
    features = accuracy.board_text_features(Path("text.tar.gz"))
    text = np.ones((1, 17, 256), np.float32)
    pixels = np.zeros((1008, 1008, 3), np.float32)
    (detections, masks), *_ = accuracy.board_image_model(Path("image.tar.gz"))([text])(pixels)

    assert len(features) == len(runs) - 1 == 80
    assert detections.shape == (200, 6) and masks().shape == (200, 192, 192)
    # The image model takes the inputs in its spec order; every input is a copied HWC tensor.
    (text_input, text_options), (pixel_input, pixel_options) = runs[-1]
    assert text_input.shape == (17, 256) and pixel_input.shape == (1008, 1008, 3)
    assert {"copy": True, "layout": pyneat.TensorLayout.HWC} == text_options == pixel_options
    assert all(options == {"copy": True, "layout": pyneat.TensorLayout.HWC} for _, options in runs[0])
    # The scripts prepare every input, so Neat must not resize, convert or normalize it again.
    for options in models:
        assert options.preprocess.kind == pyneat.InputKind.Tensor
        for step in (options.preprocess.resize, options.preprocess.color_convert, options.preprocess.normalize):
            assert step.enable == pyneat.AutoFlag.Off


class _Recorder:
    """Record a chain of builder calls and return itself from each."""

    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs)) or self


def test_compile_quantizes_and_compiles_with_the_bf16_afe_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    recorder, importer, compiled = _Recorder(), {}, {}
    monkeypatch.setattr(compile_module, "default_quantization", recorder)
    monkeypatch.setattr(compile_module, "HistogramMSEMethod", lambda bins: ("histogram_mse", bins))
    monkeypatch.setattr(compile_module, "onnx_source", lambda path, **options: importer.update(options) or path)
    bf16 = compile_module.bfloat16_scheme()

    compile_module._quantize_component(tmp_path / "graph.onnx", tmp_path, "text_encoder", "bf16", 2048, {})

    assert importer == {"shape_dict": {}, "layout": "NCHW", "flexible_batch_size": False}
    assert [(name, args) for name, args, _ in recorder.calls] == [
        ("with_activation_quantization", (bf16,)),
        ("with_weight_quantization", (bf16,)),
        ("with_requantization_mode", (compile_module.RequantizationMode.sima,)),
        ("with_calibration", (("histogram_mse", 2048),)),
        ("with_bias_correction", (compile_module.BiasCorrectionType.NONE,)),
        ("with_channel_equalization", (False,)),
        ("with_smooth_quant", (False,)),
        ("with_prefer_int8_udf", (True,)),
        ("with_per_token_quant", (False,)),
    ]
    with pytest.raises(ValueError, match="unsupported precision"):
        compile_module._quantization("int8", 2048)

    def compile_mpk(output_path: str, **options) -> None:
        compiled.update(options)
        (Path(output_path) / "model_mpk.tar.gz").write_bytes(b"")

    monkeypatch.setattr(compile_module, "audit_mpk", lambda archive, component_id: None)
    compile_module._compile_component(SimpleNamespace(compile=compile_mpk), tmp_path, "text_encoder")

    assert (
        compiled.items()
        >= {
            "batch_size": 1,
            "l2_caching_mode": compile_module.L2CachingMode.NONE,
            "layer_norm_use_fp32_intermediates": True,
            "rms_norm_use_fp32_intermediates": False,
            "enable_filter_sharing": False,
            "layout_search_effort_level": 0,
        }.items()
    )


def _model_metadata() -> dict:
    return yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))


def _mpk_manifest(compile_module: ModuleType, component_id: str) -> dict:
    contract = compile_module.COMPONENT_INTERFACES[component_id]
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": name, "size": size} for name, size in contract["inputs"].items()],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in contract["outputs"]],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict, names: tuple[str, ...] = ("model.elf",)) -> None:
    files = {name: b"" for name in names} | {"model_mpk.json": json.dumps(manifest).encode()}
    with tarfile.open(path, "w:gz") as stream:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize(
    ("component_id", "case"),
    [
        ("text_encoder", "valid"),
        ("image_model", "valid"),
        ("image_model", "reordered"),
        ("image_model", "inputs"),
        ("image_model", "two_mla"),
        ("image_model", "host"),
        ("image_model", "shared_object"),
        ("image_model", "compiler"),
    ],
)
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component_id: str, case: str
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    manifest = _mpk_manifest(compile_module, component_id)
    plugins, names = manifest["plugins"], ("model.elf",)
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[1] = outputs[1], outputs[0]
    elif case == "inputs":
        # A BF16 input instead of the float32 interface.
        manifest["input_nodes"][0]["size"] //= 2
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    elif case == "shared_object":
        names += ("libpost.so",)
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest, names)

    if case == "valid":
        assert compile_module.audit_mpk(archive, component_id)["mla_plugins"] == 1
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive, component_id)


@pytest.mark.parametrize(
    ("field", "value", "message"), [("offset", 15, "begin after the calibration"), ("seed", 1, "calibration seed 0")]
)
def test_compile_requires_the_fidelity_selection_to_follow_the_calibration_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: int, message: str
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    metadata = _model_metadata()
    metadata["fidelity"][0]["selection"][field] = value
    monkeypatch.setattr(compile_module, "_model", lambda: metadata)

    with pytest.raises(ValueError, match=message):
        compile_module.compile_model(tmp_path / "model.onnx", tmp_path, tmp_path, tmp_path / "out")


def test_graph_preparation_saves_the_float_reference_from_the_split_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    reference = {"detections": np.ones((4, 200, 6), np.float32), "mask_logits": np.zeros((4, 200, 2, 2), np.float32)}
    calls = []
    monkeypatch.setattr(compile_module, "prepare_model", lambda source, output_dir: ("text", "image"))
    monkeypatch.setattr(compile_module, "calibration_inputs", lambda dataset, names: (["cat"] * 4, "images"))
    monkeypatch.setattr(
        compile_module, "validate_float_split", lambda *args: calls.append(args) or ({"scores": 0.0}, reference)
    )

    compile_module._prepare_graphs_and_reference(Path("model.onnx"), Path("coco"), ["a"] * 4, tmp_path)

    assert calls == [(Path("model.onnx"), "text", "image", "images", ["cat"] * 4)]
    with np.load(tmp_path / "reference_outputs.npz") as saved:
        assert sorted(saved) == ["detections", "mask_logits"]
        np.testing.assert_array_equal(saved["detections"], reference["detections"])


def test_compile_runs_each_phase_in_a_spawned_process(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    contexts, exit_codes = [], [0, 3]

    class Process:
        def __init__(self, name: str, target, args: tuple) -> None:
            self.exitcode = exit_codes.pop(0)

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    monkeypatch.setattr(
        compile_module.multiprocessing,
        "get_context",
        lambda method: contexts.append(method) or SimpleNamespace(Process=Process),
    )

    compile_module._run_phase("first", print)
    with pytest.raises(RuntimeError, match="second failed with exit code 3"):
        compile_module._run_phase("second", print)
    assert contexts == ["spawn", "spawn"]


def test_compile_runs_the_phases_in_order_and_records_their_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    dataset = tmp_path / "coco"
    (dataset / "cal2017").mkdir(parents=True)
    for index in range(30):
        (dataset / "cal2017" / f"{index:012d}.jpg").write_bytes(b"")
    phases, calls = [], []

    def write_archive(temporary_dir: Path, component_id: str) -> None:
        build_dir = temporary_dir / f"efficientsam3_repvit_m_{component_id}_modalix_bf16"
        build_dir.mkdir()
        (build_dir / f"{component_id}_mpk.tar.gz").write_bytes(component_id.encode())

    def prepare(source, dataset, fidelity_names, temporary_dir) -> None:
        calls.append(("prepare", len(fidelity_names)))
        for path in compile_module.graph_paths(temporary_dir / "graphs"):
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(path.name.encode())

    def text_encoder(graph, temporary_dir, calibration_prompts, fidelity_prompts, precision, bins) -> None:
        calls.append(("text_encoder", graph.name, len(calibration_prompts), len(fidelity_prompts), precision, bins))
        write_archive(temporary_dir, "text_encoder")

    def image_model(graph, dataset, calibration_names, fidelity_names, temporary_dir, precision, bins, config) -> None:
        calls.append(("image_model", graph.name, set(calibration_names) & set(fidelity_names), precision, bins))
        write_archive(temporary_dir, "image_model")
        (temporary_dir / "fidelity.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(compile_module, "_prepare_graphs_and_reference", prepare)
    monkeypatch.setattr(compile_module, "_compile_text_encoder", text_encoder)
    monkeypatch.setattr(compile_module, "_compile_image_model", image_model)
    monkeypatch.setattr(compile_module, "_run_phase", lambda name, target, *args: phases.append(name) or target(*args))
    output_dir = tmp_path / "out"

    archives = compile_module.compile_model(tmp_path / "model.onnx", dataset, dataset, output_dir)

    assert phases == [
        "EfficientSAM3 graph preparation",
        "EfficientSAM3 text encoder compilation",
        "EfficientSAM3 image model compilation and fidelity",
    ]
    assert calls == [
        ("prepare", 4),
        ("text_encoder", "efficientsam3_repvit_m_text_encoder.onnx", 16, 4, "bf16", 2048),
        ("image_model", "efficientsam3_repvit_m_image_model.onnx", set(), "bf16", 2048),
    ]
    manifest = json.loads((output_dir / "compile-outputs.json").read_text(encoding="utf-8"))
    sha256 = compile_module.sha256_file
    assert manifest["outputs"] == [
        {
            "component_id": component_id,
            "mpk_path": f"{component_id}/{component_id}_mpk.tar.gz",
            "mpk_sha256": sha256(archives[component_id])[0],
        }
        for component_id in ("text_encoder", "image_model")
    ]
    assert [entry["component_id"] for entry in manifest["derived_graphs"]] == ["text_encoder", "image_model"]
    assert manifest["correctness"] == {
        "component_id": "image_model",
        "path": "image_model/fidelity.json",
        "sha256": sha256(output_dir / "image_model" / "fidelity.json")[0],
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--compile-invocation", "other", "--fidelity-dataset", "coco"],
        ["--compile-invocation", "efficientsam3"],
    ],
)
def test_compile_requires_its_invocation_and_a_fidelity_dataset(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    common = ["--model", "model.onnx", "--dataset", "coco", "--output-dir", "out", "--build-profile", "modalix_bf16"]
    monkeypatch.setattr(sys, "argv", ["compile.py", *common, *arguments])

    with pytest.raises(SystemExit):
        compile_module.main()


def test_model_declares_two_components_compiled_together() -> None:
    model = _model_metadata()
    profile = model["build_profiles"][0]
    components = {component["id"]: component for component in model["components"]}

    assert profile["compile_invocations"] == [{"id": "efficientsam3", "components": ["text_encoder", "image_model"]}]
    assert components["image_model"]["depends_on"] == ["text_encoder"]
    assert all(binding["parameters"]["calibration_samples"] == 16 for binding in profile["components"])
    assert model["fidelity"][0]["selection"]["offset"] == 16
    assert model["accuracy"][0]["required_metrics"] == ["coco_map", "coco_mask_map"]
    assert [entry["scope"]["component"] for entry in model["benchmark"]] == ["text_encoder", "image_model"]
