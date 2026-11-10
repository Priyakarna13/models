from __future__ import annotations

import importlib.util
import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]
# Decided before load_script stubs a missing onnxruntime; CI's package environment installs it.
HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None


def _png(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="PNG")
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _mpk(path: Path, plugins: list[dict], inputs: list[dict]) -> Path:
    manifest = {"plugins": plugins, "input_nodes": inputs}
    return _write_archive(path, {"model_mpk.json": json.dumps(manifest).encode(), "model.elf": b"elf"})


def test_preprocessing_is_imagenets_mean_and_std(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.SIZE == 224
    assert calibration.RESCALE == pytest.approx(1 / 255)
    np.testing.assert_allclose(calibration.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(calibration.STD, [0.229, 0.224, 0.225])


def test_preprocessing_resizes_straight_to_the_model_input(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, squeezes to 224x224 without a crop: 112 rows of each.
    image = Image.new("RGB", (128, 256), (255, 0, 0))
    image.paste((0, 0, 255), (0, 128, 128, 256))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert (result.shape, result.dtype) == ((224, 224, 3), np.float32)
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[100, 112], red, atol=1e-5)
    np.testing.assert_allclose(result[124, 112], blue, atol=1e-5)


def test_selection_is_seeded_and_reads_an_archive(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    names = [f"imagenet/cal2012/n01440764_{index:02d}.JPEG" for index in range(6)]
    archive = _write_archive(tmp_path / "cal.tar.gz", {name: _png(300, 260) for name in names})
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.select_names(archive, 42, 2, 3) == expected[2:5]
    assert calibration.load_images(archive, expected[:2]).shape == (2, 224, 224, 3)
    with pytest.raises(ValueError, match="needs 7 images"):
        calibration.select_names(archive, 42, 1, 6)


def test_split_names_read_an_extracted_directory_and_require_the_split(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    (tmp_path / "cal2012").mkdir()
    (tmp_path / "cal2012" / "x.JPEG").write_bytes(_png(256, 256))

    assert calibration.split_names(tmp_path, "cal2012") == ["cal2012/x.JPEG"]
    with pytest.raises(ValueError, match="no images under val2012/"):
        calibration.split_names(tmp_path, "val2012")


def test_read_images_reports_a_missing_image(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(tmp_path / "cal.tar.gz", {"imagenet/cal2012/a.JPEG": b"a"})

    assert calibration.read_images(archive, ["imagenet/cal2012/a.JPEG"]) == {"imagenet/cal2012/a.JPEG": b"a"}
    with pytest.raises(ValueError, match="missing 1 requested images"):
        calibration.read_images(archive, ["imagenet/cal2012/a.JPEG", "imagenet/cal2012/b.JPEG"])


def _export_like_model(surgery: ModuleType) -> onnx.ModelProto:
    """An opset-20 graph with the export's patterns: a SplitToSequence of the prefix and patch tokens read by
    SequenceAt, an exact Gelu, and the full token and pooled outputs."""
    helper = onnx.helper
    tokens = [1, surgery.PREFIX + surgery.PATCHES, surgery.HIDDEN_SIZE]
    constants = {
        "sizes": np.array([surgery.PREFIX, surgery.PATCHES], dtype=np.int64),
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
         helper.make_tensor_value_info("pooler_output", onnx.TensorProto.FLOAT, [1, surgery.HIDDEN_SIZE])],
        [numpy_helper.from_array(value, name) for name, value in constants.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def test_the_rewrites_keep_the_exports_embeddings(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    # ONNX's own reference evaluator, so the check runs without ONNX Runtime.
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    path = tmp_path / "model.onnx"
    onnx.save(_export_like_model(surgery), path)
    tokens = np.random.default_rng(0).normal(size=(1, surgery.PREFIX + surgery.PATCHES, surgery.HIDDEN_SIZE))
    feed = {"tokens": tokens.astype(np.float32)}
    expected = surgery.embeddings(*ReferenceEvaluator(str(path)).run(list(surgery.SOURCE_OUTPUTS), feed))

    surgery.rewrite(path)

    model = onnx.load(path)
    assert [item.version for item in model.opset_import if not item.domain] == [surgery.OPSET]
    assert not {node.op_type for node in model.graph.node} & set(surgery.DYNAMIC_OPS)
    for name, actual in zip(surgery.OUTPUTS, ReferenceEvaluator(model).run(list(surgery.OUTPUTS), feed), strict=True):
        np.testing.assert_allclose(actual, expected[name], rtol=1e-5, atol=1e-6, err_msg=name)


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="the folding runs ONNX Runtime")
def test_folding_turns_shape_computations_and_constants_into_initializers(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    helper = onnx.helper
    width = surgery.INPUT_SHAPE[-1]
    nodes = [
        helper.make_node("Shape", ["pixel_values"], ["shape"]),
        helper.make_node("Reshape", ["pixel_values", "shape"], ["same"]),
        helper.make_node("Transpose", ["weight_t"], ["weight"], perm=[1, 0]),
        helper.make_node("MatMul", ["same", "weight"], ["projected"]),
    ]
    weight_t = np.linspace(-1.0, 1.0, 2 * width, dtype=np.float32).reshape(2, width)
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, list(surgery.INPUT_SHAPE))],
        [helper.make_tensor_value_info("projected", onnx.TensorProto.FLOAT, None)],
        [numpy_helper.from_array(weight_t, "weight_t")],
    )
    source, folded = tmp_path / "source.onnx", tmp_path / "folded.onnx"
    # IR 10 is the newest that the registry's test environment, ONNX Runtime 1.21, loads.
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), source)

    surgery.fold_constants(source, folded)

    model = onnx.load(folded)
    assert [node.op_type for node in model.graph.node] == ["Reshape", "MatMul"]
    assert sorted(tensor.name for tensor in model.graph.initializer) == ["shape", "weight"]
    image = np.random.default_rng(0).normal(size=surgery.INPUT_SHAPE).astype(np.float32)
    (expected,) = ReferenceEvaluator(str(source)).run(None, {"pixel_values": image})
    (actual,) = ReferenceEvaluator(model).run(None, {"pixel_values": image})
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def _prepared_model(surgery: ModuleType, case: str) -> onnx.ModelProto:
    helper = onnx.helper
    image = helper.make_tensor_value_info(surgery.INPUT, onnx.TensorProto.FLOAT, list(surgery.INPUT_SHAPE))
    outputs = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.OUTPUT_SHAPES.items()
    ]
    if case == "order":
        outputs.reverse()
    elif case == "shape":
        outputs[1].type.tensor_type.shape.dim[1].dim_value = surgery.PREFIX + surgery.PATCHES
    nodes = []
    for output in outputs:
        zeros = np.zeros([d.dim_value for d in output.type.tensor_type.shape.dim], np.float32)
        nodes.append(helper.make_node("Constant", [], [output.name], value=numpy_helper.from_array(zeros)))
    if case == "dynamic":
        nodes.append(helper.make_node("Shape", [surgery.INPUT], ["shape"]))
    graph = helper.make_graph(nodes, "prepared", [image], outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20 if case == "opset" else surgery.OPSET)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "shape", "dynamic"])
def test_the_prepared_graph_interface(tmp_path: Path, load_script: Callable[[str], ModuleType], case: str) -> None:
    surgery = load_script("surgery")
    path = tmp_path / "prepared.onnx"
    onnx.save(_prepared_model(surgery, case), path)
    if case == "valid":
        surgery.validate_prepared(path)
    else:
        with pytest.raises(ValueError):
            surgery.validate_prepared(path)


def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()

    for profile in metadata["build_profiles"]:
        parameters = recipe._variant(metadata, profile["id"])["parameters"]
        assert parameters["precision"] == profile["id"]
        # Only INT8 activations calibrate, on images ahead of the fidelity images.
        if profile["id"] == "w8a8":
            assert int(parameters["calibration_samples"]) <= metadata["numerical_fidelity"]["selection"]["offset"]
        else:
            assert parameters == {"precision": profile["id"]}
    assert metadata["numerical_fidelity"]["outputs"] == list(recipe.OUTPUT_SHAPES)
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._variant(metadata, "w32a32")


@pytest.mark.parametrize("precision", ["w8a8", "w8a16", "w16a16"])
def test_the_quantization_recipe_follows_the_precision(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    stubbed = (
        "quantization_scheme", "bfloat16_scheme", "HistogramMSEMethod", "SkipCalibration", "default_quantization",
    )
    for name in stubbed:
        monkeypatch.setattr(recipe, name, MagicMock(name=name))
    parameters = {"precision": precision, "calibration_bins": 2048} if precision == "w8a8" else {"precision": precision}
    recipe._quantization_config(parameters)
    if precision == "w8a8":
        # INT8 activations per tensor, INT8 weights per channel.
        assert recipe.quantization_scheme.call_args_list == [
            call(asymmetric=True, per_channel=False, bits=8),
            call(asymmetric=False, per_channel=True, bits=8),
        ]
        recipe.bfloat16_scheme.assert_not_called()
    elif precision == "w8a16":
        # BF16 activations, INT8 weights per channel.
        assert recipe.quantization_scheme.call_args_list == [call(asymmetric=False, per_channel=True, bits=8)]
        recipe.bfloat16_scheme.assert_called_once_with()
    else:
        recipe.quantization_scheme.assert_not_called()
        recipe.bfloat16_scheme.assert_called_once_with()
    # Only W8A16 scales each token's activations on its own.
    chain = recipe.default_quantization.with_activation_quantization.return_value.with_weight_quantization.return_value
    chain = chain.with_requantization_mode.return_value.with_calibration.return_value
    chain = chain.with_channel_equalization.return_value.with_smooth_quant.return_value
    chain.with_per_token_quant.assert_called_once_with(precision == "w8a16")
    # Only INT8 activations calibrate; BF16 builds skip calibration.
    if precision == "w8a8":
        recipe.HistogramMSEMethod.assert_called_once_with(2048)
        recipe.SkipCalibration.assert_not_called()
    else:
        recipe.HistogramMSEMethod.assert_not_called()
        recipe.SkipCalibration.assert_called_once_with()
    with pytest.raises(ValueError, match="only the declared Modalix BF16 and INT8 variants"):
        recipe._quantization_config({"precision": "w4a8", "calibration_bins": 2048})


def test_compile_rejects_an_undeclared_precision_before_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["components"][0]["parameters"]["precision"] = "fp32"
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match="only the declared Modalix BF16 and INT8 variants"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile["id"], tmp_path / "cal.tar.gz")
    assert not (tmp_path / "model").exists()


def test_the_int8_build_needs_its_calibration_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["id"] = "w8a8"
    profile["components"][0]["parameters"] = {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 50}
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match="pass --dataset"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w8a8", None)
    assert not (tmp_path / "model").exists()


def test_fidelity_error_sums_match_the_metrics_over_all_samples(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    error = recipe._OutputError("class_embedding")
    error.add(np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]]))
    error.add(np.asarray([[0.0, 2.0]]), np.asarray([[0.0, 1.0]]))
    summary = error.summary()

    assert summary["name"] == "class_embedding"
    assert summary["cosine_similarity"] == pytest.approx(3 / np.sqrt(5 * 2))
    assert summary["max_absolute_error"] == pytest.approx(1.0)
    assert summary["mean_absolute_error"] == pytest.approx(0.25)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        error.add(np.zeros((1, 2)), np.zeros((1, 3)))


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_input(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    plugins = [{"processor": "EV74"}, {"processor": "MLA"}, {"processor": "EV74"}]
    inputs = [{"name": "pixel_values", "size": 3 * 224 * 224 * 4}]
    assert recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", plugins, inputs))["mla_plugins"] == 1
    for name, arguments, message in (
        ("cpu", ([*plugins, {"processor": "A65"}], inputs), "host=1"),
        ("input", (plugins, [{"name": "pixel_values", "size": 4}]), "input interface"),
    ):
        with pytest.raises(RuntimeError, match=message):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", *arguments))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--build-profile", "w16a16", "--compile-invocation", "w8a8"], "named after it"),
        (["--build-profile", "w16a16", "--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_incomplete_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    base = ["compile.py", "--model", str(tmp_path / "model.onnx"), "--output-dir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", base + arguments)

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_build_holds_one_audited_mpk_that_matches_the_onnx(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    (archive,) = sorted((compiled / "model").glob("*_mpk.tar.gz"))
    recipe.audit_mpk(archive)
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    assert [metric["name"] for metric in fidelity["metrics"]] == list(recipe.OUTPUT_SHAPES)
    # BF16 keeps every output within rounding of the ONNX, and INT8 weights add a little more error; a broken
    # operator or a collapsed quantization (w8a8 measured 0.01) shows up as a low cosine.
    floor = {"w16a16": 0.99, "w8a16": 0.98}.get(build_profile)
    if floor is not None:
        assert all(metric["cosine_similarity"] > floor for metric in fidelity["metrics"]), fidelity["metrics"]
