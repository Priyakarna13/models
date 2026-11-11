from __future__ import annotations

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
from onnx import helper, numpy_helper
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]


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


def _weights_graph(surgery: ModuleType, names: set[str]) -> onnx.ModelProto:
    """A graph holding the named weights as model.<name> initializers, zero-filled at their real shapes."""
    shapes = {"attn.qkv.weight": (3 * surgery.HIDDEN_SIZE, surgery.HIDDEN_SIZE)}
    initializers = [
        numpy_helper.from_array(np.zeros(shapes.get(name.split(".", 2)[-1], (1,)), np.float32), f"model.{name}")
        for name in sorted(names)
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "weights",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def test_surgery_reads_exactly_the_weights_it_rebuilds_from(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    names = surgery.weight_names()
    assert len(names) == 15 * surgery.DEPTH + 6
    path = tmp_path / "weights.onnx"
    onnx.save(_weights_graph(surgery, names), path)
    assert set(surgery.load_weights(path)) == names

    onnx.save(_weights_graph(surgery, names - {"blocks.3.gamma_2"}), path)
    with pytest.raises(ValueError, match="missing"):
        surgery.load_weights(path)


def test_surgery_rope_tables_follow_dinov3(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    sin, cos = surgery.rope_tables()

    assert sin.shape == cos.shape == (surgery.PATCHES, surgery.HEAD_DIM)
    np.testing.assert_allclose(sin**2 + cos**2, 1.0, atol=1e-6)
    # The top-left patch sits at (-13/14, -13/14); its first channel turns once per period 1, and channel 16 repeats
    # channel 0 for rotate-half.
    angle = 2 * np.pi * (2 * 0.5 / 14 - 1)
    np.testing.assert_allclose([sin[0, 0], cos[0, 0]], [np.sin(angle), np.cos(angle)], atol=1e-6)
    # Channel i turns once per period temperature^(i/16): 100^(1/16) for channel 1.
    np.testing.assert_allclose(sin[0, 1], np.sin(angle / 100 ** (1 / 16)), atol=1e-6)
    np.testing.assert_array_equal(sin[:, : surgery.HEAD_DIM // 2], sin[:, surgery.HEAD_DIM // 2 :])
    # Moving one column right changes the column angle (channel 16 of the half) but not the row angle (channel 0).
    assert sin[1, 0] == sin[0, 0] and sin[1, 16] != sin[0, 16]


def test_surgery_rejects_a_prepared_graph_that_still_computes_rope(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    shapes = {
        "class_embedding": [1, surgery.HIDDEN_SIZE],
        "patch_embeddings": [1, surgery.PATCHES, surgery.HIDDEN_SIZE],
    }
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in shapes.items()]
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.zeros(shape, np.float32)))
        for name, shape in shapes.items()
    ]
    image = helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, list(surgery.INPUT_SHAPE))
    for case in ("static", "rope"):
        extra = [helper.make_node("Sin", ["pixel_values"], ["angle"])] if case == "rope" else []
        model = helper.make_model(
            helper.make_graph(nodes + extra, case, [image], outputs),
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        path = tmp_path / f"{case}.onnx"
        onnx.save(model, path)
        if case == "static":
            surgery.validate_prepared(path)
        else:
            with pytest.raises(ValueError, match="Sin"):
                surgery.validate_prepared(path)


def test_surgery_requires_the_prepared_graph_to_match_the_converted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "INPUT_SHAPE", (1, 64))

    def graph(scale: float) -> Path:
        nodes = [
            helper.make_node("Mul", ["pixel_values", "scale"], ["class_embedding"]),
            helper.make_node("Neg", ["pixel_values"], ["patch_embeddings"]),
        ]
        values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 64]) for name in surgery.OUTPUTS]
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "embeddings",
                [helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, [1, 64])],
                values,
                [numpy_helper.from_array(np.asarray(scale, np.float32), "scale")],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        path = tmp_path / f"scale_{scale}.onnx"
        onnx.save(model, path)
        return path

    # A scale changes no direction; a sign flip does.
    surgery.check_equivalent(graph(1.0), graph(2.0))
    with pytest.raises(ValueError, match="below"):
        surgery.check_equivalent(graph(1.0), graph(-1.0))


def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()

    for profile in metadata["build_profiles"]:
        parameters = recipe._variant(metadata, profile["id"])["parameters"]
        assert parameters["precision"] == profile["id"]
        assert int(parameters["calibration_samples"]) <= metadata["numerical_fidelity"]["selection"]["offset"]
    assert metadata["numerical_fidelity"]["outputs"] == list(recipe.OUTPUTS)
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._variant(metadata, "w32a32")


@pytest.mark.parametrize("precision", ["w8a8", "w8a16", "w16a16"])
def test_the_quantization_recipe_follows_the_precision(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    for name in ("quantization_scheme", "bfloat16_scheme", "HistogramMSEMethod", "default_quantization"):
        monkeypatch.setattr(recipe, name, MagicMock(name=name))
    recipe._quantization_config({"precision": precision, "calibration_bins": 2048})
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
    # Only W8A16 moves activation outliers into the weights with SmoothQuant and scales each token on its own.
    chain = recipe.default_quantization.with_activation_quantization.return_value.with_weight_quantization.return_value
    chain = chain.with_requantization_mode.return_value.with_calibration.return_value
    chain = chain.with_channel_equalization.return_value
    chain.with_smooth_quant.assert_called_once_with(precision == "w8a16")
    chain.with_smooth_quant.return_value.with_per_token_quant.assert_called_once_with(precision == "w8a16")
    # Every precision calibrates.
    recipe.HistogramMSEMethod.assert_called_once_with(2048)
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
        (
            ["--build-profile", "w16a16", "--compile-invocation", "w16a16", "--fidelity-dataset", "cal"],
            "pass --dataset",
        ),
        (
            ["--build-profile", "w16a16", "--compile-invocation", "w16a16", "--dataset", "cal"],
            "pass --fidelity-dataset",
        ),
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
    assert [metric["name"] for metric in fidelity["metrics"]] == list(recipe.OUTPUTS)
    # BF16 keeps every output within rounding of the ONNX, and INT8 weights add a little more error; a broken
    # operator or a collapsed quantization (w8a8 measured 0.01) shows up as a low cosine.
    floor = {"w16a16": 0.99, "w8a16": 0.98}.get(build_profile)
    if floor is not None:
        assert all(metric["cosine_similarity"] > floor for metric in fidelity["metrics"]), fidelity["metrics"]
