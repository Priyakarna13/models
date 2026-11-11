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
COMPONENTS = ["blocks_0", "blocks_1", "blocks_2", "blocks_3"]


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


def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()

    for profile in metadata["build_profiles"]:
        parameters = recipe._component_parameters(metadata, profile["id"])
        assert list(parameters) == COMPONENTS
        for component in COMPONENTS:
            assert parameters[component]["precision"] == profile["id"]
            assert (
                int(parameters[component]["calibration_samples"])
                <= metadata["numerical_fidelity"]["selection"]["offset"]
            )
    assert metadata["numerical_fidelity"]["outputs"] == list(recipe.OUTPUTS)
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._component_parameters(metadata, "w32a32")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda components: components.reverse(), "must bind blocks_0, blocks_1, blocks_2, blocks_3 in order"),
        (lambda components: components[0]["parameters"].update(precision="fp32"), "Modalix BF16 and INT8"),
        (lambda components: components[1]["parameters"].update(calibration_samples=8), "same images"),
    ],
)
def test_compile_rejects_an_undeclared_component_setting(
    load_script: Callable[[str], ModuleType], change: Callable[[list], None], message: str
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    change(metadata["build_profiles"][0]["components"])

    with pytest.raises(ValueError, match=message):
        recipe._component_parameters(metadata, metadata["build_profiles"][0]["id"])


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


def test_tokens_cross_to_afe_channels_last_and_back(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    tokens = np.arange(1 * 1280 * 1 * 201, dtype=np.float32).reshape(1, 1280, 1, 201)

    afe = recipe.to_afe(tokens)
    assert afe.shape == (1, 1, 201, 1280) and afe.flags["C_CONTIGUOUS"]
    (back,) = recipe.from_afe("blocks_1", [afe])
    np.testing.assert_array_equal(back, tokens)
    class_embedding, patches = np.ones((1, 1280), np.float32), np.zeros((1, 196, 1280), np.float32)
    outputs = recipe.from_afe("blocks_3", [patches, class_embedding])
    assert [output.shape for output in outputs] == [(1, 1280), (1, 196, 1280)]
    with pytest.raises(RuntimeError, match="returned no patch_embeddings"):
        recipe.from_afe("blocks_3", [class_embedding])


def test_compile_takes_the_converted_model(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]["id"]
    for name in ("blocks_0.onnx", "model.onnx.data"):
        with pytest.raises(ValueError, match="--model must be the converted model.onnx"):
            recipe.compile_model(tmp_path / name, tmp_path / "out", profile, tmp_path, tmp_path)
    assert not (tmp_path / "out").exists()


def _component_graph(surgery: ModuleType, component: str, case: str) -> onnx.ModelProto:
    """A graph with the component's interface, or one broken in the named way; each output is a constant."""
    shapes = {name: list(shape) for name, shape in surgery.INTERFACES[component].items()}
    names = list(shapes)
    if case == "names":
        names[1] = "last_hidden_state"
    elif case == "shape":
        shapes[names[-1]][1] += 1
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in zip(names, shapes.values())
    ]
    nodes = [
        helper.make_node("Constant", [], [value.name], value=numpy_helper.from_array(np.zeros(shape, np.float32)))
        for value, shape in zip(values[1:], list(shapes.values())[1:])
    ]
    if case == "rope":
        nodes.append(helper.make_node("Cos", [names[0]], ["angle"]))
    graph = helper.make_graph(nodes, component, values[:1], values[1:])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


@pytest.mark.parametrize("component", ["blocks_0", "blocks_1", "blocks_3"])
@pytest.mark.parametrize("case", ["valid", "names", "shape", "rope"])
def test_surgery_checks_each_components_interface(
    tmp_path: Path, load_script: Callable[[str], ModuleType], component: str, case: str
) -> None:
    surgery = load_script("surgery")
    path = tmp_path / f"{component}.onnx"
    onnx.save(_component_graph(surgery, component, case), path)
    if case == "valid":
        surgery.validate_prepared(path, component)
    else:
        with pytest.raises(ValueError):
            surgery.validate_prepared(path, component)


def test_surgery_keeps_each_component_one_self_contained_file_of_at_most_2_gb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    model = _component_graph(surgery, "blocks_1", "valid")
    path = tmp_path / "blocks_1.onnx"
    onnx.save(model, path)
    monkeypatch.setattr(surgery, "MAX_ONNX_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match="larger than one ONNX file holds"):
        surgery.validate_prepared(path, "blocks_1")
    monkeypatch.undo()
    external = tmp_path / "external.onnx"
    model.graph.initializer.append(numpy_helper.from_array(np.zeros(1024, np.float32), "weights"))
    onnx.save(model, external, save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=0)
    with pytest.raises(ValueError, match="self-contained"):
        surgery.validate_prepared(external, "blocks_1")


def test_surgery_reads_exactly_the_weights_it_rebuilds_from(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    names = surgery.weight_names()
    assert len(names) == 15 * surgery.DEPTH + 6 == 486

    def save(selected: set[str]) -> Path:
        shapes = {"attn.qkv.weight": (3 * surgery.HIDDEN_SIZE, surgery.HIDDEN_SIZE)}
        initializers = [
            numpy_helper.from_array(np.zeros(shapes.get(name.split(".", 2)[-1], (1,)), np.float32), f"model.{name}")
            for name in sorted(selected)
        ]
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "weights",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
            initializers,
        )
        path = tmp_path / "weights.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)
        return path

    assert set(surgery.load_weights(save(names))) == names
    with pytest.raises(ValueError, match="missing"):
        surgery.load_weights(save(names - {"blocks.31.mlp.fc2.bias"}))


def test_surgery_rope_tables_follow_dinov3(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    sin, cos = surgery.rope_tables()

    assert sin.shape == cos.shape == (196, 64)
    np.testing.assert_allclose(sin**2 + cos**2, 1.0, atol=1e-6)
    # The top-left patch sits at (-13/14, -13/14); channel i turns once per period 100^(i/16), and channel 32 repeats
    # channel 0 for rotate-half.
    angle = 2 * np.pi * (2 * 0.5 / 14 - 1)
    np.testing.assert_allclose([sin[0, 0], cos[0, 0]], [np.sin(angle), np.cos(angle)], atol=1e-6)
    np.testing.assert_allclose(sin[0, 1], np.sin(angle / 100 ** (1 / 16)), atol=1e-6)
    np.testing.assert_array_equal(sin[:, :32], sin[:, 32:])
    # Moving one column right changes the column angle (channel 16) but not the row angle (channel 0).
    assert sin[1, 0] == sin[0, 0] and sin[1, 16] != sin[0, 16]


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


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_components_input(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    plugins = [{"processor": "EV74"}, {"processor": "MLA"}, {"processor": "EV74"}]
    image = [{"name": "pixel_values", "size": 3 * 224 * 224 * 4}]
    tokens = [{"name": "input_tokens", "size": 1280 * 201 * 4}]
    assert recipe.audit_mpk(_mpk(tmp_path / "first.tar.gz", plugins, image), "blocks_0")["mla_plugins"] == 1
    assert recipe.audit_mpk(_mpk(tmp_path / "later.tar.gz", plugins, tokens), "blocks_2")["mla_plugins"] == 1
    for name, arguments, component, message in (
        ("cpu", ([*plugins, {"processor": "A65"}], tokens), "blocks_1", "host=1"),
        ("input", (plugins, image), "blocks_1", "input interface"),
    ):
        with pytest.raises(RuntimeError, match=message):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", *arguments), component)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--build-profile", "w16a16", "--compile-invocation", "w8a16"], "named after it"),
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


def test_build_holds_one_audited_mpk_per_component_that_matches_the_onnx(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    manifest = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [output["component_id"] for output in manifest["outputs"]] == COMPONENTS
    for output in manifest["outputs"]:
        archive = compiled / output["mpk_path"]
        assert archive.name == f"dinov3_vit_h_plus_16_{build_profile}_{output['component_id']}_mpk.tar.gz"
        recipe.audit_mpk(archive, output["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    assert [metric["name"] for metric in fidelity["metrics"]] == list(recipe.OUTPUTS)
    # BF16 keeps every output within rounding of the ONNX, and INT8 weights add a little more error; a broken
    # operator or a collapsed quantization shows up as a low cosine.
    floor = {"w16a16": 0.99, "w8a16": 0.98}.get(build_profile)
    if floor is not None:
        assert all(metric["cosine_similarity"] > floor for metric in fidelity["metrics"]), fidelity["metrics"]


@pytest.mark.parametrize("stage", ["quantize", "compile"])
def test_a_compiler_exit_fails_the_component_instead_of_hanging_its_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], stage: str
) -> None:
    recipe = load_script("compile")
    loaded = MagicMock(name="loaded")
    quantized = loaded.quantize.return_value
    quantized.execute.return_value = [np.zeros((1, 1, 201, 1280), np.float32)]
    quantized._net.nodes = {"MLA_0": MagicMock(input_names=["input_tokens"])}
    if stage == "quantize":
        loaded.quantize.side_effect = SystemExit(1)
    else:
        quantized.compile.side_effect = SystemExit(1)
    monkeypatch.setattr(recipe, "load_model", MagicMock(return_value=loaded))
    for name in ("onnx_source", "convert_data_generator_to_iterable", "DataGenerator", "_quantization_config"):
        monkeypatch.setattr(recipe, name, MagicMock(name=name))
    tokens = np.zeros((1, 1280, 1, 201), np.float32)

    with pytest.raises(RuntimeError, match=f"{'quantizing' if stage == 'quantize' else 'compiling'} blocks_1 failed"):
        recipe._build(
            tmp_path / "blocks_1.onnx",
            tmp_path / "build",
            "name",
            "blocks_1",
            {"precision": "w8a16"},
            [tokens],
            [tokens],
        )
