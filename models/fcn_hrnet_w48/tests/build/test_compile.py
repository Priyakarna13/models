from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image


def _jpeg(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_stretches_to_512_and_normalizes_in_rgb(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, stretches without cropping: each colour keeps half the rows.
    image = Image.new("RGB", (100, 300), (255, 0, 0))
    image.paste((0, 0, 255), (0, 150, 100, 300))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert result.shape == (512, 512, 3)
    assert result.dtype == np.float32
    red = (np.asarray([255, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 255], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[200, 256], red, atol=1e-5)
    np.testing.assert_allclose(result[312, 256], blue, atol=1e-5)


def test_normalization_is_the_imagenet_statistics_in_pixel_units(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    np.testing.assert_allclose(calibration.MEAN, np.asarray([0.485, 0.456, 0.406]) * 255, rtol=1e-6)
    np.testing.assert_allclose(calibration.STD, np.asarray([0.229, 0.224, 0.225]) * 255, rtol=1e-6)


def test_calibration_draws_a_seeded_sample_from_the_training_images_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    # Each training image has its own red level, so the stacked samples show which images the seed picked.
    names = [f"ade20k-cal/images/training/ADE_train_{index:08d}.jpg" for index in range(6)]
    files = {name: _jpeg(64, 48, (40 * index, 0, 0)) for index, name in enumerate(names)}
    files["ade20k-cal/images/validation/ADE_val_00000001.jpg"] = _jpeg(64, 48, (255, 255, 255))
    archive = _write_archive(tmp_path / "ade20k.tgz", files)

    samples = calibration.calibration_samples(archive, 4)

    assert samples.shape == (4, 512, 512, 3)
    reds = samples[:, 256, 256, 0] * calibration.STD[0] + calibration.MEAN[0]
    np.testing.assert_allclose(reds, [40 * names.index(name) for name in random.Random(42).sample(names, 4)], atol=2)
    with pytest.raises(ValueError, match="cannot select 7 calibration images from 6"):
        calibration.calibration_samples(archive, 7)


def test_fidelity_selection_is_the_declared_seeded_training_sample(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"ade20k-cal/images/training/ADE_train_{index:08d}.jpg" for index in range(10)]
    for name in [*names, "ade20k-cal/images/validation/ADE_val_00000001.jpg"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 6, 5)


def test_read_images_reports_a_missing_image(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    name = "ade20k-cal/images/training/a.jpg"
    archive = _write_archive(tmp_path / "ade20k.tgz", {name: b"a"})

    assert calibration.read_images(archive, [name]) == {name: b"a"}
    with pytest.raises(ValueError, match="missing 1 requested images"):
        calibration.read_images(archive, [name, "ade20k-cal/images/training/b.jpg"])


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe._parameters(profile_id)

    assert parameters["precision"] == profile_id
    if profile_id == "w8a8":
        assert (parameters["calibration_bins"], parameters["calibration_samples"]) == (2048, 50)


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="no build profile 'w8a16'"):
        recipe._parameters("w8a16")


def test_logits_come_back_in_onnx_order(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    logits = np.random.RandomState(0).normal(size=recipe.LOGITS_SHAPE)

    np.testing.assert_array_equal(recipe._logits([logits.transpose(0, 2, 3, 1)]), logits)
    np.testing.assert_array_equal(recipe._logits([logits]), logits)
    with pytest.raises(RuntimeError, match="unexpected logits shape"):
        recipe._logits([logits[:, :, :64]])


def test_mpk_interface_is_the_float32_onnx_interface(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    assert recipe.INPUT_BYTES == {"image": 3 * 512 * 512 * 4}
    assert recipe.OUTPUT_BYTES == [150 * 128 * 128 * 4]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_unknown_invocation_or_missing_fidelity_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"), "--build-profile", "w16a16"]
        + arguments,
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(("precision", "calibrates"), [("w8a8", True), ("w16a16", False)])
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048 if calibrates else None)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.HistogramMSEMethod.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration
    if calibrates:
        recipe.HistogramMSEMethod.assert_called_once_with(2048)
        assert arguments["biascorr_type"] is recipe.BiasCorrectionType.REGULAR


def test_quantization_rejects_an_undeclared_precision(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="unsupported precision: w8a16"):
        recipe._quantization("w8a16", None)


@pytest.mark.parametrize("precision", ["w16a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    drawn = []

    def samples(dataset: Path, count: int) -> np.ndarray:
        drawn.append(count)
        return np.zeros((count, 4, 4, 3), np.float32)

    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(recipe, "calibration_samples", samples)

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "ade20k.tgz")

    data = recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    if precision == "w8a8":
        assert drawn == [2]
        assert data is recipe.convert_data_generator_to_iterable.return_value
    else:
        assert drawn == []
        assert data is None


def test_the_int8_build_requires_the_calibration_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)

    with pytest.raises(ValueError, match="requires --dataset"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", "w8a8", None)
    assert not (tmp_path / "out").exists()


def _mpk_manifest(recipe: ModuleType) -> dict:
    (input_size,) = recipe.INPUT_BYTES.values()
    return {
        "input_nodes": [{"name": recipe.INPUT, "size": input_size}],
        "plugins": [
            {"sequence": 1, "processor": "EV74"},
            {"sequence": 2, "processor": "MLA"},
            {"sequence": 3, "processor": "EV74"},
            {"sequence": 4, "processor": "EV74", "output_nodes": [{"size": size} for size in recipe.OUTPUT_BYTES]},
        ],
    }


def _write_mpk(path: Path, manifest: dict, extra: tuple[str, ...] = ()) -> None:
    members = {"model.elf": b"elf", "model_mpk.json": json.dumps(manifest).encode(), **{name: b"" for name in extra}}
    with tarfile.open(path, "w:gz") as stream:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_mla", "host", "shared_object", "logits", "input"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    manifest = _mpk_manifest(recipe)
    plugins = manifest["plugins"]
    if case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA"})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65"})
    elif case == "logits":
        plugins[-1]["output_nodes"][0]["size"] //= 2
    elif case == "input":
        manifest["input_nodes"][0]["name"] = "input"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest, ("libhost.so",) if case == "shared_object" else ())

    if case == "valid":
        recipe.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(archive)
