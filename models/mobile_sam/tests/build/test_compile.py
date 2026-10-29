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


def _jpeg(width: int, height: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_resizes_the_longer_side_normalizes_and_pads(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    image = np.full((300, 400, 3), 128, np.uint8)

    result = pipeline.preprocess(image)

    assert (result.shape, result.dtype) == ((1024, 1024, 3), np.float32)
    assert pipeline.resized_size(300, 400) == (768, 1024)
    np.testing.assert_allclose(result[100, 100], (128 - pipeline.PIXEL_MEAN) / pipeline.PIXEL_STD, atol=1e-5)
    # Padding below the resized image is zero, the normalized mean.
    assert (result[768:] == 0).all()


def test_prompt_inputs_normalize_pixel_centres_and_pad_the_slots(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")

    inputs = pipeline.prompt_inputs([(0, 0), (399, 299)], ["foreground", "background"], (300, 400))

    coords, types = inputs["point_coords"][0, :, 0, :].T, inputs["point_types"][0, :, 0, :].T
    assert inputs["point_coords"].shape == (1, 2, 1, pipeline.POINTS)
    assert inputs["point_types"].shape == (1, len(pipeline.POINT_TYPES), 1, pipeline.POINTS)
    # Pixel (0, 0) is the centre of the resized image's first pixel; (399, 299) scales by 1024 / 400.
    np.testing.assert_allclose(coords[0], [0.5 / 1024 * 2 - 1] * 2)
    np.testing.assert_allclose(coords[1], [(399 * 2.56 + 0.5) / 512 - 1, (299 * 2.56 + 0.5) / 512 - 1], rtol=1e-6)
    assert (coords[2:] == 0).all()
    assert [pipeline.POINT_TYPES[index] for index in types.argmax(1)] == [
        "foreground",
        "background",
        "padding",
        "padding",
        "padding",
    ]


@pytest.mark.parametrize(
    ("points", "types", "message"),
    [
        ([(1, 1)] * 6, ["foreground"] * 6, "1 to 5 points"),
        ([(1, 1)], ["foreground", "background"], "1 to 5 points"),
        ([(1, 1)], ["padding"], "unknown point types"),
    ],
)
def test_prompt_inputs_reject_bad_prompts(points, types, message, load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")

    with pytest.raises(ValueError, match=message):
        pipeline.prompt_inputs(points, types, (300, 400))


def test_postprocessing_returns_the_original_size(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    masks = np.full((1, 4, 256, 256), -1, np.float32)
    # The top-left quarter of the 1024x1024 input is positive.
    masks[:, :, :128, :128] = 1

    result = pipeline.postprocess_masks(masks, (300, 400))

    assert result.shape == (4, 300, 400)
    # The resized image is 768x1024, so the positive 512x512 corner is 200x200 of the original.
    assert (result[:, :190, :190] > 0).all() and (result[:, 210:, 210:] < 0).all()


def test_fixed_prompts_cover_a_click_and_a_box(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    pipeline = load_script("pipeline")

    click, box = calibration.image_prompts((480, 640))

    assert [pipeline.POINT_TYPES[i] for i in click["point_types"][0, :, 0, :].T.argmax(1)][:1] == ["foreground"]
    assert [pipeline.POINT_TYPES[i] for i in box["point_types"][0, :, 0, :].T.argmax(1)][:3] == [
        "box_top_left",
        "box_bottom_right",
        "background",
    ]


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    files = {f"cal2017/{index:02d}.jpg": _jpeg(64, 48) for index in range(6)}
    files["val2017/x.jpg"] = _jpeg(64, 48)
    archive = _write_archive(tmp_path / "coco.tgz", files)

    images = calibration.calibration_images(archive, 4)

    assert [image.shape for image in images] == [(48, 64, 3)] * 4
    with pytest.raises(ValueError, match="cannot select 7 calibration images from 6"):
        calibration.calibration_images(archive, 7)


def test_fidelity_selection_is_the_declared_seeded_val2017_sample(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"val2017/{index:02d}.jpg" for index in range(10)]
    for name in [*names, "cal2017/c.jpg"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 1, 10)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_binds_both_components_in_each_profile(
    load_script: Callable[[str], ModuleType], profile_id: str
) -> None:
    recipe = load_script("compile")
    parameters = recipe._component_parameters(profile_id)

    assert list(parameters) == ["image_encoder", "prompt_decoder"]
    assert {item["precision"] for item in parameters.values()} == {profile_id}
    with pytest.raises(ValueError, match="no build profile 'w8a16'"):
        recipe._component_parameters("w8a16")


def test_outputs_come_back_in_onnx_order(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    embeddings = np.random.RandomState(0).normal(size=(1, 256, 64, 64)).astype(np.float32)

    np.testing.assert_array_equal(recipe.channels_first(embeddings.transpose(0, 2, 3, 1), (1, 256, 64, 64)), embeddings)
    np.testing.assert_array_equal(recipe.channels_first(np.ones((1, 1, 1, 4)), (1, 4)), np.ones((1, 4)))


def _mpk(path: Path, processors: list[str], version: str) -> Path:
    manifest = {"model_sdk_version": version, "plugins": [{"processor": p} for p in processors]}
    content = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("model_mpk.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return path


def test_audit_requires_one_mla_segment_and_no_host_code(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    version = recipe.COMPILER_VERSION

    recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", ["EV74", "MLA", "EV74"], version), "image_encoder")
    for name, processors, compiler in [
        ("two", ["MLA", "MLA"], version),
        ("host", ["MLA", "A65"], version),
        ("old", ["MLA"], "0.0.0"),
    ]:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", processors, compiler), "image_encoder")


def test_compile_main_rejects_an_unknown_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--build-profile",
            "w16a16",
            "--compile-invocation",
            "model",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err
