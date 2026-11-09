from __future__ import annotations

import io
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profiles = [profile["id"] for profile in metadata["build_profiles"]]
    assert profiles[0] == "w16a16" and set(profiles) <= recipe.PRECISIONS

    for profile_id in profiles:
        variant = recipe._variant(metadata, profile_id)
        assert variant["parameters"]["precision"] == profile_id
        assert variant["parameters"]["input_shape"] == [1, 3, 256, 256]
        if profile_id != "w16a16":
            assert variant["parameters"]["calibration_samples"] == 8


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="exactly one build profile named 'w4a16'"):
        recipe._variant(recipe._metadata(), "w4a16")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision", "w4a16", "only the declared Modalix BF16, INT8 and INT8-weight variants"),
        ("input_shape", [1, 3, 224, 224], "unsupported FAN input shape"),
    ],
)
def test_compile_rejects_an_undeclared_precision_or_shape_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    field: str,
    value: object,
    message: str,
) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    profile = metadata["build_profiles"][0]
    profile["components"][0]["parameters"][field] = value
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)

    with pytest.raises(ValueError, match=message):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile["id"], None)
    assert not (tmp_path / "model").exists()


def test_calibrated_builds_require_the_calibration_dataset(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    calibrated = [p["id"] for p in recipe._metadata()["build_profiles"] if p["id"] != "w16a16"]
    for profile_id in calibrated:
        with pytest.raises(ValueError, match="requires --dataset"):
            recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", profile_id, None)
    assert not (tmp_path / "model").exists()


def test_compile_main_rejects_an_invocation_named_otherwise(
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


# landmarks.py, the crop, decode and AFW reader that compile.py and accuracy.py share.


@pytest.fixture
def landmarks(load_script: Callable[[str], ModuleType]) -> ModuleType:
    pytest.importorskip("cv2")
    return load_script("landmarks")


def test_crop_box_enlarges_the_face_box_by_its_crop_ratio(landmarks: ModuleType) -> None:
    box = landmarks.crop_box(np.array([100, 100, 210, 210]))
    # A 110-pixel face box becomes a 200-pixel square (110 / 0.55) around the same centre, plus ibug's +1.
    assert list(box) == [55, 55, 256, 256]


def test_crop_zero_pads_beyond_the_image_and_scales_to_unit_range(landmarks: ModuleType) -> None:
    image = np.full((100, 100, 3), 255, np.uint8)
    patch, box = landmarks.crop(image, landmarks.whole_image_box(image))
    assert patch.shape == (256, 256, 3) and patch.dtype == np.float32
    assert patch.max() == 1.0 and patch[0, 0, 0] == 0.0  # the corner lies in the zero padding
    assert box[2] - box[0] == box[3] - box[1] > 100


def test_decode_finds_each_heatmap_peak_with_subpixel_centroids(landmarks: ModuleType) -> None:
    rows, columns = np.mgrid[0:64, 0:64].astype(np.float32)
    heatmaps = np.zeros((68, 64, 64), np.float32)
    for index in range(68):
        y, x = 10.0 + index * 0.5, 50.0 - index * 0.5
        heatmaps[index] = np.exp(-((rows + 0.5 - y) ** 2 + (columns + 0.5 - x) ** 2) / 2.0)
    # A 64x64 heatmap over a 128-pixel crop at (100, 200): each heatmap pixel is 2 image pixels.
    points, scores = landmarks.decode(heatmaps, np.array([100, 200, 228, 328]))

    expected = np.stack([100 + 2 * (50.0 - np.arange(68) * 0.5), 200 + 2 * (10.0 + np.arange(68) * 0.5)], axis=1)
    np.testing.assert_allclose(points, expected, atol=0.05)
    # Every other peak falls between pixel centres, so its highest sample is below 1.
    assert scores.shape == (68,) and (scores > 0.7).all()


def test_nme_normalizes_by_the_outer_eye_corners(landmarks: ModuleType) -> None:
    truth = np.zeros((68, 2))
    truth[36], truth[45] = (0.0, 0.0), (100.0, 0.0)
    assert landmarks.nme(truth + [3.0, 4.0], truth) == pytest.approx(0.05)


def _archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _jpeg(size: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".jpg", np.full((size, size, 3), 128, np.uint8))
    assert ok
    return encoded.tobytes()


def _pts(offset: float) -> bytes:
    points = "\n".join(f"{offset + i} {offset + 2 * i}" for i in range(68))
    return f"version: 1\nn_points: 68\n{{\n{points}\n}}\n".encode()


def test_afw_reader_pairs_faces_with_points_and_skips_macos_metadata(landmarks: ModuleType, tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "afw.tgz",
        {
            "test_data/1_1_0.jpg": _jpeg(120),
            "test_data/1_1_0.pts": _pts(1.0),
            "test_data/2_1_0.jpg": _jpeg(90),  # no annotation: not scored
            "test_data/._1_1_0.jpg": b"\x00\x05\x16\x07",
            "test_data/.DS_Store": b"junk",
            "cal_data/3_1_0.jpg": _jpeg(100),
            "cal_data/4_1_0.jpg": _jpeg(140),
            "cal_data/._3_1_0.jpg": b"junk",
        },
    )

    ((name, image, truth),) = landmarks.test_faces(archive)
    assert name == "1_1_0.jpg" and image.shape == (120, 120, 3)
    assert truth.shape == (68, 2) and truth[67].tolist() == [68.0, 135.0]
    samples = landmarks.calibration_samples(archive, 2)
    assert samples.shape == (2, 256, 256, 3) and samples.dtype == np.float32
    with pytest.raises(ValueError, match="calibration requires 3 faces"):
        landmarks.calibration_samples(archive, 3)


def test_calibration_method_follows_the_declared_parameters(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    assert recipe.calibration_method({}) is not None
    assert recipe.calibration_method({"calibration_method": "percentile", "calibration_percentile": 99.99}) is not None
    with pytest.raises(ValueError, match="unsupported calibration method: kl"):
        recipe.calibration_method({"calibration_method": "kl"})


def dense_decode_oracle(heatmaps, box):
    """Original full-map FAN formula, deliberately independent of the optimized neighborhood gather."""
    heatmaps = heatmaps.astype(np.float32)
    rows, columns = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(64, dtype=np.float32), indexing="ij")
    flat = heatmaps.reshape(68, -1).argmax(axis=1)
    distance = np.hypot(rows[None] - (flat // 64)[:, None, None], columns[None] - (flat % 64)[:, None, None])
    masked = np.where(distance <= np.float32(6.4), heatmaps, 0).clip(min=0)
    total = np.maximum(masked.sum(axis=(1, 2)), np.finfo(np.float32).eps)
    centres = np.arange(0.5, 64, dtype=np.float32)
    xs = (masked.sum(axis=1) * centres).sum(axis=1) / total
    ys = (masked.sum(axis=2) * centres).sum(axis=1) / total
    left, top, right, bottom = box
    return np.stack([xs * (right - left) / 64 + left, ys * (bottom - top) / 64 + top], axis=1), heatmaps.max((1, 2))


@pytest.mark.parametrize("case", ["random", "ties", "corners", "zero", "negative", "tiny", "strided"])
def test_local_centroid_matches_independent_dense_oracle(landmarks, case):
    heatmaps = np.random.default_rng(42).normal(size=(68, 64, 64)).astype(np.float32)
    if case == "ties":
        heatmaps[:, 10, 10] = heatmaps[:, 10, 11] = 10
    elif case == "corners":
        for i in range(68):
            heatmaps[i, 63 * (i % 2), 63 * ((i // 2) % 2)] = 10
    elif case == "zero":
        heatmaps[:] = 0
    elif case == "negative":
        heatmaps[:] = -np.abs(heatmaps)
    elif case == "tiny":
        heatmaps *= 1e-10
    elif case == "strided":
        heatmaps = heatmaps.transpose(0, 2, 1)
    heatmaps.flags.writeable = False
    before = heatmaps.copy()
    box = np.array([-77, 31, 1348, 946])
    actual, scores = landmarks.decode(heatmaps, box)
    expected, peak_scores = dense_decode_oracle(heatmaps, box)
    # Different float32 reduction trees may differ by a few ulps; image-coordinate tolerance is sub-millipixel.
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-4)
    np.testing.assert_array_equal(scores, peak_scores)
    np.testing.assert_array_equal(heatmaps, before)


@pytest.mark.parametrize("bad", [np.zeros((68, 4096)), np.full((68, 64, 64), np.nan), np.full((68, 64, 64), np.inf)])
def test_decoder_rejects_malformed_or_nonfinite_heatmaps(landmarks, bad):
    with pytest.raises(ValueError, match="heatmaps"):
        landmarks.decode(bad, np.array([0, 0, 256, 256]))


@pytest.mark.parametrize("box", [[0, 0, 0, 4], [10, 10, 9, 11], [0, 0, float("nan"), 3], [0, 1, 2]])
def test_crop_rejects_invalid_face_boxes_before_allocation(landmarks, box):
    with pytest.raises(ValueError, match="box"):
        landmarks.crop(np.zeros((12, 16, 3), np.uint8), np.array(box))


def test_nme_rejects_zero_eye_distance(landmarks):
    with pytest.raises(ValueError, match="inter-ocular"):
        landmarks.nme(np.zeros((68, 2)), np.zeros((68, 2)))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0, -1, 100.1])
def test_percentile_requires_a_finite_supported_value(load_script, value):
    recipe = load_script("compile")
    parameters = {"calibration_method": "percentile"}
    if value is not None:
        parameters["calibration_percentile"] = value
    with pytest.raises(ValueError, match="calibration_percentile"):
        recipe.calibration_method(parameters)


def test_percentile_passes_exact_declared_method_to_compiler(load_script, monkeypatch):
    recipe = load_script("compile")
    calls = []
    monkeypatch.setattr(recipe, "HistogramPercentileMethod", lambda **kwargs: calls.append(kwargs))
    recipe.calibration_method({"calibration_method": "percentile", "calibration_percentile": 99.99})
    assert calls == [{"percentile_value": 99.99, "num_bins": 1024}]


def assert_fan_public_contract(manifest):
    """The original FP32 interface survives the compiler's channels-last adapters."""
    assert [(item["name"], item["size"]) for item in manifest["input_nodes"]] == [("image", 1 * 256 * 256 * 3 * 4)]
    stages = sorted(manifest["plugins"], key=lambda item: item["sequence"])
    assert len({item["sequence"] for item in stages}) == len(stages)
    assert [item["processor"] for item in stages].count("MLA") == 1
    assert {item["processor"] for item in stages} <= {"MLA", "EV74"}
    assert [(n["name"], n["size"]) for n in stages[0]["input_nodes"]] == [("image", 786432)]
    assert stages[0]["config_params"]["params"]["input_shapes"] == [[1, 256, 256, 3]]
    assert stages[0]["config_params"]["params"]["in_dtype"] == "float32"
    terminal = stages[-1]
    assert terminal["config_params"]["params"]["output_shapes"] == [[1, 64, 64, 68]]
    assert terminal["config_params"]["params"]["out_dtype"] == "float32"
    assert [node["size"] for node in terminal["output_nodes"]] == [1 * 64 * 64 * 68 * 4]
    for previous, following in zip(stages, stages[1:]):
        assert [(n["name"], n["size"]) for n in previous["output_nodes"]] == [
            (n["name"], n["size"]) for n in following["input_nodes"]
        ]


def test_compiled_fan_preserves_the_public_interface(mpk_json):
    assert_fan_public_contract(mpk_json())


def test_compiled_mla_output_keeps_physical_padding_separate_from_68_landmarks(mpk_json, build_profile):
    assert build_profile in {"w16a16", "w8a16"}, "qualify new precision's physical carriers before enabling it"
    manifest = mpk_json()
    (mla,) = [item for item in manifest["plugins"] if item["processor"] == "MLA"]
    assert [node["size"] for node in mla["input_nodes"]] == [1 * 256 * 256 * 3 * 2]
    # Both shipped profiles carry BF16 activations; MLA rounds 68 channels to 72, not to the public extent.
    assert [node["size"] for node in mla["output_nodes"]] == [1 * 64 * 64 * 72 * 2]
    (sliced,) = [item for item in manifest["plugins"] if item["config_params"].get("kernel") == "slice_transform"]
    assert sliced["config_params"]["params"]["begin"] == [0, 0, 0, 0]
    assert sliced["config_params"]["params"]["end"] == [1, 64, 64, 68]
    assert sliced["config_params"]["params"]["input_shape"] == [1, 64, 64, 72]
    assert sliced["config_params"]["params"]["output_shape"] == [1, 64, 64, 68]
    assert [node["size"] for node in sliced["output_nodes"]] == [1 * 64 * 64 * 68 * 2]


@pytest.mark.parametrize(
    "bad", ["source_size", "unbound_source", "output_shape", "terminal_dtype", "unbound_input", "extra_mla"]
)
def test_public_contract_rejects_wrong_model_interface(bad):
    # Independent minimal carrier records, not a copied manifest from the real model package.
    manifest = {
        "input_nodes": [{"name": "image", "size": 786432}],
        "plugins": [
            {
                "sequence": 0,
                "processor": "EV74",
                "config_params": {"params": {"input_shapes": [[1, 256, 256, 3]], "in_dtype": "float32"}},
                "input_nodes": [{"name": "image", "size": 786432}],
                "output_nodes": [{"name": "prepared", "size": 393216}],
            },
            {
                "sequence": 1,
                "processor": "MLA",
                "input_nodes": [{"name": "prepared", "size": 393216}],
                "output_nodes": [{"name": "output", "size": 589824}],
            },
            {
                "sequence": 2,
                "processor": "EV74",
                "input_nodes": [{"name": "output", "size": 589824}],
                "output_nodes": [{"name": "heatmaps", "size": 1114112}],
                "config_params": {"params": {"output_shapes": [[1, 64, 64, 68]], "out_dtype": "float32"}},
            },
        ],
    }
    assert_fan_public_contract(manifest)
    if bad == "source_size":
        manifest["input_nodes"][0]["size"] //= 2
    elif bad == "unbound_source":
        manifest["plugins"][0]["input_nodes"][0]["name"] = "unrelated"
    elif bad == "output_shape":
        manifest["plugins"][-1]["config_params"]["params"]["output_shapes"] = [[1, 64, 64, 72]]
    elif bad == "terminal_dtype":
        manifest["plugins"][-1]["config_params"]["params"]["out_dtype"] = "bfloat16"
    elif bad == "unbound_input":
        manifest["plugins"][1]["input_nodes"][0]["name"] = "unrelated"
    else:
        manifest["plugins"][0]["processor"] = "MLA"
    with pytest.raises(AssertionError):
        assert_fan_public_contract(manifest)
