"""V3 source geometry, decoding and calibration regressions (not target qualification)."""

import io
import random
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _heads(reference):
    heads = [np.zeros(shape, np.float32) for shape in reference.OUTPUT_SHAPES]
    for head in heads[6:]:
        head.reshape(-1, 91)[:, 0] = 100
    return heads


def _png(pixels):
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def test_geometry_and_prior_order(reference):
    anchors = reference.anchors_numpy()
    assert anchors.shape == (3234, 4) and anchors.dtype == np.float32
    assert reference.GRIDS == (20, 10, 5, 3, 2, 1)
    assert [shape[3] for shape in reference.OUTPUT_SHAPES] == [24] * 6 + [546] * 6
    np.testing.assert_allclose(anchors[0], [-24, -24, 40, 40])
    np.testing.assert_allclose((anchors[6, :2] + anchors[6, 2:]) / 2, [24, 8])
    np.testing.assert_allclose((anchors[120, :2] + anchors[120, 2:]) / 2, [8, 24])


def test_box_order_scales_and_exp_clamp(reference):
    anchors = np.array([[100, 100, 140, 160]], np.float32)
    reg = np.array([[10, -10, 5 * np.log(2), 5 * np.log(3)]], np.float32)
    np.testing.assert_allclose(reference.decode_numpy(reg, anchors), [[120, 0, 200, 160]], atol=2e-5)
    reg[0, 2:] = 10000
    decoded = reference.decode_numpy(reg, anchors)
    assert np.isfinite(decoded).all()
    np.testing.assert_array_equal(decoded, [[0, 0, 320, 320]])


def test_softmax_includes_background_and_empty_shape(reference):
    heads = _heads(reference)
    rows = reference.decode(heads, 123, 640, 480)
    assert rows.shape == (0, 7)
    result = reference.post_numpy(
        np.zeros((1, 4), np.float32), np.array([[100] + [0] * 90], np.float32), reference.ANCHORS[:1]
    )
    assert result["boxes"].shape == (0, 4) and result["boxes"].dtype == np.float32
    assert result["labels"].shape == (0,) and result["labels"].dtype == np.int64
    assert result["scores"].shape == (0,) and result["scores"].dtype == np.float32


def test_multiclass_same_anchor_is_not_argmax_only(reference):
    logits = np.full((1, 91), -100, np.float32)
    logits[0, 0:3] = [0, 10, 9]
    result = reference.post_numpy(np.zeros((1, 4), np.float32), logits, reference.ANCHORS[:1])
    np.testing.assert_array_equal(result["labels"], [1, 2])
    np.testing.assert_allclose(result["scores"].sum(), 1, atol=4e-5)


def test_per_class_top300_precedes_nms(reference):
    anchors = np.repeat(np.array([[10, 10, 20, 20]], np.float32), 301, axis=0)
    anchors[-1] = [100, 100, 110, 110]
    logits = np.full((301, 91), -100, np.float32)
    logits[:, 0] = 0
    logits[:, 1] = np.linspace(10, 5, 301)
    result = reference.post_numpy(np.zeros((301, 4), np.float32), logits, anchors)
    # All top300 are overlapping. Candidate301 must not be refilled after their NMS.
    assert result["labels"].tolist() == [1]
    np.testing.assert_array_equal(result["boxes"], [[10, 10, 20, 20]])


def test_final_global_cap_and_degenerate_source_policy(reference):
    anchors = np.array([[i % 20 * 10, i // 20 * 10, i % 20 * 10 + 1, i // 20 * 10 + 1] for i in range(400)], np.float32)
    logits = np.full((400, 91), -100, np.float32)
    logits[:, 0] = 0
    logits[:200, 1] = np.linspace(10, 8, 200)
    logits[200:, 2] = np.linspace(7, 5, 200)
    result = reference.post_numpy(np.zeros((400, 4), np.float32), logits, anchors)
    assert len(result["scores"]) == 300
    assert np.all(result["scores"][:-1] >= result["scores"][1:])
    # TorchVision clips but does not remove a zero-area box in SSD postprocessing.
    result = reference.post_numpy(np.array([[10000, 0, 0, 0]], np.float32), logits[:1], anchors[:1])
    assert len(result["scores"]) == 1 and result["boxes"][0, 0] == result["boxes"][0, 2]


def test_coco_xywh_scaling_and_category90(reference):
    heads = _heads(reference)
    logits = heads[6].reshape(-1, 91)
    logits[0, 0], logits[0, 90] = 0, 20
    result = reference.decode(heads, 123, 640, 160)
    assert result.shape == (1, 7)
    np.testing.assert_array_equal(result[0, [0, 1, 2, 3, 4, 6]], [123, 0, 0, 80, 20, 90])
    expected_score = np.float32(1) / (np.float32(1) + np.float32(90) * np.exp(np.float32(-20)))
    np.testing.assert_allclose(result[0, 5], expected_score, atol=2e-7, rtol=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("head", [0, 6])
def test_nonfinite_is_rejected(reference, bad, head):
    heads = _heads(reference)
    heads[head].flat[0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        reference.decode(heads, 1, 100, 100)


def test_wrong_heads_and_dimensions_rejected(reference):
    heads = _heads(reference)
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        reference.decode(heads[:-1], 1, 10, 10)
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        reference.decode([heads[6], *heads[1:]], 1, 10, 10)
    with pytest.raises(ValueError, match="positive"):
        reference.decode(heads, 1, 0, 10)


def test_seeded_calibration_uses_only_cal2017_and_full_archive(tmp_path, calibration):
    archive = tmp_path / "cal.tar.gz"
    paths = [f"coco/cal2017/{i:012}.jpg" for i in range(8)]
    with tarfile.open(archive, "w:gz") as stream:
        for name in [*reversed(paths), "coco/val2017/999.jpg"]:
            content = b"bytes are not decoded by the selection test"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    selected = calibration.calibration_names(archive, 4)
    assert selected == random.Random(20261028).sample(sorted(paths), 4)
    assert selected == calibration.calibration_names(archive, 4)
    with pytest.raises(ValueError, match="cannot select"):
        calibration.calibration_names(archive, 9)


@pytest.mark.parametrize("name", ["/absolute.jpg", "../escape.jpg"])
def test_unsafe_archive_rejected(tmp_path, calibration, name):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo(name)
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        calibration.calibration_names(archive, 1)


@pytest.mark.parametrize("dtype", [np.float64, np.int8, np.uint8])
def test_wrong_head_dtype_rejected(reference, dtype):
    heads = _heads(reference)
    heads[0] = heads[0].astype(dtype)
    with pytest.raises(ValueError, match="float32"):
        reference.decode(heads, 1, 100, 100)


@pytest.fixture
def reference(load_script):
    return load_script("accuracy")


@pytest.fixture
def calibration(load_script):
    return load_script("calibration")


def test_accuracy_meets_the_source_relative_gate(accuracy, model_dir, build_profile):
    from model_registry.model_tests import measured_shortfalls

    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def _scorer(accuracy: ModuleType, **overrides: float) -> SimpleNamespace:
    stats = dict.fromkeys(accuracy.METRICS.values(), 0.5)
    stats.update(overrides)
    return SimpleNamespace(params=SimpleNamespace(maxDets=[1, 10, 100]), stats_as_dict=stats)


def test_local_subset_omits_undefined_size_metrics_without_inventing_zero(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    scorer = _scorer(accuracy, AP_small=-1, AR_small=-1)

    measured = accuracy.coco_stats(scorer, allow_missing_size_buckets=True)

    assert measured["coco_map"] == 0.5
    assert set(accuracy.METRICS) - measured.keys() == {"coco_ap_small", "coco_ar_small"}
    assert all(value == 0.5 for value in measured.values())
    with pytest.raises(SystemExit, match="no ground-truth objects"):
        accuracy.coco_stats(scorer)


@pytest.mark.parametrize("metric", ["AP_all", "AP_50", "AR_third"])
def test_subset_cannot_omit_global_metrics(metric: str, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    with pytest.raises(SystemExit, match="no ground-truth objects"):
        accuracy.coco_stats(_scorer(accuracy, **{metric: -1}), allow_missing_size_buckets=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1.1, -2, -0.5])
@pytest.mark.parametrize("metric", ["AP_all", "AP_small"])
def test_subset_rejects_invalid_metrics_instead_of_hiding_them(
    value: float, metric: str, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    with pytest.raises(SystemExit):
        accuracy.coco_stats(_scorer(accuracy, **{metric: value}), allow_missing_size_buckets=True)


def test_accuracy_rejects_missing_required_coco_map(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    scorer = _scorer(accuracy)
    del scorer.stats_as_dict["AP_all"]
    with pytest.raises(KeyError, match="AP_all"):
        accuracy.coco_stats(scorer, allow_missing_size_buckets=True)


def test_undeclared_performance_route_fails_without_a_report(tmp_path):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/performance.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mpk",
            "unused.tar.gz",
            "--dataset",
            "coco-2017-val=unused",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not qualified" in result.stderr
    assert not (tmp_path / "model-report.json").exists()
