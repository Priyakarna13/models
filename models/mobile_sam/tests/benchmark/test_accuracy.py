from __future__ import annotations

import json
import weakref
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_a_box_prompt_has_only_two_active_corners(accuracy_script: ModuleType) -> None:
    prompt = accuracy_script.box_prompt([100.0, 50.0, 200.0, 100.0], (480, 640))

    # 640 wide scales to 1024: corners (100, 50) and (300, 150), shifted to pixel centres and normalized.
    coords = prompt["point_coords"][0, :, 0, :2].T
    np.testing.assert_allclose(coords, (np.array([[100, 50], [300, 150]]) * 1.6 + 0.5) / 1024 * 2 - 1, rtol=1e-6)
    types = prompt["point_types"][0, :, 0, :].argmax(axis=0)
    assert types.tolist() == [3, 4, 0, 0, 0, 0]
    assert (prompt["point_types"][..., 2:] == 0).all()


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_mask_layout_reads_both_layouts(accuracy_script: ModuleType, layout: str) -> None:
    masks = np.random.default_rng(0).normal(size=(4, 256, 256)).astype(np.float32)
    board = masks[None] if layout == "nchw" else masks.transpose(1, 2, 0)[None]

    read, scores = accuracy_script.mask_layout(board, np.arange(4, dtype=np.float32)[None])

    np.testing.assert_array_equal(read, masks)
    assert scores.tolist() == [0, 1, 2, 3]


def test_image_mask_drops_the_padding_and_fits_the_image(accuracy_script: ModuleType) -> None:
    # A 1024x512 resized image fills the top half of the padded input; its mask covers that half.
    logits = np.full((256, 256), -5.0, np.float32)
    logits[:128] = 5.0

    mask = accuracy_script.image_mask(logits, (300, 600))

    assert mask.shape == (300, 600)
    assert mask.all()


def test_a_result_scores_the_first_mask_by_its_predicted_iou(
    accuracy_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    masks = np.full((4, 256, 256), -5.0, np.float32)
    masks[0, 64:128, 64:128] = 5.0
    encoded = []
    monkeypatch.setattr(accuracy_script.mask_util, "encode", lambda array: encoded.append(array) or [{"counts": b"x"}])

    item = accuracy_script.result(
        {"image_id": 7, "category_id": 3}, masks, np.array([0.9, 0.1, 0.2, 0.3], np.float32), (1024, 1024)
    )

    (array,) = encoded
    # The 64x64 square upsamples 4x to rows and columns 256-511; bilinear leaves its four corner pixels at exactly 0.
    rows, columns = np.nonzero(array[:, :, 0])
    assert array.shape == (1024, 1024, 1) and array.sum() == 256 * 256 - 4
    assert (rows.min(), rows.max(), columns.min(), columns.max()) == (256, 511, 256, 511)
    assert item["segmentation"] == {"counts": "x"}
    assert (item["image_id"], item["category_id"], item["score"]) == (7, 3, pytest.approx(0.9))


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def _bilinear_reference(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Independent float64 pixel-centre interpolation: align_corners=False, no antialiasing."""
    height, width = size
    rows = np.maximum((np.arange(height) + 0.5) * image.shape[0] / height - 0.5, 0)
    columns = np.maximum((np.arange(width) + 0.5) * image.shape[1] / width - 0.5, 0)
    y0, x0 = rows.astype(int), columns.astype(int)
    y1, x1 = np.minimum(y0 + 1, image.shape[0] - 1), np.minimum(x0 + 1, image.shape[1] - 1)
    dy, dx = rows - y0, columns - x0
    top = image[y0[:, None], x0] * (1 - dx) + image[y0[:, None], x1] * dx
    bottom = image[y1[:, None], x0] * (1 - dx) + image[y1[:, None], x1] * dx
    return top * (1 - dy[:, None]) + bottom * dy[:, None]


@pytest.mark.parametrize("size", [(300, 600), (640, 480), (331, 417), (720, 1280), (1024, 1024), (17, 13)])
def test_mask_geometry_matches_independent_sam_pixel_centres(size, load_script):
    pipeline = load_script("pipeline")
    logits = np.random.default_rng(10).normal(size=(256, 256)).astype(np.float32)
    resized_height, resized_width = pipeline.resized_size(*size)
    upsampled = _bilinear_reference(logits.astype(np.float64), (1024, 1024))
    expected = _bilinear_reference(upsampled[:resized_height, :resized_width], size)
    actual = pipeline.resize_mask_logits(logits, size)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
    assert actual.shape == size
    # Floating-point rounding around zero is not a licence to change interpolation or threshold policy.
    away_from_zero = np.abs(expected) > 2e-5
    np.testing.assert_array_equal((actual > 0)[away_from_zero], (expected > 0)[away_from_zero])


def test_application_accuracy_and_performance_share_one_mask_policy(load_script):
    pipeline, accuracy = load_script("pipeline"), load_script("accuracy")
    masks = np.random.default_rng(10).normal(size=(4, 256, 256)).astype(np.float32)
    size = (300, 600)
    np.testing.assert_array_equal(pipeline.postprocess_masks(masks, size)[0] > 0, accuracy.image_mask(masks[0], size))
    # The single-mask token is not a candidate in multimask click mode, even with its highest score.
    scores = np.array([99, 0.2, 0.8, 0.4], np.float32)
    np.testing.assert_array_equal(
        pipeline.selected_mask(masks, scores, size, multimask=True), pipeline.image_mask(masks[2], size)
    )
    np.testing.assert_array_equal(
        pipeline.selected_mask(masks, scores, size, multimask=False), pipeline.image_mask(masks[0], size)
    )


@pytest.mark.parametrize("field", ["masks", "scores"])
def test_decoder_nonfinite_outputs_fail_before_mask_publication(field, load_script):
    pipeline = load_script("pipeline")
    masks, scores = np.zeros((4, 256, 256), np.float32), np.ones(4, np.float32)
    (masks if field == "masks" else scores).flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pipeline.selected_mask(masks, scores, (300, 600), multimask=True)


def test_accuracy_closes_all_runners_without_masking_inference_error(load_script):
    script = load_script("accuracy")
    calls = []

    class Runner:
        def __init__(self, name, fails=False):
            self.name, self.fails = name, fails

        def close(self):
            calls.append(self.name)
            if self.fails:
                raise RuntimeError("cleanup")

    runners = script.Runners.__new__(script.Runners)
    runners.runners = {"first": Runner("first"), "second": Runner("second", True)}
    with pytest.raises(ValueError, match="inference"):
        with runners:
            raise ValueError("inference")
    assert calls == ["second", "first"]


def test_fp32_runner_feeds_the_real_component_chain(load_script):
    script = load_script("fp32_baseline_accuracy")
    feeds = []
    runners = script.FloatRunners.__new__(script.FloatRunners)
    feature, embedding = object(), object()

    def run(component, values):
        feeds.append((component, values))
        return {"image_encoder_0": [feature], "image_encoder_1": [embedding], "prompt_decoder": ["masks", "scores"]}[
            component
        ]

    runners.run = run
    image = np.zeros((1024, 1024, 3), np.float32)
    actual = runners.encode(image)
    assert actual is embedding
    assert feeds[0][1]["image"].shape == (1, 3, 1024, 1024)
    assert feeds[1][1]["features"] is feature
    assert runners.decode(actual, {"point_coords": "coordinates", "point_types": "types"}) == ("masks", "scores")
    assert feeds[-1][1] == {"image_embeddings": embedding, "point_coords": "coordinates", "point_types": "types"}


@pytest.fixture
def fp32_run(load_script, monkeypatch, tmp_path):
    """Exercise the real baseline CLI/persistence with a tiny fake inference result, not an accuracy claim."""
    script = load_script("fp32_baseline_accuracy")
    source = tmp_path / "source.onnx"
    source.write_bytes(b"test source")
    dataset = tmp_path / "dataset"
    annotations = dataset / script.accuracy.ANNOTATIONS
    annotations.parent.mkdir(parents=True)
    annotations.write_text("{}")
    images = dataset / script.accuracy.IMAGES
    images.mkdir()
    for image_id in (7, 9):
        (images / f"{image_id}.jpg").write_bytes(f"image {image_id}".encode())
    rows = [
        {"image_id": image_id, "category_id": 3, "score": 0.75, "segmentation": {"size": [2, 2], "counts": "4"}}
        for image_id in (7, 9)
    ]

    class GroundTruth:
        def getImgIds(self):
            return [9, 7]

        def getAnnIds(self, imgIds, iscrowd):
            assert iscrowd is False
            return imgIds

        def loadAnns(self, ids):
            return [item for item in rows if item["image_id"] in ids]

        def loadImgs(self, ids):
            return [{"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in ids]

    references = []

    class Runners:
        def __init__(self, graphs):
            references.append(weakref.ref(self))

        def encode(self, value):
            raise AssertionError("fake evaluator does not infer")

        def decode(self, value, prompt):
            raise AssertionError("fake evaluator does not infer")

    monkeypatch.setattr(script, "FloatRunners", Runners)
    monkeypatch.setattr(script, "split_components", lambda *args: {})
    monkeypatch.setattr(script.accuracy, "COCO", lambda *args: GroundTruth())
    monkeypatch.setattr(script.accuracy, "VAL2017_IMAGES", 2)
    monkeypatch.setattr(script.accuracy, "evaluate", lambda *args: ([dict(item) for item in rows], len(rows)))
    monkeypatch.setattr(script.accuracy, "score", lambda *args: {"test_metric": 0.5})
    output = tmp_path / "report.json"
    args = ["--onnx", str(source), "--dataset", f"coco-2017-val={dataset}", "--output", str(output)]
    return script, args, output, output.with_suffix(".predictions.json"), references


def test_fp32_predictions_survive_scoring_failure_and_reuse_the_same_scorer(fp32_run, monkeypatch):
    script, args, output, predictions, runners = fp32_run

    def fail_score(ground_truth, results, image_ids):
        assert predictions.is_file()
        assert runners[0]() is None  # ORT sessions are released before scoring.
        assert image_ids == [7, 9]
        raise RuntimeError("score failed")

    monkeypatch.setattr(script.accuracy, "score", fail_score)
    with pytest.raises(RuntimeError, match="score failed"):
        script.main(args)
    assert not output.exists()
    saved_bytes = predictions.read_bytes()
    saved = json.loads(saved_bytes)
    assert saved["provenance"]["image_ids"] == [7, 9]
    assert saved["prompts"] == len(saved["predictions"]) == 2
    assert len(saved["provenance"]["onnx_sha256"]) == 64
    assert len(saved["provenance"]["annotations_sha256"]) == 64

    def score(ground_truth, results, image_ids):
        assert results == saved["predictions"] and image_ids == [7, 9]
        return {"test_metric": 0.5}

    def cannot_infer(*args):
        raise AssertionError("score-only must not create ORT sessions or run inference")

    monkeypatch.setattr(script, "FloatRunners", cannot_infer)
    monkeypatch.setattr(script, "split_components", cannot_infer)
    monkeypatch.setattr(script.accuracy, "evaluate", cannot_infer)
    monkeypatch.setattr(script.accuracy, "score", score)
    assert script.main([*args, "--score-only"]) == 0
    assert json.loads(output.read_text())["evaluated_samples"] == 2
    assert predictions.read_bytes() == saved_bytes


@pytest.mark.parametrize(
    "field",
    ["onnx_sha256", "annotations_sha256", "selected_images_sha256", "image_ids", "scripts_sha256", "conditions"],
)
def test_fp32_score_only_rejects_mismatched_provenance(fp32_run, field):
    script, args, _, predictions, _ = fp32_run
    script.main(args)
    saved = json.loads(predictions.read_text())
    saved["provenance"][field] = "changed"
    predictions.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="provenance"):
        script.main([*args, "--score-only"])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "prompt_count", "nonfinite"])
def test_fp32_score_only_rejects_incomplete_or_illegal_predictions(fp32_run, damage):
    script, args, _, predictions, _ = fp32_run
    script.main(args)
    saved = json.loads(predictions.read_text())
    if damage == "missing":
        saved["predictions"].pop()
    elif damage == "duplicate":
        saved["predictions"][0] = saved["predictions"][1]
    elif damage == "prompt_count":
        saved["prompts"] += 1
    else:
        saved["predictions"][0]["score"] = float("nan")
    predictions.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="counts|finite"):
        script.main([*args, "--score-only"])


def test_fp32_inference_does_not_overwrite_existing_predictions(fp32_run):
    script, args, _, predictions, _ = fp32_run
    script.main(args)
    saved = predictions.read_bytes()
    with pytest.raises(SystemExit):
        script.main(args)
    assert predictions.read_bytes() == saved


def test_fp32_score_only_detects_changed_image_bytes(fp32_run):
    script, args, output, _, _ = fp32_run
    script.main(args)
    (output.parent / "dataset" / script.accuracy.IMAGES / "7.jpg").write_bytes(b"different image")
    with pytest.raises(ValueError, match="provenance"):
        script.main([*args, "--score-only"])


def test_fp32_prediction_publication_does_not_clobber_a_late_writer(fp32_run, monkeypatch):
    script, args, output, predictions, _ = fp32_run
    evaluate = script.accuracy.evaluate

    def infer(*args):
        predictions.write_text("another completed prediction artifact")
        return evaluate(*args)

    monkeypatch.setattr(script.accuracy, "evaluate", infer)
    with pytest.raises(FileExistsError):
        script.main(args)
    assert predictions.read_text() == "another completed prediction artifact"
    assert not output.exists()


def test_fp32_failed_serialization_does_not_publish_a_partial_file(load_script, tmp_path):
    script = load_script("fp32_baseline_accuracy")
    output = tmp_path / "predictions.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        script.write_json(output, {"score": float("nan")})
    assert list(tmp_path.iterdir()) == []
