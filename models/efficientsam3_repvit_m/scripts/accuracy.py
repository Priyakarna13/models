"""Measure EfficientSAM3 RepViT-M COCO box and mask AP with its text encoder and image model MPKs on a Modalix board.

Every image runs once per COCO category, prompted with the category name. The text encoder runs once per category.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pyneat
from calibration import COCO_TOKENS, CONTEXT, FEATURES, IMAGE_SIZE, one_hot, packed, preprocess_image, tokens
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# The lowest 100 val2017 image IDs, each run with all 80 prompts.
EVALUATED_IMAGES = 100
QUERIES = 200
MASK_SIZE = 192
# COCO scores at most 100 detections per image: keep the 100 best of all prompts and queries.
TOP_K = 100
# Metric name in model.yaml -> (COCOeval_faster iouType, stats_as_dict label). With maxDets [1, 10, 100],
# AR_all, AR_second and AR_third are the box recall at 1, 10 and 100 detections per image.
METRICS = {
    "coco_map": ("bbox", "AP_all"),
    "coco_mask_map": ("segm", "AP_all"),
    "coco_ap50": ("bbox", "AP_50"),
    "coco_ap75": ("bbox", "AP_75"),
    "coco_ap_small": ("bbox", "AP_small"),
    "coco_ap_medium": ("bbox", "AP_medium"),
    "coco_ap_large": ("bbox", "AP_large"),
    "coco_ar1": ("bbox", "AR_all"),
    "coco_ar10": ("bbox", "AR_second"),
    "coco_ar100": ("bbox", "AR_third"),
    "coco_ar_small": ("bbox", "AR_small"),
    "coco_ar_medium": ("bbox", "AR_medium"),
    "coco_ar_large": ("bbox", "AR_large"),
}
MAX_DETECTIONS = [1, 10, 100]
TIMEOUT_MS = 120000

TextFeatures = Callable[[str], np.ndarray]
# Detections and a deferred read of the mask logits, which board runs then fetch only when a query is kept.
ImageRun = tuple[np.ndarray, Callable[[], np.ndarray]]
# Given every prompt's packed text, return a function that runs one image with each prompt in turn.
ImageModel = Callable[[list[np.ndarray]], Callable[[np.ndarray], Iterator[ImageRun]]]


def detection_rows(detections: np.ndarray) -> np.ndarray:
    """Return the image model's detections as [200, 6] rows."""
    if detections.size != QUERIES * 6:
        raise ValueError(f"unexpected detections shape: {detections.shape}")
    detections = np.asarray(detections, dtype=np.float32).reshape(QUERIES, 6)
    if not np.isfinite(detections).all():
        raise ValueError("image model detections contain non-finite values")
    return detections


def query_masks(masks: np.ndarray) -> np.ndarray:
    """Return a [200, 192, 192] view of NCHW or HWC mask logits."""
    if masks.size != QUERIES * MASK_SIZE * MASK_SIZE:
        raise ValueError(f"unexpected mask logits shape: {masks.shape}")
    if masks.shape[-1] == QUERIES:
        return np.moveaxis(masks.reshape(MASK_SIZE, MASK_SIZE, QUERIES), -1, 0)
    return masks.reshape(QUERIES, MASK_SIZE, MASK_SIZE)


def best_queries(
    kept: tuple[np.ndarray, np.ndarray, np.ndarray],
    detections: np.ndarray,
    masks: Callable[[], np.ndarray],
    category: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge one prompt's queries into the kept (detections, categories, masks) and keep the TOP_K best scores.

    Earlier prompts win ties. The masks are read only when a query can enter the kept set, and only the
    entering queries' masks are copied.
    """
    scores = detections[:, 4]
    chosen = np.argsort(-scores, kind="stable")[:TOP_K]
    if len(kept[0]) == TOP_K:
        chosen = chosen[scores[chosen] > kept[0][-1, 4]]
    if not len(chosen):
        return kept
    chosen_masks = masks()[chosen]
    if not np.isfinite(chosen_masks).all():
        raise ValueError("image model mask logits contain non-finite values")
    merged = (
        np.concatenate([kept[0], detections[chosen]]),
        np.concatenate([kept[1], np.full(len(chosen), category)]),
        np.concatenate([kept[2], chosen_masks]),
    )
    order = np.argsort(-merged[0][:, 4], kind="stable")[:TOP_K]
    return tuple(values[order] for values in merged)


def coco_results(
    image_id: int,
    width: int,
    height: int,
    kept: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Return COCO box and mask results for one image from its kept queries in 1008x1008 input pixels."""
    detections, categories, masks = kept
    scale = np.array([width, height, width, height], dtype=np.float64) / IMAGE_SIZE
    # Boxes are not clipped, as in SAM 3's processor.
    boxes = detections[:, :4].astype(np.float64) * scale
    # Bilinear with half-pixel centres, as torch interpolate(align_corners=False); a positive logit is foreground.
    # Resizing the transposed masks builds the column-major [H, W, N] array that RLE encoding reads, without a
    # strided copy.
    columns = [
        cv2.resize(np.ascontiguousarray(mask.T), (height, width), interpolation=cv2.INTER_LINEAR) for mask in masks
    ]
    encoded = mask_util.encode((np.stack(columns) > 0).astype(np.uint8).transpose(2, 1, 0)) if columns else []
    boxes_out, masks_out = [], []
    for (x1, y1, x2, y2), score, category, rle in zip(boxes, detections[:, 4], categories, encoded, strict=True):
        common = {"image_id": image_id, "category_id": int(category), "score": float(score)}
        boxes_out.append({**common, "bbox": [x1, y1, x2 - x1, y2 - y1]})
        masks_out.append({**common, "segmentation": rle})
    return boxes_out, masks_out


def coco_stats(scorers: dict[str, COCOeval_faster]) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    for scorer in scorers.values():
        if list(scorer.params.maxDets) != MAX_DETECTIONS:
            raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = {kind: scorer.stats_as_dict for kind, scorer in scorers.items()}
    stats = {name: float(labelled[kind][label]) for name, (kind, label) in METRICS.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def evaluate(
    dataset: Path,
    image_count: int,
    text_features: TextFeatures,
    image_model: ImageModel,
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    """Run every (image, category) pair through the models and score boxes and masks with COCOeval."""
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = available[:image_count]
    category_ids = sorted(ground_truth.getCatIds())
    names = [category["name"] for category in ground_truth.loadCats(category_ids)]
    if names != list(COCO_TOKENS):
        raise SystemExit("COCO categories do not match the prompt table")

    runs = image_model([packed(text_features(name), tokens(name)) for name in names])
    started = time.monotonic()
    boxes, masks = [], []
    for count, record in enumerate(ground_truth.loadImgs(image_ids), start=1):
        pixels = preprocess_image((dataset / IMAGES / record["file_name"]).read_bytes())
        kept = (np.zeros((0, 6), np.float32), np.zeros(0, np.int64), np.zeros((0, MASK_SIZE, MASK_SIZE), np.float32))
        for category, (detections, read_masks) in zip(category_ids, runs(pixels), strict=True):
            kept = best_queries(kept, detections, read_masks, category)
        image_boxes, image_masks = coco_results(record["id"], record["width"], record["height"], kept)
        boxes += image_boxes
        masks += image_masks
        if count == 1 or count % 10 == 0 or count == len(image_ids):
            seconds = time.monotonic() - started
            log(f"{count}/{len(image_ids)} images, {seconds / count / len(names) * 1000:.0f} ms/prompt")
    inference_seconds = time.monotonic() - started

    started = time.monotonic()
    scorers = {}
    for kind, results in (("bbox", boxes), ("segm", masks)):
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = ground_truth.loadRes(results)
        scorer = COCOeval_faster(ground_truth, loaded, kind, print_function=log)
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
        scorers[kind] = scorer
    stats = coco_stats(scorers)
    counts = {"evaluated_samples": len(image_ids), "available_samples": len(available), "detections": len(boxes)}
    seconds = {
        "inference_seconds": round(inference_seconds, 1),
        "scoring_seconds": round(time.monotonic() - started, 1),
    }
    return stats, counts, seconds


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already prepared every input; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def tensor(array: np.ndarray) -> pyneat.Tensor:
    # from_numpy guesses CHW when the first dimension is 1, 3 or 4; every input here is HWC.
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(array), copy=True, layout=pyneat.TensorLayout.HWC)


def board_text_features(mpk: Path) -> dict[str, np.ndarray]:
    """Return the text MPK features [1, 16, 256] of every category prompt."""
    model = pyneat.Model(str(mpk), tensor_options())
    shape = list(model.input_specs()[0].shape)
    features = {}
    for name in COCO_TOKENS:
        (output,) = model.run([tensor(one_hot(tokens(name)).reshape(shape))], timeout_ms=TIMEOUT_MS)
        features[name] = np.array(output.to_numpy(), dtype=np.float32).reshape(1, CONTEXT, FEATURES)
    return features


def board_image_model(mpk: Path) -> ImageModel:
    def build(texts: list[np.ndarray]) -> Callable[[np.ndarray], Iterator[ImageRun]]:
        model = pyneat.Model(str(mpk), tensor_options())
        shapes = [list(spec.shape) for spec in model.input_specs()]
        pixel_index = shapes.index([IMAGE_SIZE, IMAGE_SIZE, 3])
        text_tensors = [tensor(text.reshape(shapes[1 - pixel_index])) for text in texts]

        def runs(pixels: np.ndarray) -> Iterator[ImageRun]:
            pixel_tensor = tensor(pixels)
            for text_tensor in text_tensors:
                inputs = [text_tensor, text_tensor]
                inputs[pixel_index] = pixel_tensor
                outputs = model.run(inputs, timeout_ms=TIMEOUT_MS)
                small, large = sorted(outputs, key=lambda output: math.prod(output.shape))
                detections = detection_rows(np.array(small.to_numpy(), dtype=np.float32))
                yield detections, lambda large=large: query_masks(large.to_numpy(copy=False))

        return runs

    return build


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"text_encoder", "image_model"}:
        raise SystemExit("--mpk must name exactly text_encoder=<path> and image_model=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="text_encoder=<path> or image_model=<path>")
    parser.add_argument("--dataset", type=Path, required=True, help="extracted coco-2017/mini directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=EVALUATED_IMAGES, help="measure the lowest N image IDs")
    args = parser.parse_args(argv)
    mpks = parse_mpks(args.mpk)

    # Encode every prompt first so the text encoder is released before the image model loads.
    features = board_text_features(mpks["text_encoder"])
    image_model = board_image_model(mpks["image_model"])
    stats, counts, seconds = evaluate(args.dataset, args.limit, features.__getitem__, image_model)

    result = {
        "core_version": pyneat.__version__,
        "dataset": {
            "split": IMAGES,
            "evaluated_samples": counts["evaluated_samples"],
            "available_samples": counts["available_samples"],
        },
        "metrics": {name: {"value": value, "unit": "ratio", "direction": "higher"} for name, value in stats.items()},
        "details": {"detections": counts["detections"], "prompts_per_image": len(COCO_TOKENS), **seconds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: round(item["value"], 4) for name, item in result["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
