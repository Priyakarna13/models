"""Measure EfficientSAM3 RepViT-M COCO box and mask AP and AR with its two MPKs on a Modalix board.

Given --mpk text_encoder=<path> --mpk image_model=<path> and --dataset coco-2017-val=<extracted directory>, the script
scores the whole model by the package's own protocol; published SAM 3 and EfficientSAM3 scores use others, such as
box-prompted mask mIoU:

1. The 100 lowest-ID val2017 images (--limit changes the count for local checks), each prompted with all 80 COCO
   category names in category-ID order. The text encoder runs once per category; the image model runs once per image
   and category, 8000 runs.
2. A prompt is the CLIP BPE start token, the name's tokens and the end token. COCO_TOKENS is calibration.py's table,
   which convert.py checks against the pinned tokenizer; tests/benchmark/test_accuracy.py checks this copy against it.
3. The image is decoded to RGB ignoring EXIF orientation, as the COCO annotations do, stretched to 1008x1008 with
   bilinear filtering and scaled with x / 127.5 - 1, as calibration.py prepares it. The script prepares every input,
   so Neat only quantizes and tessellates.
4. COCO scores at most 100 detections per image, so of an image's 80 x 200 query and category pairs the 100 highest
   scoring are kept, an earlier category winning a tie.
5. Boxes are scaled from 1008x1008 input pixels to the image and not clipped, as in SAM 3's processor. Each kept
   query's mask logits are resized bilinearly to the image (torch interpolate, align_corners=False), and a positive
   logit is foreground.
6. faster-coco-eval scores boxes (coco_map ...) and masks (coco_mask_map ...) over the evaluated images.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
EVALUATED_IMAGES = 100
IMAGE_SIZE = 1008
CONTEXT, VOCABULARY, FEATURES = 16, 49408, 256
START, END = 49406, 49407
QUERIES, MASK_SIZE = 200, 192
TOP_K = 100
TIMEOUT_MS = 120000
# Metric name -> COCOeval_faster.stats_as_dict label, per COCO IoU type. With maxDets [1, 10, 100], AR_all,
# AR_second and AR_third are the recall at 1, 10 and 100 detections per image.
LABELS = {
    "map": "AP_all",
    "ap50": "AP_50",
    "ap75": "AP_75",
    "ap_small": "AP_small",
    "ap_medium": "AP_medium",
    "ap_large": "AP_large",
    "ar1": "AR_all",
    "ar10": "AR_second",
    "ar100": "AR_third",
    "ar_small": "AR_small",
    "ar_medium": "AR_medium",
    "ar_large": "AR_large",
}
METRICS = {
    "bbox": {f"coco_{name}": label for name, label in LABELS.items()},
    "segm": {f"coco_mask_{name}": label for name, label in LABELS.items()},
}
MAX_DETECTIONS = [1, 10, 100]
# The 80 COCO 2017 category names in category-ID order, with their BPE token IDs between the start and end tokens.
COCO_TOKENS = {
    "person": (2533,),
    "bicycle": (11652,),
    "car": (1615,),
    "motorcycle": (10297,),
    "airplane": (16451,),
    "bus": (2840,),
    "train": (3231,),
    "truck": (4629,),
    "boat": (4440,),
    "traffic light": (3399, 1395),
    "fire hydrant": (1769, 8031, 773),
    "stop sign": (1691, 2292),
    "parking meter": (5984, 10104),
    "bench": (8771,),
    "bird": (3329,),
    "cat": (2368,),
    "dog": (1929,),
    "horse": (4558,),
    "sheep": (9629,),
    "cow": (9706,),
    "elephant": (10299,),
    "bear": (4298,),
    "zebra": (22548,),
    "giraffe": (22826,),
    "backpack": (14894,),
    "umbrella": (17143,),
    "handbag": (22654,),
    "tie": (3422,),
    "suitcase": (27792,),
    "frisbee": (45768,),
    "skis": (32843,),
    "snowboard": (33403,),
    "sports ball": (2054, 1069),
    "kite": (19867,),
    "baseball bat": (3470, 6575),
    "baseball glove": (3470, 16078),
    "skateboard": (31777,),
    "surfboard": (10176, 1972),
    "tennis racket": (5298, 37691),
    "bottle": (5392,),
    "wine glass": (2604, 3313),
    "cup": (1937,),
    "fork": (13700,),
    "knife": (8960,),
    "spoon": (14024,),
    "bowl": (3814,),
    "banana": (8922,),
    "apple": (3055,),
    "sandwich": (8687,),
    "orange": (4287,),
    "broccoli": (19094,),
    "carrot": (15435,),
    "hot dog": (2069, 1929),
    "pizza": (4474,),
    "donut": (18471,),
    "cake": (2972,),
    "chair": (4269,),
    "couch": (12724,),
    "potted plant": (48581, 3912),
    "bed": (2722,),
    "dining table": (8658, 2175),
    "toilet": (11071,),
    "tv": (1583,),
    "laptop": (10464,),
    "mouse": (9301,),
    "remote": (9687,),
    "keyboard": (13017,),
    "cell phone": (5533, 1951),
    "microwave": (24240,),
    "oven": (12579,),
    "toaster": (39061,),
    "sink": (14551,),
    "refrigerator": (36662,),
    "book": (1116,),
    "clock": (6716,),
    "vase": (20431,),
    "scissors": (30867,),
    "teddy bear": (11798, 4298),
    "hair drier": (2225, 39877),
    "toothbrush": (36841,),
}

# One image's kept queries: detection rows [k, 6], COCO category IDs [k] and mask logits [k, 192, 192].
Kept = tuple[np.ndarray, np.ndarray, np.ndarray]


def tokens(name: str) -> list[int]:
    return [START, *COCO_TOKENS[name], END]


def one_hot(prompt_tokens: Sequence[int]) -> np.ndarray:
    """The text encoder input: row i is one-hot for token i; padding rows are zero."""
    value = np.zeros((1, CONTEXT, VOCABULARY), np.float32)
    value[0, np.arange(len(prompt_tokens)), prompt_tokens] = 1
    return value


def packed(features: np.ndarray, prompt_tokens: Sequence[int]) -> np.ndarray:
    """The image model's text input: rows 0-15 hold the features, row 16 is 1 for each prompt token."""
    text = np.zeros((1, CONTEXT + 1, FEATURES), np.float32)
    text[0, :CONTEXT] = np.asarray(features, np.float32).reshape(CONTEXT, FEATURES)
    text[0, CONTEXT, : len(prompt_tokens)] = 1
    return text


def preprocess_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32) / np.float32(127.5) - np.float32(1)


def detection_rows(detections: np.ndarray) -> np.ndarray:
    """The image model's detections as [200, 6] rows: x1, y1, x2, y2 in input pixels, score, 0."""
    detections = np.asarray(detections, np.float32).reshape(QUERIES, 6)
    if not np.isfinite(detections).all():
        raise ValueError("image model detections contain non-finite values")
    return detections


def query_masks(mask_logits: np.ndarray) -> np.ndarray:
    """A [200, 192, 192] view of the image model's HWC [192, 192, 200] mask logits."""
    return np.moveaxis(mask_logits.reshape(MASK_SIZE, MASK_SIZE, QUERIES), -1, 0)


def best_queries(kept: Kept, detections: np.ndarray, masks: np.ndarray, category: int) -> Kept:
    """Merge one prompt's queries into the kept ones and keep the TOP_K best scores; earlier prompts win ties.

    Only the masks of queries that enter the kept set are copied.
    """
    scores = detections[:, 4]
    chosen = np.argsort(-scores, kind="stable")[:TOP_K]
    if len(kept[0]) == TOP_K:
        chosen = chosen[scores[chosen] > kept[0][-1, 4]]
    if not len(chosen):
        return kept
    chosen_masks = masks[chosen]
    if not np.isfinite(chosen_masks).all():
        raise ValueError("image model mask logits contain non-finite values")
    merged = (
        np.concatenate([kept[0], detections[chosen]]),
        np.concatenate([kept[1], np.full(len(chosen), category)]),
        np.concatenate([kept[2], chosen_masks]),
    )
    order = np.argsort(-merged[0][:, 4], kind="stable")[:TOP_K]
    return tuple(values[order] for values in merged)


def coco_results(image_id: int, width: int, height: int, kept: Kept) -> tuple[list[dict], list[dict]]:
    """COCO box and mask results for one image from its kept queries."""
    detections, categories, masks = kept
    boxes = detections[:, :4].astype(np.float64) * np.array([width, height, width, height]) / IMAGE_SIZE
    # Resizing the transposed masks builds the column-major [H, W, N] array that RLE encoding reads without a copy.
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


def coco_stats(ground_truth: COCO, image_ids: list[int], results: dict[str, list[dict]]) -> dict[str, float]:
    """Every metric by its COCOeval label, boxes and masks; COCOeval reports -1 for an empty object-size range."""
    stats = {}
    for kind, labels in METRICS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = ground_truth.loadRes(results[kind])
        scorer = COCOeval_faster(ground_truth, loaded, kind, print_function=log)
        scorer.params.imgIds = image_ids
        if list(scorer.params.maxDets) != MAX_DETECTIONS:
            raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
        stats.update({name: float(scorer.stats_as_dict[label]) for name, label in labels.items()})
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def tensor_options() -> pyneat.ModelOptions:
    """The script prepares every input; Neat only quantizes and tessellates."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def tensor(array: np.ndarray) -> pyneat.Tensor:
    # from_numpy guesses CHW when the first dimension is 1, 3 or 4; every input here is HWC.
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(array), copy=True, layout=pyneat.TensorLayout.HWC)


def encode_prompts(mpk: Path, names: list[str]) -> list[pyneat.Tensor]:
    """Each prompt's text encoder features packed for the image model. The text encoder is released on return."""
    model = pyneat.Model(str(mpk), tensor_options())
    runner, texts = None, []
    for name in names:
        prompt = tensor(one_hot(tokens(name)))
        if runner is None:
            runner = model.build([prompt])
        (features,) = runner.run([prompt], timeout_ms=TIMEOUT_MS)
        texts.append(tensor(packed(features.to_numpy(), tokens(name))))
    runner.close()
    return texts


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"text_encoder", "image_model"}:
        raise SystemExit("--mpk must name exactly text_encoder=<path> and image_model=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="text_encoder=<path> and image_model=<path>")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, default=EVALUATED_IMAGES, help="measure the lowest N image IDs")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)
    mpks = parse_mpks(args.mpk)

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = available[: args.limit]
    category_ids = sorted(ground_truth.getCatIds())
    names = [category["name"] for category in ground_truth.loadCats(category_ids)]
    if names != list(COCO_TOKENS):
        raise SystemExit("COCO categories do not match the prompt table")

    # Encode every prompt first, so the text encoder is released before the image model loads.
    texts = encode_prompts(mpks["text_encoder"], names)
    model = pyneat.Model(str(mpks["image_model"]), tensor_options())
    runner = None
    results: dict[str, list[dict]] = {"bbox": [], "segm": []}
    started = time.monotonic()
    for count, record in enumerate(ground_truth.loadImgs(image_ids), start=1):
        pixels = tensor(preprocess_image(dataset / IMAGES / record["file_name"]))
        kept = (np.zeros((0, 6), np.float32), np.zeros(0, np.int64), np.zeros((0, MASK_SIZE, MASK_SIZE), np.float32))
        for category, text in zip(category_ids, texts, strict=True):
            if runner is None:
                runner = model.build([pixels, text])
            detections, mask_logits = runner.run([pixels, text], timeout_ms=TIMEOUT_MS)
            rows = detection_rows(detections.to_numpy())
            kept = best_queries(kept, rows, query_masks(mask_logits.to_numpy(copy=False)), category)
        boxes, masks = coco_results(record["id"], record["width"], record["height"], kept)
        results["bbox"] += boxes
        results["segm"] += masks
        if count == 1 or count % 10 == 0 or count == len(image_ids):
            seconds = time.monotonic() - started
            log(f"{count}/{len(image_ids)} images, {seconds / count / len(names) * 1000:.0f} ms/prompt")
    inference_seconds = time.monotonic() - started

    started = time.monotonic()
    stats = coco_stats(ground_truth, image_ids, results)
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": stats,
            "split": IMAGES,
            "evaluated_samples": len(image_ids),
            "available_samples": len(available),
            "details": {
                "detections": len(results["bbox"]),
                "prompts_per_image": len(names),
                "inference_seconds": round(inference_seconds, 1),
                "scoring_seconds": round(time.monotonic() - started, 1),
            },
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
