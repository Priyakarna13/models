"""Measure RF-DETR Seg Nano COCO mask AP and AR and box AP with its three MPKs on a Modalix board.

Detections follow upstream RF-DETR's PostProcess for segmentation (rfdetr a459eeade, RFDETRSegNanoConfig
num_select 100): the 100 best query/class pairs by sigmoid score, boxes scaled to the image, and each pair's
mask logits resized bilinearly (align_corners=False) to the image and thresholded at zero.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import model_report
import numpy as np
from calibration import IMAGE_SIZE, MEAN, QUERY_COUNT, STD, select_top_proposals
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util
from PIL import Image

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
CLASSES = 91
MASK_SIZE = IMAGE_SIZE // 4
TOP_K = 100
SCORE_THRESHOLD = 0.001
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label, per COCO IoU type. With maxDets [1, 10, 100],
# AR_all, AR_second and AR_third are the recall at 1, 10 and 100 detections per image.
METRICS = {
    "segm": {
        "coco_mask_map": "AP_all",
        "coco_mask_ap50": "AP_50",
        "coco_mask_ap75": "AP_75",
        "coco_mask_ap_small": "AP_small",
        "coco_mask_ap_medium": "AP_medium",
        "coco_mask_ap_large": "AP_large",
        "coco_mask_ar1": "AR_all",
        "coco_mask_ar10": "AR_second",
        "coco_mask_ar100": "AR_third",
        "coco_mask_ar_small": "AR_small",
        "coco_mask_ar_medium": "AR_medium",
        "coco_mask_ar_large": "AR_large",
    },
    "bbox": {"coco_box_map": "AP_all", "coco_box_ap50": "AP_50"},
}
MAX_DETECTIONS = [1, 10, 100]
# Images decode on worker threads while the MPKs run; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
# Resizing and run-length encoding up to 100 full-size masks per image is CPU work; processes keep up with the MPKs.
DECODERS = 8
# calibration.preprocess_image's per-channel arithmetic, tabulated for every uint8 value and laid out HWC.
NORMALIZE = ((np.arange(256, dtype=np.float32)[:, None] / 255.0 - MEAN.ravel()) / STD.ravel()).reshape(1, 256, 3)


def preprocess(content: bytes) -> np.ndarray:
    """Return the same values as calibration.preprocess_image, in the backbone's HWC layout."""
    with Image.open(io.BytesIO(content)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return cv2.LUT(np.asarray(resized), NORMALIZE)


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


class ModelArtifact:
    def __init__(self, path: Path) -> None:
        # Imported here so fp32_baseline_accuracy.py can reuse this script on a host without PyNeat.
        import pyneat

        self.pyneat = pyneat
        self.model = pyneat.Model(str(path))
        self.runner = None

    def __call__(self, *inputs: np.ndarray | pyneat.Tensor) -> list[pyneat.Tensor]:
        # Inputs stay alive until run() returns, so NumPy arrays are wrapped without a copy.
        tensor = self.pyneat.Tensor
        tensors = [item if isinstance(item, tensor) else tensor.from_numpy(item, copy=False) for item in inputs]
        if self.runner is None:
            self.runner = self.model.build(tensors)
        return self.runner.run(tensors, timeout_ms=120000)


def view(tensor: pyneat.Tensor) -> np.ndarray:
    """Read-only view of an output; valid until its MPK runs again."""
    array = tensor.to_numpy(copy=False)
    if array.dtype != np.float32:
        raise ValueError(f"expected a float32 MPK output, got {tensor.dtype}")
    return array


def select(labels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected query/class pairs, best first, and their scores.

    Takes the TOP_K best pairs as upstream does, then drops the 11 unused COCO slots and scores below
    SCORE_THRESHOLD, which cannot change the AP of what remains.
    """
    if labels.shape != (1, QUERY_COUNT, CLASSES):
        raise ValueError(f"unexpected class logits: {labels.shape}")
    scores = (1.0 / (1.0 + np.exp(-np.clip(labels[0], -80, 80)))).ravel()
    order = np.argsort(-scores, kind="stable")[:TOP_K]
    order = order[(scores[order] >= SCORE_THRESHOLD) & valid[order % CLASSES]]
    return order, scores[order]


def image_boxes(dets: np.ndarray, queries: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map the queries' normalized cx, cy, w, h boxes to x, y, w, h in image pixels, clipped to the image."""
    cx, cy, w, h = dets[0, queries].T
    x1, x2 = np.clip((cx - w / 2) * width, 0, width), np.clip((cx + w / 2) * width, 0, width)
    y1, y2 = np.clip((cy - h / 2) * height, 0, height), np.clip((cy + h / 2) * height, 0, height)
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def encode_masks(mask_logits: np.ndarray, width: int, height: int) -> list[dict]:
    """Resize each query's mask logits to the image, threshold them at zero, and run-length encode them.

    OpenCV's bilinear resize samples at pixel centres, as torch's align_corners=False does when upsampling.
    """
    binary = np.empty((height, width, len(mask_logits)), dtype=np.uint8, order="F")
    for index, logits in enumerate(mask_logits):
        binary[:, :, index] = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0
    rles = mask_util.encode(binary)
    for rle in rles:
        rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rles


def decode(
    dets: np.ndarray, labels: np.ndarray, masks: np.ndarray, image_id: int, width: int, height: int, valid: np.ndarray
) -> list[dict]:
    """Return COCO result dicts, each with a box and a mask, for one image."""
    # The decoder MPK returns the mask logits channels-last: height, width, then one channel per query.
    if dets.shape != (1, QUERY_COUNT, 4) or masks.shape != (1, MASK_SIZE, MASK_SIZE, QUERY_COUNT):
        raise ValueError(f"unexpected decoder outputs: dets={dets.shape}, masks={masks.shape}")
    if not (np.isfinite(dets).all() and np.isfinite(labels).all() and np.isfinite(masks).all()):
        raise ValueError("MPK output contains non-finite values")
    order, scores = select(labels, valid)
    queries, categories = order // CLASSES, order % CLASSES
    boxes = image_boxes(dets, queries, width, height)
    rles = encode_masks(np.ascontiguousarray(masks[0][:, :, queries].transpose(2, 0, 1)), width, height) if len(order) else []
    return [
        {
            "image_id": image_id,
            "category_id": int(category),
            "score": float(score),
            "bbox": [float(value) for value in box],
            "segmentation": rle,
        }
        for category, score, box, rle in zip(categories, scores, boxes, rles, strict=True)
    ]


def coco_stats(scorer: COCOeval_faster, metrics: dict[str, str]) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in metrics.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def score(ground_truth: COCO, detections: list[dict], image_ids: list[int]) -> dict[str, float]:
    """Score the detections' masks and boxes against the ground truth."""
    stats = {}
    for iou_type, metrics in METRICS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            results = ground_truth.loadRes(detections)
        scorer = COCOeval_faster(ground_truth, results, iou_type, print_function=log)
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
        stats.update(coco_stats(scorer, metrics))
    return stats


def evaluate(
    dataset: Path,
    records: list[dict],
    valid: np.ndarray,
    infer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict]:
    """Run infer on each image's preprocessed HWC tensor and decode its boxes and masks on worker processes."""
    detections: list[dict] = []
    started = time.monotonic()
    images = prefetched(lambda record: preprocess((dataset / IMAGES / record["file_name"]).read_bytes()), records)
    with ProcessPoolExecutor(DECODERS, mp_context=multiprocessing.get_context("forkserver")) as decoders:
        pending: deque = deque()
        for count, (record, image) in enumerate(zip(records, images), start=1):
            dets, labels, masks = infer(image)
            pending.append(
                decoders.submit(decode, dets, labels, masks, record["id"], record["width"], record["height"], valid)
            )
            while len(pending) > 2 * DECODERS:
                detections.extend(pending.popleft().result())
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
        while pending:
            detections.extend(pending.popleft().result())
    return detections


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "decoder", "mask_head"}:
        raise SystemExit("--mpk must name exactly backbone=<path>, decoder=<path> and mask_head=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each of the three")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    mpks = parse_mpks(args.mpk)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    valid = np.isin(np.arange(CLASSES), ground_truth.getCatIds())

    backbone, decoder, mask_head = (ModelArtifact(mpks[name]) for name in ("backbone", "decoder", "mask_head"))

    def infer(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, scores, boxes = backbone(image)
        # The decoder's and mask head's EV74 input stages need their inputs in EV74 memory; only the proposals
        # pass through NumPy.
        proposals = select_top_proposals(view(scores), view(boxes))
        dets, labels, queries = decoder(features.cvu(), proposals)
        (masks,) = mask_head(features.cvu(), queries.cvu())
        # Outputs are views into the MPKs' buffers, so copy them before the next image runs.
        return view(dets).copy(), view(labels).copy(), view(masks).copy()

    started = time.monotonic()
    detections = evaluate(dataset, records, valid, infer)
    inference_seconds = time.monotonic() - started

    if not detections:
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    stats = score(ground_truth, detections, image_ids)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": stats,
            "split": IMAGES, "evaluated_samples": len(image_ids), "available_samples": len(available),
            "details": {
                "detections": len(detections),
                "inference_seconds": round(inference_seconds, 1),
                "scoring_seconds": round(scoring_seconds, 1),
            },
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
