"""Measure SSD-MobileNet V2 COCO bounding-box AP and AR with its MPK on a Modalix board.

The heads are decoded as the source graph's TensorFlow postprocessing does: MultipleGridAnchorGenerator
anchors, the FasterRcnnBoxCoder with scales (10, 10, 5, 5), sigmoid scores, and per-class NMS at IoU 0.6
keeping at most 100 detections above 1e-8 (ssd_mobilenet_v2_coco's eval settings).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import preprocess_image
from faster_coco_eval import COCO, COCOeval_faster

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# Background plus the 90 ids of TensorFlow's COCO label map, which are the COCO category ids.
CLASSES = 91
# Six feature maps with three boxes per cell on the lowest one and six on the others.
GRIDS = (19, 10, 5, 3, 2, 1)
BOXES_PER_CELL = (3, 6, 6, 6, 6, 6)
OUTPUT_SHAPES = [
    (1, size, size, boxes * channels) for channels in (4, CLASSES) for size, boxes in zip(GRIDS, BOXES_PER_CELL)
]
MIN_SCALE, MAX_SCALE = 0.2, 0.95
ASPECT_RATIOS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
BOX_SCALES = np.asarray([10.0, 10.0, 5.0, 5.0], np.float32)
SCORE_THRESHOLD = 1e-8
IOU_THRESHOLD = 0.6
MAX_DETECTIONS = 100
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label. With maxDets [1, 10, 100], AR_all,
# AR_second and AR_third are the recall at 1, 10 and 100 detections per image.
METRICS = {
    "coco_map": "AP_all",
    "coco_ap50": "AP_50",
    "coco_ap75": "AP_75",
    "coco_ap_small": "AP_small",
    "coco_ap_medium": "AP_medium",
    "coco_ap_large": "AP_large",
    "coco_ar1": "AR_all",
    "coco_ar10": "AR_second",
    "coco_ar100": "AR_third",
    "coco_ar_small": "AR_small",
    "coco_ar_medium": "AR_medium",
    "coco_ar_large": "AR_large",
}
COCO_MAX_DETECTIONS = [1, 10, 100]
# Images decode on worker threads while the MPK runs.
WORKERS = 4
PREFETCH = 8


def make_anchors() -> np.ndarray:
    """[1917, 4] (ymin, xmin, ymax, xmax) anchors in normalized coordinates, in the heads' cell-major order."""
    scales = [MIN_SCALE + (MAX_SCALE - MIN_SCALE) * level / (len(GRIDS) - 1) for level in range(len(GRIDS))] + [1.0]
    levels = []
    for level, size in enumerate(GRIDS):
        if level == 0:
            # reduce_boxes_in_lowest_layer: a 0.1 square box and the 2:1 and 1:2 boxes at the lowest scale.
            shapes = [(0.1, 1.0), (scales[0], 2.0), (scales[0], 0.5)]
        else:
            shapes = [(scales[level], ratio) for ratio in ASPECT_RATIOS]
            shapes.append((np.sqrt(scales[level] * scales[level + 1]), 1.0))
        heights = np.asarray([scale / np.sqrt(ratio) for scale, ratio in shapes], np.float32)
        widths = np.asarray([scale * np.sqrt(ratio) for scale, ratio in shapes], np.float32)
        centers = (np.arange(size, dtype=np.float32) + 0.5) / size
        center_y, center_x = (axis[..., None] for axis in np.meshgrid(centers, centers, indexing="ij"))
        boxes = np.stack(
            [center_y - heights / 2, center_x - widths / 2, center_y + heights / 2, center_x + widths / 2], axis=-1
        )
        levels.append(boxes.reshape(-1, 4))
    return np.concatenate(levels).astype(np.float32)


ANCHORS = make_anchors()


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized and scaled the image; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def decode_boxes(encodings: np.ndarray) -> np.ndarray:
    """Apply the box coder to [1917, 4] (ty, tx, th, tw) encodings; returns (ymin, xmin, ymax, xmax) clipped to 0-1."""
    anchor_height, anchor_width = ANCHORS[:, 2] - ANCHORS[:, 0], ANCHORS[:, 3] - ANCHORS[:, 1]
    anchor_y, anchor_x = ANCHORS[:, 0] + anchor_height / 2, ANCHORS[:, 1] + anchor_width / 2
    ty, tx, th, tw = (encodings / BOX_SCALES).T
    center_y, center_x = ty * anchor_height + anchor_y, tx * anchor_width + anchor_x
    height, width = np.exp(th) * anchor_height, np.exp(tw) * anchor_width
    boxes = np.stack([center_y - height / 2, center_x - width / 2, center_y + height / 2, center_x + width / 2], -1)
    return boxes.clip(0.0, 1.0)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    top, left = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    bottom, right = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    inter = np.maximum(bottom - top, 0) * np.maximum(right - left, 0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-12)


def select(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class greedy NMS keeping the MAX_DETECTIONS best detections overall: (anchor indices, class indices).

    Visiting every anchor/class pair in descending score and suppressing it against the kept boxes of its own
    class gives TensorFlow's per-class NMS followed by its top-100 merge, without running NMS on every class.
    """
    flat = scores.ravel()
    candidates = np.flatnonzero(flat > SCORE_THRESHOLD)
    order = candidates[np.argsort(-flat[candidates], kind="stable")]
    kept: dict[int, list[int]] = {}
    anchors, classes = [], []
    for pair in order:
        anchor, label = divmod(int(pair), scores.shape[1])
        same_class = kept.setdefault(label, [])
        if same_class and (_iou(boxes[anchor], boxes[same_class]) > IOU_THRESHOLD).any():
            continue
        same_class.append(anchor)
        anchors.append(anchor)
        classes.append(label)
        if len(anchors) == MAX_DETECTIONS:
            break
    return np.asarray(anchors, np.int64), np.asarray(classes, np.int64)


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int) -> np.ndarray:
    """Return COCO result rows [image_id, x, y, w, h, score, category_id] for one image."""
    shapes = [tuple(output.shape) for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    encodings = np.concatenate([output.reshape(-1, 4) for output in outputs[: len(GRIDS)]])
    logits = np.concatenate([output.reshape(-1, CLASSES) for output in outputs[len(GRIDS) :]])
    if not (np.isfinite(encodings).all() and np.isfinite(logits).all()):
        raise ValueError("MPK output contains non-finite values")
    boxes = decode_boxes(encodings)
    # Class 0 is the background; scores cover the 90 label-map ids.
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 1:], -80, 80)))
    anchor, label = select(boxes, scores)
    top, left, bottom, right = (boxes[anchor] * [height, width, height, width]).T
    ids = np.full(anchor.shape, image_id)
    return np.stack([ids, left, top, right - left, bottom - top, scores[anchor, label], label + 1], axis=1)


def coco_stats(scorer: COCOeval_faster) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != COCO_MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {COCO_MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in METRICS.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)

    runner = None
    rows: list[np.ndarray] = []
    started = time.monotonic()
    image_dir = dataset / IMAGES
    images = prefetched(lambda record: preprocess_image((image_dir / record["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
        rows.append(decode(outputs, record["id"], record["width"], record["height"]))
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    inference_seconds = time.monotonic() - started

    detections = np.concatenate(rows)
    if not len(detections):
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
    scorer = COCOeval_faster(ground_truth, results, "bbox", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    stats = coco_stats(scorer)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": stats,
                "split": IMAGES,
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "detections": len(detections),
                    "inference_seconds": round(inference_seconds, 1),
                    "scoring_seconds": round(scoring_seconds, 1),
                },
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
