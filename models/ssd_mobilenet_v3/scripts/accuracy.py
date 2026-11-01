"""Measure the compiled TorchVision SSDLite320 source contract on COCO.

The entry point measures native image preprocessing and native SSD decoding. The independent
NumPy helpers below remain a source oracle, not a fallback or an equivalence claim.

The CPU oracle preserves softmax91/background0, source score0.001 and IoU0.55,
per-class pre-NMS300 and final300. This is accuracy, not a native throughput claim.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
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
from faster_coco_eval import COCO, COCOeval_faster

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
WORKERS = 4
PREFETCH = 8
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
SIZE_METRICS = {name for name in METRICS if name.endswith(("_small", "_medium", "_large"))}


IMAGE_SIZE = 320
CLASSES = 91
GRIDS = (20, 10, 5, 3, 2, 1)
ANCHORS_PER_CELL = 6
ANCHOR_COUNT = 3234
OUTPUT_SHAPES = [(1, size, size, ANCHORS_PER_CELL * channels) for channels in (4, CLASSES) for size in GRIDS]
SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.55
PRE_NMS_TOPK = 300
MAX_DETECTIONS = 300


def anchors_numpy() -> np.ndarray:
    """Return [3234,4] xyxy priors in320-pixel coordinates, cell/anchor order."""
    scales = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
    levels = []
    for level, size in enumerate(GRIDS):
        scale = scales[level]
        pairs = [[scale, scale], [math.sqrt(scale * scales[level + 1])] * 2]
        for ratio in (2, 3):
            pairs.extend(
                [
                    [scale * math.sqrt(ratio), scale / math.sqrt(ratio)],
                    [scale / math.sqrt(ratio), scale * math.sqrt(ratio)],
                ]
            )
        widths_heights = np.clip(np.asarray(pairs, np.float32), 0, 1)
        xy = (np.arange(size, dtype=np.float32) + np.float32(0.5)) / size
        centers = np.stack(np.meshgrid(xy, xy), axis=-1).reshape(-1, 1, 2)
        centers = np.broadcast_to(centers, (size * size, ANCHORS_PER_CELL, 2)).reshape(-1, 2)
        widths_heights = np.tile(widths_heights, (size * size, 1))
        levels.append(
            np.concatenate([centers - 0.5 * widths_heights, centers + 0.5 * widths_heights], axis=1) * IMAGE_SIZE
        )
    return np.concatenate(levels)


ANCHORS = anchors_numpy()


def decode_numpy(reg: np.ndarray, anc: np.ndarray = ANCHORS) -> np.ndarray:
    """Apply Tx/Ty/Tw/Th with10/10/5/5 scales and source exp cap; clip xyxy to320."""
    widths_heights = anc[:, 2:] - anc[:, :2]
    center = anc[:, :2] + widths_heights * 0.5
    xy = reg[:, :2] / np.float32(10) * widths_heights + center
    wh = np.exp(np.minimum(reg[:, 2:] / np.float32(5), np.float32(math.log(1000 / 16)))) * widths_heights
    return np.clip(np.concatenate([xy - wh * 0.5, xy + wh * 0.5], axis=1), 0, IMAGE_SIZE)


def post_numpy(reg: np.ndarray, logits: np.ndarray, anc: np.ndarray = ANCHORS) -> dict[str, np.ndarray]:
    """Softmax91 including background; per-class top300/NMS then global top300.

    Preserve source degenerate-box behavior: clipping does not add a positive-area filter.
    """
    if reg.shape != (len(anc), 4) or logits.shape != (len(anc), CLASSES):
        raise ValueError("unexpected localization, confidence or anchor shape")
    if not (np.isfinite(reg).all() and np.isfinite(logits).all() and np.isfinite(anc).all()):
        raise ValueError("MPK output or anchors contain non-finite values")
    boxes = decode_numpy(reg, anc)
    exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores = exponential / exponential.sum(axis=1, keepdims=True)
    survivors = []
    for label in range(1, CLASSES):
        indices = np.flatnonzero(scores[:, label] > SCORE_THRESHOLD)
        indices = indices[np.argsort(-scores[indices, label], kind="stable")[:PRE_NMS_TOPK]]
        while len(indices):
            index = indices[0]
            survivors.append((float(scores[index, label]), label, int(index)))
            others = indices[1:]
            low, high = np.maximum(boxes[index, :2], boxes[others, :2]), np.minimum(boxes[index, 2:], boxes[others, 2:])
            intersection = np.maximum(0, high - low).prod(axis=1)
            area = np.maximum(0, boxes[index, 2:] - boxes[index, :2]).prod()
            areas = np.maximum(0, boxes[others, 2:] - boxes[others, :2]).prod(axis=1)
            union = area + areas - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
            indices = others[iou <= IOU_THRESHOLD]
    survivors = sorted(survivors, key=lambda item: -item[0])[:MAX_DETECTIONS]
    return {
        "scores": np.array([item[0] for item in survivors], np.float32),
        "labels": np.array([item[1] for item in survivors], np.int64),
        "boxes": np.array([boxes[item[2]] for item in survivors], np.float32).reshape(-1, 4),
    }


def decode(outputs: Sequence[np.ndarray], image_id: int, width: int, height: int) -> np.ndarray:
    """Map12 raw NHWC heads to COCO rows [image_id,x,y,w,h,score,category_id]."""
    if any(not isinstance(output, np.ndarray) or output.dtype != np.float32 for output in outputs):
        raise ValueError("MPK outputs must be float32 NumPy arrays")
    shapes = [tuple(output.shape) for output in outputs]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"unexpected MPK outputs: {shapes}")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    reg = np.concatenate([output.reshape(-1, 4) for output in outputs[: len(GRIDS)]]).astype(np.float32, copy=False)
    logits = np.concatenate([output.reshape(-1, CLASSES) for output in outputs[len(GRIDS) :]]).astype(
        np.float32, copy=False
    )
    result = post_numpy(reg, logits)
    # TorchVision transform.postprocess rescales xyxy to the original image before COCO xywh conversion.
    boxes = result["boxes"] * np.asarray([width / IMAGE_SIZE, height / IMAGE_SIZE] * 2, np.float32)
    ids = np.full(len(boxes), image_id)
    return np.column_stack([ids, boxes[:, :2], boxes[:, 2:] - boxes[:, :2], result["scores"], result["labels"]])


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def coco_stats(scorer: COCOeval_faster, *, allow_missing_size_buckets: bool = False) -> dict[str, float]:
    """Keep undefined size buckets out of local subset reports, never inventing zero accuracy for them.

    COCOeval uses -1 for an empty object-size range. Only --limit may omit these optional metrics;
    the required coco_map and every full-dataset metric remain mandatory, finite ratios.
    """
    if list(scorer.params.maxDets) != COCO_MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {COCO_MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in METRICS.items()}
    invalid = [name for name, value in stats.items() if not np.isfinite(value) or value > 1 or value < -1]
    if invalid:
        raise SystemExit(f"invalid COCO metrics: {', '.join(invalid)}")
    absent = {name for name, value in stats.items() if value == -1 and name in SIZE_METRICS}
    omitted = absent if allow_missing_size_buckets else set()
    invalid = [name for name, value in stats.items() if value < 0 and name not in omitted]
    if invalid:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(invalid)}")
    return {name: value for name, value in stats.items() if name not in omitted}


def parse_mpks(values: list[str]) -> Path:
    if len(values) != 1:
        raise SystemExit("--mpk must name exactly model=<path>")
    component, separator, path = values[0].partition("=")
    if component != "model" or not separator or not path.strip():
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(path)


def one_argument(values: list[str], name: str) -> str:
    if len(values) != 1 or not values[0].strip():
        raise SystemExit(f"{name} requires exactly one nonempty value")
    return values[0]


def parse_dataset(values: list[str]) -> tuple[str, Path]:
    value = one_argument(values, "--dataset")
    dataset_id, separator, directory = value.partition("=")
    if dataset_id != "coco-2017-val" or not separator or not directory.strip():
        raise SystemExit("--dataset must name coco-2017-val=<directory>")
    return dataset_id, Path(directory)


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def native_options() -> pyneat.ModelOptions:
    """Explicit source-multiclass Image route; the independent Float32 oracle remains above."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = 1280
    options.preprocess.input_max_height = 720
    options.preprocess.input_max_depth = 3
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = [0.5, 0.5, 0.5]
    options.preprocess.normalize.stddev = [0.5, 0.5, 0.5]
    options.preprocess.normalize.has_explicit_stats = True
    options.decode_type = pyneat.BoxDecodeType.Ssd
    options.num_classes = CLASSES
    options.score_threshold = SCORE_THRESHOLD
    options.nms_iou_threshold = IOU_THRESHOLD
    options.top_k = MAX_DETECTIONS
    # Geometry comes from each original image, never from the capacity seed.
    options.boxdecode_original_width = 0
    options.boxdecode_original_height = 0
    return options


def read_rgb_record(record, image_dir: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(image_dir / record["file_name"]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.dtype != np.uint8 or image.shape != (record["height"], record["width"], 3):
        raise ValueError(f"image bytes/geometry disagree with annotations: {record['file_name']}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def native_rows(outputs, image_id: int, width: int, height: int) -> np.ndarray:
    """Parse, without rescoring/rescaling/clamping, the one native typed BBOX output."""
    if len(outputs) != 1 or not pyneat.detections.format_is_bbox(pyneat.detections.read_detection_format(outputs[0])):
        raise ValueError("expected one typed BBOX output, not raw heads")
    parsed = pyneat.detections.decode_bbox_tensor(outputs[0], 0, 0, expected_topk=MAX_DETECTIONS, strict=True)
    if len(parsed.boxes) > MAX_DETECTIONS:
        raise ValueError("native detection count exceeds topK")
    rows = []
    for box in parsed.boxes:
        values = [box.x1, box.y1, box.x2, box.y2, box.score, box.class_id]
        if not np.isfinite(values).all():
            raise ValueError("non-finite native detection")
        if not (
            float(np.float32(SCORE_THRESHOLD)) <= box.score <= 1
            and int(box.class_id) == box.class_id
            and not isinstance(box.class_id, bool)
            and 1 <= box.class_id <= 90
            and 0 <= box.x1 <= width
            and 0 <= box.y1 <= height
            and box.x1 <= box.x2 <= width + 1
            and box.y1 <= box.y2 <= height + 1
        ):
            raise ValueError("invalid native score, category or original-image geometry")
        # The native format independently rounds x/y/w/h; a right/bottom edge can overhang by1px.
        # Preserve that and zero-area integer boxes for honest evaluation; do not silently repair output.
        rows.append([image_id, box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1, box.score, box.class_id])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


def native_predictions(model, records, image_dir: Path) -> list[np.ndarray]:
    """Build a bounded image route once, then use each frame's original dimensions."""
    if not records:
        raise ValueError("no images to evaluate")
    if any(not (0 < r["width"] <= 1280 and 0 < r["height"] <= 720) for r in records):
        raise ValueError("original image exceeds the 1280x720 capacity envelope")
    seed = np.full((720, 1280, 3), [32, 96, 160], dtype=np.uint8)
    seed_tensor = pyneat.Tensor.from_numpy(seed, copy=True, image_format=pyneat.PixelFormat.RGB)
    run_options = pyneat.RunOptions()
    run_options.startup_preflight = False
    runner = model.build([seed_tensor], run_options=run_options)
    rows = []
    started = time.monotonic()
    try:
        images = prefetched(lambda record: read_rgb_record(record, image_dir), records)
        for count, (record, image) in enumerate(zip(records, images), start=1):
            tensor = pyneat.Tensor.from_numpy(image, copy=True, image_format=pyneat.PixelFormat.RGB)
            outputs = runner.run([tensor], timeout_ms=120000)
            try:
                rows.append(native_rows(outputs, record["id"], record["width"], record["height"]))
            finally:
                outputs.clear()  # Release output loans before the next push.
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    finally:
        runner.close()
    return rows


def coco_results(ground_truth, detections: np.ndarray):
    """An empty detector is a genuine zero-AP result, not missing evidence."""
    if len(detections):
        return ground_truth.loadRes(detections)
    result = COCO()
    result.dataset = {
        "images": ground_truth.dataset["images"],
        "categories": ground_truth.dataset["categories"],
        "annotations": [],
    }
    result.createIndex()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", action="append", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", action="append", required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    mpk = parse_mpks(args.mpk)
    dataset_id, dataset = parse_dataset(args.dataset)
    output = Path(one_argument(args.output, "--output"))

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    if args.limit is not None and not 0 < args.limit <= len(available):
        raise SystemExit("--limit must be positive and no larger than the available dataset")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)

    model = pyneat.Model(str(mpk), native_options())
    started = time.monotonic()
    rows = native_predictions(model, records, dataset / IMAGES)
    inference_seconds = time.monotonic() - started

    detections = np.concatenate(rows)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = coco_results(ground_truth, detections)
    scorer = COCOeval_faster(ground_truth, results, "bbox", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    stats = coco_stats(scorer, allow_missing_size_buckets=len(image_ids) < len(available))
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
                    "route": "original RGB -> native Image preprocessing -> native Ssd -> typed BBOX",
                    "score_threshold": SCORE_THRESHOLD,
                    "nms_iou_threshold": IOU_THRESHOLD,
                    "top_k": MAX_DETECTIONS,
                    "reference_note": "Native preprocessing/decoder are not claimed bit-equivalent to the FP32 oracle",
                    "unavailable_metrics": sorted(set(METRICS) - stats.keys()),
                    "inference_seconds": round(inference_seconds, 1),
                    "scoring_seconds": round(scoring_seconds, 1),
                },
            },
        }
    )
    report.save(output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
