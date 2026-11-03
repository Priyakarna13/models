"""Score Faster R-CNN ResNet-50 FPN v2 on COCO val2017 as torchvision evaluates it.

Each image is preprocessed as torchvision's GeneralizedRCNNTransform does, into the top-left corner of the
backbone's 1344x1344 input. The backbone MPK returns the FPN maps and the RPN head's outputs; calibration.py
selects the 1,000 proposals and pools their regions on the CPU; the box head MPK scores them; and
calibration.py decodes the detections, at most 100 per image, mapped back to the original image.
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
from calibration import (
    decode_image,
    detections,
    preprocess,
    proposals,
    region_rows,
    roi_align,
    tile_regions,
    unpack_backbone,
)
from faster_coco_eval import COCO, COCOeval_faster

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
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
MAX_DETECTIONS = [1, 10, 100]
# Images decode and preprocess on worker threads while the MPKs run; OpenCV releases the GIL.
WORKERS = 4
PREFETCH = 8


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized, normalized and padded the image; Neat still quantizes and tessellates.
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


class ModelArtifact:
    def __init__(self, path: Path) -> None:
        self.model = pyneat.Model(str(path), tensor_options())
        self.runner = None

    def __call__(self, array: np.ndarray) -> list[np.ndarray]:
        # The input stays alive until run() returns, so it is wrapped without a copy.
        tensors = [pyneat.Tensor.from_numpy(array, copy=False)]
        if self.runner is None:
            self.runner = self.model.build(tensors)
        return [view(output) for output in self.runner.run(tensors, timeout_ms=120000)]


def view(tensor: pyneat.Tensor) -> np.ndarray:
    """Read-only view of an output; valid until its MPK runs again."""
    array = tensor.to_numpy(copy=False)
    if array.dtype != np.float32:
        raise ValueError(f"expected a float32 MPK output, got {tensor.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("MPK output contains non-finite values")
    return array


def detect(backbone: ModelArtifact, box_head: ModelArtifact, image: np.ndarray, size: tuple[int, int]):
    """One preprocessed image -> boxes in the resized image, scores and COCO category IDs."""
    features, rpn = unpack_backbone(backbone(image)[0])
    boxes = proposals(rpn, size)
    class_logits, box_regression = box_head(tile_regions(roi_align(features, boxes)))
    count = len(boxes)
    return detections(region_rows(class_logits)[:count], region_rows(box_regression)[:count], boxes, size)


def result_rows(image_id: int, width: int, height: int, size: tuple[int, int], found) -> np.ndarray:
    """COCO result rows [image_id, x, y, w, h, score, category_id], in the original image's pixels."""
    boxes, scores, labels = found
    boxes = boxes * np.array([width / size[1], height / size[0]] * 2, np.float32)
    ids = np.full(len(boxes), image_id)
    return np.stack(
        [ids, boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], scores, labels], axis=1
    )


def coco_stats(scorer: COCOeval_faster) -> dict[str, float]:
    """Every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in METRICS.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "box_head"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and box_head=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for backbone and box_head")
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

    backbone, box_head = ModelArtifact(mpks["backbone"]), ModelArtifact(mpks["box_head"])
    rows: list[np.ndarray] = []
    started = time.monotonic()
    images = prefetched(
        lambda record: preprocess(decode_image((dataset / IMAGES / record["file_name"]).read_bytes())), records
    )
    for count, (record, (image, size)) in enumerate(zip(records, images), start=1):
        found = detect(backbone, box_head, image, size)
        rows.append(result_rows(record["id"], record["width"], record["height"], size, found))
        if count == 1 or count % 250 == 0 or count == len(records):
            log(f"{count}/{len(records)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    inference_seconds = time.monotonic() - started

    found = np.concatenate(rows)
    if not len(found):
        raise SystemExit("the model produced no detections")
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(found)
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
                    "detections": len(found),
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
