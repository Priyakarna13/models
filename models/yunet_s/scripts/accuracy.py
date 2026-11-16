"""Measure YuNet-s face-detection AP on the WIDER FACE validation split with its MPK on a Modalix board.

Each image is letterboxed on the host as calibration.py does (centered, zero-padded, BGR pixel values). The MPK returns
the 12 raw head maps; they are decoded as OpenCV's FaceDetectorYN decodes them, kept by NMS at the thresholds OpenCV
Zoo evaluates YuNet with, mapped back through the letterbox, and scored with the official WIDER FACE evaluation
(widerface.py) on the easy, medium and hard subsets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import widerface
from calibration import IMAGE_SIZE, decode_image, letterbox, letterbox_geometry

DATASET = "wider-face-val"
IMAGES = "WIDER_val/images"
GROUND_TRUTH = "eval_tools/ground_truth"
STRIDES = (8, 16, 32)
# Channels per head, in the MPK's output order: face and object scores, box offsets, five landmark offsets.
HEADS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
OUTPUTS = [f"{head}_{stride}" for head in HEADS for stride in STRIDES]
GRIDS = {stride: IMAGE_SIZE // stride for stride in STRIDES}
# OpenCV Zoo's YuNet evaluation settings: score threshold, NMS IoU and the most detections kept per image.
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
TOP_K = 5000
# Images decode and letterbox on worker threads while the MPK runs; OpenCV releases the GIL meanwhile.
WORKERS = 4
PREFETCH = 8


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def head_maps(outputs: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """The 12 head maps by name, each flattened to [H*W, C] in row-major cell order; Neat returns them channels-last."""
    if len(outputs) != len(OUTPUTS):
        raise RuntimeError(f"expected {len(OUTPUTS)} head maps, got {len(outputs)}")
    maps = {}
    for name, output in zip(OUTPUTS, outputs, strict=True):
        head, stride = name.split("_")
        grid, channels = GRIDS[int(stride)], HEADS[head]
        array = np.asarray(output, dtype=np.float32)
        if array.size != grid * grid * channels or array.shape[-1] != channels:
            raise RuntimeError(f"unexpected {name} shape {array.shape}")
        maps[name] = array.reshape(grid * grid, channels)
    return maps


def decode(maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's box [x, y, w, h], score and five [x, y] landmarks in input pixels, as FaceDetectorYN decodes them."""
    boxes, scores, landmarks = [], [], []
    for stride in STRIDES:
        grid = GRIDS[stride]
        cells = np.arange(grid * grid)
        column, row = (cells % grid).astype(np.float32), (cells // grid).astype(np.float32)
        # The face score is the geometric mean of the clipped class and objectness probabilities.
        cls = np.clip(maps[f"cls_{stride}"][:, 0], 0, 1)
        obj = np.clip(maps[f"obj_{stride}"][:, 0], 0, 1)
        offsets, points = maps[f"bbox_{stride}"], maps[f"kps_{stride}"].reshape(-1, 5, 2)
        width, height = np.exp(offsets[:, 2]) * stride, np.exp(offsets[:, 3]) * stride
        centre_x, centre_y = (column + offsets[:, 0]) * stride, (row + offsets[:, 1]) * stride
        boxes.append(np.stack([centre_x - width / 2, centre_y - height / 2, width, height], axis=1))
        scores.append(np.sqrt(cls * obj))
        landmarks.append((points + np.stack([column, row], axis=1)[:, None, :]) * stride)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(landmarks)


def detect(maps: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    """The image's face detections as [x, y, w, h, score] rows in image pixels, after NMS, best first."""
    import cv2

    boxes, scores, _ = decode(maps)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), SCORE_THRESHOLD, NMS_THRESHOLD, top_k=TOP_K)
    keep = np.asarray(keep, dtype=int).reshape(-1)
    gain, _, (left, top, _, _) = letterbox_geometry(width, height)
    found = boxes[keep].astype(np.float64)
    found[:, 0] = (found[:, 0] - left) / gain
    found[:, 1] = (found[:, 1] - top) / gain
    found[:, 2:4] /= gain
    # OpenCV Zoo rounds boxes to 0.1 pixel and scores to three decimals before scoring.
    return np.concatenate([np.round(found, 1), np.round(scores[keep], 3)[:, None]], axis=1)


def load_image(root: Path, item: tuple[str, str]) -> tuple[np.ndarray, int, int]:
    """The letterboxed model input for one validation image, with the image's width and height."""
    image = decode_image((root / IMAGES / item[0] / f"{item[1]}.jpg").read_bytes())
    height, width = image.shape[:2]
    return letterbox(image).astype(np.float32), width, height


def parse_dataset(parser: argparse.ArgumentParser, value: str) -> Path:
    dataset_id, _, directory = value.partition("=")
    if dataset_id != DATASET or not directory:
        parser.error(f"--dataset must name {DATASET}=<extracted directory>")
    # The archive extracts to one wider-face-val/ folder; take that folder or the directory holding it.
    root = Path(directory) / DATASET if (Path(directory) / DATASET).is_dir() else Path(directory)
    for required in (IMAGES, GROUND_TRUTH):
        if not (root / required).is_dir():
            parser.error(f"{root} has no {required}/")
    return root


def read_ground_truth(root: Path) -> list:
    """The validation images and their faces from the official ground truth, in its order."""
    return widerface.load_ground_truth(root / GROUND_TRUTH)


def parse_mpks(values: list[str]) -> Path:
    paths = dict(value.split("=", 1) for value in values)
    if set(paths) != {"model"}:
        raise SystemExit("pass the model component's MPK as --mpk model=<path>")
    return Path(paths["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def model_detections(mpk: Path, root: Path, items: Sequence[tuple[str, str]]) -> Iterator[np.ndarray]:
    """Run each letterboxed image through the MPK and yield its detections."""
    import pyneat

    options = pyneat.ModelOptions()
    # The host letterboxes and keeps BGR pixel values; Neat only quantizes and tessellates.
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(str(mpk), options)
    runner = None
    try:
        for image, width, height in prefetched(lambda item: load_image(root, item), items):
            tensor = pyneat.Tensor.from_numpy(image, copy=False)
            if runner is None:
                runner = model.build([tensor])
            outputs = [output.to_numpy(copy=True) for output in runner.run([tensor], timeout_ms=120000)]
            yield detect(head_maps(outputs), width, height)
    finally:
        if runner is not None:
            runner.close()


def score(
    root: Path, items: Sequence[tuple[str, str]], detections: Iterator[np.ndarray], ground_truth: list
) -> tuple[dict[str, float], int, float]:
    """WIDER FACE AP per subset over the given images, the detection count and the inference time."""
    found, started = {}, time.monotonic()
    for count, (item, rows) in enumerate(zip(items, detections, strict=True), start=1):
        found[item] = rows
        if count == 1 or count % 500 == 0 or count == len(items):
            log(f"{count}/{len(items)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    seconds = time.monotonic() - started
    selected = set(items)
    truth = [entry for entry in ground_truth if (entry[0], entry[1]) in selected]
    aps = widerface.evaluate(found, truth)
    metrics = {f"widerface_ap_{setting}": value for setting, value in aps.items()}
    return metrics, sum(len(rows) for rows in found.values()), seconds


def main(argv: list[str] | None = None) -> int:
    import model_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="package resource directory (license only)")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    root = parse_dataset(parser, args.dataset)
    ground_truth = read_ground_truth(root)
    available = [(event, name) for event, name, _, _ in ground_truth]
    items = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    metrics, detections, seconds = score(root, items, model_detections(parse_mpks(args.mpk), root, items), ground_truth)
    report = model_report.from_accuracy(
        {
            DATASET: {
                "metrics": metrics,
                "split": "validation",
                "evaluated_samples": len(items),
                "available_samples": len(available),
                "details": {
                    "detections": detections,
                    "protocol": "640x640 centered zero-padded letterbox, BGR pixel values; FaceDetectorYN decode; "
                    f"score threshold {SCORE_THRESHOLD}, NMS IoU {NMS_THRESHOLD}, top {TOP_K}",
                    "inference_seconds": round(seconds, 1),
                },
            },
        }
    )
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
