"""Measure MobileSAM's COCO mask AP and AR with box prompts, with its MPKs on a Modalix board.

Given one --mpk <component>=<path> per component and --dataset coco-2017-val=<extracted directory>, the script
encodes each val2017 image once, then prompts the decoder with the ground-truth box of each object that is not a
crowd, as segment_anything's predictor does with multimask_output=False: the box's two corners, and the first of
the decoder's masks. Each mask is upsampled to the image as segment_anything does, thresholded at 0, and scored
with COCO's mask AP and AR under the object's own category, with the decoder's predicted IoU as its score. The
boxes come from the annotations, so the scores measure the masks alone, not detection.
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
from typing import Any

import cv2
import model_report
import numpy as np
import pyneat
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core import mask as mask_util
from pipeline import COMPONENTS, IMAGE_SIZE, LOW_RES_MASK_SIZE, MASKS, preprocess, prompt_inputs, resized_size

ANNOTATIONS = "annotations/instances_val2017.json"
IMAGES = "val2017"
# The published metrics are over the full val2017 split.
VAL2017_IMAGES = 5000
TIMEOUT_MS = 120000
# Metric name in model.yaml -> COCOeval_faster.stats_as_dict label. With maxDets [1, 10, 100], AR_all, AR_second
# and AR_third are the recall at 1, 10 and 100 masks per image.
METRICS = {
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
}
MAX_DETECTIONS = [1, 10, 100]
# Images decode and masks encode on worker threads while the MPKs run; OpenCV and Pillow release the GIL.
WORKERS = 4
PREFETCH = 8


def box_prompt(bbox: Sequence[float], original_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The decoder's point inputs for a COCO x, y, width, height box: its top-left and bottom-right corners."""
    x, y, width, height = bbox
    return prompt_inputs([(x, y), (x + width, y + height)], ["box_top_left", "box_bottom_right"], original_size)


def mask_layout(masks: np.ndarray, iou_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The decoder's outputs as [MASKS, 256, 256] logits and MASKS scores, from NCHW or NHWC."""
    masks = np.asarray(masks, np.float32)
    if masks.size != MASKS * LOW_RES_MASK_SIZE**2:
        raise ValueError(f"unexpected mask output shape {masks.shape}")
    if masks.shape[-1] == MASKS and masks.shape[-2] == LOW_RES_MASK_SIZE:
        masks = masks.reshape(LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE, MASKS).transpose(2, 0, 1)
    masks = masks.reshape(MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE)
    scores = np.asarray(iou_predictions, np.float32).ravel()
    if scores.size != MASKS:
        raise ValueError(f"expected {MASKS} IoU predictions, got {scores.size}")
    return masks, scores


def image_mask(logits: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """One low-resolution mask's logits as a binary mask on the image, as segment_anything's postprocess_masks
    does: bilinear upsampling (align_corners=False) to IMAGE_SIZE, the padding dropped, then a bilinear resize to
    the image, thresholded at 0."""
    height, width = original_size
    resized_height, resized_width = resized_size(height, width)
    upsampled = cv2.resize(logits, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    cropped = np.ascontiguousarray(upsampled[:resized_height, :resized_width])
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR) > 0


def result(annotation: dict[str, Any], masks: np.ndarray, scores: np.ndarray, original_size: tuple[int, int]) -> dict:
    """The COCO result for one object: the decoder's first mask, scored by its predicted IoU."""
    mask = image_mask(masks[0], original_size)
    rle = mask_util.encode(np.asfortranarray(mask[:, :, None].astype(np.uint8)))[0]
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return {
        "image_id": annotation["image_id"],
        "category_id": annotation["category_id"],
        "segmentation": rle,
        "score": float(scores[0]),
    }


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def evaluate(
    ground_truth: COCO,
    image_ids: list[int],
    image_dir: Path,
    encode: Callable[[np.ndarray], Any],
    decode: Callable[[Any, dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], int]:
    """Every object's COCO result, and the number of decoder runs. encode takes the encoder's HWC input and
    returns the image's embedding in any form decode takes; decode returns the masks and IoU predictions."""
    records = ground_truth.loadImgs(image_ids)
    results: list[dict] = []
    started, prompts = time.monotonic(), 0
    load = lambda record: preprocess(read_rgb(image_dir / record["file_name"]))  # noqa: E731
    with ThreadPoolExecutor(WORKERS) as encoders:
        pending: deque = deque()
        for count, (record, image) in enumerate(zip(records, prefetched(load, records)), start=1):
            size = (record["height"], record["width"])
            embedding = encode(image)
            for annotation in ground_truth.loadAnns(ground_truth.getAnnIds(imgIds=record["id"], iscrowd=False)):
                masks, scores = mask_layout(*decode(embedding, box_prompt(annotation["bbox"], size)))
                pending.append(encoders.submit(result, annotation, masks, scores, size))
                prompts += 1
            while len(pending) > 4 * WORKERS:
                results.append(pending.popleft().result())
            if count == 1 or count % 250 == 0 or count == len(records):
                log(f"{count}/{len(records)} images, {prompts} prompts, "
                    f"{(time.monotonic() - started) / count * 1000:.0f} ms/image")
        results.extend(future.result() for future in pending)
    return results, prompts


def coco_stats(scorer: COCOeval_faster) -> dict[str, float]:
    """Return every declared metric by its COCOeval label; COCOeval reports -1 for an empty object-size range."""
    if list(scorer.params.maxDets) != MAX_DETECTIONS:
        raise ValueError(f"expected COCOeval maxDets {MAX_DETECTIONS}, got {scorer.params.maxDets}")
    labelled = scorer.stats_as_dict
    stats = {name: float(labelled[label]) for name, label in METRICS.items()}
    empty = [name for name, value in stats.items() if value < 0]
    if empty:
        raise SystemExit(f"no ground-truth objects to score for: {', '.join(empty)}")
    return stats


def score(ground_truth: COCO, results: list[dict], image_ids: list[int]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = ground_truth.loadRes(results)
    scorer = COCOeval_faster(ground_truth, loaded, "segm", print_function=log)
    scorer.params.imgIds = image_ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    return coco_stats(scorer)


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: the script prepares the inputs; Neat still quantizes and tessellates them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def hwc(value: np.ndarray) -> pyneat.Tensor:
    """An HWC array, or an NCHW batch of one, as the HWC tensor Neat takes."""
    if value.ndim == 4:
        value = value[0].transpose(1, 2, 0)
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(value), copy=True, layout=pyneat.TensorLayout.HWC)


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """One component's outputs as fresh HWC inputs of the next, as performance.handoff explains. Neat 0.6 gives
    them a leading batch axis. Neat's own views of its outputs are read-only, which from_numpy rejects."""
    arrays = [tensor.to_numpy(copy=True) for tensor in tensors]
    return [
        pyneat.Tensor.from_numpy(
            np.ascontiguousarray(array[0] if array.ndim == 4 else array), copy=True, layout=pyneat.TensorLayout.HWC
        )
        for array in arrays
    ]


class Runners:
    """The three components on the board, each runner built from its first real input."""

    def __init__(self, mpks: dict[str, Path]) -> None:
        self.models = {name: pyneat.Model(str(mpks[name]), tensor_options()) for name in COMPONENTS}
        self.runners: dict[str, pyneat.ModelRunner] = {}

    def run(self, component: str, inputs: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
        if component not in self.runners:
            self.runners[component] = self.models[component].build(inputs)
        return self.runners[component].run(inputs, timeout_ms=TIMEOUT_MS)

    def encode(self, image: np.ndarray) -> list[pyneat.Tensor]:
        features = handoff(self.run("image_encoder_0", [hwc(image)]))
        return handoff(self.run("image_encoder_1", features))

    def decode(self, embedding: list[pyneat.Tensor], prompt: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        inputs = [*embedding, hwc(prompt["point_coords"]), hwc(prompt["point_types"])]
        masks, scores = self.run("prompt_decoder", inputs)
        # Copies, because the runner reuses its output buffers while a worker may still read them.
        return masks.to_numpy(copy=True), scores.to_numpy(copy=True)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of images (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    runners = Runners(parse_mpks(args.mpk))
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset / ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    if len(available) != VAL2017_IMAGES:
        raise SystemExit(f"expected {VAL2017_IMAGES} val2017 images, found {len(available)}")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    started = time.monotonic()
    results, prompts = evaluate(ground_truth, image_ids, dataset / IMAGES, runners.encode, runners.decode)
    inference_seconds = time.monotonic() - started
    if not results:
        raise SystemExit("the images have no objects to prompt")
    started = time.monotonic()
    stats = score(ground_truth, results, image_ids)
    scoring_seconds = time.monotonic() - started

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": stats,
            "split": IMAGES, "evaluated_samples": len(image_ids), "available_samples": len(available),
            "details": {
                "prompts": prompts,
                "prompt": "ground-truth box of each non-crowd object",
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
