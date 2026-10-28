"""Measure compiled PointNet++ SSG semantic segmentation on S3DIS Area 5 and save its model report.

Two protocols cut each room into 4096-point blocks, centred and featured as yanx27's ScannetDatasetWholeScene does:

- tiles (the default, which CI runs): each non-overlapping 1 m x 1 m column gives one block of 4096 of its points,
  drawn with a fixed seed; the scored points are the ones drawn. Area 5 is 2,409 blocks.
- whole (yanx27's test_semseg.py, for local runs): columns every 0.5 m, every point of a column resampled into
  blocks, and each point's label voted over the blocks that hold it. Area 5 is 65,955 blocks, too many for a
  DevKit stage.

A block runs through the eight MPKs with pointcloud.py's CPU steps between them. As in performance.py's
throughput loop, the sampling and neighbour plans of upcoming blocks are computed in PLAN_WORKERS processes, so the
tile protocol fits the DevKit stage; the blocks and their order do not change. Scores are over the scored
points of the area: per-class IoU, their mean (mIoU), overall accuracy and mean class accuracy.

    python accuracy.py --mpk sa1=<archive> ... --mpk fp1=<archive> \
        --dataset s3dis-val=<extracted directory> --output <directory>
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import model_report
import numpy as np
import pyneat
from pointcloud import CLASS_NAMES, CLASSES, COMPONENTS, FEATURES, POINTS, channels_last, plan, segment

DATASET = "s3dis-val"
SPLIT = "Area_5"
BLOCK_SIZE = 1.0
STRIDE = 0.5
PADDING = 0.001
# Sampling a column's points is random; a fixed seed makes every run measure the same blocks.
SEED = 0
PROTOCOLS = ("tiles", "whole")
METRICS = ("miou", "overall_accuracy", "mean_class_accuracy")
# One block is ~880 ms on a DevKit, mostly the CPU plan: serially the tile protocol outlasts the 45-minute stage.
PLAN_WORKERS = 4


def load_rooms(directory: Path) -> list[tuple[str, dict[str, np.ndarray]]]:
    """The area's rooms in manifest order: xyz (shifted to the room's minimum), rgb and label of every point, the
    labels renumbered from the dataset's class order to the model's.

    The dataset holds val/<area>/<room>/{xyz,rgb}.npy and annotations/<area>/<room>.npy, with the class names in
    annotations/classes.txt.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rooms = manifest.get("rooms", [])
    if (manifest.get("id"), manifest.get("variant")) != ("s3dis", "val") or not all(
        room.startswith(f"{SPLIT}/") for room in rooms
    ):
        raise SystemExit(f"{directory} is not the {DATASET} dataset ({SPLIT})")
    classes = (directory / "annotations" / "classes.txt").read_text(encoding="utf-8").split()
    if sorted(classes) != sorted(CLASS_NAMES):
        raise SystemExit(f"{directory} does not label the model's classes: {classes}")
    # The model scores chair before table and bookcase before sofa; S3DIS archives often order them otherwise.
    to_model = np.array([CLASS_NAMES.index(name) for name in classes], np.uint8)
    loaded = []
    for room in rooms:
        data = {key: np.load(directory / "val" / room / f"{key}.npy", allow_pickle=False) for key in ("xyz", "rgb")}
        data["label"] = np.load(directory / "annotations" / f"{room}.npy", allow_pickle=False)
        if data["xyz"].shape != (len(data["label"]), 3) or data["label"].max() >= CLASSES:
            raise SystemExit(f"{room} is malformed")
        data["label"] = to_model[data["label"]]
        loaded.append((room, data))
    return loaded


def blocks(
    room: dict[str, np.ndarray], rng: np.random.RandomState, protocol: str = "tiles"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Each [4096, 9] block of a room and the room indices of its points, under the given protocol."""
    xyz = room["xyz"].astype(np.float64)
    rgb = room["rgb"].astype(np.float64) / 255.0
    low, high = xyz.min(0), xyz.max(0)
    stride = BLOCK_SIZE if protocol == "tiles" else STRIDE
    columns_x = int(np.ceil((high[0] - low[0] - BLOCK_SIZE) / stride) + 1)
    columns_y = int(np.ceil((high[1] - low[1] - BLOCK_SIZE) / stride) + 1)
    for row in range(columns_y):
        for column in range(columns_x):
            end_x = min(low[0] + column * stride + BLOCK_SIZE, high[0])
            end_y = min(low[1] + row * stride + BLOCK_SIZE, high[1])
            start_x, start_y = end_x - BLOCK_SIZE, end_y - BLOCK_SIZE
            inside = np.where(
                (xyz[:, 0] >= start_x - PADDING) & (xyz[:, 0] <= end_x + PADDING)
                & (xyz[:, 1] >= start_y - PADDING) & (xyz[:, 1] <= end_y + PADDING)
            )[0]
            if inside.size == 0:
                continue
            if protocol == "tiles":
                # One block: 4096 distinct points, or every point and random repeats when the column has fewer.
                if inside.size >= POINTS:
                    order = rng.choice(inside, POINTS, replace=False)
                else:
                    order = np.concatenate([inside, rng.choice(inside, POINTS - inside.size, replace=True)])
                    rng.shuffle(order)
            else:
                total = int(np.ceil(inside.size / POINTS)) * POINTS
                extra = rng.choice(inside, total - inside.size, replace=total - inside.size > inside.size)
                order = np.concatenate([inside, extra])
                rng.shuffle(order)
            centred = xyz[order] - [start_x + BLOCK_SIZE / 2, start_y + BLOCK_SIZE / 2, 0.0]
            features = np.concatenate([centred, rgb[order], xyz[order] / high], axis=1).astype(np.float32)
            for start in range(0, len(order), POINTS):
                yield features[start : start + POINTS], order[start : start + POINTS]


def scores(confusion: np.ndarray) -> dict[str, float]:
    """mIoU, overall accuracy and mean class accuracy of a [true, predicted] confusion matrix."""
    true_positive = np.diag(confusion).astype(np.float64)
    seen, predicted = confusion.sum(1), confusion.sum(0)
    present = seen > 0
    iou = true_positive[present] / (seen + predicted - true_positive)[present]
    return {
        "miou": float(iou.mean()),
        "overall_accuracy": float(true_positive.sum() / confusion.sum()),
        "mean_class_accuracy": float((true_positive[present] / seen[present]).mean()),
    }


def planned(items: Iterable[tuple], workers: int) -> Iterator[tuple]:
    """Each (..., features, points) item with plan(features' xyz) appended, in order. With workers, up to twice as
    many plans are computed ahead in spawned processes, which import only NumPy and pointcloud."""
    if not workers:
        for item in items:
            yield (*item, plan(item[-2][:, :3]))
        return
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        ahead = collections.deque()
        for item in items:
            ahead.append((item, pool.apply_async(plan, (np.ascontiguousarray(item[-2][:, :3]),))))
            if len(ahead) > 2 * workers:
                item, result = ahead.popleft()
                yield (*item, result.get())
        while ahead:
            item, result = ahead.popleft()
            yield (*item, result.get())


def evaluate(rooms, logits_of: Callable[[np.ndarray, dict], np.ndarray], limit: int | None = None,
             protocol: str = "tiles", workers: int = 0):
    """Vote each room's labels over its blocks and score the points that got a vote; return the metrics,
    per-class IoU and the block count. logits_of takes a block's features and its plan."""
    rng = np.random.RandomState(SEED)
    rooms = rooms[:limit]
    confusion = np.zeros((CLASSES, CLASSES), np.int64)
    count = 0
    started = time.monotonic()
    items = ((index, features, points) for index, (_, room) in enumerate(rooms)
             for features, points in blocks(room, rng, protocol))
    votes, current = None, None

    def score(index: int) -> None:
        nonlocal confusion
        name, room = rooms[index]
        scored = votes.any(1)
        pairs = room["label"][scored].astype(np.int64) * CLASSES + votes[scored].argmax(1)
        confusion += np.bincount(pairs, minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)
        print(f"{index + 1}/{len(rooms)} rooms ({name}), {count} blocks, "
              f"{(time.monotonic() - started) / count * 1000:.0f} ms/block", file=sys.stderr, flush=True)

    for index, features, points, indices in planned(items, workers):
        if index != current:
            if current is not None:
                score(current)
            votes, current = np.zeros((len(rooms[index][1]["label"]), CLASSES), np.int32), index
        np.add.at(votes, (points, logits_of(features, indices).argmax(1)), 1)
        count += 1
    if current is not None:
        score(current)
    seen, predicted, hit = confusion.sum(1), confusion.sum(0), np.diag(confusion)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = hit / (seen + predicted - hit)
    return scores(confusion), [None if np.isnan(value) else round(float(value), 4) for value in per_class], count


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if list(mpks) != list(COMPONENTS):
        raise SystemExit(f"--mpk must name {', '.join(COMPONENTS)} in that order, got {list(mpks)}")
    return {name: Path(path) for name, path in mpks.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="component=<archive>, once per component")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first rooms only (local runs only)")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="tiles", help="whole: yanx27's (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    rooms = load_rooms(Path(directory))

    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The inputs are grouped point features, not images: Neat only quantizes and tessellates them.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    models = {name: pyneat.Model(str(path), options) for name, path in parse_mpks(args.mpk).items()}
    runners = {}

    def run(component: str, value: np.ndarray) -> np.ndarray:
        # Each MPK takes its input as one [H, W, C] tensor.
        tensor = pyneat.Tensor.from_numpy(channels_last(value), copy=False)
        if component not in runners:
            runners[component] = models[component].build([tensor])
        (output,) = runners[component].run([tensor], timeout_ms=60000)
        return output.to_numpy(copy=True)

    started = time.monotonic()
    metrics, per_class, count = evaluate(rooms, lambda features, indices: segment(features, run, indices),
                                         args.limit, args.protocol, PLAN_WORKERS)
    result = {
        "metrics": metrics,
        "split": SPLIT,
        "evaluated_samples": len(rooms[: args.limit]),
        "available_samples": len(rooms),
        "details": {
            "protocol": args.protocol, "blocks": count, "points_per_block": POINTS, "features": FEATURES,
            "class_iou": per_class,
            "seconds": round(time.monotonic() - started, 1), "plan_workers": PLAN_WORKERS,
        },
    }
    report = model_report.from_accuracy({DATASET: result})
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
