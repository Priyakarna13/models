"""Benchmark Faster R-CNN ResNet-50 FPN v2 with Neat Core and save its report.

Given --mpk backbone=<path> --mpk box_head=<path> and --dataset coco-2017-val=<extracted directory>, the
script measures the whole model as it runs on one COCO image in a 720p frame: Neat's preprocessing and the
backbone, the 1,000 proposals and their RoIAlign on the CPU, the box head, and the detections' decoding and NMS
on the CPU, each stage working on a different frame at the same time. It also records each MPK's MLA layer and
stage stats and its execution plan.

The CPU steps follow torchvision's fasterrcnn_resnet50_fpn_v2 at inference, as calibration.py documents them;
this script keeps its own copy so it runs with only NumPy, OpenCV and Neat. Neat letterboxes the frame into the
1344x1344 input, centered, where torchvision places the image top-left: the script clips and maps boxes to the
frame's place in the input.
"""

from __future__ import annotations

import argparse
import math
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

INPUT_SIZE = 1344
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114
# torchvision's inference settings.
ANCHOR_SIZES = (32, 64, 128, 256, 512)
ASPECT_RATIOS = (0.5, 1.0, 2.0)
PRE_NMS_TOP_N = 1000
POST_NMS_TOP_N = 1000
RPN_NMS_THRESHOLD = 0.7
RPN_MIN_SIZE = 1e-3
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BOX_CLIP = math.log(1000.0 / 16)
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
BOX_MIN_SIZE = 1e-2
DETECTIONS = 100
CHANNELS = 256
CLASSES = 91
POOL = 7
SAMPLING_RATIO = 2
# The backbone's one output: each FPN level's cells as rows, P2-P6 in order, with P2-P5's 256 features then the
# RPN head's 3 objectness logits and 12 box deltas per cell, scaled by these gains into the features' range.
LEVEL_SIZES = tuple(INPUT_SIZE // stride for stride in (4, 8, 16, 32, 64))
LEVEL_ROWS = tuple(int(np.cumsum([0] + [size * size for size in LEVEL_SIZES])[i]) for i in range(6))
ANCHORS = 3
BACKBONE_CHANNELS = CHANNELS + 5 * ANCHORS
OBJECTNESS_GAIN, DELTA_GAIN = 1 / 16, 1 / 2
# The box head's input: the 1,000 regions' 7x7 maps in a 25x40 grid at an 8-pixel pitch, zero between them.
GRID_ROWS, GRID_COLS, PITCH = 25, 40, 8
REGIONS = GRID_ROWS * GRID_COLS
# RoIAlign and the RPN's per-level NMS run on worker threads; OpenCV and NumPy release the GIL.
WORKERS = 8
_POOL = ThreadPoolExecutor(WORKERS)


def image_options() -> pyneat.ModelOptions:
    """The backbone's: Neat letterboxes, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # Without a preset Neat only scales pixels to [0, 1]; the model expects ImageNet mean and std.
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The box head's: its input is the tiled regions."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


@dataclass(frozen=True)
class Window:
    """Where the letterboxed frame sits in the 1344x1344 input, and its scale."""

    top: float
    left: float
    bottom: float
    right: float
    scale: float


def letterbox_window() -> Window:
    scale = min(INPUT_SIZE / FRAME_WIDTH, INPUT_SIZE / FRAME_HEIGHT)
    width, height = FRAME_WIDTH * scale, FRAME_HEIGHT * scale
    top, left = (INPUT_SIZE - height) / 2, (INPUT_SIZE - width) / 2
    return Window(top, left, top + height, left + width, scale)


def unpack_backbone(output: np.ndarray):
    """The backbone's output -> P2-P5 as (H, W, 256) maps, and per level P2-P6 its objectness logits and deltas."""
    rows = output.reshape(-1, BACKBONE_CHANNELS)
    features = [
        np.ascontiguousarray(rows[LEVEL_ROWS[i] : LEVEL_ROWS[i + 1], :CHANNELS]).reshape(size, size, CHANNELS)
        for i, size in enumerate(LEVEL_SIZES[:4])
    ]
    rpn = [
        (
            rows[LEVEL_ROWS[i] : LEVEL_ROWS[i + 1], CHANNELS : CHANNELS + ANCHORS].reshape(-1)
            * np.float32(1 / OBJECTNESS_GAIN),
            rows[LEVEL_ROWS[i] : LEVEL_ROWS[i + 1], CHANNELS + ANCHORS :].reshape(-1, 4) * np.float32(1 / DELTA_GAIN),
        )
        for i in range(len(LEVEL_SIZES))
    ]
    return features, rpn


def base_anchors(size: int) -> np.ndarray:
    h_ratios = np.sqrt(np.array(ASPECT_RATIOS, np.float32))
    widths, heights = size / h_ratios, size * h_ratios
    return np.round(np.stack([-widths, -heights, widths, heights], axis=1) / 2).astype(np.float32)


def decode_boxes(deltas: np.ndarray, boxes: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    center_x, center_y = boxes[:, 0] + 0.5 * widths, boxes[:, 1] + 0.5 * heights
    dx, dy = deltas[:, 0] / weights[0], deltas[:, 1] / weights[1]
    dw = np.minimum(deltas[:, 2] / weights[2], BOX_CLIP)
    dh = np.minimum(deltas[:, 3] / weights[3], BOX_CLIP)
    x, y = dx * widths + center_x, dy * heights + center_y
    half_w, half_h = np.float32(0.5) * np.exp(dw) * widths, np.float32(0.5) * np.exp(dh) * heights
    return np.stack([x - half_w, y - half_h, x + half_w, y + half_h], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, window: Window) -> np.ndarray:
    return np.stack(
        [
            np.clip(boxes[:, 0], window.left, window.right),
            np.clip(boxes[:, 1], window.top, window.bottom),
            np.clip(boxes[:, 2], window.left, window.right),
            np.clip(boxes[:, 3], window.top, window.bottom),
        ],
        axis=1,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, groups: np.ndarray, threshold: float) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, np.int64)
    rects = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1).astype(np.float64)
    keep = np.asarray(
        cv2.dnn.NMSBoxesBatched(rects, scores.astype(np.float32), groups.astype(np.int32), -1.0, threshold),
        np.int64,
    ).reshape(-1)
    return keep[np.argsort(-scores[keep], kind="stable")]


def level_proposals(logits: np.ndarray, deltas: np.ndarray, level: int, window: Window):
    width = LEVEL_SIZES[level]
    count = min(PRE_NMS_TOP_N, logits.size)
    top = np.argpartition(-logits, count - 1)[:count] if count < logits.size else np.arange(count)
    top = top[np.argsort(-logits[top], kind="stable")]
    anchor, cell = top % ANCHORS, top // ANCHORS
    row, column = cell // width, cell % width
    shifts = np.stack([column, row, column, row], axis=1).astype(np.float32) * (INPUT_SIZE // width)
    anchors = base_anchors(ANCHOR_SIZES[level])[anchor] + shifts
    boxes = clip_boxes(decode_boxes(deltas[top], anchors, (1, 1, 1, 1)), window)
    scores = 1 / (1 + np.exp(-logits[top]))
    keep = np.where((boxes[:, 2] - boxes[:, 0] >= RPN_MIN_SIZE) & (boxes[:, 3] - boxes[:, 1] >= RPN_MIN_SIZE))[0]
    keep = keep[nms(boxes[keep], scores[keep], np.zeros(len(keep)), RPN_NMS_THRESHOLD)]
    return boxes[keep], scores[keep]


def proposals(rpn: Sequence[tuple[np.ndarray, np.ndarray]], window: Window) -> np.ndarray:
    kept = list(_POOL.map(lambda level: level_proposals(*rpn[level], level, window), range(len(rpn))))
    boxes = np.concatenate([item[0] for item in kept])
    scores = np.concatenate([item[1] for item in kept])
    return boxes[np.argsort(-scores, kind="stable")[:POST_NMS_TOP_N]]


def sample_coordinates(start: np.ndarray, end: np.ndarray, size: int) -> np.ndarray:
    bin_size = np.maximum(end - start, np.float32(1.0)) / np.float32(POOL)
    bins = np.arange(POOL, dtype=np.float32).repeat(SAMPLING_RATIO)
    offsets = np.tile(np.arange(SAMPLING_RATIO, dtype=np.float32) + np.float32(0.5), POOL) / np.float32(SAMPLING_RATIO)
    coordinates = start[:, None] + (bins + offsets)[None, :] * bin_size[:, None]
    inside = (coordinates >= -1.0) & (coordinates <= size)
    return np.where(inside, np.clip(coordinates, 0, size - 1), np.float32(-4.0)).astype(np.float32)


def level_roi_align(feature: np.ndarray, boxes: np.ndarray, scale: float) -> np.ndarray:
    height, width = feature.shape[:2]
    scaled = boxes * np.float32(scale)
    rows = sample_coordinates(scaled[:, 1], scaled[:, 3], height)
    columns = sample_coordinates(scaled[:, 0], scaled[:, 2], width)
    samples = POOL * SAMPLING_RATIO
    map_y = np.ascontiguousarray(np.broadcast_to(rows[:, :, None], (len(boxes), samples, samples)).reshape(-1, samples))
    map_x = np.ascontiguousarray(
        np.broadcast_to(columns[:, None, :], (len(boxes), samples, samples)).reshape(-1, samples)
    )
    sampled = cv2.remap(feature, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled = sampled.reshape(len(boxes), POOL, SAMPLING_RATIO, POOL, SAMPLING_RATIO, -1)
    return (sampled[:, :, 0, :, 0] + sampled[:, :, 0, :, 1] + sampled[:, :, 1, :, 0] + sampled[:, :, 1, :, 1]) * 0.25


def tiled_regions(features: Sequence[np.ndarray], boxes: np.ndarray) -> np.ndarray:
    """MultiScaleRoIAlign of the boxes on P2-P5, written into the box head's (200, 320, 256) grid."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    levels = (np.clip(np.floor(4 + np.log2(np.sqrt(area) / 224 + np.float32(1e-6))), 2, 5) - 2).astype(np.int64)
    grid = np.zeros((GRID_ROWS, GRID_COLS, PITCH, PITCH, CHANNELS), np.float32)
    cells = grid.reshape(REGIONS, PITCH, PITCH, CHANNELS)
    jobs = []
    for level, feature in enumerate(features):
        indices = np.where(levels == level)[0]
        scale = 2.0 ** round(math.log2(feature.shape[0] / INPUT_SIZE))
        for chunk in np.array_split(indices, max(1, math.ceil(len(indices) / 64))):
            if len(chunk):
                jobs.append((chunk, _POOL.submit(level_roi_align, feature, boxes[chunk], scale)))
    for chunk, job in jobs:
        cells[chunk, :POOL, :POOL] = job.result()
    return grid.transpose(0, 2, 1, 3, 4).reshape(GRID_ROWS * PITCH, GRID_COLS * PITCH, CHANNELS)


def region_rows(output: np.ndarray) -> np.ndarray:
    if output.shape[1:3] == (GRID_ROWS, GRID_COLS):
        return output.reshape(REGIONS, -1)
    return output.reshape(output.shape[1], REGIONS).T


def detections(class_logits: np.ndarray, box_regression: np.ndarray, boxes: np.ndarray, window: Window) -> np.ndarray:
    """(N, 6) rows of x1, y1, x2, y2 in the 720p frame's pixels, score and COCO category ID."""
    logits = class_logits - class_logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores = scores[:, 1:] / scores.sum(axis=1, keepdims=True)
    region, column = np.nonzero(scores > SCORE_THRESHOLD)
    labels = column + 1
    deltas = box_regression.reshape(-1, CLASSES, 4)[region, labels]
    decoded = clip_boxes(decode_boxes(deltas, boxes[region], BOX_WEIGHTS), window)
    scores = scores[region, column]
    keep = np.where((decoded[:, 2] - decoded[:, 0] >= BOX_MIN_SIZE) & (decoded[:, 3] - decoded[:, 1] >= BOX_MIN_SIZE))[
        0
    ]
    decoded, scores, labels = decoded[keep], scores[keep], labels[keep]
    keep = nms(decoded, scores, labels, NMS_THRESHOLD)[:DETECTIONS]
    frame = (decoded[keep] - np.array([window.left, window.top] * 2, np.float32)) / np.float32(window.scale)
    return np.concatenate([frame, scores[keep, None], labels[keep, None].astype(np.float32)], axis=1)


def box_head_input(outputs: Sequence[pyneat.Tensor], window: Window) -> tuple[pyneat.Tensor, np.ndarray]:
    """The backbone's outputs -> the box head's tiled regions, as a tensor, and the proposals they hold."""
    features, rpn = unpack_backbone(outputs[0].to_numpy(copy=False))
    boxes = proposals(rpn, window)
    tiles = tiled_regions(features, boxes)
    return pyneat.Tensor.from_numpy(tiles, copy=True, memory=pyneat.TensorMemory.EV74), boxes


def decode_outputs(outputs: Sequence[pyneat.Tensor], boxes: np.ndarray, window: Window) -> np.ndarray:
    class_logits, box_regression = (region_rows(output.to_numpy(copy=False)) for output in outputs)
    return detections(class_logits[: len(boxes)], box_regression[: len(boxes)], boxes, window)


def percentile_ms(seconds: list[float], percentile: float) -> float:
    return float(np.percentile(seconds, percentile) * 1000)


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for one MPK, in the shape Core saves them. Their throughput is not the
    model's, so Core's default run serves."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    report = pyneat.Model(str(mpk)).benchmark(options)
    layer_stats, stage_stats = report.mla_layer_stats, report.mla_stage_stats
    layers = [
        {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
        for layer in layer_stats.layers
    ]
    stages = [{"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stage_stats.stages]
    return {"status": layer_stats.status, "layers": layers}, {"status": stage_stats.status, "stages": stages}


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the backbone reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def build_runners(
    models: dict[str, pyneat.Model], image: pyneat.Tensor, window: Window
) -> tuple[pyneat.ModelRunner, pyneat.ModelRunner]:
    """A runner for each model. The box head's is built from one real backbone output."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    backbone = models["backbone"].build([image], run_options=run_options)
    tiles, _ = box_head_input(backbone.run([image], timeout_ms=TIMEOUT_MS), window)
    box_head = models["box_head"].build([tiles], run_options=run_options)
    return backbone, box_head


def infer_one(
    backbone: pyneat.ModelRunner, box_head: pyneat.ModelRunner, image: pyneat.Tensor, window: Window
) -> np.ndarray:
    tiles, boxes = box_head_input(backbone.run([image], timeout_ms=TIMEOUT_MS), window)
    return decode_outputs(box_head.run([tiles], timeout_ms=TIMEOUT_MS), boxes, window)


def measure_latency(
    backbone: pyneat.ModelRunner, box_head: pyneat.ModelRunner, image: pyneat.Tensor, window: Window
) -> list[float]:
    """Seconds per frame through every stage with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(backbone, box_head, image, window)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(backbone, box_head, image, window)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames decoded, how long they took, and the board
    telemetry sampled over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"the {name} rejected a frame")


def drain(runner: pyneat.ModelRunner, name: str, pushed: dict[str, int]) -> Iterator[list[pyneat.Tensor]]:
    """Each output until the runner's end of stream. Fewer outputs than frames pushed means a stall.

    Core's pull returns no tensors both at end of stream and on a timeout, so the count tells them apart.
    """
    pulled = 0
    while tensors := runner.pull(TIMEOUT_MS).tensors:
        pulled += 1
        yield tensors
        # Return the outputs to the runner before waiting for the next frame.
        del tensors
    if pulled != pushed[name]:
        raise RuntimeError(f"the {name} stalled: {pulled} of {pushed[name]} frames came out within {TIMEOUT_MS} ms")


def run_together(*stages: Callable[[], None], runners: tuple[pyneat.ModelRunner, ...]) -> None:
    """Run each stage on its own thread. The first failure ends every runner's input, which unblocks a stage
    waiting to push, and is raised once all stages stop. Each stage waits in Core with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            for runner in runners:
                runner.close_input()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def measure_throughput(
    backbone: pyneat.ModelRunner, box_head: pyneat.ModelRunner, image: pyneat.Tensor, window: Window
) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"backbone": 0, "box head": 0}
    # The proposals of each frame in the box head, in order.
    in_flight: list[np.ndarray] = []

    def push_inputs() -> None:
        try:
            while not measured.is_set():
                push(backbone, [image], "backbone")
                pushed["backbone"] += 1
        finally:
            # End of stream follows the last frame through the backbone, then the box head.
            backbone.close_input()

    def forward_regions() -> None:
        try:
            for outputs in drain(backbone, "backbone", pushed):
                tiles, boxes = box_head_input(outputs, window)
                # The tiles are a copy, so the backbone gets its output buffers back now.
                del outputs
                in_flight.append(boxes)
                push(box_head, [tiles], "box head")
                pushed["box head"] += 1
        finally:
            box_head.close_input()

    def decode_detections() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(box_head, "box head", pushed):
            decode_outputs(tensors, in_flight.pop(0), window)
            del tensors
            outputs += 1
            if outputs == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_FRAMES and not measured.is_set():
                frames, seconds = outputs - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(frames, seconds, monitor.stop())
                    measured.set()
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    try:
        run_together(push_inputs, forward_regions, decode_detections, runners=(backbone, box_head))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        backbone.close()
        box_head.close()
    return throughput


def write_report(
    output: Path,
    image: str,
    latencies: list[float],
    throughput: Throughput,
    models: dict[str, pyneat.Model],
    mla_layer_stats: dict[str, dict[str, object]],
    mla_stage_stats: dict[str, dict[str, object]],
) -> None:
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = percentile_ms(latencies, 50)
    benchmark.latency_p95_ms = percentile_ms(latencies, 95)
    benchmark.latency_p99_ms = percentile_ms(latencies, 99)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"{image} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing letterboxed to "
            f"{INPUT_SIZE}x{INPUT_SIZE}, backbone, {POST_NMS_TOP_N} proposals and RoIAlign on the CPU, box head, "
            f"and box decoding and NMS on the CPU, pipelined {QUEUE_DEPTH} frames deep"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.1f} FPS over {throughput.frames} frames in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms:.2f} ms over {len(latencies)} frames"
    )


def benchmark_pipeline(mpks: dict[str, Path], dataset_id: str, dataset: Path, output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        "backbone": pyneat.Model(str(mpks["backbone"]), image_options()),
        "box_head": pyneat.Model(str(mpks["box_head"]), tensor_options()),
    }
    path = sample_image(dataset)
    image = frame_tensor(path)
    window = letterbox_window()
    backbone, box_head = build_runners(models, image, window)
    latencies = measure_latency(backbone, box_head, image, window)
    throughput = measure_throughput(backbone, box_head, image, window)
    write_report(output, f"{dataset_id} {path.name}", latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"backbone", "box_head"}:
        raise SystemExit("--mpk must name exactly backbone=<path> and box_head=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_pipeline(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), Path(args.output))


if __name__ == "__main__":
    main()
