"""Benchmark RTMDet-Ins S with Neat Core and save its report.

Given --mpk <path> and --dataset coco-2017-val=<extracted directory>, the script measures the whole model as an
application runs it on one COCO image in a 720p frame: Neat's letterboxing, normalization and the MPK, then on
the CPU the 10 best detections above 0.3 after class-aware NMS, and their masks from the dynamic mask kernels,
resized to the 640x640 input and thresholded. The candidates above the threshold are read straight from the MPK's
output buffers, with OpenCV's reduction finding each cell's best class; only their kernels and the mask features are
copied, so the buffers return to the runner at once. NMS and the masks then run on a thread pool, so frames decode
in parallel on the A65 cores while the MPK runs the next ones. It also records the MPK's MLA layer and stage stats and
its execution plan.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

IMAGE_SIZE = 640
CLASSES = 80
STRIDES = (8, 16, 32)
LEVELS = [IMAGE_SIZE // stride for stride in STRIDES]
MASK_SIZE = LEVELS[0]
MASK_FEATURES = 8
KERNEL_SPLITS = [(MASK_FEATURES + 2) * MASK_FEATURES, MASK_FEATURES * MASK_FEATURES, MASK_FEATURES,
                 MASK_FEATURES, MASK_FEATURES, 1]
KERNEL_PARAMETERS = sum(KERNEL_SPLITS)
# An application keeps a few confident instances: the TOP_INSTANCES best above SCORE_THRESHOLD after NMS.
TOP_INSTANCES = 10
SCORE_THRESHOLD = 0.3
NMS_IOU = 0.6
# NumPy releases the GIL in its matrix products, so decoding threads run on separate A65 cores.
DECODE_THREADS = 8
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames the runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def _grid(size: int, stride: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1) * np.float32(stride)


PRIORS = [_grid(size, stride) for size, stride in zip(LEVELS, STRIDES, strict=True)]
MASK_GRID = _grid(MASK_SIZE, STRIDES[0])
LOGIT_THRESHOLD = float(np.log(SCORE_THRESHOLD / (1 - SCORE_THRESHOLD)))


def image_options() -> pyneat.ModelOptions:
    """Neat letterboxes, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    # Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # The model takes RGB normalized with ImageNet's mean and std.
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Class-aware NMS, best first: each class's boxes are offset apart so different classes never overlap."""
    boxes = boxes + labels[:, None] * (boxes.max() + 1)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size and len(keep) < TOP_INSTANCES:
        best, rest = order[0], order[1:]
        keep.append(best)
        inter = np.maximum(0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])) * np.maximum(
            0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])
        )
        order = rest[inter / np.maximum(areas[best] + areas[rest] - inter, 1e-9) <= NMS_IOU]
    return np.asarray(keep, dtype=np.int64)


@dataclass(frozen=True)
class Candidates:
    """One frame's detections above the score threshold, and the mask features, copied out of the MPK's buffers."""

    boxes: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    kernels: np.ndarray
    priors: np.ndarray
    mask_features: np.ndarray


def select_candidates(outputs: list[np.ndarray]) -> Candidates:
    """Read the candidates above the threshold from the MPK's channels-last outputs, which may be its buffers."""
    boxes, logits, labels, kernels, priors = [], [], [], [], []
    for level, stride in enumerate(STRIDES):
        scores = outputs[level].reshape(-1, CLASSES)
        # A logit above the threshold's logit is a score above it. OpenCV finds each cell's best class with NEON, so
        # only cells with a candidate are scanned class by class.
        cells = np.flatnonzero(cv2.reduce(scores, 1, cv2.REDUCE_MAX).ravel() > LOGIT_THRESHOLD)
        rows, label = np.nonzero(scores[cells] > LOGIT_THRESHOLD)
        index = cells[rows]
        points = PRIORS[level][index]
        distances = outputs[3 + level].reshape(-1, 4)[index]
        boxes.append(np.concatenate([points - distances[:, :2], points + distances[:, 2:]], axis=1).clip(0, IMAGE_SIZE))
        logits.append(scores[index, label])
        labels.append(label)
        kernels.append(outputs[6 + level].reshape(-1, KERNEL_PARAMETERS)[index])
        priors.append(np.concatenate([points, np.full((len(index), 1), stride, np.float32)], axis=1))
    features = np.ascontiguousarray(outputs[9].reshape(-1, MASK_FEATURES).T)
    return Candidates(*(np.concatenate(values) for values in (boxes, logits, labels, kernels, priors)), features)


def finish(candidates: Candidates) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The confident detections' boxes in the 640x640 input, scores, labels, and binary 640x640 masks."""
    if not len(candidates.boxes):
        empty = np.zeros(0, np.float32)
        return candidates.boxes, empty, candidates.labels, np.zeros((0, IMAGE_SIZE, IMAGE_SIZE), bool)
    keep = nms(candidates.boxes, candidates.logits, candidates.labels)
    count, priors = len(keep), candidates.priors[keep]
    relative = (priors[:, None, :2] - MASK_GRID[None]) / (priors[:, None, 2:] * 8)
    features = np.broadcast_to(candidates.mask_features, (count, *candidates.mask_features.shape))
    x = np.concatenate([relative.transpose(0, 2, 1), features], axis=1)
    w0, w1, w2, b0, b1, b2 = np.split(candidates.kernels[keep], np.cumsum(KERNEL_SPLITS)[:-1], axis=1)
    x = np.maximum(np.matmul(w0.reshape(count, MASK_FEATURES, -1), x) + b0[:, :, None], 0)
    x = np.maximum(np.matmul(w1.reshape(count, MASK_FEATURES, -1), x) + b1[:, :, None], 0)
    x = (np.matmul(w2.reshape(count, 1, -1), x) + b2[:, :, None]).reshape(count, MASK_SIZE, MASK_SIZE)
    masks = cv2.resize(np.ascontiguousarray(x.transpose(1, 2, 0)), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    masks = masks.reshape(IMAGE_SIZE, IMAGE_SIZE, count).transpose(2, 0, 1) > 0
    logits = candidates.logits[keep]
    return candidates.boxes[keep], 1 / (1 + np.exp(-logits)), candidates.labels[keep], masks


def decode(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one frame's candidates and finish its detections and masks."""
    return finish(select_candidates(outputs))


def percentile_ms(seconds: list[float], percentile: float) -> float:
    return float(np.percentile(seconds, percentile) * 1000)


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats, in the shape Core saves them. Their throughput is not the model's, so
    Core's default run serves."""
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


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image letterboxed into a 1280x720 RGB frame, placed once in the memory the MPK reads."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74)


def candidates_of(tensors: list[pyneat.Tensor]) -> Candidates:
    """The frame's candidates, read from the MPK's output buffers without copying the outputs."""
    return select_candidates([tensor.to_numpy(copy=False) for tensor in tensors])


def measure_latency(runner: pyneat.ModelRunner, image: pyneat.Tensor) -> list[float]:
    """Seconds per frame through every stage with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        finish(candidates_of(runner.run([image], timeout_ms=TIMEOUT_MS)))
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        finish(candidates_of(runner.run([image], timeout_ms=TIMEOUT_MS)))
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span: the frames decoded, how long they took, and board telemetry over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def measure_throughput(runner: pyneat.ModelRunner, image: pyneat.Tensor) -> Throughput:
    """Decoded frames per second and board telemetry with the MPK and the decoding threads all busy. The measured
    span starts after WARMUP_FRAMES decoded frames and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    errors: list[BaseException] = []
    pushed = 0
    throughput: Throughput | None = None

    def push_inputs() -> None:
        nonlocal pushed
        try:
            while not measured.is_set():
                if not runner.push([image]):
                    raise RuntimeError("the runner rejected a frame")
                pushed += 1
        except BaseException as error:
            errors.append(error)
        finally:
            runner.close_input()

    def pull_and_decode() -> None:
        nonlocal throughput
        decoded, started = 0, 0.0
        pending: deque = deque()

        def finish_one() -> None:
            nonlocal decoded, started, throughput
            pending.popleft().result()
            decoded += 1
            if decoded == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif decoded > WARMUP_FRAMES and not measured.is_set():
                frames, seconds = decoded - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(frames, seconds, monitor.stop())
                    measured.set()

        try:
            with ThreadPoolExecutor(DECODE_THREADS) as pool:
                pulled = 0
                while tensors := runner.pull(TIMEOUT_MS).tensors:
                    pulled += 1
                    # The candidates are copies, so the runner gets its buffers back before the pool finishes the frame.
                    pending.append(pool.submit(finish, candidates_of(tensors)))
                    del tensors
                    while len(pending) > 2 * DECODE_THREADS:
                        finish_one()
                while pending:
                    finish_one()
            if pulled != pushed:
                raise RuntimeError(f"the runner stalled: {pulled} of {pushed} frames came out within {TIMEOUT_MS} ms")
        except BaseException as error:
            errors.append(error)
            runner.close_input()

    threads = [threading.Thread(target=push_inputs), threading.Thread(target=pull_and_decode)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        runner.close()
    if errors:
        raise errors[0]
    if throughput is None:
        raise RuntimeError("the pipeline stopped before the measured span ended")
    return throughput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    mpk = Path(args.mpk)
    mla_layer_stats, mla_stage_stats = mla_stats(mpk)
    model = pyneat.Model(str(mpk), image_options())
    path = sample_image(Path(dataset_dir))
    image = frame_tensor(path)
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runner = model.build([image], run_options=run_options)
    latencies = measure_latency(runner, image)
    throughput = measure_throughput(runner, image)

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
        {"model": model},
        workload=(
            f"{dataset_id} {path.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing letterboxed "
            f"to {IMAGE_SIZE}x{IMAGE_SIZE}, then on the CPU the top {TOP_INSTANCES} detections above "
            f"{SCORE_THRESHOLD} after NMS and their {IMAGE_SIZE}x{IMAGE_SIZE} masks, on {DECODE_THREADS} threads, "
            f"pipelined {QUEUE_DEPTH} frames deep"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats={"model": mla_layer_stats},
        mla_stage_stats={"model": mla_stage_stats},
    )
    report.save(Path(args.output))
    print(
        f"{benchmark.fps:.1f} FPS over {throughput.frames} frames in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms:.2f} ms over {len(latencies)} frames"
    )


if __name__ == "__main__":
    main()
