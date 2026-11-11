"""Benchmark DINOv3 ViT-H+/16 with Neat Core and save its report.

Given one --mpk <component>=<path> per component, blocks_0 to blocks_3, the script measures the whole encoder on
a fixed 720p frame: Neat's preprocessing and blocks_0, then blocks_1 to blocks_3, each component working on a
different frame at the same time. The model declares no accuracy dataset, so the script builds its own input: a
seeded outdoor scene, a sky-to-field gradient with soft shapes and sensor noise, the same on every run. It also
records each MPK's MLA layer and stage stats.

    PYTHONPATH=tools python performance.py --mpk blocks_0=<archive> ... --mpk blocks_3=<archive> --output <directory>
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import model_report
import numpy as np
import pyneat

# As in pipeline.py; the model's tests keep the two the same.
IMAGE_SIZE = 224
COMPONENTS = ("blocks_0", "blocks_1", "blocks_2", "blocks_3")
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720


def scene() -> np.ndarray:
    """A fixed 1280x720 RGB frame: sky above a field, with seeded soft shapes and mild noise."""
    rng = np.random.default_rng(0)
    rows = np.linspace(0.0, 1.0, FRAME_HEIGHT, dtype=np.float32)[:, None, None]
    sky = np.asarray([120, 170, 230], np.float32) * (1.0 - 0.35 * rows)
    field = np.asarray([70, 130, 60], np.float32) * (1.25 - 0.5 * rows)
    frame = np.where(rows < 0.55, sky, field) * np.ones((1, FRAME_WIDTH, 1), np.float32)
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
    for _ in range(12):
        x, y = rng.uniform(0, FRAME_WIDTH), rng.uniform(0, FRAME_HEIGHT)
        radius, colour = rng.uniform(30, 140), rng.uniform(20, 235, 3).astype(np.float32)
        weight = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * radius**2))[..., None]
        frame = frame * (1 - weight) + colour * weight
    frame += rng.normal(0.0, 3.0, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def image_options() -> pyneat.ModelOptions:
    """blocks_0's: Neat stretches, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    # Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # Without a preset Neat only scales pixels to [0, 1]; the model expects ImageNet mean and std.
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def frame_tensor() -> pyneat.Tensor:
    """The fixed frame, placed once in the memory blocks_0 reads, reused for every frame."""
    return pyneat.Tensor.from_numpy(
        scene(), copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """One component's tokens as the next component's input. The next input stage cannot read the previous
    runner's output memory, so the tokens are copied with cvu()."""
    return [tensor.cvu() for tensor in tensors]


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


def build_runners(models: dict[str, pyneat.Model], image: pyneat.Tensor) -> list[pyneat.ModelRunner]:
    """A runner for each component, in pipeline order, each built from one real output of the one before."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runners, tensors = [], [image]
    for component in COMPONENTS:
        runners.append(models[component].build(tensors, run_options=run_options))
        tensors = handoff(runners[-1].run(tensors, timeout_ms=TIMEOUT_MS))
    return runners


def infer_one(runners: list[pyneat.ModelRunner], image: pyneat.Tensor) -> None:
    tensors = [image]
    for index, runner in enumerate(runners):
        outputs = runner.run(tensors, timeout_ms=TIMEOUT_MS)
        tensors = handoff(outputs) if index < len(runners) - 1 else outputs


def measure_latency(runners: list[pyneat.ModelRunner], image: pyneat.Tensor) -> list[float]:
    """Seconds per frame through every component with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(runners, image)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(runners, image)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the last component output, how long they took, and the
    board telemetry sampled over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"{name} rejected a frame")


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
        raise RuntimeError(f"{name} stalled: {pulled} of {pushed[name]} frames came out within {TIMEOUT_MS} ms")


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


def measure_throughput(runners: list[pyneat.ModelRunner], image: pyneat.Tensor) -> Throughput:
    """Frames per second and board telemetry with every component busy, each on a different frame. The measured
    span starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = dict.fromkeys(COMPONENTS, 0)

    def push_frames() -> None:
        try:
            while not measured.is_set():
                push(runners[0], [image], COMPONENTS[0])
                pushed[COMPONENTS[0]] += 1
        finally:
            # End of stream follows the last frame through every component.
            runners[0].close_input()

    def forward(index: int) -> Callable[[], None]:
        source, target = COMPONENTS[index], COMPONENTS[index + 1]

        def forward_tokens() -> None:
            try:
                for tensors in drain(runners[index], source, pushed):
                    push(runners[index + 1], handoff(tensors), target)
                    pushed[target] += 1
                    # The next runner holds its copy now, so this one gets its output buffers back.
                    del tensors
            finally:
                runners[index + 1].close_input()

        forward_tokens.__name__ = f"forward_{source}"
        return forward_tokens

    def pull_embeddings() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(runners[-1], COMPONENTS[-1], pushed):
            # Return the last component's output buffers before the next pull.
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

    stages = [push_frames, *(forward(index) for index in range(len(runners) - 1)), pull_embeddings]
    try:
        run_together(*stages, runners=tuple(runners))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        for runner in runners:
            runner.close()
    return throughput


def write_report(
    output: Path,
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
            f"a fixed {FRAME_WIDTH}x{FRAME_HEIGHT} RGB outdoor scene the script builds, Neat preprocessing stretched "
            f"to {IMAGE_SIZE}x{IMAGE_SIZE}, and the {len(COMPONENTS)} components pipelined {QUEUE_DEPTH} frames deep"
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


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        component: pyneat.Model(str(mpks[component]), image_options())
        if index == 0
        else pyneat.Model(str(mpks[component]))
        for index, component in enumerate(COMPONENTS)
    }
    image = frame_tensor()
    runners = build_runners(models, image)
    latencies = measure_latency(runners, image)
    throughput = measure_throughput(runners, image)
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(mpks[name]) for name in COMPONENTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
