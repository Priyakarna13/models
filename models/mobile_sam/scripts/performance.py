"""Benchmark MobileSAM with Neat Core and save its report.

Given one --mpk <component>=<path> per component (image_encoder_0, image_encoder_1, prompt_decoder), the script
measures the whole model on a fixed, seeded 720p frame: the two image encoder parts, then the prompt decoder with
one foreground click at the frame's centre, each component working on a different frame at the same time. The
frame is resized, normalized and padded on the CPU once, as segment_anything does, and Neat quantizes and
tessellates it inside the measured pipeline. The script also records each MPK's MLA layer and stage stats and its
execution plan.
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
from pipeline import COMPONENTS, preprocess, prompt_inputs

WARMUP_FRAMES = 20
MEASURED_FRAMES = 300
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
# The model declares no accuracy dataset, so the benchmark input is a fixed, seeded 720p camera-like frame:
# smooth colour gradients with sensor-like noise.
FRAME_WIDTH, FRAME_HEIGHT, SEED = 1280, 720, 0


def synthetic_frame() -> np.ndarray:
    rows = np.linspace(0, 1, FRAME_HEIGHT, dtype=np.float32)[:, None]
    columns = np.linspace(0, 1, FRAME_WIDTH, dtype=np.float32)[None, :]
    channels = [rows * 200 + 30, columns * 180 + 40, (1 - rows) * columns * 220 + 20]
    base = np.stack([np.broadcast_to(channel, (FRAME_HEIGHT, FRAME_WIDTH)) for channel in channels], axis=-1)
    noise = np.random.default_rng(SEED).normal(0, 8, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def tensor_options() -> pyneat.ModelOptions:
    """Both components': the script prepares their inputs; Neat still quantizes and tessellates them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def hwc(value: np.ndarray) -> pyneat.Tensor:
    """An NCHW batch of one as the HWC tensor Neat takes."""
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(value[0].transpose(1, 2, 0)), copy=True)


def handoff(tensors: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """One component's outputs as the next component's inputs: fresh CPU [H, W, C] tensors holding a copy, the form
    hwc() gives the first input. Neat Core builds a runner's input stage from the tensors it receives; an output
    passed on as is, or with Neat 0.6's leading batch axis, has media caps the next model's input stage rejects
    ([misconfig.media_caps])."""
    arrays = [tensor.to_numpy(copy=True) for tensor in tensors]
    return [
        pyneat.Tensor.from_numpy(np.ascontiguousarray(array[0] if array.ndim == 4 else array), copy=True)
        for array in arrays
    ]


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


def stage_inputs(component: str, tensors: list[pyneat.Tensor], prompt: list[pyneat.Tensor]) -> list[pyneat.Tensor]:
    """A component's inputs from the previous component's outputs; the decoder also takes the prompt."""
    return [*tensors, *prompt] if component == "prompt_decoder" else tensors


def build_runners(
    models: dict[str, pyneat.Model], image: pyneat.Tensor, prompt: list[pyneat.Tensor]
) -> list[pyneat.ModelRunner]:
    """A runner for each component, in COMPONENTS order, each built from one real output of the one before."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    runners, tensors = [], [image]
    for component in COMPONENTS:
        inputs = stage_inputs(component, tensors, prompt)
        try:
            runners.append(models[component].build(inputs, run_options=run_options))
        except pyneat.NeatError as error:
            shapes = [tuple(tensor.shape) for tensor in inputs]
            raise RuntimeError(f"building the {component} runner on input shapes {shapes} failed") from error
        tensors = handoff(runners[-1].run(inputs, timeout_ms=TIMEOUT_MS))
    return runners


def infer_one(runners: list[pyneat.ModelRunner], image: pyneat.Tensor, prompt: list[pyneat.Tensor]) -> None:
    tensors = [image]
    for component, runner in zip(COMPONENTS, runners):
        tensors = handoff(runner.run(stage_inputs(component, tensors, prompt), timeout_ms=TIMEOUT_MS))


def measure_latency(
    runners: list[pyneat.ModelRunner], image: pyneat.Tensor, prompt: list[pyneat.Tensor]
) -> list[float]:
    """Seconds per frame through every component with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(runners, image, prompt)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(runners, image, prompt)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the decoder output, how long they took, and the
    board telemetry sampled over the same span."""

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
    runners: list[pyneat.ModelRunner], image: pyneat.Tensor, prompt: list[pyneat.Tensor]
) -> Throughput:
    """Frames per second and board telemetry with every component busy, each on a different frame. The measured
    span starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = dict.fromkeys(COMPONENTS, 0)

    def push_images() -> None:
        try:
            while not measured.is_set():
                push(runners[0], [image], COMPONENTS[0])
                pushed[COMPONENTS[0]] += 1
        finally:
            # End of stream follows the last frame through every component.
            runners[0].close_input()

    def forward(index: int) -> Callable[[], None]:
        source, target = COMPONENTS[index], COMPONENTS[index + 1]

        def forward_outputs() -> None:
            try:
                for tensors in drain(runners[index], source, pushed):
                    push(runners[index + 1], stage_inputs(target, handoff(tensors), prompt), target)
                    pushed[target] += 1
                    # The next runner holds its inputs now, so this one gets its output buffers back.
                    del tensors
            finally:
                runners[index + 1].close_input()

        forward_outputs.__name__ = f"forward_{source}"
        return forward_outputs

    def pull_masks() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(runners[-1], COMPONENTS[-1], pushed):
            # Return the decoder's output buffers before the next pull.
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

    stages = [push_images, *(forward(index) for index in range(len(runners) - 1)), pull_masks]
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
            f"Seeded synthetic {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, resized, normalized and padded to 1024x1024 "
            "once on the CPU; the two image encoder parts, then the prompt decoder with one foreground click, "
            f"pipelined {QUEUE_DEPTH} frames deep"
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
    models = {name: pyneat.Model(str(mpks[name]), tensor_options()) for name in COMPONENTS}
    frame = synthetic_frame()
    image = hwc(preprocess(frame).transpose(2, 0, 1)[None])
    prompt_arrays = prompt_inputs([(FRAME_WIDTH / 2, FRAME_HEIGHT / 2)], ["foreground"], frame.shape[:2])
    prompt = [hwc(prompt_arrays["point_coords"]), hwc(prompt_arrays["point_types"])]
    runners = build_runners(models, image, prompt)
    latencies = measure_latency(runners, image, prompt)
    throughput = measure_throughput(runners, image, prompt)
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
