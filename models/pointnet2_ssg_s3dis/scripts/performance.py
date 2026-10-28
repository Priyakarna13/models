"""Benchmark PointNet++ SSG with Neat Core and save its report.

Given one --mpk <component>=<path> per component, in model.yaml order, the script measures the whole model on one
4096-point block: farthest point sampling, ball queries and 3-NN weights on the CPU (the block's plan), then the
eight MPKs with the CPU grouping and interpolation between them. Latency runs one block at a time. The plan costs
far more CPU time than the MPKs take, so the throughput pass computes the plans of upcoming blocks in
PLAN_WORKERS processes while the main process runs the grouping and the MPKs, as an application streaming blocks
would. It also records each MPK's MLA layer and stage stats. With --dataset s3dis-val=<extracted directory> the
input is the first 1 m tile block of one random.Random(0) room, built exactly as accuracy.py builds it; without
it, pointcloud.synthetic_block(), a fixed seeded block. Either way every run measures the same points.

    PYTHONPATH=tools python performance.py --mpk sa1=<archive> ... --mpk fp1=<archive> \
        [--dataset s3dis-val=<extracted directory>] --output <directory>
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import random
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from accuracy import DATASET, SEED, blocks
from pointcloud import COMPONENTS, POINTS, channels_last, plan, segment, synthetic_block

WARMUP_BLOCKS = 10
MEASURED_BLOCKS = 200
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Processes computing block plans in the throughput pass.
PLAN_WORKERS = 4


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: its input is grouped or interpolated point features, not an image."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for one MPK. Their throughput is not the model's, so Core's default run
    serves."""
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


def pipeline(models: dict[str, pyneat.Model]):
    """run(component, input) for pointcloud.segment: each MPK takes its input as one [H, W, C] tensor."""
    runners = {}

    def run(component: str, value: np.ndarray) -> np.ndarray:
        tensor = pyneat.Tensor.from_numpy(channels_last(value), copy=False)
        if component not in runners:
            runners[component] = models[component].build([tensor])
        (output,) = runners[component].run([tensor], timeout_ms=TIMEOUT_MS)
        return output.to_numpy(copy=True)

    return run, runners


def measure_latency(run, block: np.ndarray) -> list[float]:
    """Seconds per block, each block's plan and MPKs run in turn with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_BLOCKS):
        segment(block, run)
    seconds: list[float] = []
    started = time.perf_counter()
    while len(seconds) < MEASURED_BLOCKS or time.perf_counter() - started < MIN_DURATION_S:
        begun = time.perf_counter()
        logits = segment(block, run)
        seconds.append(time.perf_counter() - begun)
    if not np.isfinite(logits).all():
        raise RuntimeError("the pipeline produced non-finite logits")
    return seconds


def measure_throughput(run, block: np.ndarray) -> tuple[int, float, pyneat.Telemetry]:
    """Blocks, seconds and board telemetry with the plans computed ahead in PLAN_WORKERS processes. The measured
    span starts after WARMUP_BLOCKS blocks and lasts at least MEASURED_BLOCKS blocks and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    xyz = np.ascontiguousarray(block[:, :3])
    blocks, started = 0, 0.0
    # Spawned workers import only NumPy and pointcloud, not the Neat runtime this process holds.
    with multiprocessing.get_context("spawn").Pool(PLAN_WORKERS) as pool:
        for indices in pool.imap(plan, itertools.repeat(xyz), chunksize=1):
            segment(block, run, indices)
            blocks += 1
            if blocks == WARMUP_BLOCKS:
                monitor.start()
                started = time.perf_counter()
            elif blocks > WARMUP_BLOCKS:
                measured, seconds = blocks - WARMUP_BLOCKS, time.perf_counter() - started
                if measured >= MEASURED_BLOCKS and seconds >= MIN_DURATION_S:
                    pool.terminate()
                    return measured, seconds, monitor.stop()
    raise RuntimeError("the plan workers stopped before the measured span ended")


def dataset_block(directory: Path) -> tuple[str, np.ndarray]:
    """The first tile block of one random.Random(0) room of s3dis val, as accuracy.py's tile protocol builds it."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    room = random.Random(0).choice(manifest["rooms"])
    points = {key: np.load(directory / "val" / room / f"{key}.npy", allow_pickle=False) for key in ("xyz", "rgb")}
    features, _ = next(blocks(points, np.random.RandomState(SEED), "tiles"))
    return f"{DATASET} {room}, its first 1 m tile block", features


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or list(mpks) != list(COMPONENTS):
        raise SystemExit(f"--mpk must name {', '.join(COMPONENTS)} in that order")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    mpks = parse_mpks(args.mpk)
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpk), tensor_options()) for name, mpk in mpks.items()}
    if args.dataset:
        dataset_id, directory = args.dataset.split("=", 1)
        if dataset_id != DATASET:
            raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
        source, block = dataset_block(Path(directory))
    else:
        source, block = f"a seeded synthetic {POINTS}-point block", synthetic_block()
    run, runners = pipeline(models)
    try:
        seconds = measure_latency(run, block)
        frames, span, telemetry = measure_throughput(run, block)
    finally:
        for runner in runners.values():
            runner.close()

    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(seconds) * 1000)
    benchmark.latency_p50_ms = float(np.percentile(seconds, 50) * 1000)
    benchmark.latency_p95_ms = float(np.percentile(seconds, 95) * 1000)
    benchmark.latency_p99_ms = float(np.percentile(seconds, 99) * 1000)
    benchmark.fps = frames / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"{source}: CPU sampling, ball query and 3-NN weights, then the eight "
            f"MPKs with CPU grouping and interpolation between them; throughput with {PLAN_WORKERS} processes "
            "computing upcoming blocks' sampling, ball queries and 3-NN"
        ),
        telemetry=telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(Path(args.output))
    print(f"{benchmark.fps:.2f} blocks/s over {frames} blocks; latency {benchmark.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
