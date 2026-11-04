"""Benchmark DiT-XL/2 256 with Neat Core and save its report.

Given one --mpk <component>=<path> per component (transformer_0 to transformer_3, vae_decoder), the script measures
the whole model generating one image: a seeded latent, 50 DDIM steps that each run the four transformer parts for
the class and again for the null class, with classifier-free guidance and the DDIM update on the CPU, then the VAE
decoder. One inference is one generated 256x256 image, so throughput is in images per second. Each call depends on
the one before, so images run one at a time. The script also records each MPK's MLA layer and stage stats and its
execution plan.

The input is fixed: class 207 (golden retriever), seed 0, guidance scale 4, as the model's README describes. The
model takes no image, so it has no accuracy dataset to draw one from.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import model_report
import numpy as np
import pyneat
from dit import COMPONENTS, LATENT_CHANNELS, LATENT_SIZE, TRANSFORMER, class_one_hot, guided_noise, timestep_features

STEPS = 50
GUIDANCE_SCALE = 4.0
CLASS_LABEL = 207
SEED = 0
# DiT's training schedule: 1000 linear betas from 1e-4 to 0.02.
TRAIN_TIMESTEPS, BETA_START, BETA_END = 1000, 1e-4, 0.02
WARMUP_IMAGES = 1
# Each image takes about 15 s, so a pass is bounded by its duration rather than its count.
MEASURED_IMAGES = 4
MIN_DURATION_S = 60
TIMEOUT_MS = 60000


def tensor_options() -> pyneat.ModelOptions:
    """Every component's: the script prepares their inputs; Neat still quantizes and tessellates them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def hwc(value: np.ndarray) -> pyneat.Tensor:
    """An NCHW batch of one as the HWC tensor Neat takes."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(value[0].transpose(1, 2, 0)), copy=True, layout=pyneat.TensorLayout.HWC
    )


def handoff(tensor: pyneat.Tensor) -> pyneat.Tensor:
    """A transformer part's output as the next part's input: a fresh CPU [H, W, C] tensor holding a copy. An output
    passed on as is has media caps the next model's input stage rejects. The layout is stated: Neat reads an
    unlabelled 3D array as CHW."""
    array = tensor.to_numpy(copy=True)
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(array[0] if array.ndim == 4 else array), copy=True, layout=pyneat.TensorLayout.HWC
    )


def nchw(tensor: pyneat.Tensor) -> np.ndarray:
    array = tensor.to_numpy(copy=True)
    array = array if array.ndim == 4 else array[None]
    return np.ascontiguousarray(array.transpose(0, 3, 1, 2)).astype(np.float32)


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


@dataclass(frozen=True)
class Schedule:
    """The DDIM timesteps, from 999 to 0, and the cumulative alphas of DiT's training schedule."""

    timesteps: np.ndarray
    alphas: np.ndarray

    @classmethod
    def ddim(cls, steps: int) -> Schedule:
        betas = np.linspace(BETA_START, BETA_END, TRAIN_TIMESTEPS, dtype=np.float64)
        timesteps = np.linspace(TRAIN_TIMESTEPS - 1, 0, steps).round().astype(int)
        return cls(timesteps, np.cumprod(1 - betas))

    def step(self, index: int, latent: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """The deterministic DDIM update from timesteps[index] to the next timestep, or to the clean latent."""
        alpha = self.alphas[self.timesteps[index]]
        following = self.alphas[self.timesteps[index + 1]] if index + 1 < len(self.timesteps) else 1.0
        clean = (latent - np.sqrt(1 - alpha) * noise) / np.sqrt(alpha)
        return (np.sqrt(following) * clean + np.sqrt(1 - following) * noise).astype(np.float32)


@dataclass(frozen=True)
class Conditioning:
    """Each step's timestep features and the class one-hot, as Neat tensors for both guidance slots."""

    timesteps: list[tuple[pyneat.Tensor, pyneat.Tensor]]
    classes: tuple[pyneat.Tensor, pyneat.Tensor]

    @classmethod
    def of(cls, schedule: Schedule, label: int) -> Conditioning:
        one_hot = class_one_hot(label)
        steps = []
        for timestep in schedule.timesteps:
            features = timestep_features(float(timestep))
            steps.append((hwc(features[0:1]), hwc(features[1:2])))
        return cls(steps, (hwc(one_hot[0:1]), hwc(one_hot[1:2])))


def initial_latent() -> np.ndarray:
    shape = (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    return np.random.RandomState(SEED).standard_normal(shape).astype(np.float32)


def build_runners(models: dict[str, pyneat.Model], conditioning: Conditioning) -> dict[str, pyneat.ModelRunner]:
    """A runner for each component, each transformer part built from one real output of the part before."""
    runners: dict[str, pyneat.ModelRunner] = {}
    latent = hwc(initial_latent())
    features, one_hot = conditioning.timesteps[0][0], conditioning.classes[0]
    value = latent
    for part in TRANSFORMER:
        inputs = [value, features, one_hot]
        runners[part] = models[part].build(inputs)
        value = handoff(runners[part].run(inputs, timeout_ms=TIMEOUT_MS)[0])
    runners["vae_decoder"] = models["vae_decoder"].build([latent])
    return runners


def generate(runners: dict[str, pyneat.ModelRunner], schedule: Schedule, conditioning: Conditioning) -> np.ndarray:
    """One image: every DDIM step's two guidance passes through the transformer parts, then the decoder. Returns the
    decoder's [1, 3, 256, 256] output in [-1, 1]."""
    latent = initial_latent()
    for index in range(len(schedule.timesteps)):
        predictions = []
        for slot in range(2):
            value = hwc(latent)
            for part in TRANSFORMER[:-1]:
                value = handoff(runners[part].run(
                    [value, conditioning.timesteps[index][slot], conditioning.classes[slot]], timeout_ms=TIMEOUT_MS
                )[0])
            last = runners[TRANSFORMER[-1]].run(
                [value, conditioning.timesteps[index][slot], conditioning.classes[slot]], timeout_ms=TIMEOUT_MS
            )
            predictions.append(nchw(last[0]))
        latent = schedule.step(index, latent, guided_noise(np.concatenate(predictions), GUIDANCE_SCALE))
    # The decoder divides by the VAE's latent scale itself.
    return nchw(runners["vae_decoder"].run([hwc(latent)], timeout_ms=TIMEOUT_MS)[0])


def timed_images(
    runners: dict[str, pyneat.ModelRunner], schedule: Schedule, conditioning: Conditioning
) -> tuple[list[float], np.ndarray]:
    """Seconds per image over at least MEASURED_IMAGES images and MIN_DURATION_S seconds, and the last image."""
    seconds: list[float] = []
    image = np.empty(0)
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_IMAGES or time.perf_counter() < ends:
        started = time.perf_counter()
        image = generate(runners, schedule, conditioning)
        seconds.append(time.perf_counter() - started)
    return seconds, image


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the images generated, how long they took, and the board telemetry
    sampled over the same span."""

    images: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.images / self.seconds


def measure_throughput(
    runners: dict[str, pyneat.ModelRunner], schedule: Schedule, conditioning: Conditioning
) -> Throughput:
    """Images per second and board telemetry, generating images back to back."""
    monitor = pyneat.TelemetryMonitor()
    monitor.start()
    try:
        started = time.perf_counter()
        seconds, _ = timed_images(runners, schedule, conditioning)
        elapsed = time.perf_counter() - started
        return Throughput(len(seconds), elapsed, monitor.stop())
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()


def check_image(image: np.ndarray) -> None:
    """A generated image is finite and spans the decoder's range, so a broken build does not pass as fast."""
    if image.shape != (1, 3, 256, 256) or not np.isfinite(image).all() or float(image.std()) < 0.05:
        raise RuntimeError(f"the decoder produced no image: shape {image.shape}, std {float(image.std()):.4f}")


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
    benchmark.throughput_samples = throughput.images
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"One 256x256 image per inference: class {CLASS_LABEL}, seed {SEED}, {STEPS} DDIM steps, each the four "
            f"transformer parts for the class and the null class with guidance {GUIDANCE_SCALE:g} and the DDIM update "
            "on the CPU, then the VAE decoder; images one at a time"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.4f} images/s over {throughput.images} images in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms / 1000:.2f} s over {len(latencies)} images"
    )


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {name: pyneat.Model(str(mpks[name]), tensor_options()) for name in COMPONENTS}
    schedule = Schedule.ddim(STEPS)
    conditioning = Conditioning.of(schedule, CLASS_LABEL)
    runners = build_runners(models, conditioning)
    try:
        for _ in range(WARMUP_IMAGES):
            check_image(generate(runners, schedule, conditioning))
        latencies, image = timed_images(runners, schedule, conditioning)
        check_image(image)
        throughput = measure_throughput(runners, schedule, conditioning)
    finally:
        for runner in runners.values():
            runner.close()
    write_report(output, latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly {', '.join(f'{name}=<path>' for name in COMPONENTS)}")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", help="unused: the model takes no image, so its input is fixed")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    benchmark_pipeline(parse_mpks(args.mpk), Path(args.output))


if __name__ == "__main__":
    main()
