"""Measure complete ASR processing and live-style time to first visible text.

Usage: PYTHONPATH=tools python performance.py --mpk encoder_mpk.tar.gz \
    --resources extracted-package --audio speech.wav --output results

Use the same 30 s, 16 kHz mono PCM16 speech WAV for both models: concatenate
LibriSpeech test-clean 6930-75918-0000 through 0003 and retain the first 30 s.
Other recordings are accepted and identified by SHA-256 in the report. The unpaced
pass measures full-recording latency and RTF, including log-mel extraction,
encoder calls and INT8 CPU prediction/joining. Overlapping context counts once
in audio duration. A separate paced pass waits for each chunk's input, including
lookahead, before processing it. Time to first text starts at the first audio
sample and ends at the first non-whitespace decoded text (leading silence counts).
Loading, WAV decoding, warm-up and diagnostic profiling are outside both passes.

CI stage declaration and regression baselines await a pinned speech dataset and
real DevKit measurements; this script can already benchmark a local package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import wave
from pathlib import Path

import model_report
import numpy as np
from transcribe import Transcriber

SAMPLE_RATE = 16000
WARMUP_RECORDINGS = 2
MEASURED_RECORDINGS = 10
MIN_DURATION_S = 60
FIRST_TEXT_TRIALS = 5


def load_audio(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (
            SAMPLE_RATE, 1, 2, "NONE"
        ):
            raise ValueError("Expected a 16 kHz mono PCM16 WAV")
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if len(audio) < 30 * SAMPLE_RATE:
        raise ValueError("Use at least 30 seconds of real speech to exercise steady-state windows")
    return audio


def available_samples(kind, index, total):
    """Last raw sample required by transcribe.py's chunk, including FFT context."""
    if kind == "nemotron":
        needed = 7880 if index == 0 else (49 + 56 * (index - 1)) * 160 - 256 + 9360
    elif kind == "parakeet":
        needed = index * 32000 + 64000
    else:
        raise ValueError(f"Unknown ASR kind: {kind}")
    return min(total, needed)


def transcribe_recording(transcriber, audio):
    started = time.perf_counter()
    text = ""
    for text in transcriber.transcribe(audio, "auto"):
        pass
    elapsed = time.perf_counter() - started
    if not text.strip():
        raise RuntimeError("No visible text: use speech audio and check the runtime before benchmarking")
    return elapsed, text


def time_to_first_text(transcriber, audio):
    started = time.perf_counter()
    iterator = transcriber.transcribe(audio, "auto")
    try:
        index = 0
        while True:
            ready = available_samples(transcriber.config["kind"], index, len(audio)) / SAMPLE_RATE
            delay = started + ready - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            try:
                text = next(iterator)
            except StopIteration:
                raise RuntimeError("No visible text in the paced recording") from None
            if text.strip():
                return time.perf_counter() - started
            index += 1
    finally:
        iterator.close()


def encoder_stats(transcriber, audio):
    """Profile separately, using real tensors captured during an untimed warm-up."""
    import pyneat

    encode = transcriber.encode
    seeds = {}

    def capture(inputs):
        seeds.clear()
        seeds.update(inputs)
        return encode(inputs)

    transcriber.encode = capture
    try:
        transcribe_recording(transcriber, audio)
    finally:
        transcriber.encode = encode
    options = pyneat.BenchmarkOptions()
    options.inputs = [seeds[port["name"]] for port in transcriber.config["inputs"]]
    options.num_samples = 10
    options.min_duration_ms = 0
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    stats = transcriber.model.benchmark(options)
    layers, stages = stats.mla_layer_stats, stats.mla_stage_stats
    return (
        {"model": {"status": layers.status, "layers": [
            {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
            for layer in layers.layers
        ]}},
        {"model": {"status": stages.status, "stages": [
            {"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stages.stages
        ]}},
    )


def benchmark_pipeline(transcriber, audio, audio_path, output):
    import pyneat

    layer_stats, stage_stats = encoder_stats(transcriber, audio)
    for _ in range(WARMUP_RECORDINGS):
        transcribe_recording(transcriber, audio)
    monitor = pyneat.TelemetryMonitor()
    seconds = []
    monitor.start()
    started = time.perf_counter()
    try:
        while len(seconds) < MEASURED_RECORDINGS or time.perf_counter() - started < MIN_DURATION_S:
            elapsed, transcript = transcribe_recording(transcriber, audio)
            seconds.append(elapsed)
        span = time.perf_counter() - started
    finally:
        telemetry = monitor.stop()
    # Each call resets predictor/encoder history while keeping model weights and the Run loaded.
    first_text = [time_to_first_text(transcriber, audio) for _ in range(FIRST_TEXT_TRIALS)]
    duration = len(audio) / SAMPLE_RATE
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(seconds) * 1000)
    for percentile in (50, 95, 99):
        setattr(benchmark, f"latency_p{percentile}_ms", float(np.percentile(seconds, percentile) * 1000))
    benchmark.fps = len(seconds) / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = len(seconds)
    kind = transcriber.config["kind"]
    profile = "560 ms cached chunks" if kind == "nemotron" else "14 s windows (10 s past + 2 s new + 2 s lookahead)"
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    report = model_report.from_performance(
        benchmark, {"model": transcriber.model}, telemetry=telemetry,
        mla_layer_stats=layer_stats, mla_stage_stats=stage_stats,
        workload=(
            f"One {duration:.3f} s 16 kHz mono recording, SHA-256 {digest}; {profile}; auto language. "
            "Full log-mel frontend, BF16 encoder and INT8 CPU predictor/joiner, one recording at a time. "
            "Latency is unpaced full-recording transcription; throughput is recordings/s. "
            "RTF is unpaced processing seconds / unique audio seconds (overlap counted once). "
            "Time to first text is the mean of five paced replays from first audio sample to first visible "
            "text, including input buffering/lookahead and leading silence. Models stay loaded; "
            "WAV decoding, warm-up and separate profiling are excluded."
        ),
    )
    report.document["metrics"].update(
        real_time_factor=span / (len(seconds) * duration),
        time_to_first_text_ms=float(np.mean(first_text) * 1000),
    )
    report.evidence["asr-timings.json"] = {
        "audio_sha256": digest, "audio_samples": len(audio), "sample_rate": SAMPLE_RATE,
        "model_kind": kind, "language": "auto", "unpaced_seconds": seconds,
        "unpaced_span_seconds": span, "paced_first_text_seconds": first_text,
        "final_transcript": transcript,
    }
    report.save(output)
    print(json.dumps(report.document["metrics"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="encoder archive, optionally model=path")
    parser.add_argument("--resources", type=Path, required=True, help="extracted package or its model directory")
    parser.add_argument("--audio", type=Path, required=True, help="same speech WAV for both models")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mpk = Path(args.mpk.removeprefix("model=")).resolve(strict=True)
    package = args.resources.resolve(strict=True)
    if (package / "model/runtime_config.json").is_file():
        package = package / "model"
    audio = load_audio(args.audio)
    # Use the supplied archive with the package's CPU assets; never edit or repack the archive.
    with tempfile.TemporaryDirectory(prefix="asr-benchmark-") as temporary:
        staged = Path(temporary)
        for path in package.iterdir():
            if path.is_file() and not path.name.endswith(".tar.gz"):
                (staged / path.name).symlink_to(path)
        (staged / "encoder_mpk.tar.gz").symlink_to(mpk)
        with Transcriber(staged) as transcriber:
            benchmark_pipeline(transcriber, audio, args.audio, args.output)


if __name__ == "__main__":
    main()
