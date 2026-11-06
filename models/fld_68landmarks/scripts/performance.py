"""Benchmark one compiled FAN 2dfan2_alt MPK with Neat Core on an AFW face and save its report.

The benchmark input is one AFW test face, chosen with random.Random(0), letterboxed into a 1280x720 RGB frame. Neat
stretches the frame to the model's 256x256 input and scales it to [0, 1] inside the measured pipeline, as FAN takes
it (no mean or standard deviation). An application crops the face around a detector's box first (landmarks.py);
that CPU step and the heatmap decode are not timed here.
"""

import argparse
import random
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

SPLIT = "test_data"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def sample_face(dataset: Path) -> Path:
    """One annotated AFW test face, chosen with a fixed seed so every run measures the same input."""
    faces = sorted(path for path in (dataset / SPLIT).glob("*.jpg")
                   if not path.name.startswith(".") and path.with_suffix(".pts").is_file())
    if not faces:
        raise SystemExit(f"{dataset} has no annotated faces under {SPLIT}/")
    return random.Random(0).choice(faces)


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


def image_options() -> pyneat.ModelOptions:
    """Neat stretches the frame to 256x256 and scales it to [0, 1]; FAN takes no mean or std preset."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="afw-faces-default=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    face = sample_face(Path(dataset_dir))

    model = pyneat.Model(args.mpk, image_options())
    # Request 1000 samples and a 60 s minimum measurement duration.
    benchmark_options = pyneat.BenchmarkOptions()
    benchmark_options.num_samples = 1000
    benchmark_options.min_duration_ms = 60000
    benchmark_options.include_mla_layer_stats = True
    benchmark_options.include_mla_stage_stats = True
    frame = letterbox_to_720p(read_rgb(face))
    benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
    report = model_report.from_performance(
        model.benchmark(benchmark_options),
        {"model": model},
        workload=(
            f"{dataset_id} {SPLIT}/{face.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing "
            "stretched to 256x256 and scaled to 0-1"
        ),
    )
    report.save(args.output)


if __name__ == "__main__":
    main()
