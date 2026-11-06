"""Measure FAN 2dfan2_alt's NME on the AFW test faces with its MPK on a Modalix board.

Each AFW test image is a face crop, so its bounds are the face box. The CPU crops the face as the model takes it
(landmarks.py), the MPK returns 68 heatmaps, and the CPU decodes them to landmark coordinates. NME is each face's mean
point error over its inter-ocular distance (the outer eye corners, points 36 and 45), averaged over the faces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from landmarks import HEATMAP_SIZE, LANDMARKS, TEST_SPLIT, crop, decode, nme, test_faces, whole_image_box


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already cropped and scaled the face; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def heatmaps_of(tensor: pyneat.Tensor) -> np.ndarray:
    """The MPK's [64, 64, 68] (or batched) heatmaps as [68, 64, 64]."""
    array = tensor.to_numpy(copy=True)
    array = array[0] if array.ndim == 4 else array
    if array.shape != (HEATMAP_SIZE, HEATMAP_SIZE, LANDMARKS) or not np.isfinite(array).all():
        raise ValueError(f"unexpected heatmaps: shape {array.shape}")
    return array.transpose(2, 0, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="afw-faces-default=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)

    faces = test_faces(Path(dataset_dir))
    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    errors = []
    started = time.monotonic()
    for name, image, truth in faces:
        patch, box = crop(image, whole_image_box(image))
        tensor = pyneat.Tensor.from_numpy(patch, copy=True, layout=pyneat.TensorLayout.HWC)
        if runner is None:
            runner = model.build([tensor])
        landmarks, _ = decode(heatmaps_of(runner.run([tensor], timeout_ms=120000)[0]), box)
        errors.append(nme(landmarks, truth))
    print(f"{len(faces)} faces in {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {"nme": float(np.mean(errors))},
            "split": TEST_SPLIT, "evaluated_samples": len(faces), "available_samples": len(faces),
            "details": {"normalization": "inter-ocular (outer eye corners)", "median_nme": float(np.median(errors))},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 5) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
