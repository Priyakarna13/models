"""Measure the FP32 baseline: RF-DETR Seg Nano's ONNX on a host, scored exactly as accuracy.py scores the MPKs.

The baseline is the accuracy the package would reach without quantization, at the same 312x312 input and with
the same top-100 selection, mask resize and COCO scoring. model.yaml publishes it as the reference_metrics entry
with origin fp32_baseline. Run the model.onnx convert.py writes on a host with numpy, opencv-python-headless,
pillow, faster-coco-eval and onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/rf_detr_seg_nano/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset coco-2017-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path

import accuracy
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx convert.py exports")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of images, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val":
        raise SystemExit(f"--dataset must be coco-2017-val, got {dataset_id}")
    dataset = Path(directory)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = accuracy.COCO(str(dataset / accuracy.ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    records = ground_truth.loadImgs(image_ids)
    valid = np.isin(np.arange(accuracy.CLASSES), ground_truth.getCatIds())

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    def infer(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # accuracy.py passes the backbone a channels-last image and reads channels-last masks, as the MPKs use;
        # the ONNX takes and returns them channels-first.
        dets, labels, masks = session.run(None, {"input": np.ascontiguousarray(image.transpose(2, 0, 1)[None])})
        return dets, labels, masks.transpose(0, 2, 3, 1)

    detections = accuracy.evaluate(dataset, records, valid, infer)
    metrics = accuracy.score(ground_truth, detections, image_ids)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    args.output.write_text(json.dumps({"images": len(image_ids), "metrics": rounded}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
