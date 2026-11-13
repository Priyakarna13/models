"""Score the FP32 ONNX from surgery.export_model, before prepare_model replaces decoder functions.

Use requirements-accuracy.txt and onnxruntime (or onnxruntime-gpu). The model's accuracy.py supplies all
preprocessing, prompt tokenization, query selection, mask resizing and COCO scoring. Default: the same 100
lowest-ID images and all 80 prompts as DevKit accuracy. The registered reference was measured with FP32
PyTorch CUDA at this same grid and protocol; this runner evaluates the exported ONNX counterpart.

    PYTHONPATH=tools python models/efficientsam3_repvit_m/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --resources <package-root> --dataset coco-2017-val=<directory> \\
        --output fp32-baseline.json --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import accuracy
import numpy as np


def main(argv: list[str] | None = None) -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True, help="package root containing tokenizer/")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=accuracy.EVALUATED_IMAGES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val" or args.limit <= 0:
        parser.error("use coco-2017-val and a positive image limit")
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        parser.error(f"{provider} is unavailable in this ONNX Runtime installation")
    providers = [(provider, {"use_tf32": "0"})] if args.device == "cuda" else [provider]
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"ONNX Runtime did not activate {provider}")
    options = ort.RunOptions()
    options.only_execute_path_to_fetches = True
    tokenizer = accuracy.load_tokenizer(args.resources)
    dataset = Path(directory)
    with contextlib.redirect_stdout(io.StringIO()):
        truth = accuracy.COCO(str(dataset / accuracy.ANNOTATIONS))
    image_ids = sorted(truth.getImgIds())[:args.limit]
    categories = sorted(truth.getCatIds())
    names = [record["name"] for record in truth.loadCats(categories)]
    if len(names) != 80:
        raise ValueError("expected all 80 COCO categories")
    feeds = {
        "token_one_hot": np.zeros((1, accuracy.CONTEXT, accuracy.VOCABULARY), np.float32),
        "pixel_values": np.zeros((1, 3, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE), np.float32),
        "text": np.zeros((1, accuracy.CONTEXT + 1, accuracy.FEATURES), np.float32),
    }
    texts = []
    for name in names:
        tokens = accuracy.tokens(name, tokenizer)
        features, = session.run(["text_features"], {**feeds, "token_one_hot": accuracy.one_hot(tokens)}, options)
        texts.append(accuracy.packed(features, tokens))
    results: dict[str, list] = {"bbox": [], "segm": []}
    for count, record in enumerate(truth.loadImgs(image_ids), start=1):
        pixels = accuracy.preprocess_image(dataset / accuracy.IMAGES / record["file_name"])
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        kept = (np.zeros((0, 6), np.float32), np.zeros(0, np.int64),
                np.zeros((0, accuracy.MASK_SIZE, accuracy.MASK_SIZE), np.float32))
        for category, text in zip(categories, texts, strict=True):
            detections, masks = session.run(
                ["detections", "mask_logits"], {**feeds, "pixel_values": pixels, "text": text}, options
            )
            kept = accuracy.best_queries(kept, accuracy.detection_rows(detections), masks[0], category)
        boxes, masks = accuracy.coco_results(record["id"], record["width"], record["height"], kept)
        results["bbox"].extend(boxes)
        results["segm"].extend(masks)
        print(f"{count}/{len(image_ids)} images", flush=True)
    metrics = accuracy.coco_stats(truth, image_ids, results)
    args.output.write_text(json.dumps({"dataset": dataset_id, "images": len(image_ids), "metrics": metrics}, indent=2)
                           + "\n", encoding="utf-8")
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
