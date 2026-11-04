"""Export the pinned Microsoft ResNet-50 checkpoint for the disposable registry canary.

Run from the repository root with scripts/run_model_conversion.sh models/resnet_50
default /tmp/resnet50-export. All checkpoint bytes come from the verified source
file set; the Transformers loader stays offline. Three seeded inputs compare the
export with PyTorch. This checks conversion fidelity, not ImageNet accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
INTERFACE = {"pixel_values": [1, 3, 224, 224], "logits": [1, 1000]}


def validate_export(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected pixel_values input and logits output")
    for value in values:
        tensor = value.type.tensor_type
        if (tensor.elem_type != onnx.TensorProto.FLOAT
                or [dim.dim_value for dim in tensor.shape.dim] != INTERFACE[value.name]):
            raise ValueError(f"Unexpected interface for {value.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", choices=["default"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_and_validate_model(MODEL_DIR)
    source = metadata["conversion"][args.conversion]["source"]
    checkpoint, _, _, _ = Fetcher().fetch_source(source)

    import onnxruntime as ort
    import torch
    from transformers import ResNetForImageClassification

    torch.set_num_threads(4)
    torch.manual_seed(0)
    network = ResNetForImageClassification.from_pretrained(
        checkpoint.parent, local_files_only=True, use_safetensors=True
    ).eval()

    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = network

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).logits

    wrapper = Logits().eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, torch.zeros(1, 3, 224, 224), str(output),
            input_names=["pixel_values"], output_names=["logits"],
            opset_version=12, dynamo=False,
        )
    validate_export(output)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    comparisons = []
    for seed in range(3):
        sample = np.random.default_rng(seed).standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.inference_mode():
            expected = wrapper(torch.from_numpy(sample)).numpy()
        actual = session.run(["logits"], {"pixel_values": sample})[0]
        if not np.isfinite(actual).all():
            raise ValueError("Export produced non-finite logits")
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        comparisons.append({"seed": seed, "max_absolute_error": float(np.max(np.abs(actual - expected)))})
    report = {"status": "passed", "rtol": 1e-3, "atol": 1e-4, "comparisons": comparisons}
    (args.output_dir / "conversion-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
