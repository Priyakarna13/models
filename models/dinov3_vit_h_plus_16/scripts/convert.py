"""Export DINOv3 ViT-H+/16 from the pinned timm checkpoint and check ONNX against the timm model.

The export is timm's model as it is, at 1x3x224x224. Its outputs are the final-norm class token (`class_embedding`)
and the 196 patch tokens (`patch_embeddings`), without the 4 register tokens, as Meta's DINOv3 returns them. The 840
million parameters take 3.4 GB in float32, more than one ONNX file holds, so model.onnx keeps its weights in one
external data file, model.onnx.data. Each weight keeps its timm name, which surgery.py reads to rebuild the compiler's
components.
"""

import argparse
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from calibration import preprocess_image
from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from PIL import Image
from pipeline import HIDDEN_SIZE, IMAGE_SIZE, OUTPUTS, PATCHES, PREFIX

MODEL_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURE = "vit_huge_plus_patch16_dinov3"
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "LICENSE.md"])
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "class_embedding": [1, HIDDEN_SIZE],
    "patch_embeddings": [1, PATCHES, HIDDEN_SIZE],
}
INPUTS = ("pixel_values",)
DATA_FILE = "model.onnx.data"
SEEDS = range(123, 127)
# The export is the timm model itself, so ONNX Runtime matches it to float32 rounding.
MIN_COSINE = 0.99999


def embedding_model(reference):
    """The timm model returning the class and patch embeddings. It holds timm's model as `model`, so each exported
    weight is named model.<timm name>."""
    import torch

    class Embeddings(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            tokens = self.model.forward_features(pixel_values)
            return tokens[:, 0], tokens[:, PREFIX:]

    return Embeddings(reference).eval()


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_outputs(names):
    if names != ["model.onnx"]:
        raise ValueError(f"The conversion writes model.onnx and its {DATA_FILE}; model.yaml declares {names}")


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def validate_export(path):
    """An opset-17 ONNX with the declared float32 interface, every weight in the one external file beside it."""
    model = onnx.load(str(path), load_external_data=False)
    locations = {
        field.value for tensor in model.graph.initializer for field in tensor.external_data if field.key == "location"
    }
    if locations != {DATA_FILE}:
        raise ValueError(f"Every weight must be in {DATA_FILE}, not {sorted(locations) or 'the ONNX file'}")
    # The checker reads external data by path.
    onnx.checker.check_model(str(path))
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The export requires opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected input {INPUTS} and outputs {OUTPUTS}")
    for value in values:
        tensor = value.type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]
        ):
            raise ValueError(f"Unexpected dtype or shape for {value.name}")


def synthetic(seed):
    """A random image through the model's own preprocessing, as an NCHW float32 array."""
    pixels = np.random.RandomState(seed).randint(0, 256, (320, 288, 3), dtype=np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    return preprocess_image(encoded.getvalue()).transpose(2, 0, 1)[None].copy()


def convert(source, output_dir):
    import gc

    import onnxruntime as ort
    import timm
    import torch

    validate_outputs(conversion_outputs(MODEL_DIR))
    config = json.loads((source.parent / "config.json").read_text())
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Expected {ARCHITECTURE}, config.json declares {config.get('architecture')}")
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    reference = timm.create_model(
        ARCHITECTURE, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": str(source)}
    ).eval()
    samples = [torch.from_numpy(synthetic(seed)) for seed in SEEDS]
    with torch.inference_mode():
        expected = []
        for sample in samples:
            tokens = reference.forward_features(sample)
            expected.append((tokens[:, 0].numpy(), tokens[:, PREFIX:].numpy()))

    # Beyond 2 GB the exporter writes each weight to its own file; they are gathered into one data file.
    with tempfile.TemporaryDirectory(prefix="dinov3_vit_h_plus_16-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        exported = Path(temporary) / "model.onnx"
        with torch.inference_mode():
            # Without constant folding every weight stays an initializer under its own name.
            torch.onnx.export(
                embedding_model(reference),
                (samples[0],),
                str(exported),
                input_names=list(INPUTS),
                output_names=list(OUTPUTS),
                opset_version=17,
                dynamo=False,
                do_constant_folding=False,
            )
        del reference
        gc.collect()
        model = onnx.load(str(exported))
        onnx.save_model(
            model, str(output), save_as_external_data=True, all_tensors_to_one_file=True, location=DATA_FILE
        )
        del model
        gc.collect()
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed, sample, wanted in zip(SEEDS, samples, expected):
        exported_outputs = session.run(list(OUTPUTS), {"pixel_values": sample.numpy()})
        if not all(np.isfinite(value).all() for value in exported_outputs):
            raise ValueError(f"Non-finite ONNX output for seed {seed}")
        source_cosine = min(cosine(w, e) for w, e in zip(wanted, exported_outputs))
        print(f"seed {seed}: ONNX vs timm cosine {source_cosine:.8f}")
        if source_cosine < MIN_COSINE:
            raise ValueError(f"Seed {seed}: ONNX vs timm cosine {source_cosine:.8f} is below {MIN_COSINE}")
    print(f"Synthetic-input PyTorch/ONNX check passed: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
