"""Export DINOv3 ConvNeXt-T for the MLA and check ONNX against the timm model.

The network is exported as timm defines it, except that each GELU is written as x * sigmoid(x * (a + b * x^2)),
the tanh approximation, so it compiles to native BF16 operations instead of an INT8 lookup table. The outputs
are those of Meta's DINOv3 ConvNeXt: the final 7x7 feature map averaged over its positions and layer-normalized
(`class_embedding`), and the same layer norm applied at each of its 49 positions (`patch_embeddings`).
"""

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import onnx
from calibration import preprocess_image
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURE = "convnext_tiny"
IMAGE_SIZE = 224
HIDDEN_SIZE = 768
GRID = IMAGE_SIZE // 32
PATCHES = GRID * GRID
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "class_embedding": [1, HIDDEN_SIZE],
    "patch_embeddings": [1, PATCHES, HIDDEN_SIZE],
}
INPUTS = ("pixel_values",)
OUTPUTS = ("class_embedding", "patch_embeddings")
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "LICENSE.md"])
SEEDS = range(123, 127)
# The rewrite matches its ONNX export almost exactly; the tanh GELU stays within 1e-4 of exact GELU.
ONNX_MIN_COSINE = 0.99999
SOURCE_MIN_COSINE = 0.9999


def rewritten_model(reference):
    """The timm ConvNeXt with tanh-form GELUs, returning the class and patch embeddings."""
    import torch
    from timm.layers.activations import GELU as TimmGELU

    class TanhGelu(torch.nn.Module):
        def forward(self, x):
            a = 2.0 * math.sqrt(2.0 / math.pi)
            return x * torch.sigmoid(x * (a + a * 0.044715 * x * x))

    class Embeddings(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # timm's head norm is the final LayerNorm, over the channels of a [1, C, H, W] map.
            self.norm = model.head.norm

        def forward(self, pixel_values):
            features = self.model.forward_features(pixel_values)
            class_embedding = self.norm(features.mean((2, 3), keepdim=True)).reshape(1, HIDDEN_SIZE)
            patches = self.norm(features).reshape(1, HIDDEN_SIZE, PATCHES).transpose(1, 2)
            return class_embedding, patches

    replaced = 0
    for parent in list(reference.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, (torch.nn.GELU, TimmGELU)):
                setattr(parent, name, TanhGelu())
                replaced += 1
    if replaced == 0:
        raise ValueError("The model has no GELU to rewrite")
    return Embeddings(reference).eval()


def reference_embeddings(model, pixel_values):
    """The outputs as transformers' DINOv3ConvNextModel defines them: the averaged map as a leading token before
    the 49 positions, all through one LayerNorm."""
    import torch
    import torch.nn.functional as F

    features = model.forward_features(pixel_values)
    tokens = torch.cat([features.mean((2, 3))[:, None], features.flatten(2).transpose(1, 2)], dim=1)
    norm = model.head.norm
    tokens = F.layer_norm(tokens, (HIDDEN_SIZE,), norm.weight, norm.bias, norm.eps)
    return tokens[:, 0], tokens[:, 1:]


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def validate_export(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The export requires opset 17")
    if any(node.op_type == "Erf" for node in model.graph.node):
        raise ValueError("The export still holds an exact GELU")
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
    import onnxruntime as ort
    import timm
    import torch

    config = json.loads((source.parent / "config.json").read_text())
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Expected {ARCHITECTURE}, config.json declares {config.get('architecture')}")
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)

    def load():
        return timm.create_model(
            ARCHITECTURE, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": str(source)}
        ).eval()

    reference = load()
    rewritten = rewritten_model(load())

    samples = [torch.from_numpy(synthetic(seed)) for seed in SEEDS]
    with torch.inference_mode():
        torch.onnx.export(
            rewritten,
            (samples[0],),
            str(output),
            input_names=list(INPUTS),
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed, sample in zip(SEEDS, samples):
        with torch.inference_mode():
            expected = reference_embeddings(reference, sample)
            pooled = reference(sample)
            candidate = rewritten(sample)
        # timm's own pooled embedding is the same class embedding.
        if cosine(pooled, expected[0]) < ONNX_MIN_COSINE:
            raise ValueError(f"Seed {seed}: timm's pooled embedding differs from the class embedding")
        exported = session.run(list(OUTPUTS), {"pixel_values": sample.numpy()})
        if not all(np.isfinite(value).all() for value in exported):
            raise ValueError(f"Non-finite ONNX output for seed {seed}")
        onnx_cosine = min(cosine(c, e) for c, e in zip(candidate, exported))
        source_cosine = min(cosine(r, e) for r, e in zip(expected, exported))
        print(f"seed {seed}: ONNX vs rewrite cosine {onnx_cosine:.8f}, ONNX vs timm cosine {source_cosine:.8f}")
        if onnx_cosine < ONNX_MIN_COSINE or source_cosine < SOURCE_MIN_COSINE:
            raise ValueError(f"Seed {seed} is below the export marks {ONNX_MIN_COSINE} / {SOURCE_MIN_COSINE}")
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
