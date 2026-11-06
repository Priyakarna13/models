"""Export DINOv3 ViT-S+/16 in an MLA-friendly layout and check ONNX against the timm model.

The export keeps every activation as [1, C, 1, T] (channels on C, tokens on W): linear layers are 1x1
convolutions, LayerNorm runs over channels, and attention uses two Einsums with the keys on the channel
axis, all heads in one pair. Exact folds move the attention scale into the query weights, LayerScale into
the following convolution, each pre-norm affine into the next convolution, and RoPE's rotate-half into
extra query and key channels. GELU is written as x * sigmoid(x * (a + b * x^2)), the tanh approximation,
so it compiles to native BF16 operations. The class and register tokens and the 14x14 RoPE sines and
cosines are constants for 224x224. The outputs are the final-norm class token (`class_embedding`) and the 196
patch tokens (`patch_embeddings`), without the 4 register tokens, as Meta's DINOv3 returns them.
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
ARCHITECTURE = "vit_small_plus_patch16_dinov3"
IMAGE_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 384
PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
REGISTERS = 4
PREFIX = 1 + REGISTERS
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
    """The MLA-friendly equivalent of a loaded timm DINOv3 ViT, as a PyTorch module."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv1x1(weight, bias):
        layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        layer.weight.data = weight.detach().float().reshape(*weight.shape, 1, 1).clone()
        layer.bias.data = bias.detach().float().clone()
        return layer

    def fold_affine(weight, bias, gamma, beta):
        """conv(gamma * x + beta) == conv'(x)."""
        return weight * gamma[None, :], bias + weight @ beta

    def rotate_half(dim):
        """R with rotate_half(x) == R @ x along channels: [-x2, x1]."""
        half = dim // 2
        matrix = torch.zeros(dim, dim)
        matrix[:half, half:] = -torch.eye(half)
        matrix[half:, :half] = torch.eye(half)
        return matrix

    class ChannelNorm(nn.Module):
        """LayerNorm over the channel axis of a [1, C, 1, T] tensor."""

        def __init__(self, dim, eps, weight=None, bias=None):
            super().__init__()
            self.dim, self.eps = dim, eps
            self.register_buffer("weight", torch.ones(dim) if weight is None else weight.detach().float().clone())
            self.register_buffer("bias", torch.zeros(dim) if bias is None else bias.detach().float().clone())

        def forward(self, x):
            y = F.layer_norm(x.permute(0, 2, 3, 1), (self.dim,), self.weight, self.bias, self.eps)
            return y.permute(0, 3, 1, 2)

    def gelu_tanh(x):
        a = 2.0 * math.sqrt(2.0 / math.pi)
        return x * torch.sigmoid(x * (a + a * 0.044715 * x * x))

    class Block(nn.Module):
        def __init__(self, block, dim, eps, prefix, sin, cos):
            super().__init__()
            attention, mlp = block.attn, block.mlp
            if attention.qkv.bias is not None or getattr(attention, "q_bias", None) is not None:
                raise ValueError("The rewrite expects bias-free query, key and value projections")
            if not all(isinstance(m, nn.Identity) for m in (attention.q_norm, attention.k_norm, attention.norm)):
                raise ValueError("The rewrite expects no query, key or attention-output norms")
            self.dim, self.heads, self.head_dim = dim, attention.num_heads, attention.head_dim
            g1, b1 = block.norm1.weight.detach(), block.norm1.bias.detach()
            wq, wk, wv = attention.qkv.weight.detach().chunk(3)
            zero = torch.zeros(dim)
            wq, bq = fold_affine(wq * attention.scale, zero, g1, b1)
            wk, bk = fold_affine(wk, zero, g1, b1)
            wv, bv = fold_affine(wv, zero, g1, b1)
            # q * cos + rotate_half(q) * sin, with rotate_half folded into extra output channels.
            rotation = torch.block_diag(*[rotate_half(self.head_dim)] * self.heads)
            self.qkv = conv1x1(
                torch.cat([wq, rotation @ wq, wk, rotation @ wk, wv]),
                torch.cat([bq, rotation @ bq, bk, rotation @ bk, bv]),
            )
            tokens = prefix + sin.shape[1]
            cosine, sine = torch.ones(1, dim, 1, tokens), torch.zeros(1, dim, 1, tokens)
            cosine[..., prefix:] = cos.repeat(self.heads, 1)[None, :, None]
            sine[..., prefix:] = sin.repeat(self.heads, 1)[None, :, None]
            self.register_buffer("cosine", cosine)
            self.register_buffer("sine", sine)
            self.norm1, self.norm2 = ChannelNorm(dim, eps), ChannelNorm(dim, eps)
            ls1, ls2 = block.gamma_1.detach(), block.gamma_2.detach()
            self.proj = conv1x1(attention.proj.weight.detach() * ls1[:, None], attention.proj.bias.detach() * ls1)
            g2, b2 = block.norm2.weight.detach(), block.norm2.bias.detach()
            self.swiglu = hasattr(mlp, "fc1_g")
            fc1 = mlp.fc1_x if self.swiglu else mlp.fc1
            self.fc1 = conv1x1(*fold_affine(fc1.weight.detach(), fc1.bias.detach(), g2, b2))
            if self.swiglu:
                self.gate = conv1x1(*fold_affine(mlp.fc1_g.weight.detach(), mlp.fc1_g.bias.detach(), g2, b2))
            self.fc2 = conv1x1(mlp.fc2.weight.detach() * ls2[:, None], mlp.fc2.bias.detach() * ls2)

        def attend(self, x):
            q, q_rotated, k, k_rotated, v = self.qkv(x).split(self.dim, dim=1)
            q = q * self.cosine + q_rotated * self.sine
            k = k * self.cosine + k_rotated * self.sine
            # Heads move to H: [1, head_dim, heads, T]; one Einsum pair and one Softmax cover all heads.
            q, k, v = (torch.cat(torch.split(t, self.head_dim, dim=1), dim=2) for t in (q, k, v))
            scores = torch.einsum("nchw,nchq->nqhw", q, k)
            weights = torch.softmax(scores, dim=1)
            out = torch.einsum("nchw,nqhc->nqhw", weights, v)
            return torch.cat(torch.split(out, 1, dim=2), dim=1)

        def forward(self, x):
            x = x + self.proj(self.attend(self.norm1(x)))
            z = self.norm2(x)
            if self.swiglu:
                gate = self.gate(z)
                hidden = gate * torch.sigmoid(gate) * self.fc1(z)
            else:
                hidden = gelu_tanh(self.fc1(z))
            return x + self.fc2(hidden)

    class Dinov3Embedding(nn.Module):
        """Image [1, 3, 224, 224] -> final-norm class-token embedding [1, C, 1, 1]."""

        def __init__(self, model):
            super().__init__()
            dim, patch = model.embed_dim, model.patch_embed.patch_size[0]
            grid = IMAGE_SIZE // patch
            self.patch = nn.Conv2d(3, dim, patch, patch)
            self.patch.weight.data = model.patch_embed.proj.weight.detach().clone()
            self.patch.bias.data = model.patch_embed.proj.bias.detach().clone()
            prefix = torch.cat([model.cls_token, model.reg_token], dim=1).detach()
            self.register_buffer("prefix", prefix.permute(0, 2, 1).reshape(1, dim, 1, -1).clone())
            with torch.no_grad():
                sin, cos = model.rope.get_embed(shape=[grid, grid]).tensor_split(2, -1)
            eps = model.norm.eps
            self.blocks = nn.ModuleList(
                Block(block, dim, eps, prefix.shape[1], sin.T.contiguous(), cos.T.contiguous())
                for block in model.blocks
            )
            self.norm = ChannelNorm(dim, eps, model.norm.weight, model.norm.bias)
            self.dim = dim

        def forward(self, pixel_values):
            x = self.patch(pixel_values).reshape(1, self.dim, 1, -1)
            x = torch.cat([self.prefix, x], dim=3)
            for block in self.blocks:
                x = block(x)
            class_embedding = self.norm(x[..., :1]).reshape(1, self.dim)
            patches = self.norm(x[..., PREFIX:]).reshape(1, self.dim, PATCHES).transpose(1, 2)
            return class_embedding, patches

    return Dinov3Embedding(reference).eval()


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
    expected = INTERFACE
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(expected):
        raise ValueError(f"Expected input {INPUTS} and outputs {OUTPUTS}")
    for value in values:
        tensor = value.type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in tensor.shape.dim] != expected[value.name]
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
    reference = timm.create_model(
        ARCHITECTURE, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": str(source)}
    ).eval()
    rewritten = rewritten_model(reference)

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
            tokens = reference.forward_features(sample)
            expected = (tokens[:, 0], tokens[:, PREFIX:])
            candidate = rewritten(sample)
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
