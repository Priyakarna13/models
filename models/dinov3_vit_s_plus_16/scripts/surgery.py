"""Rebuild the converted DINOv3 ViT-S+/16 for the Model Compiler from its weights, and check its embeddings.

The conversion is timm's model as it is. For the compiler, every activation becomes [1, C, 1, T] (channels on C,
tokens on W): linear layers are 1x1 convolutions, LayerNorm runs over channels, and attention uses two Einsums with
the keys on the channel axis, all heads in one pair. Exact folds move the attention scale into the query weights,
LayerScale into the following convolution, each pre-norm affine into the next convolution, and RoPE's rotate-half
into extra query and key channels. The class and register tokens and the 14x14 RoPE sines and cosines are constants
for 224x224. The weights come from the converted ONNX by their timm names, and ONNX Runtime then checks the rebuilt
graph against the converted one.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

INPUT = "pixel_values"
IMAGE_SIZE = 224
INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
OUTPUTS = ("class_embedding", "patch_embeddings")
HIDDEN_SIZE = 384
HEADS = 6
HEAD_DIM = HIDDEN_SIZE // HEADS
DEPTH = 12
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE
PATCHES = GRID * GRID
# The class token and the four register tokens come before the patch tokens.
REGISTERS = 4
PREFIX = 1 + REGISTERS
EPS = 1e-5
ROPE_TEMPERATURE = 100.0
# convert.py exports timm's model as the attribute `model` of its wrapper, so each weight keeps its timm name after it.
WEIGHT_PREFIX = "model."
OPSET = 17
SEEDS = range(123, 127)
# The rebuild is exact up to float32 rounding.
MIN_COSINE = 0.99999
BLOCK_WEIGHTS = (
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.proj.weight",
    "attn.proj.bias",
    "gamma_1",
    "norm2.weight",
    "norm2.bias",
    "mlp.fc1_g.weight",
    "mlp.fc1_g.bias",
    "mlp.fc1_x.weight",
    "mlp.fc1_x.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
    "gamma_2",
)
MODEL_WEIGHTS = (
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "cls_token",
    "reg_token",
    "norm.weight",
    "norm.bias",
)


def weight_names() -> set[str]:
    """Every weight the rebuild reads, by its timm name."""
    blocks = {f"blocks.{index}.{name}" for index in range(DEPTH) for name in BLOCK_WEIGHTS}
    return blocks | set(MODEL_WEIGHTS)


def load_weights(source: Path) -> dict[str, np.ndarray]:
    """The converted graph's weights by their timm names. It must hold exactly the weights the rebuild reads."""
    model = onnx.load(str(source))
    weights = {
        tensor.name[len(WEIGHT_PREFIX) :]: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
        if tensor.name.startswith(WEIGHT_PREFIX)
    }
    if set(weights) != weight_names():
        missing, unexpected = sorted(weight_names() - set(weights)), sorted(set(weights) - weight_names())
        raise ValueError(
            f"The converted graph's weights differ from DINOv3 ViT-S+/16's: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )
    if weights["blocks.0.attn.qkv.weight"].shape != (3 * HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError("The converted graph is not DINOv3 ViT-S+/16: its query, key and value weights differ")
    return weights


def rope_tables() -> tuple[np.ndarray, np.ndarray]:
    """DINOv3's RoPE sines and cosines for the 14x14 patch grid, [196, head_dim] each, as timm computes them in
    float32: 0.5-centered row and column coordinates over the grid mapped to [-1, 1], periods temperature^(2i/32), and
    the angles of both axes tiled twice for rotate-half."""
    quarter = HEAD_DIM // 4
    periods = (ROPE_TEMPERATURE ** (2.0 * np.arange(quarter, dtype=np.float32) / (HEAD_DIM // 2))).astype(np.float32)
    axis = (np.arange(0.5, GRID, dtype=np.float32) / np.float32(GRID)).astype(np.float32)
    rows, columns = np.meshgrid(axis, axis, indexing="ij")
    coords = (2.0 * np.stack([rows, columns], axis=-1).reshape(PATCHES, 2) - 1.0).astype(np.float32)
    angles = (np.float32(2 * math.pi) * coords[:, :, None] / periods[None, None, :]).reshape(PATCHES, 2 * quarter)
    angles = np.tile(angles, 2).astype(np.float32)
    return np.sin(angles), np.cos(angles)


def build_encoder(weights: dict[str, np.ndarray]):
    """The MLA-friendly DINOv3 ViT-S+/16 as a PyTorch module, from its timm-named weights."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def tensor(name):
        return torch.from_numpy(np.array(weights[name], np.float32))

    def conv1x1(weight, bias):
        layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        layer.weight.data = weight.float().reshape(*weight.shape, 1, 1).clone()
        layer.bias.data = bias.float().clone()
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

        def __init__(self, weight=None, bias=None):
            super().__init__()
            self.register_buffer("weight", torch.ones(HIDDEN_SIZE) if weight is None else weight.clone())
            self.register_buffer("bias", torch.zeros(HIDDEN_SIZE) if bias is None else bias.clone())

        def forward(self, x):
            y = F.layer_norm(x.permute(0, 2, 3, 1), (HIDDEN_SIZE,), self.weight, self.bias, EPS)
            return y.permute(0, 3, 1, 2)

    sin, cos = (torch.from_numpy(table.T.copy()) for table in rope_tables())

    class Block(nn.Module):
        def __init__(self, index):
            super().__init__()

            def get(name):
                return tensor(f"blocks.{index}.{name}")

            g1, b1 = get("norm1.weight"), get("norm1.bias")
            wq, wk, wv = get("attn.qkv.weight").chunk(3)
            zero = torch.zeros(HIDDEN_SIZE)
            wq, bq = fold_affine(wq * HEAD_DIM**-0.5, zero, g1, b1)
            wk, bk = fold_affine(wk, zero, g1, b1)
            wv, bv = fold_affine(wv, zero, g1, b1)
            # q * cos + rotate_half(q) * sin, with rotate_half folded into extra output channels.
            rotation = torch.block_diag(*[rotate_half(HEAD_DIM)] * HEADS)
            self.qkv = conv1x1(
                torch.cat([wq, rotation @ wq, wk, rotation @ wk, wv]),
                torch.cat([bq, rotation @ bq, bk, rotation @ bk, bv]),
            )
            tokens = PREFIX + PATCHES
            cosine, sine = torch.ones(1, HIDDEN_SIZE, 1, tokens), torch.zeros(1, HIDDEN_SIZE, 1, tokens)
            cosine[..., PREFIX:] = cos.repeat(HEADS, 1)[None, :, None]
            sine[..., PREFIX:] = sin.repeat(HEADS, 1)[None, :, None]
            self.register_buffer("cosine", cosine)
            self.register_buffer("sine", sine)
            self.norm1, self.norm2 = ChannelNorm(), ChannelNorm()
            ls1, ls2 = get("gamma_1"), get("gamma_2")
            self.proj = conv1x1(get("attn.proj.weight") * ls1[:, None], get("attn.proj.bias") * ls1)
            g2, b2 = get("norm2.weight"), get("norm2.bias")
            self.fc1 = conv1x1(*fold_affine(get("mlp.fc1_x.weight"), get("mlp.fc1_x.bias"), g2, b2))
            self.gate = conv1x1(*fold_affine(get("mlp.fc1_g.weight"), get("mlp.fc1_g.bias"), g2, b2))
            self.fc2 = conv1x1(get("mlp.fc2.weight") * ls2[:, None], get("mlp.fc2.bias") * ls2)

        def attend(self, x):
            q, q_rotated, k, k_rotated, v = self.qkv(x).split(HIDDEN_SIZE, dim=1)
            q = q * self.cosine + q_rotated * self.sine
            k = k * self.cosine + k_rotated * self.sine
            # Heads move to H: [1, head_dim, heads, T]; one Einsum pair and one Softmax cover all heads.
            q, k, v = (torch.cat(torch.split(t, HEAD_DIM, dim=1), dim=2) for t in (q, k, v))
            scores = torch.einsum("nchw,nchq->nqhw", q, k)
            attention = torch.softmax(scores, dim=1)
            out = torch.einsum("nchw,nqhc->nqhw", attention, v)
            return torch.cat(torch.split(out, 1, dim=2), dim=1)

        def forward(self, x):
            x = x + self.proj(self.attend(self.norm1(x)))
            z = self.norm2(x)
            gate = self.gate(z)
            # SwiGLU: SiLU(gate) times the value branch.
            return x + self.fc2(gate * torch.sigmoid(gate) * self.fc1(z))

    class Encoder(nn.Module):
        """Image [1, 3, 224, 224] -> class and patch embeddings."""

        def __init__(self):
            super().__init__()
            self.patch = nn.Conv2d(3, HIDDEN_SIZE, PATCH_SIZE, PATCH_SIZE)
            self.patch.weight.data = tensor("patch_embed.proj.weight").clone()
            self.patch.bias.data = tensor("patch_embed.proj.bias").clone()
            prefix = torch.cat([tensor("cls_token"), tensor("reg_token")], dim=1)
            self.register_buffer("prefix", prefix.permute(0, 2, 1).reshape(1, HIDDEN_SIZE, 1, PREFIX).clone())
            self.blocks = nn.ModuleList(Block(index) for index in range(DEPTH))
            self.norm = ChannelNorm(tensor("norm.weight"), tensor("norm.bias"))

        def forward(self, pixel_values):
            x = self.patch(pixel_values).reshape(1, HIDDEN_SIZE, 1, PATCHES)
            x = torch.cat([self.prefix, x], dim=3)
            for block in self.blocks:
                x = block(x)
            class_embedding = self.norm(x[..., :1]).reshape(1, HIDDEN_SIZE)
            patches = self.norm(x[..., PREFIX:]).reshape(1, HIDDEN_SIZE, PATCHES).transpose(1, 2)
            return class_embedding, patches

    return Encoder().eval()


def validate_prepared(path: Path) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"The prepared graph requires opset {OPSET}")
    if [value.name for value in model.graph.input] != [INPUT] or [value.name for value in model.graph.output] != list(
        OUTPUTS
    ):
        raise ValueError(f"Expected input {INPUT} and outputs {OUTPUTS} in that order")
    left = sorted({node.op_type for node in model.graph.node} & {"If", "Loop", "Shape", "Erf", "Sin", "Cos"})
    if left:
        raise ValueError(f"The prepared graph must be static with constant RoPE tables; it still has {', '.join(left)}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_equivalent(source: Path, prepared: Path) -> None:
    """Require the prepared graph's embeddings to match the converted graph's on seeded inputs."""
    import onnxruntime as ort

    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (source, prepared)]
    for seed in SEEDS:
        image = np.random.RandomState(seed).standard_normal(INPUT_SHAPE).astype(np.float32)
        expected, actual = (session.run(list(OUTPUTS), {INPUT: image}) for session in sessions)
        if not all(np.isfinite(value).all() for value in actual):
            raise ValueError(f"The prepared graph returns non-finite values for seed {seed}")
        worst = min(_cosine(e, a) for e, a in zip(expected, actual, strict=True))
        print(f"seed {seed}: prepared vs converted cosine {worst:.8f}", flush=True)
        if worst < MIN_COSINE:
            raise ValueError(f"Seed {seed}: the prepared graph's cosine {worst:.6f} is below {MIN_COSINE}")


def prepare_model(source: Path, output_dir: Path) -> Path:
    """Write the compiler's graph for the converted ONNX into output_dir and return its path."""
    import torch

    encoder = build_encoder(load_weights(source))
    prepared = output_dir / "prepared.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (torch.zeros(INPUT_SHAPE),),
            str(prepared),
            input_names=[INPUT],
            output_names=list(OUTPUTS),
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
        )
    validate_prepared(prepared)
    check_equivalent(source, prepared)
    print(f"Prepared {prepared}", flush=True)
    return prepared
