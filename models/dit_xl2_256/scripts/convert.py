#!/usr/bin/env python3
"""Export DiT-XL/2 256's transformer step and VAE decoder side by side in one ONNX and check it against PyTorch.

The source is the pinned diffusers checkpoint. Each transformer block embeds the timestep and the class itself;
the export feeds those embedders the CPU's timestep features and a one-hot class (dit.py), which equals
diffusers' own embedding. At float32 the two components' weights exceed ONNX's 2 GB single-file limit, so
weights are stored as float16 behind a Cast; the w16a16 build keeps fewer mantissa bits than float16 has. The
check compares ONNX Runtime with PyTorch running the same float16-rounded weights.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import onnx
from dit import (
    COMPONENTS,
    INPUT_SHAPES,
    LATENT_CHANNELS,
    OPSET,
    OUTPUT_SHAPES,
    VAE_SCALING,
    class_one_hot,
    tensor_name,
    timestep_features,
)
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from onnx import TensorProto, helper, numpy_helper

MODEL_DIR = Path(__file__).resolve().parents[1]
FILES = ("model_index.json", "transformer/config.json", "transformer/diffusion_pytorch_model.bin",
         "vae/config.json", "vae/diffusion_pytorch_model.bin")
# Initializers this large are stored as float16; smaller ones, such as norms' and shapes', stay as they are.
FLOAT16_MIN_ELEMENTS = 4096
MIN_COSINE = 0.9999


def validate_source(source: dict) -> None:
    files = {item["path"] for item in source.get("files", [])}
    if source.get("type") != "file_set" or source.get("entrypoint") != "model_index.json" or files != set(FILES):
        raise ValueError(f"the source must be the DiT pipeline's {', '.join(FILES)}, opened at model_index.json")


def validate_config(root: Path) -> None:
    transformer = json.loads((root / "transformer/config.json").read_text(encoding="utf-8"))
    expected = {"num_layers": 28, "num_attention_heads": 16, "attention_head_dim": 72, "patch_size": 2,
                "sample_size": 32, "in_channels": 4, "out_channels": 8, "norm_type": "ada_norm_zero",
                "num_embeds_ada_norm": 1000}
    actual = {key: transformer.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Unsupported DiT transformer config: {actual}")


def build(root: Path):
    """The diffusers transformer, its embedders fed the CPU's features, and the VAE, in float32 eval mode."""
    import torch
    from diffusers import AutoencoderKL, Transformer2DModel

    class FeatureEmbedding(torch.nn.Module):
        """CombinedTimestepLabelEmbeddings with the sinusoid on the CPU and the label lookup as a matrix product."""

        def __init__(self, embedding):
            super().__init__()
            self.timestep_embedder = embedding.timestep_embedder
            self.table = embedding.class_embedder.embedding_table

        def forward(self, features, one_hot, hidden_dtype=None):
            batch = features.shape[0]
            return self.timestep_embedder(features.reshape(batch, -1)) + one_hot.reshape(batch, -1) @ self.table.weight

    transformer = Transformer2DModel.from_pretrained(root / "transformer", use_safetensors=False).eval()
    original = transformer.transformer_blocks[0].norm1.emb
    for block in transformer.transformer_blocks:
        block.norm1.emb = FeatureEmbedding(block.norm1.emb)
    vae = AutoencoderKL.from_pretrained(root / "vae", use_safetensors=False).eval()

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer, self.vae = transformer, vae

        def forward(self, latent, features, one_hot, decoder_latent):
            prediction = self.transformer(latent, timestep=features, class_labels=one_hot).sample
            image = self.vae.decode(decoder_latent / VAE_SCALING).sample
            return prediction, image

    return SideBySide().eval(), transformer, original


def check_embedding(transformer, original) -> None:
    """The feature embedding equals diffusers' own, `original`, on its timestep and label inputs."""
    import torch

    with torch.inference_mode():
        expected = original(torch.tensor([500.0, 500.0]), torch.tensor([207, 1000]), hidden_dtype=torch.float32)
        actual = transformer.transformer_blocks[0].norm1.emb(
            torch.from_numpy(timestep_features(500.0)), torch.from_numpy(class_one_hot(207)))
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4, err_msg="embedding")


def store_float16(path: Path, output: Path) -> None:
    """Rewrite large float32 initializers as float16 consumed through a Cast, in one file. The export keeps its
    weights in external files; each is read from disk on its own, so only the float16 model is held at once."""
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    def internalize(tensor: onnx.TensorProto) -> None:
        if tensor.data_location != TensorProto.EXTERNAL:
            return
        info = {entry.key: entry.value for entry in tensor.external_data}
        with open(path.parent / info["location"], "rb") as stream:
            stream.seek(int(info.get("offset", 0)))
            raw = stream.read(int(info["length"])) if "length" in info else stream.read()
        del tensor.external_data[:]
        tensor.data_location = TensorProto.DEFAULT
        tensor.raw_data = raw

    # The export also stores some Constant nodes' values externally.
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                internalize(attribute.t)
    casts, kept = [], []
    for initializer in graph.initializer:
        internalize(initializer)
        if initializer.data_type != TensorProto.FLOAT or np.prod(initializer.dims) < FLOAT16_MIN_ELEMENTS:
            kept.append(initializer)
            continue
        stored = numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16),
                                         f"{initializer.name}_float16")
        initializer.ClearField("raw_data")
        kept.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [initializer.name], to=TensorProto.FLOAT,
                                      name=f"{initializer.name}_cast"))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    # The transformer's output comes out of the export without a static shape; declare it.
    declared = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    for value in graph.output:
        dims = value.type.tensor_type.shape.dim
        del dims[:]
        for size in declared[value.name]:
            dims.add().dim_value = size
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.save(model, str(output))


def round_to_float16(module) -> None:
    """Round the PyTorch weights as store_float16 does, so the check isolates the export."""
    import torch

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.numel() >= FLOAT16_MIN_ELEMENTS:
                parameter.copy_(parameter.half().float())


def validate_export(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    expected_inputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()}
    expected_outputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"Unexpected DiT export interface: inputs {inputs}, outputs {outputs}")
    if not any(opset.domain == "" and opset.version == OPSET for opset in model.opset_import):
        raise ValueError(f"the export must use opset {OPSET}")


def cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected output: shape {actual.shape}")
    expected, actual = expected.ravel().astype(np.float64), actual.ravel().astype(np.float64)
    return float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))


def sample_inputs() -> dict[str, np.ndarray]:
    """The class slot of one guidance step, and a decoder latent."""
    rng = np.random.RandomState(0)
    latent = rng.standard_normal(INPUT_SHAPES["transformer"]["latent"]).astype(np.float32)
    return {
        tensor_name("transformer", "latent"): latent,
        tensor_name("transformer", "timestep_features"): timestep_features(500.0)[:1],
        tensor_name("transformer", "class_one_hot"): class_one_hot(207)[:1],
        tensor_name("vae_decoder", "latent"): (latent * VAE_SCALING).astype(np.float32),
    }


def convert(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch

    root = source.parent
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(root)
    model, transformer, original = build(root)
    check_embedding(transformer, original)
    del original
    round_to_float16(model)
    inputs = sample_inputs()
    names = list(inputs)
    float32_dir = output_dir / "float32"
    float32_dir.mkdir()
    with torch.inference_mode():
        expected = [value.numpy() for value in model(*(torch.from_numpy(inputs[name]) for name in names))]
        torch.onnx.export(
            model, tuple(torch.from_numpy(inputs[name]) for name in names), str(float32_dir / "model.onnx"),
            input_names=names,
            output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
            opset_version=OPSET, dynamo=False,
        )
    del model, transformer
    gc.collect()
    store_float16(float32_dir / "model.onnx", output)
    for item in float32_dir.iterdir():
        item.unlink()
    float32_dir.rmdir()
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)
    scores = {tensor_name(c, n): cosine(want, got) for (c, n), want, got in zip(
        [(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]], expected, actual)}
    if min(scores.values()) < MIN_COSINE:
        raise ValueError(f"ONNX Runtime disagrees with PyTorch: cosine {scores}")
    noise = actual[0][:, :LATENT_CHANNELS]
    print(f"PyTorch/ONNX check passed (cosine {scores}; noise std {noise.std():.3f}): {output}")
    return output


def main() -> None:
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
