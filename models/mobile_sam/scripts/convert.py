"""Export MobileSAM's two image encoder parts and its prompt decoder side by side in one ONNX and check it against
PyTorch.

MobileSAM is segment_anything's prompt encoder and mask decoder with a TinyViT image encoder. The checkpoint is
MobileSAM's mobile_sam.pt; the encoder is built from the vendored tiny_vit_sam.py and the rest from the
segment_anything wheel.

The exported encoder is tiny_vit_4d.ImageEncoder: upstream's TinyViT with its attention stages in the 4D layout
the compiler handles, as two components (pipeline.py), since the whole encoder needs more memory to compile than a
CI runner has. Chained, the two parts are checked against upstream's TinyViT before export.

The prompt decoder takes each prompt point as normalized coordinates and a one-hot point type (pipeline.py) and
embeds it itself, so an application needs no weights on the CPU. Its embedding equals segment_anything's.
"""

import argparse
import math
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from pipeline import (
    COMPONENTS,
    EMBEDDING_SHAPE,
    IMAGE_SIZE,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
    POINTS,
    prompt_inputs,
    resized_size,
    tensor_name,
)

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "mobile_sam.pt"
OPSET = 17
SEEDS = range(123, 128)
# Prompts the checks feed the decoder, in pixels of a 1024x768 image: (points, types).
PROMPTS = [
    ([(512, 384)], ["foreground"]),
    ([(300, 200), (700, 500)], ["foreground", "background"]),
    ([(100, 80), (900, 700)], ["box_top_left", "box_bottom_right"]),
    (
        [(100, 80), (900, 700), (500, 400), (120, 650), (850, 90)],
        ["box_top_left", "box_bottom_right", "foreground", "background", "background"],
    ),
]
ORIGINAL_SIZE = (768, 1024)


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def build(checkpoint):
    """MobileSAM as MobileSAM's build_sam_vit_t builds it, with the checkpoint loaded strictly."""
    import torch
    from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
    from tiny_vit_sam import TinyViT

    encoder = TinyViT(
        img_size=IMAGE_SIZE,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    embedding = EMBEDDING_SHAPE[1]
    sam = Sam(
        image_encoder=encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=embedding,
            image_embedding_size=EMBEDDING_SHAPE[2:],
            input_image_size=(IMAGE_SIZE, IMAGE_SIZE),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=embedding, mlp_dim=2048, num_heads=8),
            transformer_dim=embedding,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return sam.eval()


def modules(sam):
    """The side-by-side export module and the prompt decoder."""
    import torch

    class PromptDecoder(torch.nn.Module):
        """segment_anything's point embedding and mask prediction, without a mask prompt."""

        def __init__(self):
            super().__init__()
            prompt = sam.prompt_encoder
            self.decoder = sam.mask_decoder
            self.register_buffer("gaussian", prompt.pe_layer.positional_encoding_gaussian_matrix.clone())
            # One row per point type, in POINT_TYPES order.
            table = torch.cat([prompt.not_a_point_embed.weight, *(item.weight for item in prompt.point_embeddings)])
            self.register_buffer("table", table.clone())
            dense = prompt.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, *EMBEDDING_SHAPE[2:])
            self.register_buffer("dense", dense.clone())
            self.register_buffer("image_pe", prompt.get_dense_pe().clone())

        def forward(self, image_embeddings, point_coords, point_types):
            coords = point_coords[0, :, 0, :].transpose(0, 1)
            types = point_types[0, :, 0, :].transpose(0, 1)
            projected = 2 * math.pi * (coords @ self.gaussian)
            # A padding point gets no positional encoding, only its own embedding.
            encoding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1) * (1 - types[:, :1])
            sparse = (encoding + types @ self.table)[None]
            return self.decoder.predict_masks(image_embeddings, self.image_pe, sparse, self.dense)

    from tiny_vit_4d import ImageEncoder

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = ImageEncoder(sam.image_encoder)
            self.prompt_decoder = PromptDecoder()

        def forward(self, image, features, image_embeddings, point_coords, point_types):
            return (
                self.image_encoder.start(image),
                self.image_encoder.finish(features),
                *self.prompt_decoder(image_embeddings, point_coords, point_types),
            )

    return SideBySide().eval()


def reference_decoder(sam, image_embeddings, points, types):
    """segment_anything's own point embedding and mask prediction for the same prompt."""
    import torch

    # segment_anything's ResizeLongestSide.apply_coords: pixels scaled to the resized image.
    height, width = ORIGINAL_SIZE
    resized_height, resized_width = resized_size(height, width)
    pixels = np.zeros((POINTS, 2))
    pixels[: len(points)] = np.asarray(points, np.float64) * (resized_width / width, resized_height / height)
    pixels = torch.tensor(pixels[None], dtype=torch.float32)
    slots = [*types, *["padding"] * (POINTS - len(types))]
    prompt = sam.prompt_encoder
    # Clicks and padding go through _embed_points (labels 0, 1 and -1); a box's corners through _embed_boxes.
    labels = torch.tensor([[{"padding": -1, "background": 0, "foreground": 1}.get(kind, -1) for kind in slots]])
    tokens = list(prompt._embed_points(pixels, labels, pad=False)[0])
    if "box_top_left" in slots:
        corners = [slots.index("box_top_left"), slots.index("box_bottom_right")]
        box = prompt._embed_boxes(pixels[:, corners].reshape(1, 4))[0]
        for slot, token in zip(corners, box):
            tokens[slot] = token
    sparse = torch.stack(tokens)[None]
    dense = prompt.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, *EMBEDDING_SHAPE[2:])
    return sam.mask_decoder.predict_masks(image_embeddings, prompt.get_dense_pe(), sparse, dense)


def seeded_image(seed):
    """Normalized pixels lie in about [-2.1, 2.7]; the image's bottom rows are padding, as for a 4:3 image."""
    image = np.random.RandomState(seed).uniform(-2.1, 2.7, INPUT_SHAPES["image_encoder_0"]["image"]).astype(np.float32)
    image[:, :, 768:] = 0
    return image


def check_encoder(sam, side_by_side):
    """The 4D encoder's two parts, chained, give upstream TinyViT's embedding on seeded images."""
    import torch

    for seed in SEEDS[:2]:
        image = torch.from_numpy(seeded_image(seed))
        with torch.inference_mode():
            expected = sam.image_encoder(image).numpy()
            encoder = side_by_side.image_encoder
            actual = encoder.finish(encoder.start(image)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, err_msg=f"4D encoder (seed {seed})")


def check_embedding(sam, side_by_side):
    """The decoder's own point embedding gives segment_anything's masks and scores for every check prompt."""
    import torch

    embeddings = torch.from_numpy(np.random.RandomState(SEEDS.start).standard_normal(EMBEDDING_SHAPE) * 0.3).float()
    for points, types in PROMPTS:
        inputs = {key: torch.from_numpy(value) for key, value in prompt_inputs(points, types, ORIGINAL_SIZE).items()}
        with torch.inference_mode():
            actual = side_by_side.prompt_decoder(embeddings, inputs["point_coords"], inputs["point_types"])
            expected = reference_decoder(sam, embeddings, points, types)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=1e-4, err_msg=str(types))


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"MobileSAM export requires opset {OPSET}")
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    expected = (
        {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()},
        {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()},
    )
    actual = ({v.name: shape(v) for v in model.graph.input}, {v.name: shape(v) for v in model.graph.output})
    if actual != expected:
        raise ValueError(f"Unexpected interface: {actual}")
    if any(
        value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        for value in [*model.graph.input, *model.graph.output]
    ):
        raise ValueError("Every input and output must be float32")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    # Folding only: onnxsim's optimizer would fuse token Linears into 2D Gemms, whose bias the compiler mis-types.
    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded MobileSAM export")
    onnx.save(model, path)


def compare(expected, actual, name):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg=name)


def check_onnx(side_by_side, path):
    """Each component's ONNX outputs match PyTorch's on seeded images and the check prompts. The second encoder
    part takes the first's PyTorch features of the same image."""
    import onnxruntime as ort
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = [tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]]
    for seed, (points, types) in zip(SEEDS, [*PROMPTS, PROMPTS[0]]):
        image = seeded_image(seed)
        with torch.inference_mode():
            features = side_by_side.image_encoder.start(torch.from_numpy(image)).numpy()
        embeddings = (np.random.RandomState(seed).standard_normal(EMBEDDING_SHAPE) * 0.3).astype(np.float32)
        prompt = prompt_inputs(points, types, ORIGINAL_SIZE)
        feeds = {
            tensor_name("image_encoder_0", "image"): image,
            tensor_name("image_encoder_1", "features"): features,
            tensor_name("prompt_decoder", "image_embeddings"): embeddings,
            **{tensor_name("prompt_decoder", key): value for key, value in prompt.items()},
        }
        with torch.inference_mode():
            expected = side_by_side(*(torch.from_numpy(value) for value in feeds.values()))
        for name, want, got in zip(outputs, expected, session.run(outputs, feeds)):
            compare(want.numpy(), got, f"{name} (seed {seed})")


def convert(source, output_dir):
    import torch

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    sam = build(source)
    side_by_side = modules(sam)
    check_encoder(sam, side_by_side)
    check_embedding(sam, side_by_side)
    inputs = [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]
    with tempfile.TemporaryDirectory(prefix="mobile-sam-") as temporary:
        exported = Path(temporary) / "model.onnx"
        torch.onnx.export(
            side_by_side,
            tuple(torch.zeros(INPUT_SHAPES[c][n]) for c in COMPONENTS for n in INPUT_SHAPES[c]),
            str(exported),
            input_names=inputs,
            output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        fold_constants(exported)
        validate_export(exported)
        exported.replace(output)
    check_onnx(side_by_side, output)
    seeds = f"{SEEDS.start}-{SEEDS.stop - 1}"
    print(f"4D encoder, prompt embedding and PyTorch/ONNX checks passed with seeds {seeds}: {output}")
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
