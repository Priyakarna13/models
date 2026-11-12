"""Export MobileSAM's whole image encoder and reusable prompt decoder, and validate upstream parity.

The image encoder runs once per image. The decoder accepts each prompt against the retained embedding;
there is no separately compiled intermediate feature component. Fourier encoding stays in the reference
export and is cut only by surgery for compilation.
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
    MAX_POINTS,
    OUTPUT_SHAPES,
    preprocess,
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
    (
        [(200 + 100 * index, 150 + 80 * index) for index in range(count)],
        ["foreground" if index % 2 == 0 else "background" for index in range(count)],
    )
    for count in range(1, MAX_POINTS + 1)
] + [
    ([(512, 384)], ["background"]),
    *[
        (
            [(100, 80), (900, 700), *[(300 + 100 * index, 200 + 100 * index) for index in range(count)]],
            [
                "box_top_left",
                "box_bottom_right",
                *["foreground" if index % 2 == 0 else "background" for index in range(count)],
            ],
        )
        for count in range(MAX_POINTS - 1)
    ],
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
    from sam_decoder import predict_masks

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
            encoding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1) * types[:, 1:].sum(
                dim=-1, keepdim=True
            )
            sparse = (encoding + types @ self.table)[None]
            return predict_masks(
                self.decoder, image_embeddings, self.image_pe, sparse, self.dense, types.sum(dim=-1)[None]
            )

    from tiny_vit_4d import ImageEncoder

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = ImageEncoder(sam.image_encoder)
            self.prompt_decoder = PromptDecoder()

        def forward(self, image, image_embeddings, point_coords, point_types):
            return (
                self.image_encoder(image),
                *self.prompt_decoder(image_embeddings, point_coords, point_types),
            )

    return SideBySide().eval()


def reference_decoder(sam, image_embeddings, points, types):
    """Independent upstream public prompt encoder and decoder; never manually pad or mask its tokens."""
    import torch

    height, width = ORIGINAL_SIZE
    resized_height, resized_width = resized_size(height, width)
    scaled = np.asarray(points, np.float64) * (resized_width / width, resized_height / height)
    clicks = [index for index, kind in enumerate(types) if kind in ("foreground", "background")]
    point_input = None
    if clicks:
        coordinates = torch.as_tensor(scaled[clicks][None], dtype=torch.float32)
        labels = torch.tensor([[int(types[index] == "foreground") for index in clicks]])
        point_input = (coordinates, labels)
    boxes = None
    if "box_top_left" in types:
        corners = [types.index("box_top_left"), types.index("box_bottom_right")]
        boxes = torch.as_tensor(scaled[corners].reshape(1, 4), dtype=torch.float32)
    sparse, dense = sam.prompt_encoder(points=point_input, boxes=boxes, masks=None)
    inputs = dict(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
    )
    single = sam.mask_decoder(**inputs, multimask_output=False)
    multiple = sam.mask_decoder(**inputs, multimask_output=True)
    return tuple(torch.cat([one, three], dim=1) for one, three in zip(single, multiple))


def seeded_image(seed):
    """Normalized pixels lie in about [-2.1, 2.7]; the image's bottom rows are padding, as for a 4:3 image."""
    image = np.random.RandomState(seed).uniform(-2.1, 2.7, INPUT_SHAPES["image_encoder"]["image"]).astype(np.float32)
    image[:, :, 768:] = 0
    return image


def check_encoder(sam, side_by_side):
    """The whole 4D encoder gives upstream TinyViT's embedding on seeded images."""
    import torch

    for seed in SEEDS[:2]:
        image = torch.from_numpy(seeded_image(seed))
        with torch.inference_mode():
            expected = sam.image_encoder(image).numpy()
            encoder = side_by_side.image_encoder
            actual = encoder(image).numpy()
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
        # Mask logits reach about 16; float32 reordering on the CI runners moves a few by up to 2e-4.
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=1e-3, err_msg=str(types))


def check_predictor(sam, side_by_side):
    """Check full image-to-low-resolution-mask behavior against the unmodified public SamPredictor API."""
    import torch
    from segment_anything import SamPredictor

    predictor = SamPredictor(sam)
    for size in ((300, 600), (433, 301)):
        image = np.random.RandomState(SEEDS.start).randint(0, 256, (*size, 3), dtype=np.uint8)
        with torch.inference_mode():
            predictor.set_image(image)
            pixels = torch.from_numpy(preprocess(image).transpose(2, 0, 1)[None])
            embedding = side_by_side.image_encoder(pixels)
            np.testing.assert_allclose(embedding.numpy(), predictor.features.numpy(), rtol=1e-4, atol=1e-4)
            for points, types in PROMPTS:
                points = np.asarray(points) * (size[1] / ORIGINAL_SIZE[1], size[0] / ORIGINAL_SIZE[0])
                click_indices = [index for index, kind in enumerate(types) if kind in ("foreground", "background")]
                clicks = points[click_indices] if click_indices else None
                labels = (
                    np.asarray([int(types[index] == "foreground") for index in click_indices])
                    if click_indices
                    else None
                )
                box = None
                if "box_top_left" in types:
                    box = points[[types.index("box_top_left"), types.index("box_bottom_right")]].reshape(4)
                _, score0, mask0 = predictor.predict(
                    clicks, labels, box=box, multimask_output=False, return_logits=True
                )
                _, scores, masks = predictor.predict(clicks, labels, box=box, multimask_output=True, return_logits=True)
                prompt = {name: torch.from_numpy(value) for name, value in prompt_inputs(points, types, size).items()}
                actual = side_by_side.prompt_decoder(embedding, prompt["point_coords"], prompt["point_types"])
                compare(
                    np.concatenate([mask0, masks])[None], actual[0].numpy(), f"public predictor masks {size} {types}"
                )
                compare(
                    np.concatenate([score0, scores])[None], actual[1].numpy(), f"public predictor scores {size} {types}"
                )
    print(f"Public SamPredictor parity passed: {len(PROMPTS)} prompt modes on two image geometries")


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
    """The whole encoder and decoder ONNX outputs match PyTorch on seeded images and the check prompts."""
    import onnxruntime as ort
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = [tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]]
    for index, (points, types) in enumerate(PROMPTS):
        seed = SEEDS[index % len(SEEDS)]
        image = seeded_image(seed)
        embeddings = (np.random.RandomState(seed).standard_normal(EMBEDDING_SHAPE) * 0.3).astype(np.float32)
        prompt = prompt_inputs(points, types, ORIGINAL_SIZE)
        feeds = {
            tensor_name("image_encoder", "image"): image,
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
    check_predictor(sam, side_by_side)
    inputs = [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]
    with tempfile.TemporaryDirectory(prefix=".mobile-sam-", dir=output_dir) as temporary:
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
        check_onnx(side_by_side, exported)
        exported.replace(output)
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
