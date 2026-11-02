"""Mask2Former's interface, shared by conversion, compile and the application: component shapes, the image
preprocessing and the instance decoding. It needs only NumPy and Pillow, and no model weights.

The model runs as three components. The backbone (Swin-T) turns a 384x384 image into four feature maps, at 1/4 to
1/32 of the input. The pixel decoder turns them into per-pixel mask features at 1/4 and three multi-scale feature
maps at 1/32, 1/16 and 1/8. The transformer decoder predicts, for each of 100 queries, class logits over the 80
COCO classes plus "no object", and mask logits at 1/4 of the input.
"""

from __future__ import annotations

import numpy as np

COMPONENTS = ("backbone", "pixel_decoder", "transformer_decoder")
IMAGE_SIZE = 384
QUERIES = 100
CLASSES = 80
MASK_SIZE = IMAGE_SIZE // 4
SCALE_SIZES = (IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8)
BACKBONE_SHAPES = tuple((1, 96 * 2**i, IMAGE_SIZE // 4 // 2**i, IMAGE_SIZE // 4 // 2**i) for i in range(4))
INPUT_SHAPES = {
    "backbone": {"image": (1, 3, IMAGE_SIZE, IMAGE_SIZE)},
    "pixel_decoder": {f"features{i}": shape for i, shape in enumerate(BACKBONE_SHAPES)},
    "transformer_decoder": {
        "mask_features": (1, 256, MASK_SIZE, MASK_SIZE),
        **{f"scale{i}": (1, 256, size, size) for i, size in enumerate(SCALE_SIZES)},
    },
}
OUTPUT_SHAPES = {
    "backbone": dict(INPUT_SHAPES["pixel_decoder"]),
    "pixel_decoder": dict(INPUT_SHAPES["transformer_decoder"]),
    "transformer_decoder": {
        # Class logits per query, classes along the channels: [1, classes + 1, 1, queries].
        "class_logits": (1, CLASSES + 1, 1, QUERIES),
        "mask_logits": (1, QUERIES, MASK_SIZE, MASK_SIZE),
    },
}
# The checkpoint's ImageNet normalization, on 0-255 RGB.
PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def tensor_name(component: str, name: str) -> str:
    """A component's tensor in the side-by-side ONNX that conversion writes."""
    return f"{component}_{name}"


def preprocess(image: np.ndarray) -> np.ndarray:
    """An RGB uint8 HWC image as the backbone's float32 HWC input: stretched bilinearly to IMAGE_SIZE x IMAGE_SIZE
    and normalized, as transformers' Mask2Former image processor does. At this input size, stretching scored
    higher COCO mask AP than resizing the longer side and padding (float32, 200 seeded val2017 images: 0.397 vs
    0.374)."""
    from PIL import Image

    resized = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(resized, np.float32) - PIXEL_MEAN) / PIXEL_STD


def _resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float32 map to (height, width)."""
    from PIL import Image

    height, width = size
    return np.asarray(
        Image.fromarray(mask.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BILINEAR)
    )


def decode(
    class_logits: np.ndarray, mask_logits: np.ndarray, original_size: tuple[int, int], top_k: int = QUERIES
) -> list[tuple[int, float, np.ndarray]]:
    """The model's outputs as instances (class index, score, binary mask at the original size), as Mask2Former's
    instance inference does: the top_k (query, class) pairs by class probability, "no object" excluded; each mask's
    logits upsampled to the input, resized to the image and thresholded at 0; each score the
    class probability times the mean mask probability inside the mask."""
    logits = np.asarray(class_logits, np.float32).reshape(CLASSES + 1, QUERIES).T
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = (probabilities / probabilities.sum(axis=1, keepdims=True))[:, :CLASSES]
    flat = probabilities.ravel()
    order = np.argsort(-flat, kind="stable")[:top_k]
    masks = np.asarray(mask_logits, np.float32).reshape(QUERIES, MASK_SIZE, MASK_SIZE)
    instances = []
    for index in order:
        query, label = divmod(int(index), CLASSES)
        logit = _resize(_resize(masks[query], (IMAGE_SIZE, IMAGE_SIZE)), original_size)
        binary = logit > 0
        mask_probability = 1 / (1 + np.exp(-logit))
        mask_score = float((mask_probability * binary).sum() / (binary.sum() + 1e-6))
        instances.append((label, float(flat[index]) * mask_score, binary))
    return instances
