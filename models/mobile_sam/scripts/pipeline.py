"""MobileSAM's interface, shared by conversion, compile and the application: component shapes, the image
preprocessing, the prompt inputs and the mask postprocessing. It needs only NumPy and Pillow, and no model weights.

The image encoder turns a 1024x1024 image into a [1, 256, 64, 64] embedding once per image, in two components
compiled apart: image_encoder_0 (patch embedding and stages 0 and 1) gives [1, 160, 64, 64] features, and
image_encoder_1 (stages 2 and 3 and the neck) turns them into the embedding. The prompt decoder then predicts masks
for each prompt from that embedding. A prompt is up to POINTS points, each a foreground or
background click or a box corner, given as normalized coordinates and a one-hot point type; the decoder embeds
them itself.
"""

from __future__ import annotations

import numpy as np

COMPONENTS = ("image_encoder_0", "image_encoder_1", "prompt_decoder")
IMAGE_SIZE = 1024
FEATURE_SHAPE = (1, 160, 64, 64)
EMBEDDING_SHAPE = (1, 256, 64, 64)
# Prompt slots; unused slots are padding, as segment_anything pads a prompt without a box.
POINTS = 5
# One-hot point types in this order; segment_anything labels them -1, 0, 1, 2 and 3.
POINT_TYPES = ("padding", "background", "foreground", "box_top_left", "box_bottom_right")
LOW_RES_MASK_SIZE = 256
MASKS = 4
INPUT_SHAPES = {
    "image_encoder_0": {"image": (1, 3, IMAGE_SIZE, IMAGE_SIZE)},
    "image_encoder_1": {"features": FEATURE_SHAPE},
    "prompt_decoder": {
        "image_embeddings": EMBEDDING_SHAPE,
        "point_coords": (1, 2, 1, POINTS),
        "point_types": (1, len(POINT_TYPES), 1, POINTS),
    },
}
OUTPUT_SHAPES = {
    "image_encoder_0": {"features": FEATURE_SHAPE},
    "image_encoder_1": {"image_embeddings": EMBEDDING_SHAPE},
    "prompt_decoder": {
        "masks": (1, MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE),
        "iou_predictions": (1, MASKS),
    },
}
# segment_anything's pixel normalization, on 0-255 RGB.
PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def tensor_name(component: str, name: str) -> str:
    """A component's tensor in the side-by-side ONNX that conversion writes."""
    return f"{component}_{name}"


def resized_size(height: int, width: int) -> tuple[int, int]:
    """The (height, width) segment_anything resizes an image to: its longer side becomes IMAGE_SIZE."""
    scale = IMAGE_SIZE / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def preprocess(image: np.ndarray) -> np.ndarray:
    """An RGB uint8 HWC image as the encoder's float32 HWC input: resized bilinearly so its longer side is
    IMAGE_SIZE, normalized, then padded with zeros on the right and bottom."""
    from PIL import Image

    height, width = resized_size(*image.shape[:2])
    resized = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR), np.float32)
    padded = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    padded[:height, :width] = (resized - PIXEL_MEAN) / PIXEL_STD
    return padded


def prompt_inputs(points: np.ndarray, types: list[str], original_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The decoder's point inputs for up to POINTS points, given as (x, y) pixels in the original image.

    Coordinates are scaled to the resized image, shifted to pixel centres and normalized to [-1, 1], as
    segment_anything's prompt encoder does before its positional encoding. Unused slots are padding.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) != len(types) or not 0 < len(points) <= POINTS:
        raise ValueError(f"expected 1 to {POINTS} points with one type each, got {len(points)} and {len(types)}")
    unknown = sorted(set(types) - set(POINT_TYPES[1:]))
    if unknown:
        raise ValueError(f"unknown point types {unknown}; use {', '.join(POINT_TYPES[1:])}")
    height, width = original_size
    resized_height, resized_width = resized_size(height, width)
    scaled = points * (resized_width / width, resized_height / height)
    coords = np.zeros((POINTS, 2), np.float32)
    coords[: len(points)] = (scaled + 0.5) / IMAGE_SIZE * 2 - 1
    one_hot = np.zeros((POINTS, len(POINT_TYPES)), np.float32)
    one_hot[np.arange(POINTS), [POINT_TYPES.index(kind) for kind in types] + [0] * (POINTS - len(types))] = 1
    return {
        "point_coords": np.ascontiguousarray(coords.T[None, :, None, :]),
        "point_types": np.ascontiguousarray(one_hot.T[None, :, None, :]),
    }


def postprocess_masks(masks: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Low-resolution mask logits [MASKS, 256, 256] as logits at the original size, as segment_anything does:
    upsample to IMAGE_SIZE, drop the padding, then resize to the original image."""
    from PIL import Image

    height, width = original_size
    resized_height, resized_width = resized_size(height, width)

    def resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        return np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR))

    return np.stack(
        [
            resize(resize(mask, (IMAGE_SIZE, IMAGE_SIZE))[:resized_height, :resized_width], (width, height))
            for mask in np.asarray(masks, np.float32).reshape(MASKS, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE)
        ]
    )
