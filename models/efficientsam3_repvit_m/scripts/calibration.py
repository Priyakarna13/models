"""Prepare EfficientSAM3 inputs: COCO category prompts, packed text and stretched images."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

IMAGE_SIZE = 1008
CONTEXT = 16
VOCABULARY = 49408
FEATURES = 256
START, END = 49406, 49407
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 0
# The 80 COCO 2017 category names in category-id order, with their BPE token IDs between the start
# and end tokens. convert.py checks this table against the pinned CLIP tokenizer.
COCO_TOKENS = {
    "person": (2533,),
    "bicycle": (11652,),
    "car": (1615,),
    "motorcycle": (10297,),
    "airplane": (16451,),
    "bus": (2840,),
    "train": (3231,),
    "truck": (4629,),
    "boat": (4440,),
    "traffic light": (3399, 1395),
    "fire hydrant": (1769, 8031, 773),
    "stop sign": (1691, 2292),
    "parking meter": (5984, 10104),
    "bench": (8771,),
    "bird": (3329,),
    "cat": (2368,),
    "dog": (1929,),
    "horse": (4558,),
    "sheep": (9629,),
    "cow": (9706,),
    "elephant": (10299,),
    "bear": (4298,),
    "zebra": (22548,),
    "giraffe": (22826,),
    "backpack": (14894,),
    "umbrella": (17143,),
    "handbag": (22654,),
    "tie": (3422,),
    "suitcase": (27792,),
    "frisbee": (45768,),
    "skis": (32843,),
    "snowboard": (33403,),
    "sports ball": (2054, 1069),
    "kite": (19867,),
    "baseball bat": (3470, 6575),
    "baseball glove": (3470, 16078),
    "skateboard": (31777,),
    "surfboard": (10176, 1972),
    "tennis racket": (5298, 37691),
    "bottle": (5392,),
    "wine glass": (2604, 3313),
    "cup": (1937,),
    "fork": (13700,),
    "knife": (8960,),
    "spoon": (14024,),
    "bowl": (3814,),
    "banana": (8922,),
    "apple": (3055,),
    "sandwich": (8687,),
    "orange": (4287,),
    "broccoli": (19094,),
    "carrot": (15435,),
    "hot dog": (2069, 1929),
    "pizza": (4474,),
    "donut": (18471,),
    "cake": (2972,),
    "chair": (4269,),
    "couch": (12724,),
    "potted plant": (48581, 3912),
    "bed": (2722,),
    "dining table": (8658, 2175),
    "toilet": (11071,),
    "tv": (1583,),
    "laptop": (10464,),
    "mouse": (9301,),
    "remote": (9687,),
    "keyboard": (13017,),
    "cell phone": (5533, 1951),
    "microwave": (24240,),
    "oven": (12579,),
    "toaster": (39061,),
    "sink": (14551,),
    "refrigerator": (36662,),
    "book": (1116,),
    "clock": (6716,),
    "vase": (20431,),
    "scissors": (30867,),
    "teddy bear": (11798, 4298),
    "hair drier": (2225, 39877),
    "toothbrush": (36841,),
}


def tokens(prompt: str) -> list[int]:
    """Return the prompt's token IDs, start and end tokens included, without padding."""
    value = [START, *COCO_TOKENS[prompt], END]
    if len(value) > CONTEXT:
        raise ValueError(
            f"prompt {prompt!r} has {len(value) - 2} tokens; the {CONTEXT}-position context holds at most "
            f"{CONTEXT - 2} between the start and end tokens"
        )
    return value


def one_hot(prompt_tokens: Sequence[int]) -> np.ndarray:
    """Return the text encoder input: one row per context position, padding rows all zero."""
    value = np.zeros((1, CONTEXT, VOCABULARY), np.float32)
    value[0, np.arange(len(prompt_tokens)), prompt_tokens] = 1
    return value


def packed(features: np.ndarray, prompt_tokens: Sequence[int]) -> np.ndarray:
    """Return the image model's text input: rows 0-15 hold the features, row 16 the token validity mask."""
    text = np.zeros((1, CONTEXT + 1, FEATURES), np.float32)
    text[0, :CONTEXT] = np.asarray(features, np.float32).reshape(CONTEXT, FEATURES)
    text[0, CONTEXT, : len(prompt_tokens)] = 1
    return text


def prompts(count: int) -> list[str]:
    """Spread count prompts evenly over the category list."""
    names = list(COCO_TOKENS)
    return [names[index * len(names) // count] for index in range(count)]


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the image model input for one encoded image: RGB stretched to 1008x1008, scaled to [-1, 1], HWC.

    SAM 3 resizes the uint8 image with antialiased bilinear filtering. Area averaging stands in for the
    antialiasing along an axis that shrinks; an axis that enlarges uses plain bilinear interpolation.
    """
    image = decode_image(content)
    return resize(image).astype(np.float32) / np.float32(127.5) - np.float32(1)


def resize(image: np.ndarray) -> np.ndarray:
    """Stretch an HWC image to IMAGE_SIZE x IMAGE_SIZE, area-averaging each shrinking axis and bilinear otherwise."""
    import cv2

    height, width = image.shape[:2]
    shrinks = (width > IMAGE_SIZE, height > IMAGE_SIZE)
    if shrinks[0] == shrinks[1]:
        interpolation = cv2.INTER_AREA if shrinks[0] else cv2.INTER_LINEAR
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)
    # One axis shrinks and the other enlarges: resize each axis on its own.
    width_method, height_method = (cv2.INTER_AREA if shrink else cv2.INTER_LINEAR for shrink in shrinks)
    image = cv2.resize(image, (IMAGE_SIZE, height), interpolation=width_method)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=height_method)


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


def _image_names(dataset: Path) -> list[str]:
    if dataset.is_dir():
        return sorted(
            path.relative_to(dataset).as_posix()
            for path in dataset.rglob("*")
            if path.is_file() and _is_image(path.name)
        )
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            return sorted(member.name for member in archive.getmembers() if member.isfile() and _is_image(member.name))
    raise ValueError(f"unsupported image dataset input: {dataset}")


def image_bytes(dataset: Path, names: Sequence[str]) -> Iterator[bytes]:
    if dataset.is_dir():
        for name in names:
            yield (dataset / name).read_bytes()
        return
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            members = {member.name: member for member in archive.getmembers()}
            for name in names:
                stream = archive.extractfile(members[name])
                if stream is None:
                    raise ValueError(f"could not read image from dataset: {name}")
                yield stream.read()
        return
    raise ValueError(f"unsupported image dataset input: {dataset}")


def select_images(dataset: Path, calibration_count: int, fidelity_count: int) -> tuple[list[str], list[str]]:
    """Return disjoint seeded samples of the cal2017 images for calibration and fidelity, in sampling order."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    required = calibration_count + fidelity_count
    if calibration_count < 1 or fidelity_count < 1 or len(names) < required:
        raise ValueError(f"cannot select {required} images from {len(names)} {CALIBRATION_SPLIT} images")
    selected = random.Random(CALIBRATION_SEED).sample(names, required)
    return selected[:calibration_count], selected[calibration_count:]


def calibration_inputs(dataset: Path, names: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Return one prompt per image, spread over the categories, and the images as one NHWC batch."""
    return prompts(len(names)), np.stack([preprocess_image(content) for content in image_bytes(dataset, names)])
