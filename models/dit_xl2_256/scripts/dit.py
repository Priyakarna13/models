"""DiT-XL/2 256's component interface and CPU steps, shared by convert.py and compile.py.

Generation runs the transformer once per denoising step and the VAE decoder once at the end. The transformer
component runs one classifier-free-guidance step: batch slot 0 is the requested class, slot 1 the null class.
Its conditioning arrives as inputs every tensor of which is 4D: the timestep's sinusoidal features, which need no
weights, and the class as a one-hot vector, so each block's label embedding is a matrix product. The graph takes
one slot; the transformer compiles at batch size GUIDANCE_BATCH, so one call runs both. The CPU combines the two
noise predictions, runs the DDIM update, and scales the final latent for the decoder.
"""

from __future__ import annotations

import math

import numpy as np

COMPONENTS = ("transformer", "vae_decoder")
GUIDANCE_BATCH = 2
LATENT_CHANNELS, LATENT_SIZE = 4, 32
IMAGE_SIZE = 256
TIMESTEP_FEATURES = 256
# ImageNet's 1000 classes and the null class classifier-free guidance uses.
CLASSES = 1000
NULL_CLASS = CLASSES
# The noise and the learned variance, 4 channels each; DDIM uses the noise.
PREDICTION_CHANNELS = 8
# sd-vae-ft-ema's latent scale.
VAE_SCALING = 0.18215
OPSET = 17

# One batch slot of each component's graph.
INPUT_SHAPES = {
    "transformer": {
        "latent": (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
        "timestep_features": (1, TIMESTEP_FEATURES, 1, 1),
        "class_one_hot": (1, CLASSES + 1, 1, 1),
    },
    "vae_decoder": {"latent": (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)},
}
OUTPUT_SHAPES = {
    "transformer": {"prediction": (1, PREDICTION_CHANNELS, LATENT_SIZE, LATENT_SIZE)},
    "vae_decoder": {"image": (1, 3, IMAGE_SIZE, IMAGE_SIZE)},
}


def tensor_name(component: str, name: str) -> str:
    """Graph tensor names are prefixed with their component, so both fit in one ONNX."""
    return f"{component}_{name}"


def timestep_features(timestep: float) -> np.ndarray:
    """[2, 256, 1, 1] sinusoidal features of one timestep for both guidance slots, as diffusers' Timesteps(256,
    flip_sin_to_cos=True, downscale_freq_shift=1) computes them: cosines, then sines."""
    half = TIMESTEP_FEATURES // 2
    frequencies = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / (half - 1))
    angles = np.float32(timestep) * frequencies
    features = np.concatenate([np.cos(angles), np.sin(angles)]).astype(np.float32)
    return np.repeat(features[None, :, None, None], GUIDANCE_BATCH, axis=0)


def class_one_hot(label: int) -> np.ndarray:
    """[2, 1001, 1, 1]: the requested class in slot 0, the null class in slot 1."""
    if not 0 <= label < CLASSES:
        raise ValueError(f"class label must be in [0, {CLASSES}), got {label}")
    one_hot = np.zeros((GUIDANCE_BATCH, CLASSES + 1, 1, 1), np.float32)
    one_hot[0, label] = one_hot[1, NULL_CLASS] = 1.0
    return one_hot


def guided_noise(prediction: np.ndarray, scale: float) -> np.ndarray:
    """[1, 4, 32, 32] classifier-free-guided noise from the transformer's two slots."""
    conditional, unconditional = prediction[0:1, :LATENT_CHANNELS], prediction[1:2, :LATENT_CHANNELS]
    return unconditional + scale * (conditional - unconditional)
