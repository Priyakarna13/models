"""DINOv3 ViT-H+/16's interface, shared by the conversion, surgery, compile and benchmark scripts: the image shape, the
token layout and the components. It needs no model weights.

The encoder runs as four components of eight transformer blocks each. `blocks_0` also embeds the 16x16 patches
and prepends the class and four register tokens; `blocks_3` also applies the final norm and returns the class
embedding and the 196 patch embeddings. Between components the tokens travel as [1, HIDDEN_SIZE, 1, TOKENS]:
channels on C, tokens on W.
"""

from __future__ import annotations

IMAGE_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 1280
REGISTERS = 4
PREFIX = 1 + REGISTERS
PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
TOKENS = PREFIX + PATCHES
DEPTH = 32
COMPONENTS = ("blocks_0", "blocks_1", "blocks_2", "blocks_3")
# The transformer blocks [first, last) each component runs.
BLOCKS = {"blocks_0": (0, 8), "blocks_1": (8, 16), "blocks_2": (16, 24), "blocks_3": (24, 32)}
IMAGE_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
TOKEN_SHAPE = [1, HIDDEN_SIZE, 1, TOKENS]
OUTPUTS = ("class_embedding", "patch_embeddings")
# Each component's ONNX interface, in order: inputs, then outputs.
INTERFACES = {
    "blocks_0": {"pixel_values": IMAGE_SHAPE, "tokens": TOKEN_SHAPE},
    "blocks_1": {"input_tokens": TOKEN_SHAPE, "tokens": TOKEN_SHAPE},
    "blocks_2": {"input_tokens": TOKEN_SHAPE, "tokens": TOKEN_SHAPE},
    "blocks_3": {
        "input_tokens": TOKEN_SHAPE,
        "class_embedding": [1, HIDDEN_SIZE],
        "patch_embeddings": [1, PATCHES, HIDDEN_SIZE],
    },
}
INPUTS = {component: next(iter(interface)) for component, interface in INTERFACES.items()}
