"""Rebuild EfficientSAM3 RepViT-M from its checkpoints, export one ONNX and check it against PyTorch.

The sam3 wheel builds SAM 3's text encoder, neck, encoder, decoder and heads; timm builds the RepViT-M1.1
trunk. The Stage 1 checkpoint supplies SAM 3's weights; the fine-tuned student supplies the trunk and neck.
The student image encoder around the trunk is written here after SimonZeng7108/efficientsam3 at
bd0936c788fed8d51fa799437f05abd97b401b06, which publishes no package. That student resamples the trunk's
features to a 72x72 grid of image tokens (embed_size=72 in sam3/sam3/model_builder.py); this export resamples
them to 48x48 instead, so it returns 192x192 mask logits where the upstream model returns 288x288.

The exported graph re-expresses the same computation with four-dimensional token tensors, finite masks and
explicit constants. It runs the neck's leading 1x1 and 2x2 transposed convs on the trunk's 32x32 map before the
resize, as 1x1 convs and PixelShuffle, folds the instance head's 1x1 conv into the mask queries, and computes group
norms with 1x1 convs. It must match the stock PyTorch model before it is accepted. The stock model is built at
the same 48x48 grid, so this check proves the export faithful at grid 48, not equal to the upstream grid-72
model.
"""

import argparse
import gzip
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from calibration import COCO_TOKENS, CONTEXT, END, FEATURES, IMAGE_SIZE, START, VOCABULARY, one_hot, packed, tokens
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from surgery import INTERFACE, QUERIES, image_differences

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "efficient_sam3_repvit_m.pt"
# The fine-tuned student (efficientsam3_ft) recalls about three times as many COCO objects from a text prompt as the
# Stage 1 trunk and neck. It carries a MobileCLIP-S0 text encoder instead of SAM 3's, so only its trunk and neck load.
BACKBONE = "efficientsam3_repvit.pt"
BACKBONE_MODULES = ("backbone.vision_backbone.trunk.", "backbone.vision_backbone.convs.")
TRUNK_PREFIX = "backbone.vision_backbone.trunk.model.backbone.model."
WHEEL = "sam3-0.1.4-py3-none-any.whl"
PACKAGE = "sam3"
SOURCE = ("file_set", CHECKPOINT, [CHECKPOINT, BACKBONE, WHEEL, "vocab.json", "merges.txt"])
# The student encoder resamples the trunk's 32x downsampled features to this grid of image tokens. The
# upstream student uses 72; 48 makes the mask logits 4 * 48 = 192 pixels square instead of 288.
GRID = 48
TOKENS = GRID * GRID
# The MLA cannot gather from a 49,408-row table, so the token lookup runs as a one-hot MatMul in this many
# vocabulary chunks.
VOCABULARY_CHUNKS = 16
INPUTS = ("token_one_hot", "pixel_values", "text")
OUTPUTS = ("text_features", "detections", "mask_logits")
# CLIP's Hugging Face vocabulary names the special tokens differently from sam3's tokenizer.
SPECIAL_TOKENS = {"<|startoftext|>": "<start_of_text>", "<|endoftext|>": "<end_of_text>"}
# Synthetic images with single-token, two-token and three-token prompts.
CASES = [(123, "person"), (123, "traffic light"), (124, "fire hydrant"), (124, "giraffe")]
# Fused BatchNorm, folded attention scales and reordered float32 sums change rounding only; a wrong
# rewrite moves values by O(1). Twelve refinement layers amplify rounding in queries the model rejects,
# so boxes and masks are bounded relative to their scale over all queries and absolutely over
# confident queries (stock score >= surgery.CONFIDENT).
TOLERANCE = {
    "text_features": 1e-3,
    "scores": 1e-3,
    "boxes_relative": 1e-3,
    "masks_relative": 1e-3,
    "boxes_px_confident": 0.1,
    "mask_logits_confident": 0.05,
}


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(
            f"Expected the checkpoints, sam3 wheel and CLIP tokenizer {SOURCE}, model.yaml declares {declared}"
        )


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("EfficientSAM3 export requires opset 17")
    if [value.name for value in model.graph.input] != list(INPUTS) or [
        value.name for value in model.graph.output
    ] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def batched_outputs(outputs):
    """Return one case's (detections, mask logits) in the image model's batched output layout."""
    detections = np.concatenate([outputs["boxes_px"], outputs["scores"][:, None]], 1)
    return detections[None], outputs["mask_logits"][None]


def differences(expected, actual, valid):
    """Return the largest errors of one case; text features are compared at the prompt's tokens only."""
    text = np.abs(expected["text_features"].astype(np.float64) - actual["text_features"])[:valid].max()
    return {"text_features": float(text), **image_differences(batched_outputs(expected), batched_outputs(actual))}


def compare_outputs(expected, actual, valid):
    for name in expected:
        if expected[name].shape != actual[name].shape:
            raise ValueError(f"Unexpected shape for {name}")
        if not np.isfinite(expected[name]).all() or not np.isfinite(actual[name]).all():
            raise ValueError(f"Non-finite output: {name}")
    errors = differences(expected, actual, valid)
    failed = {name: value for name, value in errors.items() if not value <= TOLERANCE[name]}
    if failed:
        raise ValueError(f"ONNX differs from the stock model beyond tolerance: {failed}")
    return errors


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in INPUTS},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded EfficientSAM3 export")
    onnx.save(model, path)


def unpack_package(wheel, directory):
    """Unpack only the pure-Python package tree; it is installed without its declared dependencies."""
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.startswith(f"{PACKAGE}/")]
        if not members or any(".." in Path(name).parts for name in members):
            raise ValueError(f"Unexpected contents in {wheel.name}")
        archive.extractall(directory, members)
    sys.path.insert(0, str(directory))


def check_tokenizer(source_dir, work_dir):
    """Build sam3's tokenizer from the pinned CLIP merges and check it against the vocabulary and COCO_TOKENS."""
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe = work_dir / "bpe_simple_vocab_16e6.txt.gz"
    bpe.write_bytes(gzip.compress((source_dir / "merges.txt").read_bytes(), mtime=0))
    tokenizer = SimpleTokenizer(bpe_path=str(bpe))
    vocabulary = json.loads((source_dir / "vocab.json").read_text(encoding="utf-8"))
    if tokenizer.encoder != {SPECIAL_TOKENS.get(key, key): value for key, value in vocabulary.items()}:
        raise ValueError("merges.txt does not rebuild the vocab.json vocabulary")
    if [tokenizer.sot_token_id, tokenizer.eot_token_id] != [START, END]:
        raise ValueError("Unexpected start and end token IDs")
    for name in COCO_TOKENS:
        expected = tokens(name)
        if len(expected) > CONTEXT:
            raise ValueError(f"Prompt exceeds the exported context: {name}")
        if tokenizer([name], context_length=CONTEXT)[0].tolist() != expected + [0] * (CONTEXT - len(expected)):
            raise ValueError(f"COCO_TOKENS disagrees with the tokenizer for {name!r}")
    return bpe


def trunk_key(key):
    """Map an upstream RepViT ``features`` checkpoint key to timm's ``stem``/``stages`` modules."""
    rest = key.split("features.", 1)[1]
    index, rest = rest.split(".", 1)
    index = int(index)
    if index == 0:
        layer, rest = rest.split(".", 1)
        return {"0": "stem.conv1.", "2": "stem.conv2."}[layer] + rest
    targets = []
    for stage, depth in enumerate((2, 2, 12, 2)):
        if stage:
            targets += [(f"stages.{stage}.downsample.pre_block.", "block"), (f"stages.{stage}.downsample.", "down")]
        targets += [(f"stages.{stage}.blocks.{block}.", "block") for block in range(depth)]
    prefix, kind = targets[index - 1]
    names = (
        {
            "token_mixer.0.": "token_mixer.",
            "token_mixer.1.": "se.",
            "channel_mixer.m.0.": "channel_mixer.conv1.",
            "channel_mixer.m.2.": "channel_mixer.conv2.",
        }
        if kind == "block"
        else {
            "token_mixer.0.": "spatial_downsample.",
            "token_mixer.2.": "channel_downsample.",
            "channel_mixer.m.0.": "ffn.conv1.",
            "channel_mixer.m.2.": "ffn.conv2.",
        }
    )
    for old, new in names.items():
        if rest.startswith(old):
            return prefix + new + rest[len(old) :]
    raise KeyError(f"Unmapped RepViT checkpoint key: {key}")


def backbone_weights(checkpoint):
    """Return the fine-tuned student's trunk and neck under the detector's key names."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    weights = {}
    for key, value in state.items():
        if key.startswith(BACKBONE_MODULES):
            weights[TRUNK_PREFIX + trunk_key(key) if key.startswith(TRUNK_PREFIX) else key] = value
    return weights


def build_stock(checkpoint, fine_tuned, bpe):
    """Return the stock sam3 image model with the EfficientSAM3 student image encoder and its weights."""
    import timm
    import torch
    from sam3 import model_builder as builder
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.model_misc import TransformerWrapper
    from torch import nn
    from torch.nn import functional as F

    class Trunk(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = nn.Module()
            self.model.stem, self.model.stages = model.stem, model.stages

        def forward(self, x):
            return self.model.stages(self.model.stem(x))

    class Student(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Conv2d(512, 1024, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.GELU(),
                nn.Conv2d(1024, 1024, 3, padding=1),
            )

        def forward(self, x):
            return F.interpolate(self.head(self.backbone(x)), size=(GRID, GRID), mode="bilinear", align_corners=False)

    class OneLevel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model, self.channel_list = model, [1024]

        def forward(self, x):
            return [self.model(x)]

    trunk = OneLevel(Student(Trunk(timm.create_model("repvit_m1_1", pretrained=False))))
    neck = builder._create_vit_neck(builder._create_position_encoding(None), trunk, enable_inst_interactivity=False)
    backbone = builder._create_vl_backbone(neck, builder._create_text_encoder(str(bpe)))
    coordinates = TransformerDecoder._get_coords
    # The stock builder precomputes decoder coordinates on CUDA; build the same cache on the CPU.
    TransformerDecoder._get_coords = staticmethod(lambda height, width, device: coordinates(height, width, "cpu"))
    try:
        decoder = builder._create_transformer_decoder()
    finally:
        TransformerDecoder._get_coords = staticmethod(coordinates)
    transformer = TransformerWrapper(encoder=builder._create_transformer_encoder(), decoder=decoder, d_model=256)
    model = builder._create_sam3_model(
        backbone,
        transformer,
        builder._create_geometry_encoder(),
        builder._create_segmentation_head(),
        builder._create_dot_product_scoring(),
        None,
        True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if list(state) != ["model"]:
        raise ValueError(f"Expected one 'model' state dict in the checkpoint, found {list(state)}")
    state = state["model"]
    if any(not key.startswith(("detector.", "tracker.")) for key in state):
        raise ValueError("Checkpoint keys must belong to the detector or the tracker")
    detector = {}
    for key, value in state.items():
        if key.startswith("detector."):
            key = key[len("detector.") :]
            detector[TRUNK_PREFIX + trunk_key(key) if key.startswith(TRUNK_PREFIX) else key] = value
    tuned = backbone_weights(fine_tuned)
    stage1 = {key: value.shape for key, value in detector.items() if key.startswith(BACKBONE_MODULES)}
    if {key: value.shape for key, value in tuned.items()} != stage1:
        raise ValueError("The fine-tuned trunk and neck do not match the Stage 1 trunk and neck")
    detector.update(tuned)
    missing, unexpected = model.load_state_dict(detector, strict=False)
    # The checkpoint also carries the SAM 2 neck used only by the video tracker.
    stray = [key for key in unexpected if not key.startswith("backbone.vision_backbone.sam2_convs.")]
    if missing or stray:
        raise ValueError(f"Checkpoint does not match the model: missing={missing}, unexpected={stray}")
    return model.eval()


def build_graph(stock):
    """Return the exported computation, built from the stock modules' weights."""
    import copy

    import torch
    from timm.models.repvit import ConvNorm, RepVggDw
    from torch import nn
    from torch.nn import functional as F

    def constant(value):
        return nn.Parameter(value.detach().clone().float(), requires_grad=False)

    def gelu(x):
        # Exact erf GELU written out, so the compiler does not lower it through its GELU composite.
        return x * (torch.erf(x * 0.7071067811865476) * 0.5 + 0.5)

    class Gelu(nn.Module):
        def forward(self, x):
            return gelu(x)

    class Linear(nn.Module):
        def __init__(self, weight, bias, scale=1.0):
            super().__init__()
            self.weight, self.bias = constant(weight * scale), constant(bias * scale)

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    def linear(layer, scale=1.0):
        return Linear(layer.weight, layer.bias, scale)

    class Norm(nn.Module):
        """LayerNorm over the channel axis of a (1, 1, tokens, channels) tensor."""

        def __init__(self, norm):
            super().__init__()
            self.weight, self.bias, self.eps = constant(norm.weight), constant(norm.bias), norm.eps

        def forward(self, x):
            return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)

    class Mlp(nn.Module):
        def __init__(self, mlp, first=None):
            super().__init__()
            if mlp.residual or not isinstance(mlp.out_norm, nn.Identity):
                raise ValueError("Plain MLP expected")
            self.layers = nn.ModuleList(linear(layer) for layer in mlp.layers)
            if first is not None:
                self.layers[0].weight = constant(first)

        def forward(self, x):
            for index, layer in enumerate(self.layers):
                x = layer(x) if index == len(self.layers) - 1 else F.relu(layer(x))
            return x

    class Attention(nn.Module):
        """nn.MultiheadAttention on token tensors with a folded query scale and an additive bias."""

        def __init__(self, attention):
            super().__init__()
            if not attention._qkv_same_embed_dim or attention.bias_k is not None or attention.add_zero_attn:
                raise ValueError("Unsupported attention configuration")
            width, self.heads = attention.embed_dim, attention.num_heads
            self.dim = width // self.heads
            weight, bias = attention.in_proj_weight, attention.in_proj_bias
            self.q = Linear(weight[:width], bias[:width], self.dim**-0.5)
            self.k = Linear(weight[width : 2 * width], bias[width : 2 * width])
            self.v = Linear(weight[2 * width :], bias[2 * width :])
            self.o = linear(attention.out_proj)

        def heads_first(self, x):
            return x.reshape(1, x.shape[2], self.heads, self.dim).transpose(1, 2)

        def forward(self, query, key, value, bias=None):
            q, k, v = self.heads_first(self.q(query)), self.heads_first(self.k(key)), self.heads_first(self.v(value))
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
            if bias is not None:
                scores = scores + bias
            out = torch.einsum("bhqk,bhkd->bhqd", scores.softmax(-1), v)
            return self.o(out.transpose(1, 2).reshape(1, 1, query.shape[2], self.heads * self.dim))

    class TextBlock(nn.Module):
        def __init__(self, block):
            super().__init__()
            if not isinstance(block.ls_1, nn.Identity) or not isinstance(block.ls_2, nn.Identity):
                raise ValueError("Unexpected text layer scale")
            self.ln_1, self.attn, self.ln_2 = Norm(block.ln_1), Attention(block.attn), Norm(block.ln_2)
            self.fc, self.proj = linear(block.mlp.c_fc), linear(block.mlp.c_proj)

        def forward(self, x, causal):
            h = self.ln_1(x)
            x = x + self.attn(h, h, h, causal)
            return x + self.proj(gelu(self.fc(self.ln_2(x))))

    class TextEncoder(nn.Module):
        def __init__(self, language):
            super().__init__()
            encoder = language.encoder
            # Each one-hot row has at most one nonzero, so all chunks but one add exact zeros and the
            # sum equals the embedding lookup.
            self.chunk = VOCABULARY // VOCABULARY_CHUNKS
            self.embedding = nn.ParameterList(
                constant(part) for part in encoder.token_embedding.weight.split(self.chunk)
            )
            self.position = constant(encoder.positional_embedding[:CONTEXT])
            self.blocks = nn.ModuleList(TextBlock(block) for block in encoder.transformer.resblocks)
            self.ln_final, self.resizer = Norm(encoder.ln_final), linear(language.resizer)
            # Finite causal mask: exp(-1e4) is exactly zero in float32, like the stock -inf.
            self.register_buffer("causal", torch.full((1, 1, CONTEXT, CONTEXT), -1e4).triu(1))

        def forward(self, one_hot):
            one_hot = one_hot.reshape(1, 1, CONTEXT, VOCABULARY)
            x = self.position
            for index, part in enumerate(self.embedding):
                x = x + torch.matmul(one_hot[..., index * self.chunk : (index + 1) * self.chunk], part)
            for block in self.blocks:
                x = block(x, self.causal)
            return self.resizer(self.ln_final(x)).reshape(1, CONTEXT, FEATURES)

    def fuse_conv_bn(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups
        )
        fused.weight.data.copy_(conv.weight * scale[:, None, None, None])
        base = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.data.copy_(bn.bias + (base - bn.running_mean) * scale)
        return fused

    def reparameterize(module):
        for name, child in module.named_children():
            if isinstance(child, (RepVggDw, ConvNorm)):
                setattr(module, name, child.fuse())
            elif isinstance(child, nn.GELU):
                if child.approximate != "none":
                    raise ValueError("Exact GELU expected")
                setattr(module, name, Gelu())
            else:
                reparameterize(child)

    class ImageTrunk(nn.Module):
        def __init__(self, student):
            super().__init__()
            self.body = copy.deepcopy(student.backbone.model)
            reparameterize(self.body)
            self.reduce, self.mix = fuse_conv_bn(student.head[0], student.head[1]), copy.deepcopy(student.head[3])

        def forward(self, x):
            # The neck resamples to the token grid after its leading per-pixel layers.
            x = self.body.stages(self.body.stem(x))
            return self.mix(gelu(self.reduce(x)))

    def shuffle_conv(transposed, after=None):
        """Return a 2x2 stride-2 transposed conv, and an optional 1x1 conv after it, as one 1x1 conv to four times
        the channels, whose PixelShuffle(2) equals the original output."""
        weight, bias = transposed.weight, transposed.bias
        if after is not None:
            mix = after.weight[:, :, 0, 0]
            weight, bias = torch.einsum("oc,icab->ioab", mix, weight), mix @ bias + after.bias
        conv = nn.Conv2d(weight.shape[0], 4 * weight.shape[1], 1)
        conv.weight.data.copy_(weight.permute(1, 2, 3, 0).reshape(conv.weight.shape))
        conv.bias.data.copy_(bias.repeat_interleave(4))
        return conv

    class NeckLevel(nn.Module):
        """One neck level whose leading per-pixel affine layers run on the trunk's 32x32 map. Bilinear weights sum
        to 1, so resampling commutes with a per-pixel affine map; resampling a transposed conv's 4C channels before
        PixelShuffle equals resampling its input."""

        def __init__(self, level):
            super().__init__()
            names = [name for name, _ in level.named_children()]
            if names == ["dconv_2x2_0", "gelu", "dconv_2x2_1", "conv_1x1", "conv_3x3"]:
                self.pre = shuffle_conv(level.dconv_2x2_0)
                tail = [nn.PixelShuffle(2), Gelu(), shuffle_conv(level.dconv_2x2_1, level.conv_1x1), nn.PixelShuffle(2)]
            elif names == ["dconv_2x2", "conv_1x1", "conv_3x3"]:
                self.pre, tail = shuffle_conv(level.dconv_2x2, level.conv_1x1), [nn.PixelShuffle(2)]
            elif names == ["conv_1x1", "conv_3x3"]:
                self.pre, tail = copy.deepcopy(level.conv_1x1), []
            else:
                raise ValueError(f"Unexpected neck level {names}")
            self.tail = nn.Sequential(*tail, copy.deepcopy(level.conv_3x3))

        def forward(self, x):
            return self.tail(F.interpolate(self.pre(x), size=(GRID, GRID), mode="bilinear", align_corners=False))

    class Neck(nn.Module):
        def __init__(self, neck):
            super().__init__()
            self.scale4, self.scale2, self.scale1 = (NeckLevel(neck.convs[index]) for index in range(3))

        def forward(self, x):
            return self.scale4(x), self.scale2(x), self.scale1(x)

    class EncoderLayer(nn.Module):
        """Pre-norm encoder layer; geometry layers use one query and keys with positions."""

        def __init__(self, layer, geometry):
            super().__init__()
            expected = (True, not geometry, geometry, False, "relu")
            actual = (
                layer.pre_norm,
                layer.pos_enc_at_attn,
                layer.pos_enc_at_cross_attn_keys,
                layer.pos_enc_at_cross_attn_queries,
                layer.activation_str,
            )
            if actual != expected:
                raise ValueError(f"Unexpected encoder layer settings: {actual}")
            self.geometry = geometry
            self.norm1, self.norm2, self.norm3 = Norm(layer.norm1), Norm(layer.norm2), Norm(layer.norm3)
            if geometry:
                # Self-attention over one token returns its value projection exactly.
                width = layer.self_attn.embed_dim
                weight, bias = layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
                self.value = Linear(weight[2 * width :], bias[2 * width :])
                self.out = linear(layer.self_attn.out_proj)
            else:
                self.self_attn = Attention(layer.self_attn)
            self.cross_attn = Attention(layer.cross_attn_image)
            self.linear1, self.linear2 = linear(layer.linear1), linear(layer.linear2)

        def forward(self, x, pos, key, value, bias=None):
            h = self.norm1(x)
            x = x + (self.out(self.value(h)) if self.geometry else self.self_attn(h + pos, h + pos, h))
            x = x + self.cross_attn(self.norm2(x), key, value, bias)
            return x + self.linear2(F.relu(self.linear1(self.norm3(x))))

    class DecoderLayer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.self_attn, self.norm2 = Attention(layer.self_attn), Norm(layer.norm2)
            self.ca_text, self.catext_norm = Attention(layer.ca_text), Norm(layer.catext_norm)
            self.cross_attn, self.norm1 = Attention(layer.cross_attn), Norm(layer.norm1)
            self.linear1, self.linear2, self.norm3 = linear(layer.linear1), linear(layer.linear2), Norm(layer.norm3)

        def forward(self, t, pos, prompt, prompt_bias, key, memory, box_bias):
            t = self.norm2(t + self.self_attn(t + pos, t + pos, t))
            t = self.catext_norm(t + self.ca_text(t + pos, prompt, prompt, prompt_bias))
            t = self.norm1(t + self.cross_attn(t + pos, key, memory, box_bias))
            return self.norm3(t + self.linear2(F.relu(self.linear1(t))))

    def log_scale(t):
        # sign(t) * log2(|t| + 1) / 3 without Sign or Abs.
        return (torch.log(1 + F.relu(t)) - torch.log(1 + F.relu(-t))) * (1 / (3 * math.log(2)))

    class Decoder(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            if (
                decoder.boxRPB != "log"
                or not decoder.box_refine
                or decoder.presence_token is None
                or not decoder.use_normed_output_consistently
            ):
                raise ValueError("Unexpected decoder configuration")
            self.layers = nn.ModuleList(DecoderLayer(layer) for layer in decoder.layers)
            self.norm, self.box_head = Norm(decoder.norm), Mlp(decoder.bbox_embed)
            self.rpb_x, self.rpb_y = Mlp(decoder.boxRPB_embed_x), Mlp(decoder.boxRPB_embed_y)
            self.presence_norm = Norm(decoder.presence_token_out_norm)
            self.presence_head = Mlp(decoder.presence_token_head)
            heads = decoder.layers[0].cross_attn.num_heads
            self.register_buffer("query", decoder.query_embed.weight.detach().reshape(1, 1, QUERIES, 256).clone())
            self.register_buffer("presence", decoder.presence_token.weight.detach().reshape(1, 1, 1, 256).clone())
            reference = decoder.reference_points.weight.detach().sigmoid()
            self.register_buffer("reference", reference.reshape(1, 1, QUERIES, 4))
            # Sine embedding: angles for (y, x, w, h) blocks of 64 frequencies. The stock embedding
            # interleaves sin/cos pairs; the first ref_point_head columns are permuted to match.
            steps = torch.arange(128, dtype=torch.float32)
            frequency = (2 * math.pi) / (10000 ** (2 * torch.div(steps, 2, rounding_mode="floor") / 128))
            angles = torch.zeros(4, 256)
            stock_columns = []
            for block, coordinate in enumerate((1, 0, 2, 3)):
                angles[coordinate, block * 64 : (block + 1) * 64] = frequency[0::2]
                stock_columns.append([block * 128 + 2 * k for k in range(64)])
            sin_columns = [column for block in stock_columns for column in block]
            permutation = sin_columns + [column + 1 for column in sin_columns]
            self.register_buffer("angles", angles)
            self.ref_head = Mlp(decoder.ref_point_head, decoder.ref_point_head.layers[0].weight[:, permutation])
            # 8 * (grid coordinate - box edge) for every (grid line, edge) pair, as one affine map of the
            # reference box instead of a coordinate column broadcast against the edges. Each edge sums at
            # most two exact products and 8 is a power of two, so the map is exact.
            edges = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]])
            coordinates = torch.arange(GRID, dtype=torch.float32) / GRID
            for axis, columns in (("x", edges[:, 0::2]), ("y", edges[:, 1::2])):
                self.register_buffer(f"{axis}_offset_weight", (-8 * columns).repeat(1, GRID))
                self.register_buffer(f"{axis}_offset_bias", (8 * coordinates).repeat_interleave(2))
            expand = torch.zeros(2, GRID, TOKENS)
            for row in range(GRID):
                for column in range(GRID):
                    expand[0, row, row * GRID + column] = 1
                    expand[1, column, row * GRID + column] = 1
            self.register_buffer("expand_y", expand[0].clone())
            self.register_buffer("expand_x", expand[1].clone())
            self.register_buffer("presence_bias", torch.zeros(1, heads, 1, TOKENS))
            self.register_buffer("presence_pos", torch.zeros(1, 1, 1, 256))

        def box_bias(self, reference):
            offsets_y = torch.matmul(reference, self.y_offset_weight) + self.y_offset_bias
            offsets_x = torch.matmul(reference, self.x_offset_weight) + self.x_offset_bias
            bias_y = self.rpb_y(log_scale(offsets_y.reshape(1, QUERIES, GRID, 2))).permute(0, 3, 1, 2)
            bias_x = self.rpb_x(log_scale(offsets_x.reshape(1, QUERIES, GRID, 2))).permute(0, 3, 1, 2)
            bias = torch.matmul(bias_y, self.expand_y) + torch.matmul(bias_x, self.expand_x)
            return torch.cat([self.presence_bias, bias], 2)

        def forward(self, memory, pos, prompt, prompt_bias):
            key, target, presence, reference = memory + pos, self.query, self.presence, self.reference
            for layer in self.layers:
                angles = torch.matmul(reference, self.angles)
                query_pos = self.ref_head(torch.cat([torch.sin(angles), torch.cos(angles)], -1))
                t = layer(
                    torch.cat([presence, target], 2),
                    torch.cat([self.presence_pos, query_pos], 2),
                    prompt,
                    prompt_bias,
                    key,
                    memory,
                    self.box_bias(reference),
                )
                presence, target = t[:, :, :1], t[:, :, 1:]
                states = self.norm(target)
                # sigmoid(delta + inverse_sigmoid(reference)) with the stock eps, without Log.
                delta = self.box_head(states)
                inside = reference.clamp(1e-3, 1) * torch.sigmoid(delta)
                reference = inside / (inside + (1 - reference).clamp(1e-3, 1) * torch.sigmoid(-delta))
            return states, reference, self.presence_head(self.presence_norm(presence))

    class Scoring(nn.Module):
        def __init__(self, scoring):
            super().__init__()
            mlp = scoring.prompt_mlp
            if not mlp.residual or mlp.num_layers != 2 or not scoring.clamp_logits:
                raise ValueError("Unexpected scoring configuration")
            self.fc1, self.fc2, self.norm = linear(mlp.layers[0]), linear(mlp.layers[1]), Norm(mlp.out_norm)
            # The logit scale is a power of two, so folding it into the projection is exact.
            self.prompt_proj, self.hs_proj = linear(scoring.prompt_proj, scoring.scale), linear(scoring.hs_proj)
            self.limit = scoring.clamp_max_val

        def forward(self, states, prompt, valid):
            prompt = self.norm(prompt + self.fc2(F.relu(self.fc1(prompt))))
            count = valid.sum(-1, keepdim=True).clamp(1, valid.shape[-1])
            pooled = self.prompt_proj(torch.einsum("bnqk,bnkc->bnqc", valid, prompt) / count)
            logits = torch.einsum("bnqc,bnkc->bnqk", self.hs_proj(states), pooled)
            return logits.clamp(-self.limit, self.limit)

    class GroupNorm(nn.Module):
        """GroupNorm from 1x1 convs and global pooling: one conv averages each group's channels, another copies a
        group's value back to its channels, so no tensor is split or reshaped per group."""

        def __init__(self, norm):
            super().__init__()
            groups, channels = norm.num_groups, norm.num_channels
            size = channels // groups
            members = (torch.arange(channels) // size == torch.arange(groups)[:, None]).float()
            self.register_buffer("average", (members / size).reshape(groups, channels, 1, 1))
            self.register_buffer("expand", members.t().reshape(channels, groups, 1, 1))
            self.weight = constant(norm.weight.reshape(1, -1, 1, 1))
            self.bias, self.eps = constant(norm.bias.reshape(1, -1, 1, 1)), norm.eps

        def forward(self, x):
            centered = x - F.conv2d(F.adaptive_avg_pool2d(F.conv2d(x, self.average), 1), self.expand)
            variance = F.adaptive_avg_pool2d(F.conv2d(centered * centered, self.average), 1)
            return centered * (F.conv2d(torch.rsqrt(variance + self.eps), self.expand) * self.weight) + self.bias

    class Segmentation(nn.Module):
        def __init__(self, head):
            super().__init__()
            decoder = head.pixel_decoder
            if decoder.interpolation_mode != "nearest" or decoder.shared_conv or head.no_dec:
                raise ValueError("Unexpected segmentation head")
            self.norm, self.attn = Norm(head.cross_attn_norm), Attention(head.cross_attend_prompt)
            self.conv = nn.ModuleList(copy.deepcopy(decoder.conv_layers[index]) for index in range(2))
            self.group = nn.ModuleList(GroupNorm(decoder.norms[index]) for index in range(2))
            # (W p + b) . m = p . (W^T m) + b . m: the instance head's 1x1 conv moves from the 36,864 pixels to the
            # 200 queries, whose last column carries b . m.
            self.mask_embed = Mlp(head.mask_predictor.mask_embed)
            instance, last = head.instance_seg_head, self.mask_embed.layers[-1]
            weight, bias = instance.weight[:, :, 0, 0], instance.bias
            last.weight = constant(torch.cat([weight.t() @ last.weight, (bias @ last.weight)[None]]))
            last.bias = constant(torch.cat([weight.t() @ last.bias, (bias @ last.bias)[None]]))

        def forward(self, memory, prompt, prompt_bias, features, states):
            x = memory + self.attn(self.norm(memory), prompt, prompt, prompt_bias)
            x = x.reshape(1, GRID, GRID, 256).permute(0, 3, 1, 2)
            for conv, norm, feature in zip(self.conv, self.group, features):
                x = feature + F.interpolate(x, size=feature.shape[-2:], mode="nearest")
                x = F.relu(norm(conv(x)))
            queries = self.mask_embed(states)
            # Channels first on both sides, so the product writes the (queries, height, width) output directly.
            masks = torch.einsum("ndhw,ndhq->nqhw", x, queries[..., :256].permute(0, 3, 1, 2))
            return masks + queries[..., 256:].reshape(1, QUERIES, 1, 1)

    class Graph(nn.Module):
        def __init__(self, model):
            super().__init__()
            vision, geometry = model.backbone.vision_backbone, model.geometry_encoder
            encoder = model.transformer.encoder
            if encoder.level_embed is not None or encoder.add_pooled_text_to_img_feat:
                raise ValueError("Unexpected encoder fusion")
            self.text = TextEncoder(model.backbone.language_backbone)
            self.trunk, self.neck = ImageTrunk(vision.trunk.model), Neck(vision)
            pos = vision.position_encoding(torch.zeros(1, 256, GRID, GRID))
            self.register_buffer("pos", pos.permute(0, 2, 3, 1).reshape(1, 1, TOKENS, 256).clone())
            cls = geometry.norm(geometry.final_proj(geometry.cls_embed.weight))
            self.register_buffer("cls", cls.detach().reshape(1, 1, 1, 256).clone())
            self.geometry = nn.ModuleList(EncoderLayer(layer, True) for layer in geometry.encode)
            self.geometry_norm = Norm(geometry.encode_norm)
            self.encoder = nn.ModuleList(EncoderLayer(layer, False) for layer in encoder.layers)
            self.decoder, self.scoring = Decoder(model.transformer.decoder), Scoring(model.dot_prod_scoring)
            self.segmentation = Segmentation(model.segmentation_head)
            corners = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]) * IMAGE_SIZE
            self.register_buffer("corners", corners)
            self.register_buffer("one", torch.ones(1, 1, 1, 1))

        def image(self, pixel_values, text):
            scale4, scale2, scale1 = self.neck(self.trunk(pixel_values))
            memory = scale1.permute(0, 2, 3, 1).reshape(1, 1, TOKENS, 256)
            key, g = memory + self.pos, self.cls
            for layer in self.geometry:
                g = layer(g, None, key, memory)
            prompt = torch.cat([text[:, :CONTEXT].reshape(1, 1, CONTEXT, 256), self.geometry_norm(g)], 2)
            valid = torch.cat([text[:, CONTEXT:, :CONTEXT].reshape(1, 1, 1, CONTEXT), self.one], -1)
            # Padding keys get -1e4, whose softmax weight is exactly zero in float32.
            prompt_bias = (valid - 1) * 1e4
            x = memory
            for layer in self.encoder:
                x = layer(x, self.pos, prompt, prompt, prompt_bias)
            states, reference, presence = self.decoder(x, self.pos, prompt, prompt_bias)
            score = torch.sigmoid(self.scoring(states, prompt, valid)) * torch.sigmoid(presence)
            boxes = torch.matmul(reference, self.corners)
            detections = torch.cat([boxes, score, torch.zeros_like(score)], -1).reshape(1, QUERIES, 6)
            return detections, self.segmentation(x, prompt, prompt_bias, (scale2, scale4), states)

        def forward(self, token_one_hot, pixel_values, text):
            return (self.text(token_one_hot), *self.image(pixel_values, text))

    return Graph(stock).eval()


def stock_outputs(stock, pixels, prompt):
    import torch
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage

    backbone = stock.backbone.forward_image(torch.from_numpy(pixels))
    backbone.update(stock.backbone.forward_text([prompt], device="cpu"))
    find = FindStage(
        img_ids=torch.tensor([0]),
        text_ids=torch.tensor([0]),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    # The geometry encoder pins a host tensor for a CUDA copy; pinning has no numerical effect.
    pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda tensor, *args, **kwargs: tensor
    try:
        out = stock.forward_grounding(
            backbone_out=backbone, find_input=find, find_target=None, geometric_prompt=stock._get_dummy_prompt()
        )
    finally:
        torch.Tensor.pin_memory = pin_memory
    scores = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1))[0, :, 0]
    return {
        "text_features": backbone["language_features"][:CONTEXT, 0].numpy(),
        "scores": scores.numpy(),
        "boxes_px": (box_ops.box_cxcywh_to_xyxy(out["pred_boxes"])[0] * IMAGE_SIZE).numpy(),
        "mask_logits": out["pred_masks"][0].numpy(),
    }


def synthetic_pixels(seed):
    image = np.random.RandomState(seed).randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return np.ascontiguousarray((image.astype(np.float32) / np.float32(127.5) - 1).transpose(2, 0, 1)[None])


def export(graph, output):
    import torch

    prompt_tokens = tokens(CASES[0][1])
    example = (
        torch.from_numpy(one_hot(prompt_tokens)),
        torch.from_numpy(synthetic_pixels(CASES[0][0])),
        torch.from_numpy(packed(np.zeros((CONTEXT, FEATURES), np.float32), prompt_tokens)),
    )
    torch.onnx.export(
        graph,
        example,
        str(output),
        input_names=list(INPUTS),
        output_names=list(OUTPUTS),
        opset_version=17,
        dynamo=False,
    )
    fold_constants(output)
    validate_export(output)


def check_against_stock(stock, output):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    empty_text = np.zeros(INTERFACE["text"], np.float32)
    for seed, prompt in CASES:
        prompt_tokens = tokens(prompt)
        pixels = synthetic_pixels(seed)
        expected = stock_outputs(stock, pixels, prompt)
        # Feed the ONNX text features back as an application does.
        features = session.run(
            ["text_features"], {"token_one_hot": one_hot(prompt_tokens), "pixel_values": pixels, "text": empty_text}
        )[0]
        text_features, detections, masks = session.run(
            None,
            {"token_one_hot": one_hot(prompt_tokens), "pixel_values": pixels, "text": packed(features, prompt_tokens)},
        )
        if np.abs(detections[0, :, 5]).max() != 0:
            raise ValueError("Detection column 5 must be zero")
        actual = {
            "text_features": text_features[0],
            "scores": detections[0, :, 4],
            "boxes_px": detections[0, :, :4],
            "mask_logits": masks[0],
        }
        errors = compare_outputs(expected, actual, len(prompt_tokens))
        print(f"Seed {seed}, prompt {prompt!r}: " + ", ".join(f"{name} {value:.2e}" for name, value in errors.items()))


def convert(source, output_dir):
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory() as work_dir:
        unpack_package(source.parent / WHEEL, Path(work_dir))
        stock = build_stock(source, source.parent / BACKBONE, check_tokenizer(source.parent, Path(work_dir)))
        export(build_graph(stock), output)
        check_against_stock(stock, output)
    print(f"Synthetic-input PyTorch/ONNX smoke check passed: {output}")
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
