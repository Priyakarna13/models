"""Export Mask2Former Swin-T (COCO instance) as its three components side by side in one ONNX, and check it against
transformers.

The checkpoint is facebook/mask2former-swin-tiny-coco-instance, built with transformers' Mask2Former. Each exported
component is mask2former_4d's rearrangement of transformers' modules into the layout the compiler maps to the MLA,
for a fixed 384x384 input (pipeline.py). Each component is checked against its transformers counterpart on seeded
images before export, then each component's ONNX against PyTorch.

The masked attention masks a key by the sign of a mask logit, so a logit within float rounding of 0 can be masked
in one implementation and not another (different operation order, thread counts or CPUs round differently), and
through self-attention move every query's logits. The transformer decoder's checks therefore take mask features
scaled by MASK_SCALE: every mask logit scales with them, which makes such a logit MASK_SCALE times rarer, while the
masked path is still compared.

Loading reports two expected differences from transformers' current Swin, both checked: the checkpoint lacks the
Swin classifier's final LayerNorm, which the backbone never uses (it normalizes each feature map with its own
hidden_states_norms), and holds relative_position_index buffers that transformers now rebuilds itself.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from pipeline import BACKBONE_SHAPES, COMPONENTS, IMAGE_SIZE, INPUT_SHAPES, OUTPUT_SHAPES, SCALE_SIZES, tensor_name

MODEL_DIR = Path(__file__).resolve().parents[1]

FILES = ("config.json", "model.safetensors")
EXPORTER = "5.16.1"
OPSET = 17
SEEDS = range(123, 126)
MASK_SCALE = 100.0
SWIN = "model.pixel_level_module.encoder.swin"
EXPECTED_MISSING = {f"{SWIN}.layernorm.weight", f"{SWIN}.layernorm.bias"}


def validate_source(source):
    files = {item["path"] for item in source.get("files", [])}
    if source.get("type") != "file_set" or source.get("entrypoint") != "config.json" or files != set(FILES):
        raise ValueError(f"the source must be Mask2Former's {', '.join(FILES)}, opened at config.json")


def build(directory):
    """transformers' Mask2Former from the checkpoint, with only the expected loading differences."""
    import transformers
    from transformers import Mask2FormerForUniversalSegmentation

    if transformers.__version__ != EXPORTER:
        raise RuntimeError(f"Expected transformers {EXPORTER}, found {transformers.__version__}")
    model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
        directory, local_files_only=True, output_loading_info=True
    )
    unexpected = {key for key in info["unexpected_keys"] if not key.endswith(".relative_position_index")}
    if set(info["missing_keys"]) != EXPECTED_MISSING or unexpected or info["mismatched_keys"]:
        raise ValueError(f"unexpected checkpoint differences: {info}")
    if model.config.num_labels != 80 or model.config.num_queries != 100:
        raise ValueError("expected the 80-class COCO checkpoint with 100 queries")
    return model.eval()


def modules(model):
    """The side-by-side export module."""
    import torch
    from mask2former_4d import PixelDecoder4D, Swin4D, TransformerDecoder4D

    pixel = model.model.pixel_level_module

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Swin4D(pixel.encoder, IMAGE_SIZE)
            self.pixel_decoder = PixelDecoder4D(pixel.decoder, [(shape[1], shape[2]) for shape in BACKBONE_SHAPES])
            self.transformer_decoder = TransformerDecoder4D(
                model.model.transformer_module, model.class_predictor, list(SCALE_SIZES)
            )

        def forward(self, image, f0, f1, f2, f3, mask_features, s0, s1, s2):
            return (
                *self.backbone(image),
                *self.pixel_decoder(f0, f1, f2, f3),
                *self.transformer_decoder(mask_features, s0, s1, s2),
            )

    return SideBySide().eval()


def seeded_image(seed):
    """A smooth seeded image in the normalized range."""
    import torch

    noise = torch.from_numpy(np.random.RandomState(seed).uniform(-2.1, 2.6, (1, 3, 24, 24)).astype(np.float32))
    return torch.nn.functional.interpolate(noise, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear").numpy()


def check_parts(model, side_by_side):
    """Each component gives its transformers counterpart's outputs on seeded images, fed transformers' outputs of
    the component before it: the backbone's feature maps, the pixel decoder's mask and multi-scale features, and
    the transformer decoder's class and mask logits (with mask features scaled by MASK_SCALE)."""
    import torch

    pixel = model.model.pixel_level_module
    for seed in SEEDS:
        image = torch.from_numpy(seeded_image(seed))
        with torch.inference_mode():
            features = pixel.encoder(image).feature_maps
            for index, (want, got) in enumerate(zip(features, side_by_side.backbone(image))):
                compare(want.numpy(), got.numpy(), f"backbone features{index} (seed {seed})")
            decoded = pixel.decoder(list(features))
            expected = (decoded.mask_features, *decoded.multi_scale_features)
            for name, want, got in zip(OUTPUT_SHAPES["pixel_decoder"], expected, side_by_side.pixel_decoder(*features)):
                compare(want.numpy(), got.numpy(), f"pixel decoder {name} (seed {seed})")
            mask_features = decoded.mask_features * MASK_SCALE
            reference = model.model.transformer_module(list(decoded.multi_scale_features), mask_features)
            classes = model.class_predictor(reference.intermediate_hidden_states[-1].transpose(0, 1))
            got_classes, got_masks = side_by_side.transformer_decoder(mask_features, *decoded.multi_scale_features)
        compare_decoder(classes.numpy(), got_classes[0, :, 0].T[None].numpy(), f"class logits (seed {seed})")
        compare_masks(reference.masks_queries_logits[-1].numpy(), got_masks.numpy(), f"mask logits (seed {seed})")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"Mask2Former export requires opset {OPSET}")
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

    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded Mask2Former export")
    onnx.save(model, path)


def compare(expected, actual, name):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg=name)


def compare_decoder(expected, actual, name, signs=False):
    """The transformer decoder's logits, which the masked attention makes discontinuous (see the module docstring):
    a rare flipped mask decision moves about one query's logits. So all but 2% of them, two queries' worth, must
    agree within compare's tolerance (1e-3 absolute plus 1e-3 relative), and with signs, mask logits must agree in
    sign at nearly every pixel. A real export error moves far more."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    agreement = float(np.mean((expected > 0) == (actual > 0))) if signs else 1.0
    outside = float(np.mean(np.abs(actual - expected) > 1e-3 + 1e-3 * np.abs(expected)))
    if agreement < 0.9999 or outside > 0.02:
        raise ValueError(f"{name}: signs agree at {agreement:.4%} of values, {outside:.4%} outside the tolerance")


def compare_masks(expected, actual, name):
    compare_decoder(expected, actual, name, signs=True)


def check_onnx(side_by_side, path):
    """Each component's ONNX outputs match PyTorch's, each component fed the previous one's PyTorch outputs of the
    same seeded image; the transformer decoder takes those mask features scaled by MASK_SCALE (see the module
    docstring)."""
    import onnxruntime as ort
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = [tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]]
    for seed in SEEDS[:2]:
        image = torch.from_numpy(seeded_image(seed))
        with torch.inference_mode():
            features = side_by_side.backbone(image)
            mask_features, *scales = side_by_side.pixel_decoder(*features)
            inputs = (image, *features, mask_features * MASK_SCALE, *scales)
            expected = side_by_side(*inputs)
        feeds = {name: value.numpy() for name, value in zip(session_inputs(), inputs)}
        for name, want, got in zip(outputs, expected, session.run(outputs, feeds)):
            checks = {
                tensor_name("transformer_decoder", "class_logits"): compare_decoder,
                tensor_name("transformer_decoder", "mask_logits"): compare_masks,
            }
            checks.get(name, compare)(want.numpy(), got, f"{name} (seed {seed})")


def session_inputs():
    return [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]


def convert(source, output_dir):
    import torch

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    model = build(source.parent)
    side_by_side = modules(model)
    check_parts(model, side_by_side)
    with tempfile.TemporaryDirectory(prefix="mask2former-") as temporary:
        exported = Path(temporary) / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                side_by_side,
                tuple(torch.zeros(INPUT_SHAPES[c][n]) for c in COMPONENTS for n in INPUT_SHAPES[c]),
                str(exported),
                input_names=session_inputs(),
                output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
        fold_constants(exported)
        validate_export(exported)
        shutil.move(exported, output)
    check_onnx(side_by_side, output)
    seeds = f"{SEEDS.start}-{SEEDS.stop - 1}"
    print(f"4D components and PyTorch/ONNX checks passed with seeds {seeds}: {output}")
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
