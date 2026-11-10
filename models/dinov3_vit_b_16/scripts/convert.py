"""Convert the onnx-community export of DINOv3 ViT-B/16 into one ONNX file at its shipped input size, and check it
against that export.

onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX exports Meta's weights for any input size, at opset 20, with the
weights in a separate file. The conversion fixes the input at 1x3x224x224 and saves the graph and its weights as one
file; the graph and its pooler_output and last_hidden_state outputs stay as exported. The changes the compiler needs
happen in surgery.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

MODEL_TYPE = "dinov3_vit"
IMAGE_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
REGISTERS = 4
PREFIX = 1 + REGISTERS
OPSET = 20
INPUT = "pixel_values"
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
OUTPUT_SHAPES = {"last_hidden_state": [1, PREFIX + PATCHES, HIDDEN_SIZE], "pooler_output": [1, HIDDEN_SIZE]}
SOURCE_FILES = ["onnx/model.onnx", "onnx/model.onnx_data", "config.json", "preprocessor_config.json"]
SOURCE = ("file_set", "onnx/model.onnx", SOURCE_FILES)
SEEDS = range(123, 127)
# The preprocessing calibration.py and performance.py apply. The conversion stops if the checkpoint declares other.
PREPROCESSING = {
    "do_resize": True,
    "size": {"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    "resample": 2,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = {key: config.get(key) for key in ("model_type", "image_size", "patch_size", "hidden_size", "num_register_tokens")}
    expected = {"model_type": MODEL_TYPE, "image_size": IMAGE_SIZE, "patch_size": PATCH_SIZE, "hidden_size": HIDDEN_SIZE,
                "num_register_tokens": REGISTERS}
    if declared != expected:
        raise ValueError(f"config.json declares {declared}, expected {expected}")


def validate_preprocessor(directory):
    declared = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
    if declared.get("do_center_crop"):
        raise ValueError("preprocessor_config.json crops the image; the scripts resize it straight to 224x224")
    for key, expected in PREPROCESSING.items():
        value = declared.get(key)
        same = np.allclose(value, expected) if isinstance(expected, (float, list)) and value is not None else value == expected
        if not same:
            raise ValueError(f"preprocessor_config.json declares {key}={value!r}, the scripts apply {expected!r}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"The DINOv3 export is at opset {OPSET}")
    names = ([value.name for value in model.graph.input], [value.name for value in model.graph.output])
    if names != ([INPUT], list(OUTPUT_SHAPES)):
        raise ValueError(f"Expected input {INPUT} and outputs {tuple(OUTPUT_SHAPES)} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Unexpected dtype for {value.name}")
    if [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] != INPUT_SHAPE:
        raise ValueError(f"The input must be fixed at {INPUT_SHAPE}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fix_input_shape(source, output):
    """Fix the export's input at the shipped size and save the graph with its weights as one file."""
    model = onnx.load(str(source))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if model.graph.input[0].name != INPUT or len(dims) != len(INPUT_SHAPE):
        raise ValueError(f"Expected a rank-{len(INPUT_SHAPE)} {INPUT} input")
    for dim, size in zip(dims, INPUT_SHAPE, strict=True):
        dim.Clear()
        dim.dim_value = size
    onnx.save(model, str(output))


def _synthetic(seed):
    return np.random.RandomState(seed).standard_normal(INPUT_SHAPE).astype(np.float32)


def convert(source, output_dir):
    import onnxruntime as ort

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    root = source.parents[1]
    validate_config(root)
    validate_preprocessor(root)
    original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    expected = [original.run(list(OUTPUT_SHAPES), {INPUT: _synthetic(seed)}) for seed in SEEDS]
    del original
    fix_input_shape(source, output)
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for seed, want in zip(SEEDS, expected, strict=True):
        got = session.run(list(OUTPUT_SHAPES), {INPUT: _synthetic(seed)})
        for name, a, b in zip(OUTPUT_SHAPES, got, want, strict=True):
            if a.shape != tuple(OUTPUT_SHAPES[name]):
                raise ValueError(f"{name} has shape {a.shape}, expected {OUTPUT_SHAPES[name]}")
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5, err_msg=name)
    print(f"Source/converted ONNX check passed on {len(SEEDS)} synthetic inputs: {output}")
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
