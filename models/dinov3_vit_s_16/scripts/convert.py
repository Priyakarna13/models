"""Convert the onnx-community export of DINOv3 ViT-S/16 into a static ONNX graph with its class and patch embeddings
as outputs, and check it against that export.

onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX exports Meta's weights for any input size, at opset 20. Fixing the
input at 224x224 lets constant folding turn the RoPE cos/sin computation into tables for the 14x14 patch grid. Each
layer still splits the five prefix tokens (class and registers) from the 196 patch tokens with SplitToSequence and
SequenceAt; those become one Split with the same constant sizes, after which the reshape targets fold too. Each exact
Gelu becomes its erf form, so the graph needs only opset 17. The outputs are the pooled class embedding and the patch
tokens; the four register tokens, which DINOv3 uses only inside the encoder, are dropped. Each folding pass runs in
its own process, so the family's largest graph, ViT-L/16, fits in 9 GiB.
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from onnx import helper, numpy_helper

MODEL_DIR = Path(__file__).resolve().parents[1]

MODEL_TYPE = "dinov3_vit"
IMAGE_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 384
PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
REGISTERS = 4
PREFIX = 1 + REGISTERS
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "class_embedding": [1, HIDDEN_SIZE],
    "patch_embeddings": [1, PATCHES, HIDDEN_SIZE],
}
INPUTS = ("pixel_values",)
OUTPUTS = ("class_embedding", "patch_embeddings")
SOURCE_OUTPUTS = ("pooler_output", "last_hidden_state")
SOURCE_FILES = ["onnx/model.onnx", "onnx/model.onnx_data", "config.json", "preprocessor_config.json"]
SOURCE = ("file_set", "onnx/model.onnx", SOURCE_FILES)
SEEDS = range(123, 127)
# Operators the converted graph must not keep: dynamic shapes, control flow, sequences, and opset-20 Gelu.
DYNAMIC_OPS = ("If", "Loop", "Shape", "Range", "SplitToSequence", "SequenceAt", "Gelu")
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
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The DINOv3 export requires opset 17")
    if [value.name for value in model.graph.input] != list(INPUTS) or [value.name for value in model.graph.output] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    left = sorted({node.op_type for node in model.graph.node if node.op_type in DYNAMIC_OPS})
    if left:
        raise ValueError(f"The export must have a static opset-17 graph; it still has {', '.join(left)}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(source, output):
    """Fix the input size and fold every constant expression, then save one self-contained file."""
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(str(source)),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in INPUTS},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded DINOv3 graph")
    onnx.save(model, str(output))


def _constant(graph, name):
    for tensor in graph.initializer:
        if tensor.name == name:
            return numpy_helper.to_array(tensor)
    raise ValueError(f"{name} is not a constant; fold the graph first")


def _rename_uses(graph, old, new):
    for node in graph.node:
        node.input[:] = [new if name == old else name for name in node.input]


def split_prefix_tokens(graph):
    """Replace each SplitToSequence and the SequenceAt nodes that read it with one Split of the same sizes."""
    parts, renamed = {}, {}
    for node in graph.node:
        if node.op_type == "SplitToSequence":
            sizes = _constant(graph, node.input[1]).reshape(-1)
            parts[node.output[0]] = [f"{node.output[0]}_part{index}" for index in range(len(sizes))]
        elif node.op_type == "SequenceAt":
            renamed[node.output[0]] = parts[node.input[0]][int(_constant(graph, node.input[1]))]
        elif any(name in parts for name in node.input):
            raise ValueError(f"{node.name} reads a token sequence; only SequenceAt is rewritten")
    nodes = []
    for node in graph.node:
        if node.op_type == "SequenceAt":
            continue
        if node.op_type == "SplitToSequence":
            axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
            node = helper.make_node("Split", node.input[:2], parts[node.output[0]], name=f"{node.name}_split", axis=axis)
        node.input[:] = [renamed.get(name, name) for name in node.input]
        nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(nodes)


def gelu_as_erf(graph):
    """Rewrite each exact Gelu as 0.5 * x * (1 + erf(x / sqrt(2))), which opset 17 provides."""
    graph.initializer.extend([
        numpy_helper.from_array(np.array(np.sqrt(2.0), dtype=np.float32), "gelu_sqrt2"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "gelu_one"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "gelu_half"),
    ])
    nodes = []
    for node in graph.node:
        if node.op_type != "Gelu":
            nodes.append(node)
            continue
        if any(a.name == "approximate" and a.s not in (b"", b"none") for a in node.attribute):
            raise ValueError(f"{node.name} is a tanh-approximated Gelu; only the exact form is rewritten")
        x, y, stem = node.input[0], node.output[0], node.name or node.output[0]
        nodes += [
            helper.make_node("Div", [x, "gelu_sqrt2"], [f"{stem}_scaled"], name=f"{stem}_div"),
            helper.make_node("Erf", [f"{stem}_scaled"], [f"{stem}_erf"], name=f"{stem}_erf"),
            helper.make_node("Add", [f"{stem}_erf", "gelu_one"], [f"{stem}_shifted"], name=f"{stem}_add"),
            helper.make_node("Mul", [x, f"{stem}_shifted"], [f"{stem}_gated"], name=f"{stem}_mul"),
            helper.make_node("Mul", [f"{stem}_gated", "gelu_half"], [y], name=f"{stem}_half"),
        ]
    graph.ClearField("node")
    graph.node.extend(nodes)


def select_outputs(graph):
    """Expose the pooled class embedding and the patch tokens, without the class and register tokens."""
    pooled, hidden = SOURCE_OUTPUTS
    for node in graph.node:
        node.output[:] = [OUTPUTS[0] if name == pooled else name for name in node.output]
    _rename_uses(graph, pooled, OUTPUTS[0])
    graph.initializer.extend([
        numpy_helper.from_array(np.array([PREFIX], dtype=np.int64), "patch_start"),
        numpy_helper.from_array(np.array([PREFIX + PATCHES], dtype=np.int64), "patch_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "patch_axis"),
    ])
    graph.node.append(helper.make_node("Slice", [hidden, "patch_start", "patch_end", "patch_axis"], [OUTPUTS[1]],
                                       name="patch_tokens"))
    graph.ClearField("output")
    graph.output.extend(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, INTERFACE[name]) for name in OUTPUTS)


def make_static(path):
    model = onnx.load(str(path))
    split_prefix_tokens(model.graph)
    gelu_as_erf(model.graph)
    select_outputs(model.graph)
    model.ClearField("opset_import")
    model.opset_import.extend([helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def _in_own_process(function, *args):
    process = multiprocessing.get_context("spawn").Process(target=function, args=args)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{function.__name__} failed with exit code {process.exitcode}")


def _synthetic(seed):
    return np.random.RandomState(seed).standard_normal(INTERFACE["pixel_values"]).astype(np.float32)


def convert(source, output_dir):
    import onnxruntime as ort

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    root = source.parents[1]
    validate_config(root)
    validate_preprocessor(root)
    # The expected outputs come from the unchanged export, so the check covers folding and every rewrite.
    original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    expected = []
    for seed in SEEDS:
        pooled, hidden = original.run(list(SOURCE_OUTPUTS), {"pixel_values": _synthetic(seed)})
        expected.append((pooled, hidden[:, PREFIX:]))
    del original
    _in_own_process(fold_constants, source, output)
    _in_own_process(make_static, output)
    _in_own_process(fold_constants, output, output)
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for seed, want in zip(SEEDS, expected, strict=True):
        got = session.run(list(OUTPUTS), {"pixel_values": _synthetic(seed)})
        for name, a, b in zip(OUTPUTS, got, want, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3, err_msg=name)
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
