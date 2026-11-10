"""Split EfficientSAM3 into its text encoder and image model for BF16 compilation.

The compiler lowers Sin, Cos and Log through BF16 lookup tables. They occur only in the decoder:
Sin and Cos build the sine query embedding from reference boxes (angles in [0, 2*pi]); Log builds the box
relative-position bias from log(1 + relu(t)) with |t| <= 12. Each becomes a polynomial with a stated float32
error bound over its clipped domain, and Log adds Newton steps through Exp. The text encoder is cut out
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from calibration import CONTEXT, FEATURES, IMAGE_SIZE, VOCABULARY, one_hot, packed, tokens
from onnx import helper, numpy_helper

QUERIES = 200
MASK_SIZE = 192

TEXT_INPUT, TEXT_OUTPUT = "token_one_hot", "text_features"
IMAGE_INPUTS, IMAGE_OUTPUTS = ["pixel_values", "text"], ["detections", "mask_logits"]
INTERFACE = {
    "token_one_hot": [1, CONTEXT, VOCABULARY],
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "text": [1, CONTEXT + 1, FEATURES],
    "text_features": [1, CONTEXT, FEATURES],
    "detections": [1, QUERIES, 6],
    "mask_logits": [1, QUERIES, MASK_SIZE, MASK_SIZE],
}
# Data-dependent uses in the folded export; onnxsim folds the first decoder layer's.
EXPECTED = {"Sin": 5, "Cos": 5, "Log": 20}

# Near-minimax float64 fits on the domains below, evaluated by Horner's rule.
# sin(theta) = t * A(t^2) and cos(theta) = B(t^2) with t = clip(theta, 0, 2*pi) - pi.
SIN = [
    -0.9999999992796773, 0.16666665931858382, -0.008333321350174647, 0.00019840536491520168,
    -2.7535897092468157e-06, 2.4729237242536458e-08, -1.361453851749336e-10,
]  # fmt: skip
COS = [
    -0.9999999963716656, 0.499999998694325, -0.04166666312537827, 0.0013888859079996924,
    -2.4800407827067865e-05, 2.753229457456087e-07, -2.0583100355451864e-09, 9.658306828471712e-12,
]  # fmt: skip
# log(w) for w = clip(w, 1, 16): y = C(2w/15 - 17/15), then y <- y - 1 + w * exp(-y).
LOG = [
    2.1492067526874274, 0.9237985730862641, -0.587449189721419, -0.047279953972127316,
    0.5108554787988003, 0.498664101231361, -0.6863134993307177,
]  # fmt: skip
NEWTON_STEPS = 3
DOMAINS = {"Sin": (0.0, 2 * np.pi), "Cos": (0.0, 2 * np.pi), "Log": (1.0, 16.0)}

# The replacements' largest float32 error over their domains, which tests/build/test_compile.py checks on a dense
# grid. The text encoder is a pure cut. The replacements sit below float32 rounding, which the refinement layers
# amplify in rejected queries, so boxes and masks are bounded relative to their scale over all queries and
# absolutely over confident ones.
FUNCTION_BOUND = 2e-6
TEXT_TOLERANCE = 1e-5
IMAGE_TOLERANCE = {
    "scores": 1e-4,
    "boxes_relative": 1e-3,
    "masks_relative": 1e-3,
    "boxes_px_confident": 0.1,
    "mask_logits_confident": 0.05,
}
CONFIDENT = 0.05


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported EfficientSAM3 graph: {message}")


def _interface(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = (*model.graph.input, *model.graph.output)
    return {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim] for value in values}


class _Builder:
    """Emit replacement nodes with unique names and scalar float32 constants."""

    def __init__(self, model: onnx.ModelProto):
        self.model, self.nodes, self.count = model, [], 0

    def name(self, stem):
        self.count += 1
        return f"surgery/{stem}_{self.count}"

    def constant(self, value):
        name = self.name("constant")
        self.model.graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), name))
        return name

    def op(self, op_type, *inputs, output=None):
        output = output or self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [output], name=self.name(op_type)))
        return output

    def horner(self, coefficients, x, output=None):
        value = self.op("Add", self.op("Mul", x, self.constant(coefficients[-1])), self.constant(coefficients[-2]))
        for index, coefficient in enumerate(reversed(coefficients[:-2])):
            last = index == len(coefficients) - 3
            value = self.op(
                "Add", self.op("Mul", value, x), self.constant(coefficient), output=output if last else None
            )
        return value

    def reduced_angle(self, theta):
        low, high = DOMAINS["Sin"]
        t = self.op("Sub", self.op("Clip", theta, self.constant(low), self.constant(high)), self.constant(np.pi))
        return t, self.op("Mul", t, t)

    def log(self, w, output):
        low, high = DOMAINS["Log"]
        w = self.op("Clip", w, self.constant(low), self.constant(high))
        z = self.op("Add", self.op("Mul", w, self.constant(2 / 15)), self.constant(-17 / 15))
        y = self.horner(LOG, z)
        for step in range(NEWTON_STEPS):
            correction = self.op("Mul", w, self.op("Exp", self.op("Neg", y)))
            y = self.op(
                "Add",
                self.op("Add", y, self.constant(-1.0)),
                correction,
                output=output if step == NEWTON_STEPS - 1 else None,
            )
        return y


def function_inputs(model: onnx.ModelProto) -> dict[str, list[str]]:
    """Return the input names of the Sin, Cos and Log nodes, by operator."""
    return {op: [node.input[0] for node in model.graph.node if node.op_type == op] for op in EXPECTED}


def replace_functions(model: onnx.ModelProto, expected=EXPECTED) -> None:
    """Replace Sin, Cos and Log in place by their polynomial forms."""
    counts = {op: sum(node.op_type == op for node in model.graph.node) for op in expected}
    _require(counts == expected, f"expected {expected} data-dependent functions, found {counts}")
    builder, nodes, angles = _Builder(model), [], {}
    for node in model.graph.node:
        if node.op_type not in expected:
            nodes.append(node)
            continue
        (source,), (target,) = node.input, node.output
        builder.nodes = []
        if node.op_type == "Log":
            builder.log(source, target)
        else:
            # Sin and Cos of one angle tensor share the reduction.
            if source not in angles:
                angles[source] = builder.reduced_angle(source)
            t, u = angles[source]
            if node.op_type == "Sin":
                builder.op("Mul", t, builder.horner(SIN, u), output=target)
            else:
                builder.horner(COS, u, output=target)
        nodes.extend(builder.nodes)
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def graph_paths(output_dir: Path) -> tuple[Path, Path]:
    """Return where prepare_model writes the text encoder and image model graphs."""
    return tuple(output_dir / f"efficientsam3_repvit_m_{part}.onnx" for part in ("text_encoder", "image_model"))


def prepare_model(source: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full = onnx.load(source)
    onnx.checker.check_model(full)
    _require(_interface(full) == INTERFACE, f"interface {_interface(full)}")
    # convert.py folds the export with onnxsim; the function counts assume a folded graph.
    unfolded = sorted({node.op_type for node in full.graph.node} & {"Constant", "Shape"})
    _require(not unfolded, f"source still has {unfolded} nodes; fold it with onnxsim first")
    replace_functions(full)
    del full.graph.value_info[:]
    full = onnx.shape_inference.infer_shapes(full, strict_mode=True)
    full.ir_version = 8
    onnx.checker.check_model(full)
    text_path, image_path = graph_paths(output_dir)
    full_path = output_dir / "efficientsam3_repvit_m_full.onnx"
    onnx.save(full, full_path)
    del full
    try:
        onnx.utils.extract_model(str(full_path), str(text_path), [TEXT_INPUT], [TEXT_OUTPUT])
        onnx.utils.extract_model(str(full_path), str(image_path), IMAGE_INPUTS, IMAGE_OUTPUTS)
    finally:
        full_path.unlink()
    for path in (text_path, image_path):
        model = onnx.load(path)
        model.ir_version = 8
        left = {node.op_type for node in model.graph.node} & set(EXPECTED)
        _require(not left, f"{path.name} still has {sorted(left)}")
        onnx.checker.check_model(model)
        onnx.save(model, path)
    return text_path, image_path


def image_differences(expected, actual) -> dict[str, float]:
    (detections, masks), (actual_detections, actual_masks) = expected, actual
    scores = detections[0, :, 4].astype(np.float64)
    boxes = np.abs(detections[0, :, :4].astype(np.float64) - actual_detections[0, :, :4]).max(1)
    mask = np.abs(masks[0].astype(np.float64) - actual_masks[0]).reshape(QUERIES, -1).max(1)
    confident = scores >= CONFIDENT
    return {
        "scores": float(np.abs(scores - actual_detections[0, :, 4]).max()),
        "boxes_relative": float(boxes.max() / IMAGE_SIZE),
        "masks_relative": float(mask.max() / np.abs(masks).max()),
        "boxes_px_confident": float(boxes[confident].max(initial=0.0)),
        "mask_logits_confident": float(mask[confident].max(initial=0.0)),
    }


def validate_float_split(
    source: Path, text_path: Path, image_path: Path, images: np.ndarray, prompts
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compare both components with the converted graph on real images and prompts.

    Return the largest errors and the converted graph's image outputs, stacked over the images.
    """
    import onnxruntime as ort

    failures = []
    options = {"providers": ["CPUExecutionProvider"]}
    reference = ort.InferenceSession(str(source), **options)
    replaced = function_inputs(onnx.load(source, load_external_data=False))
    text_session = ort.InferenceSession(str(text_path), **options)
    # Also return the replaced functions' inputs to check that real values stay inside their domains.
    probes = sorted({name for names in replaced.values() for name in names})
    image = onnx.load(image_path)
    image.graph.output.extend(helper.make_empty_tensor_value_info(name) for name in probes)
    image_session = ort.InferenceSession(image.SerializeToString(), **options)
    del image
    ranges = {op: [np.inf, -np.inf] for op in replaced}
    largest = {"text_features": 0.0, **dict.fromkeys(IMAGE_TOLERANCE, 0.0)}
    empty = np.zeros(INTERFACE["text"], np.float32)
    outputs = {name: [] for name in IMAGE_OUTPUTS}
    for index, (pixels, prompt) in enumerate(zip(images, prompts, strict=True)):
        prompt_tokens = tokens(prompt)
        encoded = one_hot(prompt_tokens)
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        feeds = {TEXT_INPUT: encoded, "pixel_values": pixels, "text": empty}
        expected_text = reference.run([TEXT_OUTPUT], feeds)[0]
        (features,) = text_session.run(None, {TEXT_INPUT: encoded})
        text = packed(expected_text, prompt_tokens)
        expected = reference.run(IMAGE_OUTPUTS, {**feeds, "text": text})
        actual = image_session.run(None, {"pixel_values": pixels, "text": text})
        for name, value in zip(IMAGE_OUTPUTS, expected, strict=True):
            outputs[name].append(value[0])
        values = dict(zip(probes, actual[2:], strict=True))
        for op, names in replaced.items():
            ranges[op][0] = min(ranges[op][0], *(float(values[name].min()) for name in names))
            ranges[op][1] = max(ranges[op][1], *(float(values[name].max()) for name in names))
        case = {
            "text_features": float(np.abs(features.astype(np.float64) - expected_text).max()),
            **image_differences(expected, actual[:2]),
        }
        largest = {name: max(largest[name], value) for name, value in case.items()}
        limits = {"text_features": TEXT_TOLERANCE, **IMAGE_TOLERANCE}
        failures += [
            f"{index}/{prompt}/{name} {case[name]:.2e}" for name, limit in limits.items() if not case[name] <= limit
        ]
        if np.abs(actual[0][0, :, 5]).max() != 0:
            failures.append(f"{index}/{prompt}/detection column 5 is not zero")
    for op, (low, high) in ranges.items():
        if not (DOMAINS[op][0] <= low and high <= DOMAINS[op][1]):
            failures.append(f"{op} inputs [{low:.4g}, {high:.4g}] leave the domain {DOMAINS[op]}")
    if failures:
        raise RuntimeError(f"EfficientSAM3 split differs from the converted graph: {failures}")
    report = {"observed_inputs": ranges, **largest}
    return report, {name: np.stack(values) for name, values in outputs.items()}
