"""Split a validated static D-FINE export at its CPU proposal-selection boundary.

Shape freezing and the proposal split preserve numerical operations. The bounded
reference-logit rewrite below is a measured approximation, checked separately.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

# Decoder width and feature strides, not the encoder width (X uses 384 there).
SIZES = {
    "n": (128, (16, 32)),
    "s": (256, (8, 16, 32)),
    "m": (256, (8, 16, 32)),
    "l": (256, (8, 16, 32)),
    "x": (256, (8, 16, 32)),
}
OUTPUTS = {"pred_boxes": (1, 300, 4), "pred_logits": (1, 300, 80)}
INDEX_ROUTING = {"Identity", "Unsqueeze", "Reshape", "Expand", "Tile"}


def require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported D-FINE graph: {message}")


def geometry(model_id):
    parts = model_id.split("_")
    require(len(parts) in (2, 3) and parts[0] == "dfine" and parts[1] in SIZES, "unknown family identity")
    require(len(parts) == 2 or parts[2] == "discrete", "unknown sampling variant")
    width, strides = SIZES[parts[1]]
    levels = tuple((640 // stride) ** 2 for stride in strides)
    return width, levels


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def shape(value):
    dims = value.type.tensor_type.shape.dim
    return tuple(d.dim_value for d in dims) if all(d.HasField("dim_value") for d in dims) else None


def index(model):
    values = {v.name: v for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    producers = {name: node for node in model.graph.node for name in node.output}
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant" and "value" in attrs(node):
            constants[node.output[0]] = numpy_helper.to_array(attrs(node)["value"])
    return values, producers, constants


def validate_source(model, model_id):
    geometry(model_id)
    metadata = {p.key: p.value for p in model.metadata_props}
    require(metadata.get("model_id") == model_id, "model identity mismatch")
    require(
        metadata.get("sampling") == ("discrete" if model_id.endswith("_discrete") else "default"),
        "sampling identity mismatch",
    )
    require(
        metadata.get("box_format") == "normalized_cxcywh" and metadata.get("class_output") == "logits",
        "unexpected public output semantics",
    )
    require([(x.domain, x.version) for x in model.opset_import] == [("", 17)], "expected ONNX opset 17")
    require(
        {x.name: shape(x) for x in model.graph.input} == {"input": (1, 3, 640, 640)},
        "expected static batch-one 640 RGB input",
    )
    require(
        [(x.name, shape(x)) for x in model.graph.output] == list(OUTPUTS.items()),
        "unexpected public output shapes/order",
    )
    require(
        all(x.type.tensor_type.elem_type == TensorProto.FLOAT for x in (*model.graph.input, *model.graph.output)),
        "expected float32 public I/O",
    )
    onnx.checker.check_model(model)


def infer_and_freeze_shapes(model):
    """Use existing symbolic inference without merging/guessing unknown dimensions."""
    source = deepcopy(model)
    operations = [x.SerializeToString() for x in source.graph.node]
    weights = [x.SerializeToString() for x in source.graph.initializer]
    inferred = SymbolicShapeInference.infer_shapes(source, auto_merge=False, guess_output_rank=False)
    require(operations == [x.SerializeToString() for x in inferred.graph.node], "shape inference changed operations")
    require(weights == [x.SerializeToString() for x in inferred.graph.initializer], "shape inference changed weights")
    values, _, _ = index(inferred)
    frozen = []
    for node in inferred.graph.node:
        if node.op_type != "Shape" or node.input[0] not in values:
            continue
        dimensions = shape(values[node.input[0]])
        if dimensions is None:
            continue
        attributes = attrs(node)
        dimensions = dimensions[attributes.get("start", 0) : attributes.get("end", len(dimensions))]
        output, name = node.output[0], node.name
        node.CopyFrom(
            helper.make_node(
                "Constant", [], [output], name=name, value=numpy_helper.from_array(np.asarray(dimensions, np.int64))
            )
        )
        frozen.append(output)
    return inferred, frozen


def ancestors(value, producers, stops=()):
    seen, pending = set(), [value]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in stops and name in producers:
            pending.extend(producers[name].input)
    return seen


def follows_indices(value, root, producers):
    while value != root:
        node = producers.get(value)
        if node is None or node.op_type not in INDEX_ROUTING:
            return False
        value = node.input[0]
    return True


def rename(model, mapping):
    for node in model.graph.node:
        for fields in (node.input, node.output):
            for i, name in enumerate(fields):
                fields[i] = mapping.get(name, name)
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value.name = mapping.get(value.name, value.name)


def split_model(model, model_id):
    """Return backbone, decoder and evidence of the matched proposal contract."""
    validate_source(model, model_id)
    width, levels = geometry(model_id)
    tokens = sum(levels)
    model, frozen = infer_and_freeze_shapes(model)
    values, producers, constants = index(model)

    def matches(name, dimensions, dtype=TensorProto.FLOAT):
        return name in values and shape(values[name]) == dimensions and values[name].type.tensor_type.elem_type == dtype

    topks = [
        n
        for n in model.graph.node
        if n.op_type == "TopK"
        and n.input[1] in constants
        and constants[n.input[1]].size == 1
        and constants[n.input[1]].item() == 300
    ]
    require(len(topks) == 1, "expected one Top300 proposal selector")
    topk = topks[0]
    require(
        attrs(topk).get("axis", -1) == 1 and attrs(topk).get("largest", 1) == 1 and attrs(topk).get("sorted", 1) == 1,
        "unexpected Top300 attributes",
    )
    scores, indices = topk.input[0], topk.output[1]
    require(
        matches(scores, (1, tokens)) and matches(indices, (1, 300), TensorProto.INT64),
        "unexpected proposal count or Top300 index type",
    )
    gathers = [
        n for n in model.graph.node if n.op_type == "GatherElements" and follows_indices(n.input[1], indices, producers)
    ]
    require(len(gathers) == 2, "expected exactly two Top300 gathers")
    by_width = {}
    for node in gathers:
        require(attrs(node).get("axis", 0) == 1, "unexpected proposal gather axis")
        dimensions = shape(values[node.input[0]]) if node.input[0] in values else None
        require(dimensions in ((1, tokens, width), (1, tokens, 4)), "unexpected gathered tensor shape")
        size = dimensions[-1]
        require(size not in by_width and matches(node.output[0], (1, 300, size)), "ambiguous gathered content")
        require(matches(node.input[1], (1, 300, size), TensorProto.INT64), "unexpected gather index layout")
        by_width[size] = node
    content, references = by_width[width], by_width[4]
    scorer = producers.get(scores)
    require(
        scorer is not None
        and scorer.op_type == "ReduceMax"
        and attrs(scorer).get("axes") in ([-1], [2])
        and attrs(scorer).get("keepdims", 1) == 0
        and matches(scorer.input[0], (1, tokens, 80)),
        "expected maximum raw class logit",
    )
    reference_add = producers.get(references.input[0])
    require(
        reference_add is not None and reference_add.op_type == "Add",
        "expected unactivated bbox-plus-anchor reference logits",
    )
    require(
        sum(content.input[0] in ancestors(x, producers) for x in reference_add.input) == 1,
        "expected one content-dependent bbox delta and one independent anchor",
    )
    memory_candidates = [
        n
        for n in model.graph.node
        if n.op_type == "Concat"
        and attrs(n).get("axis") == 1
        and matches(n.output[0], (1, tokens, width))
        and [shape(values[x]) if x in values else None for x in n.input] == [(1, size, width) for size in levels]
    ]
    require(len(memory_candidates) == 1, "expected one ordered multi-level attention memory")
    memory = memory_candidates[0].output[0]
    require(
        memory != content.input[0] and memory in ancestors(content.input[0], producers),
        "query content must be transformed from attention memory",
    )
    require(
        producers.get(content.input[0]) is not None and producers[content.input[0]].op_type == "LayerNormalization",
        "expected transformed query normalization",
    )
    require(content.input[0] in ancestors(scores, producers), "proposal scores do not use query content")
    require(
        content.input[0] in ancestors(references.input[0], producers), "proposal references do not use query content"
    )

    # Prove the permitted routing copies each selected index across its last axis.
    # Shape freezing removes all model-data dependencies from this tiny integer graph.
    index_graph = onnx.utils.Extractor(model).extract_model([indices], [n.input[1] for n in gathers])
    require([x.name for x in index_graph.graph.input] == [indices], "index expansion depends on model data")
    expected_indices = np.arange(300, dtype=np.int64)[None]
    expanded = ReferenceEvaluator(index_graph).run(None, {indices: expected_indices})
    for node, actual in zip(gathers, expanded, strict=True):
        expected = np.broadcast_to(expected_indices[..., None], (1, 300, shape(values[node.input[0]])[-1]))
        require(np.array_equal(actual, expected), "index routing changes proposal order or values")

    backbone_outputs = [memory, content.input[0], scores, references.input[0]]
    decoder_inputs = [memory, content.output[0], references.output[0]]
    needed = set().union(*(ancestors(name, producers, decoder_inputs) for name in OUTPUTS))
    require("input" not in needed, "decoder retains a backbone dependency")
    backbone = onnx.utils.Extractor(model).extract_model(["input"], backbone_outputs)
    decoder = onnx.utils.Extractor(model).extract_model(decoder_inputs, list(OUTPUTS))
    require(all(name in needed for name in decoder_inputs), "unused decoder boundary")
    require(not any(n.op_type == "GatherElements" for n in decoder.graph.node), "unresolved dynamic gather")
    require(not any(n.op_type == "TopK" for n in backbone.graph.node), "proposal selection remained in backbone")
    rename(
        backbone,
        dict(
            zip(
                backbone_outputs,
                ("attention_memory", "query_content", "proposal_scores", "reference_logits"),
                strict=True,
            )
        ),
    )
    rename(decoder, dict(zip(decoder_inputs, ("attention_memory", "query_content", "reference_logits"), strict=True)))
    for graph in (backbone, decoder):
        onnx.checker.check_model(graph, full_check=True)
    return (
        backbone,
        decoder,
        {
            "model_id": model_id,
            "tokens": tokens,
            "width": width,
            "frozen_shapes": frozen,
            "source_boundaries": backbone_outputs,
            "source_indices": indices,
        },
    )


def bounded_reference_logits(model):
    """Replace the decoder's inverse-sigmoid reference chain, not arbitrary Log ops.

    AFE lowers Log to a calibrated INT8 lookup even under a BF16 policy. Its wide
    positive input range loses useful reference values. For this checked chain,
    clamp the original logits to +/- -log(eps). FP32 sigmoid saturation makes this
    an approximation, not an algebraically exact floating-point cancellation.
    """
    model = deepcopy(model)
    _, producers, constants = index(model)
    logs = [n for n in model.graph.node if n.op_type == "Log"]
    require(len(logs) == 1, "expected one reference inverse-sigmoid Log")
    log = logs[0]
    division = producers.get(log.input[0])
    require(division is not None and division.op_type == "Div", "expected inverse-sigmoid ratio")
    clips = [producers.get(x) for x in division.input]
    require(all(n is not None and n.op_type == "Clip" for n in clips), "expected clamped ratio operands")
    epsilon = np.float32(1e-5)
    for node in clips:
        require(
            len(node.input) >= 2
            and node.input[1] in constants
            and np.asarray(constants[node.input[1]]).size == 1
            and np.asarray(constants[node.input[1]]).item() == epsilon,
            "unexpected inverse-sigmoid epsilon",
        )
        if len(node.input) > 2 and node.input[2]:
            require(
                node.input[2] in constants and np.asarray(constants[node.input[2]]).item() == 1,
                "unexpected inverse-sigmoid upper bound",
            )
    sigmoid_clip = producers.get(clips[0].input[0])
    complement = producers.get(clips[1].input[0])
    require(sigmoid_clip is not None and sigmoid_clip.op_type == "Clip", "expected bounded sigmoid")
    require(
        complement is not None
        and complement.op_type == "Sub"
        and complement.input[1] == sigmoid_clip.output[0]
        and complement.input[0] in constants
        and np.asarray(constants[complement.input[0]]).item() == 1,
        "unexpected inverse-sigmoid complement",
    )
    require(
        len(sigmoid_clip.input) == 3
        and all(x in constants for x in sigmoid_clip.input[1:])
        and [np.asarray(constants[x]).item() for x in sigmoid_clip.input[1:]] == [0, 1],
        "unexpected sigmoid clamp",
    )
    sigmoid = producers.get(sigmoid_clip.input[0])
    require(
        sigmoid is not None and sigmoid.op_type == "Sigmoid" and list(sigmoid.input) == ["reference_logits"],
        "Log is not the reference-logit chain",
    )
    names = {name for n in model.graph.node for name in (*n.input, *n.output)} | set(constants)
    lower, upper = "dfine_reference_logit_min", "dfine_reference_logit_max"
    require(lower not in names and upper not in names, "reference bound name collision")
    bound = np.float32(-np.log(epsilon))
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(-bound, np.float32), lower),
            numpy_helper.from_array(np.array(bound, np.float32), upper),
        ]
    )
    replacement = helper.make_node(
        "Clip", ["reference_logits", lower, upper], list(log.output), name="bounded_reference_logits"
    )
    log.CopyFrom(replacement)
    # Prune only dead ancestors. Keep the exact public three-input/two-output ABI.
    model = onnx.utils.Extractor(model).extract_model(
        [v.name for v in model.graph.input], [v.name for v in model.graph.output]
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def prepare_model(source, destination):
    """Write the two components; compatibility constraints are Nano/default only."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    backbone, decoder, _ = split_model(onnx.load(source), "dfine_n")
    values, _, _ = index(decoder)
    samples = [node for node in decoder.graph.node if node.op_type == "GridSample"]
    require(len(samples) == 48, "expected 48 per-head Nano attention samples")
    for node in samples:
        dimensions = shape(values[node.input[0]]) if node.input[0] in values else None
        require(dimensions in ((1, 16, 20, 20), (1, 16, 40, 40)), "unexpected attention geometry")
        policy = attrs(node)
        require(
            policy.get("mode", b"bilinear") == b"bilinear"
            and policy.get("padding_mode", b"zeros") == b"zeros"
            and policy.get("align_corners", 0) == 0,
            "unexpected attention interpolation",
        )
    decoder = bounded_reference_logits(decoder)
    paths = [destination / f"{name}.onnx" for name in ("backbone", "decoder")]
    for model, path in zip((backbone, decoder), paths, strict=True):
        onnx.save(model, path)
    return tuple(paths)
