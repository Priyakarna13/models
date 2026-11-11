from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper


def _source_graph(case: str = "valid") -> onnx.ModelProto:
    """The converted interface: three channel Pads and a spatial Pad on the kept outputs, two outputs to drop."""
    rng = np.random.RandomState(0)
    nodes = [helper.make_node("Transpose", ["input_1"], ["nchw"], perm=[0, 3, 1, 2])]
    initializers = []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def pad(name, pads, **attributes):
        nodes.append(
            helper.make_node("Pad", ["nchw", constant(f"{name}_pads", np.array(pads, np.int64))], [name], **attributes)
        )

    def head(name, value, channels, size):
        nodes.append(helper.make_node("GlobalAveragePool", [value], [f"{name}_pool"]))
        nodes.append(helper.make_node("Flatten", [f"{name}_pool"], [f"{name}_flat"]))
        weight = constant(f"{name}_weight", rng.uniform(-1, 1, (channels, size)).astype(np.float32))
        nodes.append(helper.make_node("MatMul", [f"{name}_flat", weight], [f"{name}_out"]))
        return f"{name}_out"

    added = [1, 2, 0 if case == "two_channel_pads" else 3]
    for index, count in enumerate(added):
        pad(
            f"channel_{index}",
            [0, 0, 0, 0, 0, count, 0, 0],
            **({"mode": "reflect"} if case == "reflect" and index == 0 else {}),
        )
    pad("spatial", [0, 0, 1, 1, 0, 0, 1, 1])
    landmarks = head("landmarks", "channel_0", 4, 195)
    nodes.append(helper.make_node("Identity", [landmarks], ["Identity"]))
    nodes.append(helper.make_node("Sigmoid", [head("presence", "channel_1", 5, 1)], ["Identity_1"]))
    world = [head("world", "channel_2", 3 + added[2], 117), head("spatial_world", "spatial", 3, 117)]
    nodes.append(helper.make_node("Add", world, ["Identity_4"]))
    nodes.append(helper.make_node("ReduceMean", ["input_1"], ["Identity_2"], axes=[3]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["Identity_3"], value=numpy_helper.from_array(np.zeros((1, 64, 64, 39), np.float32))
        )
    )
    input_shape = [1, 256, 256, 4] if case == "input_shape" else [1, 256, 256, 3]
    outputs = {
        "Identity": [1, 195],
        "Identity_1": [1, 1],
        "Identity_2": [1, 256, 256, 1],
        "Identity_3": [1, 64, 64, 39],
        "Identity_4": [1, 117],
    }
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in outputs.items()],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=8)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    onnx.save(_source_graph(), path)
    return path


def test_surgery_keeps_three_outputs_and_turns_channel_pads_into_identity_convolutions(
    tmp_path: Path, source: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")

    prepared = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in prepared.graph.output}
    assert list(outputs) == surgery.OUTPUTS == ["Identity", "Identity_1", "Identity_4"]
    assert outputs == {"Identity": [1, 195], "Identity_1": [1, 1], "Identity_4": [1, 117]}
    assert not {"ReduceMean", "Constant"} & {node.op_type for node in prepared.graph.node}
    weights = {value.name: numpy_helper.to_array(value) for value in prepared.graph.initializer}
    convolutions = [node for node in prepared.graph.node if node.op_type == "Conv"]
    assert [weights[node.input[1]].shape for node in convolutions] == [(4, 3, 1, 1), (5, 3, 1, 1), (6, 3, 1, 1)]
    for node in convolutions:
        weight = weights[node.input[1]][:, :, 0, 0]
        np.testing.assert_array_equal(weight, np.eye(*weight.shape, dtype=np.float32))
    (spatial,) = [node for node in prepared.graph.node if node.op_type == "Pad"]
    assert weights[spatial.input[1]].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


@pytest.mark.parametrize(
    ("case", "match"),
    [("two_channel_pads", "found 2 channel Pads"), ("reflect", "does not pad with zeros"), ("input_shape", "inputs")],
)
def test_surgery_rejects_an_unexpected_graph(
    tmp_path: Path, case: str, match: str, load_script: Callable[[str], ModuleType]
) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_source_graph(case), path)

    with pytest.raises(ValueError, match=match):
        load_script("surgery").prepare_model(path, tmp_path / "out")


def test_rewrite_check_rejects_a_graph_that_changes_an_output(
    tmp_path: Path, source: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    prepared = onnx.load(surgery.prepare_model(source, tmp_path / "out"))
    weight = next(value for value in prepared.graph.initializer if value.name.endswith("/identity_weight"))
    weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight) * 1.01, weight.name))
    changed = tmp_path / "changed.onnx"
    onnx.save(prepared, changed)

    with pytest.raises(RuntimeError, match="prepared output Identity differs"):
        surgery.validate_rewrite(source, changed)


class Quantized:
    """Stands in for the compiler's quantized model: execute returns the source's outputs plus an offset each."""

    def __init__(self, source: Path, offsets: dict[str, float]) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        self.offsets = offsets

    def execute(self, inputs, use_jax):
        (sample,) = inputs.values()
        values = self.session.run(list(self.offsets), {"input_1": sample})
        return [value + offset for value, offset in zip(values, self.offsets.values())]


@pytest.mark.parametrize("case", ["valid", "world_landmarks", "nan"])
def test_quantized_check_bounds_each_output_against_the_converted_graph(
    case: str, source: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    offsets = {name: bound / 2 for name, bound in recipe.MAX_MEAN_ERROR.items()}
    if case == "valid":
        errors = recipe.check_quantized_outputs(source, Quantized(source, offsets))
        assert errors == pytest.approx(offsets, rel=1e-3)
        return
    offsets["Identity_4"] = 0.1 if case == "world_landmarks" else np.nan
    with pytest.raises(RuntimeError, match="Identity_4"):
        recipe.check_quantized_outputs(source, Quantized(source, offsets))


def _mpk_manifest(recipe: ModuleType) -> dict:
    """The plugin chain of the compiled MPK: a cast of the frame, the MLA, then slices and casts per output."""
    return {
        "input_nodes": [{"name": "input_1", "size": 1 * 256 * 256 * 3 * 4}],
        "plugins": [
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 2, "processor": "MLA", "config_params": {}},
            {"sequence": 3, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {
                "sequence": 4,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in recipe.OUTPUT_BYTES],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_mla", "host", "input", "reordered"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    manifest = _mpk_manifest(recipe)
    plugins = manifest["plugins"]
    if case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65", "config_params": {}})
    elif case == "input":
        manifest["input_nodes"][0]["size"] //= 2
    elif case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[2] = outputs[2], outputs[0]
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        recipe.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(archive)


@pytest.mark.parametrize("profile_id", ["w16a16"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    assert load_script("compile")._parameters(profile_id) == {"precision": profile_id}


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w8a8")


def test_bf16_quantization_skips_calibration(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    recipe._quantization("w16a16")

    assert recipe.QuantizationParams.call_args.kwargs == {
        "calibration_method": recipe.SkipCalibration.return_value,
        "activation_quantization_scheme": recipe.bfloat16_scheme.return_value,
        "weight_quantization_scheme": recipe.bfloat16_scheme.return_value,
    }
    with pytest.raises(ValueError, match="unsupported precision"):
        recipe._quantization("w8a8")


def test_compile_quantizes_without_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "prepared.onnx")
    monkeypatch.setattr(recipe, "check_quantized_outputs", lambda source, quantized: {})

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", "w16a16")

    assert recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"] is None


def test_compiled_build_holds_the_audited_mpk(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    assert archive.name == f"blazepose_ghum_heavy_{build_profile}_mpk.tar.gz"
    load_script("compile").audit_mpk(archive)
