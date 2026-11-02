"""Precision-profile, graph-rewrite and installed artifact contracts for Nano."""

from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from model_registry.model import load_model
from PIL import Image


@pytest.fixture
def compiler(load_script):
    return load_script("compile")


def test_only_registered_profiles_are_admitted_and_decoder_stays_w16a16(compiler, model_dir):
    metadata = load_model(model_dir)
    parameters = compiler.component_parameters(metadata, "w16a16_w16a16")
    assert list(parameters) == ["backbone", "decoder"]
    assert all(
        p == {"precision": "w16a16", "calibration_bins": 2048, "calibration_samples": 50} for p in parameters.values()
    )
    with pytest.raises(ValueError):
        compiler.component_parameters(metadata, "w8a8_w16a16")
    for key, value in [("precision", "w8a8"), ("calibration_bins", 0), ("calibration_samples", True)]:
        changed = copy.deepcopy(metadata)
        changed["build_profiles"][0]["components"][0]["parameters"][key] = value
        with pytest.raises(ValueError):
            compiler.component_parameters(changed, "w16a16_w16a16")


def test_both_quantization_schemes_are_bf16_without_unmeasured_option_changes(compiler, monkeypatch):
    calls = []

    class Config:
        def with_activation_quantization(self, value):
            calls.append(("activation", value))
            return self

        def with_weight_quantization(self, value):
            calls.append(("weight", value))
            return self

        def with_calibration(self, value):
            calls.append(("bins", value))
            return self

    monkeypatch.setattr(compiler, "default_quantization", Config())
    monkeypatch.setattr(compiler, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(compiler, "HistogramMSEMethod", lambda bins: bins)
    compiler.quantization_config({"precision": "w16a16", "calibration_bins": 2048})
    assert calls == [("activation", "bf16"), ("weight", "bf16"), ("bins", 2048)]


@pytest.mark.parametrize("profile", ["w8a16_w16a16", "w8a8_w16a16", "w8a8_mixed_w16a16"])
def test_candidate_profile_parameters_do_not_change_the_decoder(compiler, model_dir, profile):
    metadata = copy.deepcopy(load_model(model_dir))
    candidate = copy.deepcopy(metadata["build_profiles"][0])
    candidate["id"] = profile
    candidate["components"][0]["parameters"]["precision"] = compiler.PROFILES[profile]
    if "mixed" in profile:
        candidate["components"][0]["parameters"]["precision_map"] = {"graph_sha256": "0" * 64, "nodes": {}}
    metadata["build_profiles"].append(candidate)
    parameters = compiler.component_parameters(metadata, profile)
    assert parameters["decoder"] == metadata["build_profiles"][0]["components"][1]["parameters"]
    candidate["components"][1]["parameters"]["precision"] = "w8a8"
    with pytest.raises(ValueError, match="decoder"):
        compiler.component_parameters(metadata, profile)


@pytest.mark.parametrize("precision", ["w8a16", "w8a8"])
def test_int8_policies_keep_a16_defaults_and_only_enable_measured_int8_correction(compiler, monkeypatch, precision):
    calls = []

    class Config:
        def __getattr__(self, name):
            def apply(value):
                calls.append((name, value))
                return self

            return apply

    monkeypatch.setattr(compiler, "default_quantization", Config())
    monkeypatch.setattr(compiler, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(compiler, "quantization_scheme", lambda **kwargs: kwargs)
    monkeypatch.setattr(compiler, "HistogramMSEMethod", lambda bins: bins)
    monkeypatch.setattr(compiler, "BiasCorrectionType", SimpleNamespace(REGULAR="regular"))
    compiler.quantization_config({"precision": precision, "calibration_bins": 1024})
    assert calls[1] == ("with_weight_quantization", {"asymmetric": False, "per_channel": True, "bits": 8})
    assert calls[2] == ("with_calibration", 1024)
    if precision == "w8a8":
        assert calls[0][1] == {"asymmetric": True, "per_channel": False, "bits": 8}
        assert calls[3:] == [("with_bias_correction", "regular"), ("with_per_token_quant", False)]
    else:
        assert calls[0] == ("with_activation_quantization", "bf16") and len(calls) == 3


def precision_graph(tmp_path):
    h = onnx.helper

    def value(name):
        return h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])

    model = h.make_model(
        h.make_graph(
            [
                h.make_node("Relu", ["input"], ["hidden"], name="first"),
                h.make_node("Relu", ["hidden"], ["output"], name="last"),
            ],
            "precision",
            [value("input")],
            [value("output")],
        ),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    model.graph.value_info.append(value("hidden"))
    onnx.helper.set_model_props(model, {"contract": "preserved"})
    graph = tmp_path / "backbone.onnx"
    onnx.save(model, graph)
    from model_registry.fetch import sha256_file

    return graph, {"graph_sha256": sha256_file(graph)[0], "nodes": {"first": "int8", "last": "bfloat16"}}


@pytest.mark.parametrize("case", ["hash", "unknown", "dtype", "not_mixed", "missing", "duplicate"])
def test_precision_maps_cannot_silently_drift_or_promote_everything(compiler, tmp_path, case):
    graph, precision_map = precision_graph(tmp_path)
    original = graph.read_bytes()
    if case == "hash":
        precision_map["graph_sha256"] = "0" * 64
    elif case == "unknown":
        precision_map["nodes"]["typo"] = "int8"
    elif case == "dtype":
        precision_map["nodes"]["first"] = "fp32"
    elif case == "not_mixed":
        precision_map["nodes"]["first"] = "bfloat16"
    elif case == "missing":
        del precision_map["nodes"]
    else:
        model = onnx.load(graph)
        model.graph.node[1].name = "first"
        onnx.save(model, graph)
        from model_registry.fetch import sha256_file

        precision_map["graph_sha256"] = sha256_file(graph)[0]
        original = graph.read_bytes()
    with pytest.raises(ValueError):
        compiler.annotate_backbone(graph, precision_map)
    assert graph.read_bytes() == original


def test_mixed_uses_bundled_exporter_without_changing_model_metadata(compiler, tmp_path, monkeypatch):
    graph, precision_map = precision_graph(tmp_path)
    seen = []

    def annotate_graph(model_graph, nodes):
        seen.append(nodes)
        # The bundled helper rebuilds GraphProto and drops its internal value_info.
        return onnx.helper.make_graph(model_graph.node, model_graph.name, model_graph.input, model_graph.output)

    monkeypatch.setitem(
        sys.modules, "neural_compressor.annotate_precision", SimpleNamespace(annotate_graph=annotate_graph)
    )
    compiler.annotate_backbone(graph, precision_map)
    assert seen == [precision_map["nodes"]]
    model = onnx.load(graph)
    assert dict((entry.key, entry.value) for entry in model.metadata_props) == {"contract": "preserved"}
    assert [v.name for v in model.graph.input] == ["input"]
    assert [v.name for v in model.graph.output] == ["output"]
    assert [v.name for v in model.graph.value_info] == ["hidden"]


def test_decoder_calibration_uses_owned_outputs_from_the_selected_backbone(compiler, tmp_path, monkeypatch):
    image = np.zeros((1, 3, 640, 640), np.float32)
    tensors = [
        np.full((1, 2000, 128), 2, np.float32),
        np.full((1, 2000, 128), 3, np.float32),
        np.arange(2000, dtype=np.float32)[None],
        np.full((1, 2000, 4), 4, np.float32),
    ]
    chosen = object()
    calls = []
    monkeypatch.setattr(compiler, "load_images", lambda *args: image)
    monkeypatch.setattr(compiler, "quantize_component", lambda *args: calls.append(args) or chosen)

    def execute(model, pixels):
        assert model is chosen
        return tensors

    def compile_component(*args):
        # Simulate AFE recycling the execution storage after it has been saved.
        for tensor in tensors:
            tensor.fill(-99)

    monkeypatch.setattr(compiler, "execute_backbone", execute)
    monkeypatch.setattr(compiler, "compile_component", compile_component)
    parameters = {"precision": "w8a16", "calibration_bins": 2048, "calibration_samples": 1}
    compiler.backbone_phase("w8a16_w16a16", parameters, tmp_path, ("cal",), ("eval",), tmp_path)
    assert calls[0][1] == "dfine_n_w8a16_w16a16_backbone"
    assert calls[0][2] is parameters
    with np.load(tmp_path / "decoder-calibration.npz", allow_pickle=False) as values:
        assert np.all(values["attention_memory"] == 2)
        assert np.all(values["query_content"] == 3)
        assert np.all(values["reference_logits"] == 4)


def test_calibration_is_rgb_over_255_and_selection_is_disjoint(load_script, tmp_path):
    calibration = load_script("calibration")
    for i in range(8):
        Image.new("RGB", (4, 2), (255, 128, 0)).save(tmp_path / f"{i}.jpg")
    first = calibration.select_images(tmp_path, 42, 0, 3)
    next_ = calibration.select_images(tmp_path, 42, 3, 2)
    assert not set(first) & set(next_)
    assert first + next_ == calibration.select_images(tmp_path, 42, 0, 5)
    stream = io.BytesIO()
    Image.new("RGB", (4, 2), (255, 128, 0)).save(stream, format="PNG")
    pixels = calibration.preprocess_image(stream.getvalue())
    assert pixels.shape == (3, 640, 640) and pixels.dtype == np.float32
    np.testing.assert_array_equal(pixels[:, 0, 0], np.array([1, 128 / 255, 0], np.float32))


def test_query_bridge_preserves_memory_and_stable_owned_rows(load_script):
    calibration = load_script("calibration")
    memory = np.ones((1, 2000, 128), np.float32)
    content = np.arange(2000 * 128, dtype=np.float32).reshape(1, 2000, 128)
    scores = np.zeros((1, 2000), np.float32)
    refs = np.arange(2000 * 4, dtype=np.float32).reshape(1, 2000, 4)
    forwarded, selected, references = calibration.decoder_inputs([memory, content, scores, refs])
    assert forwarded is memory
    np.testing.assert_array_equal(selected, content[:, :300])
    np.testing.assert_array_equal(references, refs[:, :300])
    assert not np.shares_memory(selected, content) and not np.shares_memory(references, refs)


def reference_graph():
    h, n = onnx.helper, onnx.numpy_helper
    constants = [
        n.from_array(np.array(value, np.float32), name) for name, value in [("zero", 0), ("one", 1), ("eps", 1e-5)]
    ]
    nodes = [
        h.make_node("Sigmoid", ["reference_logits"], ["sigmoid"]),
        h.make_node("Clip", ["sigmoid", "zero", "one"], ["bounded"]),
        h.make_node("Sub", ["one", "bounded"], ["complement"]),
        h.make_node("Clip", ["bounded", "eps"], ["numerator"]),
        h.make_node("Clip", ["complement", "eps"], ["denominator"]),
        h.make_node("Div", ["numerator", "denominator"], ["ratio"]),
        h.make_node("Log", ["ratio"], ["result"]),
    ]

    def value(name):
        return h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 300, 4])

    graph = h.make_graph(nodes, "reference_chain", [value("reference_logits")], [value("result")], constants)
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=8)


def test_reference_rewrite_is_narrow_and_retains_the_tensor_contract(load_script):
    surgery = load_script("surgery")
    original = reference_graph()
    candidate = surgery.bounded_reference_logits(original)
    assert [v.SerializeToString() for v in candidate.graph.input] == [
        v.SerializeToString() for v in original.graph.input
    ]
    assert [v.SerializeToString() for v in candidate.graph.output] == [
        v.SerializeToString() for v in original.graph.output
    ]
    assert len(candidate.graph.node) == 1 and candidate.graph.node[0].op_type == "Clip"
    assert any(n.op_type == "Log" for n in original.graph.node), "caller graph was mutated"


@pytest.mark.parametrize("case", ["epsilon", "upper", "complement", "unrelated", "two_logs"])
def test_reference_rewrite_rejects_other_log_semantics(load_script, case):
    surgery = load_script("surgery")
    graph = reference_graph()
    if case == "epsilon":
        graph.graph.initializer[2].CopyFrom(onnx.numpy_helper.from_array(np.array(1e-4, np.float32), "eps"))
    elif case == "upper":
        graph.graph.node[3].input.append("eps")
    elif case == "complement":
        graph.graph.node[2].input[0] = "zero"
    elif case == "unrelated":
        graph.graph.node[0].input[0] = "other_input"
    else:
        graph.graph.node.append(onnx.helper.make_node("Log", ["ratio"], ["unrelated_log"]))
    with pytest.raises(ValueError):
        surgery.bounded_reference_logits(graph)


def test_reference_rewrite_handles_saturation_but_is_not_claimed_bitwise_equal(load_script):
    ort = pytest.importorskip("onnxruntime")
    surgery = load_script("surgery")
    original = reference_graph()
    candidate = surgery.bounded_reference_logits(original)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = [
        ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for m in (original, candidate)
    ]
    values = np.resize(np.array([-30, -11.25, 0, 1, 11.2525, 30], np.float32), 1200).reshape(1, 300, 4)
    before, after = [s.run(None, {"reference_logits": values})[0] for s in sessions]
    assert np.isfinite(after).all() and not np.array_equal(before, after)
    assert np.max(np.abs(before - after)) < 0.011


def test_sdk_system_exit_cannot_look_like_a_successful_phase(compiler):
    def sdk_failure():
        raise SystemExit()

    with pytest.raises(RuntimeError, match="failed"):
        compiler.execute_phase(sdk_failure, {})


def test_failed_output_setup_never_builds_runners(load_script, monkeypatch):
    p = load_script("performance")
    called = []
    monkeypatch.setattr(p, "mla_stats", lambda *args: called.append("hardware"))
    monkeypatch.setattr(p, "build_runners", lambda *args: called.append("build"))
    with pytest.raises(OSError):
        p.benchmark_pipeline({}, "coco-2017-val", Path("/unused"), Path("/dev/null/not-a-directory"))
    assert called == []


def test_compiled_packages_have_native_placement_and_real_checksums(compiler, compiled, build_profile):
    from model_registry.fetch import sha256_file

    receipt = json.loads((compiled / "compile-outputs.json").read_text())
    assert receipt["model_id"] == "dfine_n" and receipt["build_profile_id"] == build_profile
    assert build_profile in compiler.PROFILES
    assert [item["component_id"] for item in receipt["outputs"]] == ["backbone", "decoder"]
    for item in receipt["outputs"]:
        path = compiled / item["mpk_path"]
        assert sha256_file(path)[0] == item["mpk_sha256"]
        compiler.audit_mpk(path, item["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text())
    assert fidelity["samples"] == 10 and fidelity["selection"]["offset"] >= 50
    assert [m["name"] for m in fidelity["metrics"]] == ["pred_boxes", "pred_logits"]
    assert len(receipt["derived_graphs"]) == 2
