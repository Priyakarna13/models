"""Host assembly tests; native execution remains a board qualification gate."""

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def p(load_script):
    return load_script("performance")


def test_complete_model_ingress_then_explicit_raw_head_decode(p, monkeypatch):
    calls = []

    class Route:
        include_input = False
        include_output = True

    class Graph:
        def add(self, node):
            calls.append(node)

    class Model:
        def graph(self, route):
            assert route.include_input is True and route.include_output is False
            return Graph()

    model = Model()

    def boxdecode(bound, **options):
        assert bound == "DetrFocal"
        assert options == {
            "original_width": 1280,
            "original_height": 720,
            "model_width": 640,
            "model_height": 640,
            "detection_threshold": 0.0,
            "nms_iou_threshold": 0.0,
            "top_k": 300,
            "decode_type_option": "InterleavedByHeadLogit",
            "source_storage": "DenseHwc",
            "detess": False,
            "dequant": False,
            "resize_mode": "Stretch",
        }
        return "native_boxdecode"

    monkeypatch.setattr(p.pyneat, "ModelRouteOptions", Route, raising=False)
    monkeypatch.setattr(
        p.pyneat,
        "BoxDecodeType",
        types.SimpleNamespace(DetrFocal="DetrFocal"),
        raising=False,
    )
    monkeypatch.setattr(p.pyneat, "ResizeMode", types.SimpleNamespace(Stretch="Stretch"), raising=False)
    monkeypatch.setattr(
        p.pyneat,
        "BoxDecodeTypeOption",
        types.SimpleNamespace(InterleavedByHeadLogit="InterleavedByHeadLogit"),
        raising=False,
    )
    monkeypatch.setattr(p.pyneat, "BoxDecodeSourceStorage", types.SimpleNamespace(DenseHwc="DenseHwc"), raising=False)
    monkeypatch.setattr(
        p.pyneat,
        "nodes",
        types.SimpleNamespace(sima_box_decode=boxdecode, output=lambda: "output"),
        raising=False,
    )
    p.decoder_graph(model)
    assert calls == ["native_boxdecode", "output"]


def tensor(format):
    return types.SimpleNamespace(semantic=types.SimpleNamespace(detection=types.SimpleNamespace(format=format)))


def test_completion_requires_native_bbox_and_parses_inside_boundary(p, monkeypatch):
    outputs = [tensor("BBOX")]
    called = []

    def decode(values, **options):
        assert values is outputs and options == {"strict": True}
        called.append(True)
        return [types.SimpleNamespace(to_numpy=lambda: np.zeros((300, 6), np.float32))]

    monkeypatch.setattr(p.pyneat, "decode_bbox", decode, raising=False)
    p.require_native_boxes(outputs)
    assert called == [True]


@pytest.mark.parametrize("outputs", [[], [tensor("RAW")], [tensor("BBOX"), tensor("BBOX")]])
def test_raw_or_missing_results_cannot_count_as_native(p, outputs):
    with pytest.raises(RuntimeError):
        p.require_native_boxes(outputs)


def test_dropped_pairs_rejected(p, monkeypatch):
    monkeypatch.setattr(
        p.pyneat,
        "decode_bbox",
        lambda *a, **k: [types.SimpleNamespace(to_numpy=lambda: np.zeros((299, 6)))],
        raising=False,
    )
    with pytest.raises(RuntimeError, match="299 of 300"):
        p.require_native_boxes([tensor("BBOX")])


def test_drain_checks_eos_counts_and_accepts_optional_sample(p):
    class Run:
        def __init__(self):
            self.outputs = iter([types.SimpleNamespace(tensors=[1]), None])

        def pull(self, timeout):
            return next(self.outputs)

    assert list(p.drain(Run(), "decoder", {"decoder": 1})) == [[1]]
    with pytest.raises(RuntimeError, match="1 of 2"):
        list(p.drain(Run(), "decoder", {"decoder": 2}))


def test_query_bridge_owns_gathers_not_attention_memory(p, monkeypatch):
    monkeypatch.setattr(p.pyneat, "TensorLayout", types.SimpleNamespace(Unknown="Unknown"), raising=False)

    def tensor_of(array):
        return types.SimpleNamespace(
            to_numpy=lambda **kwargs: array,
            semantic=types.SimpleNamespace(preprocess="same-frame-meta"),
        )

    content = np.arange(2000 * 128, dtype=np.float32).reshape(1, 2000, 128)
    scores = np.arange(2000, dtype=np.float32)[None]
    refs = np.arange(2000 * 4, dtype=np.float32).reshape(1, 2000, 4)
    copies = []

    def from_numpy(array, **options):
        assert options == {"copy": True, "memory": "EV74", "layout": "Unknown"}
        copies.append(array.copy())
        return types.SimpleNamespace(array=copies[-1], semantic=types.SimpleNamespace())

    monkeypatch.setattr(p.pyneat, "Tensor", types.SimpleNamespace(from_numpy=from_numpy), raising=False)
    monkeypatch.setattr(p.pyneat, "TensorMemory", types.SimpleNamespace(EV74="EV74"), raising=False)
    selected = p.select_queries(*map(tensor_of, (content, scores, refs)))
    np.testing.assert_array_equal(selected[0].array, content[:, 1999:1699:-1])
    np.testing.assert_array_equal(selected[1].array, refs[:, 1999:1699:-1])
    assert all(not np.shares_memory(x.array, y) for x, y in zip(selected, (content, refs), strict=True))

    assert all(x.semantic.preprocess == "same-frame-meta" for x in selected)


def test_native_normalization_declares_rgb_over_255(p, monkeypatch):
    options = types.SimpleNamespace(
        advanced_execution=types.SimpleNamespace(),
        preprocess=types.SimpleNamespace(normalize=types.SimpleNamespace(), resize=types.SimpleNamespace()),
    )
    monkeypatch.setattr(p.pyneat, "ModelOptions", lambda: options, raising=False)
    monkeypatch.setattr(p.pyneat, "InputKind", types.SimpleNamespace(Image="Image"), raising=False)
    monkeypatch.setattr(p.pyneat, "AutoFlag", types.SimpleNamespace(On="On"), raising=False)
    monkeypatch.setattr(p.pyneat, "ResizeMode", types.SimpleNamespace(Stretch="Stretch"), raising=False)
    actual = p.image_options()
    assert actual is options
    assert actual.advanced_execution.postprocess_target == "EV74"
    assert actual.preprocess.kind == "Image"
    assert actual.preprocess.resize.mode == "Stretch"
    norm = actual.preprocess.normalize
    assert norm.enable == "On" and norm.has_explicit_stats is True
    assert norm.mean == [0.0, 0.0, 0.0] and norm.stddev == [1.0, 1.0, 1.0]


def test_sequential_inference_forwards_attention_without_conversion(p, monkeypatch):
    memory, content, scores, references, selected = (object() for _ in range(5))
    backbone = types.SimpleNamespace(run=lambda inputs, timeout_ms: [memory, content, scores, references])
    decoder = object()

    def select(c, s, r):
        assert (c, s, r) == (content, scores, references)
        return [selected]

    def invoke(run, inputs):
        assert run is decoder and inputs[0] is memory and inputs[1] is selected

    monkeypatch.setattr(p, "select_queries", select)
    monkeypatch.setattr(p, "invoke_decoder", invoke)
    p.infer_one(backbone, decoder, object())


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["backbone", "decoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id
