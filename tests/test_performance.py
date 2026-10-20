import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from model_registry.errors import ValidationError
from model_registry.performance import (
    DEFAULT_OPTIONS,
    PERFORMANCE_RUNNER,
    benchmark_options,
    performance_script,
    validate_benchmarks,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("decoder,decode_type,selection", [
    (None, None, None),
    ("yolo26_boxes", "boxes", "YoloV26 box decoder, top 25"),
    ("yolo26_pose", "pose", "YoloV26Pose box decoder, top 25"),
    ("yolo26_seg", "seg", "YoloV26Seg decoder, top 25, 160x160 masks"),
])
def test_shared_runner_applies_workload_and_preserves_core_report(
    decoder, decode_type, selection, monkeypatch, tmp_path
):
    calls = []
    plan = {"schema": "sima.neat.execution_plan", "schema_version": 1, "nodes": [{"id": "mla"}], "edges": []}
    report = {"layer_stats": [{"name": "conv", "stage_index": 0, "total_execution_time_us": 12.5}],
              "future_core_field": {"preserved": True}}

    class Model:
        def __init__(self, mpk, options=None):
            calls.append((mpk, options))

        def benchmark(self, options):
            calls.append(options)
            return SimpleNamespace(save=lambda path: path.write_text(json.dumps(report)))

        def execution_plan(self):
            return SimpleNamespace(to_json=lambda: json.dumps(plan))

        def info(self):
            return SimpleNamespace(selection=SimpleNamespace(selected_post_kind="decoder"))

    monkeypatch.setitem(sys.modules, "pyneat", SimpleNamespace(
        __version__="test", Model=Model, ModelOptions=SimpleNamespace, BenchmarkOptions=SimpleNamespace,
        BoxDecodeType=SimpleNamespace(YoloV26="boxes", YoloV26Pose="pose", YoloV26Seg="seg"),
    ))
    declared = {"num_samples": 45, "min_duration_ms": 120, "include_layer_stats": False}
    if decoder:
        declared.update(decoder=decoder, top_k=25)
    options = benchmark_options(declared)
    script = ROOT / PERFORMANCE_RUNNER
    monkeypatch.setattr(sys, "argv", [str(script), "--mpk", "model.tar.gz", "--output", str(tmp_path),
                                     "--options", json.dumps(options)])
    runpy.run_path(str(script), run_name="__main__")

    assert calls[0][0] == "model.tar.gz"
    if decoder:
        assert vars(calls[0][1]) == {"decode_type": decode_type, "top_k": 25}
    else:
        assert calls[0][1] is None
    assert vars(calls[1]) == {key: options[key] for key in DEFAULT_OPTIONS}
    assert json.loads((tmp_path / "benchmark-report.json").read_text()) == report
    context = json.loads((tmp_path / "benchmark-context.json").read_text())
    expected = {"core_version": "test", **{key: options[key] for key in DEFAULT_OPTIONS}, "execution_plan": plan}
    if decoder:
        expected["workload"] = {"selection": selection, "postprocess": "decoder"}
    assert context == expected


def test_workload_defaults_and_custom_override_are_explicit(tmp_path):
    model_dir = tmp_path / "models/example"
    assert benchmark_options({}) == DEFAULT_OPTIONS
    assert benchmark_options({"decoder": "yolo26_pose"})["top_k"] == 100
    assert benchmark_options({}, custom=True) == {}
    assert performance_script(model_dir) == tmp_path / PERFORMANCE_RUNNER
    script = model_dir / "scripts/performance.py"
    script.parent.mkdir(parents=True)
    script.write_text("custom pipeline")
    assert performance_script(model_dir) == script
    for option in ({"num_samples": 100}, {"include_layer_stats": True}, {"decoder": "yolo26_boxes"}):
        with pytest.raises(ValidationError, match="custom.*unsupported benchmark settings"):
            benchmark_options(option, custom=True)
    with pytest.raises(ValidationError, match="top_k requires an explicit decoder"):
        benchmark_options({"top_k": 10})


def test_shared_runner_rejects_package_and_language_workloads(tmp_path):
    model = {"components": [{"interface": {"format": "onnx"}}],
             "benchmark": [{"scope": {"type": "component"}}]}
    validate_benchmarks(model, tmp_path / "models/example")
    model["benchmark"][0]["scope"]["type"] = "package"
    with pytest.raises(ValidationError, match="add scripts/performance.py"):
        validate_benchmarks(model, tmp_path / "models/example")
    model["benchmark"][0]["scope"]["type"] = "component"
    model["components"][0]["interface"]["format"] = "huggingface"
    with pytest.raises(ValidationError, match="one ONNX component"):
        validate_benchmarks(model, tmp_path / "models/example")


def test_shared_runner_help_does_not_load_the_devkit_sdk():
    result = subprocess.run([sys.executable, str(ROOT / PERFORMANCE_RUNNER), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--mpk" in result.stdout
