from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import model_report
import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def assert_native_plan(plan):
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    nodes = {node["id"]: node for node in plan["nodes"]}
    boxes = [node for node in nodes.values() if node["kind"] == "SimaBoxDecode"]
    assert len(boxes) == 1
    (box,) = boxes
    assert box["runs_on"] == "A65"
    settings = box["settings"]
    assert settings["decode_type"] == "ssd"
    assert settings["top_k"] == 100
    assert float(np.float32(settings["detection_threshold"])) == float(np.float32(0.2))
    assert float(np.float32(settings["nms_iou_threshold"])) == float(np.float32(0.6))
    edges = plan["edges"]
    assert edges

    def reachable(start, removed=None):
        seen, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in seen or node == removed:
                continue
            seen.add(node)
            pending.extend(edge["to"] for edge in edges if edge["from"] == node)
        return seen

    mla = [key for key, node in nodes.items() if node["runs_on"] == "MLA"]
    outputs = {key for key, node in nodes.items() if node["kind"] == "Output"}
    assert mla and outputs
    assert any(box["id"] in reachable(key) for key in mla)
    assert outputs <= reachable(box["id"])
    assert all(not (outputs & reachable(key, box["id"])) for key in mla), "raw-head output bypasses SSD"


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path, build_profile: str):
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_complete_native_execution_plan(performance):
    (component,) = performance["components"].values()
    assert_native_plan(component["execution_plan"])


def test_mla_stats_collected(performance):
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"])
    assert stage_stats["stages"]


def test_performance_explicit_native_customer_policy(load_script):
    script = load_script("performance")
    options = script.native_options()
    assert options.decode_type == sys.modules["pyneat"].BoxDecodeType.Ssd
    assert (options.score_threshold, options.nms_iou_threshold, options.top_k) == (0.2, 0.6, 100)
    assert options.preprocess.kind == sys.modules["pyneat"].InputKind.Image
    assert options.preprocess.resize.mode == sys.modules["pyneat"].ResizeMode.Stretch
    assert options.preprocess.normalize.mean == [0.5] * 3
    assert options.preprocess.normalize.stddev == [0.5] * 3
    assert (options.boxdecode_original_width, options.boxdecode_original_height) == (0, 0)


def test_performance_uses_seeded_real_720p_input_and_whole_model_report(load_script, monkeypatch, tmp_path):
    script = load_script("performance")
    folder = tmp_path / "val2017"
    folder.mkdir()
    for name in ["c.jpg", "a.jpg", "b.jpg"]:
        (folder / name).touch()
    assert script.sample_image(tmp_path).name == "b.jpg"
    monkeypatch.setattr(script, "read_rgb", lambda _: np.zeros((40, 40, 3), np.uint8))
    imported, benchmarks, reports, saved = [], [], [], []

    def tensor(frame, **kw):
        imported.append((frame.copy(), kw))
        return "image-tensor"

    monkeypatch.setattr(sys.modules["pyneat"].Tensor, "from_numpy", tensor)
    model = SimpleNamespace(benchmark=lambda options: benchmarks.append(options) or "actual-core-report")
    monkeypatch.setattr(sys.modules["pyneat"], "Model", lambda path, options: model)

    def report(core_report, models, **kw):
        reports.append((core_report, models, kw))
        return SimpleNamespace(save=lambda path: saved.append(path))

    monkeypatch.setattr(model_report, "from_performance", report)
    script.benchmark("model.tar.gz", "coco-2017-val", tmp_path, tmp_path / "out")
    (options,) = benchmarks
    assert options.inputs == ["image-tensor"]
    assert (options.num_samples, options.min_duration_ms) == (1000, 60000)
    assert options.include_mla_layer_stats is True and options.include_mla_stage_stats is True
    image, kwargs = imported[0]
    assert image.shape == (720, 1280, 3) and image.dtype == np.uint8
    assert (image[:, :280] == 114).all() and (image[:, 280:1000] == 0).all()
    assert kwargs == dict(copy=True, image_format=sys.modules["pyneat"].PixelFormat.RGB)
    assert reports[0][:2] == ("actual-core-report", {"model": model})
    assert "top 100" in reports[0][2]["workload"] and "complete detector" in reports[0][2]["workload"]
    assert saved == [tmp_path / "out"]


def test_performance_is_standalone_without_registry_or_sibling_imports(model_dir):
    tree = ast.parse((model_dir / "scripts/performance.py").read_text())
    imports = {
        name.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    }
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {"__future__", "argparse", "random", "pathlib", "cv2", "model_report", "numpy", "pyneat"}


def _plan():
    return {
        "schema": "sima.neat.execution_plan",
        "schema_version": 1,
        "nodes": [
            dict(id="mla", kind="ModelFragment", runs_on="MLA"),
            dict(
                id="ssd",
                kind="SimaBoxDecode",
                runs_on="A65",
                settings=dict(decode_type="ssd", top_k=100, detection_threshold=0.2, nms_iou_threshold=0.6),
            ),
            dict(id="out", kind="Output", runs_on="A65"),
        ],
        "edges": [dict(**{"from": "mla", "to": "ssd"}), dict(**{"from": "ssd", "to": "out"})],
    }


def test_native_plan_check_rejects_raw_heads_and_wrong_policy():
    assert_native_plan(_plan())
    plan = _plan()
    plan["edges"].append({"from": "mla", "to": "out"})
    with pytest.raises(AssertionError, match="bypasses"):
        assert_native_plan(plan)
    for key, value in [
        ("decode_type", "yolov8"),
        ("top_k", 10),
        ("detection_threshold", 0.25),
        ("nms_iou_threshold", 0.55),
    ]:
        plan = _plan()
        plan["nodes"][1]["settings"][key] = value
        with pytest.raises(AssertionError):
            assert_native_plan(plan)
