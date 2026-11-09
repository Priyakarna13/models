from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"


def test_report_names_the_full_landmark_boundary(performance):
    workload = performance["workload"]
    assert "CPU crop/resize/normalize" in workload and "landmark decode" in workload
    assert "face detection excluded" in workload and "HEATMAP-ONLY" not in workload
    (component,) = performance["components"].values()
    (prepared,) = component["execution_plan"]["edges"][0]["tensors"]
    assert prepared["shape"][-3:] == [256, 256, 3]  # 720p is outside the tensor-mode Neat graph.


def test_benchmark_maps_the_face_box_not_the_padding(load_script):
    import numpy as np

    script = load_script("performance")
    frame, face_box = script.benchmark_frame(np.full((101, 199, 3), 200, np.uint8))
    scale = min(1280 / 199, 720 / 101)
    width, height = round(199 * scale), round(101 * scale)
    left, top = (1280 - width) // 2, (720 - height) // 2
    assert frame.shape == (720, 1280, 3)
    np.testing.assert_array_equal(face_box, [left, top, left + width, top + height])
    assert (frame[top : top + height, left : left + width] == 200).all()
    assert face_box.tolist() != [0, 0, 1280, 720]


def test_diagnostic_input_capacity_is_real_720p(load_script):
    options = load_script("performance").image_options()
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (1280, 720)


def test_sequential_frame_includes_prepare_and_same_decode(load_script, monkeypatch):
    from types import SimpleNamespace

    script = load_script("performance")
    calls = []
    monkeypatch.setattr(script, "prepare_input", lambda frame, box: (calls.append("prepare") or ["crop"], "box"))
    monkeypatch.setattr(script, "consume_outputs", lambda tensors, box: calls.append(("decode", tensors, box)))
    runner = SimpleNamespace(run=lambda inputs, timeout_ms: calls.append(("run", inputs)) or ["heatmap"])
    script.infer_one(runner, "frame", "face")
    assert calls == ["prepare", ("run", ["crop"]), ("decode", ["heatmap"], "box")]


def _throughput_fixture(script, monkeypatch, *, decode_failure=False, push_failure=False):
    import queue
    import threading
    from types import SimpleNamespace

    import numpy as np

    monkeypatch.setattr(script, "WARMUP_FRAMES", 2)
    monkeypatch.setattr(script, "MEASURED_FRAMES", 3)
    monkeypatch.setattr(script, "MIN_DURATION_S", 0)
    events = []
    box = np.array([0, 0, 10, 10])
    monkeypatch.setattr(script, "crop", lambda frame, face: ("prepared", box))
    monkeypatch.setattr(script, "prepare_input", lambda frame, face: (events.append("prepare") or ["input"], box))

    def consume(tensors, output_box):
        events.append("decode")
        assert tensors == ["heatmap"]
        np.testing.assert_array_equal(output_box, box)
        if decode_failure:
            raise ValueError("decode fault")

    monkeypatch.setattr(script, "consume_outputs", consume)

    class Monitor:
        active = False

        def start(self):
            self.active = True
            events.append("monitor-start")

        def running(self):
            return self.active

        def stop(self):
            self.active = False
            events.append("monitor-stop")
            return "telemetry"

    monitor = Monitor()
    monkeypatch.setattr(script.pyneat, "TelemetryMonitor", lambda: monitor)

    class Runner:
        def __init__(self):
            self.queue = queue.Queue(maxsize=4)
            self.closed = threading.Event()
            self.pushed = 0

        def push(self, inputs):
            if push_failure:
                return False
            while not self.closed.is_set():
                try:
                    self.queue.put(["heatmap"], timeout=0.01)
                    self.pushed += 1
                    return True
                except queue.Full:
                    pass
            return False

        def pull(self, timeout):
            while True:
                try:
                    return SimpleNamespace(tensors=self.queue.get(timeout=0.01))
                except queue.Empty:
                    if self.closed.is_set():
                        return SimpleNamespace(tensors=[])

        def close_input(self):
            self.closed.set()

    return Runner(), monitor, events


def test_throughput_decodes_every_output_and_stops_telemetry(load_script, monkeypatch):
    script = load_script("performance")
    runner, monitor, events = _throughput_fixture(script, monkeypatch)
    result = script.measure_throughput(runner, "frame", "face")
    assert result.frames >= 3 and result.telemetry == "telemetry"
    assert events.count("prepare") == events.count("decode") == runner.pushed
    assert events.count("monitor-start") == events.count("monitor-stop") == 1
    assert not monitor.running() and runner.closed.is_set()


def test_throughput_propagates_decode_error_without_live_threads(load_script, monkeypatch):
    import threading

    import pytest

    script = load_script("performance")
    runner, monitor, _ = _throughput_fixture(script, monkeypatch, decode_failure=True)
    with pytest.raises(ValueError, match="decode fault"):
        script.measure_throughput(runner, "frame", "face")
    assert runner.closed.is_set() and not monitor.running()
    assert not any(t.name in {"fan-produce", "fan-consume"} for t in threading.enumerate())


def test_throughput_rejects_failed_submission(load_script, monkeypatch):
    import pytest

    script = load_script("performance")
    runner, monitor, _ = _throughput_fixture(script, monkeypatch, push_failure=True)
    with pytest.raises(RuntimeError, match="rejected"):
        script.measure_throughput(runner, "frame", "face")
    assert runner.closed.is_set() and not monitor.running()


def test_benchmark_closes_runner_when_latency_raises(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    script = load_script("performance")
    closed = []
    runner = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(script, "read_rgb", lambda _: "image")
    monkeypatch.setattr(script, "benchmark_frame", lambda _: ("frame", "box"))
    monkeypatch.setattr(script, "prepare_input", lambda *args: (["input"], "box"))
    monkeypatch.setattr(script.pyneat, "Model", lambda *args: SimpleNamespace(build=lambda *a, **k: runner))

    def failed(*args):
        raise RuntimeError("latency fault")

    monkeypatch.setattr(script, "measure_latency", failed)
    with pytest.raises(RuntimeError, match="latency fault"):
        script.benchmark_pipeline(Path("model"), Path("image"), tmp_path)
    assert closed == [True]


def test_benchmark_uses_a_seeded_annotated_face(load_script, tmp_path):
    import random

    script = load_script("performance")
    split = tmp_path / "test_data"
    split.mkdir()
    names = [f"{index}_1_0.jpg" for index in range(3)]
    for name in names:
        (split / name).write_bytes(b"image")
        (split / name).with_suffix(".pts").write_text("annotation")
    (split / "unannotated.jpg").write_bytes(b"image")
    (split / "._0_1_0.jpg").write_bytes(b"metadata")
    assert script.sample_face(tmp_path).name == random.Random(0).choice(names)


def test_throughput_joins_started_producer_if_consumer_cannot_start(load_script, monkeypatch):
    import threading

    import pytest

    script = load_script("performance")
    runner, monitor, _ = _throughput_fixture(script, monkeypatch)
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == "fan-consume":
            raise RuntimeError("thread creation fault")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="thread creation fault"):
        script.measure_throughput(runner, "frame", "face")
    assert runner.closed.is_set() and not monitor.running()
    assert not any(t.name in {"fan-produce", "fan-consume"} for t in threading.enumerate())
