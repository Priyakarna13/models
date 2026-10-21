import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from model_registry import measurement_environment
from model_registry.errors import ValidationError
from model_registry.measurement_environment import _require_internals_runtime
from model_registry.model_report import load_report

PLAN = {
    "schema": "sima.neat.execution_plan",
    "schema_version": 1,
    "model": {"name": "resnet_50", "batch": 4},
    "nodes": [{"id": "p0", "kind": "Input", "runs_on": "A65"}],
    "edges": [],
}
ROOT = Path(__file__).resolve().parents[1]


def component_benchmark_scripts():
    return sorted(
        path for path in ROOT.glob("models/*/scripts/performance.py") if not path.parents[1].name.startswith("rf_detr")
    )


def summary(key, unit, mean, maximum, samples=15):
    return {
        "key": key,
        "label": key,
        "unit": unit,
        "samples": samples,
        "mean": mean,
        "minimum": mean - 1.0,
        "maximum": maximum,
    }


def no_samples(key, unit):
    return summary(key, unit, 0.0, 0.0, samples=0)


def sentinel_telemetry():
    """Telemetry in the shape BenchmarkReport.save() writes from Sentinel on a DevKit."""
    return {
        "status": "ok",
        "trace_id": "neat-benchmark-test",
        "sentinel_version": "main:test",
        "sample_interval_ms": 2000,
        "samples": 15,
        "power_som_watts": summary("power_current_watts", "W", 11.3, 11.7),
        "power_mla_watts": summary("power_rail_mla_watts", "W", 1.9, 2.0),
        "power_rails_watts": [summary("soc", "W", 3.5, 3.8), summary("mla", "W", 1.9, 2.0)],
        "die_temperatures_c": [summary("rtsn_0", "C", 41.9, 42.7), summary("rtsn_1", "C", 42.2, 43.3)],
        "die_temperature_max_c": summary("die_temperature_max_c", "C", 42.3, 43.3),
        "board_temperatures_c": [summary("temp1", "C", 40.9, 42.0), summary("temp2", "C", 43.9, 44.6)],
        "cpu_usage_pct": summary("cpu_usage_pct", "%", 7.8, 8.7),
        "cpu_cores_usage_pct": [
            summary("cpu_core_0_usage_pct", "%", 6.0, 7.0),
            summary("cpu_core_1_usage_pct", "%", 61.0, 98.5),
        ],
        "mla_memory_mib": summary("mla_memory_mib", "MiB", 29.6, 29.7),
        "cma_memory_mib": summary("cma_memory_mib", "MiB", 863.8, 867.2),
        "linux_memory_used_mib": summary("linux_memory_used_mib", "MiB", 600.0, 607.5),
    }


def unavailable_telemetry():
    """What save() writes when Sentinel did not answer: every reading present with no samples."""
    telemetry = sentinel_telemetry()
    for name, value in telemetry.items():
        if isinstance(value, dict):
            telemetry[name] = no_samples(value["key"], value["unit"])
        elif isinstance(value, list):
            telemetry[name] = []
    return {**telemetry, "status": "sentinel_unreachable", "samples": 0}


def saved_report(**fields):
    """A report in the shape BenchmarkReport.save() writes; Core writes 0 for what it did not measure."""
    return {
        "latency_ms": 2.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
        "fps": 2.0,
        "avg_power_watts": 0.0,
        "energy_joules": 2.0,
        "peak_power_watts": 0.0,
        "throughput_duration_ms": 60000.0,
        "throughput_samples": 120,
        "telemetry": unavailable_telemetry(),
        "layer_stats": [],
        **fields,
    }


@pytest.mark.parametrize(
    "packages",
    [
        {"neat-runtime": "0.5.0+release-3.0.0-prep.464889585182"},
        {"neat-runtime:arm64": "0.5.0+release-3.0.0-prep.464889585182"},
    ],
)
def test_accepts_versioned_internals_runtime_package(packages):
    _require_internals_runtime(packages)


@pytest.mark.parametrize(
    "packages",
    [
        {},
        {"neat-runtime": ""},
        {"neat-internals": "0.4.0+legacy"},
        {"neat-internals-dev": "0.5.0+test"},
        {"sima-neat": "0.4.0+test"},
    ],
)
def test_rejects_missing_internals_runtime_package(packages):
    with pytest.raises(ValidationError, match="installed Neat Internals version"):
        _require_internals_runtime(packages)


def test_runner_environment_records_only_installed_packages(monkeypatch):
    package_output = "\n".join(
        [
            "ii \tneat-runtime:arm64\t0.5.0+release-3.0.0-prep.464889585182",
            "ii \tsima-neat\t0.4.0+release-3.0.0-prep.02a5ca96b124",
            "rc \tneat-gst-plugins\t0.4.0+removed",
            "un \tneat-internals\t",
        ]
    )

    def run(command, **kwargs):
        assert command == [
            "dpkg-query",
            "-W",
            "-f=${db:Status-Abbrev}\t${binary:Package}\t${Version}\n",
            "*neat*",
        ]
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return SimpleNamespace(stdout=package_output)

    monkeypatch.setattr(Path, "read_text", lambda self, encoding: "DISTRO_VERSION = 3.0.0\nBUILD = B1800\n")
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(measurement_environment.subprocess, "run", run)

    environment = measurement_environment.runner_environment()

    assert environment["packages"] == {
        "neat-runtime:arm64": "0.5.0+release-3.0.0-prep.464889585182",
        "sima-neat": "0.4.0+release-3.0.0-prep.02a5ca96b124",
    }
    assert environment["mla_frequency_hz"] is None
    assert environment["cpu_frequency_khz"] is None


def test_runner_environment_records_the_mla_and_cpu_clocks(monkeypatch):
    files = {
        "/etc/buildinfo": "DISTRO_VERSION = 3.0.0\nBUILD = B1800\n",
        str(measurement_environment.MLA_FREQUENCY): "1000000000\n",
        str(measurement_environment.CPU_FREQUENCY): "1400000\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: files[str(self)])
    monkeypatch.setattr(Path, "exists", lambda self: str(self) in files)
    packages = SimpleNamespace(stdout="ii \tneat-runtime\t0.5.0+test\n")
    monkeypatch.setattr(measurement_environment.subprocess, "run", lambda command, **kwargs: packages)

    environment = measurement_environment.runner_environment()

    assert environment["mla_frequency_hz"] == 1_000_000_000
    assert environment["cpu_frequency_khz"] == 1_400_000


@pytest.mark.parametrize("script", component_benchmark_scripts(), ids=lambda path: path.parent.parent.name)
def test_every_component_runner_requests_and_saves_layer_stats(script, monkeypatch, tmp_path):
    import runpy
    import sys

    report = {
        "latency_ms": 2.0,
        "fps": 100.0,
        "layer_stats": [{"name": "conv", "stage_index": 0, "total_execution_time_us": 12.5}],
        "future_core_field": "preserved",
    }
    requests = []

    class Report:
        def save(self, path):
            Path(path).write_text(json.dumps(report))

    selected = []

    class Model:
        def __init__(self, *args):
            selected.append(args)

        def benchmark(self, options):
            requests.append(options)
            return Report()

        def execution_plan(self):
            return SimpleNamespace(to_json=lambda: json.dumps(PLAN))

        def compiled_batch_size(self):
            return 4

        def input_specs(self):
            return [SimpleNamespace(shape=[4, 224, 224, 3], dtypes=[SimpleNamespace(name="UINT8")])]

        def info(self):
            return SimpleNamespace(selection=SimpleNamespace(selected_post_kind="boxdecode"))

    pyneat = SimpleNamespace(
        __version__="test",
        Model=Model,
        ModelOptions=SimpleNamespace,
        BenchmarkOptions=SimpleNamespace,
        BoxDecodeType=SimpleNamespace(YoloV26="YoloV26", YoloV26Pose="YoloV26Pose", YoloV26Seg="YoloV26Seg"),
    )
    monkeypatch.setitem(sys.modules, "pyneat", pyneat)
    standalone = tmp_path / "performance.py"
    shutil.copyfile(script, standalone)
    monkeypatch.setattr(sys, "argv", [str(standalone), "--mpk", "model.tar.gz", "--output", str(tmp_path)])
    runpy.run_path(str(standalone), run_name="__main__")
    model_id = script.parents[1].name
    if model_id.startswith("yolo26") and model_id != "yolo26_tiny_drone":
        decoder = (
            "YoloV26Pose" if model_id.endswith("_pose") else ("YoloV26Seg" if model_id.endswith("_seg") else "YoloV26")
        )
        assert selected[0][0] == "model.tar.gz"
        assert vars(selected[0][1]) == {"decode_type": decoder, "top_k": 100}
    else:
        assert selected == [("model.tar.gz",)]
    assert len(requests) == 1
    assert requests[0].include_layer_stats is True
    assert requests[0].num_samples == 1000
    assert requests[0].min_duration_ms == 60000
    assert json.loads((tmp_path / "benchmark-report.json").read_text()) == report
    context = json.loads((tmp_path / "benchmark-context.json").read_text())
    assert context["include_layer_stats"] is True
    assert context["execution_plan"] == PLAN
    standard = load_report(tmp_path / "performance-report.json", required_metrics=["model.inference_rate"])
    assert standard["components"]["model"]["layer_stats"] == report["layer_stats"]
    assert standard["context"]["workload"]["batch_size"] == 4


def report_writer(path):
    """Load only the model-owned pure report conversion, without importing a board runtime."""
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "write_performance_report"
        or isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"REPORT_FIELDS", "TELEMETRY_FIELDS"}
    ]
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["write_performance_report"]


@pytest.mark.parametrize(
    "script", sorted(ROOT.glob("models/*/scripts/performance.py")), ids=lambda p: p.parents[1].name
)
def test_standalone_reports_preserve_telemetry_energy_and_optional_availability(script, tmp_path):
    raw = saved_report(avg_power_watts=11.3, energy_joules=2.0, telemetry=sentinel_telemetry())
    context = {"num_samples": 100, "workload": {"batch_size": 4}, "core_version": "test"}
    components = {
        "model": {
            "layer_stats": [{"name": "conv", "stage_index": 0, "total_execution_time_us": 5}],
            "execution_plan": PLAN,
        }
    }
    report_writer(script)(tmp_path, raw, context, components)
    report = load_report(tmp_path / "performance-report.json")
    metrics = {item["name"]: item for item in report["metrics"]}
    assert metrics["model.energy_per_inference"]["value"] == 0.005
    assert metrics["board.cpu_core_usage.avg"]["value"] == 61
    assert metrics["board.cpu_core_usage.peak"]["value"] == 98.5
    assert metrics["board.som_temperature.avg"]["value"] == 43.9
    assert metrics["board.die_temperature.peak"]["unit"] == "°C"
    assert metrics["model.latency.p99"]["status"] == "unavailable"
    assert "value" not in metrics["model.latency.p99"]
    assert report["components"] == components
    raw["telemetry"]["die_temperature_max_c"]["unit"] = "K"
    with pytest.raises(ValueError, match="unexpected telemetry unit"):
        report_writer(script)(tmp_path, raw, context, components)
