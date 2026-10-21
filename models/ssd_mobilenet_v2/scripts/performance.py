"""Benchmark one compiled MPK with Neat Core and save its report."""

import argparse
import json
from pathlib import Path

import pyneat

REPORT_FIELDS = {
    "latency_ms": ("model.latency.avg", "ms"),
    "latency_p50_ms": ("model.latency.p50", "ms"),
    "latency_p95_ms": ("model.latency.p95", "ms"),
    "latency_p99_ms": ("model.latency.p99", "ms"),
    "fps": ("model.inference_rate", "inferences/s"),
    "avg_power_watts": ("board.power.avg", "W"),
    "energy_joules": ("board.energy", "J"),
}
TELEMETRY_FIELDS = [
    ("power_som_watts", "maximum", "W", "board.power.peak"),
    ("power_mla_watts", "mean", "W", "board.mla_power.avg"),
    ("power_mla_watts", "maximum", "W", "board.mla_power.peak"),
    ("die_temperature_max_c", "mean", "C", "board.die_temperature.avg"),
    ("die_temperature_max_c", "maximum", "C", "board.die_temperature.peak"),
    ("board_temperatures_c", "mean", "C", "board.som_temperature.avg"),
    ("board_temperatures_c", "maximum", "C", "board.som_temperature.peak"),
    ("cpu_usage_pct", "mean", "%", "board.cpu_usage.avg"),
    ("cpu_usage_pct", "maximum", "%", "board.cpu_usage.peak"),
    ("cpu_cores_usage_pct", "mean", "%", "board.cpu_core_usage.avg"),
    ("cpu_cores_usage_pct", "maximum", "%", "board.cpu_core_usage.peak"),
    ("mla_memory_mib", "mean", "MiB", "model.mla_memory.avg"),
    ("mla_memory_mib", "maximum", "MiB", "model.mla_memory.peak"),
    ("cma_memory_mib", "mean", "MiB", "board.cma_memory.avg"),
    ("cma_memory_mib", "maximum", "MiB", "board.cma_memory.peak"),
    ("linux_memory_used_mib", "mean", "MiB", "board.linux_memory.avg"),
    ("linux_memory_used_mib", "maximum", "MiB", "board.linux_memory.peak"),
]


def write_performance_report(output: Path, report: dict, context: dict, components: dict) -> None:
    """Write whole-model measurements while retaining Core's complete raw evidence."""
    json.dumps(report, allow_nan=False)
    metrics = []
    for field, (name, unit) in REPORT_FIELDS.items():
        value = report.get(field, 0)
        if value < 0:
            raise ValueError(f"negative benchmark reading: {field}")
        metric = {"name": name, "unit": unit, "aggregation": "raw"}
        if value > 0:
            metric.update(status="measured", value=value)
        else:
            metric.update(status="unavailable", reason="Core did not measure this value")
        metrics.append(metric)
    for field, statistic, unit, name in TELEMETRY_FIELDS:
        values = report.get("telemetry", {}).get(field, [])
        entries = values if isinstance(values, list) else [values]
        entries = [entry for entry in entries if entry.get("samples", 0) > 0]
        metric = {
            "name": name,
            "unit": "°C" if unit == "C" else unit,
            "aggregation": "mean" if statistic == "mean" else "max",
        }
        if entries:
            if any(entry["unit"] != unit for entry in entries):
                raise ValueError(f"unexpected telemetry unit for {field}")
            selected = max(entries, key=lambda entry: entry[statistic])
            metric.update(
                status="measured",
                value=selected[statistic],
                sample_count=selected["samples"],
                minimum=selected["minimum"],
                maximum=selected["maximum"],
            )
        else:
            metric.update(status="unavailable", reason="Telemetry supplied no samples")
        if name == "board.power.peak" and entries:
            metric["details"] = {
                "breakdown": [
                    {key: entry[key] for key in ("key", "label", "unit", "samples", "mean", "maximum")}
                    for entry in report.get("telemetry", {}).get("power_rails_watts", [])
                    if entry["samples"]
                ]
            }
        metrics.append(metric)
    inferences = context["num_samples"] * context["workload"]["batch_size"]
    if report.get("energy_joules", 0) > 0:
        metrics.append(
            {
                "name": "model.energy_per_inference",
                "status": "measured",
                "value": report["energy_joules"] / inferences,
                "unit": "J/inference",
            }
        )
    document = {
        "schema": "sima.model-report",
        "schema_version": 1,
        "stage": "performance",
        "context": {key: value for key, value in context.items() if key not in {"components", "execution_plan"}},
        "metrics": metrics,
        "components": components,
        "details": {
            "throughput_samples": report.get("throughput_samples"),
            "throughput_duration_ms": report.get("throughput_duration_ms"),
            "telemetry": {key: report.get("telemetry", {}).get(key) for key in (
                "status", "sentinel_version", "samples", "sample_interval_ms"
            )},
            "cpu_cores": [{"mean": item["mean"], "maximum": item["maximum"]}
                          for item in report.get("telemetry", {}).get("cpu_cores_usage_pct", []) if item["samples"]],
        },
        "evidence": [
            {"path": "benchmark-report.json", "media_type": "application/json"},
            {"path": "benchmark-context.json", "media_type": "application/json"},
        ],
    }
    (output / "performance-report.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mpk", required=True, help="compiled model archive")
parser.add_argument("--output", required=True, help="directory for benchmark-report.json and benchmark-context.json")
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

model = pyneat.Model(args.mpk)
# Request 1000 samples and a 60 s minimum measurement duration.
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = 1000
benchmark_options.min_duration_ms = 60000
benchmark_options.include_layer_stats = True
model.benchmark(benchmark_options).save(output / "benchmark-report.json")
# What the report does not record: the Core release, the requested run, and the graph that ran.
context = {
    "core_version": pyneat.__version__,
    "num_samples": benchmark_options.num_samples,
    "min_duration_ms": benchmark_options.min_duration_ms,
    "include_layer_stats": benchmark_options.include_layer_stats,
    "execution_plan": json.loads(model.execution_plan().to_json()),
}

context.setdefault("workload", {}).update(
    {
        "kind": "model.synthetic",
        "batch_size": model.compiled_batch_size(),
        "inputs": [{"shape": spec.shape, "dtype": spec.dtypes[0].name} for spec in model.input_specs()],
        "boundary": "Core synthetic model benchmark including configured preprocessing and decoding",
    }
)
(output / "benchmark-context.json").write_text(json.dumps(context, allow_nan=False), encoding="utf-8")
raw_report = json.loads((output / "benchmark-report.json").read_text(encoding="utf-8"))
write_performance_report(
    output,
    raw_report,
    context,
    {
        "model": {"execution_plan": context["execution_plan"], "layer_stats": raw_report["layer_stats"]},
    },
)
