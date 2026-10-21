"""The same contract covers vision pipelines and repeated GenAI workloads."""

import copy
import json
from pathlib import Path

import pytest
from model_registry.errors import ValidationError
from model_registry.model_report import load_report, metric_identity, metric_records, validate_report


def report(stage="performance"):
    return {
        "schema": "sima.model-report",
        "schema_version": 1,
        "stage": stage,
        "context": {"batch_size": 1},
        "metrics": [
            {"name": "model.latency.avg", "status": "measured", "value": 12.5, "unit": "ms"},
            {"name": "model.inference_rate", "status": "measured", "value": 80, "unit": "inferences/s"},
        ],
        "evidence": [],
    }


def test_whole_model_report_round_trip(tmp_path):
    document = report()
    document["components"] = {"backbone": {"metrics": [], "layer_stats": [], "execution_plan": None}}
    path = tmp_path / "performance-report.json"
    path.write_text(json.dumps(document))
    assert load_report(path, stage="performance", required_metrics=["model.inference_rate"]) == document


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(schema="legacy"), "schema"),
        (lambda value: value.update(schema_version=7), "schema_version"),
        (lambda value: value.update(unknown_field=True), "Additional properties"),
        (lambda value: value["metrics"][0].update(value=float("nan")), "finite"),
        (lambda value: value["metrics"][0].update(value=True), "not of type"),
        (lambda value: value["metrics"][0].update(unit="s"), "unit"),
        (lambda value: value["metrics"][0].update(direction="higher"), "direction"),
        (lambda value: value["metrics"].append(copy.deepcopy(value["metrics"][0])), "duplicate"),
        (lambda value: value["metrics"][0].update(context={"batch_size": 2}), "conflicting"),
    ],
)
def test_invalid_producer_report_is_rejected(mutation, match):
    document = report()
    mutation(document)
    with pytest.raises(ValidationError, match=match):
        validate_report(document)


def test_native_metrics_keep_every_workload_and_unknown_metric():
    document = report()
    document["context"] = {"tool": "llima"}
    document["metrics"] = [
        {
            "name": "language.decode_rate",
            "status": "measured",
            "value": 10 + tokens,
            "unit": "tokens/s",
            "context": {"input_tokens": tokens},
        }
        for tokens in (16, 32, 64)
    ]
    validate_report(document)
    measured = list(metric_records(document))
    assert [item["value"] for item in measured] == [26, 42, 74]
    assert len({metric_identity(item) for item in measured}) == 3
    assert len(list(metric_records(document))) == 3


def test_required_headline_does_not_select_arbitrary_workload():
    document = report()
    document["metrics"].append({**document["metrics"][1], "context": {"image_size": 704}})
    with pytest.raises(ValidationError, match="exactly one"):
        validate_report(document, required_metrics=["model.inference_rate"])


def test_component_reading_cannot_satisfy_whole_model_requirement():
    document = report()
    component_metric = document["metrics"].pop()
    document["components"] = {"decoder": {"metrics": [component_metric]}}
    with pytest.raises(ValidationError, match="found 0"):
        validate_report(document, required_metrics=["model.inference_rate"])


def test_unavailable_telemetry_retains_reason_without_fake_zero():
    document = report()
    document["metrics"].append({"name": "board.power.avg", "status": "unavailable", "reason": "No samples"})
    validate_report(document)
    records = list(metric_records(document))
    assert len([item for item in records if item["status"] == "measured"]) == 2
    unavailable = records[-1]
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "No samples"
    assert "value" not in unavailable
    with pytest.raises(ValidationError, match="found 0"):
        validate_report(document, required_metrics=["board.power.avg"])


def test_accuracy_samples_and_unit():
    document = report("accuracy")
    document["context"] = {"dataset": {"evaluated_samples": 10, "available_samples": 10}}
    document["metrics"] = [{"name": "model.accuracy.coco_map", "status": "measured", "unit": "ratio", "value": 0.5}]
    validate_report(document)
    document["context"]["dataset"]["evaluated_samples"] = 11
    with pytest.raises(ValidationError, match="exceeds"):
        validate_report(document)


def test_canonical_metrics_are_used_by_every_model_script():
    import ast

    from model_registry.model_report import load_metric_catalog

    root = Path(__file__).resolve().parents[1]
    catalog = load_metric_catalog()
    for path in root.glob("models/*/scripts/performance.py"):
        tree = ast.parse(path.read_text())
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"REPORT_FIELDS", "TELEMETRY_FIELDS"}
        }
        for name, unit in constants["REPORT_FIELDS"].values():
            assert catalog[name]["unit"] == unit
        for _, _, unit, name in constants["TELEMETRY_FIELDS"]:
            assert catalog[name]["unit"] == ("°C" if unit == "C" else unit)
        assert "model_registry" not in path.read_text()
