from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from model_registry.errors import ValidationError
from model_registry.fetch import sha256_file
from model_registry.report_publication import publish_report


@pytest.fixture
def publication(tmp_path):
    evidence = tmp_path / "raw.json"
    evidence.write_text('{"raw": true}')
    sha, size = sha256_file(evidence)
    inputs = [
        {
            "component_id": component,
            "sha256": digest * 64,
            "size_bytes": 12,
            "compile_run_id": f"compile-{component}",
            "artifact_id": f"artifact-{component}",
            "artifact_key": "mpk",
            "s3_bucket": "compiled",
            "s3_key": f"{component}.mpk",
            "s3_version_id": "immutable",
        }
        for component, digest in [("backbone", "a"), ("decoder", "b")]
    ]
    runtime = {"core": {"install_target": "core@ref:" + "c" * 40}}
    source = {"models_commit": "d" * 40, "script_sha256": "e" * 64}
    manifest = {
        "model_id": "detector",
        "build_profile_id": "int8",
        "branch": "feature/report",
        "runtime": runtime,
        "expected_component_ids": ["backbone", "decoder"],
        "inputs": inputs,
        "stages": {
            stage: {"source": source, "stage_key": key * 64, "required_metrics": [metric]}
            for stage, key, metric in [
                ("performance", "f", "model.inference_rate"),
                ("accuracy", "0", "model.accuracy.top1"),
            ]
        },
    }
    report = {
        "schema": "sima.model-report",
        "schema_version": 1,
        "stage": "performance",
        "context": {},
        "model": {"id": "detector", "build_profile_id": "int8"},
        "provenance": {
            "source": source,
            "environment": {"runtime": runtime, "core_version": "test"},
            "inputs": [{key: item[key] for key in ("component_id", "sha256", "size_bytes")} for item in inputs],
        },
        "metrics": [{"name": "model.inference_rate", "status": "measured", "value": 25, "unit": "inferences/s"}],
        "evidence": [{"path": "raw.json", "media_type": "application/json", "sha256": sha, "size_bytes": size}],
    }
    return manifest, report, tmp_path / "performance-result.json"


class FakeStore:
    def __init__(self):
        self.uploads = []

    def ensure_native_sha256(self, path, key, sha, *, media_type):
        self.uploads.append((path, key, sha, media_type))
        return SimpleNamespace(key=key, version_id="stored-version", sha256=sha, size_bytes=path.stat().st_size)


def publish(publication, *, status="passed"):
    manifest, report, path = publication
    path.write_text(json.dumps(report))
    calls = []
    store = FakeStore()

    def request(method, endpoint, body, key):
        calls.append((method, endpoint, body, key))
        return {"run_id": "report-run", "artifact_id": "report-artifact"}

    result = publish_report(
        manifest=manifest,
        result_path=path,
        api_url="https://registry.invalid",
        aws_region="us-west-2",
        bucket="reports",
        environment="staging",
        test_status=status,
        request=request,
        artifact_store=store,
    )
    return result, calls, store


@pytest.mark.parametrize("stage", ["performance", "accuracy"])
@pytest.mark.parametrize("status", ["passed", "failed", "skipped"])
def test_profile_report_is_parentless_and_references_every_compiled_input(publication, stage, status):
    manifest, report, _ = publication
    report["stage"] = stage
    if stage == "accuracy":
        report["metrics"] = [{"name": "model.accuracy.top1", "status": "measured", "value": 0.9, "unit": "ratio"}]
    result, calls, store = publish(publication, status=status)
    created = calls[0][2]
    assert created["run_type"] == ("model_benchmark" if stage == "performance" else "model_accuracy")
    assert "parent_run_id" not in created
    assert created["metadata"]["model_id"] == "detector"
    assert created["environment"]["fingerprint_version"] == (2 if stage == "performance" else 1)
    links = [body for _, endpoint, body, _ in calls if "/inputs/" in endpoint]
    assert [item["source_reference"] for item in links] == [
        "vulcan://runs/compile-backbone/artifacts/mpk",
        "vulcan://runs/compile-decoder/artifacts/mpk",
    ]
    payload = next(body for _, endpoint, body, _ in calls if endpoint == "/v1/runs/report-run/model-report")
    assert payload["report"] == report
    assert "artifact_key" not in payload["manifest"]["inputs"][0]
    assert payload["manifest"]["expected_component_ids"] == ["backbone", "decoder"]
    assert len(store.uploads) == 2
    assert calls[-1][1:][:2] == ("/v1/runs/report-run/finalize", {"status": "completed"})
    assert result["test_status"] == status
    assert next(body["status"] for _, endpoint, body, _ in calls if "/test-results/" in endpoint) == status


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "profile",
        "source",
        "runtime",
        "input_digest",
        "input_size",
        "input_missing",
        "input_extra",
        "input_duplicate",
        "trusted_duplicate",
        "trusted_partial",
        "undeclared",
        "missing_metric",
        "component_only",
        "evidence_escape",
        "evidence_absolute",
        "evidence_missing",
        "evidence_digest",
        "evidence_size",
    ],
)
def test_invalid_report_fails_before_any_external_write(publication, change):
    manifest, report, path = publication
    if change == "model":
        report["model"]["id"] = "other"
    elif change == "profile":
        report["model"]["build_profile_id"] = "other"
    elif change == "source":
        report["provenance"]["source"] = {"models_commit": "0" * 40, "script_sha256": "e" * 64}
    elif change == "runtime":
        report["provenance"]["environment"]["runtime"] = {"core": {"install_target": "latest"}}
    elif change == "input_digest":
        report["provenance"]["inputs"][0]["sha256"] = "0" * 64
    elif change == "input_size":
        report["provenance"]["inputs"][0]["size_bytes"] += 1
    elif change == "input_missing":
        report["provenance"]["inputs"].pop()
    elif change == "input_extra":
        report["provenance"]["inputs"].append({"component_id": "extra", "sha256": "0" * 64, "size_bytes": 12})
    elif change == "input_duplicate":
        report["provenance"]["inputs"][1] = copy.deepcopy(report["provenance"]["inputs"][0])
    elif change == "trusted_duplicate":
        manifest["inputs"][1] = copy.deepcopy(manifest["inputs"][0])
    elif change == "trusted_partial":
        manifest["expected_component_ids"].append("extra")
    elif change == "undeclared":
        manifest["stages"].pop("performance")
    elif change == "missing_metric":
        report["metrics"] = []
    elif change == "component_only":
        report["components"] = {"backbone": {"metrics": report.pop("metrics")}}
        report["metrics"] = []
    elif change == "evidence_escape":
        report["evidence"][0]["path"] = "../raw.json"
    elif change == "evidence_absolute":
        report["evidence"][0]["path"] = str(path.parent / "raw.json")
    elif change == "evidence_missing":
        report["evidence"][0]["path"] = "missing.json"
    elif change == "evidence_digest":
        report["evidence"][0]["sha256"] = "0" * 64
    elif change == "evidence_size":
        report["evidence"][0]["size_bytes"] += 1
    path.write_text(json.dumps(report))

    def forbidden(*args, **kwargs):
        pytest.fail("publication wrote externally before finishing validation")

    store = FakeStore()
    store.ensure_native_sha256 = forbidden
    with pytest.raises(ValidationError):
        publish_report(
            manifest=manifest,
            result_path=path,
            api_url="https://registry.invalid",
            aws_region="us-west-2",
            bucket="reports",
            environment="staging",
            test_status="passed",
            request=forbidden,
            artifact_store=store,
        )


def test_repeat_publication_has_stable_idempotency_keys(publication):
    _, first, _ = publish(publication)
    _, second, _ = publish(publication)
    assert [call[3] for call in first] == [call[3] for call in second]
