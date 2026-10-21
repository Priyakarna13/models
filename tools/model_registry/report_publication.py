"""Publish one profile report, linked to every immutable compiled input."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from .errors import ValidationError
from .fetch import sha256_file
from .model_report import load_report, validate_report
from .model_tests import require_test_outcome
from .registry import S3ArtifactStore, SigV4RegistryClient, canonical_sha256
from .stage_keys import ACCURACY_FINGERPRINT_VERSION, PERFORMANCE_FINGERPRINT_VERSION


def validate_publication(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare board-produced evidence with the trusted manifest retained before execution."""
    stage = report["stage"]
    expected = manifest["stages"].get(stage)
    if expected is None:
        raise ValidationError(f"undeclared measurement stage: {stage}")
    if report.get("model") != {"id": manifest["model_id"], "build_profile_id": manifest["build_profile_id"]}:
        raise ValidationError("report model/profile differs from the trusted measurement manifest")
    provenance = report.get("provenance", {})
    if provenance.get("source") != expected["source"]:
        raise ValidationError("report source differs from the trusted measurement manifest")
    if provenance.get("environment", {}).get("runtime") != manifest["runtime"]:
        raise ValidationError("report runtime differs from the planned runtime")
    observed = provenance.get("inputs", [])
    if not isinstance(observed, list) or any(not isinstance(item, dict) for item in observed):
        raise ValidationError("report inputs must list the verified component identities")
    fields = ("component_id", "sha256", "size_bytes")

    def identity(item):
        return tuple(item.get(field) for field in fields)

    expected_inputs = manifest["inputs"]
    if len({item.get("component_id") for item in expected_inputs}) != len(expected_inputs):
        raise ValidationError("trusted manifest contains duplicate component inputs")
    if {item.get("component_id") for item in expected_inputs} != set(manifest["expected_component_ids"]):
        raise ValidationError("trusted manifest inputs do not cover the complete profile")
    if (
        len(observed) != len(expected_inputs)
        or any(identity(item) not in [identity(expected) for expected in expected_inputs] for item in observed)
        or len({item.get("component_id") for item in observed}) != len(observed)
    ):
        raise ValidationError("report inputs differ from the complete trusted profile")
    return {
        "model_id": manifest["model_id"],
        "build_profile_id": manifest["build_profile_id"],
        "stage_key": expected["stage_key"],
        "expected_component_ids": manifest["expected_component_ids"],
        "inputs": [{key: value for key, value in item.items() if key != "artifact_key"} for item in manifest["inputs"]],
        "source": expected["source"],
        "environment": {"runtime": manifest["runtime"]},
    }


def publish_report(
    *,
    manifest: dict[str, Any],
    result_path: Path,
    api_url: str,
    aws_region: str,
    bucket: str,
    environment: str,
    test_status: str,
    kms_key_id: str | None = None,
    github_run_url: str | None = None,
    request=None,
    artifact_store=None,
) -> dict[str, Any]:
    path = Path(result_path).resolve()
    report = load_report(path)
    trusted = validate_publication(report, manifest)
    # Required scores are whole-model records. Optional diagnostics cannot satisfy them.
    validate_report(report, required_metrics=manifest["stages"][report["stage"]]["required_metrics"])
    test_status = require_test_outcome(test_status)
    files = [(path, "model-report", "application/json")]
    for index, evidence in enumerate(report["evidence"]):
        evidence_path = (path.parent / evidence["path"]).resolve()
        if Path(evidence["path"]).is_absolute() or not evidence_path.is_relative_to(path.parent):
            raise ValidationError("evidence path escapes report directory")
        if not evidence_path.is_file():
            raise ValidationError(f"evidence file is missing: {evidence['path']}")
        actual_sha, size = sha256_file(evidence_path)
        if evidence.get("sha256") != actual_sha or evidence.get("size_bytes") != size:
            raise ValidationError(f"evidence checksum differs: {evidence['path']}")
        files.append((evidence_path, f"evidence-{index}", evidence["media_type"]))
    stage = report["stage"]
    key = trusted["stage_key"]
    digest = canonical_sha256(report)
    stable = f"model-report-{key}-{digest}"
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)
    observed_environment = report["provenance"]["environment"]
    body = {
        "schema_version": 1,
        "run_type": "model_benchmark" if stage == "performance" else "model_accuracy",
        "commit_sha": trusted["source"]["models_commit"],
        "git_ref": f"refs/heads/{manifest['branch']}",
        "run_fingerprint": key,
        "environment": {
            "fingerprint": canonical_sha256(observed_environment),
            "fingerprint_version": PERFORMANCE_FINGERPRINT_VERSION
            if stage == "performance"
            else ACCURACY_FINGERPRINT_VERSION,
            "toolchain_name": "sima-neat",
            "toolchain_version": observed_environment.get("core_version"),
            "properties": {"vulcan_environment": environment, **observed_environment},
        },
        "metadata": {
            "model_id": manifest["model_id"],
            "build_profile_id": manifest["build_profile_id"],
            "stage": stage,
            "stage_key": key,
        },
    }
    if github_run_url:
        body.update(ci_provider="github-actions", ci_run_url=github_run_url)
    run_id = client.request("POST", "/v1/runs", body, f"{stable}-create")["run_id"]
    parameters = {
        "stage": stage,
        "stage_key": key,
        "model_id": manifest["model_id"],
        "build_profile_id": manifest["build_profile_id"],
        "runtime": manifest["runtime"],
    }
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/parameters",
        {
            "parameters": parameters,
            "canonical_sha256": canonical_sha256(parameters),
            "canonicalization_version": 1,
        },
        f"{stable}-parameters",
    )
    for item in manifest["inputs"]:
        component = item["component_id"]
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/inputs/compiled-model-{component.replace('_', '-')}",
            {
                "input_type": "model-pack",
                "source_reference": f"vulcan://runs/{item['compile_run_id']}/artifacts/{item['artifact_key']}",
                **{field: item[field] for field in ("sha256", "size_bytes", "s3_bucket", "s3_key", "s3_version_id")},
                "metadata": {
                    "model_id": manifest["model_id"],
                    "build_profile_id": manifest["build_profile_id"],
                    "component_id": component,
                    "compile_run_id": item["compile_run_id"],
                    "artifact_id": item["artifact_id"],
                },
            },
            f"{stable}-input-{component}",
        )
    profile_path = f"{manifest['model_id']}/{manifest['build_profile_id']}"
    prefix = f"models/{quote(manifest['branch'], safe='')}/{profile_path}/reports/{stage}/{key}/{digest}"
    artifact_id = None
    for file, name, media_type in files:
        sha, _ = sha256_file(file)
        stored = store.ensure_native_sha256(file, f"{prefix}/{name}/{file.name}", sha, media_type=media_type)
        artifact = client.request(
            "PUT",
            f"/v1/runs/{run_id}/artifacts/{name}",
            {
                "artifact_type": "benchmark-result" if name == "model-report" else "report-evidence",
                "s3_bucket": bucket,
                "s3_key": stored.key,
                "s3_version_id": stored.version_id,
                "sha256": stored.sha256,
                "size_bytes": stored.size_bytes,
                "media_type": media_type,
                "retention_class": "branch",
                "metadata": {
                    "model_id": manifest["model_id"],
                    "build_profile_id": manifest["build_profile_id"],
                    "stage": stage,
                },
            },
            f"{stable}-artifact-{name}",
        )
        if name == "model-report":
            artifact_id = artifact["artifact_id"]
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/test-results/{stage}/model",
        {
            "status": test_status,
            "version": "1",
            "report_artifact_id": artifact_id,
            "summary": {"stage": stage},
        },
        f"{stable}-test",
    )
    client.request(
        "PUT", f"/v1/runs/{run_id}/model-report", {"report": report, "manifest": trusted}, f"{stable}-report"
    )
    client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable}-finalize")
    return {
        "run_id": run_id,
        "model_id": manifest["model_id"],
        "build_profile_id": manifest["build_profile_id"],
        "stage": stage,
        "stage_key": key,
        "test_status": test_status,
    }
