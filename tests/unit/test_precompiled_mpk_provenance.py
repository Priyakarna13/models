from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from model_registry.catalog import build_matrix, discover_catalog, model_build_fingerprint
from model_registry.errors import ValidationError
from model_registry.model import load_model, validate_model
from model_registry.registry import S3Object, publish_compilations

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "runtime_lightglue_superpoint_600"
VARIANT_ID = "modalix_int8_neat68_regression"
MODEL_DIR = ROOT / "models" / MODEL_ID
WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Compile.yml"


class FakeStore:
    def ensure_native_sha256(self, path: Path, key: str, sha256: str) -> S3Object:
        return S3Object(key=key, version_id="version-1", sha256=sha256, size_bytes=path.stat().st_size)


def test_precompiled_mpk_source_requires_explicit_compiler_provenance() -> None:
    model = load_model(MODEL_DIR)
    validate_model(MODEL_DIR, model)

    missing = copy.deepcopy(model)
    del missing["source"]["compiler_provenance"]
    with pytest.raises(ValidationError, match="source.compiler_provenance.*required"):
        validate_model(MODEL_DIR, missing)


def test_precompiled_mpk_matrix_uses_producer_compiler_for_provenance_and_fingerprint() -> None:
    catalog = discover_catalog(ROOT)
    matrix = build_matrix(
        catalog,
        [MODEL_ID],
        repository=ROOT,
        compiler_version="2.1.3",
    )

    assert matrix == {
        "include": [
            {
                "model_dir": f"models/{MODEL_ID}",
                "model_id": MODEL_ID,
                "recipe": "import_mpk.py",
                "target": "modalix",
                "variant_id": VARIANT_ID,
                "provenance_compiler_version": "2.1.0",
                "provenance_compiler_label": "model-sdk-2.1.0",
                "build_fingerprint": model_build_fingerprint(ROOT, MODEL_ID, "2.1.0"),
            }
        ]
    }


def test_compile_workflow_separates_runner_and_packaged_artifact_compilers() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    compile_job = workflow["jobs"]["compile"]
    verify_step = next(step for step in compile_job["steps"] if step.get("name") == "Validate and checksum the MPK")

    assert compile_job["env"]["EXPECTED_COMPILER_VERSION"] == "${{ needs.select.outputs.model_compiler_version }}"
    assert compile_job["env"]["PROVENANCE_COMPILER_VERSION"] == (
        "${{ matrix.provenance_compiler_version || needs.select.outputs.model_compiler_version }}"
    )
    assert compile_job["env"]["PROVENANCE_COMPILER_LABEL"] == (
        "${{ matrix.provenance_compiler_label || needs.select.outputs.model_compiler_label }}"
    )
    assert '--compiler-version "$PROVENANCE_COMPILER_VERSION"' in verify_step["run"]
    assert '--compiler-label "$PROVENANCE_COMPILER_LABEL"' in verify_step["run"]


def test_registry_records_precompiled_mpk_source_and_producer_compiler(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    output = dist / MODEL_ID / VARIANT_ID
    output.mkdir(parents=True)
    archive = output / "fixture_mpk.tar.gz"
    archive.write_bytes(b"immutable precompiled MPK")
    artifact_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    model = discover_catalog(ROOT)[MODEL_ID]
    provenance = {
        "model_id": MODEL_ID,
        "variant_id": VARIANT_ID,
        "target": "modalix",
        "compiler_version": "2.1.0",
        "compiler_label": "model-sdk-2.1.0",
        "source_sha256": model["source"]["sha256"],
        "git_sha": "a" * 40,
        "environment": "staging",
        "build_fingerprint": model_build_fingerprint(ROOT, MODEL_ID, "2.1.0"),
    }
    (output / "compile-result.json").write_text(
        json.dumps(
            {
                "archive": archive.name,
                "sha256": artifact_sha,
                "size_bytes": archive.stat().st_size,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        return {"run_id": "00000000-0000-4000-8000-000000000001"} if path == "/v1/runs" else {"ok": True}

    result = publish_compilations(
        repository=ROOT,
        dist_dir=dist,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        kms_key_id=None,
        environment="staging",
        branch="feature/import",
        git_sha="a" * 40,
        github_run_id="1234",
        github_run_attempt="1",
        github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
        request=request,
        artifact_store=FakeStore(),
    )

    assert result["count"] == 1
    create = calls[0][2]
    source = calls[1][2]
    parameters = calls[2][2]
    assert create is not None
    assert source is not None
    assert parameters is not None
    assert create["environment"]["toolchain_version"] == "2.1.0"
    assert create["environment"]["properties"]["compiler_label"] == "model-sdk-2.1.0"
    assert source["input_type"] == "precompiled_mpk"
    assert source["media_type"] == "application/gzip"
    assert parameters["parameters"]["compiler_version"] == "2.1.0"
