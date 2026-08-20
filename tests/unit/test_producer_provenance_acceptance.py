from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.catalog import discover_catalog
from model_registry.dependencies import (
    model_producer_compiler,
    resolve_model_compiler,
    validate_result_producer_compiler,
)
from model_registry.errors import ValidationError
from model_registry.package import package_compiled_models
from model_registry.registry import publish_compilations

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "runtime_lightglue_superpoint_600"
VARIANT_ID = "modalix_int8_neat68_regression"
GIT_SHA = "a" * 40


def _write_result(root: Path, *, compiler_version: str, compiler_label: str) -> Path:
    output = root / MODEL_ID / VARIANT_ID
    output.mkdir(parents=True)
    model = discover_catalog(ROOT)[MODEL_ID]
    (output / "compile-result.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "model_id": MODEL_ID,
                    "variant_id": VARIANT_ID,
                    "target": "modalix",
                    "compiler_version": compiler_version,
                    "compiler_label": compiler_label,
                    "source_sha256": model["source"]["sha256"],
                    "git_sha": GIT_SHA,
                    "environment": "staging",
                    "build_fingerprint": "b" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_package_rejects_result_with_wrong_precompiled_producer_version(tmp_path: Path) -> None:
    compiled = _write_result(
        tmp_path / "compiled",
        compiler_version="2.1.3",
        compiler_label="model-sdk-2.1.0",
    )

    with pytest.raises(
        ValidationError,
        match=r"provenance compiler_version must be '2.1.0'.*found '2.1.3'",
    ):
        package_compiled_models(ROOT, compiled, tmp_path / "dist", GIT_SHA)


def test_registry_rejects_result_with_wrong_precompiled_producer_label(tmp_path: Path) -> None:
    dist = _write_result(
        tmp_path / "dist",
        compiler_version="2.1.0",
        compiler_label="model-compiler-2.1.3",
    )

    with pytest.raises(
        ValidationError,
        match=r"provenance compiler_label must be 'model-sdk-2.1.0'.*found 'model-compiler-2.1.3'",
    ):
        publish_compilations(
            repository=ROOT,
            dist_dir=dist,
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            kms_key_id=None,
            environment="staging",
            branch="feature/import",
            git_sha=GIT_SHA,
            github_run_id="1234",
            github_run_attempt="1",
            github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
            request=lambda *_args: {},
        )


def test_onnx_source_keeps_repository_compiler_as_expected_producer() -> None:
    model = discover_catalog(ROOT)["resnet_50"]
    repository_compiler = resolve_model_compiler(ROOT)

    assert model_producer_compiler(model, repository_compiler) == {
        "label": "model-compiler-2.1.3",
        "version": "2.1.3",
    }
    validate_result_producer_compiler(
        model,
        {"compiler_version": "2.1.3", "compiler_label": "current-artifact-label"},
        repository_compiler,
        context="fixture",
    )
    with pytest.raises(ValidationError, match="compiler_version must be '2.1.3'"):
        validate_result_producer_compiler(
            model,
            {"compiler_version": "2.1.2", "compiler_label": "current-artifact-label"},
            repository_compiler,
            context="fixture",
        )
