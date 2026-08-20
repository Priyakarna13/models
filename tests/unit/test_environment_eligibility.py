from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from model_registry.catalog import build_matrix
from model_registry.cli import main
from model_registry.errors import ValidationError
from model_registry.model import load_model, validate_model
from model_registry.package import package_compiled_models

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Compile.yml"


def _catalog() -> dict[str, dict]:
    policies = {
        "legacy": None,
        "nowhere": [],
        "production_only": ["production"],
        "staging_only": ["staging"],
    }
    return {
        model_id: {
            "maturity": "experimental",
            "source": {},
            "compile_variants": [
                {
                    "id": "modalix",
                    "target": "modalix",
                    "recipe": "compile.py",
                    "parameters": {},
                }
            ],
            **({"publication": {"environments": environments}} if environments is not None else {}),
        }
        for model_id, environments in policies.items()
    }


def test_matrix_filters_catalog_models_by_explicit_publication_environment() -> None:
    catalog = _catalog()

    staging = build_matrix(catalog, list(catalog), environment="staging")
    production = build_matrix(catalog, list(catalog), environment="production")

    assert [entry["model_id"] for entry in staging["include"]] == ["legacy", "staging_only"]
    assert [entry["model_id"] for entry in production["include"]] == ["legacy", "production_only"]


def test_explicit_ineligible_production_model_fails_with_evidence() -> None:
    with pytest.raises(
        ValidationError,
        match=r"not eligible for the production catalog: staging_only .*publication environments: staging",
    ):
        build_matrix(
            _catalog(),
            ["staging_only"],
            environment="production",
            require_all_models_eligible=True,
        )


def test_production_all_cli_excludes_staging_lightglue(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--all", "--environment", "production"])

    assert result == 0
    model_ids = {entry["model_id"] for entry in json.loads(capsys.readouterr().out)["include"]}
    assert "runtime_lightglue_superpoint_600" not in model_ids
    assert "resnet_50" in model_ids


def test_packaging_reapplies_environment_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = {"staging_only": _catalog()["staging_only"]}
    monkeypatch.setattr("model_registry.package.discover_catalog", lambda _root: catalog)
    compiled = tmp_path / "compiled"
    compiled.mkdir()

    result = package_compiled_models(
        ROOT,
        compiled,
        tmp_path / "dist",
        "a" * 40,
        environment="production",
    )

    assert result["models"] == []
    latest = json.loads((tmp_path / "dist" / "latest.json").read_text(encoding="utf-8"))
    assert latest["models"] == {}


def test_packaging_rejects_ineligible_compiled_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = {"staging_only": _catalog()["staging_only"]}
    monkeypatch.setattr("model_registry.package.discover_catalog", lambda _root: catalog)
    result = tmp_path / "compiled" / "staging_only" / "modalix" / "compile-result.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ValidationError,
        match=r"compiled output.*not eligible for the production catalog.*publication environments: staging",
    ):
        package_compiled_models(
            ROOT,
            tmp_path / "compiled",
            tmp_path / "dist",
            "a" * 40,
            environment="production",
        )


def test_restricted_redistribution_requires_explicit_publication_policy() -> None:
    fixture = ROOT / "tests" / "fixtures" / "fixture_model"
    model = copy.deepcopy(load_model(fixture))
    model["license"]["redistribution"] = "restricted"

    with pytest.raises(ValidationError, match="publication.*required"):
        validate_model(fixture, model, ROOT / "schemas" / "model.schema.json")


def test_workflow_passes_resolved_environment_to_every_matrix_path() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    matrix_step = next(step for step in workflow["jobs"]["select"]["steps"] if step.get("id") == "matrix")

    assert matrix_step["env"]["VULCAN_ENVIRONMENT"] == "${{ steps.environment.outputs.vulcan_environment }}"
    assert matrix_step["run"].count('--environment "$VULCAN_ENVIRONMENT"') == 3
