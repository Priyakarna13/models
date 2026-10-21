from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.model import component_pipelines, load_and_validate_model, load_model, validate_model
from model_registry.taxonomy import load_taxonomy, validate_model_categorization

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/model.schema.json"
TAXONOMY = ROOT / "taxonomies/model-taxonomy.yaml"
FIXTURES = ROOT / "tests/fixtures"


def authored(name: str = "resnet_50") -> dict:
    return yaml.safe_load((ROOT / "models" / name / "model.yaml").read_text())


def check(value: dict, name: str = "resnet_50") -> None:
    validate_model(ROOT / "models" / name, value, SCHEMA, TAXONOMY)


def test_valid_fixture_passes() -> None:
    assert load_and_validate_model(FIXTURES / "fixture_model")["id"] == "fixture_model"


def test_invalid_fixture_reports_field_paths() -> None:
    with pytest.raises(ValidationError, match="compilation.source"):
        load_and_validate_model(FIXTURES / "invalid_model")


def test_catalog_model_is_valid() -> None:
    for path in (ROOT / "models").glob("*/model.yaml"):
        load_and_validate_model(path.parent)


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "architecture",
        "modalities",
        "maturity",
        "owners",
        "components",
        "packages",
        "compositions",
        "benchmark",
        "fidelity",
        "build-on-demand",
    ],
)
def test_obsolete_authored_fields_are_rejected(field: str) -> None:
    value = authored()
    value[field] = "obsolete"
    with pytest.raises(ValidationError, match="Additional properties"):
        check(value)


@pytest.mark.parametrize(
    "field", ["display_name", "model_type", "tasks", "model_family", "compilation", "build_profiles", "performance"]
)
def test_required_sections_are_not_implicit(field: str) -> None:
    value = authored()
    del value[field]
    with pytest.raises(ValidationError, match="required"):
        check(value)


def test_duplicate_yaml_keys_fail_before_normalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_ROOT", str(ROOT))
    path = tmp_path / "model.yaml"
    path.write_text("model_family: resnet\nmodel_family: rf_detr\n")
    with pytest.raises(ValidationError, match="duplicate YAML key: 'model_family'"):
        load_model(tmp_path)


def test_yaml_uses_safe_loader(tmp_path: Path) -> None:
    (tmp_path / "model.yaml").write_text("!!python/object/apply:os.system ['echo unsafe']")
    with pytest.raises(ValidationError, match="could not load YAML"):
        load_model(tmp_path)


@pytest.mark.parametrize("runner", ["default", "expensive", "most-expensive"])
def test_runner_classes_are_preserved(runner: str) -> None:
    value = authored()
    value["build_profiles"]["w8a8"]["runner_class"] = runner
    check(value)


def test_invalid_runner_class_fails() -> None:
    value = authored()
    value["build_profiles"]["w8a8"]["runner_class"] = "arbitrary-instance"
    with pytest.raises(ValidationError, match="runner_class"):
        check(value)


def test_unsupported_build_requires_reason() -> None:
    value = authored()
    value["build_profiles"]["w8a8"]["components"]["model"]["status"] = "unsupported"
    with pytest.raises(ValidationError, match="reason"):
        check(value)


@pytest.mark.parametrize("components", [{}, {"other": {"parameters": {}}}])
def test_profile_requires_complete_component_set(components: dict) -> None:
    value = authored()
    value["build_profiles"]["w8a8"]["components"] = components
    with pytest.raises(ValidationError):
        check(value)


def test_multi_component_invocation_requires_one_runner_class() -> None:
    value = authored("rf_detr_large")
    value["build_profiles"]["w8a8_w16a16"]["components"]["decoder"]["runner_class"] = "expensive"
    with pytest.raises(ValidationError, match="one runner class"):
        check(value, "rf_detr_large")


@pytest.mark.parametrize(
    "pipeline", [[{"component": "backbone"}], [{"component": "backbone"}, {"component": "backbone"}]]
)
def test_pipeline_must_list_each_component_once(pipeline: list) -> None:
    value = authored("rf_detr_large")
    value["pipeline"] = pipeline
    with pytest.raises(ValidationError, match="pipeline"):
        check(value, "rf_detr_large")


def test_pipeline_is_display_metadata() -> None:
    model = load_model(ROOT / "models/rf_detr_large")
    assert component_pipelines({model["id"]: model})[model["id"]] == [
        {"component": "backbone"},
        {"cpu": "Top 300 proposals"},
        {"component": "decoder"},
    ]


@pytest.mark.parametrize("alias", ["missing", ""])
def test_calibration_dataset_must_exist(alias: str) -> None:
    value = authored()
    value["build_profiles"]["w8a8"]["calibration_dataset"] = alias
    with pytest.raises(ValidationError):
        check(value)


@pytest.mark.parametrize("stage", ["numerical_fidelity", "accuracy"])
def test_measurement_dataset_must_be_declared(stage: str) -> None:
    value = authored()
    value[stage]["dataset"] = "missing"
    with pytest.raises(ValidationError, match="unknown dataset"):
        check(value)


def test_source_policy_is_enforced() -> None:
    value = authored()
    value["compilation"]["source"]["uri"] = "https://unapproved.example/model.onnx"
    with pytest.raises(ValidationError, match="source policy"):
        check(value)


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "uri"])
def test_source_integrity_fields_are_required(field: str) -> None:
    value = authored()
    del value["compilation"]["source"][field]
    with pytest.raises(ValidationError, match="compilation.source"):
        check(value)


def test_model_family_preserves_paradigm() -> None:
    model = load_model(ROOT / "models/rf_detr_large")
    categories = validate_model_categorization(model, load_taxonomy(TAXONOMY))
    assert categories["model_family"] == "rf_detr"
    assert categories["architecture_paradigm"] == "transformer"
    assert "modalities" not in categories


@pytest.mark.parametrize("field,value", [("tasks", ["unknown"]), ("model_family", "unknown"), ("model_type", "llm")])
def test_taxonomy_references_are_checked(field: str, value: object) -> None:
    model = authored()
    model[field] = value
    with pytest.raises(ValidationError):
        check(model)


@pytest.mark.parametrize("field", ["spdx", "attribution", "customer_redistribution"])
def test_license_requires_terms_attribution_and_customer_policy(field: str) -> None:
    value = authored()
    del value["license"][field]
    with pytest.raises(ValidationError, match="license"):
        check(value)


def test_meaningful_attribution_is_retained() -> None:
    value = authored("rf_detr_large")
    assert "Roboflow" in value["license"]["attribution"]
    assert value["license"]["customer_redistribution"] == "prohibited"


@pytest.mark.parametrize("stage", ["accuracy", "performance"])
def test_measurement_requires_nonempty_metric_contract(stage: str) -> None:
    value = authored()
    value[stage]["required_metrics"] = []
    with pytest.raises(ValidationError):
        check(value)


def test_unregistered_accuracy_metric_fails() -> None:
    value = authored()
    value["accuracy"]["required_metrics"] = ["model.accuracy.typo"]
    with pytest.raises(ValidationError, match="unregistered.*metrics"):
        check(value)


@pytest.mark.parametrize(
    "missing",
    ["scripts/compile.py", "scripts/performance.py", "scripts/accuracy.py", "scripts/requirements-accuracy.txt"],
)
def test_required_script_files_are_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_ROOT", str(ROOT))
    directory = tmp_path / "resnet_50"
    shutil.copytree(ROOT / "models/resnet_50", directory)
    (directory / missing).unlink()
    with pytest.raises(ValidationError, match="referenced file does not exist"):
        load_and_validate_model(directory)


@pytest.mark.parametrize("entry", ["recipes", "components", "resources", "notes.md"])
def test_unexpected_model_folder_entries_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_ROOT", str(ROOT))
    directory = tmp_path / "resnet_50"
    shutil.copytree(ROOT / "models/resnet_50", directory)
    (directory / entry).mkdir()
    (directory / entry / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ValidationError, match="unexpected model folder entries"):
        load_and_validate_model(directory)


def test_native_compilation_requires_complete_checkpoint_file_set() -> None:
    value = authored("qwen3_5_0_8b")
    value["compilation"]["source"] = authored()["compilation"]["source"]
    with pytest.raises(ValidationError, match="file set"):
        check(value, "qwen3_5_0_8b")


def test_performance_required_metrics_use_registered_stage_vocabulary() -> None:
    value = authored()
    value["performance"]["required_metrics"].append("model.accuracy.coco_map")
    with pytest.raises(ValidationError, match="performance.*metric"):
        check(value)


def test_pipeline_follows_declared_component_order() -> None:
    value = authored("rf_detr_large")
    value["pipeline"] = [{"component": "decoder"}, {"component": "backbone"}]
    with pytest.raises(ValidationError, match="pipeline"):
        check(value, "rf_detr_large")
