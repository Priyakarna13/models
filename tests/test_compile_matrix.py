from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from model_registry.compile_matrix import build_matrix
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model, load_models
from model_registry.stage_keys import (
    ACCURACY_TOOLING,
    COMPILE_TOOLING,
    PERFORMANCE_TOOLING,
    accuracy_key,
    compile_key,
    performance_key,
)

ROOT = Path(__file__).resolve().parents[1]
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]
NATIVE_BASE_KEYS = {
    "gemma_4_e2b_it_textonly": "6ce2e4f84bc280a85aa6ea6896d793b3d02950c12cd9abc9f031b3aa66f1bdb4",
    "qwen3_5_0_8b": "1cf204eed4f2c104f480e3196be8c5cc58dfb2fa6530d67039b9fbd85549f5f0",
    "whisper_small": "73322147deeed19e453183ebd4c8f8265f13547d873d180cc77d1b1cea917012",
}


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for relative in ("models", "schemas", "taxonomies", "deps"):
        shutil.copytree(ROOT / relative, tmp_path / relative, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in {*COMPILE_TOOLING, *PERFORMANCE_TOOLING, *ACCURACY_TOOLING}:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def key(root: Path, name: str = "resnet_50", profile: str = "w8a8") -> str:
    return compile_key(root, name, profile, "default" if name.startswith("rf_detr_") else "model", COMPILER_VERSION)


def edit_yaml(root: Path, name: str, change) -> None:
    path = root / "models" / name / "model.yaml"
    value = yaml.safe_load(path.read_text())
    change(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_catalog_inventory_is_preserved() -> None:
    models = load_models(ROOT)
    matrix = build_matrix(models, sorted(models))["include"]
    assert len(models) == 27
    assert len(matrix) == 50
    assert sum(len(entry["component_ids"]) for entry in matrix) == 58
    assert sum(entry["performance"] for entry in matrix) == 47
    assert sum(entry["accuracy"] for entry in matrix) == 39


@pytest.mark.parametrize("name", list(NATIVE_BASE_KEYS))
def test_native_compile_keys_match_pr346_baseline(name: str) -> None:
    model = load_and_validate_model(ROOT / "models" / name)
    assert key(ROOT, name, model["build_profiles"][0]["id"]) == NATIVE_BASE_KEYS[name]


def test_grouped_compile_and_measurement_are_one_profile() -> None:
    model = load_and_validate_model(ROOT / "models/rf_detr_large")
    matrix = build_matrix({model["id"]: model}, [model["id"]])["include"]
    assert len(matrix) == 2
    assert all(entry["component_ids"] == ["backbone", "decoder"] for entry in matrix)
    assert all(entry["performance"] and entry["accuracy"] for entry in matrix)
    assert all("benchmark_component_ids" not in entry for entry in matrix)


def test_partial_component_selection_is_rejected() -> None:
    models = load_models(ROOT)
    with pytest.raises(ValidationError, match="complete model build"):
        build_matrix(models, ["rf_detr_large"], component_ids=["decoder"])


@pytest.mark.parametrize(
    "models,profiles,components",
    [(["unknown"], None, None), (["resnet_50"], ["unknown"], None), (["resnet_50"], None, ["unknown"])],
)
def test_unknown_selection_fails(models, profiles, components) -> None:
    with pytest.raises(ValidationError, match="unknown"):
        build_matrix(load_models(ROOT), models, profiles, components)


def test_profile_filter_retains_expensive_runner_grant_request() -> None:
    rows = build_matrix(load_models(ROOT), ["gemma_4_e2b_it_textonly"], ["w4a16"])["include"]
    assert rows[0]["requested_runner_class"] == "expensive"
    assert rows[0]["build_on_demand"] is True
    assert rows[0]["runner_script"] == "scripts/run_llima_compilation.sh"


@pytest.mark.parametrize(
    "relative,changes",
    [
        ("scripts/compile.py", True),
        ("scripts/calibration.py", True),
        ("scripts/performance.py", False),
        ("scripts/accuracy.py", False),
        ("README.md", False),
        ("tests/tests.yaml", False),
    ],
)
def test_compile_key_tracks_only_executed_compile_code(repository: Path, relative: str, changes: bool) -> None:
    before = key(repository)
    path = repository / "models/resnet_50" / relative
    path.write_text(path.read_text() + "\n# edited\n")
    assert (key(repository) != before) is changes


@pytest.mark.parametrize(
    "field,value",
    [
        ("description", "Updated description"),
        ("display_name", "Different display name"),
        ("model_family", "resnet"),
        ("build_on_demand", True),
    ],
)
def test_catalog_metadata_does_not_invalidate_compile(repository: Path, field: str, value: object) -> None:
    before = key(repository)
    edit_yaml(repository, "resnet_50", lambda model: model.update({field: value}))
    assert key(repository) == before


def test_calibration_pin_and_parameters_invalidate_only_affected_profile(repository: Path) -> None:
    before = {profile: key(repository, profile=profile) for profile in ("w8a8", "w16a16")}
    edit_yaml(
        repository,
        "resnet_50",
        lambda model: model["build_profiles"]["w8a8"]["components"]["model"]["parameters"].update(
            calibration_samples=501
        ),
    )
    assert key(repository) != before["w8a8"]
    assert key(repository, profile="w16a16") == before["w16a16"]


def test_grouped_key_tracks_decoder_configuration(repository: Path) -> None:
    before = key(repository, "rf_detr_large", "w8a8_w16a16")
    edit_yaml(
        repository,
        "rf_detr_large",
        lambda model: model["build_profiles"]["w8a8_w16a16"]["components"]["decoder"]["parameters"].update(
            calibration_samples=51
        ),
    )
    assert key(repository, "rf_detr_large", "w8a8_w16a16") != before


def test_tiny_drone_helpers_are_owned_by_model(repository: Path) -> None:
    before = key(repository, "yolo26_tiny_drone")
    path = repository / "models/yolo26_tiny_drone/scripts/deployment.py"
    path.write_text(path.read_text() + "\n# changed audit\n")
    assert key(repository, "yolo26_tiny_drone") != before
    assert not (repository / "training").exists()


def test_shared_loader_invalidates_onnx_but_not_native(repository: Path) -> None:
    onnx, native = key(repository), key(repository, "qwen3_5_0_8b", "w4a16")
    path = repository / "tools/model_registry/model.py"
    path.write_text(path.read_text() + "\n# changed loader\n")
    assert key(repository) != onnx
    assert key(repository, "qwen3_5_0_8b", "w4a16") == native


@pytest.mark.parametrize(
    "name,field,value", [("qwen3_5_0_8b", "max_num_tokens", 2048), ("qwen3_5_0_8b", "language_group_size", 64)]
)
def test_native_graph_parameters_invalidate(repository: Path, name: str, field: str, value: int) -> None:
    before = key(repository, name, "w4a16")
    edit_yaml(
        repository,
        name,
        lambda model: model["build_profiles"]["w4a16"]["components"]["model"]["parameters"].update({field: value}),
    )
    assert key(repository, name, "w4a16") != before


def builds(model: dict) -> dict:
    return {
        component["id"]: {"run_id": component["id"], "artifact_sha256": str(i) * 64}
        for i, component in enumerate(model["components"], 1)
    }


def test_model_report_key_requires_all_exact_inputs() -> None:
    model = load_and_validate_model(ROOT / "models/rf_detr_large")
    inputs = builds(model)
    before = performance_key(ROOT, model, "w8a8_w16a16", inputs)
    inputs["backbone"]["artifact_sha256"] = "f" * 64
    assert performance_key(ROOT, model, "w8a8_w16a16", inputs) != before
    del inputs["backbone"]
    with pytest.raises(ValidationError, match="exactly the compiled components"):
        performance_key(ROOT, model, "w8a8_w16a16", inputs)


def test_report_key_is_independent_of_component_order() -> None:
    model = load_and_validate_model(ROOT / "models/rf_detr_large")
    inputs = builds(model)
    before = performance_key(ROOT, model, "w8a8_w16a16", inputs)
    model["components"].reverse()
    assert performance_key(ROOT, model, "w8a8_w16a16", inputs) == before


@pytest.mark.parametrize("field,value,changes", [("display_name", "Better label", False), ("unit", "seconds", True)])
def test_report_key_uses_only_applicable_metric_semantics(
    repository: Path, field: str, value: str, changes: bool
) -> None:
    model = load_and_validate_model(repository / "models/resnet_50")
    before = performance_key(repository, model, "w8a8", builds(model))
    path = repository / "taxonomies/metrics.json"
    catalog = json.loads(path.read_text())
    catalog["model.latency.avg"][field] = value
    catalog["unrelated.metric"] = {"unit": "count", "type": "number", "direction": "higher", "stage": "performance"}
    path.write_text(json.dumps(catalog))
    assert (performance_key(repository, model, "w8a8", builds(model)) != before) is changes


def test_explicit_runtime_identity_enters_both_measurement_keys() -> None:
    model = load_and_validate_model(ROOT / "models/resnet_50")
    for function in (performance_key, accuracy_key):
        assert function(ROOT, model, "w8a8", builds(model), runtime={"sha256": "a" * 64}) != function(
            ROOT, model, "w8a8", builds(model), runtime={"sha256": "b" * 64}
        )
