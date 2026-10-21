from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.model import component_pipelines, load_and_validate_model, load_model, validate_model
from model_registry.taxonomy import load_taxonomy, validate_model_categorization

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FIXTURE = FIXTURES / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"


def _authored(directory: Path = FIXTURE) -> dict:
    return yaml.safe_load((directory / "model.yaml").read_text(encoding="utf-8"))


PARAMETERS = {"parameters": {}}


def _two_component_model() -> dict:
    authored = _authored()
    authored["compilation"]["components"] = ["backbone", "decoder"]
    authored["build_profiles"]["fixture"]["components"] = {"backbone": PARAMETERS, "decoder": PARAMETERS}
    return authored


def test_valid_fixture_passes() -> None:
    model = load_and_validate_model(FIXTURE, SCHEMA)
    assert model["id"] == "fixture_model"


def test_invalid_fixture_reports_field_paths() -> None:
    with pytest.raises(ValidationError) as error:
        load_and_validate_model(FIXTURES / "invalid_model", SCHEMA)
    message = str(error.value)
    assert "compilation.source" in message
    assert "not valid under any of the given schemas" in message


def test_model_id_is_its_directory_name(tmp_path: Path) -> None:
    directory = tmp_path / "renamed_model"
    shutil.copytree(FIXTURE, directory)

    assert load_and_validate_model(directory, SCHEMA)["id"] == "renamed_model"
    with pytest.raises(ValidationError, match="'id' was unexpected"):
        validate_model(directory, {**_authored(), "id": "fixture_model"}, SCHEMA)


def test_runner_class_is_optional_and_validated() -> None:
    authored = _authored()
    validate_model(FIXTURE, authored, SCHEMA)
    authored["build_profiles"]["fixture"]["runner_class"] = "expensive"
    assert validate_model(FIXTURE, authored, SCHEMA)["build_profiles"][0]["runner_class"] == "expensive"
    authored["build_profiles"]["fixture"]["runner_class"] = "arbitrary-instance"
    with pytest.raises(ValidationError, match="runner_class"):
        validate_model(FIXTURE, authored, SCHEMA)


def test_build_profile_compiles_every_component_in_one_invocation_named_after_it() -> None:
    model = validate_model(FIXTURE, _two_component_model(), SCHEMA)

    assert [component["id"] for component in model["components"]] == ["backbone", "decoder"]
    assert model["build_profiles"][0]["compile_invocations"] == [
        {"id": "fixture", "components": ["backbone", "decoder"]}
    ]
    assert model["compositions"] == [{"id": "default", "type": "declarative"}]
    assert model["packages"][0]["components"] == ["backbone", "decoder"]


@pytest.mark.parametrize(
    "components",
    [{"backbone": PARAMETERS}, {"backbone": PARAMETERS, "decoder": PARAMETERS, "extra": PARAMETERS}],
    ids=["missing component", "unknown component"],
)
def test_build_profile_sets_the_parameters_of_exactly_the_compiled_components(components: dict) -> None:
    authored = _two_component_model()
    authored["build_profiles"]["fixture"]["components"] = components
    with pytest.raises(ValidationError, match="must set the parameters of exactly the components backbone, decoder"):
        validate_model(FIXTURE, authored, SCHEMA)


def test_build_profile_calibrates_its_first_component_on_a_declared_dataset() -> None:
    authored = _two_component_model()
    authored["build_profiles"]["fixture"]["calibration_dataset"] = "missing"
    with pytest.raises(ValidationError, match="names unknown calibration_dataset 'missing'"):
        validate_model(FIXTURE, authored, SCHEMA)

    pin = {"id": "fixture-set", "variant": "cal", "sha256": "a" * 64, "size_bytes": 1}
    authored["datasets"] = {"calibration": pin}
    authored["build_profiles"]["fixture"]["calibration_dataset"] = "calibration"
    model = validate_model(FIXTURE, authored, SCHEMA)

    assert model["calibrations"] == [{"id": "calibration", "status": "ready", "dataset": pin}]
    assert [binding.get("calibration") for binding in model["build_profiles"][0]["components"]] == ["calibration", None]


def test_pipeline_is_read_as_declared() -> None:
    authored = _two_component_model()
    authored["pipeline"] = [{"component": "backbone"}, {"cpu": "Top 10 boxes"}, {"component": "decoder"}]
    model = validate_model(FIXTURE, authored, SCHEMA)

    assert component_pipelines({"fixture_model": model}) == {"fixture_model": authored["pipeline"]}
    assert component_pipelines({"fixture_model": validate_model(FIXTURE, _two_component_model(), SCHEMA)}) == {}


@pytest.mark.parametrize(
    "pipeline",
    [
        [{"component": "backbone"}, {"cpu": "Top 10 boxes"}],
        [{"component": "backbone"}, {"component": "backbone"}, {"component": "decoder"}],
        [{"component": "backbone"}, {"component": "unknown"}],
        [{"cpu": "Top 10 boxes"}, {"cpu": "Sort boxes"}],
    ],
)
def test_pipeline_must_list_each_component_once(pipeline: list[dict]) -> None:
    authored = _two_component_model()
    authored["pipeline"] = pipeline
    with pytest.raises(ValidationError, match="pipeline must list each component exactly once"):
        validate_model(FIXTURE, authored, SCHEMA)


def test_unapproved_https_host_is_a_validation_error() -> None:
    authored = _authored()
    authored["compilation"]["source"]["uri"] = "https://unapproved.example/model.onnx"
    with pytest.raises(ValidationError, match="source policy.*not approved"):
        validate_model(FIXTURE, authored, SCHEMA)


@pytest.mark.parametrize("model_card", (
    "https://huggingface.co/example/model",
    "https://github.com/example/model",
    "https://docs.example.org/model.html",
))
def test_model_card_accepts_any_https_page(model_card: str) -> None:
    authored = _authored()
    authored["upstream"]["model_card"] = model_card
    assert validate_model(FIXTURE, authored, SCHEMA)["upstream"]["model_card"] == model_card


@pytest.mark.parametrize("model_card", ("http://example.org/model", "javascript:alert(1)", "hf://models/a/b"))
def test_model_card_must_be_https(model_card: str) -> None:
    authored = _authored()
    authored["upstream"]["model_card"] = model_card
    with pytest.raises(ValidationError, match="upstream.model_card"):
        validate_model(FIXTURE, authored, SCHEMA)


def test_missing_compile_script_fails(tmp_path: Path) -> None:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURE, directory)
    (directory / "scripts" / "compile.py").unlink()
    with pytest.raises(ValidationError, match="referenced file does not exist.*scripts/compile.py"):
        validate_model(directory, _authored(directory), SCHEMA)


def test_hugging_face_model_is_one_component_compiled_from_its_config() -> None:
    authored = _authored(ROOT / "models" / "qwen3_5_0_8b")
    authored["compilation"]["source"]["main_file"] = "tokenizer.json"
    with pytest.raises(ValidationError, match="main_file is config.json"):
        validate_model(ROOT / "models" / "qwen3_5_0_8b", authored, SCHEMA)


RUN_TESTS = {"run_tests": True}
DATASET = {"id": "fixture-set", "variant": "mini", "sha256": "a" * 64, "size_bytes": 1}
MEASUREMENTS = {
    "accuracy": {"status": "ready", "dataset": "fixture", "required_metrics": ["top1_accuracy"]},
    "performance": {"required_metrics": ["latency_avg_ms", "throughput_fps"]},
}
EXPECTED = {
    "accuracy": {"run_tests": True, "expected": {"fixture": {"top1_accuracy": 0.5}}},
    "performance": {"run_tests": True, "expected": {"fixture": {"throughput_fps": 9.0}}},
}


def _write_model_tests(directory: Path, tests: dict) -> None:
    (directory / "tests" / "tests.yaml").write_text(yaml.safe_dump(tests), encoding="utf-8")


def _measured_model(tmp_path: Path, *stages: str) -> tuple[Path, dict]:
    """A copy of the fixture that measures the given stages, with their scripts, test files and expected values."""
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURE, directory)
    (directory / "tests" / "measure").mkdir()
    authored = _authored(directory)
    authored["datasets"] = {"fixture": DATASET}
    tests = {"compile": RUN_TESTS}
    for stage in stages:
        (directory / "scripts" / f"{stage}.py").write_text("", encoding="utf-8")
        (directory / "tests" / "measure" / f"test_{stage}.py").write_text("", encoding="utf-8")
        authored[stage] = copy.deepcopy(MEASUREMENTS[stage])
        tests[stage] = EXPECTED[stage]
    _write_model_tests(directory, tests)
    return directory, authored


def test_measured_model_is_valid(tmp_path: Path) -> None:
    directory, authored = _measured_model(tmp_path, "accuracy", "performance")

    model = validate_model(directory, authored, SCHEMA)

    assert model["accuracy"] == {**MEASUREMENTS["accuracy"], "dataset": DATASET}
    assert model["performance"] == MEASUREMENTS["performance"]


@pytest.mark.parametrize("stage", ["accuracy", "performance"])
def test_measurement_requires_its_script(tmp_path: Path, stage: str) -> None:
    directory, authored = _measured_model(tmp_path, stage)
    (directory / "scripts" / f"{stage}.py").unlink()
    with pytest.raises(ValidationError, match=f"{stage} needs scripts/{stage}.py"):
        validate_model(directory, authored, SCHEMA)


@pytest.mark.parametrize(
    ("stage", "settings", "expected"),
    [
        (
            "accuracy", {"required_metrics": ["missing"]},
            "accuracy requires metrics model_report does not define: missing",
        ),
        # A performance metric is not one the accuracy script reports.
        ("accuracy", {"required_metrics": ["throughput_fps"]}, "does not define: throughput_fps"),
        ("accuracy", {"required_metrics": ["Score"]}, "required_metrics.*does not match"),
        ("accuracy", {"build_profiles": ["other"]}, "accuracy names unknown build profiles: other"),
        ("accuracy", {"dataset": "missing"}, "accuracy: unknown dataset 'missing'"),
        # Published scores live in the top-level reference_metrics.
        ("accuracy", {"reference": {"top1_accuracy": 0.5}}, "'reference' was unexpected"),
        ("performance", {"required_metrics": ["top1_accuracy"]}, "does not define: top1_accuracy"),
        ("performance", {"unsupported_setting": True}, "'unsupported_setting' was unexpected"),
    ],
)
def test_measurement_is_validated(tmp_path: Path, stage: str, settings: dict, expected: str) -> None:
    directory, authored = _measured_model(tmp_path, stage)
    authored[stage].update(settings)
    with pytest.raises(ValidationError, match=expected):
        validate_model(directory, authored, SCHEMA)


def _numerical_fidelity(**settings) -> dict:
    return {
        "status": "diagnostic",
        "dataset": "fixture",
        "selection": {"method": "seeded_random_without_replacement", "seed": 1, "offset": 0, "samples": 2},
        "outputs": ["output"],
        **settings,
    }


def test_numerical_fidelity_resolves_its_dataset() -> None:
    authored = {**_authored(), "datasets": {"fixture": DATASET}, "numerical_fidelity": _numerical_fidelity()}

    model = validate_model(FIXTURE, authored, SCHEMA)

    assert model["numerical_fidelity"] == _numerical_fidelity(dataset=DATASET)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"dataset": "missing"}, "numerical_fidelity: unknown dataset 'missing'"),
        ({"engine": "model_recipe"}, "'engine' was unexpected"),
        ({"outputs": []}, "numerical_fidelity.outputs"),
    ],
)
def test_numerical_fidelity_is_validated(settings: dict, expected: str) -> None:
    authored = {**_authored(), "datasets": {"fixture": DATASET}, "numerical_fidelity": _numerical_fidelity(**settings)}
    with pytest.raises(ValidationError, match=expected):
        validate_model(FIXTURE, authored, SCHEMA)


def _reference_metrics(**settings) -> dict:
    return {
        "source": "https://example.com/card",
        "dataset": {"id": "coco-2017", "split": "val"},
        "conditions": "640x640",
        "metrics": {"coco_map": 0.5},
        **settings,
    }


def test_reference_metrics_accept_scores_of_known_metrics() -> None:
    authored = _authored()
    authored["reference_metrics"] = [
        _reference_metrics(note="The card does not name the split."),
        _reference_metrics(dataset={"id": "coco-2014", "split": "minival"}),
    ]
    validate_model(FIXTURE, authored, SCHEMA)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([_reference_metrics(metrics={"coco_mapp": 0.5})], "'coco_mapp' is not an accuracy metric"),
        ([_reference_metrics(metrics={"throughput_fps": 0.5})], "'throughput_fps' is not an accuracy metric"),
        ([_reference_metrics(metrics={"coco_map": 53.0})], "coco_map must be a ratio from 0 to 1; found 53.0"),
        (
            [_reference_metrics(), _reference_metrics(source="https://example.com/other")],
            "coco_map on coco-2017 is already published",
        ),
        ([_reference_metrics(source="http://example.com/card")], "does not match"),
        ([{key: value for key, value in _reference_metrics().items() if key != "conditions"}], "'conditions'"),
    ],
)
def test_reference_metrics_are_validated(entries: list[dict], expected: str) -> None:
    authored = _authored()
    authored["reference_metrics"] = entries
    with pytest.raises(ValidationError, match=expected):
        validate_model(FIXTURE, authored, SCHEMA)


def _model_with_conversion(tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURE, directory)
    for relative in ("scripts/convert.py", "scripts/requirements-convert.txt"):
        (directory / relative).write_text("", encoding="utf-8")
    _write_model_tests(directory, {"conversion": RUN_TESTS, "compile": RUN_TESTS})
    (directory / "tests" / "build" / "test_conversion.py").write_text("", encoding="utf-8")
    authored = _authored(directory)
    authored["conversion"] = {"source": copy.deepcopy(authored["compilation"]["source"])}
    return directory, authored


def test_conversion_sources_are_validated(tmp_path: Path) -> None:
    directory, authored = _model_with_conversion(tmp_path)
    model = validate_model(directory, authored, SCHEMA)
    assert model["conversion"] == {"default": {"source": {"type": "file", **authored["conversion"]["source"]}}}
    authored["conversion"]["config"] = "convert.yaml"
    with pytest.raises(ValidationError, match="conversion"):
        validate_model(directory, authored, SCHEMA)


def test_conversion_sources_follow_source_policy(tmp_path: Path) -> None:
    directory, authored = _model_with_conversion(tmp_path)
    authored["conversion"]["source"]["uri"] = "https://unapproved.example/model.pth"
    with pytest.raises(ValidationError, match="source policy.*not approved"):
        validate_model(directory, authored, SCHEMA)


def test_conversion_file_set_opens_one_of_its_files(tmp_path: Path) -> None:
    directory, authored = _model_with_conversion(tmp_path)
    member = {key: value for key, value in authored["compilation"]["source"].items() if key != "media_type"}
    authored["conversion"]["source"] = {"main_file": "missing.safetensors", "files": {"model.safetensors": member}}
    with pytest.raises(ValidationError, match="source main_file 'missing.safetensors' is not one of its files"):
        validate_model(directory, authored, SCHEMA)


@pytest.mark.parametrize("missing", ["scripts/convert.py", "scripts/requirements-convert.txt"])
def test_declared_conversion_requires_its_workflow_files(tmp_path: Path, missing: str) -> None:
    directory, authored = _model_with_conversion(tmp_path)
    (directory / missing).unlink()
    with pytest.raises(ValidationError, match=f"referenced file does not exist inside model directory: {missing}"):
        validate_model(directory, authored, SCHEMA)


@pytest.mark.parametrize("entry", ["compile.py", "convert.yaml"])
def test_legacy_model_folder_entries_are_rejected(tmp_path: Path, entry: str) -> None:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURE, directory)
    (directory / entry).write_text("", encoding="utf-8")
    with pytest.raises(ValidationError, match=f"unexpected model folder entries: {entry}.*scripts/"):
        load_and_validate_model(directory, SCHEMA, TAXONOMY)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda tests: (tests / "tests.yaml").unlink(), "tests/tests.yaml is required"),
        (
            lambda tests: _write_model_tests(tests.parent, {"conversion": RUN_TESTS, "compile": RUN_TESTS}),
            "set up exactly the stages this model has: unknown conversion",
        ),
        (
            lambda tests: _write_model_tests(tests.parent, {"compile": {"run_tests": False}}),
            "compile: 'skip_reason' is a required property",
        ),
        (
            lambda tests: _write_model_tests(tests.parent, {"compile": {"run_tests": False, "skip_reason": ""}}),
            "compile.skip_reason",
        ),
        (
            lambda tests: _write_model_tests(tests.parent, {"compile": {"run_tests": True, "skip_reason": "stale"}}),
            "compile: .* should not be valid",
        ),
        (lambda tests: (tests / "build" / "test_compile.py").unlink(), "tests/build/test_compile.py is missing"),
        (
            lambda tests: (tests / "build" / "test_helpers.py").write_text("", encoding="utf-8"),
            "unexpected test files: tests/build/test_helpers.py",
        ),
        # The model-level tests/correctness.yaml these folders once held is not a test file.
        (
            lambda tests: (tests / "correctness.yaml").write_text("", encoding="utf-8"),
            "unexpected test files: tests/correctness.yaml",
        ),
    ],
    ids=[
        "missing tests file",
        "unknown entry",
        "skipped without reason",
        "skipped with empty reason",
        "running with a skip reason",
        "running without its file",
        "unexpected build file",
        "legacy correctness file",
    ],
)
def test_model_tests_follow_their_tests_file(tmp_path: Path, change, expected: str) -> None:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURE, directory)
    change(directory / "tests")
    with pytest.raises(ValidationError, match=expected):
        load_and_validate_model(directory, SCHEMA, TAXONOMY)


def test_model_tests_set_up_every_stage_the_model_has(tmp_path: Path) -> None:
    directory, authored = _measured_model(tmp_path, "accuracy", "performance")
    _write_model_tests(directory, {"compile": RUN_TESTS, "accuracy": EXPECTED["accuracy"]})
    with pytest.raises(ValidationError, match="set up exactly the stages this model has: missing performance"):
        validate_model(directory, authored, SCHEMA)

    _write_model_tests(
        directory,
        {
            "compile": RUN_TESTS,
            "accuracy": EXPECTED["accuracy"],
            "performance": {"run_tests": False, "skip_reason": "no published result"},
        },
    )
    validate_model(directory, authored, SCHEMA)


@pytest.mark.parametrize(
    ("stage", "entry", "expected"),
    [
        ("accuracy", {**EXPECTED["accuracy"], "expected": {"other": {"top1_accuracy": 0.5}}},
         "accuracy.expected names unmeasured build profiles: other"),
        ("accuracy", {**EXPECTED["accuracy"], "expected": {"fixture": {"top5_accuracy": 0.5}}},
         "expected.fixture lacks top1_accuracy"),
        ("accuracy", {**EXPECTED["accuracy"], "expected": {"fixture": {}}}, "accuracy.expected.fixture"),
        ("accuracy", {"run_tests": True}, "'expected' is a required"),
        ("accuracy", {**EXPECTED["accuracy"], "allowed_drop": "50%"}, "accuracy: Unevaluated properties"),
        ("accuracy", {**EXPECTED["accuracy"], "unknown": 1}, "accuracy: Unevaluated properties"),
        ("performance", {**EXPECTED["performance"], "expected": {"fixture": {"latency_avg_ms": 1.0}}},
         "performance.expected.fixture: 'throughput_fps' is a required property"),
        (
            "performance",
            {**EXPECTED["performance"], "expected": {"fixture": {"throughput_fps": 9.0, "top1_accuracy": 0.5}}},
            "performance.expected.fixture has unknown performance metrics: top1_accuracy",
        ),
    ],
    ids=[
        "accuracy of an unmeasured build profile",
        "accuracy without a required metric",
        "accuracy without values",
        "missing expected values",
        "a model's own allowed drop",
        "unknown setting",
        "performance without its throughput",
        "performance with an accuracy metric",
    ],
)
def test_measured_stages_state_what_the_model_must_deliver(
    tmp_path: Path, stage: str, entry: dict, expected: str
) -> None:
    directory, authored = _measured_model(tmp_path, "accuracy", "performance")
    validate_model(directory, authored, SCHEMA)

    _write_model_tests(directory, {"compile": RUN_TESTS, **EXPECTED, stage: entry})
    with pytest.raises(ValidationError, match=expected):
        validate_model(directory, authored, SCHEMA)


def test_yaml_uses_safe_loader(tmp_path: Path) -> None:
    directory = tmp_path / "unsafe"
    directory.mkdir()
    (directory / "model.yaml").write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="could not load YAML"):
        load_model(directory)


def test_catalog_model_is_valid() -> None:
    model = load_and_validate_model(ROOT / "models" / "resnet_50", SCHEMA)
    assert model["tasks"] == ["image_classification"]
    categorization = validate_model_categorization(model, load_taxonomy(TAXONOMY))
    assert categorization is not None
    assert categorization["architecture"] == {"id": "resnet", "paradigm": "cnn"}
    assert model["components"][0]["source"]["sha256"] == (
        "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"
    )
    profiles = {profile["id"]: profile for profile in model["build_profiles"]}
    assert set(profiles) == {"w16a16", "w8a8"}
    assert profiles["w16a16"]["components"][0]["parameters"]["precision"] == "w16a16"
    assert profiles["w8a8"]["components"][0]["parameters"]["precision"] == "w8a8"
    assert profiles["w8a8"]["components"][0]["parameters"]["calibration_samples"] == 500
    assert model["calibrations"][0]["dataset"] == {
        "id": "imagenet-2012",
        "variant": "mini",
        "sha256": "d769e759abba1fe6e03db704020dcff3f1e8309a8cac8885ab60f5a0e512c239",
        "size_bytes": 356052889,
    }


def test_unknown_task_fails_semantic_validation() -> None:
    authored = _authored()
    authored["tasks"] = ["unknown_task"]
    with pytest.raises(ValidationError, match="tasks references unknown taxonomy ID"):
        validate_model(FIXTURE, authored, SCHEMA, TAXONOMY)


def test_incompatible_model_type_and_architecture_fail() -> None:
    authored = _authored()
    authored["model_type"] = "llm"
    authored["tasks"] = ["text_generation"]
    authored["modalities"] = {"inputs": ["text"], "outputs": ["text"]}
    with pytest.raises(ValidationError, match="architecture 'resnet' is not compatible"):
        validate_model(FIXTURE, authored, SCHEMA, TAXONOMY)


def test_model_rejects_obsolete_taxonomy_version() -> None:
    authored = _authored()
    authored["taxonomy_version"] = 4
    with pytest.raises(ValidationError, match="taxonomy_version.*unexpected"):
        validate_model(FIXTURE, authored, SCHEMA, TAXONOMY)


def test_category_fields_are_required() -> None:
    authored = _authored()
    for field in ("model_type", "tasks", "modalities", "architecture"):
        authored.pop(field)
    with pytest.raises(ValidationError, match="'model_type' is a required property"):
        validate_model(FIXTURE, authored, SCHEMA, TAXONOMY)
