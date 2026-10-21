from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from model_registry.accuracy_config import find_devkit_accuracy_config
from model_registry.errors import ValidationError
from model_registry.model import load_model, normalize_model, validate_model
from model_registry.source import source_identity

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _authored(name: str) -> dict:
    return yaml.safe_load((MODELS / name / "model.yaml").read_text())


@pytest.mark.parametrize(
    "name,entrypoint", [("resnet_50", "default"), ("qwen3_5_0_8b", "generate"), ("whisper_small", "transcribe")]
)
def test_default_package_preserves_runtime_entrypoint(name: str, entrypoint: str) -> None:
    model = load_model(MODELS / name)
    assert model["compositions"][0]["entrypoint"] == entrypoint
    assert model["packages"][0]["id"] == "default"
    assert model["packages"][0]["components"] == ["model"]


def test_normalization_does_not_modify_authored_settings() -> None:
    value = _authored("rf_detr_large")
    original = copy.deepcopy(value)
    normalized = normalize_model(value, MODELS / "rf_detr_large")
    assert value == original
    assert normalized["packages"][0]["components"] == ["backbone", "decoder"]
    assert normalized["components"][0]["source"] == normalized["components"][1]["source"]
    assert "interface" not in normalized["components"][0]


def test_file_set_preserves_layout_media_and_content_identity() -> None:
    value = _authored("qwen3_5_0_8b")
    model = normalize_model(value, MODELS / "qwen3_5_0_8b")
    source = model["compilation"]["source"]
    original = value["compilation"]["source"]
    assert source["main_file"] == original["main_file"]
    assert {member["path"]: member["sha256"] for member in source["files"]} == {
        path: member["sha256"] for path, member in original["files"].items()
    }
    changed = copy.deepcopy(source)
    for member in changed["files"]:
        member["uri"] = member["uri"].replace("simaai/", "mirror/")
    assert source_identity(source) == source_identity(changed)


@pytest.mark.parametrize("name,section", [("qwen3_5_0_8b", "compilation"), ("rf_detr_large", "conversion")])
def test_upstream_defaults_to_checkpoint(name: str, section: str) -> None:
    value = _authored(name)
    del value["upstream"]
    model = normalize_model(value, MODELS / name)
    assert model["upstream"]["repository"] == "https://huggingface.co/" + value[section]["source"]["repository"]
    assert model["upstream"]["revision"] == value[section]["source"]["revision"]


def test_explicit_upstream_is_preserved() -> None:
    value = _authored("resnet_50")
    model = normalize_model(value, MODELS / "resnet_50")
    assert model["upstream"]["derived_from"] == value["upstream"]["derived_from"]


def test_dataset_aliases_preserve_pins_and_accuracy_profile_restrictions() -> None:
    value = _authored("resnet_50")
    model = normalize_model(value, MODELS / "resnet_50")
    for stage in ("numerical_fidelity", "accuracy"):
        assert model[stage]["dataset"] == value["datasets"]["default"]
    assert find_devkit_accuracy_config(model, "w16a16") is None
    assert find_devkit_accuracy_config(model, "w8a8")["status"] == "provisional"
    assert "accuracy" not in load_model(MODELS / "ssd_mobilenet_v1")


def test_decoder_calibration_survives_backbone_w8a16() -> None:
    model = load_model(MODELS / "rf_detr_large")
    profile = next(p for p in model["build_profiles"] if p["id"] == "w8a16_w16a16")
    assert profile["calibration_dataset"] == "coco_images"
    decoder = profile["components"][1]["parameters"]
    assert decoder["calibration_samples"] == 50
    assert decoder["calibration_bins"] == 2048
    assert profile["components"][0]["parameters"]["per_token"] is True


def test_mutable_source_revision_is_rejected() -> None:
    value = _authored("qwen3_5_0_8b")
    value["compilation"]["source"]["revision"] = "main"
    with pytest.raises(ValidationError, match="immutable 40-character commit"):
        normalize_model(value, MODELS / "qwen3_5_0_8b")


def test_source_main_file_must_be_a_member() -> None:
    value = _authored("qwen3_5_0_8b")
    value["compilation"]["source"]["main_file"] = "missing.json"
    with pytest.raises(ValidationError, match="main_file is not a declared file"):
        validate_model(MODELS / "qwen3_5_0_8b", value)
