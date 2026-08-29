from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml
from model_registry.catalog import model_build_fingerprint, select_model_ids
from model_registry.model import load_and_validate_model

from models.depthart_base.compile import (
    EXPECTED_ELF_SHA256,
    EXPECTED_GUARDED_EMITTERS,
    PUBLIC_INPUT,
    PUBLIC_OUTPUT,
    _disable_impossible_int16_exact_mode,
    audit_mpk,
    variant_config,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "depthart_base"


def test_depthart_registry_entry_pins_strict_int8_sources() -> None:
    model = load_and_validate_model(MODEL_DIR)
    variant = variant_config("modalix_strict_int8")
    parameters = variant["parameters"]

    assert model["architecture"] == {"id": "depthart", "variant": "base_quant_conv_mixer"}
    assert model["source"] == {
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/534820229",
        "sha256": "25fb78ac659c6936dc2cac9a739262e6c8a028d4e2f6f8a65350016ee8458d79",
        "size_bytes": 12192947,
        "media_type": "application/onnx",
    }
    assert model["training"] == {"method": "qat", "config": "training/depthart_base/qat.yaml"}
    assert parameters["precision"] == "int8"
    assert parameters["public_input_shape"] == [1, 448, 576, 3]
    assert parameters["public_output_shape"] == [1, 448, 576, 16]
    assert parameters["quantized_source"]["sha256"] == (
        "8b2518ef6492865d7f7cde2bb8be6879d00259c08d2565557573d290928575a3"
    )
    assert parameters["qat_checkpoint"]["sha256"] == (
        "7e7909271692ff3b2e24d4ce82f7cf17cd99393f3dbb4557c73120bfc1a4997c"
    )
    assert parameters["expected_elf_sha256"] == EXPECTED_ELF_SHA256


def test_depthart_qat_config_is_model_specific_and_strict_w8a8() -> None:
    config = yaml.safe_load((ROOT / "training" / "depthart_base" / "qat.yaml").read_text(encoding="utf-8"))

    assert config["method"] == "qat"
    assert config["runner_contract"]["package"] == "sima-neat/qat"
    assert config["source"]["checkpoint"]["sha256"] == (
        "87bdec998a91c262142e2d8df058a708a76585cf0b82e2916204fca1e392f653"
    )
    assert config["quantization"]["activations"]["dtype"] == "int8"
    assert config["quantization"]["weights"]["dtype"] == "int8"
    assert config["quantization"]["forbidden_compute_dtypes"] == ["bfloat16", "float16"]
    assert config["data"]["disjoint_validation_required"] is True
    assert config["resume"]["required"] is True
    assert config["acceptance"]["minimum_delta1_retention_vs_matched_fp32"] == 0.95


def test_depthart_accuracy_reference_is_honest_about_absolute_gates() -> None:
    result = yaml.safe_load((MODEL_DIR / "tests" / "correctness.yaml").read_text(encoding="utf-8"))

    assert result["dataset"]["samples"] == 654
    assert result["qualification_artifact"]["mla_elf_sha256"] == EXPECTED_ELF_SHA256
    assert result["acceptance"] == {
        "metric": "delta1_retention_vs_matched_fp32",
        "minimum": 0.95,
        "measured": 0.9505142586891481,
        "passed": True,
    }
    assert [item["passed"] for item in result["reported_non_gates"]] == [False, False]
    assert result["execution"]["all_outputs_finite"] is True
    assert result["execution"]["c16_max_absolute_channel_delta"] == 0
    assert result["execution"]["repeat_runs_bit_exact"] is True
    assert result["execution"]["registry_recipe_board_smoke"]["matches_qualification_prediction_set"] is True


def test_int16_guard_changes_only_impossible_int8_request() -> None:
    class Dtype:
        def __init__(self, is_int16: bool) -> None:
            self.is_int16_type = is_int16

    class Layout:
        def __init__(self, is_int16: bool) -> None:
            self.dtype = Dtype(is_int16)

    assert _disable_impossible_int16_exact_mode(True, [Layout(False)], Layout(False)) is True
    assert _disable_impossible_int16_exact_mode(True, [Layout(True)], Layout(False)) is False
    assert _disable_impossible_int16_exact_mode(True, [Layout(False)], Layout(True)) is False
    assert _disable_impossible_int16_exact_mode(False, [Layout(False)], Layout(False)) is False
    assert EXPECTED_GUARDED_EMITTERS == {
        "MLA_0/tuple_concat_47_conv",
        "MLA_0/tuple_concat_67_conv",
    }


def test_mpk_audit_requires_one_mla_elf_and_exact_int8_abi(tmp_path: Path) -> None:
    manifest = {
        "input_nodes": [dict(PUBLIC_INPUT)],
        "plugins": [
            {
                "name": "MLA_0",
                "sequence": 1,
                "processor": "MLA",
                "output_nodes": [{"name": "MLA_0", **PUBLIC_OUTPUT}],
            }
        ],
    }
    archive = tmp_path / "depthart_mpk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        payloads = {
            "depthart_mpk.json": json.dumps(manifest).encode(),
            "depthart_stage1_mla.elf": b"qualified-elf",
        }
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))

    fixture_elf_sha256 = hashlib.sha256(b"qualified-elf").hexdigest()
    report = audit_mpk(archive, expected_elf_sha256=fixture_elf_sha256)

    assert report["plugin_count"] == 1
    assert report["elfs"] == ["depthart_stage1_mla.elf"]
    assert report["elf_sha256"] == fixture_elf_sha256
    assert report["public_input"]["dtype"] == "int8"
    assert report["public_output"]["shape"] == [1, 448, 576, 16]
    with pytest.raises(RuntimeError, match="does not match the board-qualified program"):
        audit_mpk(archive, expected_elf_sha256="0" * 64)


def test_generic_training_path_selects_and_fingerprints_only_depthart() -> None:
    catalog = {
        "depthart_base": load_and_validate_model(MODEL_DIR),
        "resnet_50": load_and_validate_model(ROOT / "models" / "resnet_50"),
    }
    changed = "training/depthart_base/qat.yaml"

    assert select_model_ids(catalog, [changed]) == ["depthart_base"]
    before = model_build_fingerprint(ROOT, "depthart_base", "2.1.3")
    original = (ROOT / changed).read_text(encoding="utf-8")
    try:
        (ROOT / changed).write_text(original + "\n", encoding="utf-8")
        assert model_build_fingerprint(ROOT, "depthart_base", "2.1.3") != before
    finally:
        (ROOT / changed).write_text(original, encoding="utf-8")
