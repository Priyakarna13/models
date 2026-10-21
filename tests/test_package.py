from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from model_registry import package as package_module
from model_registry.errors import ValidationError
from model_registry.package import assemble_available_packages, assemble_package

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"
MPK_NAME = "resnet_50_w16a16_mpk.tar.gz"


def _compiled_component(tmp_path: Path) -> Path:
    output = tmp_path / "compiled" / "resnet_50" / "w16a16" / "model"
    output.mkdir(parents=True)
    archive = output / MPK_NAME
    with tarfile.open(archive, "w:gz") as stream:
        payload = b"compiled model"
        info = tarfile.TarInfo("model.elf")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    result = {
        "archive": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size,
        "provenance": {
            "model_id": "resnet_50",
            "build_profile_id": "w16a16",
            "component_id": "model",
            "target": "modalix",
            "source_sha256": SOURCE_SHA,
            "git_sha": "a" * 40,
            "environment": "staging",
            "compiler_version": "2.1.3",
            "compiler_label": "model-compiler-2.1.3",
        },
    }
    (output / "compile-result.json").write_text(json.dumps(result), encoding="utf-8")
    return tmp_path / "compiled"


def _two_components(monkeypatch: pytest.MonkeyPatch, compiled: Path, *, duplicate_path: bool = False) -> None:
    model = copy.deepcopy(package_module.load_and_validate_model(ROOT / "models/resnet_50"))
    model["components"].append({**model["components"][0], "id": "aux"})
    model["packages"][0]["components"].append("aux")
    model["compositions"] = [{"id": "default", "type": "pipeline"}]
    source = compiled / "resnet_50/w16a16/model"
    target = source.with_name("aux")
    target.mkdir()
    name = MPK_NAME if duplicate_path else "resnet_50_w16a16_aux_mpk.tar.gz"
    (target / name).write_bytes((source / MPK_NAME).read_bytes())
    result = json.loads((source / "compile-result.json").read_text())
    result["archive"] = name
    result["provenance"]["component_id"] = "aux"
    (target / "compile-result.json").write_text(json.dumps(result))
    monkeypatch.setattr(package_module, "load_and_validate_model", lambda *_args: model)


def test_single_component_package_is_the_compiled_mpk(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    result = assemble_package(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages",
        model_id="resnet_50",
        build_profile_id="w16a16",
        package_id="default",
    )
    package_dir = tmp_path / "packages" / "resnet_50" / "w16a16" / "packages" / "default"
    compiled_mpk = compiled / "resnet_50" / "w16a16" / "model" / MPK_NAME
    assert result["archive"] == MPK_NAME
    assert sorted(path.name for path in package_dir.iterdir()) == ["package-result.json", MPK_NAME]
    assert (package_dir / MPK_NAME).read_bytes() == compiled_mpk.read_bytes()
    manifest = result["manifest"]
    assert manifest["components"][0]["id"] == "model"
    assert manifest["components"][0]["path"] == MPK_NAME
    assert manifest["composition"] == {
        "id": "default",
        "type": "single",
        "component": "model",
        "entrypoint": "default",
    }
    assert manifest["files"] == [
        {
            "path": MPK_NAME,
            "sha256": hashlib.sha256(compiled_mpk.read_bytes()).hexdigest(),
            "size_bytes": compiled_mpk.stat().st_size,
        }
    ]


def test_assemble_available_packages_skips_incomplete_builds(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    results = assemble_available_packages(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages",
    )
    assert [(item["model_id"], item["build_profile_id"]) for item in results] == [("resnet_50", "w16a16")]


def test_multi_component_package_is_a_flat_deterministic_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = _compiled_component(tmp_path)
    _two_components(monkeypatch, compiled)
    results = [
        assemble_package(
            repository=ROOT,
            compiled_dir=compiled,
            output_dir=tmp_path / output,
            model_id="resnet_50",
            build_profile_id="w16a16",
            package_id="default",
        )
        for output in ("packages-a", "packages-b")
    ]
    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["archive"] == "resnet_50_w16a16_default.zip"
    bundle = tmp_path / "packages-a/resnet_50/w16a16/packages/default" / results[0]["archive"]
    with zipfile.ZipFile(bundle) as stream:
        assert stream.namelist() == ["resnet_50_w16a16_aux_mpk.tar.gz", MPK_NAME]
    assert results[0]["manifest"]["composition"] == {"id": "default", "type": "pipeline"}
    assert results[0]["manifest"]["resources"] == []


def test_package_rejects_colliding_component_archive_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = _compiled_component(tmp_path)
    _two_components(monkeypatch, compiled, duplicate_path=True)
    with pytest.raises(ValidationError, match="distinct package paths"):
        assemble_package(
            repository=ROOT,
            compiled_dir=compiled,
            output_dir=tmp_path / "packages",
            model_id="resnet_50",
            build_profile_id="w16a16",
            package_id="default",
        )
