from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def test_repository_model_compiler_dependency_is_pinned() -> None:
    assert resolve_model_compiler(ROOT) == {
        "label": "model-compiler-2.1.3",
        "source_commit": "0beeab8d4f4df457b9dabf3af32073399d116453",
        "version": "2.1.3",
    }


@pytest.mark.parametrize("version", ["2.1", "develop", "2.1.3-develop", 213, None])
def test_model_compiler_dependency_requires_semantic_version(tmp_path: Path, version: object) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps({"model-compiler": {"version": version, "source_commit": "a" * 40}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="MAJOR.MINOR.PATCH"):
        resolve_model_compiler(tmp_path)


def test_missing_model_compiler_dependency_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="must define model-compiler"):
        resolve_model_compiler(tmp_path)


@pytest.mark.parametrize("source_commit", ["develop", "a" * 39, "A" * 40, None])
def test_model_compiler_dependency_requires_source_commit(tmp_path: Path, source_commit: object) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps({"model-compiler": {"version": "2.1.3", "source_commit": source_commit}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="source_commit"):
        resolve_model_compiler(tmp_path)


def test_model_compiler_dependency_rejects_unapproved_runner_label(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "model-compiler": {
                    "version": "2.1.3",
                    "artifact_label": "default",
                    "source_commit": "a" * 40,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="artifact_label"):
        resolve_model_compiler(tmp_path)
