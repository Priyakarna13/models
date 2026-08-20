from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import ValidationError

MODEL_COMPILER_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def resolve_model_compiler(repository: str | Path) -> dict[str, str]:
    root = Path(repository).expanduser().resolve()
    manifest_path = root / "deps" / "manifest.json"
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not read dependency manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"dependency manifest is not valid JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValidationError("dependency manifest must be a JSON object")
    dependency = manifest.get("model-compiler")
    if not isinstance(dependency, dict):
        raise ValidationError("dependency manifest must define model-compiler as an object")
    version = dependency.get("version")
    if not isinstance(version, str) or MODEL_COMPILER_VERSION.fullmatch(version) is None:
        raise ValidationError("model-compiler.version must use MAJOR.MINOR.PATCH format")
    return {"label": f"model-compiler-{version}", "version": version}


def model_producer_compiler(
    model: dict[str, Any],
    repository_compiler: dict[str, str],
) -> dict[str, str]:
    declared = model["source"].get("compiler_provenance")
    if isinstance(declared, dict):
        return {"label": str(declared["label"]), "version": str(declared["version"])}
    return repository_compiler


def validate_result_producer_compiler(
    model: dict[str, Any],
    provenance: dict[str, Any],
    repository_compiler: dict[str, str],
    *,
    context: str | Path,
) -> None:
    expected = model_producer_compiler(model, repository_compiler)
    fields = [("compiler_version", "version")]
    if isinstance(model["source"].get("compiler_provenance"), dict):
        fields.append(("compiler_label", "label"))
    for provenance_field, expected_field in fields:
        actual = provenance.get(provenance_field)
        if actual != expected[expected_field]:
            raise ValidationError(
                f"{context}: provenance {provenance_field} must be {expected[expected_field]!r} "
                f"for the declared model source, found {actual!r}"
            )
