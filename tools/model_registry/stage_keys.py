"""Stage keys: what a compile, a DevKit performance run, or a DevKit accuracy run depends on.

A key hashes everything a stage's result depends on. Each published run carries its key as the run
fingerprint, so a registry run with the same key means that stage need not run again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .accuracy_config import find_devkit_accuracy_config
from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError
from .model import (
    ACCURACY_REQUIREMENTS,
    ACCURACY_SCRIPT,
    CONVERSION_FILES,
    PERFORMANCE_SCRIPT,
    SCRIPTS_DIR,
    compile_invocations_for,
    load_and_validate_model,
    performance_config,
)
from .source import source_identity

# Shared code used by ONNX builds.
COMPILE_TOOLING = ("scripts/run_model_compilation.sh", "tools/model_registry/model.py")

_IGNORED_BUILD_PARTS = {".git", "__pycache__"}
_IGNORED_BUILD_SUFFIXES = {".pyc", ".pyo"}
# Conversion produces the pinned source, and the board stages consume the MPK; none changes compilation.
NON_COMPILE_SCRIPTS = {PERFORMANCE_SCRIPT, ACCURACY_SCRIPT, *CONVERSION_FILES}

PERFORMANCE_RUN_TYPE = "model_benchmark"
ACCURACY_RUN_TYPE = "model_accuracy"
# Fingerprint versions are registry lookup identities, independent of report schema majors.
PERFORMANCE_FINGERPRINT_VERSION = 2
ACCURACY_FINGERPRINT_VERSION = 1
BOARD_STAGE_TOOLING = (
    "tools/model_registry/measurement.py",
    "tools/model_registry/model_report.py",
    "tools/model_registry/measurement_environment.py",
    "tools/model_registry/artifact.py",
    "tools/model_registry/fetch.py",
    "tools/model_registry/source.py",
    "tools/model_registry/model.py",
    "tools/model_registry/dependencies.py",
    "tools/model_registry/accuracy_config.py",
)
PERFORMANCE_TOOLING = (*BOARD_STAGE_TOOLING, "scripts/run_model_performance.sh")
ACCURACY_TOOLING = (*BOARD_STAGE_TOOLING, "scripts/run_model_accuracy.sh")


def _hash_files(digest: Any, root: Path, paths: set[Path] | list[Path]) -> None:
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def _key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_digests(root: Path, required: list[str], optional: list[str] = ()) -> dict[str, str | None]:
    digests: dict[str, str | None] = {}
    for relative in [*required, *optional]:
        path = root / relative
        if not path.is_file():
            if relative in required:
                raise ValidationError(f"stage input not found: {relative}")
            digests[relative] = None
            continue
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _compile_input_files(root: Path, model_id: str) -> list[Path]:
    """Code the compile runs: the shared runner, model scripts, and any declared training code."""
    candidates: list[Path] = []
    for prefix in COMPILE_TOOLING:
        path = root / prefix.rstrip("/")
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
        else:
            raise ValidationError(f"model compile input not found: {path}")
    model_dir = root / "models" / model_id
    candidates.extend(
        item
        for item in (model_dir / SCRIPTS_DIR).rglob("*")
        if item.is_file()
        and item.suffix in {".py", ".sh"}
        and item.relative_to(model_dir).as_posix() not in NON_COMPILE_SCRIPTS
    )
    symlinks = sorted(item for item in candidates if item.is_symlink())
    if symlinks:
        relative = ", ".join(item.relative_to(root).as_posix() for item in symlinks)
        raise ValidationError(f"build inputs must not be symbolic links: {relative}")
    return sorted(
        {
            item
            for item in candidates
            if not (_IGNORED_BUILD_PARTS & set(item.relative_to(root).parts))
            and item.suffix not in _IGNORED_BUILD_SUFFIXES
        },
        key=lambda item: item.relative_to(root).as_posix(),
    )


def _compile_consumed_fidelity(model: dict[str, Any], profile_id: str) -> dict[str, Any] | None:
    config = model.get("numerical_fidelity")
    if config is None or profile_id not in config["build_profiles"]:
        return None
    return {key: value for key, value in config.items() if key != "build_profiles"}


def compile_key(
    repository: str | Path,
    model_id: str,
    profile_id: str,
    compile_invocation_id: str,
    compiler_version: str,
) -> str:
    """Identity of one compile invocation: what it compiles, how, and with which compiler."""
    root = Path(repository).expanduser().resolve()
    compiler = compiler_version.strip()
    if not compiler:
        raise ValidationError("compiler version must not be empty")
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    try:
        profile = next(item for item in model["build_profiles"] if item["id"] == profile_id)
        invocation = next(item for item in compile_invocations_for(profile) if item["id"] == compile_invocation_id)
    except StopIteration as exc:
        raise ValidationError(f"unknown compile invocation: {model_id}/{profile_id}/{compile_invocation_id}") from exc
    components = {item["id"]: item for item in model["components"]}
    bindings = {item["id"]: item for item in profile["components"]}
    if any(components[item]["format"] == "huggingface" for item in invocation["components"]):
        # LLiMa recipes orchestrate a pinned compiler; graph options belong in parameters.
        return _key(
            {
                "compiler_version": compiler,
                "target": profile["target"],
                "components": [
                    {
                        "source_sha256": source_identity(components[component_id]["source"]),
                        "parameters": bindings[component_id]["parameters"],
                    }
                    for component_id in invocation["components"]
                ],
            }
        )
    identity = {
        "model_id": model_id,
        "compile_invocation": {
            "id": compile_invocation_id,
            "components": [
                {
                    "source_sha256": source_identity(components[component_id]["source"]),
                    "parameters": bindings[component_id]["parameters"],
                }
                for component_id in invocation["components"]
            ],
        },
        "build_profile": {
            "id": profile_id,
            "target": profile["target"],
            "calibration_dataset": model["datasets"].get(profile.get("calibration_dataset")),
        },
        "numerical_fidelity": _compile_consumed_fidelity(model, profile_id),
        "compiler_version": compiler,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    _hash_files(digest, root, _compile_input_files(root, model_id))
    return digest.hexdigest()


def _measurement_key(
    repository: str | Path,
    model: dict[str, Any],
    build_profile_id: str,
    builds: dict[str, dict[str, Any]],
    stage: str,
    runtime: dict[str, Any] | None,
) -> str:
    root = Path(repository).expanduser().resolve()
    model_id = str(model["id"])
    config = (
        performance_config(model, build_profile_id)
        if stage == "performance"
        else find_devkit_accuracy_config(model, build_profile_id)
    )
    if config is None:
        raise ValidationError(f"no {stage} stage applies to {model_id}/{build_profile_id}")
    components = [component["id"] for component in model["components"]]
    if set(components) != set(builds):
        raise ValidationError(f"{stage} requires exactly the compiled components: {', '.join(components)}")
    catalog = json.loads((root / "taxonomies/metrics.json").read_text(encoding="utf-8"))
    required_metrics = config["required_metrics"]
    unknown = set(required_metrics) - set(catalog)
    if unknown:
        raise ValidationError(f"unknown required metrics: {', '.join(sorted(unknown))}")
    semantics = {
        name: {field: catalog[name][field] for field in ("unit", "type", "direction", "stage")}
        for name in required_metrics
    }
    script = PERFORMANCE_SCRIPT if stage == "performance" else ACCURACY_SCRIPT
    tooling = PERFORMANCE_TOOLING if stage == "performance" else ACCURACY_TOOLING
    required = [f"models/{model_id}/{script}", *tooling]
    optional = [f"models/{model_id}/{ACCURACY_REQUIREMENTS}"] if stage == "accuracy" else []
    return _key(
        {
            "report_contract": {"schema": "sima.model-report", "schema_version": 1},
            "stage": stage,
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "configuration": {key: value for key, value in config.items() if key != "build_profiles"},
            "metric_semantics": semantics,
            "builds": {
                component: {
                    "compile_run_id": builds[component]["run_id"],
                    "artifact_sha256": builds[component]["artifact_sha256"],
                }
                for component in components
            },
            "files": _file_digests(root, required, optional),
            "runtime": runtime if runtime is not None else resolve_benchmark_runtime(root),
        }
    )


def performance_key(
    repository: str | Path,
    model: dict[str, Any],
    build_profile_id: str,
    builds: dict[str, dict[str, Any]],
    *,
    runtime: dict[str, Any] | None = None,
) -> str:
    """Identity of one whole-model performance measurement over exact compiled inputs."""
    return _measurement_key(repository, model, build_profile_id, builds, "performance", runtime)


def accuracy_key(
    repository: str | Path,
    model: dict[str, Any],
    build_profile_id: str,
    builds: dict[str, dict[str, Any]],
    *,
    runtime: dict[str, Any] | None = None,
) -> str:
    """Identity of one whole-model accuracy measurement over exact compiled inputs."""
    return _measurement_key(repository, model, build_profile_id, builds, "accuracy", runtime)
