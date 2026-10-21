"""Bind registered profile inputs and expected stage keys before executing model code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .accuracy_config import find_devkit_accuracy_config
from .errors import ValidationError
from .fetch import sha256_file
from .measurement import verified_inputs
from .model import load_and_validate_model, performance_config
from .registry import _public_registry_json
from .stage_keys import accuracy_key, performance_key


def prepare_measurement(
    *,
    repository: Path,
    compiled_dir: Path,
    model_id: str,
    build_profile_id: str,
    branch: str,
    git_sha: str,
    runtime: dict[str, Any],
    read_api_url: str,
    read_request=None,
) -> dict[str, Any]:
    """Use checked local outputs and registry metadata, never values returned by model code."""
    root = Path(repository).resolve()
    model = load_and_validate_model(root / "models" / model_id)
    inputs = verified_inputs(model, build_profile_id, Path(compiled_dir))
    read = read_request or _public_registry_json
    builds = {}
    registered = []
    for item in inputs:
        component = item["component_id"]
        directory = Path(compiled_dir) / model_id / build_profile_id / component
        receipt = json.loads((directory / "registry-publication.json").read_text())
        runs = receipt.get("runs", [])
        if len(runs) != 1 or any(
            runs[0].get(key) != expected
            for key, expected in (
                ("model_id", model_id),
                ("build_profile_id", build_profile_id),
                ("component_id", component),
            )
        ):
            raise ValidationError(f"publication receipt does not identify {model_id}/{build_profile_id}/{component}")
        run_id = runs[0]["run_id"]
        kind = item["format"]
        artifact = read(f"{read_api_url.rstrip('/')}/v1/runs/{run_id}/artifacts/{kind}").get("artifact", {})
        if artifact.get("sha256") != item["sha256"] or artifact.get("size_bytes") != item["size_bytes"]:
            raise ValidationError(f"registered artifact does not match measured input: {component}")
        artifact["artifact_id"] = artifact.get("id")
        fields = ("artifact_id", "sha256", "size_bytes", "s3_bucket", "s3_key", "s3_version_id")
        if any(artifact.get(field) is None for field in fields):
            raise ValidationError(f"registered artifact is not version-bound: {component}")
        registered.append(
            {
                "component_id": component,
                "compile_run_id": run_id,
                "artifact_key": kind,
                **{field: artifact[field] for field in fields},
            }
        )
        builds[component] = {"run_id": run_id, "artifact_sha256": item["sha256"]}
    stages = {}
    for stage, config, key in (
        ("performance", performance_config(model, build_profile_id), performance_key),
        ("accuracy", find_devkit_accuracy_config(model, build_profile_id), accuracy_key),
    ):
        if config:
            script = root / "models" / model_id / "scripts" / f"{stage}.py"
            stages[stage] = {
                "stage_key": key(root, model, build_profile_id, builds, runtime=runtime),
                "source": {"models_commit": git_sha, "script_sha256": sha256_file(script)[0]},
                "required_metrics": config["required_metrics"],
            }
    return {
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "branch": branch,
        "runtime": runtime,
        "expected_component_ids": [item["component_id"] for item in inputs],
        "inputs": registered,
        "stages": stages,
        "stage_keys": {stage: item["stage_key"] for stage, item in stages.items()},
    }


def bind_measurement_keys(
    manifest: dict[str, Any], *, api_url: str, read_api_url: str, aws_region: str, request=None, read_request=None
) -> None:
    """Update expected keys without changing compiled selections, including targeted dispatches."""
    from urllib.parse import quote

    from .registry import RegistryPublicationError, SigV4RegistryClient, canonical_sha256

    read = read_request or _public_registry_json
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    ref = f"refs/heads/{manifest['branch']}"
    catalog_url = f"{read_api_url.rstrip('/')}/v1/repositories/sima-neat/models/catalogs/current"
    for attempt in range(8):
        catalog = read(f"{catalog_url}?git_ref={quote(ref, safe='')}").get("catalog")
        if not catalog:
            raise ValidationError("a branch catalog must exist before dispatching measurements")
        body = {
            "git_ref": ref,
            "expected_snapshot": catalog["canonical_sha256"],
            "measurements": manifest["stage_keys"],
        }
        try:
            client.request(
                "PUT",
                f"/v1/repositories/sima-neat/models/models/{manifest['model_id']}/measurements/{manifest['build_profile_id']}",
                body,
                f"model-measurements-{canonical_sha256(body)}",
            )
        except RegistryPublicationError as exc:
            if "catalog_snapshot_conflict" not in str(exc) or attempt == 7:
                raise
            continue
        return
