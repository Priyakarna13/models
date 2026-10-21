from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ValidationError


def validate_fidelity_result(
    config: dict[str, Any],
    report: dict[str, Any],
    path: str | Path,
) -> None:
    report_path = Path(path)
    expected = {
        "status": config["status"],
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": config["selection"],
        "samples": config["selection"]["samples"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValidationError(f"{report_path}: numerical fidelity {field} does not match its configuration")

    metrics = {metric["name"]: metric for metric in report["metrics"]}
    if set(metrics) != set(config["outputs"]):
        raise ValidationError(f"{report_path}: numerical fidelity metrics do not match configured outputs")


def find_fidelity_config(
    model_dir: str | Path,
    model: dict[str, Any],
    *,
    build_profile_id: str,
    component_id: str,
) -> dict[str, Any] | None:
    config = model.get("numerical_fidelity")
    if config is None or build_profile_id not in config["build_profiles"]:
        return None
    if component_id not in {component["id"] for component in model["components"]}:
        raise ValidationError(f"{model_dir}: unknown fidelity component {component_id!r}")
    return config
