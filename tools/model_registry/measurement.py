"""A model's whole-model measurements: what scripts/performance.py and scripts/accuracy.py must report, and for
which build profiles, plus the reference scores its authors published."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_report import METRICS

from .errors import ValidationError

MEASUREMENT_STAGES = ("performance", "accuracy")
SCRIPTS = {"performance": "scripts/performance.py", "accuracy": "scripts/accuracy.py"}
ACCURACY_REQUIREMENTS = "scripts/requirements-accuracy.txt"


def measured_profiles(model: dict[str, Any], stage: str) -> list[str]:
    """The build profiles the stage measures, in model.yaml order."""
    declaration = model.get(stage)
    if declaration is None:
        return []
    return declaration.get("build_profiles", [profile["id"] for profile in model["build_profiles"]])


def validate_measurements(model: dict[str, Any], directory: Path) -> None:
    profiles = [profile["id"] for profile in model["build_profiles"]]
    for stage in MEASUREMENT_STAGES:
        declaration = model.get(stage)
        if declaration is None:
            continue
        if not (directory / SCRIPTS[stage]).is_file():
            raise ValidationError(f"{directory}: {stage} needs {SCRIPTS[stage]}")
        unknown_profiles = set(declaration.get("build_profiles", ())) - set(profiles)
        if unknown_profiles:
            names = ", ".join(sorted(unknown_profiles))
            raise ValidationError(f"{directory}: {stage} names unknown build profiles: {names}")
        unknown = [name for name in declaration["required_metrics"] if METRICS.get(name, {}).get("stage") != stage]
        if unknown:
            names = ", ".join(unknown)
            raise ValidationError(f"{directory}: {stage} requires metrics model_report does not define: {names}")
    _validate_reference_metrics(model.get("reference_metrics", []), directory)


def _validate_reference_metrics(entries: list[dict[str, Any]], directory: Path) -> None:
    """Each published score names an accuracy metric, as a ratio, at most once per dataset."""
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        label = f"{directory}: reference_metrics[{index}]"
        for name, value in entry["metrics"].items():
            if METRICS.get(name, {}).get("stage") != "accuracy":
                raise ValidationError(f"{label}: {name!r} is not an accuracy metric")
            if not 0 <= value <= 1:
                raise ValidationError(f"{label}: {name} must be a ratio from 0 to 1; found {value}")
            if (entry["dataset"]["id"], name) in seen:
                raise ValidationError(f"{label}: {name} on {entry['dataset']['id']} is already published")
            seen.add((entry["dataset"]["id"], name))


def reference_metrics_by_model(models: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each model's published scores, keyed by model ID, for the model browser."""
    return {
        model_id: model["reference_metrics"]
        for model_id, model in sorted(models.items())
        if model.get("reference_metrics")
    }
