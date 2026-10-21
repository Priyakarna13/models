from __future__ import annotations

from typing import Any


def find_devkit_accuracy_config(model: dict[str, Any], build_profile_id: str) -> dict[str, Any] | None:
    """Return the model's accuracy settings when the profile is qualified."""
    entry = model.get("accuracy")
    return entry if entry is not None and build_profile_id in entry["build_profiles"] else None


def accuracy_components(model: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Every compiled component participates in a whole-model measurement."""
    return [component["id"] for component in model["components"]]
