"""Read normalized whole-model performance measurements."""

from __future__ import annotations

from typing import Any

from .errors import ValidationError
from .model_report import load_metric_catalog, metric_records


def measurement_name(metric: str) -> str:
    """Performance declarations already use their registered metric identity."""
    return metric


def registry_metrics(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return unambiguous whole-model measurements; never use component scores."""
    catalog = load_metric_catalog()
    metrics = {}
    for record in metric_records(result):
        if record["status"] != "measured":
            continue
        name = record["name"]
        if name in metrics:
            raise ValidationError(f"{name} has multiple contexts; select a context before comparing it")
        metrics[name] = {
            **record,
            "direction": record.get("direction", catalog.get(name, {}).get("direction", "neutral")),
        }
    return metrics
