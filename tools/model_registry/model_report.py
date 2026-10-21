"""Validate the model report and identify its metric records."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .errors import ValidationError

REPORT_SCHEMA = "sima.model-report"
REPORT_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


def contract_path(name: str) -> Path:
    bundled = Path(__file__).parent / "contracts" / name
    if bundled.is_file():
        return bundled
    folder = "schemas" if name.endswith(".schema.json") else "taxonomies"
    return ROOT / folder / name


def load_metric_catalog(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or contract_path("metrics.json")).read_text(encoding="utf-8"))


def merge_context(*contexts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for context in contexts:
        for name, value in context.items():
            if name in merged and merged[name] != value:
                raise ValidationError(f"conflicting measurement context: {name}")
            merged[name] = value
    return merged


def metric_records(report: dict[str, Any], *, include_components: bool = False) -> Iterator[dict[str, Any]]:
    """Yield records with their complete context and optional component ID."""
    for metric in report["metrics"]:
        yield {**metric, "context": merge_context(report["context"], metric.get("context", {}))}
    if include_components:
        for component_id, component in report.get("components", {}).items():
            for metric in component.get("metrics", []):
                yield {
                    **metric,
                    "component_id": component_id,
                    "context": merge_context(
                        report["context"], component.get("context", {}), metric.get("context", {})
                    ),
                }


def metric_identity(metric: dict[str, Any]) -> str:
    return json.dumps(
        [metric["name"], metric.get("context", {}), metric.get("component_id"), metric.get("aggregation", "raw")],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_report(
    report: dict[str, Any],
    *,
    required_metrics: Iterable[str] = (),
    stage: str | None = None,
    schema_path: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Reject malformed, ambiguous or semantically inconsistent producer reports."""
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"model report is not finite JSON: {exc}") from exc
    schema = json.loads((schema_path or contract_path("model-report.schema.json")).read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(report))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "report"
        raise ValidationError(f"model report {location}: {error.message}")
    if stage and report["stage"] != stage:
        raise ValidationError(f"expected {stage} report, received {report['stage']}")
    catalog = load_metric_catalog(catalog_path)
    identities = set()
    for record in metric_records(report, include_components=True):
        identity = metric_identity(record)
        if identity in identities:
            raise ValidationError(f"duplicate measurement: {record['name']}")
        identities.add(identity)
        definition = catalog.get(record["name"])
        if definition:
            for field in ("unit", "direction"):
                if field in record and record[field] != definition[field]:
                    raise ValidationError(f"{record['name']} {field} must be {definition[field]!r}")
            if definition["stage"] != report["stage"]:
                raise ValidationError(f"{record['name']} is not a {report['stage']} metric")
    whole_model = list(metric_records(report))
    for name in required_metrics:
        matches = [record for record in whole_model if record["name"] == name and record["status"] == "measured"]
        if len(matches) != 1:
            raise ValidationError(
                f"required metric {name} needs exactly one whole-model measurement; found {len(matches)}"
            )
    dataset = report["context"].get("dataset", {})
    if dataset.get("evaluated_samples", 0) > dataset.get("available_samples", math.inf):
        raise ValidationError("evaluated_samples exceeds available_samples")
    return report


def load_report(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"could not read model report {path}: {exc}") from exc
    return validate_report(report, **kwargs)
