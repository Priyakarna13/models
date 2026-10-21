"""Validate that declared performance stages have a standalone model script."""

from pathlib import Path
from typing import Any

from .errors import ValidationError


def validate_benchmarks(model: dict[str, Any], model_dir: str | Path) -> None:
    if model.get("performance"):
        script = Path(model_dir) / "scripts/performance.py"
        if not script.is_file():
            raise ValidationError(f"performance script is missing: {script}")
