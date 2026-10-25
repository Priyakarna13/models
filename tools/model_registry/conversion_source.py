"""Emit immutable source pins for a converted ONNX and its original checkpoint members."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import ValidationError
from .fetch import sha256_file
from .model import load_model, source_members


def conversion_source(model_dir: Path, model_path: Path, uri: str, conversion: str = "default") -> dict:
    sha, size = sha256_file(model_path)
    pin = dict(uri=uri, sha256=sha, size_bytes=size)
    members_path = model_path.parent / "source-members.json"
    if not members_path.exists():
        source_members(dict(type="file", **pin))
        return pin
    # The exporter may reuse upstream weights, but cannot introduce unpinned members.
    expected = load_model(model_dir)["conversion"][conversion]["source"].get("files")
    members = json.loads(members_path.read_text())
    if not expected or members != expected or any(item["path"] == model_path.name for item in members):
        raise ValidationError("Converted source members must equal the pinned conversion checkpoint")
    files = {item["path"]: {key: item[key] for key in ("uri", "sha256", "size_bytes")} for item in members}
    files[model_path.name] = pin
    source = dict(main_file=model_path.name, files=files)
    source_members(
        dict(
            type="file_set",
            entrypoint=model_path.name,
            files=[dict(path=path, **value) for path, value in files.items()],
        )
    )
    return source
