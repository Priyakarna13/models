from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ValidationError
from .fetch import sha256_file


def verify_mpk(
    output_dir: str | Path,
    *,
    write_manifest: bool = False,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    directory = Path(output_dir).expanduser().resolve()
    archives = sorted(directory.glob("*_mpk.tar.gz")) if directory.is_dir() else []
    if len(archives) != 1:
        raise ValidationError(f"{directory}: expected exactly one *_mpk.tar.gz archive, found {len(archives)}")

    archive = archives[0]
    try:
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ValidationError(f"{archive}: invalid MPK archive: {exc}") from exc

    names: list[str] = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValidationError(f"{archive}: unsafe archive member: {member.name}")
        names.append(member.name)
    if not any(name.endswith(".elf") for name in names):
        raise ValidationError(f"{archive}: MPK does not contain an ELF payload")
    if not any(name.endswith("_mpk.json") for name in names):
        raise ValidationError(f"{archive}: MPK does not contain _mpk.json metadata")

    digest, size = sha256_file(archive)
    result = {
        "archive": archive.name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {},
        "schema_version": 1,
        "sha256": digest,
        "size_bytes": size,
        "status": "success",
    }
    if write_manifest:
        (directory / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        (directory / "compile-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
