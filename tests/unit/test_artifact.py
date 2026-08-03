from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.artifact import verify_mpk
from model_registry.errors import ValidationError


def _archive(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name in names:
            content = b"fixture"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


def test_verify_mpk_writes_checksums_and_result(tmp_path: Path) -> None:
    archive = tmp_path / "resnet_50_mpk.tar.gz"
    _archive(archive, ["resnet_50/model.elf", "resnet_50/resnet_50_mpk.json"])
    result = verify_mpk(tmp_path, write_manifest=True, provenance={"model_id": "resnet_50"})
    assert result["archive"] == archive.name
    assert len(result["sha256"]) == 64
    assert archive.name in (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    written = json.loads((tmp_path / "compile-result.json").read_text(encoding="utf-8"))
    assert written["archive"] == archive.name
    assert written["built_at"].endswith("+00:00")
    assert written["provenance"]["model_id"] == "resnet_50"
    assert written["status"] == "success"


@pytest.mark.parametrize(
    "names, message",
    [
        (["model/model_mpk.json"], "ELF payload"),
        (["model/model.elf"], "_mpk.json"),
        (["../model.elf", "model_mpk.json"], "unsafe archive member"),
    ],
)
def test_verify_mpk_rejects_incomplete_or_unsafe_archives(tmp_path: Path, names: list[str], message: str) -> None:
    _archive(tmp_path / "broken_mpk.tar.gz", names)
    with pytest.raises(ValidationError, match=message):
        verify_mpk(tmp_path)


def test_verify_mpk_requires_exactly_one_archive(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        verify_mpk(tmp_path)
