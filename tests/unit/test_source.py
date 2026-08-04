from __future__ import annotations

import pytest
from model_registry.errors import SourcePolicyError
from model_registry.source import SourceSpec, redact_url

DIGEST = "a" * 64
REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_hugging_face_uri_resolves_full_commit() -> None:
    source = SourceSpec(
        uri=f"hf://models/owner/repository@{REVISION}/nested/model.onnx",
        sha256=DIGEST,
        size_bytes=10,
    )
    assert source.resolve_url() == (f"https://huggingface.co/owner/repository/resolve/{REVISION}/nested/model.onnx")
    assert source.filename == "model.onnx"


def test_github_release_asset_is_allowed() -> None:
    uri = "https://github.com/sima-neat/models/releases/download/superpoint-source-v1/model.onnx"
    source = SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10)
    assert source.resolve_url() == uri
    assert source.filename == "model.onnx"


def test_github_release_asset_api_is_allowed() -> None:
    uri = "https://api.github.com/repos/sima-neat/models/releases/assets/123456"
    source = SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10)
    assert source.resolve_url() == uri
    assert source.filename == "123456"


@pytest.mark.parametrize("repo_type", ["datasets", "spaces"])
def test_hugging_face_non_model_repo_types(repo_type: str) -> None:
    source = SourceSpec(
        uri=f"hf://{repo_type}/owner/repository@{REVISION}/data/file.bin",
        sha256=DIGEST,
        size_bytes=10,
    )
    assert source.resolve_url().startswith(f"https://huggingface.co/{repo_type}/owner/repository/resolve/{REVISION}/")


@pytest.mark.parametrize(
    "uri",
    [
        "hf://models/owner/repository@main/model.onnx",
        "http://huggingface.co/owner/repository/model.onnx",
        "https://user:secret@huggingface.co/owner/repository/model.onnx",
        "https://huggingface.co/owner/repository/resolve/main/model.onnx",
        f"https://huggingface.co/owner/repository/resolve/{REVISION}/model.onnx?token=secret",
        f"https://huggingface.co:444/owner/repository/resolve/{REVISION}/model.onnx",
        "https://unapproved.example/model.onnx",
        "https://github.com/sima-neat/models/raw/main/model.onnx",
        "https://api.github.com/repos/sima-neat/models/contents/model.onnx",
        f"hf://models/owner/repository@{REVISION}/../model.onnx",
    ],
)
def test_mutable_or_unapproved_sources_fail(uri: str) -> None:
    source = SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10)
    with pytest.raises(SourcePolicyError):
        source.resolve_url()


def test_redaction_removes_credentials_query_and_fragment() -> None:
    value = redact_url("https://user:secret@example.test/model.onnx?token=top-secret#fragment")
    assert value == "https://example.test/model.onnx"
    assert "secret" not in value
