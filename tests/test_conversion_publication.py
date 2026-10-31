from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from model_registry import conversion_publication as publication

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    output = tmp_path / "export"
    output.mkdir()
    original = tmp_path / "original.pth"
    original.write_bytes(b"exact-checkpoint-bytes")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    source = {
        "type": "file", "uri": f"https://download.pytorch.org/models/ssdlite_test-{digest[:8]}.pth",
        "sha256": digest, "size_bytes": original.stat().st_size,
    }
    model = {"id": "test_ssd", "license": {"redistribution": "allowed"},
             "conversion": {"default": {"source": source}}}
    onnx = output / "model.onnx"
    onnx.write_bytes(b"converted-model-bytes")
    onnx_sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
    key = f"models-onnx/test_ssd/{onnx_sha}/123-2/model.onnx"
    pin = {"uri": f"{publication.BASE_URI}/{key}", "sha256": onnx_sha, "size_bytes": onnx.stat().st_size}
    (output / "source.yaml").write_text(yaml.safe_dump({"source": pin}))
    state = SimpleNamespace(model=model, source=source, original=original, output=output, pin=pin,
                            objects={key: onnx.read_bytes()}, calls=[], reads=[], head_changes={}, version="version-1",
                            corrupt_leaf=None, fail_put=False)
    monkeypatch.setattr(publication, "load_and_validate_model", lambda _path: model)
    monkeypatch.setattr(publication, "Fetcher", lambda: SimpleNamespace(fetch=lambda _source: (original, False)))

    def aws(arguments):
        state.calls.append(arguments)
        key = arguments[arguments.index("--key") + 1]
        if arguments[0] == "put-object":
            assert key not in state.objects
            assert arguments[arguments.index("--if-none-match") + 1] == "*"
            assert arguments[arguments.index("--ssekms-key-id") + 1] == publication.KMS_KEY
            if state.fail_put:
                raise publication.ConversionPublicationError("create-only failure")
            data = Path(arguments[arguments.index("--body") + 1]).read_bytes()
            expected = base64.b64encode(hashlib.sha256(data).digest()).decode()
            assert arguments[arguments.index("--checksum-sha256") + 1] == expected
            state.objects[key] = data
            return {"VersionId": state.version}
        assert arguments[0] == "head-object"
        data = state.objects[key]
        return {"VersionId": state.version, "ChecksumSHA256": base64.b64encode(hashlib.sha256(data).digest()).decode(),
                "ContentLength": len(data), "ServerSideEncryption": "aws:kms", **state.head_changes}

    def opened(request, timeout):
        assert timeout == 60
        state.reads.append(request.full_url)
        key = request.full_url.removeprefix(publication.BASE_URI + "/")
        data = state.objects[key]
        return io.BytesIO(b"corrupt" if key.endswith(str(state.corrupt_leaf)) else data)

    monkeypatch.setattr(publication, "_aws", aws)
    monkeypatch.setattr(publication, "open_verified_url", opened)
    state.publish = lambda **kw: publication.publish_conversion_source(
        tmp_path / "model", output, **{"conversion": "default", "git_sha": "a" * 40,
                                     "run_id": "123", "run_attempt": "2", **kw},
    )
    return state


def test_mirror_preserves_origin_onnx_run_identity_and_versions(conversion):
    result = conversion.publish()
    assert result["origin"] == {key: conversion.source[key] for key in ("uri", "sha256", "size_bytes")}
    assert result["converted_onnx"] == conversion.pin
    assert result["license"] == conversion.model["license"]
    assert result["upstream"] == conversion.model.get("upstream", {})
    assert result["github"] == {"commit_sha": "a" * 40, "run_id": "123", "run_attempt": "2"}
    assert result["source_object"]["version_id"] == result["provenance_object"]["version_id"] == "version-1"
    assert result["mirror"]["uri"].endswith(f"/{conversion.source['sha256']}/123-2/source.pth")
    assert yaml.safe_load((conversion.output / "conversion-source.yaml").read_text()) == {
        "conversion": {"source": result["mirror"]},
    }
    assert len(conversion.reads) == 3  # full ONNX, PTH and provenance readbacks, not source-cache hits
    assert [call[0] for call in conversion.calls] == ["put-object", "head-object"] * 2
    for call in conversion.calls[::2]:
        assert json.loads(call[call.index("--metadata") + 1]) == {"git-sha": "a" * 40, "config": "default"}


@pytest.mark.parametrize("arguments", [
    {"git_sha": "short"}, {"run_id": "0"}, {"run_attempt": "0"}, {"conversion": "../bad"},
])
def test_mirror_rejects_unbound_workflow_identity_before_upload(conversion, arguments):
    with pytest.raises(publication.ConversionPublicationError):
        conversion.publish(**arguments)
    assert conversion.calls == []


@pytest.mark.parametrize("redistribution", ["conditional", "prohibited", None])
def test_mirror_requires_explicit_allowed_redistribution(conversion, redistribution):
    conversion.model["license"]["redistribution"] = redistribution
    with pytest.raises(publication.ConversionPublicationError, match="redistribution"):
        conversion.publish()
    assert conversion.calls == []


@pytest.mark.parametrize("source_type", ["huggingface", "files"])
def test_mirror_rejects_file_sets_without_changing_other_conversion_routes(conversion, source_type):
    conversion.source["type"] = source_type
    with pytest.raises(publication.ConversionPublicationError, match="single-file"):
        conversion.publish()
    assert conversion.calls == []


def test_mirror_rejects_non_official_conversion_source(conversion):
    conversion.source["uri"] = "hf://models/owner/repo@" + "a" * 40 + "/source.pth"
    with pytest.raises(publication.ConversionPublicationError, match="official"):
        conversion.publish()
    assert conversion.calls == []


def test_mirror_revalidates_checkpoint_bytes_after_fetch(conversion):
    conversion.original.write_bytes(b"changed-after-cache-verification")
    with pytest.raises(publication.ConversionPublicationError, match="changed after"):
        conversion.publish()
    assert conversion.calls == []


def test_mirror_rejects_unmatched_published_onnx_pin(conversion):
    conversion.pin["uri"] = conversion.pin["uri"].replace("/123-2/", "/123-1/")
    (conversion.output / "source.yaml").write_text(yaml.safe_dump({"source": conversion.pin}))
    with pytest.raises(publication.ConversionPublicationError, match="ONNX source pin"):
        conversion.publish()
    assert conversion.calls == []


@pytest.mark.parametrize("leaf", ["model.onnx", "source.pth", "conversion-source-provenance.json"])
def test_failed_full_readback_never_emits_a_suggested_source_pin(conversion, leaf):
    conversion.corrupt_leaf = leaf
    with pytest.raises(publication.ConversionPublicationError, match="HTTP readback"):
        conversion.publish()
    assert not (conversion.output / "conversion-source.yaml").exists()
    assert not (conversion.output / "conversion-source-publication.json").exists()


@pytest.mark.parametrize("version", [None, "", "null"])
def test_mirror_requires_an_immutable_s3_version(conversion, version):
    conversion.version = version
    with pytest.raises(publication.ConversionPublicationError, match="immutable S3 version"):
        conversion.publish()
    assert not (conversion.output / "conversion-source.yaml").exists()


@pytest.mark.parametrize("changes", [
    {"ChecksumSHA256": "bad"}, {"ContentLength": 0}, {"VersionId": "different"}, {"ServerSideEncryption": "AES256"},
])
def test_mirror_verifies_s3_checksum_size_version_and_encryption(conversion, changes):
    conversion.head_changes = changes
    with pytest.raises(publication.ConversionPublicationError, match="mismatch"):
        conversion.publish()
    assert not (conversion.output / "conversion-source.yaml").exists()


def test_create_only_upload_failure_never_emits_a_pin(conversion):
    conversion.fail_put = True
    with pytest.raises(publication.ConversionPublicationError, match="create-only"):
        conversion.publish()
    assert not (conversion.output / "conversion-source.yaml").exists()


def test_stale_publication_outputs_are_not_reused_as_new_source_pins(conversion):
    (conversion.output / "conversion-source.yaml").write_text("old pin\n")
    with pytest.raises(publication.ConversionPublicationError, match="fresh workflow attempt"):
        conversion.publish()
    assert conversion.calls == conversion.reads == []


def test_failed_http_request_never_emits_a_source_pin(conversion, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline test transfer failure")

    monkeypatch.setattr(publication, "open_verified_url", fail)
    with pytest.raises(publication.ConversionPublicationError, match="HTTP readback failed"):
        conversion.publish()
    assert conversion.calls == []


@pytest.mark.parametrize("response", [
    SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    SimpleNamespace(returncode=0, stdout="[]", stderr=""),
])
def test_aws_adapter_fails_closed_without_an_object_response(monkeypatch, response):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: response)
    with pytest.raises(publication.ConversionPublicationError):
        publication._aws(["head-object"])


def test_optional_workflow_leaves_default_onnx_publication_unchanged():
    workflow = yaml.load((ROOT / ".github/workflows/vulcan-model-convert.yml").read_text(), Loader=yaml.BaseLoader)
    selection = workflow["on"]["workflow_dispatch"]["inputs"]["mirror_conversion_source"]
    assert selection == {
        "description": "Also mirror the verified original TorchVision PTH when redistribution is allowed",
        "default": "false", "type": "boolean",
    }
    steps = workflow["jobs"]["convert"]["steps"]
    mirror = next(step for step in steps if step.get("name") == "Mirror the verified conversion PTH (opt-in)")
    assert mirror["if"] == "${{ inputs.mirror_conversion_source }}"
    assert "publish-conversion-source" in mirror["run"]
    names = [step.get("name") for step in steps]
    assert names.index("Test the converted model with the model's conversion tests") < names.index(
        "Publish ONNX and print source pin"
    ) < names.index(mirror["name"])
    onnx = next(step for step in steps if step.get("name") == "Publish ONNX and print source pin")
    assert "--if-none-match '*'" in onnx["run"]
    assert "source.pth" not in onnx["run"]
