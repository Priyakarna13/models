"""Opt-in publication of a verified original PTH beside a converted ONNX.

This CI-only job uses the conversion workflow's existing publisher identity,
bucket, KMS key and immutable-key convention. Other source/publication routes
are unchanged. No model declaration is edited by this module.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request

import yaml

from .errors import ModelRegistryError
from .fetch import Fetcher, open_verified_url, sha256_file
from .model import load_and_validate_model
from .source import SourceSpec

BUCKET = "sima-neat-artifacts-staging"
KMS_KEY = "alias/sima-neat-artifacts-staging"
BASE_URI = "https://artifacts.stg.neat.sima.ai"


class ConversionPublicationError(ModelRegistryError):
    """The source cannot be mirrored or its immutable publication was not verified."""


def _aws(arguments: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["aws", "s3api", *arguments, "--output", "json", "--no-cli-pager"],
            text=True, capture_output=True, check=False,
        )
    except OSError as error:
        raise ConversionPublicationError("Could not execute the conversion source S3 upload client") from error
    if result.returncode != 0:
        raise ConversionPublicationError("Conversion source immutable S3 operation failed")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ConversionPublicationError("Conversion source S3 response was not JSON") from error
    if not isinstance(document, dict):
        raise ConversionPublicationError("Conversion source S3 response was not an object")
    return document


def _put(path: Path, key: str, git_sha: str, conversion: str, media_type: str) -> dict[str, Any]:
    digest, size = sha256_file(path)
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    uploaded = _aws([
        "put-object", "--bucket", BUCKET, "--key", key, "--body", str(path),
        "--if-none-match", "*", "--server-side-encryption", "aws:kms", "--ssekms-key-id", KMS_KEY,
        "--checksum-algorithm", "SHA256", "--checksum-sha256", checksum,
        "--content-type", media_type, "--metadata", json.dumps({"git-sha": git_sha, "config": conversion}),
    ])
    version = uploaded.get("VersionId")
    if not isinstance(version, str) or not version or version == "null":
        raise ConversionPublicationError("Conversion source upload has no immutable S3 version")
    head = _aws([
        "head-object", "--bucket", BUCKET, "--key", key, "--version-id", version,
        "--checksum-mode", "ENABLED",
    ])
    if (head.get("VersionId") != version or head.get("ChecksumSHA256") != checksum
            or head.get("ContentLength") != size or head.get("ServerSideEncryption") != "aws:kms"):
        raise ConversionPublicationError("Conversion source S3 version/checksum/size/encryption mismatch")
    return {"bucket": BUCKET, "key": key, "version_id": version, "sha256": digest, "size_bytes": size}


def _readback(uri: str, expected_sha256: str, expected_size: int) -> None:
    digest = hashlib.sha256()
    size = 0
    # A fresh streamed HTTP transfer, never a source-cache hit. Do not publish a
    # metadata pin merely because an upload or a HEAD request returned success.
    try:
        with open_verified_url(Request(uri, headers={"User-Agent": "sima-neat-model-registry/0.1"}), 60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > expected_size:
                    raise ConversionPublicationError("Conversion source HTTP readback exceeds declared size")
                digest.update(chunk)
    except (OSError, HTTPException) as error:
        raise ConversionPublicationError("Conversion source HTTP readback failed") from error
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise ConversionPublicationError("Conversion source HTTP readback checksum/size mismatch")


def publish_conversion_source(
    model_dir: Path, output_dir: Path, *, conversion: str, git_sha: str, run_id: str, run_attempt: str,
) -> dict[str, Any]:
    if not re.fullmatch("[a-f0-9]{40}", git_sha):
        raise ConversionPublicationError("Conversion publication requires the complete Git commit")
    if any(not re.fullmatch("[1-9][0-9]*", value) for value in (run_id, run_attempt)):
        raise ConversionPublicationError("Conversion publication requires positive GitHub run and attempt identities")
    if not re.fullmatch("[a-z][a-z0-9_]*", conversion):
        raise ConversionPublicationError("Invalid conversion identifier")
    for filename in (
        "conversion-source.yaml", "conversion-source-provenance.json", "conversion-source-publication.json",
    ):
        path = output_dir / filename
        if path.exists() or path.is_symlink():
            raise ConversionPublicationError("Prior mirror publication outputs exist; use a fresh workflow attempt")
    model = load_and_validate_model(model_dir)
    if model["license"].get("redistribution") != "allowed":
        raise ConversionPublicationError("Mirroring requires explicitly allowed source redistribution")
    entry = model.get("conversion", {}).get(conversion)
    if entry is None or entry["source"].get("type") != "file":
        raise ConversionPublicationError("Mirroring supports only a declared single-file conversion source")
    original = SourceSpec.from_mapping(entry["source"])
    # This extension mirrors the original hashed TorchVision checkpoint only;
    # it does not start copying HF file sets or unrelated conversion inputs.
    if urlsplit(original.resolve_url()).hostname != "download.pytorch.org" or not original.filename.endswith(".pth"):
        raise ConversionPublicationError("Mirroring supports only an original official hashed TorchVision PTH")
    source, _cache_hit = Fetcher().fetch(original)
    if sha256_file(source) != (original.sha256, original.size_bytes):
        raise ConversionPublicationError("Conversion source changed after verified fetch")
    onnx_sha, onnx_size = sha256_file(output_dir / "model.onnx")
    pin = yaml.safe_load((output_dir / "source.yaml").read_text(encoding="utf-8"))["source"]
    expected_onnx_uri = f"{BASE_URI}/models-onnx/{model['id']}/{onnx_sha}/{run_id}-{run_attempt}/model.onnx"
    if (pin.get("uri"), pin.get("sha256"), pin.get("size_bytes")) != (expected_onnx_uri, onnx_sha, onnx_size):
        raise ConversionPublicationError("Published ONNX source pin does not match this conversion and workflow run")
    SourceSpec.from_mapping(pin)
    _readback(expected_onnx_uri, onnx_sha, onnx_size)
    key = f"models-onnx/{model['id']}/{original.sha256}/{run_id}-{run_attempt}/source.pth"
    mirror = {"uri": f"{BASE_URI}/{key}", "sha256": original.sha256, "size_bytes": original.size_bytes}
    SourceSpec.from_mapping(mirror)
    stored = _put(source, key, git_sha, conversion, original.media_type)
    _readback(mirror["uri"], original.sha256, original.size_bytes)
    provenance = {
        "schema_version": 1, "model_id": model["id"], "conversion": conversion,
        "github": {"commit_sha": git_sha, "run_id": run_id, "run_attempt": run_attempt},
        "origin": {"uri": original.uri, "sha256": original.sha256, "size_bytes": original.size_bytes},
        "mirror": mirror, "source_object": stored, "converted_onnx": pin,
        "license": model["license"], "upstream": model.get("upstream", {}),
    }
    evidence = output_dir / "conversion-source-provenance.json"
    evidence.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_key = key.removesuffix("source.pth") + evidence.name
    proof = _put(evidence, evidence_key, git_sha, conversion, "application/json")
    _readback(f"{BASE_URI}/{evidence_key}", proof["sha256"], proof["size_bytes"])
    # Only successful, fully read-back publications produce a suggested pin.
    (output_dir / "conversion-source.yaml").write_text(
        yaml.safe_dump({"conversion": {"source": mirror}}, sort_keys=False), encoding="utf-8",
    )
    result = {**provenance, "provenance_object": proof}
    (output_dir / "conversion-source-publication.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result
