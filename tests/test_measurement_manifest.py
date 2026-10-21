from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry import measurement_manifest as manifests
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    records = [
        {"component_id": name, "format": "mpk", "sha256": char * 64, "size_bytes": 12}
        for name, char in [("backbone", "a"), ("decoder", "b")]
    ]
    for record in records:
        directory = tmp_path / "rf_detr_small" / "w8a8_w16a16" / record["component_id"]
        directory.mkdir(parents=True)
        (directory / "registry-publication.json").write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "model_id": "rf_detr_small",
                            "build_profile_id": "w8a8_w16a16",
                            "component_id": record["component_id"],
                            "run_id": "compile-" + record["component_id"],
                        }
                    ]
                }
            )
        )
    monkeypatch.setattr(manifests, "verified_inputs", lambda *args: records)
    monkeypatch.setattr(manifests, "performance_key", lambda *args, **kwargs: "c" * 64)
    monkeypatch.setattr(manifests, "accuracy_key", lambda *args, **kwargs: "d" * 64)
    return records


def prepare(tmp_path, inputs, *, artifact_change=None):
    urls = []

    def read(url):
        urls.append(url)
        component = url.split("/")[-3].removeprefix("compile-")
        item = next(item for item in inputs if item["component_id"] == component)
        artifact = {
            "id": component + "-artifact",
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
            "s3_bucket": "models",
            "s3_key": component + ".mpk",
            "s3_version_id": "version",
        }
        if artifact_change:
            artifact_change(artifact)
        return {"artifact": artifact}

    manifest = manifests.prepare_measurement(
        repository=ROOT,
        compiled_dir=tmp_path,
        model_id="rf_detr_small",
        build_profile_id="w8a8_w16a16",
        branch="feature/report",
        git_sha="e" * 40,
        runtime={"core": {"install_target": "core@ref:" + "f" * 40}},
        read_api_url="https://reader.invalid",
        read_request=read,
    )
    return manifest, urls


def test_prepare_binds_complete_profile_to_every_stage(tmp_path, inputs):
    manifest, urls = prepare(tmp_path, inputs)
    assert manifest["expected_component_ids"] == ["backbone", "decoder"]
    assert set(manifest["stages"]) == {"performance", "accuracy"}
    assert manifest["stage_keys"] == {"performance": "c" * 64, "accuracy": "d" * 64}
    assert [item["compile_run_id"] for item in manifest["inputs"]] == ["compile-backbone", "compile-decoder"]
    assert len(urls) == 2
    assert all("/artifacts/mpk" in url for url in urls)
    assert manifest["stages"]["performance"]["source"]["models_commit"] == "e" * 40
    assert len(manifest["stages"]["accuracy"]["source"]["script_sha256"]) == 64


@pytest.mark.parametrize(
    "field,value", [("sha256", "0" * 64), ("size_bytes", 13), ("s3_version_id", None), ("id", None)]
)
def test_prepare_rejects_substituted_or_unversioned_artifact(tmp_path, inputs, field, value):
    with pytest.raises(ValidationError):
        prepare(tmp_path, inputs, artifact_change=lambda item: item.update({field: value}))


def test_prepare_rejects_another_profiles_receipt(tmp_path, inputs):
    path = tmp_path / "rf_detr_small/w8a8_w16a16/backbone/registry-publication.json"
    receipt = json.loads(path.read_text())
    receipt["runs"][0]["build_profile_id"] = "other"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValidationError, match="publication receipt"):
        prepare(tmp_path, inputs)


def test_native_deployment_manifest_is_an_input_not_a_generated_layer(tmp_path, inputs):
    for item in inputs:
        item["format"] = "llima"
    manifest, urls = prepare(tmp_path, inputs)
    assert all(url.endswith("/artifacts/llima") for url in urls)
    assert all(item["artifact_key"] == "llima" for item in manifest["inputs"])


def test_bind_changes_only_requested_profile_measurements(tmp_path, inputs):
    manifest, _ = prepare(tmp_path, inputs)
    calls = []
    manifests.bind_measurement_keys(
        manifest,
        api_url="https://writer.invalid",
        read_api_url="https://reader.invalid",
        aws_region="us-west-2",
        read_request=lambda url: {"catalog": {"canonical_sha256": "9" * 64}},
        request=lambda *args: calls.append(args) or {},
    )
    assert len(calls) == 1
    method, path, body, key = calls[0]
    assert method == "PUT"
    assert path.endswith("/models/rf_detr_small/measurements/w8a8_w16a16")
    assert body == {
        "git_ref": "refs/heads/feature/report",
        "expected_snapshot": "9" * 64,
        "measurements": {"performance": "c" * 64, "accuracy": "d" * 64},
    }
    assert key.startswith("model-measurements-")


def test_bind_requires_catalog_before_any_write(tmp_path, inputs):
    manifest, _ = prepare(tmp_path, inputs)

    def forbidden(*args):
        pytest.fail("no update is valid without a catalog snapshot")

    with pytest.raises(ValidationError, match="catalog must exist"):
        manifests.bind_measurement_keys(
            manifest,
            api_url="https://writer.invalid",
            read_api_url="https://reader.invalid",
            aws_region="us-west-2",
            read_request=lambda url: {},
            request=forbidden,
        )


def test_bind_reloads_catalog_after_concurrent_profile_update(tmp_path, inputs):
    from model_registry.registry import RegistryPublicationError

    manifest, _ = prepare(tmp_path, inputs)
    snapshots = iter(["8" * 64, "9" * 64])
    calls = []

    def request(method, path, body, key):
        calls.append(body)
        if len(calls) == 1:
            raise RegistryPublicationError("HTTP 409 catalog_snapshot_conflict: reload")
        return {}

    manifests.bind_measurement_keys(
        manifest,
        api_url="https://writer.invalid",
        read_api_url="https://reader.invalid",
        aws_region="us-west-2",
        read_request=lambda url: {"catalog": {"canonical_sha256": next(snapshots)}},
        request=request,
    )
    assert [call["expected_snapshot"] for call in calls] == ["8" * 64, "9" * 64]
    assert all(call["measurements"] == manifest["stage_keys"] for call in calls)
