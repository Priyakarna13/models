from __future__ import annotations

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import yaml
from model_registry import conversion
from model_registry.errors import ValidationError
from model_registry.fingerprints import compile_fingerprint
from model_registry.model import load_and_validate_model
from model_registry.registry import RegistryPublicationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    for name in (
        "schemas", "taxonomies", "datasets", "scripts",
        "tools/model_registry", "tools/model_report", ".github/workflows",
    ):
        shutil.copytree(ROOT / name, root / name)
    for name in ("conftest.py", "pyproject.toml"):
        shutil.copyfile(ROOT / name, root / name)
    shutil.copytree(ROOT / "models/yolo26x", root / "models/yolo26x")
    path = root / "models/yolo26x/model.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["compilation"]["source"] = "conversion"
    path.write_text(yaml.safe_dump(doc))
    return root


def output(directory, external=False, external_location="weights.bin"):
    directory.mkdir()
    tensor = onnx.numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="weights")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "weights"], ["y"])],
        "model",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 2])],
        [tensor],
    )
    model = onnx.helper.make_model(graph)
    onnx.save_model(
        model,
        directory / "model.onnx",
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=external_location,
        size_threshold=0,
    )
    return directory


class Registry:
    def __init__(self):
        self.calls = []
        self.inputs = []
        self.status = "running"
        self.latest = "conversion-1"
        self.artifacts = {
            run: {
                "id": run + "-artifact",
                "artifact_type": "model-onnx",
                "object_status": "available",
                "s3_bucket": "bucket",
                "s3_key": run + "/onnx.tar.gz",
                "s3_version_id": "version",
                "sha256": char * 64,
                "size_bytes": 123,
                "media_type": "application/gzip",
                "metadata": {"validation": "passed"},
            }
            for run, char in [("conversion-1", "a"), ("conversion-2", "b")]
        }

    def request(self, method, path, body, key):
        self.calls.append((method, path, body, key))
        if path == "/v1/runs":
            return {"run_id": "plan"}
        if path.endswith("/inputs/conversion"):
            self.inputs = [{"name": "conversion", **body}]
        if path.endswith("/finalize"):
            self.status = body["status"]
        return {}

    def read(self, url):
        if url.endswith("/provenance"):
            return {"run": {"status": self.status}, "inputs": self.inputs}
        if "/resolve?" in url:
            if self.latest is None:
                raise RegistryPublicationError("artifact registry HTTP 404 fingerprint_not_found")
            return {"run": {"id": self.latest}}
        return {"artifact": self.artifacts[url.split("/")[-3]]}


def plan(root, registry):
    model = load_and_validate_model(root / "models/yolo26x")
    return conversion.plan_conversions(
        root,
        {"yolo26x": model},
        read=registry.read,
        read_api_url="https://registry.test",
        client=registry,
        branch="feature",
    )


def test_schema_is_offline_and_requires_conversion(repository):
    model = load_and_validate_model(repository / "models/yolo26x")
    assert model["components"][0]["source"] == {"type": "conversion"}
    path = repository / "models/yolo26x/model.yaml"
    doc = yaml.safe_load(path.read_text())
    del doc["conversion"]
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValidationError, match="requires an ONNX"):
        load_and_validate_model(path.parent)


def test_retry_retains_selection_and_new_plan_gets_latest(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    registry = Registry()
    first = plan(repository, registry)
    registry.latest = "conversion-2"
    assert plan(repository, registry) == first
    newer = Registry()
    newer.latest = "conversion-2"
    assert plan(repository, newer)["yolo26x"]["sha256"] == "b" * 64
    assert "download_url" not in first["yolo26x"]


def test_missing_and_unavailable_fail_without_fallback(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    registry = Registry()
    registry.latest = None
    with pytest.raises(ValidationError, match="run Vulcan Model Convert"):
        plan(repository, registry)
    registry.latest = "conversion-1"
    registry.artifacts[registry.latest]["object_status"] = "deleted"
    with pytest.raises(ValidationError, match="unavailable"):
        plan(repository, registry)


def test_compile_fingerprint_depends_on_content_not_location(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    selected = plan(repository, Registry())

    def fingerprint(value):
        return compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources=value)

    first = fingerprint(selected)
    other = copy.deepcopy(selected)
    other["yolo26x"].update(run_id="new-run", s3_bucket="production", read_api_url="https://production.test")
    assert fingerprint(other) == first
    other["yolo26x"]["sha256"] = "b" * 64
    assert fingerprint(other) != first


@pytest.mark.parametrize("filename", ["model.onnx", "weights.bin", "onnx.tar.gz"])
def test_bundle_rejects_output_collisions_without_changing_model_data(tmp_path, filename):
    location = filename if filename != "model.onnx" else "weights.bin"
    directory = output(tmp_path / "output", external=True, external_location=location)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValidationError, match="would overwrite model data"):
        conversion.bundle_onnx(directory, directory / filename)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_bundle_rejects_output_alias_of_tensor_data(tmp_path, link_type):
    directory = output(tmp_path / "output", external=True)
    destination = tmp_path / "bundle.tar.gz"
    tensor = directory / "weights.bin"
    if link_type == "symlink":
        destination.symlink_to(tensor)
    else:
        destination.hardlink_to(tensor)
    original = tensor.read_bytes()
    with pytest.raises(ValidationError, match="would overwrite model data"):
        conversion.bundle_onnx(directory, destination)
    assert tensor.read_bytes() == original


@pytest.mark.parametrize("external", [False, True])
def test_bundle_is_deterministic_and_fetch_checks_exact_version(tmp_path, monkeypatch, external):
    directory = output(tmp_path / "output", external)
    first = conversion.bundle_onnx(directory, tmp_path / "first.tar.gz")
    second = conversion.bundle_onnx(directory, tmp_path / "second.tar.gz")
    assert first.read_bytes() == second.read_bytes()
    sha, size = conversion.sha256_file(first)
    source = {
        "type": "conversion",
        "run_id": "run",
        "artifact_id": "artifact",
        "sha256": sha,
        "size_bytes": size,
        "s3_bucket": "bucket",
        "s3_key": "key",
        "s3_version_id": "v1",
        "media_type": "application/gzip",
        "read_api_url": "https://registry.test",
    }
    artifact = {
        **source,
        "id": "artifact",
        "artifact_type": "model-onnx",
        "object_status": "available",
        "metadata": {"validation": "passed"},
        "download_url": first.as_uri(),
    }
    monkeypatch.setattr("model_registry.conversion.read_registry_json", lambda _: {"artifact": artifact})
    fetched, hit, _, _ = conversion.fetch_conversion(source, tmp_path / "cache")
    assert not hit
    onnx.checker.check_model(str(fetched))
    assert (fetched.parent / "weights.bin").exists() == external
    assert conversion.fetch_conversion(source, tmp_path / "cache")[1]
    artifact["s3_version_id"] = "v2"
    with pytest.raises(ValidationError, match="saved plan"):
        conversion.fetch_conversion(source, tmp_path / "cache")


def test_publication_completes_only_after_validated_output(repository, tmp_path):
    directory = output(tmp_path / "output")
    registry = Registry()
    store = SimpleNamespace(ensure_native_sha256=lambda path, key, sha: SimpleNamespace(key=key, version_id="v1"))
    kwargs = dict(
        client=registry,
        store=store,
        bucket="bucket",
        environment="staging",
        branch="feature",
        git_sha="a" * 40,
        workflow="42",
    )
    with pytest.raises(ValidationError, match="passed"):
        conversion.publish_conversion(repository, "yolo26x", directory, validation="skipped", **kwargs)
    assert not registry.calls
    result = conversion.publish_conversion(repository, "yolo26x", directory, validation="passed", **kwargs)
    assert result["run_id"] == "plan"
    assert registry.calls[-1][1].endswith("/finalize")
    artifact = next(c[2] for c in registry.calls if c[1].endswith("/artifacts/onnx"))
    assert artifact["s3_key"].startswith("models/conversions/yolo26x/")
    assert artifact["metadata"]["validation"] == "passed"


def test_planning_resolves_before_compile_reuse(repository, monkeypatch):
    from model_registry import stage_plan

    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    registry = Registry()
    selected = plan(repository, registry)
    fingerprint = compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources=selected)
    catalog = {
        "models": [
            {
                "id": "yolo26x",
                "builds": [{"component_id": "model", "build_profile_id": "w16a16", "compile_fingerprint": fingerprint}],
            }
        ]
    }
    monkeypatch.setattr(stage_plan, "read_catalog_sources", lambda *args: (None, [catalog]))
    monkeypatch.setattr(stage_plan, "benchmark_fingerprint", lambda *args: "c" * 64)
    monkeypatch.setattr(stage_plan, "stage_run_exists", lambda *args: True)

    def stages(client):
        return stage_plan.plan_stages(
            repository,
            compiler_version="test",
            read=client.read,
            read_api_url="https://registry.test",
            branch="feature",
            profile_ids=["w16a16"],
            conversion_client=client,
        )

    assert stages(registry)["compile"]["include"] == []
    newer = Registry()
    newer.latest = "conversion-2"
    assert len(stages(newer)["compile"]["include"]) == 1
    newer.artifacts["conversion-2"]["sha256"] = "a" * 64
    newer.inputs = []
    newer.status = "running"
    assert stages(newer)["compile"]["include"] == []


def test_conversion_fingerprint_tracks_recipe_and_dependency_pins(repository):
    model = load_and_validate_model(repository / "models/yolo26x")
    original = conversion.conversion_fingerprint(repository, model)
    requirements = repository / "models/yolo26x/scripts/requirements-convert.txt"
    requirements.write_text(requirements.read_text() + "\n# new exporter version\n")
    assert conversion.conversion_fingerprint(repository, model) != original
    with pytest.raises(ValidationError, match="plan-stages"):
        compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources={})


@pytest.mark.parametrize("relative", conversion.CONVERSION_TOOLING)
def test_shared_conversion_dependency_invalidates_saved_selection(repository, relative):
    model = load_and_validate_model(repository / "models/yolo26x")
    original = conversion.conversion_fingerprint(repository, model)
    path = repository / relative
    path.write_bytes(path.read_bytes() + b"\n")
    assert conversion.conversion_fingerprint(repository, model) != original
    with pytest.raises(ValidationError, match="checkout's recipe"):
        conversion.apply_selection(model, {"yolo26x": {"conversion_fingerprint": original}}, root=repository)


def test_saved_selection_rejects_a_different_recipe(repository, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    selected = plan(repository, Registry())
    script = repository / "models/yolo26x/scripts/convert.py"
    script.write_text(script.read_text() + "\n# changed recipe\n")
    with pytest.raises(ValidationError, match="checkout's recipe"):
        compile_fingerprint(repository, "yolo26x", "w16a16", "w16a16", "test", conversion_sources=selected)


def test_conversion_workflow_uses_shared_environment_and_passed_tests():
    workflow = yaml.safe_load((ROOT / ".github/workflows/vulcan-model-convert.yml").read_text())
    job = workflow["jobs"]["convert"]
    assert job["environment"] == "${{ needs.config.outputs.vulcan_env }}"
    tests = next(step for step in job["steps"] if step.get("name", "").startswith("Test the converted"))
    assert '[[ "$outcome" == passed ]]' in tests["run"]
    publish = job["steps"][-1]["run"]
    assert "publish-conversion" in publish
    assert "models-onnx" not in publish and "sima-neat-artifacts-staging" not in publish


def test_on_demand_conversion_resolves_catalog_identity_without_compiling(repository, monkeypatch):
    from model_registry.stage_plan import plan_stages
    path = repository / "models/yolo26x/model.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["build-on-demand"] = True
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setattr("model_registry.stage_plan.read_catalog_sources", lambda *args: (None, []))
    registry = Registry()
    result = plan_stages(repository, compiler_version="test", read=registry.read,
                         read_api_url="https://registry.test", branch="feature", conversion_client=registry)
    assert not result["compile"]["include"]
    assert result["conversion_sources"]["yolo26x"]["run_id"] == "conversion-1"
