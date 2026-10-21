"""Local execution uses complete verified artifacts and returns the same hosted report."""

import hashlib
import json
import sys

import pytest
from model_registry import measurement
from model_registry.cli import author_main
from model_registry.errors import ValidationError
from model_registry.model_report import load_report


def build(tmp_path, components=("model",)):
    root = tmp_path / "compiled"
    directory = root / "example" / "w8a8"
    outputs = []
    for component in components:
        path = directory / component / f"{component}_mpk.tar.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(component.encode())
        outputs.append(
            {
                "component_id": component,
                "mpk_path": str(path.relative_to(directory)),
                "mpk_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "model_id": "example",
        "build_profile_id": "w8a8",
        "compile_invocation_id": components[0] if len(components) == 1 else "default",
        "outputs": outputs,
    }
    if len(components) > 1:
        correctness = directory / components[-1] / "fidelity.json"
        correctness.write_text("{}")
        manifest.update(
            derived_graphs=[{"component_id": name, "sha256": "a" * 64} for name in components],
            correctness={
                "component_id": components[-1],
                "path": str(correctness.relative_to(directory)),
                "sha256": hashlib.sha256(correctness.read_bytes()).hexdigest(),
            },
        )
    (directory / "compile-outputs.json").write_text(json.dumps(manifest))
    model = {
        "id": "example",
        "build_profiles": [{"id": "w8a8", "components": [{"id": name} for name in components]}],
        "performance": {"build_profiles": ["w8a8"], "required_metrics": ["model.inference_rate"]},
    }
    return model, root


@pytest.mark.parametrize("components", [("model",), ("backbone", "decoder")])
def test_complete_local_compilation_needs_no_publication_receipts(tmp_path, components):
    model, root = build(tmp_path, components)
    inputs = measurement.verified_inputs(model, "w8a8", root)
    assert [item["component_id"] for item in inputs] == list(components)
    assert all(item["size_bytes"] > 0 for item in inputs)
    assert not list(root.rglob("registry-publication.json"))


@pytest.mark.parametrize("corruption", ["bytes", "missing", "extra", "path"])
def test_local_compilation_rejects_untrusted_artifacts(tmp_path, corruption):
    model, root = build(tmp_path)
    directory = root / "example/w8a8"
    manifest_path = directory / "compile-outputs.json"
    manifest = json.loads(manifest_path.read_text())
    path = directory / manifest["outputs"][0]["mpk_path"]
    if corruption == "bytes":
        path.write_bytes(b"modified")
    if corruption == "missing":
        path.unlink()
    if corruption == "extra":
        manifest["outputs"].append(manifest["outputs"][0])
    if corruption == "path":
        manifest["outputs"][0]["mpk_path"] = "../other_mpk.tar.gz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError):
        measurement.verified_inputs(model, "w8a8", root)


@pytest.mark.parametrize("explicit_runtime", [True, False])
def test_runner_calls_one_script_then_adds_verified_provenance(tmp_path, monkeypatch, explicit_runtime):
    model, root = build(tmp_path, ("backbone", "decoder"))
    model_dir = tmp_path / "models/example"
    script = model_dir / "scripts/performance.py"
    script.parent.mkdir(parents=True)
    script.write_text("""import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--mpk',action='append')
parser.add_argument('--output')
args=parser.parse_args()
assert len(args.mpk)==2
out=Path(args.output)
(out/'raw.json').write_text('{}')
(out/'performance-report.json').write_text(json.dumps({
'schema':'sima.model-report','schema_version':1,'stage':'performance',
'context':{'core_version':'1.0+aaaaaaaaaaaa'},
'metrics':[{'name':'model.inference_rate','status':'measured','unit':'inferences/s','value':100}],
'evidence':[{'path':'raw.json','media_type':'application/json'}]}))
""")
    monkeypatch.setattr(measurement, "load_and_validate_model", lambda directory: model)
    runtime = {"devkit": {"platform_version": "3.0.0"}, "neat_core": {"install_target": "core@develop:aaaaaaaaaaaa"}}
    pinned = []

    def pin(repository):
        pinned.append(repository)
        return runtime

    monkeypatch.setattr(measurement, "pin_benchmark_runtime", pin)
    original = measurement.subprocess.check_output
    monkeypatch.setattr(
        measurement.subprocess,
        "check_output",
        lambda command, **kwargs: "1.0+aaaaaaaaaaaa\n" if "-c" in command else original(command, **kwargs),
    )
    monkeypatch.setenv("SOURCE_SHA", "b" * 40)
    output = tmp_path / "results"
    result = measurement.run_measurement(
        model_dir,
        "w8a8",
        root,
        output,
        stage="performance",
        python=sys.executable,
        runtime=runtime if explicit_runtime else None,
        environment={"buildinfo": "DISTRO_VERSION = 3.0.0"},
    )
    assert pinned == ([] if explicit_runtime else [tmp_path])
    assert result["model"] == {"id": "example", "build_profile_id": "w8a8"}
    assert result["provenance"]["environment"]["runtime"] == runtime
    assert len(result["provenance"]["inputs"]) == 2
    assert result["provenance"]["source"]["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert result["evidence"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert load_report(output / "performance-result.json") == result
    with pytest.raises(ValidationError, match="immutable runtime"):
        measurement.run_measurement(
            model_dir,
            "w8a8",
            root,
            output,
            stage="performance",
            python=sys.executable,
            runtime={**runtime, "neat_core": {"install_target": "core@develop"}},
            environment={"buildinfo": "DISTRO_VERSION = 3.0.0"},
        )


def test_enrichment_rejects_forged_identity_and_escaped_evidence(tmp_path):
    model, root = build(tmp_path)
    script = tmp_path / "performance.py"
    script.write_text("# source")
    arguments = dict(
        model=model,
        build_profile_id="w8a8",
        inputs=measurement.verified_inputs(model, "w8a8", root),
        script=script,
        source_commit="a" * 40,
        environment={},
        output=tmp_path,
    )
    with pytest.raises(ValidationError, match="must not supply"):
        measurement.enrich_report({"model": {}}, **arguments)
    report = {"stage": "performance", "evidence": [{"path": "../outside.json"}]}
    with pytest.raises(ValidationError, match="inside result"):
        measurement.enrich_report(report, **arguments)


def test_dataset_resolution_uses_shared_board_lock_and_times_out(tmp_path, monkeypatch):
    import fcntl

    cache = tmp_path / "datasets/artifacts"
    lock_path = tmp_path / "datasets/locks/prepare.lock"
    lock_path.parent.mkdir(parents=True)
    monkeypatch.setenv("DATASET_LOCK_SECONDS", "0")
    monkeypatch.delenv("DATASET_LOCK", raising=False)
    monkeypatch.setattr(measurement, "resolve_dataset", lambda *args: pytest.fail("locked cache must not resolve"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValidationError, match="timed out waiting for dataset cache lock"):
            measurement.prepare_dataset({"dataset": {}}, cache, tmp_path / "unpack")


def test_default_cli_rejects_installed_core_mismatch_before_script_execution(tmp_path, monkeypatch, capsys):
    model, root = build(tmp_path)
    runtime = {"devkit": {"platform_version": "3.0.0"}, "neat_core": {"install_target": "core@develop:aaaaaaaaaaaa"}}
    monkeypatch.setattr(measurement, "load_and_validate_model", lambda _: model)
    monkeypatch.setattr(measurement, "pin_benchmark_runtime", lambda _: runtime)
    monkeypatch.setattr(measurement.subprocess, "check_output", lambda *args, **kwargs: "1.0+bbbbbbbbbbbb")
    monkeypatch.setattr(
        measurement.subprocess, "run", lambda *args, **kwargs: pytest.fail("mismatched runtime executed model code")
    )
    monkeypatch.setenv("SOURCE_SHA", "c" * 40)
    monkeypatch.setattr(measurement, "prepare_clocks", lambda: None)
    monkeypatch.setattr(measurement, "runner_environment", lambda: {"buildinfo": "DISTRO_VERSION = 3.0.0"})
    status = author_main(
        [
            "performance",
            str(tmp_path / "models/example"),
            "--build-profile",
            "w8a8",
            "--compiled-dir",
            str(root),
            "--output",
            str(tmp_path / "results"),
            "--python",
            sys.executable,
        ]
    )
    assert status == 2
    assert "does not match immutable runtime" in capsys.readouterr().err
