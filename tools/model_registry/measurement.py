"""Run the model's measurement script with verified local inputs, identically in CI and locally."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact import compile_output_fields, validate_compile_outputs
from .dependencies import pin_benchmark_runtime
from .errors import ValidationError
from .fetch import resolve_dataset, sha256_file
from .measurement_environment import runner_environment
from .model import load_and_validate_model
from .model_report import load_report, validate_report


def verified_inputs(
    model: dict[str, Any],
    build_profile_id: str,
    compiled_dir: Path,
) -> list[dict[str, Any]]:
    """Verify the complete profile, including native deployment files, without registry receipts."""
    profile = next((item for item in model["build_profiles"] if item["id"] == build_profile_id), None)
    if profile is None:
        raise ValidationError(f"unknown build profile: {build_profile_id}")
    component_ids = [item["id"] for item in profile["components"]]
    directory = compiled_dir.expanduser().resolve() / model["id"] / build_profile_id
    path = directory / "compile-outputs.json"
    manifest = validate_compile_outputs(
        path,
        directory,
        model_id=model["id"],
        build_profile_id=build_profile_id,
        compile_invocation_id=component_ids[0] if len(component_ids) == 1 else "default",
        component_ids=component_ids,
    )
    inputs = {}
    for output in manifest["outputs"]:
        path_field, digest_field = compile_output_fields(output)
        artifact = Path(output[path_field])
        digest, size = sha256_file(artifact)
        inputs[output["component_id"]] = {
            "component_id": output["component_id"],
            "path": str(artifact),
            "sha256": digest,
            "size_bytes": size,
            "format": "llima" if path_field == "artifact_path" else "mpk",
        }
    return [inputs[component_id] for component_id in component_ids]


def prepare_clocks() -> None:
    """Use the same required Modalix clocks for local and hosted measurements."""
    clocks = (
        (
            Path("/sys/class/devfreq/5000000.mla/cur_freq"),
            Path("/sys/class/devfreq/5000000.mla/userspace/set_freq"),
            1_000_000_000,
        ),
        (
            Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq"),
            Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed"),
            1_400_000,
        ),
    )
    for current, setting, expected in clocks:
        if not current.exists():
            raise ValidationError(f"measurement requires a Modalix board: missing {current}")
        if int(current.read_text()) != expected:
            subprocess.run(
                ["sudo", "-n", "tee", str(setting)],
                input=f"{expected}\n",
                text=True,
                stdout=subprocess.DEVNULL,
                check=True,
            )
        if int(current.read_text()) != expected:
            raise ValidationError(f"measurement requires {current} = {expected}")


def prepare_dataset(entry: dict[str, Any], cache: Path, directory: Path) -> Path:
    """Resolve immutable data under a shared cache lock and unpack regular files only."""
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = Path(os.environ.get("DATASET_LOCK", str(cache.parent / "locks/prepare.lock")))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(os.environ.get("DATASET_LOCK_SECONDS", "1200"))
    with lock_path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValidationError(f"timed out waiting for dataset cache lock: {lock_path}") from None
                time.sleep(0.1)
        archive = resolve_dataset(entry["dataset"], cache)
    directory.mkdir()
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        for member in members:
            destination = (directory / member.name).resolve()
            if not destination.is_relative_to(directory.resolve()) or not (member.isfile() or member.isdir()):
                raise ValidationError(f"unsafe dataset archive member: {member.name}")
        source.extractall(directory, members=members, filter="data")
    entries = list(directory.iterdir())
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else directory


def enrich_report(
    report: dict[str, Any],
    *,
    model: dict[str, Any],
    build_profile_id: str,
    inputs: list[dict[str, Any]],
    script: Path,
    source_commit: str,
    environment: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Attach verified local provenance; a publisher supplies registered identities separately."""
    if "model" in report or "provenance" in report:
        raise ValidationError("model scripts must not supply registry-owned model or provenance fields")
    report["model"] = {"id": model["id"], "build_profile_id": build_profile_id}
    report["provenance"] = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "source": {"models_commit": source_commit, "script_sha256": sha256_file(script)[0]},
        "environment": environment,
    }
    for evidence in report["evidence"]:
        path = (output / evidence["path"]).resolve()
        if Path(evidence["path"]).is_absolute() or not path.is_relative_to(output.resolve()):
            raise ValidationError(f"evidence must remain inside result directory: {evidence['path']}")
        digest, size = sha256_file(path)
        evidence.update(sha256=digest, size_bytes=size)
    return validate_report(report, required_metrics=model[report["stage"]]["required_metrics"])


def run_measurement(
    model_dir: str | Path,
    build_profile_id: str,
    compiled_dir: str | Path,
    output_dir: str | Path,
    *,
    stage: str,
    python: str | None = None,
    dataset_cache: str | Path | None = None,
    environment: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one declared stage once for the entire model/profile and save its standard result."""
    if stage not in {"performance", "accuracy"}:
        raise ValidationError(f"unknown measurement stage: {stage}")
    directory = Path(model_dir).expanduser().resolve()
    repository = directory.parents[1]
    model = load_and_validate_model(directory)
    entry = model.get(stage)
    if not entry or build_profile_id not in entry["build_profiles"]:
        raise ValidationError(f"{model['id']} does not declare {stage} for {build_profile_id}")
    inputs = verified_inputs(model, build_profile_id, Path(compiled_dir))
    if environment is None:
        prepare_clocks()
        environment = runner_environment()
    required_runtime = pin_benchmark_runtime(repository) if runtime is None else runtime
    expected_platform = required_runtime["devkit"]["platform_version"]
    match = re.search(r"^\s*DISTRO_VERSION\s*=\s*(\S+)", environment.get("buildinfo", ""), re.MULTILINE)
    if not match or match.group(1) != expected_platform:
        raise ValidationError(f"measurement requires platform {expected_platform}")
    interpreter = python or os.environ.get("PYNEAT_PYTHON", str(Path.home() / "pyneat/bin/python"))
    core_version = subprocess.check_output(
        [interpreter, "-c", "import pyneat; print(pyneat.__version__)"],
        text=True,
    ).strip()
    if not core_version:
        raise ValidationError("could not identify the loaded Core runtime")
    environment["core_version"] = core_version
    target = required_runtime["neat_core"]["install_target"]
    revision = target.rsplit(":", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{12,40}", revision) or revision[:12] not in core_version:
        raise ValidationError(f"loaded Core {core_version!r} does not match immutable runtime {target!r}")
    environment["runtime"] = required_runtime
    script = directory / "scripts" / f"{stage}.py"
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    script_report = output / f"{stage}-report.json"
    # Never accidentally accept a stale report from an earlier invocation.
    script_report.unlink(missing_ok=True)
    command = [interpreter, str(script)]
    for item in inputs:
        option = "--deployment" if item["format"] == "llima" else "--mpk"
        path = str(Path(item["path"]).parent / "deployment") if item["format"] == "llima" else item["path"]
        value = f"{item['component_id']}={path}" if stage == "accuracy" or len(inputs) > 1 else path
        command.extend([option, value])
    command.extend(["--output", str(output if stage == "performance" else script_report)])
    source_commit = (
        os.environ.get("SOURCE_SHA")
        or subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    )
    with tempfile.TemporaryDirectory(prefix="model-measurement-", dir=output) as scratch:
        child_environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        if stage == "accuracy":
            cache = Path(dataset_cache or os.environ.get("DATASET_CACHE", str(Path.home() / ".cache/sima/datasets")))
            dataset = prepare_dataset(entry, cache, Path(scratch) / "dataset")
            command.extend(["--dataset", str(dataset)])
            requirements = directory / "scripts/requirements-accuracy.txt"
            if requirements.is_file():
                site = Path(scratch) / "site-packages"
                subprocess.run(
                    [
                        interpreter,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--disable-pip-version-check",
                        "--no-deps",
                        "--only-binary=:all:",
                        "--target",
                        str(site),
                        "-r",
                        str(requirements),
                    ],
                    check=True,
                )
                child_environment["PYTHONPATH"] = str(site)
        with (output / f"{stage}.log").open("w") as log:
            subprocess.run(command, env=child_environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = load_report(script_report, stage=stage, required_metrics=entry["required_metrics"])
    if report["context"].get("core_version") != core_version:
        raise ValidationError("script Core version differs from the selected runtime")
    if stage == "accuracy":
        report["context"]["validation_status"] = entry["status"]
        dataset = report["context"].get("dataset", {})
        report["context"]["dataset"] = {**dataset, **entry["dataset"]}
    report = enrich_report(
        report,
        model=model,
        build_profile_id=build_profile_id,
        inputs=inputs,
        script=script,
        source_commit=source_commit,
        environment=environment,
        output=output,
    )
    (output / f"{stage}-result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
