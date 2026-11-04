"""Prepare four ordinary ONNX publication inputs and bind returned CI pins.

No upload is performed. Each component uses the existing publisher's model.onnx
basename and models-onnx/lewam_pusht/<sha>/<run>-<attempt>/model.onnx key.
The bind command downloads/verifies every real returned pin before writing a
combined receipt. Candidate URLs or filenames are never treated as publication.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec

sys.path.insert(0, str(Path(__file__).parent / "model/scripts"))
from surgery import COMPONENTS, digest, validate_graph  # noqa: E402

CHECKPOINT = "e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb"
SOURCE_REVISION = "83c47030a80cee28f638680b76998d2f6a410914"


def read_stage(directory: Path) -> dict:
    receipt = json.loads((directory / "source-stage.json").read_text())
    if (
        receipt.get("schema"),
        receipt.get("model_id"),
        receipt.get("checkpoint_sha256"),
        receipt.get("source_revision"),
    ) != ("lewam.source-stage.v1", "lewam_pusht", CHECKPOINT, SOURCE_REVISION):
        raise ValueError("Unexpected staged source identity")
    if set(receipt.get("files", {})) != {f"{name}.onnx" for name in COMPONENTS}:
        raise ValueError("Exactly four staged sources required")
    for name in COMPONENTS:
        path = directory / f"{name}.onnx"
        if path.is_symlink():
            raise ValueError("Source symlinks are not publication inputs")
        actual = validate_graph(path, name)
        if actual != receipt["files"][path.name]:
            raise ValueError(f"Staged source receipt mismatch: {name}")
    return receipt


def prepare(directory: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    receipt = read_stage(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lewam-ci-", dir=output.parent) as temporary:
        staged = Path(temporary) / "input"
        staged.mkdir()
        jobs = {}
        for name in COMPONENTS:
            target = staged / name / "model.onnx"
            target.parent.mkdir()
            shutil.copyfile(directory / f"{name}.onnx", target)
            row = receipt["files"][f"{name}.onnx"]
            if digest(target) != row["sha256"] or target.stat().st_size != row["size_bytes"]:
                raise ValueError(f"Source changed while preparing CI input: {name}")
            jobs[name] = {"file": f"{name}/model.onnx", "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        report = {
            "schema": "lewam.ci-source-input.v1",
            "status": "STAGED_NOT_PUBLISHED",
            "model_id": "lewam_pusht",
            "sources": jobs,
            "source_stage_sha256": digest(directory / "source-stage.json"),
            "publisher": ".github/workflows/vulcan-model-convert.yml: Publish ONNX and print source pin",
        }
        (staged / "ci-input.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(output)
    return report


def bind(directory: Path, pins: dict[str, dict], cache: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if set(pins) != set(COMPONENTS):
        raise ValueError("Exactly four publication pins required")
    receipt = copy.deepcopy(read_stage(directory))
    specs = {}
    for name in COMPONENTS:
        value = pins[name].get("source", {})
        spec = SourceSpec.from_mapping(value)
        original = receipt["files"][f"{name}.onnx"]
        if (spec.sha256, spec.size_bytes) != (original["sha256"], original["size_bytes"]):
            raise ValueError(f"Published source differs from staged bytes: {name}")
        if value.get("type") != "file":
            raise ValueError("Each CI publication pin must describe one file")
        specs[name] = spec
    fetcher = Fetcher(cache)
    for name, spec in specs.items():
        fetcher.fetch(spec)
        receipt["files"][f"{name}.onnx"]["uri"] = spec.uri
    # Do not write a success receipt if even one public download failed.
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-stage", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    commit = commands.add_parser("bind")
    commit.add_argument("--source-stage", type=Path, required=True)
    commit.add_argument("--pin", action="append", required=True, help="component=returned source.yaml")
    commit.add_argument("--cache", type=Path, required=True)
    commit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_stage, args.output)
    else:
        pairs = [value.split("=", 1) for value in args.pin]
        if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(pairs):
            parser.error("Duplicate or malformed --pin")
        bind(
            args.source_stage,
            {name: yaml.safe_load(Path(path).read_text()) for name, path in pairs},
            args.cache,
            args.output,
        )


if __name__ == "__main__":
    main()
