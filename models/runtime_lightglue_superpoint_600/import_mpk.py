#!/usr/bin/env python3
"""Import the immutable NEAT-68 LightGlue MPK into the model registry."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

EXPECTED_VARIANT = "modalix_int8_neat68_regression"
EXPECTED_ARCHIVE = "quantized_lightglue_superpoint_600_mpk.tar.gz"
EXPECTED_MODEL_JSON = "quantized_lightglue_superpoint_600_mpk.json"


def _read_json(stream: tarfile.TarFile, name: str) -> dict:
    member = stream.extractfile(name)
    if member is None:
        raise ValueError(f"archive member is not a regular file: {name}")
    return json.load(member)


def validate_archive(path: Path) -> None:
    with tarfile.open(path, "r:gz") as stream:
        members = stream.getmembers()
        names = {member.name for member in members}
        for member in members:
            candidate = PurePosixPath(member.name)
            if candidate.is_absolute() or ".." in candidate.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")

        required = {EXPECTED_MODEL_JSON, "0_postproc.json", "pipeline_sequence.json"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"regression archive is missing required members: {missing}")
        if not any(name.endswith("_stage1_mla.elf") for name in names):
            raise ValueError("regression archive is missing its MLA ELF")

        model = _read_json(stream, EXPECTED_MODEL_JSON)
        if model.get("model_sdk_version") != "2.1.0":
            raise ValueError("unexpected Model SDK provenance")
        if [node.get("name") for node in model.get("input_nodes", [])] != [
            "kpts0", "desc0", "kpts1", "desc1"
        ]:
            raise ValueError("unexpected LightGlue input order")

        post = _read_json(stream, "0_postproc.json")
        expected = {
            "input_width": [601],
            "input_channels": [601],
            "slice_width": [67],
            "slice_channels": [601],
        }
        for key, value in expected.items():
            if post.get(key) != value:
                raise ValueError(f"unexpected {key}: {post.get(key)!r}")
        if post.get("buffers", {}).get("input", [{}])[0].get("size") != 365408:
            raise ValueError("unexpected tiled CBlock input storage size")
        if post.get("buffers", {}).get("output", {}).get("size") != 1444804:
            raise ValueError("unexpected FP32 output storage size")


def import_model(source: Path, output_dir: Path, variant: str) -> Path:
    if variant != EXPECTED_VARIANT:
        raise ValueError(f"unsupported variant: {variant}")
    validate_archive(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / EXPECTED_ARCHIVE
    shutil.copyfile(source, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = import_model(source, args.output_dir.expanduser().resolve(), args.variant)
    print(f"Imported immutable regression MPK: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
