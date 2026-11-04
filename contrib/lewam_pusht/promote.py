"""Bind real publication receipts and atomically promote the staged model.

Normal source publication returns a source-stage.json copy with a `uri` added
to each file. The separate Dataset Registry receipt supplies `calibration`
{id, variant, sha256, size_bytes}. Nothing here uploads or approves artifacts.
The default promotion enables compile only. Accuracy/performance remain absent
until registered evaluation data and target qualification are available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

COMPONENTS = ("front", "body", "head", "policy")
CHECKPOINT = "e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb"
SOURCE_REVISION = "83c47030a80cee28f638680b76998d2f6a410914"


def definition(receipt: dict, calibration: dict) -> dict:
    if (
        receipt.get("schema"),
        receipt.get("model_id"),
        receipt.get("checkpoint_sha256"),
        receipt.get("source_revision"),
    ) != ("lewam.source-stage.v1", "lewam_pusht", CHECKPOINT, SOURCE_REVISION):
        raise ValueError("Unexpected source publication identity")
    names = {f"{name}.onnx" for name in COMPONENTS}
    if set(receipt.get("files", {})) != names:
        raise ValueError("Publication receipt must name exactly the four ONNX files")
    files = {}
    for name, entry in receipt["files"].items():
        if not all(key in entry for key in ("uri", "sha256", "size_bytes")):
            raise ValueError(f"Missing publication receipt fields for {name}")
        files[name] = {key: entry[key] for key in ("uri", "sha256", "size_bytes")}
    if (calibration.get("id"), calibration.get("variant")) != ("pusht", "cal"):
        raise ValueError("Expected the pusht/cal Dataset Registry receipt")
    value = yaml.safe_load((Path(__file__).parent / "model/definition.yaml").read_text())
    value["compilation"]["source"] = {"main_file": "front.onnx", "files": files}
    value["calibrations"] = [{"id": "pusht-cal", "dataset": calibration}]
    return value


def promote(receipt: dict, calibration: dict, repository: Path, cache: Path) -> Path:
    destination = repository / "models/lewam_pusht"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    authored = definition(receipt, calibration)
    # A sibling staging directory keeps final rename on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".lewam-promote-", dir=repository) as temporary:
        staged = Path(temporary) / "models/lewam_pusht"
        shutil.copytree(Path(__file__).parent / "model", staged)
        (staged / "definition.yaml").unlink()
        (staged / "model.yaml").write_text(yaml.safe_dump(authored, sort_keys=False))
        metadata = load_and_validate_model(staged, repository / "schemas/model.schema.json")
        # The public Fetcher independently checks every published byte and size.
        Fetcher(cache).fetch_source(metadata["components"][0]["source"])
        # Dataset availability is checked by the ordinary compile stage's
        # Dataset Registry resolver; a receipt is not an approved deployment.
        staged.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(
        promote(
            json.loads(args.source_receipt.read_text()),
            json.loads(args.calibration_receipt.read_text()),
            args.repository.resolve(),
            args.cache,
        )
    )


if __name__ == "__main__":
    main()
