"""Stage self-contained exports outside Git for the normal artifact publication job.

This does not upload, compile, or claim that an exported graph reproduces the
qualified direct ELF. Publication of source inputs is distinct from approval of
a compiled model. Keep the emitted directory immutable after receipt creation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "model/scripts"))
from surgery import COMPONENTS, digest, validate_graph  # noqa: E402

CHECKPOINT = "e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb"
SOURCE_REVISION = "83c47030a80cee28f638680b76998d2f6a410914"


def stage(graphs: dict[str, Path], output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if set(graphs) != set(COMPONENTS):
        raise ValueError("Exactly front, body, head, policy graphs are required")
    # Validate everything before creating a partially populated publication input.
    reports = {name: validate_graph(graphs[name], name) for name in COMPONENTS}
    manifest = {
        "schema": "lewam.source-stage.v1",
        "model_id": "lewam_pusht",
        "checkpoint_sha256": CHECKPOINT,
        "source_revision": SOURCE_REVISION,
        "scope": "Interface-checked sources, not numerical or native qualification",
        "files": {f"{name}.onnx": reports[name] for name in COMPONENTS},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lewam-source-", dir=output.parent) as temporary:
        staged = Path(temporary) / "sources"
        staged.mkdir()
        for name in COMPONENTS:
            target = staged / f"{name}.onnx"
            shutil.copyfile(graphs[name], target)
            if digest(target) != reports[name]["sha256"]:
                raise ValueError("Source changed while staging")
        (staged / "source-stage.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="append", required=True, help="component=local ONNX")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [x.split("=", 1) for x in args.graph]
    if any(len(x) != 2 for x in pairs) or len(dict(pairs)) != len(pairs):
        parser.error("Duplicate or malformed --graph")
    stage({name: Path(path) for name, path in pairs}, args.output)


if __name__ == "__main__":
    main()
