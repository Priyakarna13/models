"""Select a model's benchmark implementation and run the ordinary Core workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PERFORMANCE_RUNNER = "tools/model_registry/performance.py"
DEFAULT_OPTIONS = {"num_samples": 1000, "min_duration_ms": 60000, "include_layer_stats": True}
DECODERS = {
    "yolo26_boxes": "YoloV26",
    "yolo26_pose": "YoloV26Pose",
    "yolo26_seg": "YoloV26Seg",
}
WORKLOAD_OPTIONS = {*DEFAULT_OPTIONS, "decoder", "top_k"}


def performance_script(model_dir: str | Path) -> Path:
    directory = Path(model_dir)
    custom = directory / "scripts/performance.py"
    return custom if custom.is_file() else directory.parents[1] / PERFORMANCE_RUNNER


def benchmark_options(entry: dict[str, Any], *, custom: bool = False) -> dict[str, Any]:
    from .errors import ValidationError

    declared = {key: entry[key] for key in WORKLOAD_OPTIONS if key in entry}
    if custom:
        if declared:
            raise ValidationError(
                "custom scripts/performance.py supports only --mpk and --output; "
                f"remove unsupported benchmark settings: {', '.join(sorted(declared))}"
            )
        return {}
    if "decoder" in declared and declared["decoder"] not in DECODERS:
        raise ValidationError(f"unsupported Core benchmark decoder: {declared['decoder']}")
    if "top_k" in declared and "decoder" not in declared:
        raise ValidationError("benchmark top_k requires an explicit decoder")
    return {**DEFAULT_OPTIONS, **({"top_k": 100} if "decoder" in declared else {}), **declared}


def validate_benchmarks(model: dict[str, Any], model_dir: str | Path) -> None:
    from .errors import ValidationError

    script = performance_script(model_dir)
    custom = script == Path(model_dir) / "scripts/performance.py"
    for entry in model.get("benchmark", []):
        benchmark_options(entry, custom=custom)
        if not custom and (
            len(model["components"]) != 1
            or entry["scope"]["type"] != "component"
            or model["components"][0]["interface"]["format"] != "onnx"
        ):
            raise ValidationError(
                "the shared Core benchmark supports one ONNX component; "
                "add scripts/performance.py for this workload"
            )


def run(mpk: str, output: str | Path, options: dict[str, Any]) -> None:
    """Save Core's complete report and the execution plan from the same selected model."""
    import pyneat

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    decoder = options.get("decoder")
    if decoder:
        model_options = pyneat.ModelOptions()
        model_options.decode_type = getattr(pyneat.BoxDecodeType, DECODERS[decoder])
        model_options.top_k = options["top_k"]
        model = pyneat.Model(mpk, model_options)
    else:
        model = pyneat.Model(mpk)
    requested = pyneat.BenchmarkOptions()
    for key in DEFAULT_OPTIONS:
        setattr(requested, key, options[key])
    model.benchmark(requested).save(destination / "benchmark-report.json")
    context = {
        "core_version": pyneat.__version__,
        **{key: getattr(requested, key) for key in DEFAULT_OPTIONS},
        "execution_plan": json.loads(model.execution_plan().to_json()),
    }
    if decoder:
        name, top_k = DECODERS[decoder], options["top_k"]
        selection = (
            f"{name} decoder, top {top_k}, 160x160 masks" if decoder == "yolo26_seg"
            else f"{name} box decoder, top {top_k}"
        )
        context["workload"] = {
            "selection": selection,
            "postprocess": model.info().selection.selected_post_kind,
        }
    (destination / "benchmark-context.json").write_text(json.dumps(context, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--output", required=True, help="directory for the Core report and benchmark context")
    parser.add_argument("--options", default="{}", help="effective workload options as JSON")
    args = parser.parse_args()
    options = json.loads(args.options)
    run(args.mpk, args.output, {**DEFAULT_OPTIONS, **options})


if __name__ == "__main__":
    main()
