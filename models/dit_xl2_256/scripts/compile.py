#!/usr/bin/env python3
"""Split the verified DiT-XL/2 256 ONNX into its transformer and VAE decoder, quantize, and compile each to an MPK.

The converted ONNX holds both components side by side. Each is extracted, quantized to its build precision with one
seeded sample, compared with float32 ONNX Runtime on that sample, and compiled at its build profile's batch
size. For the feasibility check in #406 every MPK's plan is logged as a
FEASIBILITY line, which processor runs each plugin, instead of being required to be a single MLA segment.
compile-outputs.json records every MPK and derived graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnx
from afe.apis.defines import (
    BiasCorrectionType,
    InputName,
    RequantizationMode,
    SkipCalibration,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from dit import COMPONENTS, INPUT_SHAPES, OUTPUT_SHAPES, class_one_hot, tensor_name, timestep_features
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "dit_xl2_256"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
SAMPLE_SEED = 0


def _component_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    profiles = [profile for profile in load_metadata(MODEL_DIR)["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    parameters = {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}
    if list(parameters) != list(COMPONENTS):
        raise ValueError(f"build profile {profile_id} must bind {', '.join(COMPONENTS)} in order")
    return parameters


def _quantization(precision: str) -> Any:
    if precision != "w16a16":
        raise ValueError(f"unsupported precision: {precision}")
    # BF16 needs no calibrated ranges, and bias correction would only measure the one sample.
    return (
        default_quantization.with_activation_quantization(bfloat16_scheme())
        .with_weight_quantization(bfloat16_scheme())
        .with_calibration(SkipCalibration())
        .with_bias_correction(BiasCorrectionType.NONE)
        .with_requantization_mode(RequantizationMode.sima)
        .with_per_token_quant(False)
    )


def validate_source(source: Path) -> None:
    model = onnx.load(str(source), load_external_data=False)
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    expected_inputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()}
    expected_outputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"Unsupported DiT graph: inputs {inputs}, outputs {outputs}")


def split(source: Path, directory: Path) -> dict[str, Path]:
    """Extract each component's graph from the side-by-side ONNX."""
    directory.mkdir(parents=True)
    graphs = {}
    for component in COMPONENTS:
        graphs[component] = directory / f"{component}.onnx"
        onnx.utils.extract_model(
            str(source), str(graphs[component]),
            [tensor_name(component, name) for name in INPUT_SHAPES[component]],
            [tensor_name(component, name) for name in OUTPUT_SHAPES[component]],
        )
    return graphs


def sample(component: str) -> dict[str, np.ndarray]:
    """One seeded NCHW input set of one batch slot: a mid-schedule timestep, class 207, and a standard-normal latent
    or hidden grid."""
    rng = np.random.RandomState(SAMPLE_SEED)
    conditioning = {"timestep_features": timestep_features(500.0)[:1], "class_one_hot": class_one_hot(207)[:1]}
    return {name: conditioning[name] if name in conditioning else rng.standard_normal(shape).astype(np.float32)
            for name, shape in INPUT_SHAPES[component].items()}


def _channels_last(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1))


def _cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    expected, actual = expected.ravel().astype(np.float64), actual.ravel().astype(np.float64)
    return float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))


def quantize(graph: Path, name: str, component: str, precision: str) -> tuple[Any, float]:
    """The quantized component, and its output's cosine with float32 ONNX Runtime on the seeded sample."""
    import onnxruntime as ort

    inputs = sample(component)
    names = {tensor_name(component, key): value for key, value in inputs.items()}
    importer = onnx_source(str(graph), shape_dict={key: INPUT_SHAPES[component][key.removeprefix(f"{component}_")]
                                                   for key in names}, layout="NCHW")
    loaded = load_model(importer, target=gen2_target)
    try:
        quantized = loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(
                DataGenerator({InputName(key): _channels_last(value) for key, value in names.items()})
            ),
            quantization_config=_quantization(precision),
            model_name=name,
            log_level=logging.INFO,
        )
    except SystemExit as error:
        raise RuntimeError(f"quantizing {component} failed; the AFE log above names the cause") from error
    (expected,) = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"]).run(None, names)
    try:
        (actual,) = quantized.execute({key: _channels_last(value) for key, value in names.items()}, fast_mode=True)
        actual = np.asarray(actual, np.float32)
        if actual.shape != expected.shape:
            actual = actual.transpose(0, 3, 1, 2)
        similarity = _cosine(expected, actual)
    except Exception as error:  # the feasibility build still compiles when the simulator cannot run the graph
        logging.warning("Could not simulate quantized %s: %s", component, error)
        similarity = float("nan")
    return quantized, similarity


def _read_mpk_manifest(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        manifests = [member for member in stream.getmembers() if member.name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {[m.name for m in manifests]}")
        return json.load(stream.extractfile(manifests[0]))


def plan_of(archive: Path) -> dict[str, Any]:
    """Where the MPK runs: each plugin's processor and kernel, in order, and the compiler that built it."""
    manifest = _read_mpk_manifest(archive)
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    steps = [{"processor": str(p.get("processor", "")).upper(),
              "kernel": str(p.get("config_params", {}).get("kernel", ""))} for p in plugins]
    return {
        "compiler": str(manifest.get("model_sdk_version")),
        "mla_segments": sum(step["processor"] == "MLA" for step in steps),
        "host_plugins": [step for step in steps if step["processor"] in {"A65", "APU"}],
        "plugins": steps,
    }


def _compile(quantized: Any, build_dir: Path, component: str, batch_size: int) -> Path:
    build_dir.mkdir(parents=True)
    try:
        quantized.compile(
            output_path=str(build_dir),
            batch_size=batch_size,
            compress=True,
            log_level=logging.INFO,
            layer_norm_use_fp32_intermediates=True,
        )
    except SystemExit as error:
        raise RuntimeError(f"compiling {component} failed; the AFE log above names the cause") from error
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component} MPK, found {len(archives)}")
    plan = plan_of(archives[0])
    if plan["compiler"] != COMPILER_VERSION:
        raise RuntimeError(f"{component} MPK compiler mismatch: expected {COMPILER_VERSION}, found {plan['compiler']}")
    return archives[0]


def _build(graph: Path, build_dir: Path, name: str, component: str, precision: str,
           batch_size: int) -> tuple[Path, float]:
    """Quantize and compile one component: its MPK and its output's cosine with float32."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    quantized, similarity = quantize(graph, name, component, precision)
    return _compile(quantized, build_dir, component, batch_size), similarity


def compile_model(source: Path, output_dir: Path, profile_id: str) -> dict[str, Path]:
    parameters = _component_parameters(profile_id)
    validate_source(source)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build = Path(temporary)
        graphs = split(source, build / "graphs")
        output_dir.mkdir(parents=True)
        archives = {}
        for component in COMPONENTS:
            name = f"{MODEL_ID}_{profile_id}_{component}"
            precision = str(parameters[component]["precision"])
            batch_size = int(parameters[component].get("batch_size", 1))
            logging.info("Quantizing %s as %s", component, precision)
            # Each component in its own process: the compiler's memory goes back to the system between them.
            with multiprocessing.get_context("spawn").Pool(1) as pool:
                built, similarity = pool.apply(_build, (graphs[component], build / name, name, component, precision,
                                                        batch_size))
            plan = plan_of(built)
            logging.info("FEASIBILITY %s", json.dumps({"component": component, "batch_size": batch_size,
                                                        "float32_cosine": similarity, **plan}))
            (output_dir / component).mkdir()
            archives[component] = output_dir / component / built.name
            shutil.copyfile(built, archives[component])
        manifest = {
            "model_id": MODEL_ID,
            "build_profile_id": profile_id,
            "compile_invocation_id": profile_id,
            "outputs": [
                {
                    "component_id": component,
                    "mpk_path": f"{component}/{archives[component].name}",
                    "mpk_sha256": sha256_file(archives[component])[0],
                }
                for component in COMPONENTS
            ],
            "derived_graphs": [
                {"component_id": component, "sha256": sha256_file(graphs[component])[0]} for component in COMPONENTS
            ],
        }
        (output_dir / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
    return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="unused: no build of this model takes a calibration dataset")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx" or not source.is_file():
        parser.error(f"--model must be the verified ONNX from the conversion workflow: {source}")
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    for component, archive in compile_model(source, args.output_dir.expanduser().resolve(),
                                            args.build_profile).items():
        logging.info("Compiled %s: %s", component, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
