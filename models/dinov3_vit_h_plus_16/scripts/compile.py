#!/usr/bin/env python3
"""Rebuild the converted DINOv3 ViT-H+/16 as components with surgery.py, compile them into Modalix MPKs and measure
their numerical fidelity against the converted ONNX.

surgery.py writes the four components pipeline.py lists, one ONNX each. Each is quantized to its build precision and
compiled in its own process, so the compiler's memory goes back to the system between them, and its MPK is audited:
one MLA plugin, no CPU plugin and the component's input. blocks_0 calibrates on the declared calibration images and
each later component on the float32 tokens the components before it give for the same images. For numerical
fidelity, the quantized components feed each other on the declared images, and blocks_3's class and patch embeddings
are compared with the converted model's.
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    TensorDRAMLayout,
    TensorTessellateParameters,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import load_images, select_names
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from pipeline import COMPONENTS, INPUTS, INTERFACES, OUTPUTS
from sima_utils.data.data_generator import DataGenerator
from surgery import prepare_model

LOG = logging.getLogger("dinov3_vit_h_plus_16.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "dinov3_vit_h_plus_16"
PRECISIONS = ("w16a16", "w8a16", "w8a8")


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _component_parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, dict[str, Any]]:
    """Each component's parameters in pipeline order, for a declared Modalix profile."""
    matches = [item for item in metadata["build_profiles"] if item["id"] == profile_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one build profile named {profile_id!r}")
    profile = matches[0]
    parameters = {binding["id"]: binding["parameters"] for binding in profile["components"]}
    if list(parameters) != list(COMPONENTS):
        raise ValueError(f"build profile {profile_id} must bind {', '.join(COMPONENTS)} in order")
    if profile["target"] != "modalix" or any(p.get("precision") not in PRECISIONS for p in parameters.values()):
        raise ValueError("this recipe supports only the declared Modalix BF16 and INT8 variants")
    if len({int(p["calibration_samples"]) for p in parameters.values()}) != 1:
        raise ValueError("every component calibrates on the same images")
    return parameters


def _quantization_config(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    if precision == "w8a8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w8a16":
        activation = bfloat16_scheme()
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w16a16":
        activation = weights = bfloat16_scheme()
    else:
        raise ValueError("this recipe supports only the declared Modalix BF16 and INT8 variants")
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        # AFE refuses to quantize ViT encoders without calibrated ranges, even at BF16.
        .with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
        .with_channel_equalization(False)
        # W8A16 moves activation outliers into the INT8 weights with SmoothQuant, measured on the calibration images.
        .with_smooth_quant(precision == "w8a16")
        # with_activation_quantization turns per-token quantization on for BF16. W8A16 keeps it, scaling each token's
        # activations on its own before an INT8-weight operation; W16A16 does not.
        .with_per_token_quant(precision == "w8a16")
    )


def _float32_bytes(shape: list[int]) -> int:
    return int(np.prod(shape)) * 4


def audit_mpk(archive: Path, component: str) -> dict[str, Any]:
    """Require one MLA plugin, no CPU plugins and the component's input."""
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0]}")
        manifest = json.load(source)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = manifest.get("plugins", [])
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid {component} MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    input_name = INPUTS[component]
    if public_inputs != {input_name: _float32_bytes(INTERFACES[component][input_name])}:
        raise RuntimeError(f"{component} MPK input interface mismatch: {public_inputs}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins)}


def to_afe(value: np.ndarray) -> np.ndarray:
    """A component's NCHW ONNX input as AFE takes it: channels last."""
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1))


def from_afe(component: str, values: list[np.ndarray]) -> list[np.ndarray]:
    """A component's AFE outputs in the ONNX layout. AFE returns 4D outputs channels last; the class and patch
    embeddings, whose shapes differ, are matched to their names by shape."""
    names = list(INTERFACES[component])[1:]
    by_shape = {np.asarray(value).shape: np.asarray(value, np.float32) for value in values}
    outputs = []
    for name in names:
        shape = tuple(INTERFACES[component][name])
        if len(shape) == 4:
            channels_last = (shape[0], *shape[2:], shape[1])
            if channels_last not in by_shape:
                raise RuntimeError(f"{component} returned no {name} of shape {channels_last}: {list(by_shape)}")
            outputs.append(by_shape[channels_last].transpose(0, 3, 1, 2))
        else:
            if shape not in by_shape:
                raise RuntimeError(f"{component} returned no {name} of shape {shape}: {list(by_shape)}")
            outputs.append(by_shape[shape])
    return outputs


def float_chain(graphs: dict[str, Path], images: list[np.ndarray]) -> dict[str, list[list[np.ndarray]]]:
    """Each component's float32 ONNX inputs for the NCHW images, and blocks_3's outputs under "outputs"."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    chain: dict[str, list[list[np.ndarray]]] = {}
    values = [[image] for image in images]
    for component in COMPONENTS:
        chain[component] = values
        session = ort.InferenceSession(str(graphs[component]), sess_options=options, providers=["CPUExecutionProvider"])
        values = [session.run(None, {INPUTS[component]: value[0]}) for value in values]
        del session
    chain["outputs"] = values
    return chain


def _build(
    graph: Path,
    build_dir: Path,
    name: str,
    component: str,
    parameters: dict[str, Any],
    calibration: list[np.ndarray],
    fidelity_inputs: list[np.ndarray],
) -> tuple[Path, list[list[np.ndarray]]]:
    """Quantize and compile one component: its audited MPK, and its quantized outputs for the fidelity inputs."""
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    input_name = INPUTS[component]
    shape = tuple(INTERFACES[component][input_name])
    importer = onnx_source(str(graph), shape_dict={input_name: shape}, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    samples = np.concatenate([to_afe(value) for value in calibration])
    data = convert_data_generator_to_iterable(DataGenerator({InputName(input_name): samples}))
    LOG.info("Quantizing %s as %s on %d images", component, parameters["precision"], len(calibration))
    # surgery.py's rank-4 [1, C, 1, T] layout is kept as it is: every layer stays on the MLA.
    try:
        quantized = loaded.quantize(
            calibration_data=data,
            quantization_config=_quantization_config(parameters),
            any_shape_on_mla=True,
            automatic_layout_conversion=False,
            model_name=name,
            log_level=logging.INFO,
        )
    except SystemExit as error:
        raise RuntimeError(f"quantizing {component} failed; the compiler's log above names the cause") from error
    outputs = [
        from_afe(component, quantized.execute(inputs={InputName(input_name): to_afe(value)}, use_jax=True))
        for value in fidelity_inputs
    ]
    build_dir.mkdir(parents=True)
    # The MLA reads its input as plain HWC, so no EV74 reorder runs before it.
    tessellation = {
        mla_input: TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC
        )
        for mla_input in quantized._net.nodes["MLA_0"].input_names
    }
    # The compiler exits rather than raising when compilation fails; a worker that exits never returns its result.
    try:
        _compile(quantized, build_dir, parameters, tessellation)
    except SystemExit as error:
        raise RuntimeError(f"compiling {component} failed; the compiler's log above names the cause") from error
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one {component} MPK, found {len(archives)} in {build_dir}")
    LOG.info("Compiled %s MPK: %s, %s", component, archives[0], audit_mpk(archives[0], component))
    return archives[0], outputs


def _compile(quantized: Any, build_dir: Path, parameters: dict[str, Any], tessellation: dict[str, Any]) -> None:
    quantized.compile(
        output_path=str(build_dir),
        batch_size=1,
        compress=True,
        log_level=logging.INFO,
        tessellate_parameters=tessellation,
        l2_caching_mode=L2CachingMode.NONE,
        preserve=True,
        deployable=True,
        retained_temporary_directory_name=None,
        use_power_limits=False,
        max_mla_power=None,
        # The option applies to every layer norm, and INT8 ones reject it; BF16 activations keep layer norms in BF16.
        layer_norm_use_fp32_intermediates=parameters["precision"] != "w8a8",
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )


def _in_process(*arguments: Any) -> tuple[Path, list[list[np.ndarray]]]:
    # Each component in its own process: the compiler's memory goes back to the system between them. A worker that
    # dies, killed for memory for instance, fails the build with BrokenProcessPool instead of leaving it waiting.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_build, *arguments).result()


@dataclass
class _OutputError:
    """Running sums of one output's error against the original ONNX, over every fidelity sample."""

    name: str
    count: int = 0
    dot: float = 0.0
    expected_squared: float = 0.0
    actual_squared: float = 0.0
    error_squared: float = 0.0
    error_absolute: float = 0.0
    max_absolute_error: float = 0.0

    def add(self, expected: np.ndarray, actual: np.ndarray) -> None:
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{self.name} shape mismatch: {expected.shape} != {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{self.name} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        self.count += expected.size
        self.dot += float(np.dot(expected.ravel(), actual.ravel()))
        self.expected_squared += float(np.sum(np.square(expected)))
        self.actual_squared += float(np.sum(np.square(actual)))
        self.error_squared += float(np.sum(np.square(error)))
        self.error_absolute += float(np.sum(absolute))
        self.max_absolute_error = max(self.max_absolute_error, float(np.max(absolute)))

    def summary(self) -> dict[str, Any]:
        expected_rms = max(np.sqrt(self.expected_squared / self.count), np.finfo(np.float64).eps)
        norm = np.sqrt(self.expected_squared * self.actual_squared)
        cosine = self.dot / norm if norm else float(self.expected_squared == self.actual_squared)
        return {
            "name": self.name,
            "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
            "normalized_rmse": float(np.sqrt(self.error_squared / self.count) / expected_rms),
            "mean_absolute_error": float(self.error_absolute / self.count),
            "max_absolute_error": self.max_absolute_error,
        }


def converted_outputs(source: Path, images: list[np.ndarray]) -> list[list[np.ndarray]]:
    """The converted model's class and patch embeddings for the NCHW images."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
    return [session.run(list(OUTPUTS), {INPUTS[COMPONENTS[0]]: image}) for image in images]


def _nchw(images: np.ndarray) -> list[np.ndarray]:
    return [np.ascontiguousarray(image.transpose(2, 0, 1)[None]) for image in images]


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path, fidelity_dataset: Path) -> Path:
    """Compile every component, then write compile-outputs.json and fidelity.json; returns the output directory."""
    metadata = _metadata()
    parameters = _component_parameters(metadata, profile_id)
    fidelity = metadata["numerical_fidelity"]
    selection = fidelity["selection"]
    samples = int(parameters[COMPONENTS[0]]["calibration_samples"])
    if int(selection["offset"]) < samples:
        raise ValueError("the fidelity images must follow the calibration images")
    if source.name != "model.onnx":
        raise ValueError(f"--model must be the converted model.onnx, not {source.name}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    LOG.info("Loading %d calibration images", samples)
    calibration = _nchw(load_images(dataset, select_names(dataset, int(selection["seed"]), 0, samples)))
    names = select_names(fidelity_dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    fidelity_images = _nchw(load_images(fidelity_dataset, names))
    LOG.info("Running the converted model on the fidelity images")
    reference = converted_outputs(source, fidelity_images)

    archives: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build = Path(temporary)
        (build / "surgery").mkdir()
        graphs = prepare_model(source, build / "surgery")
        derived = [{"component_id": c, "sha256": sha256_file(graphs[c])[0]} for c in COMPONENTS]
        LOG.info("Running the float32 components on the calibration images")
        calibration_chain = float_chain(graphs, calibration)
        # The fidelity chain: each quantized component takes the previous one's quantized tokens.
        values = fidelity_images
        for component in COMPONENTS:
            name = f"{MODEL_ID}_{profile_id}_{component}"
            archives[component], outputs = _in_process(
                graphs[component],
                build / name,
                name,
                component,
                parameters[component],
                [value[0] for value in calibration_chain[component]],
                values,
            )
            # Between components only the tokens travel on; blocks_3's class and patch embeddings are the result.
            values = [output[0] for output in outputs]
        output_dir.mkdir(parents=True)
        for component in COMPONENTS:
            (output_dir / component).mkdir()
            target = output_dir / component / archives[component].name
            shutil.copyfile(archives[component], target)
            archives[component] = target
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
        "derived_graphs": derived,
    }
    (output_dir / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")

    errors = [_OutputError(output) for output in fidelity["outputs"]]
    if [error.name for error in errors] != list(OUTPUTS):
        raise ValueError(f"numerical_fidelity must compare {', '.join(OUTPUTS)}")
    if len(outputs) != len(reference):
        raise RuntimeError(f"the quantized chain returned {len(outputs)} images for {len(reference)}")
    for expected, actual in zip(reference, outputs):
        for error, want, got in zip(errors, expected, actual):
            error.add(want, got)
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": fidelity["status"],
        "reference": "Original ONNX (ONNX Runtime)",
        "candidate": f"AFE {profile_id.upper()} components chained (JAX execute)",
        "dataset": {key: fidelity["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": [error.summary() for error in errors],
    }
    (output_dir / "fidelity.json").write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    LOG.info("Fidelity: %s", report["metrics"])
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    if args.dataset is None:
        parser.error("every precision calibrates; pass --dataset")
    if args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    compile_model(
        source,
        args.output_dir.expanduser().resolve(),
        args.build_profile,
        args.dataset.expanduser().resolve(),
        args.fidelity_dataset.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
