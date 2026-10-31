#!/usr/bin/env python3
"""Split the verified MobileSAM ONNX into its components, quantize, and compile each to an MPK.

The converted ONNX holds the two image encoder parts and the prompt decoder side by side. Each is extracted,
quantized to its build precision and compiled in its own process, and its MPK is audited: one MLA segment, no host
code. Each component calibrates on the float32 outputs of the components before it. When model.yaml declares
numerical_fidelity, the quantized components feed each other on the declared val2017 images and fixed prompts, and
the decoder's outputs are compared with the float32 ONNX chain's. compile-outputs.json records every MPK.
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
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import calibration_images, decode, encoder_inputs, fidelity_names, image_prompts, read_images
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from pipeline import COMPONENTS, INPUT_SHAPES, OUTPUT_SHAPES, tensor_name
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "mobile_sam"
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


def _quantization(parameters: dict[str, Any]) -> Any:
    precision = str(parameters["precision"])
    if precision == "w8a8":
        config = (
            default_quantization.with_activation_quantization(
                quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            )
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
            .with_bias_correction(BiasCorrectionType.REGULAR)
        )
    elif precision == "w16a16":
        config = default_quantization.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(
            bfloat16_scheme()
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")
    # with_activation_quantization turns per-token quantization on for BF16; neither precision uses it.
    return config.with_requantization_mode(RequantizationMode.sima).with_per_token_quant(False)


def validate_source(source: Path) -> None:
    model = onnx.load(str(source), load_external_data=False)
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    expected_inputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()}
    expected_outputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"Unsupported MobileSAM graph: inputs {inputs}, outputs {outputs}")


def split(source: Path, directory: Path) -> dict[str, Path]:
    """Extract each component's graph from the side-by-side ONNX."""
    directory.mkdir(parents=True)
    graphs = {}
    for component in COMPONENTS:
        graphs[component] = directory / f"{component}.onnx"
        onnx.utils.extract_model(
            str(source),
            str(graphs[component]),
            [tensor_name(component, name) for name in INPUT_SHAPES[component]],
            [tensor_name(component, name) for name in OUTPUT_SHAPES[component]],
        )
    return graphs


def channels_last(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1)) if value.ndim == 4 else value


def channels_first(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """An SDK output in the ONNX shape: 4D outputs come back channels-last."""
    value = np.asarray(value, np.float32)
    if len(shape) == 4 and value.shape == (shape[0], *shape[2:], shape[1]):
        value = value.transpose(0, 3, 1, 2)
    if value.shape != shape:
        value = value.reshape(shape)
    return value


def onnx_session(graph: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])


def float_embeddings(graphs: dict[str, Path], images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The float32 encoder parts' NCHW features and embeddings of NHWC encoder inputs."""
    first, second = onnx_session(graphs["image_encoder_0"]), onnx_session(graphs["image_encoder_1"])
    name = tensor_name("image_encoder_0", "image")
    features = np.concatenate([first.run(None, {name: image.transpose(2, 0, 1)[None]})[0] for image in images])
    name = tensor_name("image_encoder_1", "features")
    return features, np.concatenate([second.run(None, {name: feature[None]})[0] for feature in features])


def decoder_feeds(embeddings: np.ndarray, sizes: list[tuple[int, int]]) -> list[dict[str, np.ndarray]]:
    """The decoder's NCHW inputs: each image's embedding with each of its fixed prompts."""
    return [
        {"image_embeddings": embedding[None], **prompt}
        for embedding, size in zip(embeddings, sizes)
        for prompt in image_prompts(size)
    ]


def calibration_data(component: str, samples: list[dict[str, np.ndarray]]) -> Any:
    """The quantizer's channels-last calibration batches, one input set at a time."""
    names = list(INPUT_SHAPES[component])
    stacked = {
        InputName(tensor_name(component, name)): np.concatenate([channels_last(sample[name]) for sample in samples])
        for name in names
    }
    return convert_data_generator_to_iterable(DataGenerator(stacked))


def _read_mpk_manifest(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        manifests = [member for member in members if member.name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {[member.name for member in manifests]}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0].name}")
        return json.load(source)


def audit_mpk(archive: Path, component: str) -> None:
    """Require the component on one MLA segment, with no host code, from the pinned compiler."""
    manifest = _read_mpk_manifest(archive)
    processors = [str(plugin.get("processor", "")).upper() for plugin in manifest.get("plugins", [])]
    if processors.count("MLA") != 1 or {"A65", "APU"} & set(processors):
        raise RuntimeError(f"invalid {component} MPK: plugins run on {processors}")
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"{component} MPK compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}"
        )


def _build(
    graph: Path,
    build_dir: Path,
    name: str,
    component: str,
    parameters: dict[str, Any],
    samples: list[dict[str, np.ndarray]],
    fidelity_feeds: list[dict[str, np.ndarray]],
) -> tuple[Path, list[list[np.ndarray]]]:
    """Quantize and compile one component: its MPK, and its quantized outputs for the fidelity feeds."""
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    names = list(INPUT_SHAPES[component])
    importer = onnx_source(
        str(graph),
        shape_dict={tensor_name(component, key): INPUT_SHAPES[component][key] for key in names},
        layout="NCHW",
    )
    loaded = load_model(importer, target=gen2_target)
    logging.info("Quantizing %s as %s with %d sample(s)", component, parameters["precision"], len(samples))
    try:
        quantized = loaded.quantize(
            calibration_data=calibration_data(component, samples),
            quantization_config=_quantization(parameters),
            model_name=name,
            log_level=logging.INFO,
        )
    except SystemExit as error:
        raise RuntimeError(f"quantizing {component} failed; the AFE log above names the cause") from error
    outputs = []
    for feeds in fidelity_feeds:
        result = quantized.execute(
            {InputName(tensor_name(component, key)): channels_last(value) for key, value in feeds.items()},
            fast_mode=True,
        )
        outputs.append(
            [channels_first(value, shape) for value, shape in zip(result, OUTPUT_SHAPES[component].values())]
        )
    build_dir.mkdir(parents=True)
    try:
        quantized.compile(output_path=str(build_dir), batch_size=1, compress=True, log_level=logging.INFO)
    except SystemExit as error:
        raise RuntimeError(f"compiling {component} failed; the AFE log above names the cause") from error
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component} MPK, found {len(archives)}")
    audit_mpk(archives[0], component)
    return archives[0], outputs


def _in_process(*arguments: Any) -> tuple[Path, list[list[np.ndarray]]]:
    # Each component in its own process: the compiler's memory goes back to the system between them.
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(_build, arguments)


def _metrics(name: str, expected: list[np.ndarray], actual: list[np.ndarray]) -> dict[str, Any]:
    expected_all = np.concatenate([value.ravel() for value in expected]).astype(np.float64)
    actual_all = np.concatenate([value.ravel() for value in actual]).astype(np.float64)
    if not (np.isfinite(expected_all).all() and np.isfinite(actual_all).all()):
        raise RuntimeError(f"{name} contain non-finite values")
    error = actual_all - expected_all
    norm = np.linalg.norm(expected_all) * np.linalg.norm(actual_all)
    expected_rms = max(np.sqrt(np.mean(expected_all**2)), np.finfo(np.float64).eps)
    return {
        "name": name,
        "cosine_similarity": float(np.clip(expected_all @ actual_all / norm if norm else 1.0, -1.0, 1.0)),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / expected_rms),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def compile_model(
    source: Path, output_dir: Path, profile_id: str, dataset: Path | None, fidelity_dataset: Path | None
) -> dict[str, Path]:
    parameters = _component_parameters(profile_id)
    validate_source(source)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    fidelity = load_metadata(MODEL_DIR).get("numerical_fidelity")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build = Path(temporary)
        graphs = split(source, build / "graphs")

        # Calibration: the first encoder part's images, the second's float32 features of them, and the decoder's
        # prompts on the float32 encoder's embeddings.
        int8 = [component for component in COMPONENTS if parameters[component]["precision"] == "w8a8"]
        if int8 and dataset is None:
            raise ValueError("an INT8 component requires --dataset from dataset-registry")
        rng = np.random.RandomState(SAMPLE_SEED)
        samples: dict[str, list[dict[str, np.ndarray]]] = {}
        if int8:
            count = max(int(parameters[component]["calibration_samples"]) for component in int8)
            images = calibration_images(dataset, count)
            inputs = encoder_inputs(images)
            samples["image_encoder_0"] = [{"image": image.transpose(2, 0, 1)[None]} for image in inputs]
            if {"image_encoder_1", "prompt_decoder"} & set(int8):
                features, embeddings = float_embeddings(graphs, inputs)
                samples["image_encoder_1"] = [{"features": feature[None]} for feature in features]
                samples["prompt_decoder"] = decoder_feeds(embeddings, [image.shape[:2] for image in images])
        # BF16 derives nothing from representative data, but quantize() still needs one sample.
        for component in ("image_encoder_0", "image_encoder_1"):
            (name, shape), = INPUT_SHAPES[component].items()
            samples.setdefault(component, [{name: rng.uniform(-2, 2, shape).astype(np.float32)}])
        samples.setdefault(
            "prompt_decoder",
            decoder_feeds(
                rng.standard_normal(INPUT_SHAPES["prompt_decoder"]["image_embeddings"]).astype(np.float32) * 0.3,
                [(768, 1024)],
            )[:1],
        )

        # Fidelity: the declared val2017 images, preprocessed, and their float32 ONNX chain.
        fidelity_inputs, sizes = np.zeros((0,)), []
        if fidelity is not None:
            if fidelity_dataset is None:
                raise ValueError("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
            selection = fidelity["selection"]
            names = fidelity_names(
                fidelity_dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"])
            )
            encoded = read_images(fidelity_dataset, names)
            images = [decode(encoded[name]) for name in names]
            fidelity_inputs, sizes = encoder_inputs(images), [image.shape[:2] for image in images]

        output_dir.mkdir(parents=True)
        archives = {}
        # The fidelity chain: each quantized component takes the previous one's quantized outputs.
        feeds = [{"image": image.transpose(2, 0, 1)[None]} for image in fidelity_inputs]
        for component, output in (("image_encoder_0", "features"), ("image_encoder_1", "image_embeddings")):
            name = f"{MODEL_ID}_{profile_id}_{component}"
            archives[component], outputs = _in_process(
                graphs[component], build / name, name, component, parameters[component], samples[component], feeds
            )
            feeds = [{output: item[0]} for item in outputs]
        quantized_embeddings = np.concatenate([item["image_embeddings"] for item in feeds]) if sizes else None
        name = f"{MODEL_ID}_{profile_id}_prompt_decoder"
        built, decoded_outputs = _in_process(
            graphs["prompt_decoder"],
            build / name,
            name,
            "prompt_decoder",
            parameters["prompt_decoder"],
            samples["prompt_decoder"],
            decoder_feeds(quantized_embeddings, sizes) if sizes else [],
        )
        archives["prompt_decoder"] = built

        for component, archive in list(archives.items()):
            (output_dir / component).mkdir()
            archives[component] = output_dir / component / archive.name
            shutil.copyfile(archive, archives[component])
            logging.info("Compiled %s: %s", component, archives[component])
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
        (output_dir / "compile-outputs.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if fidelity is not None:
            reference = onnx_session(graphs["prompt_decoder"])
            feeds = decoder_feeds(float_embeddings(graphs, fidelity_inputs)[1], sizes)
            expected = [
                reference.run(None, {tensor_name("prompt_decoder", key): value for key, value in feed.items()})
                for feed in feeds
            ]
            masks = _metrics("masks", [item[0] for item in expected], [item[0] for item in decoded_outputs])
            agreement = np.mean(
                [np.mean((want[0] > 0) == (got[0] > 0)) for want, got in zip(expected, decoded_outputs)]
            )
            logging.info("Mask pixels on the same side of 0 as the float32 chain: %.4f", agreement)
            report = {
                "schema_version": 1,
                "model_id": MODEL_ID,
                "build_profile_id": profile_id,
                "status": fidelity["status"],
                "reference": "float32 ONNX encoder parts and decoder (ONNX Runtime)",
                "candidate": "quantized encoder parts feeding the quantized decoder (AFE execute)",
                "dataset": {key: fidelity["dataset"][key] for key in ("id", "variant", "sha256")},
                "selection": fidelity["selection"],
                "samples": len(sizes),
                "metrics": [
                    masks,
                    _metrics("iou_predictions", [item[1] for item in expected], [item[1] for item in decoded_outputs]),
                ],
            }
            (output_dir / "fidelity.json").write_text(
                json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            logging.info("Fidelity: %s", report["metrics"])
    return archives


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
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx" or not source.is_file():
        parser.error(f"--model must be the verified ONNX from the conversion workflow: {source}")
    if load_metadata(MODEL_DIR).get("numerical_fidelity") is not None and args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    fidelity_dataset = args.fidelity_dataset.expanduser().resolve() if args.fidelity_dataset else None
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile, dataset, fidelity_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
