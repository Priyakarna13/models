"""Compile the full encoder and package separate dynamic INT8 CPU decoding models."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from model_registry.artifact import record_runtime_files
from model_registry.model import load_model as load_metadata
from model_registry.onnx_export import partition

MODEL_DIR = Path(__file__).resolve().parents[1]


def runtime_contract(archive: Path, logical_outputs: list[str]) -> dict:
    """Read native tensor shapes without modifying the compiler-produced archive."""
    with tarfile.open(archive, "r|gz") as stream:
        manifests = [json.load(stream.extractfile(member)) for member in stream if member.name.endswith("_mpk.json")]
    if len(manifests) != 1:
        raise ValueError("Expected exactly one compiled manifest")
    plugins = manifests[0]["plugins"]
    accelerators = [plugin for plugin in plugins if plugin["processor"] == "MLA"]
    casts = [plugin for plugin in plugins if plugin["config_params"].get("kernel") == "cast_transform"]
    host = [plugin for plugin in plugins if plugin["processor"] != "MLA"]
    if len(accelerators) != 1 or any(
        plugin["config_params"].get("kernel") not in {"cast_transform", "pass_through"} for plugin in host
    ):
        raise ValueError(
            "Encoder must contain one accelerator region with only boundary conversions and identity output publication"
        )
    accelerator = accelerators[0]
    produced = {port["name"]: plugin for plugin in casts for port in plugin["output_nodes"]}
    consumed = {port["name"]: plugin for plugin in casts for port in plugin["input_nodes"]}
    inputs, outputs = [], []
    for port in accelerator["input_nodes"]:
        cast = produced[port["name"]]
        params = cast["config_params"]["params"]
        if params["out_dtype"] != "bfloat16":
            raise ValueError("Expected native BF16 encoder inputs")
        inputs.append(dict(name=cast["input_nodes"][0]["name"], shape=params["output_shapes"][0], bytes=port["size"]))
    for name, port in zip(logical_outputs, accelerator["output_nodes"], strict=True):
        params = consumed[port["name"]]["config_params"]["params"]
        if params["in_dtype"] != "bfloat16":
            raise ValueError("Expected native BF16 encoder outputs")
        outputs.append(dict(name=name, shape=params["input_shapes"][0], bytes=port["size"]))
    return dict(inputs=inputs, outputs=outputs)


def cpu_model(graph, outputs: list[str], source_dir: Path, destination: Path, prefix: str) -> None:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    part = partition(graph, outputs)
    for node in part.graph.node:
        for ports in (node.input, node.output):
            for index, name in enumerate(ports):
                ports[index] = name.removeprefix(prefix)
    for values in (part.graph.input, part.graph.output, part.graph.value_info, part.graph.initializer):
        for value in values:
            value.name = value.name.removeprefix(prefix)
    onnx.external_data_helper.load_external_data_for_model(part, str(source_dir))
    onnx.external_data_helper.convert_model_from_external_data(part)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        source = Path(directory) / "float.onnx"
        onnx.save_model(part, source)
        # Fold checkpoint Transpose/Reshape constants before dynamic quantization,
        # which requires constant MatMul weights to be initializers.
        optimized = Path(directory) / "optimized.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(optimized)
        session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        del session
        quantize_dynamic(
            optimized,
            destination,
            per_channel=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Gather", "LSTM", "MatMul"],
        )
    onnx.checker.check_model(str(destination))


def prepare_model(source: Path, temporary: Path, component: Path) -> None:
    """Release all export and CPU-quantization objects before the encoder compiler starts."""
    import onnx

    graph = onnx.load(source, load_external_data=False)
    config = json.loads(next(item.value for item in graph.metadata_props if item.key == "sima.asr"))
    for name in ("predictor", "joiner"):
        cpu_model(
            graph,
            config[name + "_outputs"],
            source.parent,
            component / (name + "_cpu_int8.onnx"),
            name + "/",
        )
    encoder = partition(graph, config["encoder_outputs"])
    weights = {
        onnx.external_data_helper.ExternalDataInfo(tensor).location
        for tensor in encoder.graph.initializer
        if onnx.external_data_helper.uses_external_data(tensor)
    }
    for name in weights:
        try:
            os.link(source.parent / name, temporary / name)
        except OSError:
            shutil.copyfile(source.parent / name, temporary / name)
    onnx.save_model(encoder, temporary / "encoder.onnx")
    (temporary / "config.json").write_text(json.dumps(config))


def compile_model(source: Path, profile: str, output: Path) -> Path:
    metadata = load_metadata(MODEL_DIR)
    selected = next(item for item in metadata["build_profiles"] if item["id"] == profile)
    if selected["components"][0]["parameters"]["precision"] != "w8a16":
        raise ValueError("Only INT8 weights with BF16 activations are supported")
    name = metadata["id"] + "_" + profile
    component = output / "model"
    component.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=output, prefix="encoder-") as temporary:
        temporary = Path(temporary)
        worker = multiprocessing.get_context("spawn").Process(target=prepare_model, args=(source, temporary, component))
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Source preparation failed: {worker.exitcode}")
        config = json.loads((temporary / "config.json").read_text())
        encoder_path = temporary / "encoder.onnx"
        parameters = selected["components"][0]["parameters"]
        keys = (
            ("chunk_ms", "lookahead")
            if config["kind"] == "nemotron"
            else ("left_seconds", "chunk_seconds", "right_seconds")
        )
        expected = {"precision": "w8a16", **{key: config[key] for key in keys}}
        if parameters != expected:
            raise ValueError(f"Build parameters must match the pinned encoder: {expected}")
        from afe.apis.defines import (
            BiasCorrectionType,
            RequantizationMode,
            SkipCalibration,
            bfloat16_scheme,
            default_quantization,
            gen2_target,
            quantization_scheme,
        )
        from afe.apis.loaded_net import load_model
        from afe.load.importers.general_importer import onnx_source

        imported = load_model(
            onnx_source(str(encoder_path), layout="NCHW", flexible_batch_size=False), target=gen2_target
        )
        quantization = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(SkipCalibration())
            .with_bias_correction(BiasCorrectionType.NONE)
            .with_per_token_quant(False)
            .with_channel_equalization(False)
            .with_smooth_quant(False)
        )
        quantized = imported.quantize(
            calibration_data=None, quantization_config=quantization, model_name=name, log_level=logging.INFO
        )
        quantized.compile(
            output_path=str(temporary / "compiled"),
            batch_size=1,
            compress=True,
            deployable=True,
            preserve=True,
            layer_norm_use_fp32_intermediates=True,
        )
        archives = list((temporary / "compiled").rglob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise ValueError("Expected one full encoder archive")
        archive = component / (name + "_mpk.tar.gz")
        shutil.copyfile(archives[0], archive)
    config.update(runtime_contract(archive, config["encoder_outputs"]))
    config.pop("encoder_outputs")
    frontend = config.pop("frontend")
    arrays = {
        key: np.frombuffer(base64.b64decode(value["data"]), dtype="<f4").reshape(value["shape"])
        for key, value in frontend.items()
    }
    np.savez(component / "frontend.npz", **arrays)
    for filename in ("tokenizer.json", "processor_config.json"):
        shutil.copyfile(source.parent / filename, component / filename)
    for source_name, target_name in (
        ("transcribe.py", "transcribe.py"),
        ("requirements-runtime.txt", "requirements.txt"),
        ("LICENSE.txt", "LICENSE.txt"),
        ("NOTICE.txt", "NOTICE.txt"),
    ):
        shutil.copyfile(MODEL_DIR / "scripts" / source_name, component / target_name)
    (component / "runtime_config.json").write_text(json.dumps(config, indent=2) + "\n")
    record_runtime_files(component, metadata["runtime_files"])
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("Compile invocation must equal the build profile")
    logging.basicConfig(level=logging.INFO)
    compile_model(args.model.resolve(), args.build_profile, args.output_dir.resolve())


if __name__ == "__main__":
    main()
