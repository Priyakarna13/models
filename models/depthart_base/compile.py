#!/usr/bin/env python3
"""Compile the qualified DepthART Base strict-INT8 SiMaIR snapshot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec

MODEL_DIR = Path(__file__).resolve().parent
MODEL_NAME = "r46_branchsilu_nyu1536_int8io.sima"
EXPECTED_ONNX_SHA256 = "25fb78ac659c6936dc2cac9a739262e6c8a028d4e2f6f8a65350016ee8458d79"
EXPECTED_ELF_SHA256 = "5d887d7aed1ca44ec22c5edc3ed1efaf3389aa3da01c58dc27efe8a6027f72d0"
EXPECTED_GUARDED_EMITTERS = {
    "MLA_0/tuple_concat_47_conv",
    "MLA_0/tuple_concat_67_conv",
}
PUBLIC_INPUT = {"name": "image", "dtype": "int8", "shape": [1, 448, 576, 3]}
PUBLIC_OUTPUT = {"dtype": "int8", "shape": [1, 448, 576, 16]}


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["compile_variants"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one DepthART variant named {variant_id!r}")
    variant = matches[0]
    parameters = variant["parameters"]
    required = {
        "precision": "int8",
        "public_input_dtype": "int8",
        "public_output_dtype": "int8",
        "tessellation_owner": "mla",
        "require_single_mla": True,
        "require_zero_float_compute": True,
        "expected_elf_sha256": EXPECTED_ELF_SHA256,
    }
    mismatches = {
        name: (parameters.get(name), value) for name, value in required.items() if parameters.get(name) != value
    }
    if variant["target"] != "modalix" or mismatches:
        raise ValueError(f"unsupported DepthART compile contract: target={variant['target']}, mismatches={mismatches}")
    if parameters.get("expected_guarded_concat_emitters") != len(EXPECTED_GUARDED_EMITTERS):
        raise ValueError("expected_guarded_concat_emitters changed without updating the audited compiler guard")
    return variant


def audit_onnx_source(path: Path) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto

    digest, size = sha256_file(path)
    if digest != EXPECTED_ONNX_SHA256:
        raise RuntimeError(f"DepthART QDQ source changed: expected {EXPECTED_ONNX_SHA256}, found {digest}")
    model = onnx.load(str(path), load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    outputs = list(model.graph.output)

    def tensor_contract(value: Any) -> tuple[str, int, list[int]]:
        tensor = value.type.tensor_type
        shape = [int(dimension.dim_value) for dimension in tensor.shape.dim]
        return value.name, int(tensor.elem_type), shape

    if [tensor_contract(value) for value in inputs] != [("image", TensorProto.FLOAT, [1, 3, 448, 576])]:
        raise RuntimeError("DepthART ONNX input contract changed")
    if [tensor_contract(value) for value in outputs] != [("depth", TensorProto.FLOAT, [1, 16, 448, 576])]:
        raise RuntimeError("DepthART ONNX output contract changed")
    operator_counts: dict[str, int] = {}
    for node in model.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    expected = {"Conv": 146, "QuantizeLinear": 377, "DequantizeLinear": 595}
    if any(operator_counts.get(name) != count for name, count in expected.items()):
        raise RuntimeError(f"DepthART ONNX QAT topology changed: expected {expected}, found {operator_counts}")
    return {"sha256": digest, "size_bytes": size, "operator_counts": operator_counts}


def _internal_model_name(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".sima.yaml")]
    if len(members) != 1:
        raise RuntimeError(f"expected exactly one .sima.yaml member, found {members}")
    return Path(members[0]).name.removesuffix(".yaml")


def _stage_snapshot(source: Path, directory: Path) -> tuple[Path, str]:
    internal_name = _internal_model_name(source)
    staged = directory / internal_name
    shutil.copyfile(source, staged)
    return staged, internal_name


def _disable_impossible_int16_exact_mode(requested: bool, ifm_layouts: Any, ofm_layout: Any) -> bool:
    if not requested:
        return False
    return not (any(layout.dtype.is_int16_type for layout in ifm_layouts) or ofm_layout.dtype.is_int16_type)


@contextlib.contextmanager
def compiler_compatibility_guards() -> Iterator[set[str]]:
    """Install only the two compiler guards required by the qualified graph.

    These are process-local and restored after compilation. They must be
    removed from this recipe after the corresponding Model Compiler fixes are
    released and the rebuilt artifact is requalified.
    """

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    import afe.backends.mla.afe_to_n2a_compiler.defines as afe_mla_defines
    import afe.backends.mla.afe_to_n2a_compiler.n2a_compiler_operations as afe_mla_operations
    import mlc.compiler.model_graph.json_utils as mlc_json_utils
    from mlc.kernel.gen_conv2d import Conv2DKernelEmitter

    guarded_emitters: set[str] = set()
    original_init = Conv2DKernelEmitter.__init__
    original_export = mlc_json_utils.export_model_to_json
    original_defines_export = afe_mla_defines.export_model_to_json
    original_operations_export = afe_mla_operations.export_model_to_json

    def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("int16_exact_mode") is True and _disable_impossible_int16_exact_mode(True, args[7], args[11]):
            kwargs["int16_exact_mode"] = False
            guarded_emitters.add(str(args[3]))
        original_init(self, *args, **kwargs)

    def export_with_deep_graph_guard(*args: Any, **kwargs: Any) -> Any:
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
        pickle_module = mlc_json_utils.pickle
        accelerated_dump = pickle_module.dump

        def pure_python_dump(
            value: Any,
            stream: Any,
            protocol: int | None = None,
            *,
            fix_imports: bool = True,
            buffer_callback: Any = None,
        ) -> None:
            pickle_module._Pickler(
                stream,
                protocol=protocol,
                fix_imports=fix_imports,
                buffer_callback=buffer_callback,
            ).dump(value)

        pickle_module.dump = pure_python_dump
        try:
            return original_export(*args, **kwargs)
        finally:
            pickle_module.dump = accelerated_dump

    Conv2DKernelEmitter.__init__ = guarded_init
    mlc_json_utils.export_model_to_json = export_with_deep_graph_guard
    afe_mla_defines.export_model_to_json = export_with_deep_graph_guard
    afe_mla_operations.export_model_to_json = export_with_deep_graph_guard
    try:
        yield guarded_emitters
    finally:
        Conv2DKernelEmitter.__init__ = original_init
        mlc_json_utils.export_model_to_json = original_export
        afe_mla_defines.export_model_to_json = original_defines_export
        afe_mla_operations.export_model_to_json = original_operations_export


def tessellation_parameters(model: Any) -> dict[str, Any]:
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.ir.node import node_is_tuple

    subnet_nodes = [node for node in model._net.nodes.values() if hasattr(node.ir, "nodes")]
    if len(subnet_nodes) != 1:
        raise RuntimeError(f"expected one MLA subnet, found {[str(node.name) for node in subnet_nodes]}")
    mla = subnet_nodes[0]
    parameters = {
        name: TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC,
        )
        for name in mla.input_names
    }
    output_node = mla.ir.nodes[mla.ir.output_node_name]
    output_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    parameters.update(
        {
            f"{name}_output": TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC,
            )
            for name in output_names
        }
    )
    if len(parameters) != 2:
        raise RuntimeError(f"expected one MLA input and output tessellation contract, found {list(parameters)}")
    return parameters


def audit_mpk(archive: Path, *, expected_elf_sha256: str) -> dict[str, Any]:
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
        elf_members = [member for member in members if member.name.endswith(".elf")]
        if len(elf_members) != 1:
            raise RuntimeError(f"expected one MLA ELF, found {[member.name for member in elf_members]}")
        elf_source = stream.extractfile(elf_members[0])
        if elf_source is None:
            raise RuntimeError(f"cannot read MLA ELF: {elf_members[0].name}")
        elf_digest = hashlib.sha256(elf_source.read()).hexdigest()
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    if len(plugins) != 1 or str(plugins[0].get("processor", "")).upper() != "MLA":
        raise RuntimeError(f"strict one-MLA contract failed: {plugins}")
    if len(elfs) != 1 or shared_objects:
        raise RuntimeError(f"strict package payload failed: elfs={elfs}, shared_objects={shared_objects}")
    if elf_digest != expected_elf_sha256:
        raise RuntimeError(
            "compiled MLA ELF does not match the board-qualified program: "
            f"expected {expected_elf_sha256}, found {elf_digest}"
        )
    inputs = manifest.get("input_nodes", [])
    output_nodes = plugins[0].get("output_nodes", [])
    if len(inputs) != 1 or any(inputs[0].get(key) != value for key, value in PUBLIC_INPUT.items()):
        raise RuntimeError(f"public INT8 input contract changed: {inputs}")
    if len(output_nodes) != 1 or any(output_nodes[0].get(key) != value for key, value in PUBLIC_OUTPUT.items()):
        raise RuntimeError(f"public INT8 output contract changed: {output_nodes}")
    digest, size = sha256_file(archive)
    return {
        "sha256": digest,
        "size_bytes": size,
        "elfs": elfs,
        "elf_sha256": elf_digest,
        "plugin_count": len(plugins),
        "public_input": inputs[0],
        "public_output": output_nodes[0],
    }


def compile_variant(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    parameters = variant["parameters"]
    source_report = audit_onnx_source(source)
    quantized_source = SourceSpec.from_mapping(parameters["quantized_source"])
    quantized_path, _cache_hit = Fetcher().fetch(quantized_source)
    quantized_digest, quantized_size = sha256_file(quantized_path)
    if (quantized_digest, quantized_size) != (quantized_source.sha256, quantized_source.size_bytes):
        raise RuntimeError("verified quantized snapshot changed after fetch")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    sdk_bin = Path(sys.executable).parent
    if shutil.which("mla-masm") is None and (sdk_bin / "mla-masm").is_file():
        os.environ["PATH"] = f"{sdk_bin}:{os.environ.get('PATH', '')}"

    from afe.apis.error_handling_variables import enable_verbose_error_messages
    from afe.apis.model import Model

    enable_verbose_error_messages()
    with tempfile.TemporaryDirectory(prefix="depthart-strict-int8-") as temporary:
        staged, internal_name = _stage_snapshot(quantized_path, Path(temporary))
        try:
            model = Model.load(model_name=internal_name, network_directory=str(staged.parent))
        except Exception as exc:
            raise RuntimeError(
                "the pinned Model Compiler cannot deserialize the qualified strict-INT8 SiMaIR; "
                "install the release containing QMLA boundary configuration support"
            ) from exc
        if model is None:
            raise RuntimeError("Model Compiler returned no model for the qualified SiMaIR snapshot")
        with compiler_compatibility_guards() as guarded_emitters:
            model.compile(
                output_path=str(output_dir),
                batch_size=1,
                tessellate_parameters=tessellation_parameters(model),
                require_single_mla=True,
            )
        if guarded_emitters not in (set(), EXPECTED_GUARDED_EMITTERS):
            raise RuntimeError(f"unexpected INT16-exact Concat guard coverage: {sorted(guarded_emitters)}")

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {archives}")
    archive = archives[0]
    report = {
        "schema_version": 1,
        "model_id": "depthart_base",
        "variant_id": variant_id,
        "source": source_report,
        "quantized_source": {"sha256": quantized_digest, "size_bytes": quantized_size},
        "compiler_compatibility": {
            "guarded_int8_concat_emitters": sorted(guarded_emitters),
            "deep_graph_pickle_guard": True,
            "installed_compiler_sources_modified": False,
        },
        "mpk": audit_mpk(archive, expected_elf_sha256=parameters["expected_elf_sha256"]),
    }
    (output_dir / "depthart-base-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_variant(source, args.output_dir.expanduser().resolve(), args.variant)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
