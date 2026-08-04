from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "superpoint"


def load_module(name: str) -> ModuleType:
    path = MODEL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"superpoint_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR))
    return module


def superpoint_source(path: Path) -> Path:
    layers = [
        ("conv1a", 1, 64, 3),
        ("conv1b", 64, 64, 3),
        ("conv2a", 64, 64, 3),
        ("conv2b", 64, 64, 3),
        ("conv3a", 64, 128, 3),
        ("conv3b", 128, 128, 3),
        ("conv4a", 128, 128, 3),
        ("conv4b", 128, 128, 3),
        ("convPa", 128, 256, 3),
        ("convPb", 256, 65, 1),
        ("convDa", 128, 256, 3),
        ("convDb", 256, 256, 1),
    ]
    initializers = []
    for name, inputs, outputs, kernel in layers:
        weights = np.zeros((outputs, inputs, kernel, kernel), dtype=np.float32)
        bias = np.zeros((outputs,), dtype=np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(weights, f"{name}.weight"),
                numpy_helper.from_array(bias, f"{name}.bias"),
            ]
        )

    nodes = []
    current = "input"
    blocks = [
        ("conv1a", "/relu/Relu", False),
        ("conv1b", "/relu_1/Relu", True),
        ("conv2a", "/relu_2/Relu", False),
        ("conv2b", "/relu_3/Relu", True),
        ("conv3a", "/relu_4/Relu", False),
        ("conv3b", "/relu_5/Relu", True),
        ("conv4a", "/relu_6/Relu", False),
        ("conv4b", "/relu_7/Relu", False),
    ]
    pool_index = 0
    for name, relu_name, pool in blocks:
        conv_output = f"/{name}/Conv_output_0"
        nodes.append(
            helper.make_node(
                "Conv",
                [current, f"{name}.weight", f"{name}.bias"],
                [conv_output],
                name=f"/{name}/Conv",
                pads=[1, 1, 1, 1],
            )
        )
        relu_output = f"{relu_name}_output_0"
        nodes.append(helper.make_node("Relu", [conv_output], [relu_output], name=relu_name))
        current = relu_output
        if pool:
            pool_name = "/pool/MaxPool" if pool_index == 0 else f"/pool_{pool_index}/MaxPool"
            pool_output = f"{pool_name}_output_0"
            nodes.append(
                helper.make_node(
                    "MaxPool", [current], [pool_output], name=pool_name, kernel_shape=[2, 2], strides=[2, 2]
                )
            )
            current = pool_output
            pool_index += 1

    trunk = current
    nodes.extend(
        [
            helper.make_node(
                "Conv",
                [trunk, "convPa.weight", "convPa.bias"],
                ["/convPa/Conv_output_0"],
                name="/convPa/Conv",
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", ["/convPa/Conv_output_0"], ["/relu_8/Relu_output_0"], name="/relu_8/Relu"),
            helper.make_node(
                "Conv",
                ["/relu_8/Relu_output_0", "convPb.weight", "convPb.bias"],
                ["semi"],
                name="/convPb/Conv",
            ),
            helper.make_node(
                "Conv",
                [trunk, "convDa.weight", "convDa.bias"],
                ["/convDa/Conv_output_0"],
                name="/convDa/Conv",
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", ["/convDa/Conv_output_0"], ["/relu_9/Relu_output_0"], name="/relu_9/Relu"),
            helper.make_node(
                "Conv",
                ["/relu_9/Relu_output_0", "convDb.weight", "convDb.bias"],
                ["/convDb/Conv_output_0"],
                name="/convDb/Conv",
            ),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "source",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 480, 640])],
        [
            helper.make_tensor_value_info("semi", TensorProto.FLOAT, [1, 65, 60, 80]),
            helper.make_tensor_value_info("/convDb/Conv_output_0", TensorProto.FLOAT, [1, 256, 60, 80]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


def test_surgery_produces_static_dense_heads(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    output = surgery.prepare_model(superpoint_source(tmp_path / "source.onnx"), tmp_path / "dense.onnx")
    model = onnx.load(output, load_external_data=False)
    surgery.validate_dense_core(model)
    assert model.graph.input[0].name == "image"
    assert [value.name for value in model.graph.output] == [name for name, _ in surgery.HEADS.values()]
    assert len(model.graph.node) == 25


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype", "input_elem_type", "output_elem_type", "cast_count"),
    [
        ("bfloat16", "bfloat16", TensorProto.BFLOAT16, TensorProto.BFLOAT16, 3),
        ("bfloat16", "float32", TensorProto.BFLOAT16, TensorProto.FLOAT, 1),
    ],
)
def test_explicit_float_boundary_is_valid_onnx(
    tmp_path: Path,
    input_dtype: str,
    output_dtype: str,
    input_elem_type: int,
    output_elem_type: int,
    cast_count: int,
) -> None:
    surgery = load_module("surgery")
    compile_module = load_module("compile")
    dense = surgery.prepare_model(superpoint_source(tmp_path / "source.onnx"), tmp_path / "dense.onnx")
    output = compile_module._explicit_float_boundary(
        dense,
        tmp_path / "boundary.onnx",
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    model = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(model)
    assert model.graph.input[0].type.tensor_type.elem_type == input_elem_type
    assert {value.type.tensor_type.elem_type for value in model.graph.output} == {output_elem_type}
    assert sum(node.op_type == "Cast" for node in model.graph.node) == cast_count


def test_calibration_manifest_matches_the_pinned_snapshot() -> None:
    calibration = load_module("calibration")
    manifest = calibration.validate_manifest(MODEL_DIR / "calibration_manifest.json")
    assert manifest["sources"] == {"coco": 80, "hpatches": 32, "tum": 16}
    assert len(manifest["records"]) == 128


def test_grayscale_conversion_uses_pinned_fixed_point_formula() -> None:
    calibration = load_module("calibration")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]
    gray = calibration.bgr_to_gray(frame)
    assert gray.dtype == np.uint32
    assert gray[0, 0] == (30 * 4899 + 20 * 9617 + 10 * 1868 + 8192) >> 14


def plugin(
    name: str,
    processor: str,
    *,
    input_dtype: str = "",
    layout: str = "",
    kernel: str = "",
    frame_type: str = "",
    tensor_types: list[str] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if frame_type:
        params["frame_type"] = frame_type
    if tensor_types:
        params["tensor_types"] = tensor_types
    return {
        "name": name,
        "sequence": 1,
        "processor": processor,
        "config_params": {"kernel": kernel, "params": params},
        "input_nodes": [
            {"dtype": input_dtype, "size": 1 * 480 * 640 * {"float32": 4, "bfloat16": 2}[input_dtype]}
        ]
        if input_dtype
        else [],
        "output_nodes": [{"layout": layout}, {"layout": layout}] if layout else [],
    }


def write_mpk(path: Path, plugins: list[dict[str, Any]], *, shared_object: bool = False) -> Path:
    public_inputs = plugins[0].get("input_nodes", []) if plugins else []
    manifest = json.dumps({"input_nodes": public_inputs, "plugins": plugins}).encode()
    with tarfile.open(path, "w:gz") as stream:
        for name, payload in (("model_mpk.json", manifest), ("model.elf", b"ELF")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
        if shared_object:
            info = tarfile.TarInfo("libbad.so")
            info.size = 2
            stream.addfile(info, io.BytesIO(b"so"))
    return path


@pytest.mark.parametrize(
    ("variant", "plugins"),
    [
        (
            "modalix_int8_tessellation_ev74",
            [
                plugin("quantize", "EV74", input_dtype="float32"),
                plugin("score_tess", "EV74", frame_type="int8", kernel="tessellation_transform"),
                plugin("MLA_0", "MLA", layout="tessellated"),
                plugin("unpack", "EV74", kernel="unpack_transform"),
                plugin("score_detess", "EV74", frame_type="int8", kernel="detessellation_transform"),
                plugin("desc_detess", "EV74", frame_type="int8", kernel="detessellation_transform"),
                plugin("publish", "EV74", kernel="pass_through"),
            ],
        ),
        (
            "modalix_int8_tessellation_mla",
            [
                plugin("quantize", "EV74", input_dtype="float32"),
                plugin("MLA_0", "MLA", layout="normal"),
                plugin("publish", "EV74", kernel="pass_through"),
            ],
        ),
        (
            "modalix_bf16_tessellation_ev74",
            [
                plugin("MLA_0", "MLA", input_dtype="bfloat16", layout="tessellated"),
                plugin("score_tess", "EV74", frame_type="bfloat16", kernel="tessellation_transform"),
                plugin("unpack", "EV74", kernel="unpack_transform"),
                plugin("score_detess", "EV74", frame_type="bfloat16", kernel="detessellation_transform"),
                plugin("desc_detess", "EV74", frame_type="bfloat16", kernel="detessellation_transform"),
            ],
        ),
        (
            "modalix_bf16_tessellation_mla",
            [plugin("MLA_0", "MLA", input_dtype="bfloat16", layout="normal")],
        ),
        (
            "modalix_fp32_dense",
            [
                plugin("MLA_0", "MLA", input_dtype="bfloat16", layout="normal"),
                plugin("unpack", "EV74", tensor_types=["float32", "float32"]),
                plugin("publish", "EV74", kernel="pass_through"),
            ],
        ),
    ],
)
def test_mpk_audit_accepts_each_qualified_boundary(tmp_path: Path, variant: str, plugins: list[dict[str, Any]]) -> None:
    compile_module = load_module("compile")
    report = compile_module.audit_mpk(write_mpk(tmp_path / f"{variant}_mpk.tar.gz", plugins), variant)
    assert report["mla_plugin_count"] == 1
    assert report["shared_object_count"] == 0


def test_mpk_audit_rejects_shared_objects(tmp_path: Path) -> None:
    compile_module = load_module("compile")
    plugins = [
        plugin("MLA_0", "MLA", input_dtype="bfloat16", layout="normal"),
        plugin("unpack", "EV74", tensor_types=["float32", "float32"]),
        plugin("publish", "EV74", kernel="pass_through"),
    ]
    archive = write_mpk(tmp_path / "bad_mpk.tar.gz", plugins, shared_object=True)
    with pytest.raises(RuntimeError, match="invalid SuperPoint MPK packaging"):
        compile_module.audit_mpk(archive, "modalix_fp32_dense")


def test_mpk_audit_accepts_ev74_transforms_when_mla_layout_is_not_published(tmp_path: Path) -> None:
    compile_module = load_module("compile")
    plugins = [
        plugin("quantize", "EV74", input_dtype="float32"),
        plugin("tess", "EV74", frame_type="int8", kernel="tessellation_transform"),
        plugin("MLA_0", "MLA"),
        plugin("unpack", "EV74", kernel="unpack_transform"),
        plugin("score_detess", "EV74", frame_type="int8", kernel="detessellation_transform"),
        plugin("desc_detess", "EV74", frame_type="int8", kernel="detessellation_transform"),
        plugin("publish", "EV74", kernel="pass_through"),
    ]
    archive = write_mpk(tmp_path / "no-layout_mpk.tar.gz", plugins)
    report = compile_module.audit_mpk(archive, "modalix_int8_tessellation_ev74")
    assert report["mla_output_layouts"] == []


def test_mpk_audit_infers_the_public_input_abi_when_dtype_is_not_published(tmp_path: Path) -> None:
    compile_module = load_module("compile")
    quantize = plugin("quantize", "EV74", input_dtype="float32")
    quantize["input_nodes"][0].pop("dtype")
    plugins = [
        quantize,
        plugin("MLA_0", "MLA"),
        plugin("unpack", "EV74", kernel="unpack_transform"),
        plugin("publish", "EV74", kernel="pass_through"),
    ]
    archive = write_mpk(tmp_path / "size-only-input_mpk.tar.gz", plugins)
    report = compile_module.audit_mpk(archive, "modalix_int8_tessellation_mla")
    assert report["public_input_bytes"] == 1 * 480 * 640 * 4


def test_mpk_audit_rejects_mixed_mla_output_layouts(tmp_path: Path) -> None:
    compile_module = load_module("compile")
    mla = plugin("MLA_0", "MLA", input_dtype="float32", layout="normal")
    mla["output_nodes"][1]["layout"] = "tessellated"
    plugins = [mla, plugin("publish", "EV74", kernel="pass_through")]
    archive = write_mpk(tmp_path / "mixed-layout_mpk.tar.gz", plugins)
    with pytest.raises(RuntimeError, match="MLA layouts=.*normal.*tessellated"):
        compile_module.audit_mpk(archive, "modalix_int8_tessellation_mla")


def test_compile_matrix_contains_all_superpoint_boundaries() -> None:
    compile_module = load_module("compile")
    expected = {
        "modalix_int8_tessellation_ev74": ("float32", "int8", "ev74", False, False),
        "modalix_int8_tessellation_mla": ("float32", "int8", "mla", True, True),
        "modalix_bf16_tessellation_ev74": ("bfloat16", "bfloat16", "ev74", False, False),
        "modalix_bf16_tessellation_mla": ("bfloat16", "bfloat16", "mla", True, True),
        "modalix_fp32_dense": ("bfloat16", "float32", "mla", True, True),
    }
    for variant, contract in expected.items():
        parameters = compile_module.variant_config(variant)["parameters"]
        assert (
            parameters["public_input_dtype"],
            parameters["public_output_dtype"],
            parameters["tessellation_owner"],
            parameters["input_tesselation_mla"],
            parameters["output_tesselation_mla"],
        ) == contract
