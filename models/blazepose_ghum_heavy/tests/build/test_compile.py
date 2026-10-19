from __future__ import annotations

import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "blazepose_ghum_heavy"


def channel_pad_graph(surgery: ModuleType) -> onnx.ModelProto:
    input_shape = (1, 2, 3, 3)
    nodes = []
    initializers = []
    outputs = []
    for index in range(surgery.CHANNEL_PADDING_COUNT):
        pads_name = f"pads_{index}"
        output_name = f"output_{index}"
        initializers.append(numpy_helper.from_array(np.asarray([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.int64), pads_name))
        nodes.append(
            helper.make_node(
                "Pad",
                ["input", pads_name],
                [output_name],
                name=f"branch_{index}/channel_padding",
            )
        )
        outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, (1, 4, 3, 3)))
    graph = helper.make_graph(
        nodes,
        "channel_pad_fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        outputs,
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])


def test_surgery_replaces_each_channel_pad_with_identity_convolution(
    load_script: Callable[[str], ModuleType],
) -> None:
    surgery = load_script("surgery")
    model = channel_pad_graph(surgery)
    replaced = surgery._replace_channel_padding(model)
    assert len(replaced) == surgery.CHANNEL_PADDING_COUNT
    assert not any(node.op_type == "Pad" for node in model.graph.node)
    assert all(node.op_type == "Conv" for node in model.graph.node)
    weights = [numpy_helper.to_array(item) for item in model.graph.initializer]
    assert all(weight.shape == (4, 2, 1, 1) for weight in weights)
    assert all(np.array_equal(weight[:2, :, 0, 0], np.eye(2, dtype=np.float32)) for weight in weights)
    assert all(np.count_nonzero(weight[2:]) == 0 for weight in weights)


def test_conversion_gate_accepts_qualified_deltas_and_rejects_regression(
    load_script: Callable[[str], ModuleType],
) -> None:
    conversion = load_script("validate_conversion")
    reference = {
        name: np.ones(shape, dtype=np.float32)
        for name, shape in (("Identity", (1, 195)), ("Identity_1", (1, 1)), ("Identity_4", (1, 117)))
    }
    candidate = {name: value + np.float32(0.0001) for name, value in reference.items()}
    assert conversion.compare_outputs(reference, candidate)["passed"] is True
    candidate["Identity"][0, 0] = 2.0
    with pytest.raises(ValueError, match="exceeds tolerance"):
        conversion.compare_outputs(reference, candidate)


def test_model_declares_exact_apps_output_contract() -> None:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    component = metadata["components"][0]
    outputs = component["interface"]["entrypoints"][0]["outputs"]
    assert [(item["name"], item["shape"]) for item in outputs] == [
        ("Identity", [1, 195]),
        ("Identity_1", [1, 1]),
        ("Identity_4", [1, 117]),
    ]
    descriptions = {item["name"]: item["description"] for item in outputs}
    assert "33-34 are MediaPipe ROI auxiliaries" in descriptions["Identity"]
    assert "35-38 are ignored" in descriptions["Identity"]
    assert "exposes only 0-32 and ignores 33-38" in descriptions["Identity_4"]
    profile = metadata["build_profiles"][0]
    assert profile["id"] == "w16a16"
    assert profile["components"][0]["parameters"]["precision"] == "w16a16"


def test_documentation_distinguishes_public_and_raw_landmark_records() -> None:
    readme = " ".join((MODEL_DIR / "README.md").read_text(encoding="utf-8").split())
    assert "only `Identity` records 33 and 34 as auxiliary" in readme
    assert "records 35–38 are not part of its postprocessed landmark outputs" in readme
    assert "discards `Identity_4` records 33–38" in readme
    assert "The remaining six are auxiliary ROI-tracking landmarks" not in readme


def test_compile_cli_accepts_current_invocation_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "pose_landmark_heavy.onnx"
    source.write_bytes(b"fixture")
    output_root = tmp_path / "output"
    calls: list[tuple[Path, Path, str]] = []

    def compile_model(model: Path, output: Path, profile: str) -> Path:
        calls.append((model, output, profile))
        return output / recipe.COMPONENT_ID / "fixture_mpk.tar.gz"

    monkeypatch.setattr(recipe, "compile_model", compile_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(source),
            "--output-dir",
            str(output_root),
            "--build-profile",
            "w16a16",
            "--compile-invocation",
            "model",
        ],
    )

    assert recipe.main() == 0
    assert calls == [(source.resolve(), output_root.resolve(), "w16a16")]


def test_mpk_audit_enforces_public_abi_and_mla_placement(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    manifest = json.dumps(
        {
            "model_sdk_version": recipe.COMPILER_VERSION,
            "input_nodes": [
                {
                    "name": recipe.INPUT_NAME,
                    "size": recipe.PUBLIC_INPUT_SIZE,
                    "input_range": [0.0, 1.0],
                }
            ],
            "plugins": [
                {"sequence": 1, "processor": "MLA", "output_nodes": [{"name": "mla", "size": 10016}]},
                {
                    "sequence": 2,
                    "processor": "EV74",
                    "config_params": {"kernel": "pass_through"},
                    "output_nodes": [{"size": size} for size in recipe.PUBLIC_OUTPUT_SIZES],
                },
            ],
        }
    ).encode()
    archive = tmp_path / "blazepose_mpk.tar.gz"
    preprocessor = json.dumps(
        {
            "input_channels": 3,
            "input_img_type": "RGB",
            "normalize": True,
            "output_channels": 3,
            "output_height": 256,
            "output_img_type": "RGB",
            "output_width": 256,
        }
    ).encode()
    with tarfile.open(archive, "w:gz") as stream:
        for name, payload in (
            ("blazepose_mpk.json", manifest),
            ("0_preproc.json", preprocessor),
            ("blazepose.elf", b"ELF"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    report = recipe.audit_mpk(archive)
    assert report["public_output_sizes"] == [780, 4, 468]
    assert report["mla_plugin_count"] == 1


def test_mpk_audit_accepts_model_sdk3_manifest_boundary(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    shape = list(recipe.INPUT_SHAPE)
    manifest = json.dumps(
        {
            "model_sdk_version": recipe.COMPILER_VERSION,
            "input_nodes": [
                {
                    "name": recipe.INPUT_NAME,
                    "size": recipe.PUBLIC_INPUT_SIZE,
                    "input_range": [0.0, 1.0],
                }
            ],
            "plugins": [
                {
                    "sequence": 1,
                    "processor": "EV74",
                    "config_params": {
                        "kernel": "cast_transform",
                        "params": {
                            "in_dtype": "float32",
                            "out_dtype": "bfloat16",
                            "input_shapes": [shape],
                            "output_shapes": [shape],
                        },
                    },
                    "output_nodes": [{"size": recipe.PUBLIC_INPUT_SIZE // 2}],
                },
                {
                    "sequence": 2,
                    "processor": "MLA",
                    "output_nodes": [{"name": "mla", "size": 10016}],
                },
                {
                    "sequence": 3,
                    "processor": "EV74",
                    "config_params": {"kernel": "pass_through"},
                    "output_nodes": [{"size": size} for size in recipe.PUBLIC_OUTPUT_SIZES],
                },
            ],
        }
    ).encode()
    archive = tmp_path / "blazepose_sdk3_mpk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, payload in (
            ("blazepose_mpk.json", manifest),
            ("blazepose.elf", b"ELF"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))

    report = recipe.audit_mpk(archive)

    assert report["preprocessor"] == {
        "input_dtype": "float32",
        "input_shape": shape,
        "normalize": True,
        "output_dtype": "bfloat16",
        "output_shape": shape,
        "source": "manifest_boundary",
    }
