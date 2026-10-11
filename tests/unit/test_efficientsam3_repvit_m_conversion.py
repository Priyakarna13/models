"""Small checks for conversion failures that a synthetic smoke input can miss."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "models/efficientsam3_repvit_m/scripts"
SPEC = importlib.util.spec_from_file_location("efficientsam3_repvit_m_convert", SCRIPTS_DIR / "convert.py")
converter = importlib.util.module_from_spec(SPEC)
# Sibling scripts are imported by name, so another model's calibration or surgery must not stand in for ours.
shadowed = {dependency: sys.modules.pop(dependency, None) for dependency in ("calibration", "surgery")}
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    SPEC.loader.exec_module(converter)
finally:
    sys.path.remove(str(SCRIPTS_DIR))
    for dependency, previous in shadowed.items():
        if previous is None:
            sys.modules.pop(dependency, None)
        else:
            sys.modules[dependency] = previous


def _outputs():
    rng = np.random.default_rng(0)
    scores = np.zeros(converter.QUERIES)
    scores[0] = 0.9
    return {
        "text_features": rng.standard_normal(converter.INTERFACE["text_features"][1:]),
        "scores": scores,
        "boxes_px": rng.uniform(0, 1008, (converter.QUERIES, 4)),
        "mask_logits": rng.standard_normal((converter.QUERIES, 8, 8)),
    }


@pytest.mark.parametrize("case", ["valid", "padding", "confident_box", "score", "shape", "nan"])
def test_output_agreement(case):
    expected = _outputs()
    actual = {name: value.copy() for name, value in expected.items()}
    if case == "valid":
        actual["boxes_px"][0] += 0.05
    elif case == "padding":
        # Padding positions carry no prompt and are masked out of the image model.
        actual["text_features"][4:] += 1
    elif case == "confident_box":
        actual["boxes_px"][0, 0] += 1
    elif case == "score":
        actual["scores"][3] += 0.01
    elif case == "shape":
        actual["scores"] = actual["scores"][:-1]
    else:
        actual["mask_logits"][0, 0, 0] = np.nan
    if case in {"valid", "padding"}:
        converter.compare_outputs(expected, actual, valid=4)
    else:
        with pytest.raises(ValueError):
            converter.compare_outputs(expected, actual, valid=4)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case):
    helper = onnx.helper
    values = {
        name: helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    }
    inputs = [values[name] for name in converter.INPUTS]
    outputs = [values[name] for name in converter.OUTPUTS]
    if case == "order":
        inputs.reverse()
    elif case == "dtype":
        outputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        inputs[0].type.tensor_type.shape.dim[1].dim_value = 32
    nodes = [helper.make_node("Constant", [], [value.name], value_float=0.0) for value in outputs]
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_file_set():
    model = yaml.safe_load((ROOT / "models/efficientsam3_repvit_m/model.yaml").read_text(encoding="utf-8"))
    source = model["conversion"]["default"]["source"]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="sam3 wheel"):
        converter.validate_source({**source, "files": source["files"][:1]})


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("features.0.0.c.weight", "stem.conv1.c.weight"),
        ("features.0.2.bn.bias", "stem.conv2.bn.bias"),
        ("features.1.token_mixer.0.conv.c.weight", "stages.0.blocks.0.token_mixer.conv.c.weight"),
        ("features.2.channel_mixer.m.2.bn.weight", "stages.0.blocks.1.channel_mixer.conv2.bn.weight"),
        ("features.3.token_mixer.1.fc1.weight", "stages.1.downsample.pre_block.se.fc1.weight"),
        ("features.4.token_mixer.0.c.weight", "stages.1.downsample.spatial_downsample.c.weight"),
        ("features.4.channel_mixer.m.0.c.weight", "stages.1.downsample.ffn.conv1.c.weight"),
        ("features.20.channel_mixer.m.0.c.weight", "stages.2.blocks.11.channel_mixer.conv1.c.weight"),
        ("features.21.channel_mixer.m.0.c.weight", "stages.3.downsample.pre_block.channel_mixer.conv1.c.weight"),
        ("features.24.token_mixer.0.conv.c.weight", "stages.3.blocks.1.token_mixer.conv.c.weight"),
    ],
)
def test_trunk_keys_map_onto_timm_repvit(key, expected):
    assert converter.trunk_key(f"trunk.model.backbone.model.{key}") == expected


def test_trunk_key_rejects_unknown_layers():
    with pytest.raises(KeyError, match="Unmapped"):
        converter.trunk_key("trunk.features.1.unknown.weight")


def test_package_unpacking_keeps_only_sam3(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    wheel = tmp_path / "sam3.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sam3/__init__.py", "")
        archive.writestr("sam3-0.1.4.dist-info/METADATA", "")
    converter.unpack_package(wheel, tmp_path / "out")

    assert (tmp_path / "out/sam3/__init__.py").is_file()
    assert not (tmp_path / "out/sam3-0.1.4.dist-info").exists()
    assert sys.path[0] == str(tmp_path / "out")
