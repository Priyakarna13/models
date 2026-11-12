import json
import math
import sys
from pathlib import Path

import pytest
from model_registry.runtime_files import read_runtime_files


def test_complete_runtime_payload(compiled):
    import onnx

    validation = json.loads((compiled / "source-validation.json").read_text())
    assert validation["checked_output"] == "encoded"
    assert math.isfinite(validation["max_abs_error"])
    component = compiled / "model"
    assert len(list(component.glob("*_mpk.tar.gz"))) == 1
    names = {item["path"] for item in read_runtime_files(compiled)}
    assert {
        "model/predictor_cpu_int8.onnx",
        "model/joiner_cpu_int8.onnx",
        "model/transcribe.py",
        "model/frontend.npz",
    } <= names
    for name in ("predictor", "joiner"):
        model = onnx.load(component / (name + "_cpu_int8.onnx"))
        assert any(tensor.data_type == onnx.TensorProto.INT8 for tensor in model.graph.initializer)
        assert not {"MatMul", "Gemm", "LSTM"}.intersection(node.op_type for node in model.graph.node)
        assert not any(
            tensor.data_type == onnx.TensorProto.FLOAT and len(tensor.dims) > 1 and math.prod(tensor.dims) >= 640 * 640
            for tensor in model.graph.initializer
        )
        assert any(node.op_type in {"DynamicQuantizeLSTM", "MatMulInteger"} for node in model.graph.node)
    config = json.loads((component / "runtime_config.json").read_text())
    assert config["outputs"][0]["name"] == "encoded"


def test_predictor_runs_only_after_nonblank_tokens(load_script):
    import numpy as np

    runtime = load_script("transcribe")

    class Predictor:
        def __init__(self):
            self.tokens = []

        def run(self, _, inputs):
            self.tokens.append(int(inputs["token"].item()))
            return np.zeros((1, 1, 640), np.float32), inputs["hidden"], inputs["cell"]

    class Joiner:
        def __init__(self):
            self.decisions = iter([0, 1, 3, 2, 3])

        def run(self, _, inputs):
            output = np.zeros((1, 1, 4), np.float32)
            output[..., next(self.decisions)] = 1
            return [output]

    predictor = Predictor()
    decoder = runtime.Decoder(predictor, Joiner(), dict(kind="nemotron", blank_token_id=3, max_symbols_per_step=10))
    decoder.step(np.zeros((1, 2, 640), np.float32))
    assert decoder.tokens == [0, 1, 2]
    assert predictor.tokens == [3, 0, 1, 2]


def test_tdt_duration_carries_to_the_next_chunk(load_script):
    import numpy as np

    runtime = load_script("transcribe")

    class Predictor:
        def run(self, _, inputs):
            return np.zeros((1, 1, 640), np.float32), inputs["hidden"], inputs["cell"]

    class Joiner:
        def __init__(self):
            self.calls = 0

        def run(self, _, inputs):
            self.calls += 1
            return [np.array([[[1.0, 0.0]]], np.float32), np.array([[[0.0, 0.0, 0.0, 0.0, 1.0]]], np.float32)]

    joiner = Joiner()
    decoder = runtime.Decoder(Predictor(), joiner, dict(kind="parakeet", blank_token_id=1, max_symbols_per_step=10))
    decoder.step(np.zeros((1, 2, 640), np.float32))
    assert decoder.skip == 2
    decoder.step(np.zeros((1, 2, 640), np.float32))
    assert decoder.skip == 0 and joiner.calls == 1
    decoder.step(np.zeros((1, 2, 640), np.float32))
    assert decoder.tokens == [0, 0] and joiner.calls == 2


def test_compile_manifest_uses_shared_runtime_files_contract(tmp_path, load_script):
    import pytest
    from model_registry.artifact import validate_compile_outputs
    from model_registry.errors import ValidationError

    recipe = load_script("compile")
    component = tmp_path / "model"
    component.mkdir()
    archive = component / "encoder_mpk.tar.gz"
    archive.write_bytes(b"encoder")
    for filename in recipe.RUNTIME_FILES:
        (component / filename).write_bytes(filename.encode())
    recipe.write_compile_outputs(tmp_path, archive, "asr", "w8a16")

    def validate():
        return validate_compile_outputs(
            tmp_path / "compile-outputs.json",
            tmp_path,
            model_id="asr",
            build_profile_id="w8a16",
            compile_invocation_id="w8a16",
            component_ids=["model"],
        )

    assert len(validate()["runtime_files"]) == len(recipe.RUNTIME_FILES)
    (component / "predictor_cpu_int8.onnx").write_bytes(b"changed")
    with pytest.raises(ValidationError, match="SHA-256"):
        validate()


def test_prepare_runs_surgery_before_partitioning(tmp_path, load_script, monkeypatch):
    import onnx
    from onnx import TensorProto, helper

    recipe = load_script("compile")

    def value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

    graph = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["audio"], ["encoded"])],
            "encoder",
            [value("audio")],
            [value("encoded")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    assets = {"tokenizer.json": {"model": {}}, "processor_config.json": {"sample_rate": 16000}}
    helper.set_model_props(
        graph,
        {
            "sima.asr": json.dumps(
                {
                    "encoder_outputs": ["encoded"],
                    "predictor_outputs": [],
                    "joiner_outputs": [],
                    "assets": assets,
                }
            )
        },
    )
    from types import SimpleNamespace

    source = tmp_path / "weights.pth"
    source.write_bytes(b"checkpoint")

    def export_model(weights, output):
        assert weights == source
        output.mkdir()
        path = output / "model.onnx"
        onnx.save(graph, path)
        return path

    monkeypatch.setitem(sys.modules, "surgery", SimpleNamespace(export_model=export_model))
    component, work = tmp_path / "component", tmp_path / "work"
    component.mkdir()
    work.mkdir()
    monkeypatch.setattr(recipe, "cpu_model", lambda *args: None)
    recipe.prepare_model(source, work, component)
    for filename, document in assets.items():
        assert json.loads((component / filename).read_text()) == document
    assert "assets" not in json.loads((work / "config.json").read_text())
    onnx.checker.check_model(str(work / "encoder.onnx"))


def test_partition_retains_only_used_inputs_and_preserves_metadata(load_script):
    import pytest

    partition = load_script("onnx_export").partition
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    def value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [helper.make_node("Identity", ["audio"], ["encoded"]), helper.make_node("Identity", ["token"], ["prediction"])],
        "regions",
        [value("audio"), value("token")],
        [value("encoded"), value("prediction")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    helper.set_model_props(model, {"sima.asr": "metadata"})
    encoder = partition(model, ["encoded"])
    assert [value.name for value in encoder.graph.input] == ["audio"]
    assert [value.name for value in encoder.graph.output] == ["encoded"]
    assert encoder.metadata_props == model.metadata_props
    onnx.checker.check_model(encoder)
    with pytest.raises(ValueError, match="declared"):
        partition(model, ["missing"])


@pytest.mark.parametrize("install_fails", [False, True])
def test_export_dependencies_are_installed_only_in_the_export_environment(
    tmp_path, load_script, monkeypatch, install_fails
):
    from types import SimpleNamespace

    recipe = load_script("compile")
    calls = []

    def create(directory):
        path = recipe.sysconfig.get_path("purelib", vars={"base": str(directory), "platbase": str(directory)})
        Path(path).mkdir(parents=True)

    def run(command, **kwargs):
        assert kwargs == {"check": True}
        calls.append(command)
        assert command[1] == "-I"
        assert str(tmp_path) in command[0] and command[0] != sys.executable
        if install_fails:
            raise recipe.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(recipe.venv, "EnvBuilder", lambda **kwargs: SimpleNamespace(create=create))
    monkeypatch.setattr(recipe.subprocess, "run", run)
    if install_fails:
        with pytest.raises(recipe.subprocess.CalledProcessError):
            recipe.prepare_in_environment(tmp_path / "weights.pth", tmp_path, tmp_path / "model")
        assert len(calls) == 1
    else:
        recipe.prepare_in_environment(tmp_path / "weights.pth", tmp_path, tmp_path / "model")
        install, prepare = calls
        assert install[2:7] == ["-m", "pip", "--isolated", "install", "--disable-pip-version-check"]
        assert "transformers==5.16.1" in install and "torch==2.8.0" in install
        assert install[7:] == list(recipe.EXPORT_REQUIREMENTS)
        assert prepare[0] == install[0] and prepare[2] == "-c"
        assert "AutoModelForRNNT" in prepare[3] and "AutoModelForTDT" in prepare[3]
        assert "prepare_model" in prepare[3]
    assert not list(tmp_path.glob("export-env-*"))
