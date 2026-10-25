import json
import math

from model_registry.artifact import verify_runtime_files


def test_complete_runtime_payload(compiled):
    import onnx

    component = compiled / "model"
    assert len(list(component.glob("*_mpk.tar.gz"))) == 1
    names = {item["path"] for item in verify_runtime_files(component)}
    assert {"predictor_cpu_int8.onnx", "joiner_cpu_int8.onnx", "transcribe.py", "frontend.npz"} <= names
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
