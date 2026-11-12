import json
import math
import sys
from pathlib import Path

import pytest
from model_registry.runtime_files import read_runtime_files


@pytest.mark.parametrize("kind,expected", [
    ("nemotron", [7880, 16944, 25904]),
    ("parakeet", [64000, 96000, 128000]),
])
def test_first_text_waits_for_real_input_and_skips_empty_chunks(load_script, monkeypatch, kind, expected):
    import numpy as np

    benchmark = load_script("performance")

    class Clock:
        now = 0.0

        def perf_counter(self):
            return self.now

        def sleep(self, seconds):
            assert seconds > 0
            self.now += seconds

    clock = Clock()
    arrivals, closed = [], []

    class Transcriber:
        config = {"kind": kind}

        def transcribe(self, audio, language):
            assert language == "auto"
            try:
                for text in ("", "  ", "Hello"):
                    arrivals.append(clock.now)
                    clock.now += 0.1
                    yield text
            finally:
                closed.append(True)

    monkeypatch.setattr(benchmark, "time", clock)
    audio = np.zeros(480000, np.float32)
    elapsed = benchmark.time_to_first_text(Transcriber(), audio)
    assert arrivals == pytest.approx([samples / 16000 for samples in expected])
    assert elapsed == pytest.approx(expected[-1] / 16000 + 0.1)
    assert closed == [True]
    assert benchmark.available_samples(kind, 100, len(audio)) == len(audio)


def test_first_text_does_not_sleep_again_when_processing_falls_behind(load_script, monkeypatch):
    import numpy as np

    benchmark = load_script("performance")
    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(benchmark.time, "sleep", sleep)

    class Transcriber:
        config = {"kind": "parakeet"}

        def transcribe(self, *_):
            clock[0] += 3
            yield ""
            clock[0] += 3
            yield "Hello"

    assert benchmark.time_to_first_text(Transcriber(), np.zeros(480000)) == pytest.approx(10)
    assert sleeps == [4]


@pytest.mark.parametrize("kind", ["nemotron", "parakeet"])
def test_pacing_matches_runtime_audio_windows(load_script, monkeypatch, kind):
    from types import SimpleNamespace

    import numpy as np

    benchmark = runtime = load_script("performance")
    seen = []

    def mel(raw, *_, **kwargs):
        seen.append(int(raw.max()) + 1)
        frames = (49 if kwargs.get("center") else 56) if kind == "nemotron" else len(raw) // 160
        return np.zeros((1, frames, 128), np.float32)

    monkeypatch.setattr(runtime, "log_mel", mel)
    monkeypatch.setattr(runtime, "Decoder", lambda *_: SimpleNamespace(tokens=[1], step=lambda _: None))
    transcriber = runtime.Transcriber.__new__(runtime.Transcriber)
    transcriber.config = {"kind": kind}
    transcriber.zero_states = {}
    transcriber.processor = {"prompt_dictionary": {"auto": 0}}
    transcriber.filters = transcriber.window = transcriber.predictor = transcriber.joiner = None
    transcriber.tensor = lambda value: value
    transcriber.tokenizer = SimpleNamespace(decode=lambda *_, **__: "Hello")
    outputs = {f"b{layer:02d}/next_{name}": None for layer in range(24) for name in ("keys", "values", "conv")}
    outputs.update({"sub/next_sub1": None, "sub/next_sub2": None})
    transcriber.encode = lambda _: (np.zeros((1, 175, 640), np.float32), outputs)
    audio = np.arange(480000, dtype=np.float32)
    list(transcriber.transcribe(audio))
    assert seen == [benchmark.available_samples(kind, index, len(audio)) for index in range(len(seen))]


def test_unpaced_recording_rejects_no_visible_text(load_script):
    import numpy as np

    benchmark = load_script("performance")

    class Transcriber:
        def transcribe(self, *_):
            yield ""
            yield "  "

    with pytest.raises(RuntimeError, match="No visible text"):
        benchmark.transcribe_recording(Transcriber(), np.zeros(480000))
    with pytest.raises(ValueError, match="Unknown ASR kind"):
        benchmark.available_samples("other", 0, 480000)


def test_audio_validation_and_performance_fingerprint_boundary(load_script, model_dir, tmp_path):
    import wave

    from model_registry.fingerprints import _compile_input_files

    benchmark = load_script("performance")
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(30 * 16000 * 2))
    assert len(benchmark.load_audio(path)) == 480000
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(16000 * 2))
    with pytest.raises(ValueError, match="at least 30 seconds"):
        benchmark.load_audio(path)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(16000 * 4))
    with pytest.raises(ValueError, match="mono PCM16"):
        benchmark.load_audio(path)
    inputs = _compile_input_files(model_dir.parents[1], model_dir.name)
    assert model_dir / "scripts/performance.py" not in inputs


def test_fleurs_float32_selection_is_ordered_and_preserves_samples(load_script, tmp_path):
    import numpy as np
    from scipy.io import wavfile

    benchmark = load_script("performance")
    folder = tmp_path / "audio/test"
    folder.mkdir(parents=True)
    first = np.full(16 * 16000, 0.125, np.float32)
    second = np.full(20 * 16000, 0.25, np.float32)
    for name, audio in (("1.wav", first), ("2.wav", second)):
        wavfile.write(folder / name, 16000, audio)
    # Sentence IDs can repeat, filenames cannot; preserve literal transcript quotes.
    rows = ['5\t2.wav\t"Second sentence"\tsecond sentence\ts e c\t320000\tFEMALE',
            '5\t1.wav\tFirst sentence\tfirst sentence\tf i r\t256000\tMALE']
    (tmp_path / "test.tsv").write_text('\n'.join(rows) + '\n')
    audio, metadata = benchmark.load_dataset(tmp_path)
    assert np.array_equal(audio, np.concatenate((first, second[:14 * 16000])))
    assert [item['id'] for item in metadata['recordings']] == ['en_us/test/1', 'en_us/test/2']
    assert [item['samples_used'] for item in metadata['recordings']] == [256000, 224000]
    (tmp_path / "test.tsv").write_text('\n'.join(rows).replace('256000', '1'))
    with pytest.raises(ValueError, match="sample count differs"):
        benchmark.load_dataset(tmp_path)
    wavfile.write(folder / 'nonfinite.wav', 16000, np.array([np.nan], np.float32))
    with pytest.raises(ValueError, match="finite audio"):
        benchmark.read_audio(folder / 'nonfinite.wav')


def test_rtf_counts_unique_audio_and_uses_separate_paced_trials(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np

    benchmark = load_script("performance")
    clock = [0.0]
    recording = np.zeros(480000, np.float32)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(benchmark, "WARMUP_RECORDINGS", 2)
    monkeypatch.setattr(benchmark, "MEASURED_RECORDINGS", 2)
    monkeypatch.setattr(benchmark, "MIN_DURATION_S", 0)
    monkeypatch.setattr(benchmark, "FIRST_TEXT_TRIALS", 5)
    paced = []

    def first_text(*_):
        paced.append(True)
        clock[0] += 4.5
        return 4.5

    monkeypatch.setattr(benchmark, "time_to_first_text", first_text)

    class Monitor:
        def start(self):
            self.started = clock[0]

        def stop(self):
            assert clock[0] - self.started == 30
            return "unpaced telemetry"

    class Transcriber:
        config = {"kind": "parakeet"}
        model = "encoder"

        def transcribe(self, audio, language):
            assert audio is recording
            clock[0] += 15  # Total encoder windows are irrelevant to the unique 30 s of audio.
            yield "Full transcript"

    fake_neat = SimpleNamespace(TelemetryMonitor=Monitor, BenchmarkReport=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "pyneat", fake_neat)
    report = SimpleNamespace(document={"metrics": {}}, evidence={}, save=lambda _: None)

    def from_performance(core, models, **kwargs):
        assert core.latency_ms == 15000
        assert core.throughput_samples == 2
        assert core.throughput_duration_ms == 30000
        assert core.fps == pytest.approx(2 / 30)
        assert kwargs["telemetry"] == "unpaced telemetry"
        assert kwargs["mla_layer_stats"] == {"model": {"status": "not requested", "layers": []}}
        assert kwargs["mla_stage_stats"] == {"model": {"status": "not requested", "stages": []}}
        return report

    monkeypatch.setattr(benchmark.model_report, "from_performance", from_performance)
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"input")
    benchmark.benchmark_pipeline(Transcriber(), recording, {"source": "test"}, tmp_path)
    assert report.document["metrics"] == {"real_time_factor": 0.5, "time_to_first_text_ms": 4500}
    assert len(paced) == 5
    assert report.evidence["asr-timings.json"]["audio_samples"] == 480000


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
        "model/frontend.npz",
    } <= names
    assert not {"model/transcribe.py", "model/requirements.txt"}.intersection(names)
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

    runtime = load_script("performance")

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

    runtime = load_script("performance")

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

    manifest = validate()
    assert len(manifest["runtime_files"]) == len(recipe.RUNTIME_FILES)
    assert not any(item["path"].endswith((".py", "/requirements.txt")) for item in manifest["runtime_files"])
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
