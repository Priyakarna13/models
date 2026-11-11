from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

COMPONENTS = ["blocks_0", "blocks_1", "blocks_2", "blocks_3"]


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == COMPONENTS
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_blocks_0_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    (frame,) = performance["components"]["blocks_0"]["execution_plan"]["edges"][0]["tensors"]
    assert frame["shape"] == [1, 720, 1280, 3]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id


class _Flags:
    def __getattr__(self, name: str) -> str:
        return name


class _Options(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            preprocess=types.SimpleNamespace(resize=types.SimpleNamespace(), normalize=types.SimpleNamespace())
        )


class _Tensor:
    def __init__(self, array: np.ndarray, **options: Any) -> None:
        self.array, self.options, self.copied = array, options, False

    @classmethod
    def from_numpy(cls, array: np.ndarray, **options: Any) -> _Tensor:
        return cls(np.array(array), **options)

    def cvu(self) -> _Tensor:
        copy = _Tensor(self.array, **self.options)
        copy.copied = True
        return copy


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    flags = _Flags()
    pyneat = types.SimpleNamespace(
        ModelOptions=_Options,
        Tensor=_Tensor,
        InputKind=flags,
        AutoFlag=flags,
        ResizeMode=flags,
        NormalizePreset=flags,
        PixelFormat=flags,
        TensorMemory=flags,
    )
    monkeypatch.setitem(sys.modules, "pyneat", pyneat)
    monkeypatch.setitem(sys.modules, "model_report", types.SimpleNamespace())
    return load_script("performance")


def test_the_benchmark_frame_is_a_fixed_realistic_720p_scene(script: ModuleType) -> None:
    first, second = script.scene(), script.scene()
    assert (first.shape, first.dtype) == ((720, 1280, 3), np.uint8)
    assert np.array_equal(first, second) and first.std() > 10
    frame = script.frame_tensor()
    assert frame.options == {"copy": True, "image_format": "RGB", "memory": "EV74"}


def test_only_the_first_component_takes_the_image_route(script: ModuleType) -> None:
    options = script.image_options()
    assert (options.preprocess.kind, options.preprocess.resize.mode, options.preprocess.preset) == (
        "Image",
        "Stretch",
        "ImageNet",
    )
    assert (options.preprocess.enable, options.preprocess.resize.enable, options.preprocess.normalize.enable) == (
        "On",
        "On",
        "On",
    )
    # Neat sizes its input buffers, which the execution plan shows as its input, for the benchmark frame.
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (1280, 720)


def test_each_component_hands_a_copy_of_its_tokens_on(script: ModuleType) -> None:
    tokens = _Tensor(np.zeros((1, 201, 1280), np.float32))
    (handed,) = script.handoff([tokens])
    assert handed.copied and handed is not tokens


def test_the_script_takes_one_mpk_per_component_in_pipeline_order(script: ModuleType) -> None:
    values = [f"{component}=/mpk/{component}.tar.gz" for component in reversed(COMPONENTS)]
    assert list(script.parse_mpks(values)) == COMPONENTS
    for broken in (values[:-1], [*values, "extra=/mpk/x.tar.gz"], ["/mpk/model.tar.gz"]):
        with pytest.raises(SystemExit, match="--mpk must name exactly"):
            script.parse_mpks(broken)


def test_the_script_runs_pipeline_pys_components_at_its_image_size(
    script: ModuleType, load_script: Callable[[str], ModuleType]
) -> None:
    pipeline = load_script("pipeline")
    assert (script.COMPONENTS, script.IMAGE_SIZE) == (pipeline.COMPONENTS, pipeline.IMAGE_SIZE)


def test_the_script_measures_as_the_benchmarking_guide_requires(script: ModuleType) -> None:
    assert (script.WARMUP_FRAMES, script.MEASURED_FRAMES, script.MIN_DURATION_S) == (100, 1000, 60)


def test_the_script_keeps_cores_mla_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    layers = [types.SimpleNamespace(stage_index=0, name="conv", latency_us=12.5)]
    stages = [types.SimpleNamespace(stage_index=0, latency_us=40.0)]

    class Model:
        def __init__(self, mpk: str) -> None:
            self.mpk = mpk

        def benchmark(self, options: Any) -> Any:
            requests.append(options)
            return types.SimpleNamespace(
                mla_layer_stats=types.SimpleNamespace(status="ok", layers=layers),
                mla_stage_stats=types.SimpleNamespace(status="ok", stages=stages),
            )

    monkeypatch.setitem(
        sys.modules, "pyneat", types.SimpleNamespace(Model=Model, BenchmarkOptions=types.SimpleNamespace)
    )
    monkeypatch.setitem(sys.modules, "model_report", types.SimpleNamespace())
    path = Path(__file__).resolve().parents[2] / "scripts" / "performance.py"
    monkeypatch.syspath_prepend(str(path.parent))
    layer_stats, stage_stats = runpy.run_path(str(path))["mla_stats"](Path("blocks_0.tar.gz"))

    assert [(r.include_mla_layer_stats, r.include_mla_stage_stats) for r in requests] == [(True, True)]
    assert layer_stats == {"status": "ok", "layers": [{"stage_index": 0, "name": "conv", "latency_us": 12.5}]}
    assert stage_stats == {"status": "ok", "stages": [{"stage_index": 0, "latency_us": 40.0}]}
