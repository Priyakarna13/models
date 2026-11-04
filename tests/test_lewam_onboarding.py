"""Host contracts for the staged model; no board/compiler success is inferred."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGED = ROOT / "contrib/lewam_pusht"


@pytest.fixture
def script(monkeypatch):
    def load(name):
        path = STAGED / (
            f"{name}.py" if name in {"stage", "stage_data", "ci_sources", "promote"} else f"model/scripts/{name}.py"
        )
        monkeypatch.syspath_prepend(str(STAGED / "model/scripts"))
        for sibling in ("surgery", "calibration", "performance"):
            monkeypatch.delitem(sys.modules, sibling, raising=False)
        spec = importlib.util.spec_from_file_location(f"lewam_test_{name}", path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def request_data(script):
    return {
        "images": np.zeros((2, 224, 224, 3), np.uint8),
        "history_pad": np.array([True, False]),
        "noise": np.zeros((10, 2), np.float32),
        "time_features": np.arange(8 * 192, dtype=np.float32).reshape(8, 192),
        "horizon_features": np.ones(192, np.float32),
        "attention_mask": script("accuracy").attention_mask(True),
    }


def toy_graphs(directory, script):
    surgery = script("surgery")
    paths = {}
    for name, (input_shape, output_shape) in surgery.SHAPES.items():
        input_name = "policy_input" if name == "policy" else "input"
        output_name = "velocity_padded" if name == "policy" else "output"
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("ConstantOfShape", ["shape"], [output_name])],
            name,
            [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)],
            [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)],
            [onnx.numpy_helper.from_array(np.array(output_shape, np.int64), "shape")],
        )
        path = directory / f"{name}.onnx"
        onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)]), path)
        paths[name] = path
    return paths


def test_source_stage_hashes_every_graph_and_never_uploads(tmp_path, script):
    graphs = toy_graphs(tmp_path, script)
    report = script("stage").stage(graphs, tmp_path / "stage")
    assert len(report["files"]) == 4
    assert "not numerical or native qualification" in report["scope"]
    for name, row in report["files"].items():
        assert row["sha256"] == hashlib.sha256((tmp_path / "stage" / name).read_bytes()).hexdigest()
        assert "uri" not in row
    with pytest.raises(FileExistsError):
        script("stage").stage(graphs, tmp_path / "stage")


def test_stage_rejects_missing_component_before_writing(tmp_path, script):
    with pytest.raises(ValueError, match="Exactly"):
        script("stage").stage({}, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_stage_rejects_changed_policy_abi(tmp_path, script):
    graphs = toy_graphs(tmp_path, script)
    proto = onnx.load(graphs["policy"])
    proto.graph.input[0].name = "reduced_policy"
    onnx.save(proto, graphs["policy"])
    with pytest.raises(ValueError, match="original full19"):
        script("stage").stage(graphs, tmp_path / "absent")


def publication_receipt(tmp_path, script):
    report = script("stage").stage(toy_graphs(tmp_path, script), tmp_path / "source-stage")
    for name, row in report["files"].items():
        # Unit-test network identity only; never a real production pin.
        row["uri"] = f"https://artifacts.stg.neat.sima.ai/models-onnx/lewam_pusht/{row['sha256']}/1-1/model.onnx"
    return report


def test_definition_uses_one_profile_four_components_and_no_fake_benchmarks(tmp_path, script):
    receipt = publication_receipt(tmp_path, script)
    calibration = {"id": "pusht", "variant": "cal", "sha256": "a" * 64, "size_bytes": 1}
    metadata = script("promote").definition(receipt, calibration)
    assert metadata["compilation"]["components"] == ["front", "body", "head", "policy"]
    assert list(metadata["build_profiles"]) == ["w16a16"]
    assert not {"performance", "accuracy", "reference_metrics", "build-on-demand"} & set(metadata)


def test_promotion_fetches_before_atomic_catalog_entry(tmp_path, script, monkeypatch):
    receipt = publication_receipt(tmp_path, script)
    repository = tmp_path / "repository"
    for path in ("schemas", "taxonomies", "deps"):
        shutil.copytree(ROOT / path, repository / path)
    (repository / "models").mkdir()
    calibration = {"id": "pusht", "variant": "cal", "sha256": "a" * 64, "size_bytes": 1}
    module = script("promote")
    calls = []

    def fail_fetch(self, source):
        calls.append(source)
        raise RuntimeError("not published")

    monkeypatch.setattr(module.Fetcher, "fetch_source", fail_fetch)
    with pytest.raises(RuntimeError, match="not published"):
        module.promote(receipt, calibration, repository, tmp_path / "cache")
    assert not (repository / "models/lewam_pusht").exists()
    assert len(calls) == 1 and len(calls[0]["files"]) == 4
    monkeypatch.setattr(module.Fetcher, "fetch_source", lambda *args: None)
    destination = module.promote(receipt, calibration, repository, tmp_path / "cache")
    assert (destination / "model.yaml").is_file()
    assert not (destination / "definition.yaml").exists()


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "0" * 64), ("source_revision", "0" * 40)])
def test_promotion_rejects_identity_drift(tmp_path, script, field, value):
    receipt = publication_receipt(tmp_path, script)
    receipt[field] = value
    with pytest.raises(ValueError, match="identity"):
        script("promote").definition(receipt, {})


def test_compile_disables_both_fp32_normalizations(script):
    module = script("compile")
    assert module.compile_options()["layer_norm_use_fp32_intermediates"] is False
    assert module.compile_options()["rms_norm_use_fp32_intermediates"] is False
    assert module.compile_options()["deployable"] is True
    assert module.component_name("w16a16", "front") == "lewam_pusht_w16a16_front"
    with pytest.raises(ValueError):
        module.component_name("w8a8", "front")


def test_calibration_rejects_traversal_tar(tmp_path, script):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("../calibration.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError):
        script("compile").unpack_dataset(path, tmp_path)


def test_calibration_rejects_overlap(tmp_path, script):
    module = script("calibration")
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "schema": "lewam.calibration.v1",
                "source": "source_trajectories",
                "checkpoint_sha256": module.CHECKPOINT_SHA256,
                "episode_ids": [1],
                "evaluation_episode_ids": [1],
            }
        )
    )
    with pytest.raises(ValueError, match="disjoint"):
        module.read_calibration(tmp_path, "front", 1)


def test_eight_step_feedback_and_original_packing(script, request_data):
    module = script("performance")
    module.validate_request(request_data)
    history = np.zeros((1, 192, 1, 2), np.float32)
    history[:, :, :, 1] = 2
    goal = np.full((1, 192, 1, 1), 3, np.float32)
    inputs = []

    def velocity(packed):
        inputs.append(packed.copy())
        result = np.zeros((1, 192, 1, 10), np.float32)
        # Makes each later input depend on its own previous output.
        result[:, :2] = packed[:, :2, :, 3:13] + np.float32(1)
        return result

    output = module.integrate(velocity, history, goal, request_data)
    x = np.zeros((10, 2), np.float32)
    assert len(inputs) == 8
    for step, packed in enumerate(inputs):
        np.testing.assert_array_equal(packed[:, :, :, :2], history)
        np.testing.assert_array_equal(packed[:, :, :, 2:3], goal)
        np.testing.assert_array_equal(packed[0, :2, 0, 3:13].T, x)
        np.testing.assert_array_equal(packed[0, :, 0, 13], request_data["time_features"][step])
        assert np.all(packed[:, 2:, :, 3:13] == 0)
        x = np.add(x, np.divide(np.add(x, np.float32(1)), np.float32(8)), dtype=np.float32)
    expected = np.add(np.multiply(x, module.STD, dtype=np.float32), module.MEAN, dtype=np.float32)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("kind", ["nonfinite", "padding", "shape"])
def test_policy_output_failures_stop_before_next_step(script, request_data, kind):
    module = script("performance")
    result = np.zeros((1, 192, 1, 10), np.float32)
    if kind == "nonfinite":
        result[0, 0, 0, 0] = np.nan
    elif kind == "padding":
        result[0, 2, 0, 0] = 1
    else:
        result = result[..., :5]
    calls = []

    def run(value):
        calls.append(value)
        return result

    with pytest.raises(ValueError):
        module.integrate(run, np.zeros((1, 192, 1, 2), np.float32), np.zeros((1, 192, 1, 1), np.float32), request_data)
    assert len(calls) == 1


def test_request_cannot_infer_padding_from_zero_images(script, request_data):
    module = script("performance")
    bad = copy.deepcopy(request_data)
    bad["history_pad"][:] = False
    with pytest.raises(ValueError, match="Images"):
        module.validate_request(bad)


def test_success_uses_goal_state_not_coverage(script):
    module = script("accuracy")
    goal = np.zeros(7)
    assert module.success(goal, goal)
    moved = goal.copy()
    moved[0] = 20
    assert not module.success(goal, moved)
    moved = goal.copy()
    moved[4] = np.pi / 9
    assert not module.success(goal, moved)


def test_measure_counts_completed_replans_not_actions(script):
    module = script("performance")
    tick = iter(range(100))
    calls = []

    def infer():
        calls.append(1)
        return np.zeros((10, 2), np.float32)

    samples, elapsed = module.measure(infer, samples=3, seconds=0, warmups=2, clock=lambda: next(tick))
    assert len(samples) == 3 and len(calls) == 5 and elapsed > 0


def test_new_metric_is_a_ratio_and_has_matching_browser_label(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "pyneat", SimpleNamespace(__version__="host-test", build_info=lambda: {}))
    from model_report import from_accuracy

    result = from_accuracy(
        {
            "pusht-reproduction": {
                "metrics": {"pusht_success_rate": 0.8},
                "split": "reproduction-non-held-out",
                "evaluated_samples": 150,
                "available_samples": 150,
            }
        }
    )
    assert result is not None
    assert 'pusht_success_rate: "PushT success rate"' in (ROOT / "artifact-browser/accuracy-metrics.js").read_text()
    with pytest.raises(ValueError, match="ratio"):
        from_accuracy(
            {
                "pusht-reproduction": {
                    "metrics": {"pusht_success_rate": 80},
                    "split": "reproduction-non-held-out",
                    "evaluated_samples": 150,
                    "available_samples": 150,
                }
            }
        )


def test_staged_model_is_not_an_active_catalog_entry():
    assert not (STAGED / "model/model.yaml").exists()
    template = yaml.safe_load((STAGED / "model/definition.yaml").read_text())
    assert "source" not in template["compilation"]


def fake_neat(module, *, failure=None, close_failure=None):
    """Explicit synchronous API double, not a device/layout qualification."""
    from types import SimpleNamespace as NS

    log, builds, copies, packed = [], [], [], []

    class Tensor:
        def __init__(self, value):
            self.value = value

        @staticmethod
        def from_numpy(value, **kwargs):
            copies.append(kwargs)
            assert kwargs["copy"] is True
            return Tensor(value.copy())

        def to_numpy(self):
            return self.value

    class Runner:
        def __init__(self, name):
            self.name = name
            shape = module.OUTPUTS[name]
            self.output = Tensor(np.zeros((shape[0], shape[2], shape[3], shape[1]), np.float32))
            self.count, self.closes = 0, 0

        def run(self, inputs, *, timeout_ms):
            assert len(inputs) == 1 and timeout_ms == module.TIMEOUT_MS
            log.append(("run", self.name))
            self.count += 1
            if self.name == failure:
                raise TimeoutError("test double timeout")
            if self.name == "head":
                # Reuse one borrowed buffer to test that the caller owns latents.
                self.output.value.fill(self.count)
            if self.name == "policy":
                packed.append(inputs[0].value.copy())
            return [self.output]

        def close(self):
            log.append(("close", self.name))
            self.closes += 1
            if self.name == close_failure and self.closes == 1:
                raise RuntimeError("test double incomplete close")

    class Model:
        def __init__(self, path, options):
            self.name = Path(path).stem
            self.options = options

        def build(self, inputs, *, run_options):
            assert run_options.queue_depth == 1 and run_options.overflow_policy == "block"
            assert len(inputs) == 1
            builds.append(self.name)
            return Runner(self.name)

    def options():
        return NS(preprocess=NS(resize=NS(), color_convert=NS(), normalize=NS()))

    return NS(
        Tensor=Tensor,
        Model=Model,
        ModelOptions=options,
        RunOptions=NS,
        AutoFlag=NS(On="on", Off="off"),
        InputKind=NS(Image="image", Tensor="tensor"),
        NormalizePreset=NS(ImageNet="imagenet"),
        OverflowPolicy=NS(Block="block"),
        TensorLayout=NS(NHWC="nhwc"),
        TensorMemory=NS(EV74="ev74"),
        PixelFormat=NS(RGB="rgb"),
        log=log,
        builds=builds,
        copies=copies,
        packed=packed,
    )


def test_public_runtime_keeps_four_runners_and_owns_borrowed_outputs(script, request_data):
    module = script("performance")
    pn = fake_neat(module)
    policy = module.Policy({name: Path(name) for name in module.COMPONENTS}, pn)
    output = policy.replan(request_data)
    assert output.shape == (10, 2)
    assert pn.builds == list(module.COMPONENTS)
    assert [name for operation, name in pn.log if operation == "run"] == (
        ["front", "body", "head"] * 2 + ["policy"] * 8
    )
    assert len(pn.packed) == 8
    # Goal encoding reused the head output buffer; history must still be frame 1.
    packed = pn.packed[0].transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(packed[:, :, :, 0], 0)
    np.testing.assert_array_equal(packed[:, :, :, 1], 1)
    np.testing.assert_array_equal(packed[:, :, :, 2], 2)
    policy.reset()
    request = dict(
        request_data,
        images=np.zeros((3, 224, 224, 3), np.uint8),
        history_pad=np.zeros(2, np.bool_),
        attention_mask=module.attention_mask(False),
    )
    policy.replan(request)
    assert policy.generation == 1
    assert pn.builds == list(module.COMPONENTS)
    assert len([row for row in pn.log if row == ("run", "policy")]) == 16
    assert len([row for row in pn.log if row == ("run", "front")]) == 5
    policy.close()
    assert pn.log[-4:] == [("close", name) for name in reversed(module.COMPONENTS)]
    count = len(pn.log)
    policy.close()
    assert len(pn.log) == count
    with pytest.raises(RuntimeError, match="closed"):
        policy.replan(request)


def test_public_runtime_never_retries_native_timeout(script, request_data):
    module = script("performance")
    pn = fake_neat(module, failure="policy")
    policy = module.Policy({name: Path(name) for name in module.COMPONENTS}, pn)
    with pytest.raises(TimeoutError):
        policy.replan(request_data)
    calls = len(pn.log)
    with pytest.raises(RuntimeError, match="failed"):
        policy.replan(request_data)
    with pytest.raises(RuntimeError, match="failed"):
        policy.reset()
    assert len(pn.log) == calls
    policy.close()
    assert not policy.runners


def test_public_runtime_reports_and_retains_failed_close(script, request_data):
    module = script("performance")
    pn = fake_neat(module, close_failure="head")
    policy = module.Policy({name: Path(name) for name in module.COMPONENTS}, pn)
    policy.replan(request_data)
    with pytest.raises(RuntimeError, match="failed to close"):
        policy.close()
    assert policy.closed and list(policy.runners) == ["head"]
    before = len(pn.log)
    policy.close()
    assert pn.log[before:] == [("close", "head")]
    assert not policy.runners


def test_public_runtime_rejects_overlapping_replan_reset_and_close(script, request_data):
    module = script("performance")
    pn = fake_neat(module)
    policy = module.Policy({name: Path(name) for name in module.COMPONENTS}, pn)
    policy._request_lock.acquire()
    try:
        for operation in (lambda: policy.replan(request_data), policy.reset, policy.close):
            with pytest.raises(RuntimeError, match="in flight"):
                operation()
    finally:
        policy._request_lock.release()
    assert not policy.failed and not policy.closed and not pn.builds
    policy.replan(request_data)
    policy.close()


def test_mask_rejects_new_future_token_attention(script, request_data):
    module = script("performance")
    request_data["attention_mask"][18, 9] = 0
    with pytest.raises(ValueError, match="topology"):
        module.validate_request(request_data)


def test_real_calibration_layout_count_and_hash_contract(tmp_path, script):
    module = script("calibration")
    shutil.copyfile(ROOT / "tests/fixtures/lewam_pusht/selection.json", tmp_path / "selection.json")
    excluded = sorted({row["episode"] for row in json.loads((tmp_path / "selection.json").read_text())["episodes"]})
    data = np.arange(2 * 192 * 34, dtype=np.float32).reshape(2, 192, 1, 34)
    path = tmp_path / "policy.npz"
    np.savez(path, inputs=data, episode_ids=np.array([123, 124], np.int64))
    manifest = {
        "schema": "lewam.calibration.v1",
        "source": "source_trajectories",
        "checkpoint_sha256": module.CHECKPOINT_SHA256,
        "episode_ids": [123, 124],
        "evaluation_episode_ids": excluded,
        "components": {
            name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name in ("front", "body", "head", "policy")
        },
    }
    (tmp_path / "calibration.json").write_text(json.dumps(manifest))
    actual = module.read_calibration(tmp_path, "policy", 2)
    np.testing.assert_array_equal(actual, data.transpose(0, 2, 3, 1))
    assert actual.flags.c_contiguous
    with pytest.raises(ValueError, match="Not enough"):
        module.read_calibration(tmp_path, "policy", 3)
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="Changed"):
        module.read_calibration(tmp_path, "policy", 1)


def test_evaluation_staging_rejects_changed_frozen_bytes_before_output(tmp_path, script):
    module = script("stage_data")
    source = tmp_path / "source"
    source.mkdir()
    for name in module.PINS:
        (source / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen evaluation source changed"):
        module.stage(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_evaluation_archive_has_deterministic_metadata(tmp_path, script, monkeypatch):
    module = script("stage_data")
    source = tmp_path / "source"
    source.mkdir()
    # Deliberately fake pack tests only tar determinism, not original data validity.
    pins = {}
    for name in module.PINS:
        (source / name).write_bytes(name.encode())
        pins[name] = module.digest(source / name)
    monkeypatch.setattr(module, "PINS", pins)
    monkeypatch.setattr(module, "validate_pack", lambda path: None)
    first = module.stage(source, tmp_path / "first")
    second = module.stage(source, tmp_path / "second")
    assert first == second
    assert "uri" not in first["archive"]
    with tarfile.open(tmp_path / "first" / first["archive"]["file"]) as archive:
        assert archive.getnames() == sorted(pins)
        assert all(x.isfile() and x.mtime == 0 and x.uid == 0 and x.gid == 0 for x in archive)
    with pytest.raises(FileExistsError):
        module.stage(source, tmp_path / "first")


def test_closed_loop_uses_native_prefix_actions_and_alive_rng_batches(script):
    module = script("accuracy")
    from types import SimpleNamespace as NS

    policy = NS(resets=0, requests=[])
    actions = np.arange(20, dtype=np.float32).reshape(10, 2)

    def reset():
        policy.resets += 1

    def replan(request):
        policy.requests.append(request)
        return actions

    policy.reset, policy.replan = reset, replan
    environments, draws, seeds = [], [], []

    class Environment:
        def __init__(self):
            self.order, self.steps, self.closed = [], 0, False
            environments.append(self)

        def reset(self, *, seed):
            self.order.append(("reset", seed))

        def _set_state(self, state):
            self.order.append(("state",))

        def _set_goal_state(self, goal):
            self.order.append(("goal",))
            self.goal = goal.copy()

        def render(self):
            return np.zeros((224, 224, 3), np.uint8)

        def step(self, action):
            np.testing.assert_array_equal(action, actions[self.steps])
            self.steps += 1
            reached = self.steps == 2
            state = self.goal.copy()
            if not reached:
                state[0] += 30
            return {"state": state}, 0.0, reached, False, {}

        def close(self):
            self.closed = True

    def noise_factory(seed):
        seeds.append(seed)

        def noise(count):
            draws.append(count)
            return np.zeros((count, 10, 2), np.float32)

        return noise

    rows = [
        {"episode": i, "start": 0, "reset_seed": i, "seed": seed}
        for i, seed in enumerate([42] * 50 + [0] * 50 + [1] * 50)
    ]
    arrays = {
        "initial_states": np.zeros((150, 7), np.float64),
        "goal_states": np.zeros((150, 7), np.float64),
        "goal_images": np.zeros((150, 224, 224, 3), np.uint8),
        "time_features": np.zeros((8, 192), np.float32),
        "horizon_features": np.zeros((10, 192), np.float32),
    }
    results = module.evaluate(policy, rows, arrays, Environment, noise_factory)
    assert len(results) == 150 and all(row["success"] and row["steps"] == 2 for row in results)
    assert policy.resets == 3 and seeds == [42, 0, 1] and draws == [50, 50, 50]
    assert len(policy.requests) == 150
    assert all(env.closed and [x[0] for x in env.order] == ["reset", "state", "goal"] for env in environments)


def test_ci_sources_use_standard_filename_without_claiming_publication(tmp_path, script):
    receipt = script("stage").stage(toy_graphs(tmp_path, script), tmp_path / "stage")
    module = script("ci_sources")
    prepared = module.prepare(tmp_path / "stage", tmp_path / "ci")
    assert prepared["status"] == "STAGED_NOT_PUBLISHED"
    for component, entry in prepared["sources"].items():
        assert entry["file"] == f"{component}/model.onnx"
        assert entry["sha256"] == receipt["files"][f"{component}.onnx"]["sha256"]
        assert hashlib.sha256((tmp_path / "ci" / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
        assert "uri" not in entry


def test_ci_pin_binding_checks_all_bytes_and_all_downloads(tmp_path, script, monkeypatch):
    receipt = publication_receipt(tmp_path, script)
    module = script("ci_sources")
    pins = {
        name.removesuffix(".onnx"): {
            "source": dict(type="file", **{key: row[key] for key in ("uri", "sha256", "size_bytes")})
        }
        for name, row in receipt["files"].items()
    }
    bad = copy.deepcopy(pins)
    bad["body"]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs"):
        module.bind(tmp_path / "source-stage", bad, tmp_path / "cache", tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()
    calls = []

    def download(self, spec):
        calls.append(spec)
        if len(calls) == 3:
            raise RuntimeError("third artifact not published")
        return tmp_path / "unused", False

    monkeypatch.setattr(module.Fetcher, "fetch", download)
    with pytest.raises(RuntimeError, match="not published"):
        module.bind(tmp_path / "source-stage", pins, tmp_path / "cache", tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()
    calls.clear()
    monkeypatch.setattr(module.Fetcher, "fetch", lambda self, spec: calls.append(spec))
    bound = module.bind(tmp_path / "source-stage", pins, tmp_path / "cache", tmp_path / "receipt.json")
    assert len(calls) == 4
    assert all("uri" in row for row in bound["files"].values())
    with pytest.raises(FileExistsError):
        module.bind(tmp_path / "source-stage", pins, tmp_path / "cache", tmp_path / "receipt.json")


def mpk_fixture(tmp_path, script, *, change=None, extra=None):
    """Synthetic MPK structure only; not a compiled/precision-qualified fixture."""
    module = script("compile")
    name = module.component_name("w16a16", "policy")
    graph = {"input": "policy_input", "input_shape": [1, 192, 1, 34], "output_shape": [1, 192, 1, 10]}
    manifest = {
        "model_sdk_version": "3.0.0",
        "input_nodes": [{"name": "policy_input", "size": 1 * 192 * 1 * 34 * 4}],
        "plugins": [
            {"processor": "MLA", "sequence": 0},
            {
                "processor": "CVU",
                "sequence": 1,
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"name": "velocity_padded", "size": 1 * 192 * 1 * 10 * 4}],
            },
        ],
    }
    if change:
        change(manifest)
    members = {f"{name}_mpk.json": json.dumps(manifest).encode(), f"{name}.elf": b"\x7fELFhost-test-fixture"}
    if extra:
        members.update(extra)
    path = tmp_path / f"{name}_mpk.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for filename, contents in members.items():
            entry = tarfile.TarInfo(filename)
            entry.size = len(contents)
            archive.addfile(entry, io.BytesIO(contents))
    return module, path, name, graph


def test_mpk_audit_uses_real_registry_manifest_convention(tmp_path, script):
    module, path, name, graph = mpk_fixture(tmp_path, script)
    receipt = module.audit_archive(path, name, "3.0.0", graph)
    assert receipt["mpk_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["public_inputs"] == {"policy_input": 26112}
    assert receipt["public_output_sizes"] == [7680]
    assert "no_fp32_mla" not in receipt  # Structural acceptance is not precision proof.


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda m: m.update(model_sdk_version="2.1.3"), "compiler version"),
        (lambda m: m["input_nodes"][0].update(size=1), "input interface"),
        (lambda m: m["plugins"][-1]["output_nodes"][0].update(size=1), "output interface"),
        (lambda m: m["plugins"][0].update(processor="A65"), "host fallback"),
        (lambda m: m["plugins"][-1].update(sequence=0), "unique sequence"),
        (lambda m: m["plugins"][-1]["config_params"].update(kernel="unknown"), "output interface"),
    ],
)
def test_mpk_audit_rejects_incompatible_packages(tmp_path, script, change, match):
    module, path, name, graph = mpk_fixture(tmp_path, script, change=change)
    with pytest.raises(ValueError, match=match):
        module.audit_archive(path, name, "3.0.0", graph)


@pytest.mark.parametrize(
    "filename,contents,match",
    [
        ("libprivate.so.1", b"private runtime", "private runtime"),
        ("../escaped", b"outside", "Unsafe"),
        ("duplicate_mpk.json", b"{}", "Exactly one"),
        ("./lewam_pusht_w16a16_policy.elf", b"duplicate", "Duplicate"),
    ],
)
def test_mpk_audit_rejects_private_payloads_and_unsafe_members(tmp_path, script, filename, contents, match):
    module, path, name, graph = mpk_fixture(tmp_path, script, extra={filename: contents})
    with pytest.raises(ValueError, match=match):
        module.audit_archive(path, name, "3.0.0", graph)


def test_source_stage_is_atomic_when_copy_verification_fails(tmp_path, script, monkeypatch):
    graphs = toy_graphs(tmp_path, script)
    module = script("stage")
    monkeypatch.setattr(module.shutil, "copyfile", lambda source, target: target.write_bytes(b"changed"))
    with pytest.raises(ValueError, match="changed while staging"):
        module.stage(graphs, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".lewam-source-*"))


@pytest.mark.parametrize("field", ["checkpoint_sha256", "source_revision"])
def test_ci_preparation_rejects_provenance_drift(tmp_path, script, field):
    script("stage").stage(toy_graphs(tmp_path, script), tmp_path / "stage")
    receipt = tmp_path / "stage/source-stage.json"
    metadata = json.loads(receipt.read_text())
    metadata[field] = "0" * len(metadata[field])
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity"):
        script("ci_sources").prepare(tmp_path / "stage", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_accuracy_closes_all_environments_if_one_close_fails(script):
    from types import SimpleNamespace as NS

    module = script("accuracy")
    closed = []

    class Environment:
        def __init__(self):
            self.index = len(created)
            created.append(self)

        def reset(self, **kwargs):
            pass

        def _set_state(self, state):
            pass

        def _set_goal_state(self, goal):
            pass

        def render(self):
            return np.zeros((224, 224, 3), np.uint8)

        def step(self, action):
            return {"state": np.zeros(7)}, 0.0, True, False, {}

        def close(self):
            closed.append(self.index)
            if self.index == 0:
                raise RuntimeError("fixture close failure")

    created = []
    rows = [{"seed": 42, "episode": i, "start": 0, "reset_seed": i} for i in range(150)]
    arrays = {
        "initial_states": np.zeros((150, 7), np.float64),
        "goal_states": np.zeros((150, 7), np.float64),
        "goal_images": np.zeros((150, 224, 224, 3), np.uint8),
        "time_features": np.zeros((8, 192), np.float32),
        "horizon_features": np.zeros((10, 192), np.float32),
    }
    policy = NS(reset=lambda: None, replan=lambda request: np.zeros((10, 2), np.float32))
    with pytest.raises(RuntimeError, match="1 PushT environment"):
        module.evaluate(
            policy, rows, arrays, Environment, lambda seed: lambda count: np.zeros((count, 10, 2), np.float32)
        )
    assert closed == list(range(50))
