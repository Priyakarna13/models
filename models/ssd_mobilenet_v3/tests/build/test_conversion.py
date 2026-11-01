"""Independent export contracts; heavy source/ORT parity is run by convert.py."""

import numpy as np
import onnx
import pytest


@pytest.fixture
def converter(load_script):
    return load_script("convert")


def test_weight_digest_is_full_and_fail_closed(tmp_path, converter):
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"not the official weights")
    assert len(converter.WEIGHTS_SHA256) == 64
    with pytest.raises(ValueError, match="digest mismatch"):
        converter.verified_weights(path)


def test_raw_heads_have_3234_anchors_and_91_logits(converter):
    assert len(converter.OUTPUTS) == 12
    assert [name for name, _shape in converter.OUTPUTS] == [
        *[f"ssd_loc_{index}" for index in range(6)],
        *[f"ssd_confidence_{index}" for index in range(6)],
    ]
    assert [shape[1] for _name, shape in converter.OUTPUTS] == [24] * 6 + [546] * 6
    assert sum(shape[2] * shape[3] * 6 for _name, shape in converter.OUTPUTS[:6]) == 3234


def test_flatten_is_cell_anchor_coordinate_order_not_channel_major(converter):
    # Independent coordinate construction: each scalar labels (cell,anchor,field).
    head = np.zeros((1, 24, 2, 2), np.float32)
    expected = []
    for row in range(2):
        for column in range(2):
            for anchor in range(6):
                values = []
                for coordinate in range(4):
                    value = 1000 * row + 100 * column + 10 * anchor + coordinate
                    head[0, anchor * 4 + coordinate, row, column] = value
                    values.append(value)
                expected.append(values)
    np.testing.assert_array_equal(converter.flatten_heads([head], 4), np.array([expected], np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_parity_rejects_nonfinite(bad, converter):
    with pytest.raises(ValueError, match="nonfinite"):
        converter.parity(np.array([1.0]), np.array([bad]))


def test_parity_rejects_changed_shape_or_large_error(converter):
    with pytest.raises(ValueError, match="shape"):
        converter.parity(np.ones(1), np.ones(2))
    assert not converter.parity(np.ones(1), np.array([1.01]))["within_experimental_limits"]
    assert converter.EXPERIMENTAL_MAX_NORMALIZED_RMSE == 1e-5


def test_actual_export_has_fixed_twelve_head_fp32_contract(converted, converter):
    report = converter.audit(converted)
    assert report["operators"]["Conv"] == 98
    model = onnx.load(converted)
    properties = {item.key: item.value for item in model.metadata_props}
    assert properties["sima.precision.source"] == "fp32_pretrained_not_qat"
    assert properties["sima.input.normalization"] == "rgb_uint8/127.5-1; external to graph"
    assert properties["sima.torchvision.weights_sha256"] == converter.WEIGHTS_SHA256


@pytest.mark.parametrize("case", ["names", "order", "dtype", "shape", "opset"])
def test_actual_export_corruption_rejected(case, tmp_path, converted, converter):
    model = onnx.load(converted)
    if case == "names":
        # Rename both producer and public binding to retain ONNX validity.
        old = model.graph.output[0].name
        model.graph.output[0].name = "wrong_identity"
        for node in model.graph.node:
            for index, output in enumerate(node.output):
                if output == old:
                    node.output[index] = "wrong_identity"
    elif case == "order":
        first, second = onnx.ValueInfoProto(), onnx.ValueInfoProto()
        first.CopyFrom(model.graph.output[0])
        second.CopyFrom(model.graph.output[1])
        model.graph.output[0].CopyFrom(second)
        model.graph.output[1].CopyFrom(first)
    elif case == "dtype":
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 19
    else:
        model.opset_import[0].version = 18
    path = tmp_path / "corrupted.onnx"
    onnx.save(model, path)
    with pytest.raises((ValueError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError)):
        converter.audit(path)


def test_official_weights_use_the_canonical_verified_fetcher(monkeypatch, converter, tmp_path):
    from model_registry.source import SourceSpec

    path = tmp_path / "weights.pth"
    seen = []

    class FakeFetcher:
        def fetch(self, source):
            assert isinstance(source, SourceSpec)
            seen.append(source)
            return path, True

    monkeypatch.setattr(
        converter,
        "load_and_validate_model",
        lambda _: {"conversion": {"default": {"source": {"type": "file", **converter.WEIGHTS_SOURCE}}}},
    )
    monkeypatch.setattr(converter, "Fetcher", FakeFetcher)
    monkeypatch.setattr(converter, "verified_weights", lambda value: value)
    assert converter.resolve_weights(None) == path
    assert seen[0].uri == converter.WEIGHTS_URL
    assert seen[0].sha256 == converter.WEIGHTS_SHA256
    assert seen[0].size_bytes == 14069355


def test_explicit_local_weights_never_trigger_network(monkeypatch, converter, tmp_path):
    class NoFetcher:
        def __init__(self):
            raise AssertionError("an explicit local file must not fetch a fallback")

    monkeypatch.setattr(converter, "Fetcher", NoFetcher)
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="digest mismatch"):
        converter.resolve_weights(path)


@pytest.mark.parametrize(("machine", "vision"), [("x86_64", "0.23.0+cpu"), ("aarch64", "0.23.0")])
def test_conversion_architectures_pin_official_cpu_wheel_versions(converter, machine, vision):
    assert converter.expected_versions(machine) == {**converter.EXPECTED_VERSIONS, "torchvision": vision}


def test_conversion_rejects_unpinned_architecture(converter):
    with pytest.raises(ValueError, match="unsupported conversion architecture"):
        converter.expected_versions("unknown")


@pytest.fixture
def reference(load_script):
    return load_script("accuracy")


@pytest.fixture
def calibration(load_script):
    return load_script("calibration")


def _png(pixels):
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def test_source_priors_and_postprocessing_against_torchvision(converted, reference):
    import torch
    import torchvision
    from torchvision.models.detection.image_list import ImageList

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
    features = [torch.empty((1, 1, size, size)) for size in reference.GRIDS]
    anchors = model.anchor_generator(ImageList(torch.empty((1, 3, 320, 320)), [(320, 320)]), features)[0]
    np.testing.assert_array_equal(reference.ANCHORS, anchors.numpy())
    reg = np.random.default_rng(1).normal(0, 0.05, (3234, 4)).astype(np.float32)
    logits = np.full((3234, 91), -100, np.float32)
    logits[:, 0] = 20
    for index, label, score in [(0, 1, 7), (300, 2, 8), (1100, 90, 9)]:
        logits[index, 0] = 0
        logits[index, label] = score
    actual = reference.post_numpy(reg, logits)
    expected = model.postprocess_detections(
        {"bbox_regression": torch.from_numpy(reg)[None], "cls_logits": torch.from_numpy(logits)[None]},
        [anchors],
        [(320, 320)],
    )[0]
    np.testing.assert_array_equal(actual["labels"], expected["labels"].numpy())
    np.testing.assert_allclose(actual["scores"], expected["scores"].numpy(), atol=2e-7, rtol=0)
    np.testing.assert_allclose(actual["boxes"], expected["boxes"].numpy(), atol=4e-5, rtol=0)


def test_rgb_and_half_pixel_normalization_before_resize(converted, calibration):
    import torch
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    y, x = np.indices((450, 600))
    pixels = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    actual = calibration.preprocess_image(_png(pixels))
    transform = GeneralizedRCNNTransform(320, 320, [0.5] * 3, [0.5] * 3, size_divisible=1, fixed_size=(320, 320)).eval()
    image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    expected = transform([image])[0].tensors[0].permute(1, 2, 0).numpy()
    assert actual.shape == (320, 320, 3) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    red = np.zeros((17, 29, 3), np.uint8)
    red[:, :, 0] = 255
    np.testing.assert_array_equal(calibration.preprocess_image(_png(red))[160, 160], [1, -1, -1])


@pytest.mark.parametrize("host", ["artifact", "huggingface"])
def test_checkpoint_mirror_preserves_original_bytes_and_provenance(monkeypatch, converter, tmp_path, host):
    source = {"type": "file", **converter.WEIGHTS_SOURCE}
    source["uri"] = (
        f"https://artifacts.stg.neat.sima.ai/models-onnx/ssd_mobilenet_v3/{converter.WEIGHTS_SHA256}/123-1/source.pth"
        if host == "artifact" else
        "hf://models/example/ssd@0123456789abcdef0123456789abcdef01234567/source.pth"
    )
    path = tmp_path / "verified.pth"
    seen = []
    monkeypatch.setattr(converter, "load_and_validate_model", lambda _: {
        "conversion": {"default": {"source": source}},
        "upstream": {"derived_from": converter.WEIGHTS_URL},
    })

    class FakeFetcher:
        def fetch(self, spec):
            seen.append(spec)
            return path, True

    monkeypatch.setattr(converter, "Fetcher", FakeFetcher)
    monkeypatch.setattr(converter, "verified_weights", lambda value: value)
    assert converter.resolve_weights(None) == path
    assert seen[0].uri == source["uri"]
    assert (seen[0].sha256, seen[0].size_bytes) == (converter.WEIGHTS_SHA256, converter.WEIGHTS_SIZE)


@pytest.mark.parametrize("case", ["sha256", "size", "provenance", "wrong_hash_path"])
def test_checkpoint_mirror_rejects_changed_identity_before_fetch(monkeypatch, converter, case):
    from model_registry.errors import SourcePolicyError

    source = {"type": "file", **converter.WEIGHTS_SOURCE}
    source["uri"] = (
        f"https://artifacts.stg.neat.sima.ai/models-onnx/ssd_mobilenet_v3/{converter.WEIGHTS_SHA256}/123-1/source.pth"
    )
    model = {"conversion": {"default": {"source": source}}, "upstream": {"derived_from": converter.WEIGHTS_URL}}
    if case == "sha256":
        source["sha256"] = "0" * 64
    elif case == "size":
        source["size_bytes"] -= 1
    elif case == "provenance":
        model["upstream"]["derived_from"] = "https://example.invalid/not-the-original"
    else:
        source["uri"] = source["uri"].replace(converter.WEIGHTS_SHA256, "0" * 64)
    monkeypatch.setattr(converter, "load_and_validate_model", lambda _: model)

    class NoFetcher:
        def __init__(self):
            raise AssertionError("a rejected checkpoint identity must not fetch")

    monkeypatch.setattr(converter, "Fetcher", NoFetcher)
    with pytest.raises((ValueError, SourcePolicyError)):
        converter.resolve_weights(None)


def test_numpy_source_oracle_does_not_require_board_image_io(load_script, monkeypatch):
    import sys

    # The conversion environment needs the NumPy oracle, not native image I/O.
    monkeypatch.setitem(sys.modules, "cv2", None)
    reference = load_script("accuracy")
    assert reference.ANCHORS.shape == (3234, 4)
    logits = np.full((3234, 91), -100, np.float32)
    logits[:, 0] = 20
    actual = reference.post_numpy(np.zeros((3234, 4), np.float32), logits)
    assert len(actual["boxes"]) == 0
