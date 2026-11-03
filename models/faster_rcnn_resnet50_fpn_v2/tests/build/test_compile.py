from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def _box_head_source(surgery: ModuleType) -> onnx.ModelProto:
    """The export's box head and predictor nodes, by their names, with seeded weights."""
    rng = np.random.default_rng(0)
    weights, nodes = [], []
    tensor = surgery.SOURCE_ROIS
    for index in range(4):
        prefix = f"/roi_heads/box_head/box_head.{index}"
        weights += [
            numpy_helper.from_array(rng.standard_normal((256, 256, 3, 3)).astype(np.float32), f"w{index}"),
            numpy_helper.from_array(rng.standard_normal(256).astype(np.float32), f"b{index}"),
        ]
        nodes += [
            helper.make_node(
                "Conv",
                [tensor, f"w{index}", f"b{index}"],
                [f"{prefix}/conv"],
                f"{prefix}/Conv",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Relu", [f"{prefix}/conv"], [f"{prefix}/relu"], f"{prefix}/Relu"),
        ]
        tensor = f"{prefix}/relu"
    gemms = (
        ("/roi_heads/box_head/box_head.5/Gemm", 1024, 256 * 49),
        ("/roi_heads/box_predictor/cls_score/Gemm", 91, 1024),
        ("/roi_heads/box_predictor/bbox_pred/Gemm", 364, 1024),
    )
    nodes.append(helper.make_node("Flatten", [tensor], ["flat"], "/roi_heads/box_head/box_head.4/Flatten"))
    for index, (name, rows, columns) in enumerate(gemms):
        weights += [
            numpy_helper.from_array(rng.standard_normal((rows, columns)).astype(np.float32), f"g{index}"),
            numpy_helper.from_array(rng.standard_normal(rows).astype(np.float32), f"c{index}"),
        ]
        source = "flat" if index == 0 else "fc"
        nodes.append(
            helper.make_node("Gemm", [source, f"g{index}", f"c{index}"], ["fc" if index == 0 else name], name, transB=1)
        )
    graph = helper.make_graph(
        nodes,
        "box_head_source",
        [helper.make_tensor_value_info(surgery.SOURCE_ROIS, TensorProto.FLOAT, [None, 256, 7, 7])],
        [
            helper.make_tensor_value_info(gemms[1][0], TensorProto.FLOAT, None),
            helper.make_tensor_value_info(gemms[2][0], TensorProto.FLOAT, None),
        ],
        weights,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_box_head_runs_every_region_as_one_image_with_the_source_weights(
    load_script: Callable[[str], ModuleType],
) -> None:
    surgery = load_script("surgery")
    source = _box_head_source(surgery)

    tiled = surgery.box_head_graph(source)

    convs = [node for node in tiled.graph.node if node.op_type == "Conv"]
    assert [node.name for node in convs] == [*(f"conv{i}" for i in range(4)), "fc", "class_score", "box_prediction"]
    assert [node.op_type for node in tiled.graph.node].count("Mul") == 3
    initializers = {item.name: numpy_helper.to_array(item) for item in tiled.graph.initializer}
    source_weights = {item.name: numpy_helper.to_array(item) for item in source.graph.initializer}
    np.testing.assert_array_equal(initializers["fc_weight"].reshape(1024, -1), source_weights["g0"])
    np.testing.assert_array_equal(initializers["class_weight"][:, :, 0, 0], source_weights["g1"])
    fc = next(node for node in tiled.graph.node if node.name == "fc")
    assert {item.name: helper.get_attribute_value(item) for item in fc.attribute} == {
        "kernel_shape": [7, 7],
        "strides": [8, 8],
    }
    shapes = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in tiled.graph.output}
    assert shapes == {"class_logits": [1, 91, 25, 40], "box_regression": [1, 364, 25, 40]}


def test_the_gap_mask_keeps_each_region_and_zeroes_the_gaps(load_script: Callable[[str], ModuleType]) -> None:
    mask = load_script("surgery").gap_mask()[0, 0]

    assert mask.shape == (200, 320)
    assert mask[:7, :7].all() and not mask[7, :].any() and not mask[:, 7].any()
    assert mask.sum() == 1000 * 49


def test_tiled_regions_come_back_in_grid_order(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    pooled = np.random.default_rng(0).standard_normal((1000, 7, 7, 256)).astype(np.float32)

    tiles = calibration.tile_regions(pooled[:999])

    assert tiles.shape == (200, 320, 256)
    # Region 41 is grid row 1, column 1; its cells start at pixel (8, 8), and the gap after it is zero.
    np.testing.assert_array_equal(tiles[8:15, 8:15], pooled[41])
    assert not tiles[15].any() and not tiles[:, 15].any()
    assert tiles[192:199, 304:311].any() and not tiles[192:199, 312:319].any(), "the 1,000th cell holds no region"
    # A 1x1 conv over the tiles reads one value per region, in the order region_rows returns.
    per_region = tiles[::8, ::8, 0][None, None]
    np.testing.assert_array_equal(calibration.region_rows(per_region)[:999, 0], pooled[:999, 0, 0, 0])
    np.testing.assert_array_equal(
        calibration.region_rows(per_region.transpose(0, 2, 3, 1))[:999, 0], pooled[:999, 0, 0, 0]
    )


@pytest.mark.parametrize(
    ("size", "resized"), [((480, 640), (800, 1066)), ((640, 427), (1199, 800)), ((300, 1000), (399, 1333))]
)
def test_images_resize_as_torchvision_does(
    load_script: Callable[[str], ModuleType], size: tuple[int, int], resized: tuple[int, int]
) -> None:
    assert load_script("calibration").resized_size(*size) == resized


def test_preprocessing_normalizes_and_pads_the_top_left_corner(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    image = np.full((480, 640, 3), 255, np.uint8)

    canvas, size = calibration.preprocess(image)

    assert canvas.shape == (1344, 1344, 3) and size == (800, 1066)
    np.testing.assert_allclose(canvas[400, 500], (1 - calibration.MEAN) / calibration.STD, rtol=1e-5)
    assert not canvas[800:].any() and not canvas[:, 1066:].any()


def test_anchors_are_torchvisions(load_script: Callable[[str], ModuleType]) -> None:
    anchors = load_script("calibration")._base_anchors(32)

    np.testing.assert_array_equal(anchors, [[-23, -11, 23, 11], [-16, -16, 16, 16], [-11, -23, 11, 23]])


def _reference_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order, keep = list(np.argsort(-scores, kind="stable")), []
    while order:
        best, order = order[0], order[1:]
        keep.append(best)
        x1 = np.maximum(boxes[best, 0], boxes[order, 0])
        y1 = np.maximum(boxes[best, 1], boxes[order, 1])
        x2 = np.minimum(boxes[best, 2], boxes[order, 2])
        y2 = np.minimum(boxes[best, 3], boxes[order, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[best]) + area(boxes[order]) - inter)
        order = [index for index, value in zip(order, iou, strict=True) if value <= threshold]
    return keep


def test_nms_matches_torchvisions_greedy_suppression(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    rng = np.random.default_rng(0)
    corners = rng.uniform(0, 400, (300, 2)).astype(np.float32)
    boxes = np.concatenate([corners, corners + rng.uniform(5, 120, (300, 2)).astype(np.float32)], axis=1)
    scores = rng.uniform(0, 1, 300).astype(np.float32)

    assert calibration.nms(boxes, scores, np.zeros(300), 0.5).tolist() == _reference_nms(boxes, scores, 0.5)


def _reference_roi_align(feature: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    """torchvision.ops.roi_align for one box on an (H, W, C) map, aligned=False, sampling ratio 2."""
    height, width = feature.shape[:2]
    x1, y1, x2, y2 = box * scale
    bin_h, bin_w = max(y2 - y1, 1.0) / 7, max(x2 - x1, 1.0) / 7
    out = np.zeros((7, 7, feature.shape[2]), np.float64)
    for row in range(7):
        for column in range(7):
            for sy in range(2):
                for sx in range(2):
                    y, x = y1 + row * bin_h + (sy + 0.5) * bin_h / 2, x1 + column * bin_w + (sx + 0.5) * bin_w / 2
                    if y < -1 or y > height or x < -1 or x > width:
                        continue
                    y, x = min(max(y, 0), height - 1), min(max(x, 0), width - 1)
                    y0, x0 = int(y), int(x)
                    y1_, x1_ = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
                    ly, lx = y - y0, x - x0
                    out[row, column] += (
                        feature[y0, x0] * (1 - ly) * (1 - lx)
                        + feature[y0, x1_] * (1 - ly) * lx
                        + feature[y1_, x0] * ly * (1 - lx)
                        + feature[y1_, x1_] * ly * lx
                    ) / 4
    return out


def test_roi_align_samples_as_torchvision_within_opencvs_grid(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    rng = np.random.default_rng(0)
    features = [
        rng.uniform(0, 1, (1344 // stride, 1344 // stride, 256)).astype(np.float32) for stride in (4, 8, 16, 32)
    ]
    # Boxes on every level, and boxes crossing the input's edges.
    boxes = np.array([[10, 20, 60, 90], [100, 100, 300, 260], [-30, 400, 250, 700], [800, 600, 1344, 1344]], np.float32)

    pooled = calibration.roi_align(features, boxes)

    levels = calibration.roi_levels(boxes)
    assert levels.tolist() == [0, 1, 2, 3]
    for box, level, actual in zip(boxes, levels, pooled, strict=True):
        expected = _reference_roi_align(features[level], box, 1 / (4 << level))
        # OpenCV places samples on a 1/32-pixel grid, so values on this unit-range map move by at most 1/32.
        np.testing.assert_allclose(actual, expected, atol=1 / 32)


def test_a_lone_proposal_decodes_to_its_anchor_clipped_to_the_image(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    rpn = [
        (np.full(size * size * 3, -20, np.float32), np.zeros((size * size * 3, 4), np.float32))
        for size in calibration.LEVEL_SIZES
    ]
    # The square 128 anchor of P4's cell at row 2, column 3, centred on (48, 32).
    rpn[2][0][(2 * 84 + 3) * 3 + 1] = 20

    boxes = calibration.proposals(rpn, (800, 1066))

    np.testing.assert_allclose(boxes[0], [0, 0, 112, 96])


def test_the_backbone_output_unpacks_into_levels_in_either_layout(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    rng = np.random.default_rng(0)
    sizes = calibration.LEVEL_SIZES
    features = [rng.standard_normal((s, s, 256)).astype(np.float32) for s in sizes[:4]]
    logits = [rng.standard_normal(s * s * 3).astype(np.float32) for s in sizes]
    deltas = [rng.standard_normal((s * s * 3, 4)).astype(np.float32) for s in sizes]
    rows = np.zeros((calibration.LEVEL_ROWS[-1], 271), np.float32)
    for i, s in enumerate(sizes):
        level = rows[calibration.LEVEL_ROWS[i] : calibration.LEVEL_ROWS[i + 1]]
        if i < 4:
            level[:, :256] = features[i].reshape(-1, 256)
        level[:, 256:259] = logits[i].reshape(-1, 3) * calibration.OBJECTNESS_GAIN
        level[:, 259:] = deltas[i].reshape(-1, 12) * calibration.DELTA_GAIN

    for output in (rows[None, None], np.ascontiguousarray(rows.T)[None, :, None]):
        unpacked_features, rpn = calibration.unpack_backbone(output)
        np.testing.assert_array_equal(unpacked_features[3], features[3])
        np.testing.assert_array_equal(rpn[4][0], logits[4])
        np.testing.assert_array_equal(rpn[1][1], deltas[1])


def test_detections_decode_the_winning_class_and_map_nothing_below_the_threshold(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    proposals = np.array([[100, 100, 200, 300], [0, 0, 50, 50]], np.float32)
    logits = np.full((2, 91), -10, np.float32)
    logits[0, 18] = 10  # COCO category 18, dog
    regression = np.zeros((2, 364), np.float32)

    boxes, scores, labels = calibration.detections(logits, regression, proposals, (800, 1066))

    assert labels.tolist() == [18] and scores[0] > 0.99
    np.testing.assert_allclose(boxes[0], proposals[0])


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"coco/{split}/{index:012d}.jpg")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4 and all(Path(name).parent.name == "cal2017" for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _mpk_manifest(compile_module: ModuleType, component_id: str) -> dict:
    contract = compile_module.COMPONENT_INTERFACES[component_id]
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": name, "size": size} for name, size in contract["inputs"].items()],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in contract["outputs"]],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("component_id", ["backbone", "box_head"])
@pytest.mark.parametrize(
    "case", ["valid", "cast", "dequantized", "resized", "two_mla", "host", "compiler", "quantized"]
)
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, component_id: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module, component_id)
    plugins = manifest["plugins"]
    if case in {"cast", "dequantized", "quantized"}:
        kernel = {
            "cast": "cast_transform",
            "dequantized": "dequantization_transform",
            "quantized": "quantization_transform",
        }
        plugins[-1]["config_params"]["kernel"] = kernel[case]
    if case == "resized":
        plugins[-1]["output_nodes"][-1]["size"] += 4
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case in {"valid", "cast", "dequantized"}:
        compile_module.audit_mpk(archive, component_id)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive, component_id)


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    assert compile_module._component_parameters("w16a16") == {
        "backbone": {"precision": "w16a16"},
        "box_head": {"precision": "w16a16"},
    }
    assert compile_module._component_parameters("w8a8") == {
        "backbone": {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 64},
        "box_head": {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 32},
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._component_parameters("w32a32")


def test_compiled_build_has_one_mpk_per_component(compiled: Path) -> None:
    for component_id in ("backbone", "box_head"):
        assert len(list((compiled / component_id).glob("*_mpk.tar.gz"))) == 1


def test_compile_runs_its_three_phases_with_their_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    started = []

    class Process:
        exitcode = 0

        def __init__(self, name: str, target: Callable, args: tuple) -> None:
            self.name, (self.phase, self.arguments) = name, args

        def start(self) -> None:
            started.append((self.name, self.phase.__name__, sorted(self.arguments)))
            if "graphs" in self.arguments:
                self.arguments["graphs"].mkdir(parents=True)
                for name in ("backbone.onnx", "box_head.onnx"):
                    (self.arguments["graphs"] / name).write_bytes(name.encode())
            build_dir = self.arguments.get("build_dir")
            if build_dir is not None:
                build_dir.mkdir(parents=True)
                (build_dir / f"{self.arguments['name']}_mpk.tar.gz").write_bytes(b"mpk")

        def join(self) -> None:
            pass

    context = type("Context", (), {"Process": Process})
    monkeypatch.setattr(compile_module.multiprocessing, "get_context", lambda method: context)

    archives = compile_module.compile_model(tmp_path / "source.onnx", tmp_path / "out", "w8a8", tmp_path / "cal.tgz")

    assert [(name, phase) for name, phase, _ in started] == [
        ("graph preparation", "_prepare_phase"),
        ("backbone compilation", "_backbone_phase"),
        ("box head compilation", "_box_head_phase"),
    ]
    assert started[2][2] == ["backbone", "build_dir", "dataset", "graph", "name", "parameters", "selected"]
    manifest = json.loads((tmp_path / "out" / "compile-outputs.json").read_text())
    assert [item["component_id"] for item in manifest["derived_graphs"]] == ["backbone", "box_head"]
    assert manifest["outputs"][1]["mpk_path"] == "box_head/faster_rcnn_resnet50_fpn_v2_w8a8_box_head_mpk.tar.gz"
    assert {component: path.parent.name for component, path in archives.items()} == {
        "backbone": "backbone",
        "box_head": "box_head",
    }
