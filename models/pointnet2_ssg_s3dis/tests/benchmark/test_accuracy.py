"""Checks of accuracy.py's dataset loading, block protocols and scoring."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

S3DIS_ORDER = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


@pytest.fixture
def accuracy(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def _room(size_x: float, size_y: float, points: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    xyz = (rng.uniform(0, 1, (points, 3)) * [size_x, size_y, 3]).astype(np.float32)
    return {"xyz": xyz, "rgb": rng.randint(0, 256, (points, 3)).astype(np.uint8),
            "label": rng.randint(0, 13, points).astype(np.uint8)}


def _write_dataset(directory: Path, rooms: dict[str, dict[str, np.ndarray]], classes=S3DIS_ORDER, **manifest) -> Path:
    """An s3dis val archive's layout: val/<area>/<room>/{xyz,rgb}.npy and annotations/<area>/<room>.npy."""
    directory.mkdir()
    for name, room in rooms.items():
        (directory / "val" / name).mkdir(parents=True)
        np.save(directory / "val" / name / "xyz.npy", room["xyz"])
        np.save(directory / "val" / name / "rgb.npy", room["rgb"])
        (directory / "annotations" / name).parent.mkdir(parents=True, exist_ok=True)
        np.save(directory / "annotations" / f"{name}.npy", room["label"])
    (directory / "annotations" / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    content = {"id": "s3dis", "variant": "val", "rooms": list(rooms)}
    (directory / "manifest.json").write_text(json.dumps({**content, **manifest}), encoding="utf-8")
    return directory


def test_labels_are_renumbered_to_the_models_class_order(tmp_path: Path, accuracy: ModuleType) -> None:
    room = _room(2, 2, 100)
    room["label"] = np.array([S3DIS_ORDER.index(name) for name in ("table", "chair", "sofa", "bookcase")] * 25,
                             np.uint8)
    (name, loaded), = accuracy.load_rooms(_write_dataset(tmp_path / "s3dis-val", {"Area_5/office_1": room}))
    assert name == "Area_5/office_1"
    assert [accuracy.CLASS_NAMES[i] for i in loaded["label"][:4]] == ["table", "chair", "sofa", "bookcase"]


@pytest.mark.parametrize("case", ["area", "classes", "variant"])
def test_other_datasets_are_refused(tmp_path: Path, case: str, accuracy: ModuleType) -> None:
    room = "Area_1/office_1" if case == "area" else "Area_5/office_1"
    classes = S3DIS_ORDER[:12] if case == "classes" else S3DIS_ORDER
    changes = {"variant": "cal"} if case == "variant" else {}
    directory = _write_dataset(tmp_path / "s3dis-val", {room: _room(2, 2, 10)}, classes, **changes)
    with pytest.raises(SystemExit):
        accuracy.load_rooms(directory)


def test_tiles_give_one_block_per_occupied_metre(accuracy: ModuleType) -> None:
    room = _room(2.5, 1.5, 20000)
    blocks = list(accuracy.blocks(room, np.random.RandomState(0), "tiles"))
    # Columns start every metre and the last one ends at the room's edge: 3 x 2 of them.
    assert len(blocks) == 6
    for features, points in blocks:
        assert features.shape == (accuracy.POINTS, accuracy.FEATURES) and features.dtype == np.float32
        assert len(np.unique(points)) == accuracy.POINTS  # enough points: no repeats
        # x, y centred on the 1 m column; rgb in [0, 1]; normalized xyz in [0, 1].
        assert np.abs(features[:, :2]).max() <= 0.5 + accuracy.PADDING + 1e-6
        assert 0 <= features[:, 3:].min() and features[:, 3:].max() <= 1


def test_small_columns_repeat_points_to_fill_a_block(accuracy: ModuleType) -> None:
    (features, points), = list(accuracy.blocks(_room(0.9, 0.9, 100), np.random.RandomState(0), "tiles"))
    assert len(points) == accuracy.POINTS and set(points.tolist()) == set(range(100))


def test_whole_protocol_covers_every_point(accuracy: ModuleType) -> None:
    room = _room(1.6, 1.2, 9000)
    points = np.concatenate([p for _, p in accuracy.blocks(room, np.random.RandomState(0), "whole")])
    assert set(points.tolist()) == set(range(9000))
    assert len(points) % accuracy.POINTS == 0


def test_scores_follow_their_definitions(accuracy: ModuleType) -> None:
    confusion = np.zeros((13, 13), np.int64)
    confusion[0, 0], confusion[0, 1], confusion[1, 1] = 3, 1, 4
    result = accuracy.scores(confusion)
    # IoU: class 0 is 3 / (4 + 3 - 3), class 1 is 4 / (4 + 5 - 4); absent classes are left out.
    assert result["miou"] == pytest.approx((3 / 4 + 4 / 5) / 2)
    assert result["overall_accuracy"] == pytest.approx(7 / 8)
    assert result["mean_class_accuracy"] == pytest.approx((3 / 4 + 1) / 2)


def test_evaluate_scores_only_the_points_a_block_holds(accuracy: ModuleType) -> None:
    room = _room(2, 2, 20000)
    room["label"][:] = 2  # every point is a wall

    def walls(features):
        logits = np.zeros((len(features), 13), np.float32)
        logits[:, 2] = 1
        return logits

    metrics, per_class, count = accuracy.evaluate([("office_1", room)], walls)
    assert count == 4 and metrics["overall_accuracy"] == 1.0 and per_class[2] == 1.0
