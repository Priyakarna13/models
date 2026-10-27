"""PointNet++ SSG's CPU steps and component interface, shared by convert.py, compile.py and accuracy.py.

The network runs as eight MPKs, the four set-abstraction (SA) levels and the four feature-propagation (FP)
levels. Everything between them depends on point coordinates, which the MLA cannot gather by: `plan()` computes
farthest point sampling, ball queries and 3-NN interpolation weights for a block once, `group()` builds each SA
input, and `interpolate()` each FP input. Only NumPy is needed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

POINTS = 4096
FEATURES = 9  # xyz relative to the block centre, rgb in [0, 1], xyz divided by the room's extent
CLASSES = 13
CLASS_NAMES = (
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "chair", "table", "bookcase", "sofa", "board", "clutter",
)
NEIGHBORS = 32
SA_POINTS = (1024, 256, 64, 16)
SA_RADII = (0.1, 0.2, 0.4, 0.8)
SA_CHANNELS = ((FEATURES, (32, 32, 64)), (64, (64, 64, 128)), (128, (128, 128, 256)), (256, (256, 256, 512)))
FP_CHANNELS = ((768, (256, 256)), (384, (256, 256)), (320, (256, 128)), (128, (128, 128, 128)))
# Run order. fp4 upsamples the coarsest level; fp1 also runs the classifier head.
COMPONENTS = ("sa1", "sa2", "sa3", "sa4", "fp4", "fp3", "fp2", "fp1")
LEVEL_POINTS = (POINTS, *SA_POINTS)
INPUT_SHAPES = {
    **{f"sa{level + 1}": (1, SA_CHANNELS[level][0] + 3, SA_POINTS[level], NEIGHBORS) for level in range(4)},
    **{f"fp{4 - i}": (1, FP_CHANNELS[i][0], LEVEL_POINTS[3 - i], 1) for i in range(4)},
}
OUTPUT_SHAPES = {
    **{f"sa{level + 1}": (1, SA_CHANNELS[level][1][-1], SA_POINTS[level], 1) for level in range(4)},
    **{f"fp{4 - i}": (1, FP_CHANNELS[i][1][-1], LEVEL_POINTS[3 - i], 1) for i in range(3)},
    "fp1": (1, CLASSES, POINTS, 1),
}


def input_name(component: str) -> str:
    return f"{component}_input"


def output_name(component: str) -> str:
    return f"{component}_output"


def farthest_point_sample(pos: np.ndarray, count: int) -> np.ndarray:
    """Indices of `count` points, each the farthest from those already chosen, starting from point 0."""
    picked = np.empty(count, np.int64)
    distance = np.full(len(pos), np.inf, np.float32)
    current = 0
    for i in range(count):
        picked[i] = current
        offset = pos - pos[current]
        np.minimum(distance, np.einsum("ij,ij->i", offset, offset), out=distance)
        current = int(distance.argmax())
    return picked


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[len(a), len(b)] squared distances, summed one axis at a time without an [a, b, 3] temporary."""
    total = np.subtract.outer(a[:, 0], b[:, 0])
    total *= total
    for axis in (1, 2):
        difference = np.subtract.outer(a[:, axis], b[:, axis])
        difference *= difference
        total += difference
    return total


def ball_query(pos: np.ndarray, centres: np.ndarray, radius: float) -> np.ndarray:
    """[S, 32] indices of the lowest-indexed points within `radius` of each centre, padded with the first.

    PointNet++'s reference query keeps the lowest indices; padding with a repeat leaves the max pool unchanged.
    """
    inside = squared_distances(centres, pos) <= radius * radius
    rank = np.cumsum(inside, axis=1)
    rows, columns = np.nonzero(inside & (rank <= NEIGHBORS))
    counts = np.minimum(rank[:, -1], NEIGHBORS)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slots = np.arange(len(rows)) - np.repeat(starts, counts)
    index = np.repeat(columns[starts][:, None], NEIGHBORS, axis=1)
    index[rows, slots] = columns
    return index


def three_nearest(coarse: np.ndarray, fine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, 3] nearest coarse points of each fine point and their weights 1 / (d^2 + 1e-8), normalized."""
    nearest = np.argpartition(squared_distances(fine, coarse), 2, axis=1)[:, :3]
    offset = fine[:, None, :] - coarse[nearest]
    distance = np.einsum("nkc,nkc->nk", offset, offset)
    order = np.argsort(distance, axis=1, kind="stable")
    nearest, distance = np.take_along_axis(nearest, order, 1), np.take_along_axis(distance, order, 1)
    weight = 1.0 / (distance + 1e-8)
    return nearest, (weight / weight.sum(1, keepdims=True)).astype(np.float32)


def plan(pos: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Every coordinate-only index of one block: the points of each level, ball queries and interpolation."""
    pos = np.ascontiguousarray(pos, np.float32)
    levels, centres, groups = [pos], [], []
    for count, radius in zip(SA_POINTS, SA_RADII):
        picked = farthest_point_sample(levels[-1], count)
        groups.append(ball_query(levels[-1], levels[-1][picked], radius))
        centres.append(picked)
        levels.append(levels[-1][picked])
    interpolation = [three_nearest(levels[level + 1], levels[level]) for level in range(4)]
    return {"levels": levels, "centres": centres, "groups": groups, "interpolation": interpolation}


def group(features: np.ndarray, level: int, indices: dict[str, list[np.ndarray]]) -> np.ndarray:
    """SA input [1, C + 3, S, 32]: each neighbour's features, then its xyz minus the centre's."""
    pos, neighbours = indices["levels"][level], indices["groups"][level]
    relative = pos[neighbours] - pos[indices["centres"][level]][:, None, :]
    grouped = np.concatenate([features[neighbours], relative], axis=-1)
    return np.ascontiguousarray(grouped.transpose(2, 0, 1)[None], np.float32)


def interpolate(coarse: np.ndarray, skip: np.ndarray | None, level: int, indices: dict) -> np.ndarray:
    """FP input [1, C + C_skip, N, 1]: the inverse-distance mix of the 3 nearest coarse features, then the skip."""
    nearest, weight = indices["interpolation"][level]
    mixed = np.einsum("nkc,nk->nc", coarse[nearest], weight)
    stacked = mixed if skip is None else np.concatenate([mixed, skip], axis=1)
    return np.ascontiguousarray(stacked.T[None, :, :, None], np.float32)


def channels_last(value: np.ndarray) -> np.ndarray:
    """A component input [1, C, H, W] as the [H, W, C] tensor its MPK takes."""
    return np.ascontiguousarray(value[0].transpose(1, 2, 0))


def points_of(output: np.ndarray, channels: int, count: int) -> np.ndarray:
    """A component output as [points, channels], from NCHW [1, C, N, 1] or a channels-last layout such as Neat's
    [N, 1, C]. Every component has more than one channel, so the last axis tells the layouts apart."""
    output = np.asarray(output, np.float32)
    if output.size != channels * count:
        raise ValueError(f"expected {channels * count} values, got shape {output.shape}")
    if output.shape[-1] == channels:
        return output.reshape(count, channels)
    return output.reshape(channels, count).T


def synthetic_block(seed: int = 0) -> np.ndarray:
    """[4096, 9] features of a seeded block: a floor, two walls and a box of clutter, 1 m wide and 3 m high."""
    rng = np.random.RandomState(seed)
    parts = [
        np.c_[rng.uniform(-0.5, 0.5, (1500, 2)), rng.normal(0.0, 0.005, 1500)],
        np.c_[np.full(1000, -0.5), rng.uniform(-0.5, 0.5, 1000), rng.uniform(0, 3, 1000)],
        np.c_[rng.uniform(-0.5, 0.5, 896), np.full(896, 0.5), rng.uniform(0, 3, 896)],
        rng.uniform((-0.2, -0.2, 0.0), (0.2, 0.2, 0.8), (700, 3)),
    ]
    xyz = np.concatenate(parts)[rng.permutation(POINTS)].astype(np.float32)
    rgb = rng.uniform(0, 1, (POINTS, 3)).astype(np.float32)
    normalized = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0))
    return np.concatenate([xyz, rgb, normalized], axis=1).astype(np.float32)


def segment(x: np.ndarray, run: Callable[[str, np.ndarray], np.ndarray], indices: dict | None = None) -> np.ndarray:
    """[4096, 13] logits of one block, x being its [4096, 9] features; run(component, input) runs one MPK.
    `indices` is plan(x[:, :3]) when the caller computed it already, for example in another process."""
    if x.shape != (POINTS, FEATURES):
        raise ValueError(f"a block is [{POINTS}, {FEATURES}], got {x.shape}")
    if indices is None:
        indices = plan(x[:, :3])
    features = [np.asarray(x, np.float32)]
    for level in range(4):
        component = f"sa{level + 1}"
        output = run(component, group(features[-1], level, indices))
        features.append(points_of(output, OUTPUT_SHAPES[component][1], SA_POINTS[level]))
    upsampled = features[4]
    for level in range(3, -1, -1):
        component = f"fp{level + 1}"
        output = run(component, interpolate(upsampled, features[level] if level else None, level, indices))
        upsampled = points_of(output, OUTPUT_SHAPES[component][1], LEVEL_POINTS[level])
    return upsampled
