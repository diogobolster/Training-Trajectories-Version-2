from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


def load_raw_volume(
    path: str | Path,
    shape: tuple[int, int, int],
    dtype: str = "<u2",
    order: str = "C",
) -> np.ndarray:
    """Load a cubic or rectangular raw CT volume."""
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"expected {expected} voxels for shape {shape}, found {arr.size}")
    return arr.reshape(shape, order=order)


def block_average(volume: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 3D volume by averaging non-overlapping cubic blocks."""
    if factor <= 1:
        return np.asarray(volume)
    if any(size % factor != 0 for size in volume.shape):
        raise ValueError(f"volume shape {volume.shape} is not divisible by factor {factor}")
    new_shape = (
        volume.shape[0] // factor,
        factor,
        volume.shape[1] // factor,
        factor,
        volume.shape[2] // factor,
        factor,
    )
    return volume.reshape(new_shape).mean(axis=(1, 3, 5))


def otsu_threshold(volume: np.ndarray, bins: int = 512) -> float:
    """Return an Otsu threshold for grayscale segmentation."""
    values = np.asarray(volume).ravel()
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    weights_total = hist.sum()
    if weights_total == 0:
        raise ValueError("cannot threshold an empty volume")

    weight_bg = np.cumsum(hist)
    weight_fg = weights_total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1)
    mean_fg = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]

    variance = weight_bg[:-1] * weight_fg[:-1] * (mean_bg[:-1] - mean_fg[1:]) ** 2
    idx = int(np.argmax(variance))
    return float(centers[idx])


def segment_pore_space(
    volume: np.ndarray,
    threshold: float | None = None,
    pore_is_dark: bool = True,
) -> np.ndarray:
    """Segment pores from a grayscale CT image."""
    if threshold is None:
        threshold = otsu_threshold(volume)
    if pore_is_dark:
        return np.asarray(volume) <= threshold
    return np.asarray(volume) >= threshold


def connected_pore_network(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """Keep pores connected to both inlet and outlet faces."""
    inlet = _connected_from_face(mask, axis=axis, face_index=0)
    outlet = _connected_from_face(mask, axis=axis, face_index=mask.shape[axis] - 1)
    return inlet & outlet


def porosity(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def _connected_from_face(mask: np.ndarray, axis: int, face_index: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    connected = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()

    face = [slice(None), slice(None), slice(None)]
    face[axis] = face_index
    face_coords = np.argwhere(mask[tuple(face)])

    for coord2 in face_coords:
        coord = [0, 0, 0]
        coord[axis] = face_index
        other_axes = [idx for idx in range(3) if idx != axis]
        coord[other_axes[0]] = int(coord2[0])
        coord[other_axes[1]] = int(coord2[1])
        item = tuple(coord)
        connected[item] = True
        queue.append(item)

    shape = mask.shape
    while queue:
        i, j, k = queue.popleft()
        for ni, nj, nk in (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        ):
            if ni < 0 or nj < 0 or nk < 0:
                continue
            if ni >= shape[0] or nj >= shape[1] or nk >= shape[2]:
                continue
            if mask[ni, nj, nk] and not connected[ni, nj, nk]:
                connected[ni, nj, nk] = True
                queue.append((ni, nj, nk))

    return connected
