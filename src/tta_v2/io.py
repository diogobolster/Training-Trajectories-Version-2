from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def load_trajectories(
    path: str | Path,
    *,
    key: str | None = None,
    id_column: str | None = None,
    coordinate_columns: Sequence[str] | None = None,
    delimiter: str = ",",
) -> list[np.ndarray]:
    """Load particle trajectories from ``.npy``, ``.npz``, or ``.csv`` files.

    Supported array layouts:

    - ``(n_particles, n_steps, d)``: one trajectory per first-axis entry.
    - ``(n_steps, d)``: one trajectory.
    - object arrays containing variable-length ``(n_steps, d)`` arrays.

    Supported CSV layout:

    - header with particle id column such as ``particle_id`` or
      ``trajectory_id`` and coordinate columns ``x,y`` or ``x,y,z``.
    - optional time column named ``time``, ``t``, or ``step`` for sorting.
    - no header: interpreted as one trajectory with columns ``x,y[,z]``.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return _array_to_trajectories(np.load(file_path, allow_pickle=True))
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=True) as data:
            chosen_key = key or _default_npz_key(data.files)
            return _array_to_trajectories(data[chosen_key])
    if suffix == ".csv":
        return _load_csv_trajectories(
            file_path,
            id_column=id_column,
            coordinate_columns=coordinate_columns,
            delimiter=delimiter,
        )
    raise ValueError(f"unsupported trajectory format: {suffix}")


def save_trajectories_npz(path: str | Path, trajectories: list[np.ndarray], key: str = "trajectories") -> None:
    """Save trajectories as a compressed object array for variable lengths."""
    arr = np.empty(len(trajectories), dtype=object)
    for idx, trajectory in enumerate(trajectories):
        arr[idx] = np.asarray(trajectory, dtype=float)
    np.savez_compressed(path, **{key: arr})


def _default_npz_key(keys: list[str]) -> str:
    if "trajectories" in keys:
        return "trajectories"
    if len(keys) == 1:
        return keys[0]
    raise ValueError(f"npz contains multiple arrays; pass --key. Available keys: {keys}")


def _array_to_trajectories(arr: np.ndarray) -> list[np.ndarray]:
    if arr.dtype == object:
        return [_validate_trajectory(np.asarray(item, dtype=float)) for item in arr]
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return [_validate_trajectory(arr)]
    if arr.ndim == 3:
        return [_validate_trajectory(arr[idx]) for idx in range(arr.shape[0])]
    raise ValueError("trajectory arrays must have shape (n_steps, d) or (n_particles, n_steps, d)")


def _validate_trajectory(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("each trajectory must have shape (n_steps, n_dimensions)")
    if arr.shape[0] < 2:
        raise ValueError("each trajectory must contain at least two points")
    if arr.shape[1] < 2:
        raise ValueError("each trajectory must contain at least two spatial dimensions")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectories must contain only finite numeric values")
    return arr


def _load_csv_trajectories(
    path: Path,
    *,
    id_column: str | None,
    coordinate_columns: Sequence[str] | None,
    delimiter: str,
) -> list[np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if not has_header:
            arr = np.loadtxt(handle, delimiter=delimiter)
            if arr.ndim == 1:
                arr = arr[None, :]
            return [_validate_trajectory(arr)]

        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no field names")
        fields = list(reader.fieldnames)
        coords = list(coordinate_columns) if coordinate_columns is not None else _infer_coordinate_columns(fields)
        particle_col = id_column or _infer_id_column(fields)
        time_col = _infer_time_column(fields)

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            particle_id = row[particle_col] if particle_col else "0"
            grouped.setdefault(particle_id, []).append(row)

    trajectories: list[np.ndarray] = []
    for rows in grouped.values():
        if time_col:
            rows = sorted(rows, key=lambda row: float(row[time_col]))
        values = [[float(row[col]) for col in coords] for row in rows]
        trajectories.append(_validate_trajectory(np.asarray(values, dtype=float)))
    return trajectories


def _infer_coordinate_columns(fields: Sequence[str]) -> list[str]:
    lower_to_field = {field.lower(): field for field in fields}
    if {"x", "y", "z"}.issubset(lower_to_field):
        return [lower_to_field["x"], lower_to_field["y"], lower_to_field["z"]]
    if {"x", "y"}.issubset(lower_to_field):
        return [lower_to_field["x"], lower_to_field["y"]]
    raise ValueError("could not infer coordinate columns; pass coordinate_columns")


def _infer_id_column(fields: Sequence[str]) -> str | None:
    lower_to_field = {field.lower(): field for field in fields}
    for candidate in ("particle_id", "trajectory_id", "traj_id", "id"):
        if candidate in lower_to_field:
            return lower_to_field[candidate]
    return None


def _infer_time_column(fields: Sequence[str]) -> str | None:
    lower_to_field = {field.lower(): field for field in fields}
    for candidate in ("time", "t", "step"):
        if candidate in lower_to_field:
            return lower_to_field[candidate]
    return None
