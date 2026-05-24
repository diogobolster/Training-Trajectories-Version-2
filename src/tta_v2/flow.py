from __future__ import annotations

import numpy as np


def solve_pressure_jacobi(
    pore_mask: np.ndarray,
    *,
    axis: int = 0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    n_iters: int = 1000,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a graph-Laplace pressure field on a pore voxel network.

    This is a lightweight bootstrap solver, not a replacement for Stokes/LBM.
    Pore voxels on the inlet/outlet faces are fixed to Dirichlet pressures;
    other pore voxels relax to the average of pore-neighbor pressures.
    """
    mask = np.asarray(pore_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("pore_mask must be 3D")

    p = _initial_pressure(mask, axis, inlet_pressure, outlet_pressure)
    fixed = np.zeros_like(mask, dtype=bool)
    inlet_face = [slice(None), slice(None), slice(None)]
    outlet_face = [slice(None), slice(None), slice(None)]
    inlet_face[axis] = 0
    outlet_face[axis] = mask.shape[axis] - 1
    fixed[tuple(inlet_face)] = mask[tuple(inlet_face)]
    fixed[tuple(outlet_face)] = mask[tuple(outlet_face)]

    update = mask & ~fixed
    neighbor_count = _neighbor_count(mask)
    active_update = update & (neighbor_count > 0)

    last_delta = float("inf")
    performed = 0
    for performed in range(1, n_iters + 1):
        neighbor_sum = _neighbor_sum(p, mask)
        new_p = p.copy()
        new_p[active_update] = neighbor_sum[active_update] / neighbor_count[active_update]
        new_p[fixed & _face_mask(mask.shape, axis, 0)] = inlet_pressure
        new_p[fixed & _face_mask(mask.shape, axis, mask.shape[axis] - 1)] = outlet_pressure
        new_p[~mask] = 0.0
        last_delta = float(np.max(np.abs(new_p[active_update] - p[active_update]))) if active_update.any() else 0.0
        p = new_p
        if last_delta < tolerance:
            break

    return p, {"iterations": float(performed), "last_delta": last_delta}


def velocity_from_pressure(
    pressure: np.ndarray,
    pore_mask: np.ndarray,
    *,
    spacing: float = 1.0,
    target_mean_speed: float = 0.06,
) -> np.ndarray:
    """Compute a normalized voxel velocity from pressure gradients."""
    mask = np.asarray(pore_mask, dtype=bool)
    velocity = np.zeros(mask.shape + (3,), dtype=float)

    for axis in range(3):
        forward = _shift_with_self_boundary(pressure, mask, axis, -1)
        backward = _shift_with_self_boundary(pressure, mask, axis, 1)
        velocity[..., axis] = -(forward - backward) / (2.0 * spacing)

    velocity[~mask] = 0.0
    speed = np.linalg.norm(velocity[mask], axis=1)
    mean_speed = float(np.mean(speed[speed > 0.0])) if np.any(speed > 0.0) else 0.0
    if mean_speed > 0.0:
        velocity *= target_mean_speed / mean_speed
    return velocity


def _initial_pressure(mask: np.ndarray, axis: int, inlet_pressure: float, outlet_pressure: float) -> np.ndarray:
    coord = np.linspace(inlet_pressure, outlet_pressure, mask.shape[axis])
    shape = [1, 1, 1]
    shape[axis] = mask.shape[axis]
    p = np.broadcast_to(coord.reshape(shape), mask.shape).astype(float).copy()
    p[~mask] = 0.0
    return p


def _neighbor_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    total = np.zeros_like(values, dtype=float)
    for axis in range(3):
        total += _shift(values, mask, axis, 1)
        total += _shift(values, mask, axis, -1)
    return total


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    count = np.zeros_like(mask, dtype=int)
    ones = mask.astype(int)
    for axis in range(3):
        count += _shift(ones, mask, axis, 1).astype(int)
        count += _shift(ones, mask, axis, -1).astype(int)
    return count


def _shift(values: np.ndarray, mask: np.ndarray, axis: int, direction: int) -> np.ndarray:
    shifted_values = np.roll(values, shift=direction, axis=axis)
    shifted_mask = np.roll(mask, shift=direction, axis=axis)

    edge = [slice(None), slice(None), slice(None)]
    edge[axis] = 0 if direction > 0 else values.shape[axis] - 1
    shifted_values[tuple(edge)] = 0.0
    shifted_mask[tuple(edge)] = False

    out = np.zeros_like(values, dtype=float)
    out[shifted_mask] = shifted_values[shifted_mask]
    return out


def _shift_with_self_boundary(values: np.ndarray, mask: np.ndarray, axis: int, direction: int) -> np.ndarray:
    shifted_values = np.roll(values, shift=direction, axis=axis)
    shifted_mask = np.roll(mask, shift=direction, axis=axis)

    edge = [slice(None), slice(None), slice(None)]
    edge[axis] = 0 if direction > 0 else values.shape[axis] - 1
    shifted_mask[tuple(edge)] = False

    out = values.copy()
    out[shifted_mask] = shifted_values[shifted_mask]
    return out


def _face_mask(shape: tuple[int, int, int], axis: int, index: int) -> np.ndarray:
    face = np.zeros(shape, dtype=bool)
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = index
    face[tuple(slices)] = True
    return face

