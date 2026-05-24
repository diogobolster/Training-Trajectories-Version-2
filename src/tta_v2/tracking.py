from __future__ import annotations

import numpy as np


def trace_particles(
    velocity: np.ndarray,
    pore_mask: np.ndarray,
    *,
    n_particles: int = 500,
    n_steps: int = 800,
    dt: float = 0.5,
    diffusivity: float = 0.001,
    axis: int = 0,
    seed: int | None = 1,
    max_advective_step: float | None = None,
    max_diffusive_step: float | None = None,
    max_substeps: int = 128,
    diagnostics: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Trace advective-diffusive particles through a voxel velocity field."""
    rng = np.random.default_rng(seed)
    mask = np.asarray(pore_mask, dtype=bool)
    if velocity.shape[:3] != mask.shape or velocity.shape[3] != 3:
        raise ValueError("velocity must have shape pore_mask.shape + (3,)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if diffusivity < 0.0:
        raise ValueError("diffusivity must be non-negative")
    if max_advective_step is not None and max_advective_step <= 0.0:
        raise ValueError("max_advective_step must be positive when provided")
    if max_diffusive_step is not None and max_diffusive_step <= 0.0:
        raise ValueError("max_diffusive_step must be positive when provided")
    if max_substeps <= 0:
        raise ValueError("max_substeps must be positive")

    inlet_cells = _face_cells(mask, axis=axis, index=0)
    if len(inlet_cells) == 0:
        raise ValueError("no pore cells on inlet face")

    stats = {
        "particles_started": 0,
        "output_steps_attempted": 0,
        "output_steps_completed": 0,
        "substeps": 0,
        "max_substeps_in_output_step": 0,
        "substep_cap_hits": 0,
        "diffusive_rejections": 0,
        "advective_fallback_accepts": 0,
        "immobile_rejections": 0,
    }
    trajectories: list[np.ndarray] = []
    for _ in range(n_particles):
        stats["particles_started"] += 1
        cell = inlet_cells[int(rng.integers(0, len(inlet_cells)))]
        pos = cell.astype(float) + rng.uniform(0.1, 0.9, size=3)
        path = [pos.copy()]

        for _step in range(n_steps):
            stats["output_steps_attempted"] += 1
            pos, valid, substeps_used = _advance_particle(
                pos,
                velocity,
                mask,
                dt=dt,
                diffusivity=diffusivity,
                rng=rng,
                max_advective_step=max_advective_step,
                max_diffusive_step=max_diffusive_step,
                max_substeps=max_substeps,
                stats=stats,
            )
            stats["max_substeps_in_output_step"] = max(
                stats["max_substeps_in_output_step"],
                substeps_used,
            )
            if not valid:
                break
            stats["output_steps_completed"] += 1
            path.append(pos.copy())
            if pos[axis] >= mask.shape[axis] - 1.01:
                break

        if len(path) > 2:
            trajectories.append(np.asarray(path, dtype=float))

    if diagnostics is not None:
        diagnostics.update({key: float(value) for key, value in stats.items()})
        if stats["output_steps_completed"] > 0:
            diagnostics["mean_substeps_per_output_step"] = (
                stats["substeps"] / stats["output_steps_completed"]
            )
        else:
            diagnostics["mean_substeps_per_output_step"] = 0.0
    return trajectories


def _advance_particle(
    pos: np.ndarray,
    velocity: np.ndarray,
    mask: np.ndarray,
    *,
    dt: float,
    diffusivity: float,
    rng: np.random.Generator,
    max_advective_step: float | None,
    max_diffusive_step: float | None,
    max_substeps: int,
    stats: dict[str, float],
) -> tuple[np.ndarray, bool, int]:
    remaining = dt
    substeps = 0
    valid = True
    eps = 1e-14

    while remaining > eps:
        idx = np.clip(np.floor(pos).astype(int), 0, np.asarray(mask.shape) - 1)
        if not mask[tuple(idx)]:
            valid = False
            break

        drift = velocity[tuple(idx)]
        sub_dt = remaining
        speed = float(np.linalg.norm(drift))
        if max_advective_step is not None and speed > 0.0:
            sub_dt = min(sub_dt, max_advective_step / speed)
        if max_diffusive_step is not None and diffusivity > 0.0:
            sub_dt = min(sub_dt, max_diffusive_step**2 / (6.0 * diffusivity))

        if substeps >= max_substeps - 1 and remaining - sub_dt > eps:
            sub_dt = remaining
            stats["substep_cap_hits"] += 1

        noise = np.sqrt(2.0 * diffusivity * sub_dt) * rng.normal(size=3)
        candidate = _clip_position(pos + sub_dt * drift + noise, mask.shape)
        candidate_idx = np.clip(np.floor(candidate).astype(int), 0, np.asarray(mask.shape) - 1)

        if mask[tuple(candidate_idx)]:
            pos = candidate
        else:
            stats["diffusive_rejections"] += 1
            advective_candidate = _clip_position(pos + sub_dt * drift, mask.shape)
            advective_idx = np.clip(np.floor(advective_candidate).astype(int), 0, np.asarray(mask.shape) - 1)
            if mask[tuple(advective_idx)]:
                pos = advective_candidate
                stats["advective_fallback_accepts"] += 1
            else:
                stats["immobile_rejections"] += 1

        remaining -= sub_dt
        substeps += 1
        stats["substeps"] += 1

    return pos, valid, substeps


def _face_cells(mask: np.ndarray, axis: int, index: int) -> np.ndarray:
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = index
    coords = np.argwhere(mask[tuple(slices)])
    cells = np.zeros((len(coords), 3), dtype=int)
    cells[:, axis] = index
    other_axes = [idx for idx in range(3) if idx != axis]
    if len(coords):
        cells[:, other_axes[0]] = coords[:, 0]
        cells[:, other_axes[1]] = coords[:, 1]
    return cells


def _clip_position(position: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    upper = np.asarray(shape, dtype=float) - 1e-6
    return np.minimum(np.maximum(position, 0.0), upper)
