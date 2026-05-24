from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SegmentArchive:
    """Archive of relative trajectory segments and endpoint descriptors."""

    relative_paths: np.ndarray
    start_velocities: np.ndarray
    end_velocities: np.ndarray
    source_ids: np.ndarray
    start_indices: np.ndarray
    segment_steps: int
    match_steps: int
    stride: int
    dt: float = 1.0

    @classmethod
    def from_trajectories(
        cls,
        trajectories: Iterable[np.ndarray],
        segment_steps: int,
        match_steps: int,
        stride: int | None = None,
        dt: float = 1.0,
    ) -> "SegmentArchive":
        """Cut trajectories into overlapping relative-position segments.

        Parameters
        ----------
        trajectories:
            Iterable of arrays with shape ``(n_steps, n_dimensions)``.
        segment_steps:
            Number of increments per segment. Each stored path has
            ``segment_steps + 1`` points.
        match_steps:
            Number of increments used to define start/end velocity descriptors.
        stride:
            Window stride in increments. Defaults to ``segment_steps - match_steps``
            so neighboring windows overlap by the matching interval.
        dt:
            Time spacing between trajectory samples. Endpoint descriptors are
            stored as velocities, so increments are divided by ``dt``.
        """
        if segment_steps <= 1:
            raise ValueError("segment_steps must be greater than 1")
        if match_steps <= 0 or match_steps >= segment_steps:
            raise ValueError("match_steps must be positive and smaller than segment_steps")
        if dt <= 0:
            raise ValueError("dt must be positive")

        if stride is None:
            stride = segment_steps - match_steps
        if stride <= 0:
            raise ValueError("stride must be positive")

        paths: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        source_ids: list[int] = []
        start_indices: list[int] = []

        for source_id, trajectory in enumerate(trajectories):
            arr = np.asarray(trajectory, dtype=float)
            if arr.ndim != 2:
                raise ValueError("each trajectory must have shape (n_steps, n_dimensions)")
            if len(arr) < segment_steps + 1:
                continue

            for start_idx in range(0, len(arr) - segment_steps, stride):
                path = arr[start_idx : start_idx + segment_steps + 1]
                relative = path - path[0]
                increments = np.diff(path, axis=0) / dt
                paths.append(relative)
                starts.append(increments[:match_steps].mean(axis=0))
                ends.append(increments[-match_steps:].mean(axis=0))
                source_ids.append(source_id)
                start_indices.append(start_idx)

        if not paths:
            raise ValueError("no segments were produced; check trajectory lengths and segment_steps")

        return cls(
            relative_paths=np.stack(paths),
            start_velocities=np.stack(starts),
            end_velocities=np.stack(ends),
            source_ids=np.asarray(source_ids, dtype=int),
            start_indices=np.asarray(start_indices, dtype=int),
            segment_steps=segment_steps,
            match_steps=match_steps,
            stride=stride,
            dt=float(dt),
        )

    @property
    def size(self) -> int:
        return int(self.relative_paths.shape[0])

    @property
    def n_dimensions(self) -> int:
        return int(self.relative_paths.shape[2])

    def random_index(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.size))

    def true_successor_pairs(self) -> np.ndarray:
        """Return archive index pairs that are adjacent in the source trajectories."""
        lookup = {
            (int(source_id), int(start_idx)): idx
            for idx, (source_id, start_idx) in enumerate(zip(self.source_ids, self.start_indices))
        }
        pairs: list[tuple[int, int]] = []
        for idx, (source_id, start_idx) in enumerate(zip(self.source_ids, self.start_indices)):
            next_idx = lookup.get((int(source_id), int(start_idx + self.stride)))
            if next_idx is not None:
                pairs.append((idx, next_idx))
        return np.asarray(pairs, dtype=int)
