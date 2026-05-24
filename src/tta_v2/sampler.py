from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .segments import SegmentArchive


@dataclass
class SegmentSampler:
    """Base class for cut-copy-paste trajectory samplers."""

    archive: SegmentArchive
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample_next_index(self, end_velocity: np.ndarray, current_index: int | None = None) -> int:
        candidate_ids, weights = self.transition_weights(end_velocity, current_index=current_index)
        return int(self.rng.choice(candidate_ids, p=weights))

    def generate(
        self,
        n_segments: int,
        start_index: int | None = None,
        origin: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate one trajectory by conditionally pasting archive segments."""
        if n_segments <= 0:
            raise ValueError("n_segments must be positive")

        if start_index is None:
            start_index = self.archive.random_index(self.rng)
        if origin is None:
            origin = np.zeros(self.archive.n_dimensions, dtype=float)
        else:
            origin = np.asarray(origin, dtype=float)

        first = self.archive.relative_paths[start_index] + origin[None, :]
        pieces = [first]
        current_endpoint = first[-1]
        current_end_velocity = self.archive.end_velocities[start_index]
        current_index = start_index

        overlap = self.archive.match_steps
        for _ in range(n_segments - 1):
            next_index = self.sample_next_index(current_end_velocity, current_index=current_index)
            candidate = self.archive.relative_paths[next_index]

            shifted_tail = candidate[overlap:] - candidate[overlap] + current_endpoint
            pieces.append(shifted_tail[1:])
            current_endpoint = pieces[-1][-1]
            current_end_velocity = self.archive.end_velocities[next_index]
            current_index = next_index

        return np.concatenate(pieces, axis=0)

    def generate_many(self, n_trajectories: int, n_segments: int) -> list[np.ndarray]:
        return [self.generate(n_segments=n_segments) for _ in range(n_trajectories)]


@dataclass
class UnconditionalSegmentSampler(SegmentSampler):
    """Random segment shuffling baseline."""

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_ids = np.arange(self.archive.size)
        weights = np.full(self.archive.size, 1.0 / self.archive.size)
        return candidate_ids, weights


@dataclass
class MixtureSegmentSampler(SegmentSampler):
    """Mixture of component transition distributions."""

    components: Sequence[SegmentSampler] = ()
    component_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.components:
            raise ValueError("at least one component sampler is required")
        for component in self.components:
            if component.archive is not self.archive:
                raise ValueError("all component samplers must share the same archive object")
        if self.component_weights is None:
            self.component_weights = np.full(len(self.components), 1.0 / len(self.components))
        else:
            self.component_weights = np.asarray(self.component_weights, dtype=float)
            if self.component_weights.shape != (len(self.components),):
                raise ValueError("component_weights must match number of components")
            if np.any(self.component_weights < 0):
                raise ValueError("component_weights must be nonnegative")
            total = float(np.sum(self.component_weights))
            if total <= 0:
                raise ValueError("at least one component weight must be positive")
            self.component_weights = self.component_weights / total

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        for component_weight, component in zip(self.component_weights, self.components):
            if component_weight <= 0.0:
                continue
            candidate_ids, weights = component.transition_weights(
                end_velocity,
                current_index=current_index,
            )
            candidate_chunks.append(candidate_ids)
            weight_chunks.append(component_weight * weights)
        if not candidate_chunks:
            candidate_ids = np.arange(self.archive.size)
            weights = np.full(self.archive.size, 1.0 / self.archive.size)
            return candidate_ids, weights

        all_candidates = np.concatenate(candidate_chunks)
        all_weights = np.concatenate(weight_chunks)
        candidate_ids, inverse = np.unique(all_candidates, return_inverse=True)
        combined_weights = np.zeros(len(candidate_ids), dtype=float)
        np.add.at(combined_weights, inverse, all_weights)

        total = float(np.sum(combined_weights))
        if total <= 0 or not np.isfinite(total):
            candidate_ids = np.arange(self.archive.size)
            weights = np.full(self.archive.size, 1.0 / self.archive.size)
        else:
            positive = combined_weights > 0.0
            candidate_ids = candidate_ids[positive]
            weights = combined_weights[positive] / np.sum(combined_weights[positive])
        return candidate_ids, weights


@dataclass
class ConditionalSegmentSampler(SegmentSampler):
    """kNN conditional sampler in endpoint-velocity space.

    This is a deliberately simple TTA-v2 baseline. It samples candidate segments
    by matching the current segment's ending velocity to archive start velocities.
    The later neural version can replace ``transition_weights`` without changing
    the rest of the assembly logic.
    """

    k: int = 64
    temperature: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate indices and normalized transition weights."""
        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)
        k = min(self.k, self.archive.size)
        candidate_ids = np.argpartition(distances, kth=k - 1)[:k]
        candidate_distances = distances[candidate_ids]

        scale = np.median(candidate_distances[candidate_distances > 0.0])
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(candidate_distances.mean() + 1e-12)

        logits = -candidate_distances / (self.temperature * scale + 1e-12)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        return candidate_ids, weights


@dataclass
class GaussianBayesSegmentSampler(SegmentSampler):
    """Original TTA-style Gaussian transition kernel.

    The 2019 paper conditions candidate segment starts on the current segment
    end by asking whether the velocity mismatch is plausibly explained by
    diffusion over the matching interval. In sampled data, a diffusive
    displacement has standard deviation ``sqrt(2 D tau)`` per coordinate, so
    the average-velocity tolerance is ``sqrt(2 D tau) / tau``.
    """

    diffusivity: float = 1e-4
    candidate_limit: int | None = None
    bandwidth_multiplier: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diffusivity <= 0:
            raise ValueError("diffusivity must be positive")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided")
        if self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be positive")

    @property
    def sigma_velocity(self) -> float:
        tau = self.archive.match_steps * self.archive.dt
        return self.bandwidth_multiplier * np.sqrt(2.0 * self.diffusivity * tau) / tau

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)

        if self.candidate_limit is None or self.candidate_limit >= self.archive.size:
            candidate_ids = np.arange(self.archive.size)
            candidate_distances = distances
        else:
            k = min(self.candidate_limit, self.archive.size)
            candidate_ids = np.argpartition(distances, kth=k - 1)[:k]
            candidate_distances = distances[candidate_ids]

        sigma = max(float(self.sigma_velocity), 1e-12)
        logits = -0.5 * candidate_distances / (sigma * sigma)
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
        else:
            weights /= total
        return candidate_ids, weights


@dataclass
class AdaptiveGaussianBayesSegmentSampler(SegmentSampler):
    """Gaussian/Bayes sampler with speed-state-dependent bandwidths."""

    speed_edges: np.ndarray | None = None
    sigma_by_bin: np.ndarray | None = None
    candidate_limit: int | None = 256
    bandwidth_multiplier: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.speed_edges is None or self.sigma_by_bin is None:
            raise ValueError("use AdaptiveGaussianBayesSegmentSampler.fit(...) to estimate bandwidths")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided")
        if self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be positive")

    @classmethod
    def fit(
        cls,
        archive: SegmentArchive,
        *,
        n_bins: int = 4,
        density_k: int = 32,
        max_reference_segments: int | None = 6000,
        min_sigma: float = 1e-6,
        bandwidth_multiplier: float = 1.0,
        candidate_limit: int | None = 256,
        seed: int | None = None,
    ) -> "AdaptiveGaussianBayesSegmentSampler":
        if density_k <= 0:
            raise ValueError("density_k must be positive")
        rng = np.random.default_rng(seed)
        reference_ids = np.arange(archive.size)
        if max_reference_segments is not None and len(reference_ids) > max_reference_segments:
            reference_ids = rng.choice(reference_ids, size=max_reference_segments, replace=False)

        current_speed = np.linalg.norm(archive.end_velocities[reference_ids], axis=1)
        local_sigma = np.empty(len(reference_ids), dtype=float)
        for local_idx, archive_idx in enumerate(reference_ids):
            delta = archive.start_velocities - archive.end_velocities[archive_idx][None, :]
            distances = np.sum(delta * delta, axis=1)
            positive = distances[distances > 1e-18]
            if len(positive) == 0:
                local_sigma[local_idx] = min_sigma
                continue
            k = min(density_k, len(positive))
            nearest = np.partition(positive, kth=k - 1)[:k]
            local_sigma[local_idx] = max(
                float(np.sqrt(np.median(nearest) / archive.n_dimensions)),
                min_sigma,
            )

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        speed_edges = np.quantile(current_speed, quantiles)
        speed_edges[0] = -np.inf
        speed_edges[-1] = np.inf

        global_sigma = float(np.median(local_sigma))
        sigma_by_bin = np.empty(n_bins, dtype=float)
        for bin_id in range(n_bins):
            in_bin = (current_speed >= speed_edges[bin_id]) & (current_speed <= speed_edges[bin_id + 1])
            if np.any(in_bin):
                sigma = float(np.median(local_sigma[in_bin]))
            else:
                sigma = global_sigma
            sigma_by_bin[bin_id] = max(sigma, min_sigma)

        return cls(
            archive=archive,
            seed=seed,
            speed_edges=speed_edges,
            sigma_by_bin=sigma_by_bin,
            candidate_limit=candidate_limit,
            bandwidth_multiplier=bandwidth_multiplier,
        )

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)
        if self.candidate_limit is None or self.candidate_limit >= self.archive.size:
            candidate_ids = np.arange(self.archive.size)
            candidate_distances = distances
        else:
            k = min(self.candidate_limit, self.archive.size)
            candidate_ids = np.argpartition(distances, kth=k - 1)[:k]
            candidate_distances = distances[candidate_ids]

        speed = float(np.linalg.norm(end_velocity))
        bin_id = int(np.searchsorted(self.speed_edges[1:-1], speed, side="right"))
        sigma = max(float(self.sigma_by_bin[bin_id] * self.bandwidth_multiplier), 1e-12)
        logits = -0.5 * candidate_distances / (sigma * sigma)
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
        else:
            weights /= total
        return candidate_ids, weights


@dataclass
class ShortHorizonRerankGaussianSampler(SegmentSampler):
    """Gaussian seam sampler reranked by short-horizon archive consequences."""

    speed_edges: np.ndarray | None = None
    target_by_bin: np.ndarray | None = None
    scale_by_bin: np.ndarray | None = None
    horizon_descriptors: np.ndarray | None = None
    diffusivity: float = 1e-4
    bandwidth_multiplier: float = 1.0
    candidate_limit: int | None = 256
    horizon_weight: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.speed_edges is None
            or self.target_by_bin is None
            or self.scale_by_bin is None
            or self.horizon_descriptors is None
        ):
            raise ValueError("use ShortHorizonRerankGaussianSampler.fit(...) to estimate horizon targets")
        if self.diffusivity <= 0:
            raise ValueError("diffusivity must be positive")
        if self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be positive")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided")

    @classmethod
    def fit(
        cls,
        archive: SegmentArchive,
        *,
        seed: int | None = None,
        n_bins: int = 4,
        horizon_segments: int = 3,
        diffusivity: float = 1e-4,
        bandwidth_multiplier: float = 1.0,
        candidate_limit: int | None = 256,
        horizon_weight: float = 0.25,
        min_scale: float = 1e-6,
    ) -> "ShortHorizonRerankGaussianSampler":
        pairs = archive.true_successor_pairs()
        if len(pairs) == 0:
            raise ValueError("archive has no true successor pairs; check stride/source metadata")
        horizon_descriptors = archive_horizon_descriptors(archive, horizon_segments=horizon_segments)

        current_speed = np.linalg.norm(archive.end_velocities[pairs[:, 0]], axis=1)
        true_future = horizon_descriptors[pairs[:, 1]]
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        speed_edges = np.quantile(current_speed, quantiles)
        speed_edges[0] = -np.inf
        speed_edges[-1] = np.inf

        global_target = np.median(true_future, axis=0)
        global_scale = np.std(true_future, axis=0)
        global_scale[global_scale < min_scale] = min_scale
        target_by_bin = np.empty((n_bins, true_future.shape[1]), dtype=float)
        scale_by_bin = np.empty_like(target_by_bin)
        for bin_id in range(n_bins):
            in_bin = (current_speed >= speed_edges[bin_id]) & (current_speed <= speed_edges[bin_id + 1])
            if np.any(in_bin):
                target = np.median(true_future[in_bin], axis=0)
                scale = np.std(true_future[in_bin], axis=0)
                scale[scale < min_scale] = min_scale
            else:
                target = global_target
                scale = global_scale
            target_by_bin[bin_id] = target
            scale_by_bin[bin_id] = scale

        return cls(
            archive=archive,
            seed=seed,
            speed_edges=speed_edges,
            target_by_bin=target_by_bin,
            scale_by_bin=scale_by_bin,
            horizon_descriptors=horizon_descriptors,
            diffusivity=diffusivity,
            bandwidth_multiplier=bandwidth_multiplier,
            candidate_limit=candidate_limit,
            horizon_weight=horizon_weight,
        )

    @property
    def sigma_velocity(self) -> float:
        tau = self.archive.match_steps * self.archive.dt
        return self.bandwidth_multiplier * np.sqrt(2.0 * self.diffusivity * tau) / tau

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)
        if self.candidate_limit is None or self.candidate_limit >= self.archive.size:
            candidate_ids = np.arange(self.archive.size)
            candidate_distances = distances
        else:
            k = min(self.candidate_limit, self.archive.size)
            candidate_ids = np.argpartition(distances, kth=k - 1)[:k]
            candidate_distances = distances[candidate_ids]

        sigma = max(float(self.sigma_velocity), 1e-12)
        gaussian_logits = -0.5 * candidate_distances / (sigma * sigma)

        speed = float(np.linalg.norm(end_velocity))
        bin_id = int(np.searchsorted(self.speed_edges[1:-1], speed, side="right"))
        target = self.target_by_bin[bin_id]
        scale = self.scale_by_bin[bin_id]
        z = (self.horizon_descriptors[candidate_ids] - target[None, :]) / scale[None, :]
        horizon_logits = -0.5 * np.sum(z * z, axis=1)

        logits = gaussian_logits + self.horizon_weight * horizon_logits
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
        else:
            weights /= total
        return candidate_ids, weights


@dataclass
class PairAwareRerankGaussianSampler(SegmentSampler):
    """Gaussian seam sampler reranked by archive-level pair behavior."""

    speed_edges: np.ndarray | None = None
    target_by_bin: np.ndarray | None = None
    scale_by_bin: np.ndarray | None = None
    pair_descriptors: np.ndarray | None = None
    diffusivity: float = 1e-4
    bandwidth_multiplier: float = 1.0
    candidate_limit: int | None = 256
    pair_weight: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.speed_edges is None
            or self.target_by_bin is None
            or self.scale_by_bin is None
            or self.pair_descriptors is None
        ):
            raise ValueError("use PairAwareRerankGaussianSampler.fit(...) to estimate pair targets")
        if self.diffusivity <= 0:
            raise ValueError("diffusivity must be positive")
        if self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be positive")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided")
        if self.pair_weight < 0:
            raise ValueError("pair_weight must be nonnegative")

    @classmethod
    def fit(
        cls,
        archive: SegmentArchive,
        *,
        seed: int | None = None,
        n_bins: int = 4,
        horizon_segments: int = 3,
        neighbor_k: int = 32,
        diffusivity: float = 1e-4,
        bandwidth_multiplier: float = 1.0,
        candidate_limit: int | None = 256,
        pair_weight: float = 0.25,
        min_scale: float = 1e-6,
    ) -> "PairAwareRerankGaussianSampler":
        pairs = archive.true_successor_pairs()
        if len(pairs) == 0:
            raise ValueError("archive has no true successor pairs; check stride/source metadata")
        pair_descriptors = archive_pair_behavior_descriptors(
            archive,
            horizon_segments=horizon_segments,
            neighbor_k=neighbor_k,
        )

        current_speed = np.linalg.norm(archive.end_velocities[pairs[:, 0]], axis=1)
        true_pair_behavior = pair_descriptors[pairs[:, 1]]
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        speed_edges = np.quantile(current_speed, quantiles)
        speed_edges[0] = -np.inf
        speed_edges[-1] = np.inf

        global_target = np.median(true_pair_behavior, axis=0)
        global_scale = np.std(true_pair_behavior, axis=0)
        global_scale[global_scale < min_scale] = min_scale
        target_by_bin = np.empty((n_bins, true_pair_behavior.shape[1]), dtype=float)
        scale_by_bin = np.empty_like(target_by_bin)
        for bin_id in range(n_bins):
            in_bin = (current_speed >= speed_edges[bin_id]) & (current_speed <= speed_edges[bin_id + 1])
            if np.any(in_bin):
                target = np.median(true_pair_behavior[in_bin], axis=0)
                scale = np.std(true_pair_behavior[in_bin], axis=0)
                scale[scale < min_scale] = min_scale
            else:
                target = global_target
                scale = global_scale
            target_by_bin[bin_id] = target
            scale_by_bin[bin_id] = scale

        return cls(
            archive=archive,
            seed=seed,
            speed_edges=speed_edges,
            target_by_bin=target_by_bin,
            scale_by_bin=scale_by_bin,
            pair_descriptors=pair_descriptors,
            diffusivity=diffusivity,
            bandwidth_multiplier=bandwidth_multiplier,
            candidate_limit=candidate_limit,
            pair_weight=pair_weight,
        )

    @property
    def sigma_velocity(self) -> float:
        tau = self.archive.match_steps * self.archive.dt
        return self.bandwidth_multiplier * np.sqrt(2.0 * self.diffusivity * tau) / tau

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)
        if self.candidate_limit is None or self.candidate_limit >= self.archive.size:
            candidate_ids = np.arange(self.archive.size)
            candidate_distances = distances
        else:
            k = min(self.candidate_limit, self.archive.size)
            candidate_ids = np.argpartition(distances, kth=k - 1)[:k]
            candidate_distances = distances[candidate_ids]

        sigma = max(float(self.sigma_velocity), 1e-12)
        gaussian_logits = -0.5 * candidate_distances / (sigma * sigma)

        speed = float(np.linalg.norm(end_velocity))
        bin_id = int(np.searchsorted(self.speed_edges[1:-1], speed, side="right"))
        target = self.target_by_bin[bin_id]
        scale = self.scale_by_bin[bin_id]
        z = (self.pair_descriptors[candidate_ids] - target[None, :]) / scale[None, :]
        pair_logits = -0.5 * np.sum(z * z, axis=1)

        logits = gaussian_logits + self.pair_weight * pair_logits
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
        else:
            weights /= total
        return candidate_ids, weights


@dataclass
class ContrastiveTransitionSampler(SegmentSampler):
    """Learned transition scorer trained from true archive continuations.

    This is intentionally small and transparent: a logistic contrastive model
    learns which segment-context features distinguish true adjacent segment
    continuations from random candidates.
    """

    weights: np.ndarray | None = None
    bias: float = 0.0
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    segment_context: np.ndarray | None = None
    candidate_limit: int | None = 256
    temperature: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.weights is None
            or self.feature_mean is None
            or self.feature_scale is None
            or self.segment_context is None
        ):
            raise ValueError("use ContrastiveTransitionSampler.fit(...) to train the scorer")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive when provided")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @classmethod
    def fit(
        cls,
        archive: SegmentArchive,
        *,
        seed: int | None = None,
        negative_ratio: int = 4,
        max_positive_pairs: int | None = 6000,
        learning_rate: float = 0.2,
        epochs: int = 600,
        l2: float = 1e-3,
        candidate_limit: int | None = 256,
        temperature: float = 1.0,
    ) -> "ContrastiveTransitionSampler":
        rng = np.random.default_rng(seed)
        positive_pairs = archive.true_successor_pairs()
        if len(positive_pairs) == 0:
            raise ValueError("archive has no true successor pairs; check stride/source metadata")
        if max_positive_pairs is not None and len(positive_pairs) > max_positive_pairs:
            chosen = rng.choice(len(positive_pairs), size=max_positive_pairs, replace=False)
            positive_pairs = positive_pairs[chosen]
        segment_context = archive_segment_context(archive)

        x_positive = contextual_transition_features(
            archive,
            positive_pairs[:, 0],
            positive_pairs[:, 1],
            segment_context,
        )

        negative_count = len(positive_pairs) * negative_ratio
        current_ids = rng.choice(positive_pairs[:, 0], size=negative_count, replace=True)
        candidate_ids = rng.integers(0, archive.size, size=negative_count)
        x_negative = contextual_transition_features(
            archive,
            current_ids,
            candidate_ids,
            segment_context,
        )

        x = np.vstack([x_positive, x_negative])
        y = np.concatenate([np.ones(len(x_positive)), np.zeros(len(x_negative))])

        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale[feature_scale <= 1e-12] = 1.0
        x_scaled = (x - feature_mean) / feature_scale

        weights = np.zeros(x_scaled.shape[1], dtype=float)
        bias = float(np.log(len(x_positive) / max(len(x_negative), 1)))
        for _ in range(epochs):
            logits = x_scaled @ weights + bias
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            residual = probabilities - y
            grad_w = x_scaled.T @ residual / len(x_scaled) + l2 * weights
            grad_b = float(np.mean(residual))
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        return cls(
            archive=archive,
            seed=seed,
            weights=weights,
            bias=bias,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            segment_context=segment_context,
            candidate_limit=candidate_limit,
            temperature=temperature,
        )

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)
        if self.candidate_limit is None or self.candidate_limit >= self.archive.size:
            candidate_ids = np.arange(self.archive.size)
        else:
            k = min(self.candidate_limit, self.archive.size)
            candidate_ids = np.argpartition(distances, kth=k - 1)[:k]

        if current_index is None:
            raise ValueError("current_index is required for contextual contrastive scoring")
        current_ids = np.full(len(candidate_ids), int(current_index), dtype=int)
        features = contextual_transition_features(
            self.archive,
            current_ids,
            candidate_ids,
            self.segment_context,
        )
        scaled = (features - self.feature_mean) / self.feature_scale
        logits = (scaled @ self.weights + self.bias) / self.temperature
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
        else:
            weights /= total
        return candidate_ids, weights


@dataclass
class HybridContrastiveGaussianSampler(ContrastiveTransitionSampler):
    """Contextual contrastive score with a Gaussian continuity prior."""

    diffusivity: float = 1e-4
    bandwidth_multiplier: float = 1.0
    gaussian_weight: float = 1.0
    learned_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diffusivity <= 0:
            raise ValueError("diffusivity must be positive")
        if self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be positive")
        if self.gaussian_weight < 0:
            raise ValueError("gaussian_weight must be nonnegative")
        if self.learned_weight < 0:
            raise ValueError("learned_weight must be nonnegative")

    @classmethod
    def fit(
        cls,
        archive: SegmentArchive,
        *,
        seed: int | None = None,
        negative_ratio: int = 4,
        max_positive_pairs: int | None = 6000,
        learning_rate: float = 0.2,
        epochs: int = 600,
        l2: float = 1e-3,
        candidate_limit: int | None = 256,
        temperature: float = 1.0,
        diffusivity: float = 1e-4,
        bandwidth_multiplier: float = 1.0,
        gaussian_weight: float = 1.0,
        learned_weight: float = 1.0,
    ) -> "HybridContrastiveGaussianSampler":
        trained = ContrastiveTransitionSampler.fit(
            archive,
            seed=seed,
            negative_ratio=negative_ratio,
            max_positive_pairs=max_positive_pairs,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            candidate_limit=candidate_limit,
            temperature=temperature,
        )
        return cls(
            archive=archive,
            seed=seed,
            weights=trained.weights,
            bias=trained.bias,
            feature_mean=trained.feature_mean,
            feature_scale=trained.feature_scale,
            segment_context=trained.segment_context,
            candidate_limit=candidate_limit,
            temperature=temperature,
            diffusivity=diffusivity,
            bandwidth_multiplier=bandwidth_multiplier,
            gaussian_weight=gaussian_weight,
            learned_weight=learned_weight,
        )

    @property
    def sigma_velocity(self) -> float:
        tau = self.archive.match_steps * self.archive.dt
        return self.bandwidth_multiplier * np.sqrt(2.0 * self.diffusivity * tau) / tau

    def transition_weights(
        self,
        end_velocity: np.ndarray,
        current_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if current_index is None:
            raise ValueError("current_index is required for hybrid contrastive scoring")

        delta = self.archive.start_velocities - end_velocity[None, :]
        distances = np.sum(delta * delta, axis=1)
        if self.candidate_limit is None or self.candidate_limit >= self.archive.size:
            candidate_ids = np.arange(self.archive.size)
            candidate_distances = distances
        else:
            k = min(self.candidate_limit, self.archive.size)
            candidate_ids = np.argpartition(distances, kth=k - 1)[:k]
            candidate_distances = distances[candidate_ids]

        current_ids = np.full(len(candidate_ids), int(current_index), dtype=int)
        features = contextual_transition_features(
            self.archive,
            current_ids,
            candidate_ids,
            self.segment_context,
        )
        scaled = (features - self.feature_mean) / self.feature_scale
        learned_logits = (scaled @ self.weights + self.bias) / self.temperature

        sigma = max(float(self.sigma_velocity), 1e-12)
        gaussian_logits = -0.5 * candidate_distances / (sigma * sigma)
        logits = self.learned_weight * learned_logits + self.gaussian_weight * gaussian_logits
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
        else:
            weights /= total
        return candidate_ids, weights


def transition_features(end_velocities: np.ndarray, start_velocities: np.ndarray) -> np.ndarray:
    end = np.atleast_2d(np.asarray(end_velocities, dtype=float))
    start = np.atleast_2d(np.asarray(start_velocities, dtype=float))
    delta = start - end
    abs_delta = np.abs(delta)
    squared_delta = delta * delta
    end_speed = np.linalg.norm(end, axis=1)
    start_speed = np.linalg.norm(start, axis=1)
    speed_diff = np.abs(start_speed - end_speed)[:, None]
    dot = np.sum(end * start, axis=1)
    denom = np.maximum(end_speed * start_speed, 1e-12)
    cosine_distance = (1.0 - dot / denom)[:, None]
    return np.hstack([abs_delta, squared_delta, speed_diff, cosine_distance])


def archive_segment_context(archive: SegmentArchive) -> np.ndarray:
    """Segment descriptors used by the contextual contrastive scorer."""
    increments = np.diff(archive.relative_paths, axis=1) / archive.dt
    mean_velocity = increments.mean(axis=1)
    speed = np.linalg.norm(increments, axis=2)
    speed_mean = speed.mean(axis=1, keepdims=True)
    speed_std = speed.std(axis=1, keepdims=True)
    displacement_velocity = archive.relative_paths[:, -1, :] / (archive.segment_steps * archive.dt)
    start_to_mean = archive.start_velocities - mean_velocity
    end_to_mean = archive.end_velocities - mean_velocity
    curvature_proxy = np.linalg.norm(archive.end_velocities - archive.start_velocities, axis=1, keepdims=True)
    return np.hstack(
        [
            mean_velocity,
            displacement_velocity,
            speed_mean,
            speed_std,
            start_to_mean,
            end_to_mean,
            curvature_proxy,
        ]
    )


def archive_horizon_descriptors(
    archive: SegmentArchive,
    *,
    horizon_segments: int = 3,
) -> np.ndarray:
    """Approximate multi-segment future descriptors for each archive segment."""
    if horizon_segments <= 0:
        raise ValueError("horizon_segments must be positive")
    successor = np.full(archive.size, -1, dtype=int)
    for current, nxt in archive.true_successor_pairs():
        successor[current] = nxt

    descriptors = np.empty((archive.size, archive.n_dimensions + 4), dtype=float)
    for idx in range(archive.size):
        displacement = np.zeros(archive.n_dimensions, dtype=float)
        speeds: list[float] = []
        current = idx
        used = 0
        for _ in range(horizon_segments):
            if current < 0:
                break
            segment_disp = archive.relative_paths[current, -1]
            displacement += segment_disp
            segment_speed = np.linalg.norm(segment_disp) / max(archive.segment_steps * archive.dt, 1e-12)
            speeds.append(float(segment_speed))
            used += 1
            current = int(successor[current])

        duration = max(used * archive.segment_steps * archive.dt, 1e-12)
        total_speed = np.linalg.norm(displacement) / duration
        longitudinal_speed = displacement[0] / duration
        transverse_speed = np.linalg.norm(displacement[1:]) / duration if archive.n_dimensions > 1 else 0.0
        speed_std = float(np.std(speeds)) if speeds else 0.0
        descriptors[idx] = np.concatenate(
            [
                displacement / duration,
                np.asarray([total_speed, longitudinal_speed, transverse_speed, speed_std]),
            ]
        )
    return descriptors


def archive_pair_behavior_descriptors(
    archive: SegmentArchive,
    *,
    horizon_segments: int = 3,
    neighbor_k: int = 32,
) -> np.ndarray:
    """Estimate pair-aware behavior from local future divergence in archive state space."""
    if neighbor_k <= 0:
        raise ValueError("neighbor_k must be positive")
    horizon = archive_horizon_descriptors(archive, horizon_segments=horizon_segments)
    future_velocity = horizon[:, : archive.n_dimensions]
    n = archive.size
    descriptors = np.empty((n, 5), dtype=float)

    for idx in range(n):
        state_delta = archive.start_velocities - archive.start_velocities[idx][None, :]
        state_distances = np.sum(state_delta * state_delta, axis=1)
        positive = np.flatnonzero(state_distances > 1e-18)
        if len(positive) == 0:
            neighbor_ids = np.array([idx], dtype=int)
        else:
            k = min(neighbor_k, len(positive))
            local = np.argpartition(state_distances[positive], kth=k - 1)[:k]
            neighbor_ids = positive[local]

        future_delta = future_velocity[neighbor_ids] - future_velocity[idx][None, :]
        future_dist = np.linalg.norm(future_delta, axis=1)
        current_state_dist = np.sqrt(state_distances[neighbor_ids])
        growth = future_dist - current_state_dist
        descriptors[idx] = np.asarray(
            [
                np.median(future_dist),
                np.quantile(future_dist, 0.9),
                np.median(growth),
                np.mean(growth < 0.0),
                np.std(future_dist),
            ],
            dtype=float,
        )

    return descriptors


def contextual_transition_features(
    archive: SegmentArchive,
    current_ids: np.ndarray,
    candidate_ids: np.ndarray,
    segment_context: np.ndarray,
) -> np.ndarray:
    """Features comparing a current archive segment to a candidate successor."""
    current_ids = np.asarray(current_ids, dtype=int)
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    endpoint_features = transition_features(
        archive.end_velocities[current_ids],
        archive.start_velocities[candidate_ids],
    )

    current_context = segment_context[current_ids]
    candidate_context = segment_context[candidate_ids]
    context_delta = candidate_context - current_context
    context_abs_delta = np.abs(context_delta)

    d = archive.n_dimensions
    current_mean = current_context[:, :d]
    candidate_mean = candidate_context[:, :d]
    mean_cosine = cosine_distance(current_mean, candidate_mean)

    current_disp = current_context[:, d : 2 * d]
    candidate_disp = candidate_context[:, d : 2 * d]
    displacement_cosine = cosine_distance(current_disp, candidate_disp)

    return np.hstack(
        [
            endpoint_features,
            context_abs_delta,
            mean_cosine,
            displacement_cosine,
        ]
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(a_norm * b_norm, 1e-12)
    return (1.0 - np.sum(a * b, axis=1) / denom)[:, None]
