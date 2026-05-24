from __future__ import annotations

import numpy as np


def breakthrough_times(trajectories: list[np.ndarray], planes: list[float]) -> dict[float, np.ndarray]:
    """First time index at which each trajectory crosses each x-plane."""
    result: dict[float, list[float]] = {plane: [] for plane in planes}
    for trajectory in trajectories:
        x = np.asarray(trajectory)[:, 0]
        for plane in planes:
            crossed = np.flatnonzero(x >= plane)
            result[plane].append(float(crossed[0]) if len(crossed) else np.nan)
    return {plane: np.asarray(times, dtype=float) for plane, times in result.items()}


def summarize_breakthroughs(times_by_plane: dict[float, np.ndarray]) -> dict[float, dict[str, float]]:
    """Summarize breakthrough times with robust quantiles."""
    summary: dict[float, dict[str, float]] = {}
    for plane, times in times_by_plane.items():
        finite = times[np.isfinite(times)]
        if len(finite) == 0:
            summary[plane] = {
                "count": 0.0,
                "coverage": 0.0,
                "q10": np.nan,
                "q50": np.nan,
                "q90": np.nan,
            }
            continue
        q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
        summary[plane] = {
            "count": float(len(finite)),
            "coverage": float(len(finite) / len(times)) if len(times) else 0.0,
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
        }
    return summary


def breakthrough_quantile_mae(
    reference: dict[float, dict[str, float]],
    generated: dict[float, dict[str, float]],
) -> float:
    """Mean absolute error over finite breakthrough quantiles."""
    errors: list[float] = []
    for plane, ref_stats in reference.items():
        gen_stats = generated[plane]
        for key in ("q10", "q50", "q90"):
            ref = ref_stats[key]
            gen = gen_stats[key]
            if np.isfinite(ref) and np.isfinite(gen):
                errors.append(abs(ref - gen))
    return float(np.mean(errors)) if errors else float("nan")


def breakthrough_coverage_deficit(
    reference: dict[float, dict[str, float]],
    generated: dict[float, dict[str, float]],
) -> float:
    """Average shortfall in plane-crossing coverage relative to reference."""
    deficits: list[float] = []
    for plane, ref_stats in reference.items():
        ref_coverage = ref_stats.get("coverage", 0.0)
        gen_coverage = generated[plane].get("coverage", 0.0)
        deficits.append(max(0.0, ref_coverage - gen_coverage))
    return float(np.mean(deficits)) if deficits else float("nan")


def breakthrough_score(
    reference: dict[float, dict[str, float]],
    generated: dict[float, dict[str, float]],
    missing_penalty: float = 250.0,
) -> dict[str, float]:
    """Coverage-aware scalar score; lower is better."""
    mae = breakthrough_quantile_mae(reference, generated)
    deficit = breakthrough_coverage_deficit(reference, generated)
    if not np.isfinite(mae):
        mae = missing_penalty
    return {
        "quantile_mae": float(mae),
        "coverage_deficit": float(deficit),
        "score": float(mae + missing_penalty * deficit),
    }


def velocity_autocorrelation(trajectories: list[np.ndarray], max_lag: int = 80) -> np.ndarray:
    """Estimate scalar velocity autocorrelation from trajectory increments."""
    increments = np.concatenate([np.diff(np.asarray(t), axis=0) for t in trajectories], axis=0)
    speeds = increments[:, 0] - increments[:, 0].mean()
    denom = float(np.dot(speeds, speeds))
    if denom <= 0.0:
        return np.full(max_lag + 1, np.nan)

    corr = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        if lag == 0:
            corr[lag] = 1.0
        else:
            corr[lag] = float(np.dot(speeds[:-lag], speeds[lag:]) / denom)
    return corr


def dilution_index(
    trajectories: list[np.ndarray],
    time_indices: list[int],
    *,
    bin_size: float = 2.0,
) -> dict[int, dict[str, float]]:
    """Estimate an occupancy-entropy dilution proxy at selected times."""
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    result: dict[int, dict[str, float]] = {}
    for time_index in time_indices:
        positions = positions_at_time(trajectories, time_index)
        if len(positions) == 0:
            result[time_index] = {
                "n_particles": 0.0,
                "occupied_bins": 0.0,
                "entropy": float("nan"),
                "dilution_index": float("nan"),
            }
            continue

        bins = np.floor(positions / bin_size).astype(int)
        _, counts = np.unique(bins, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        bin_volume = bin_size ** positions.shape[1]
        result[time_index] = {
            "n_particles": float(len(positions)),
            "occupied_bins": float(len(counts)),
            "entropy": entropy,
            "dilution_index": float(bin_volume * np.exp(entropy)),
        }
    return result


def pair_separation_summary(
    trajectories: list[np.ndarray],
    time_indices: list[int],
    *,
    n_pairs: int = 2000,
    seed: int | None = 1,
) -> dict[int, dict[str, float]]:
    """Summarize sampled particle-pair separation distances."""
    rng = np.random.default_rng(seed)
    pairs = sample_pair_indices(len(trajectories), n_pairs, rng)
    result: dict[int, dict[str, float]] = {}

    for time_index in time_indices:
        distances: list[float] = []
        for i, j in pairs:
            if len(trajectories[i]) <= time_index or len(trajectories[j]) <= time_index:
                continue
            distance = np.linalg.norm(trajectories[i][time_index] - trajectories[j][time_index])
            distances.append(float(distance))
        result[time_index] = summarize_values(np.asarray(distances, dtype=float))
    return result


def reaction_encounter_probability(
    trajectories: list[np.ndarray],
    *,
    reaction_radius: float,
    n_pairs: int = 2000,
    max_time_index: int | None = None,
    seed: int | None = 1,
) -> dict[str, float]:
    """Probability that sampled particle pairs come within a reaction radius."""
    if reaction_radius <= 0:
        raise ValueError("reaction_radius must be positive")

    rng = np.random.default_rng(seed)
    pairs = sample_pair_indices(len(trajectories), n_pairs, rng)
    encounters = 0
    usable = 0
    for i, j in pairs:
        horizon = min(len(trajectories[i]), len(trajectories[j]))
        if max_time_index is not None:
            horizon = min(horizon, max_time_index + 1)
        if horizon <= 0:
            continue
        delta = trajectories[i][:horizon] - trajectories[j][:horizon]
        min_distance = float(np.min(np.linalg.norm(delta, axis=1)))
        encounters += int(min_distance <= reaction_radius)
        usable += 1

    probability = encounters / usable if usable else float("nan")
    return {
        "reaction_radius": float(reaction_radius),
        "pair_count": float(usable),
        "encounter_count": float(encounters),
        "probability": float(probability),
    }


def positions_at_time(trajectories: list[np.ndarray], time_index: int) -> np.ndarray:
    """Collect positions for trajectories that are alive at a time index."""
    if time_index < 0:
        raise ValueError("time_index must be nonnegative")
    positions = [np.asarray(traj)[time_index] for traj in trajectories if len(traj) > time_index]
    if not positions:
        return np.empty((0, 0), dtype=float)
    return np.stack(positions)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "q10": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
        }
    q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
    return {
        "count": float(len(finite)),
        "mean": float(np.mean(finite)),
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
    }


def sample_pair_indices(
    n_items: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if n_items < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for _ in range(n_pairs):
        i = int(rng.integers(0, n_items))
        j = int(rng.integers(0, n_items - 1))
        if j >= i:
            j += 1
        pairs.append((i, j))
    return pairs


def scalar_curve_log_mae(
    reference: dict[int, dict[str, float]],
    generated: dict[int, dict[str, float]],
    key: str,
) -> float:
    """Mean absolute error between positive scalar curves in log space."""
    errors: list[float] = []
    for time_index, ref_stats in reference.items():
        ref = ref_stats[key]
        gen = generated[time_index][key]
        if np.isfinite(ref) and np.isfinite(gen) and ref > 0 and gen > 0:
            errors.append(abs(np.log(gen) - np.log(ref)))
    return float(np.mean(errors)) if errors else float("nan")


def quantile_curve_mae(
    reference: dict[int, dict[str, float]],
    generated: dict[int, dict[str, float]],
) -> float:
    """Mean absolute error over q10/q50/q90 curves."""
    errors: list[float] = []
    for time_index, ref_stats in reference.items():
        gen_stats = generated[time_index]
        for key in ("q10", "q50", "q90"):
            ref = ref_stats[key]
            gen = gen_stats[key]
            if np.isfinite(ref) and np.isfinite(gen):
                errors.append(abs(ref - gen))
    return float(np.mean(errors)) if errors else float("nan")


def adjacent_segment_velocity_correlation(
    trajectories: list[np.ndarray],
    segment_steps_values: list[int],
    *,
    component: int = 0,
    dt: float = 1.0,
) -> dict[int, float]:
    """Correlation between velocities averaged over adjacent segments.

    This is the discrete analogue of the paper's K(lambda)|1,2 diagnostic.
    """
    correlations: dict[int, float] = {}
    for segment_steps in segment_steps_values:
        if segment_steps <= 0:
            raise ValueError("segment step values must be positive")
        left_values: list[float] = []
        right_values: list[float] = []
        for trajectory in trajectories:
            arr = np.asarray(trajectory, dtype=float)
            if arr.shape[0] < 2 * segment_steps + 1:
                continue
            for start in range(0, arr.shape[0] - 2 * segment_steps, segment_steps):
                middle = start + segment_steps
                end = middle + segment_steps
                left_velocity = (arr[middle, component] - arr[start, component]) / (segment_steps * dt)
                right_velocity = (arr[end, component] - arr[middle, component]) / (segment_steps * dt)
                left_values.append(float(left_velocity))
                right_values.append(float(right_velocity))

        if len(left_values) < 3:
            correlations[segment_steps] = float("nan")
            continue
        left = np.asarray(left_values)
        right = np.asarray(right_values)
        left -= left.mean()
        right -= right.mean()
        denom = np.sqrt(np.dot(left, left) * np.dot(right, right))
        correlations[segment_steps] = float(np.dot(left, right) / denom) if denom > 0 else float("nan")
    return correlations


def suggest_segment_scales(correlations: dict[int, float]) -> dict[str, float]:
    """Heuristic suggestions for match and segment lengths from K(lambda)|1,2."""
    items = [(step, corr) for step, corr in sorted(correlations.items()) if np.isfinite(corr)]
    if not items:
        return {
            "match_steps": float("nan"),
            "segment_steps": float("nan"),
            "match_correlation": float("nan"),
            "segment_correlation": float("nan"),
        }

    steps = np.asarray([item[0] for item in items], dtype=float)
    values = np.asarray([item[1] for item in items], dtype=float)
    peak_idx = int(np.argmax(values))
    match_idx = peak_idx

    segment_idx = len(values) - 1
    if len(values) >= 3:
        second_diff = values[:-2] - 2.0 * values[1:-1] + values[2:]
        for idx in range(max(1, peak_idx + 1), len(values) - 1):
            decreasing = values[idx] < values[idx - 1]
            convex = second_diff[idx - 1] > 0.0
            if decreasing and convex:
                segment_idx = idx
                break

    return {
        "match_steps": float(steps[match_idx]),
        "segment_steps": float(steps[segment_idx]),
        "match_correlation": float(values[match_idx]),
        "segment_correlation": float(values[segment_idx]),
    }
