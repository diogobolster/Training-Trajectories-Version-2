from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import (
    breakthrough_score,
    breakthrough_times,
    dilution_index,
    pair_separation_summary,
    quantile_curve_mae,
    reaction_encounter_probability,
    scalar_curve_log_mae,
    summarize_breakthroughs,
)


@dataclass(frozen=True)
class MetricSettings:
    planes: list[float]
    time_indices: list[int]
    bin_size: float = 3.0
    pair_samples: int = 3000
    reaction_radius: float = 3.0
    missing_penalty: float = 250.0
    seed: int = 123


@dataclass(frozen=True)
class ObjectiveWeights:
    btc: float = 1.0
    pair: float = 20.0
    dilution: float = 120.0
    reaction: float = 1000.0


def evaluate_ensemble(
    trajectories: list[np.ndarray],
    settings: MetricSettings,
) -> dict[str, object]:
    return {
        "breakthrough": summarize_breakthroughs(breakthrough_times(trajectories, settings.planes)),
        "dilution": dilution_index(
            trajectories,
            settings.time_indices,
            bin_size=settings.bin_size,
        ),
        "pair_separation": pair_separation_summary(
            trajectories,
            settings.time_indices,
            n_pairs=settings.pair_samples,
            seed=settings.seed,
        ),
        "reaction": reaction_encounter_probability(
            trajectories,
            reaction_radius=settings.reaction_radius,
            n_pairs=settings.pair_samples,
            max_time_index=max(settings.time_indices),
            seed=settings.seed,
        ),
    }


def compare_metrics(
    reference: dict[str, object],
    generated: dict[str, object],
    *,
    missing_penalty: float,
) -> dict[str, float]:
    btc = breakthrough_score(
        reference["breakthrough"],
        generated["breakthrough"],
        missing_penalty=missing_penalty,
    )
    dilution_log_mae = scalar_curve_log_mae(
        reference["dilution"],
        generated["dilution"],
        "dilution_index",
    )
    pair_quantile_mae = quantile_curve_mae(
        reference["pair_separation"],
        generated["pair_separation"],
    )

    ref_reaction = reference["reaction"]["probability"]
    gen_reaction = generated["reaction"]["probability"]
    reaction_abs_error = (
        float(abs(ref_reaction - gen_reaction))
        if np.isfinite(ref_reaction) and np.isfinite(gen_reaction)
        else float("nan")
    )

    return {
        "btc_score": btc["score"],
        "btc_quantile_mae": btc["quantile_mae"],
        "btc_coverage_deficit": btc["coverage_deficit"],
        "dilution_log_mae": dilution_log_mae,
        "pair_quantile_mae": pair_quantile_mae,
        "reaction_abs_error": reaction_abs_error,
    }


def multi_objective_score(
    errors: dict[str, float],
    weights: ObjectiveWeights,
) -> float:
    terms = [
        weights.btc * finite_or_penalty(errors["btc_score"], 500.0),
        weights.pair * finite_or_penalty(errors["pair_quantile_mae"], 25.0),
        weights.dilution * finite_or_penalty(errors["dilution_log_mae"], 2.0),
        weights.reaction * finite_or_penalty(errors["reaction_abs_error"], 1.0),
    ]
    return float(sum(terms))


def finite_or_penalty(value: float, penalty: float) -> float:
    return float(value) if np.isfinite(value) else float(penalty)


def generate_with_origins(
    sampler,
    *,
    n_trajectories: int,
    n_segments: int,
    origin_pool: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    generated: list[np.ndarray] = []
    for _ in range(n_trajectories):
        origin = origin_pool[int(rng.integers(0, len(origin_pool)))]
        generated.append(sampler.generate(n_segments=n_segments, origin=origin))
    return generated


def choose_origin_pool(
    source: str,
    train: list[np.ndarray],
    reference: list[np.ndarray],
) -> np.ndarray:
    if source == "zero":
        return np.zeros((1, train[0].shape[1]), dtype=float)
    pool = train if source == "train" else reference
    return np.stack([np.asarray(traj, dtype=float)[0] for traj in pool])


def split_three_way(
    trajectories: list[np.ndarray],
    fit_fraction: float,
    validation_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if len(trajectories) < 3:
        raise ValueError("at least three trajectories are needed for fit/validation/test")
    if fit_fraction <= 0 or validation_fraction <= 0 or fit_fraction + validation_fraction >= 1:
        raise ValueError("fit_fraction and validation_fraction must be positive and sum to less than one")
    n = len(trajectories)
    fit_end = max(1, int(round(fit_fraction * n)))
    validation_end = max(fit_end + 1, int(round((fit_fraction + validation_fraction) * n)))
    validation_end = min(validation_end, n - 1)
    return trajectories[:fit_end], trajectories[fit_end:validation_end], trajectories[validation_end:]
