from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import (  # noqa: E402
    ConditionalSegmentSampler,
    GaussianBayesSegmentSampler,
    HybridContrastiveGaussianSampler,
    MetricSettings,
    MixtureSegmentSampler,
    ObjectiveWeights,
    PairAwareRerankGaussianSampler,
    SegmentArchive,
    choose_origin_pool,
    compare_metrics,
    evaluate_ensemble,
    generate_with_origins,
    load_trajectories,
    multi_objective_score,
)


COMPONENT_ORDER = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average validation-selected sampler mixture weights over repeated splits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.30,
        help="Fraction of the non-test pool used for validation in each repeat.",
    )
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--planes", default="6,10,14")
    parser.add_argument("--time-indices", default="100,200,300,400")
    parser.add_argument("--segment-steps", type=int, default=36)
    parser.add_argument("--match-steps", type=int, default=20)
    parser.add_argument(
        "--segment-stride",
        type=int,
        help="Stride between archived segment windows. Defaults to segment_steps - match_steps.",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--gaussian-bandwidth", type=float, default=0.25)
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=96)
    parser.add_argument("--knn-temperature", type=float, default=0.8)
    parser.add_argument("--contrastive-epochs", type=int, default=500)
    parser.add_argument("--contrastive-negative-ratio", type=int, default=6)
    parser.add_argument("--hybrid-learned-weight", type=float, default=0.25)
    parser.add_argument("--pair-rerank-weight", type=float, default=0.25)
    parser.add_argument("--pair-neighbor-k", type=int, default=32)
    parser.add_argument("--rerank-horizon-segments", type=int, default=3)
    parser.add_argument("--adaptive-bins", type=int, default=4)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--n-validation-generated", type=int, default=50)
    parser.add_argument("--n-test-generated", type=int, default=90)
    parser.add_argument("--n-segments", type=int, default=32)
    parser.add_argument("--bin-size", type=float, default=3.0)
    parser.add_argument("--pair-samples", type=int, default=1500)
    parser.add_argument("--reaction-radius", type=float, default=3.0)
    parser.add_argument("--btc-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=20.0)
    parser.add_argument("--dilution-weight", type=float, default=120.0)
    parser.add_argument("--reaction-weight", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_bootstrap_mixture_selection.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    trajectories = load_trajectories(args.input, key=args.key)
    train_pool, test, split_payload = fixed_train_test_split(
        trajectories,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    settings = MetricSettings(
        planes=parse_float_list(args.planes),
        time_indices=parse_int_list(args.time_indices),
        bin_size=args.bin_size,
        pair_samples=args.pair_samples,
        reaction_radius=args.reaction_radius,
        seed=args.seed,
    )
    objective_weights = ObjectiveWeights(
        btc=args.btc_weight,
        pair=args.pair_weight,
        dilution=args.dilution_weight,
        reaction=args.reaction_weight,
    )
    grid = simplex_grid(len(COMPONENT_ORDER), args.grid_step)

    repeat_payloads = []
    score_sums = {weights: 0.0 for weights in grid}
    selected_weight_vectors = []

    for repeat in range(args.n_repeats):
        repeat_seed = args.seed + 1009 * repeat
        fit, validation, repeat_split = repeated_fit_validation_split(
            train_pool,
            validation_fraction=args.validation_fraction,
            seed=repeat_seed,
        )
        archive = SegmentArchive.from_trajectories(
            fit,
            segment_steps=args.segment_steps,
            match_steps=args.match_steps,
            stride=args.segment_stride,
            dt=args.dt,
        )
        components = build_components(args, archive, seed=repeat_seed)
        repeat_settings = replace(settings, seed=repeat_seed)
        validation_reference = evaluate_ensemble(validation, repeat_settings)
        validation_origin_pool = choose_origin_pool("train", fit, validation)

        validation_results = []
        for weights in grid:
            mixture = MixtureSegmentSampler(
                archive=archive,
                components=[components[name] for name in COMPONENT_ORDER],
                component_weights=np.asarray(weights),
                seed=repeat_seed,
            )
            generated = generate_with_origins(
                mixture,
                n_trajectories=args.n_validation_generated,
                n_segments=args.n_segments,
                origin_pool=validation_origin_pool,
                rng=np.random.default_rng(repeat_seed),
            )
            metrics = evaluate_ensemble(generated, repeat_settings)
            errors = compare_metrics(
                validation_reference,
                metrics,
                missing_penalty=repeat_settings.missing_penalty,
            )
            score = multi_objective_score(errors, objective_weights)
            score_sums[weights] += score
            validation_results.append(
                {
                    "weights": weight_dict(weights),
                    "score": score,
                    "errors": errors,
                }
            )

        validation_results.sort(key=lambda item: item["score"])
        best_weights = tuple(validation_results[0]["weights"][name] for name in COMPONENT_ORDER)
        selected_weight_vectors.append(best_weights)
        repeat_payloads.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "split": repeat_split,
                "archive_size": archive.size,
                "selected_weights": validation_results[0]["weights"],
                "selected_score": validation_results[0]["score"],
                "validation_top": validation_results[:8],
            }
        )
        print(
            f"repeat {repeat + 1}/{args.n_repeats}: "
            f"fit/validation {len(fit)}/{len(validation)}, "
            f"selected {validation_results[0]['weights']}, "
            f"score {validation_results[0]['score']:.2f}",
            flush=True,
        )

    mean_weights = normalize_weights(np.mean(np.asarray(selected_weight_vectors), axis=0))
    pooled_best_weights = min(grid, key=lambda weights: score_sums[weights] / args.n_repeats)

    final_archive = SegmentArchive.from_trajectories(
        train_pool,
        segment_steps=args.segment_steps,
        match_steps=args.match_steps,
        stride=args.segment_stride,
        dt=args.dt,
    )
    final_components = build_components(args, final_archive, seed=args.seed)
    test_reference = evaluate_ensemble(test, settings)
    test_origin_pool = choose_origin_pool("train", train_pool, test)

    mean_mixture = MixtureSegmentSampler(
        archive=final_archive,
        components=[final_components[name] for name in COMPONENT_ORDER],
        component_weights=mean_weights,
        seed=args.seed,
    )
    pooled_mixture = MixtureSegmentSampler(
        archive=final_archive,
        components=[final_components[name] for name in COMPONENT_ORDER],
        component_weights=np.asarray(pooled_best_weights),
        seed=args.seed,
    )
    eval_samplers = {
        "bootstrap_mean_mixture": mean_mixture,
        "pooled_validation_mixture": pooled_mixture,
        **final_components,
    }
    test_payload = {}
    for name, sampler in eval_samplers.items():
        generated = generate_with_origins(
            sampler,
            n_trajectories=args.n_test_generated,
            n_segments=args.n_segments,
            origin_pool=test_origin_pool,
            rng=np.random.default_rng(args.seed),
        )
        metrics = evaluate_ensemble(generated, settings)
        errors = compare_metrics(test_reference, metrics, missing_penalty=settings.missing_penalty)
        test_payload[name] = {
            "errors": errors,
            "objective": multi_objective_score(errors, objective_weights),
            "metrics": metrics,
        }

    mean_score_by_weight = [
        {
            "weights": weight_dict(weights),
            "mean_score": score_sums[weights] / args.n_repeats,
        }
        for weights in grid
    ]
    mean_score_by_weight.sort(key=lambda item: item["mean_score"])

    payload = {
        "input": str(args.input),
        "splits": split_payload,
        "repeat_count": args.n_repeats,
        "component_order": COMPONENT_ORDER,
        "archive": {
            "size": final_archive.size,
            "segment_steps": final_archive.segment_steps,
            "match_steps": final_archive.match_steps,
        },
        "objective_weights": {
            "btc": objective_weights.btc,
            "pair": objective_weights.pair,
            "dilution": objective_weights.dilution,
            "reaction": objective_weights.reaction,
        },
        "repeat_results": repeat_payloads,
        "mean_selected_weights": weight_dict(mean_weights),
        "pooled_validation_weights": weight_dict(pooled_best_weights),
        "pooled_validation_top": mean_score_by_weight[:12],
        "test_reference": test_reference,
        "test": test_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Train/test: {len(train_pool)}/{len(test)}")
    print(f"Final archive segments: {final_archive.size}")
    print(f"Wrote bootstrap mixture selection: {args.output}")
    print(f"Mean selected weights: {weight_dict(mean_weights)}")
    print(f"Pooled validation weights: {weight_dict(pooled_best_weights)}")
    print()
    print("Held-out test errors")
    print("sampler                    objective  btc_score  pair_mae  dilution_log  reaction_abs")
    for name, item in sorted(test_payload.items(), key=lambda pair: pair[1]["objective"]):
        errors = item["errors"]
        print(
            f"{name:<26}"
            f"{item['objective']:10.2f}"
            f"{errors['btc_score']:11.2f}"
            f"{errors['pair_quantile_mae']:10.2f}"
            f"{errors['dilution_log_mae']:14.3f}"
            f"{errors['reaction_abs_error']:14.3f}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be positive")


def fixed_train_test_split(
    trajectories: list[np.ndarray],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    if len(trajectories) < 4:
        raise ValueError("at least four trajectories are needed for repeated validation")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajectories))
    n_test = max(1, int(round(test_fraction * len(trajectories))))
    n_test = min(n_test, len(trajectories) - 2)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return (
        [trajectories[int(index)] for index in train_indices],
        [trajectories[int(index)] for index in test_indices],
        {
            "train_pool": int(len(train_indices)),
            "test": int(len(test_indices)),
            "test_fraction": test_fraction,
            "seed": seed,
        },
    )


def repeated_fit_validation_split(
    train_pool: list[np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_pool))
    n_validation = max(1, int(round(validation_fraction * len(train_pool))))
    n_validation = min(n_validation, len(train_pool) - 1)
    validation_indices = indices[:n_validation]
    fit_indices = indices[n_validation:]
    return (
        [train_pool[int(index)] for index in fit_indices],
        [train_pool[int(index)] for index in validation_indices],
        {
            "fit": int(len(fit_indices)),
            "validation": int(len(validation_indices)),
            "validation_fraction": validation_fraction,
        },
    )


def build_components(
    args: argparse.Namespace,
    archive: SegmentArchive,
    *,
    seed: int,
) -> dict[str, object]:
    gaussian = GaussianBayesSegmentSampler(
        archive=archive,
        diffusivity=args.diffusivity,
        candidate_limit=args.candidate_limit,
        bandwidth_multiplier=args.gaussian_bandwidth,
        seed=seed,
    )
    knn = ConditionalSegmentSampler(
        archive=archive,
        k=args.knn_k,
        temperature=args.knn_temperature,
        seed=seed,
    )
    hybrid = HybridContrastiveGaussianSampler.fit(
        archive=archive,
        seed=seed,
        negative_ratio=args.contrastive_negative_ratio,
        epochs=args.contrastive_epochs,
        candidate_limit=args.candidate_limit,
        diffusivity=args.diffusivity,
        bandwidth_multiplier=args.gaussian_bandwidth,
        gaussian_weight=1.0,
        learned_weight=args.hybrid_learned_weight,
    )
    pair_rerank = PairAwareRerankGaussianSampler.fit(
        archive=archive,
        seed=seed,
        n_bins=args.adaptive_bins,
        horizon_segments=args.rerank_horizon_segments,
        neighbor_k=args.pair_neighbor_k,
        diffusivity=args.diffusivity,
        bandwidth_multiplier=args.gaussian_bandwidth,
        candidate_limit=args.candidate_limit,
        pair_weight=args.pair_rerank_weight,
    )
    return {
        "gaussian_bayes": gaussian,
        "knn_conditional": knn,
        "hybrid": hybrid,
        "pair_rerank": pair_rerank,
    }


def simplex_grid(n_components: int, step: float) -> list[tuple[float, ...]]:
    levels = int(round(1.0 / step))
    if abs(levels * step - 1.0) > 1e-9:
        raise ValueError("--grid-step must divide 1.0")
    combos = []
    for counts in itertools.product(range(levels + 1), repeat=n_components):
        if sum(counts) == levels:
            combos.append(tuple(count / levels for count in counts))
    return combos


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def weight_dict(weights: np.ndarray | tuple[float, ...] | list[float]) -> dict[str, float]:
    return {name: float(weight) for name, weight in zip(COMPONENT_ORDER, weights)}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
