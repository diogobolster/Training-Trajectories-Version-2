from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_mixture_selection import (  # noqa: E402
    COMPONENT_ORDER,
    build_components,
    fixed_train_test_split,
    normalize_weights,
    parse_float_list,
    parse_int_list,
    repeated_fit_validation_split,
    simplex_grid,
    weight_dict,
)
from tta_v2 import (  # noqa: E402
    MetricSettings,
    MixtureSegmentSampler,
    ObjectiveWeights,
    SegmentArchive,
    choose_origin_pool,
    compare_metrics,
    evaluate_ensemble,
    generate_with_origins,
    load_trajectories,
    multi_objective_score,
)


BASE_COMPONENTS = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
MIXTURE_RULES = ["pooled_validation_mixture", "bootstrap_mean_mixture"]
WEIGHT_REGIMES = {
    "balanced": ObjectiveWeights(btc=1.0, pair=20.0, dilution=120.0, reaction=1000.0),
    "breakthrough_only": ObjectiveWeights(btc=1.0, pair=0.0, dilution=0.0, reaction=0.0),
    "btc_heavy": ObjectiveWeights(btc=3.0, pair=10.0, dilution=60.0, reaction=500.0),
    "pair_heavy": ObjectiveWeights(btc=0.5, pair=60.0, dilution=80.0, reaction=500.0),
    "dilution_heavy": ObjectiveWeights(btc=0.5, pair=10.0, dilution=360.0, reaction=500.0),
    "reaction_light": ObjectiveWeights(btc=1.0, pair=20.0, dilution=120.0, reaction=100.0),
    "reaction_heavy": ObjectiveWeights(btc=0.5, pair=10.0, dilution=60.0, reaction=3000.0),
    "no_reaction": ObjectiveWeights(btc=1.0, pair=20.0, dilution=120.0, reaction=0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test sampler selection sensitivity to multi-objective scoring weights."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--regimes", default=",".join(WEIGHT_REGIMES))
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--n-outer-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--outer-seed-stride", type=int, default=7919)
    parser.add_argument("--inner-seed-stride", type=int, default=1009)
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
    parser.add_argument("--contrastive-epochs", type=int, default=300)
    parser.add_argument("--contrastive-negative-ratio", type=int, default=6)
    parser.add_argument("--hybrid-learned-weight", type=float, default=0.25)
    parser.add_argument("--pair-rerank-weight", type=float, default=0.25)
    parser.add_argument("--pair-neighbor-k", type=int, default=32)
    parser.add_argument("--rerank-horizon-segments", type=int, default=3)
    parser.add_argument("--adaptive-bins", type=int, default=4)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--n-validation-generated", type=int, default=35)
    parser.add_argument("--n-test-generated", type=int, default=60)
    parser.add_argument("--n-segments", type=int, default=32)
    parser.add_argument("--bin-size", type=float, default=3.0)
    parser.add_argument("--pair-samples", type=int, default=1000)
    parser.add_argument("--reaction-radius", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_objective_weight_sensitivity.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    regimes = parse_regimes(args.regimes)

    trajectories = load_trajectories(args.input, key=args.key)
    base_settings = MetricSettings(
        planes=parse_float_list(args.planes),
        time_indices=parse_int_list(args.time_indices),
        bin_size=args.bin_size,
        pair_samples=args.pair_samples,
        reaction_radius=args.reaction_radius,
        seed=args.seed,
    )
    grid = simplex_grid(len(COMPONENT_ORDER), args.grid_step)
    outer_results = []

    for outer in range(args.n_outer_splits):
        outer_seed = args.seed + args.outer_seed_stride * outer
        result = run_outer_split(
            args=args,
            trajectories=trajectories,
            outer_index=outer,
            outer_seed=outer_seed,
            base_settings=base_settings,
            regimes=regimes,
            grid=grid,
        )
        outer_results.append(result)
        print(f"outer {outer + 1}/{args.n_outer_splits} complete", flush=True)
        for regime_name in regimes:
            winner = result["regime_results"][regime_name]["ranking"][0]
            print(
                f"  {regime_name:<15} winner {winner['sampler']:<27}"
                f" objective {winner['objective']:.2f}",
                flush=True,
            )

    summary = summarize_results(outer_results, regimes)
    payload = {
        "input": str(args.input),
        "n_outer_splits": args.n_outer_splits,
        "n_repeats": args.n_repeats,
        "component_order": COMPONENT_ORDER,
        "regime_weights": {
            name: objective_weights_payload(weights) for name, weights in regimes.items()
        },
        "outer_results": outer_results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Wrote objective-weight sensitivity: {args.output}")
    print()
    print("Regime summary")
    print("regime           best_mean_sampler            mean_obj  mean_rank  wins  beats_g  beats_h")
    for regime_name in regimes:
        sampler_summary = summary[regime_name]["samplers"]
        best_name = min(sampler_summary, key=lambda name: sampler_summary[name]["mean_objective"])
        best = sampler_summary[best_name]
        print(
            f"{regime_name:<16}"
            f"{best_name:<28}"
            f"{best['mean_objective']:9.2f}"
            f"{best['mean_rank']:11.2f}"
            f"{best['wins']:6d}"
            f"{best['beats_gaussian_bayes']:9d}"
            f"{best['beats_hybrid']:9d}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.n_outer_splits <= 0:
        raise ValueError("--n-outer-splits must be positive")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be positive")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")


def parse_regimes(value: str) -> dict[str, ObjectiveWeights]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in WEIGHT_REGIMES]
    if unknown:
        raise ValueError(f"unknown regimes: {', '.join(unknown)}")
    return {name: WEIGHT_REGIMES[name] for name in names}


def run_outer_split(
    *,
    args: argparse.Namespace,
    trajectories: list[np.ndarray],
    outer_index: int,
    outer_seed: int,
    base_settings: MetricSettings,
    regimes: dict[str, ObjectiveWeights],
    grid: list[tuple[float, ...]],
) -> dict[str, object]:
    train_pool, test, split_payload = fixed_train_test_split(
        trajectories,
        test_fraction=args.test_fraction,
        seed=outer_seed,
    )
    settings = replace(base_settings, seed=outer_seed)
    repeat_records = []
    selections = initialize_selection_state(regimes, grid)

    for repeat in range(args.n_repeats):
        repeat_seed = outer_seed + args.inner_seed_stride * repeat
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

        validation_errors_by_weight = {}
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
            validation_errors_by_weight[weights] = errors

        repeat_selections = {}
        for regime_name, objective_weights in regimes.items():
            scored = [
                (
                    weights,
                    multi_objective_score(validation_errors_by_weight[weights], objective_weights),
                    validation_errors_by_weight[weights],
                )
                for weights in grid
            ]
            scored.sort(key=lambda item: item[1])
            best_weights, best_score, best_errors = scored[0]
            selections[regime_name]["selected_vectors"].append(best_weights)
            for weights, score, _errors in scored:
                selections[regime_name]["score_sums"][weights] += score
            repeat_selections[regime_name] = {
                "selected_weights": weight_dict(best_weights),
                "selected_score": best_score,
                "selected_errors": best_errors,
                "validation_top": [
                    {
                        "weights": weight_dict(weights),
                        "score": score,
                        "errors": errors,
                    }
                    for weights, score, errors in scored[:5]
                ],
            }

        repeat_records.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "split": repeat_split,
                "archive_size": archive.size,
                "selections": repeat_selections,
            }
        )
        balanced = repeat_selections.get(next(iter(regimes)))
        print(
            f"  outer {outer_index + 1}, repeat {repeat + 1}/{args.n_repeats}: "
            f"{next(iter(regimes))} selected {balanced['selected_weights']}",
            flush=True,
        )

    selected_weights = finalize_selected_weights(selections, args.n_repeats)
    archive = SegmentArchive.from_trajectories(
        train_pool,
        segment_steps=args.segment_steps,
        match_steps=args.match_steps,
        stride=args.segment_stride,
        dt=args.dt,
    )
    components = build_components(args, archive, seed=outer_seed)
    test_reference = evaluate_ensemble(test, settings)
    test_origin_pool = choose_origin_pool("train", train_pool, test)
    generated_errors = evaluate_selected_samplers(
        args=args,
        archive=archive,
        components=components,
        selected_weights=selected_weights,
        test_reference=test_reference,
        test_origin_pool=test_origin_pool,
        settings=settings,
        seed=outer_seed,
    )
    regime_results = score_test_by_regime(generated_errors, selected_weights, regimes)

    return {
        "outer_index": outer_index,
        "seed": outer_seed,
        "splits": split_payload,
        "archive": {
            "size": archive.size,
            "segment_steps": archive.segment_steps,
            "match_steps": archive.match_steps,
        },
        "repeat_results": repeat_records,
        "selected_weights": selected_weights,
        "test_errors": generated_errors,
        "regime_results": regime_results,
    }


def initialize_selection_state(
    regimes: dict[str, ObjectiveWeights],
    grid: list[tuple[float, ...]],
) -> dict[str, dict[str, object]]:
    return {
        regime_name: {
            "selected_vectors": [],
            "score_sums": {weights: 0.0 for weights in grid},
        }
        for regime_name in regimes
    }


def finalize_selected_weights(
    selections: dict[str, dict[str, object]],
    n_repeats: int,
) -> dict[str, dict[str, object]]:
    selected_weights = {}
    for regime_name, state in selections.items():
        selected_vectors = np.asarray(state["selected_vectors"], dtype=float)
        mean_weights = normalize_weights(np.mean(selected_vectors, axis=0))
        pooled_weights = min(state["score_sums"], key=lambda weights: state["score_sums"][weights] / n_repeats)
        pooled_top = [
            {
                "weights": weight_dict(weights),
                "mean_score": state["score_sums"][weights] / n_repeats,
            }
            for weights in state["score_sums"]
        ]
        pooled_top.sort(key=lambda item: item["mean_score"])
        selected_weights[regime_name] = {
            "bootstrap_mean_mixture": weight_dict(mean_weights),
            "pooled_validation_mixture": weight_dict(pooled_weights),
            "pooled_validation_top": pooled_top[:8],
        }
    return selected_weights


def evaluate_selected_samplers(
    *,
    args: argparse.Namespace,
    archive: SegmentArchive,
    components: dict[str, object],
    selected_weights: dict[str, dict[str, object]],
    test_reference: dict[str, object],
    test_origin_pool: np.ndarray,
    settings: MetricSettings,
    seed: int,
) -> dict[str, dict[str, float]]:
    specs: dict[str, object] = dict(components)
    for regime_name, regime_weights in selected_weights.items():
        for rule in MIXTURE_RULES:
            weights = np.asarray(
                [regime_weights[rule][name] for name in COMPONENT_ORDER],
                dtype=float,
            )
            specs[f"{regime_name}::{rule}"] = MixtureSegmentSampler(
                archive=archive,
                components=[components[name] for name in COMPONENT_ORDER],
                component_weights=weights,
                seed=seed,
            )

    errors_by_sampler = {}
    for sampler_name, sampler in specs.items():
        generated = generate_with_origins(
            sampler,
            n_trajectories=args.n_test_generated,
            n_segments=args.n_segments,
            origin_pool=test_origin_pool,
            rng=np.random.default_rng(seed),
        )
        metrics = evaluate_ensemble(generated, settings)
        errors_by_sampler[sampler_name] = compare_metrics(
            test_reference,
            metrics,
            missing_penalty=settings.missing_penalty,
        )
    return errors_by_sampler


def score_test_by_regime(
    generated_errors: dict[str, dict[str, float]],
    selected_weights: dict[str, dict[str, object]],
    regimes: dict[str, ObjectiveWeights],
) -> dict[str, object]:
    regime_results = {}
    for regime_name, objective_weights in regimes.items():
        candidates = list(BASE_COMPONENTS) + [
            f"{regime_name}::pooled_validation_mixture",
            f"{regime_name}::bootstrap_mean_mixture",
        ]
        test = {}
        for candidate in candidates:
            display_name = candidate.split("::")[-1]
            errors = generated_errors[candidate]
            test[display_name] = {
                "errors": errors,
                "objective": multi_objective_score(errors, objective_weights),
            }
        ranking = [
            {"sampler": name, "objective": test[name]["objective"]}
            for name in sorted(test, key=lambda sampler: test[sampler]["objective"])
        ]
        for rank, item in enumerate(ranking, start=1):
            test[item["sampler"]]["rank"] = rank
            item["rank"] = rank
        regime_results[regime_name] = {
            "weights": selected_weights[regime_name],
            "test": test,
            "ranking": ranking,
        }
    return regime_results


def summarize_results(
    outer_results: list[dict[str, object]],
    regimes: dict[str, ObjectiveWeights],
) -> dict[str, object]:
    summary = {}
    candidate_names = BASE_COMPONENTS + MIXTURE_RULES
    for regime_name in regimes:
        sampler_summary = {}
        for candidate in candidate_names:
            objectives = np.asarray(
                [
                    result["regime_results"][regime_name]["test"][candidate]["objective"]
                    for result in outer_results
                ],
                dtype=float,
            )
            ranks = np.asarray(
                [
                    result["regime_results"][regime_name]["test"][candidate]["rank"]
                    for result in outer_results
                ],
                dtype=float,
            )
            gaussian_objectives = np.asarray(
                [
                    result["regime_results"][regime_name]["test"]["gaussian_bayes"]["objective"]
                    for result in outer_results
                ],
                dtype=float,
            )
            hybrid_objectives = np.asarray(
                [
                    result["regime_results"][regime_name]["test"]["hybrid"]["objective"]
                    for result in outer_results
                ],
                dtype=float,
            )
            sampler_summary[candidate] = {
                "mean_objective": float(np.mean(objectives)),
                "std_objective": float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0,
                "mean_rank": float(np.mean(ranks)),
                "wins": int(np.sum(ranks == 1)),
                "beats_gaussian_bayes": int(np.sum(objectives < gaussian_objectives)),
                "beats_hybrid": int(np.sum(objectives < hybrid_objectives)),
            }

        weight_summary = {}
        for rule in MIXTURE_RULES:
            matrix = np.asarray(
                [
                    [
                        result["regime_results"][regime_name]["weights"][rule][component]
                        for component in COMPONENT_ORDER
                    ]
                    for result in outer_results
                ],
                dtype=float,
            )
            weight_summary[rule] = {
                "mean": weight_dict(np.mean(matrix, axis=0)),
                "std": weight_dict(np.std(matrix, axis=0, ddof=1))
                if len(outer_results) > 1
                else weight_dict(np.zeros(len(COMPONENT_ORDER))),
            }
        summary[regime_name] = {
            "samplers": sampler_summary,
            "selected_weights": weight_summary,
        }
    return summary


def objective_weights_payload(weights: ObjectiveWeights) -> dict[str, float]:
    return {
        "btc": weights.btc,
        "pair": weights.pair,
        "dilution": weights.dilution,
        "reaction": weights.reaction,
    }


if __name__ == "__main__":
    main()
