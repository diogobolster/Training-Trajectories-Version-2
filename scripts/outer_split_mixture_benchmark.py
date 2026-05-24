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


EVAL_ORDER = [
    "bootstrap_mean_mixture",
    "pooled_validation_mixture",
    "gaussian_bayes",
    "knn_conditional",
    "hybrid",
    "pair_rerank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated-validation mixture selection over multiple held-out test splits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.30,
        help="Fraction of the non-test pool used for validation in each inner repeat.",
    )
    parser.add_argument("--n-outer-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=4)
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
    parser.add_argument("--contrastive-epochs", type=int, default=400)
    parser.add_argument("--contrastive-negative-ratio", type=int, default=6)
    parser.add_argument("--hybrid-learned-weight", type=float, default=0.25)
    parser.add_argument("--pair-rerank-weight", type=float, default=0.25)
    parser.add_argument("--pair-neighbor-k", type=int, default=32)
    parser.add_argument("--rerank-horizon-segments", type=int, default=3)
    parser.add_argument("--adaptive-bins", type=int, default=4)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--n-validation-generated", type=int, default=45)
    parser.add_argument("--n-test-generated", type=int, default=80)
    parser.add_argument("--n-segments", type=int, default=32)
    parser.add_argument("--bin-size", type=float, default=3.0)
    parser.add_argument("--pair-samples", type=int, default=1200)
    parser.add_argument("--reaction-radius", type=float, default=3.0)
    parser.add_argument("--btc-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=20.0)
    parser.add_argument("--dilution-weight", type=float, default=120.0)
    parser.add_argument("--reaction-weight", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_outer_split_mixture_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    trajectories = load_trajectories(args.input, key=args.key)
    base_settings = MetricSettings(
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

    outer_results = []
    for outer in range(args.n_outer_splits):
        outer_seed = args.seed + args.outer_seed_stride * outer
        result = run_outer_split(
            args=args,
            trajectories=trajectories,
            outer_index=outer,
            outer_seed=outer_seed,
            base_settings=base_settings,
            objective_weights=objective_weights,
            grid=grid,
        )
        outer_results.append(result)
        winner = result["ranking"][0]
        print(
            f"outer {outer + 1}/{args.n_outer_splits}: "
            f"winner {winner['sampler']} objective {winner['objective']:.2f}; "
            f"mean weights {result['mean_selected_weights']}",
            flush=True,
        )

    summary = summarize_outer_results(outer_results)
    payload = {
        "input": str(args.input),
        "n_outer_splits": args.n_outer_splits,
        "n_repeats": args.n_repeats,
        "component_order": COMPONENT_ORDER,
        "objective_weights": {
            "btc": objective_weights.btc,
            "pair": objective_weights.pair,
            "dilution": objective_weights.dilution,
            "reaction": objective_weights.reaction,
        },
        "outer_results": outer_results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Wrote outer-split benchmark: {args.output}")
    print()
    print("Summary over held-out test splits")
    print("sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h")
    for name in sorted(summary["samplers"], key=lambda item: summary["samplers"][item]["mean_objective"]):
        item = summary["samplers"][name]
        print(
            f"{name:<26}"
            f"{item['mean_objective']:9.2f}"
            f"{item['std_objective']:10.2f}"
            f"{item['mean_rank']:11.2f}"
            f"{item['wins']:6d}"
            f"{item['beats_gaussian_bayes']:9d}"
            f"{item['beats_hybrid']:9d}"
        )
    print()
    print(f"Mean outer selected weights: {summary['mean_selected_weights']}")


def validate_args(args: argparse.Namespace) -> None:
    if args.n_outer_splits <= 0:
        raise ValueError("--n-outer-splits must be positive")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be positive")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")


def run_outer_split(
    *,
    args: argparse.Namespace,
    trajectories: list[np.ndarray],
    outer_index: int,
    outer_seed: int,
    base_settings: MetricSettings,
    objective_weights: ObjectiveWeights,
    grid: list[tuple[float, ...]],
) -> dict[str, object]:
    train_pool, test, split_payload = fixed_train_test_split(
        trajectories,
        test_fraction=args.test_fraction,
        seed=outer_seed,
    )
    settings = replace(base_settings, seed=outer_seed)
    repeat_payloads = []
    score_sums = {weights: 0.0 for weights in grid}
    selected_weight_vectors = []

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
                "validation_top": validation_results[:5],
            }
        )
        print(
            f"  outer {outer_index + 1}, repeat {repeat + 1}/{args.n_repeats}: "
            f"selected {validation_results[0]['weights']} "
            f"score {validation_results[0]['score']:.2f}",
            flush=True,
        )

    mean_weights = normalize_weights(np.mean(np.asarray(selected_weight_vectors), axis=0))
    pooled_best_weights = min(grid, key=lambda weights: score_sums[weights] / args.n_repeats)
    mean_score_by_weight = [
        {
            "weights": weight_dict(weights),
            "mean_score": score_sums[weights] / args.n_repeats,
        }
        for weights in grid
    ]
    mean_score_by_weight.sort(key=lambda item: item["mean_score"])

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
    mean_mixture = MixtureSegmentSampler(
        archive=archive,
        components=[components[name] for name in COMPONENT_ORDER],
        component_weights=mean_weights,
        seed=outer_seed,
    )
    pooled_mixture = MixtureSegmentSampler(
        archive=archive,
        components=[components[name] for name in COMPONENT_ORDER],
        component_weights=np.asarray(pooled_best_weights),
        seed=outer_seed,
    )
    samplers = {
        "bootstrap_mean_mixture": mean_mixture,
        "pooled_validation_mixture": pooled_mixture,
        **components,
    }
    test_payload = {}
    for name in EVAL_ORDER:
        sampler = samplers[name]
        generated = generate_with_origins(
            sampler,
            n_trajectories=args.n_test_generated,
            n_segments=args.n_segments,
            origin_pool=test_origin_pool,
            rng=np.random.default_rng(outer_seed),
        )
        metrics = evaluate_ensemble(generated, settings)
        errors = compare_metrics(test_reference, metrics, missing_penalty=settings.missing_penalty)
        test_payload[name] = {
            "errors": errors,
            "objective": multi_objective_score(errors, objective_weights),
            "metrics": metrics,
        }

    ranking = [
        {"sampler": name, "objective": test_payload[name]["objective"]}
        for name in sorted(test_payload, key=lambda sampler: test_payload[sampler]["objective"])
    ]
    for rank, item in enumerate(ranking, start=1):
        test_payload[item["sampler"]]["rank"] = rank
        item["rank"] = rank

    return {
        "outer_index": outer_index,
        "seed": outer_seed,
        "splits": split_payload,
        "archive": {
            "size": archive.size,
            "segment_steps": archive.segment_steps,
            "match_steps": archive.match_steps,
        },
        "repeat_results": repeat_payloads,
        "mean_selected_weights": weight_dict(mean_weights),
        "pooled_validation_weights": weight_dict(pooled_best_weights),
        "pooled_validation_top": mean_score_by_weight[:8],
        "test": test_payload,
        "ranking": ranking,
    }


def summarize_outer_results(outer_results: list[dict[str, object]]) -> dict[str, object]:
    sampler_names = list(EVAL_ORDER)
    baseline_gaussian = "gaussian_bayes"
    baseline_hybrid = "hybrid"
    sampler_summary = {}
    for name in sampler_names:
        objectives = np.asarray(
            [result["test"][name]["objective"] for result in outer_results],
            dtype=float,
        )
        ranks = np.asarray([result["test"][name]["rank"] for result in outer_results], dtype=float)
        gaussian_objectives = np.asarray(
            [result["test"][baseline_gaussian]["objective"] for result in outer_results],
            dtype=float,
        )
        hybrid_objectives = np.asarray(
            [result["test"][baseline_hybrid]["objective"] for result in outer_results],
            dtype=float,
        )
        sampler_summary[name] = {
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0,
            "min_objective": float(np.min(objectives)),
            "max_objective": float(np.max(objectives)),
            "mean_rank": float(np.mean(ranks)),
            "wins": int(np.sum(ranks == 1)),
            "beats_gaussian_bayes": int(np.sum(objectives < gaussian_objectives)),
            "beats_hybrid": int(np.sum(objectives < hybrid_objectives)),
        }

    weight_matrix = np.asarray(
        [
            [result["mean_selected_weights"][name] for name in COMPONENT_ORDER]
            for result in outer_results
        ],
        dtype=float,
    )
    return {
        "samplers": sampler_summary,
        "mean_selected_weights": weight_dict(np.mean(weight_matrix, axis=0)),
        "std_selected_weights": weight_dict(np.std(weight_matrix, axis=0, ddof=1))
        if len(outer_results) > 1
        else weight_dict(np.zeros(len(COMPONENT_ORDER))),
    }


if __name__ == "__main__":
    main()
