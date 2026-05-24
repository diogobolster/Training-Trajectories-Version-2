from __future__ import annotations

import argparse
import itertools
import json
import sys
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
    split_three_way,
)


COMPONENT_ORDER = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select sampler mixture weights on validation metrics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--fit-fraction", type=float, default=0.50)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--planes", default="6,10,14")
    parser.add_argument("--time-indices", default="100,200,300,400")
    parser.add_argument("--segment-steps", type=int, default=36)
    parser.add_argument("--match-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--gaussian-bandwidth", type=float, default=0.25)
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=96)
    parser.add_argument("--knn-temperature", type=float, default=0.8)
    parser.add_argument("--contrastive-epochs", type=int, default=700)
    parser.add_argument("--contrastive-negative-ratio", type=int, default=6)
    parser.add_argument("--hybrid-learned-weight", type=float, default=0.25)
    parser.add_argument("--pair-rerank-weight", type=float, default=0.25)
    parser.add_argument("--pair-neighbor-k", type=int, default=32)
    parser.add_argument("--rerank-horizon-segments", type=int, default=3)
    parser.add_argument("--adaptive-bins", type=int, default=4)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--n-validation-generated", type=int, default=70)
    parser.add_argument("--n-test-generated", type=int, default=90)
    parser.add_argument("--n-segments", type=int, default=32)
    parser.add_argument("--bin-size", type=float, default=3.0)
    parser.add_argument("--pair-samples", type=int, default=2000)
    parser.add_argument("--reaction-radius", type=float, default=3.0)
    parser.add_argument("--btc-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=20.0)
    parser.add_argument("--dilution-weight", type=float, default=120.0)
    parser.add_argument("--reaction-weight", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sampler_mixture_selection.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.input, key=args.key)
    fit, validation, test = split_three_way(trajectories, args.fit_fraction, args.validation_fraction)
    archive = SegmentArchive.from_trajectories(
        fit,
        segment_steps=args.segment_steps,
        match_steps=args.match_steps,
        dt=args.dt,
    )
    components = build_components(args, archive)
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

    validation_reference = evaluate_ensemble(validation, settings)
    validation_origin_pool = choose_origin_pool("train", fit, validation)

    validation_results = []
    for weights in simplex_grid(len(COMPONENT_ORDER), args.grid_step):
        mixture = MixtureSegmentSampler(
            archive=archive,
            components=[components[name] for name in COMPONENT_ORDER],
            component_weights=np.asarray(weights),
            seed=args.seed,
        )
        generated = generate_with_origins(
            mixture,
            n_trajectories=args.n_validation_generated,
            n_segments=args.n_segments,
            origin_pool=validation_origin_pool,
            rng=np.random.default_rng(args.seed),
        )
        metrics = evaluate_ensemble(generated, settings)
        errors = compare_metrics(
            validation_reference,
            metrics,
            missing_penalty=settings.missing_penalty,
        )
        score = multi_objective_score(errors, objective_weights)
        validation_results.append(
            {
                "weights": weight_dict(weights),
                "score": score,
                "errors": errors,
            }
        )

    validation_results.sort(key=lambda item: item["score"])
    best_weights = [validation_results[0]["weights"][name] for name in COMPONENT_ORDER]

    test_reference = evaluate_ensemble(test, settings)
    test_origin_pool = choose_origin_pool("train", fit, test)
    test_payload = {}
    selected_mixture = MixtureSegmentSampler(
        archive=archive,
        components=[components[name] for name in COMPONENT_ORDER],
        component_weights=np.asarray(best_weights),
        seed=args.seed,
    )
    eval_samplers = {"selected_mixture": selected_mixture, **components}
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

    payload = {
        "input": str(args.input),
        "splits": {
            "fit": len(fit),
            "validation": len(validation),
            "test": len(test),
        },
        "archive": {
            "size": archive.size,
            "segment_steps": archive.segment_steps,
            "match_steps": archive.match_steps,
        },
        "component_order": COMPONENT_ORDER,
        "objective_weights": {
            "btc": objective_weights.btc,
            "pair": objective_weights.pair,
            "dilution": objective_weights.dilution,
            "reaction": objective_weights.reaction,
        },
        "validation_top": validation_results[:12],
        "selected_weights": validation_results[0]["weights"],
        "validation_reference": validation_reference,
        "test_reference": test_reference,
        "test": test_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Fit/validation/test: {len(fit)}/{len(validation)}/{len(test)}")
    print(f"Archive segments: {archive.size}")
    print(f"Wrote mixture selection: {args.output}")
    print(f"Selected weights: {validation_results[0]['weights']}")
    print()
    print("Test errors")
    print("sampler             objective  btc_score  pair_mae  dilution_log  reaction_abs")
    for name, item in sorted(test_payload.items(), key=lambda pair: pair[1]["objective"]):
        errors = item["errors"]
        print(
            f"{name:<18}"
            f"{item['objective']:10.2f}"
            f"{errors['btc_score']:11.2f}"
            f"{errors['pair_quantile_mae']:10.2f}"
            f"{errors['dilution_log_mae']:14.3f}"
            f"{errors['reaction_abs_error']:14.3f}"
        )


def build_components(args: argparse.Namespace, archive: SegmentArchive) -> dict[str, object]:
    gaussian = GaussianBayesSegmentSampler(
        archive=archive,
        diffusivity=args.diffusivity,
        candidate_limit=args.candidate_limit,
        bandwidth_multiplier=args.gaussian_bandwidth,
        seed=args.seed,
    )
    knn = ConditionalSegmentSampler(
        archive=archive,
        k=args.knn_k,
        temperature=args.knn_temperature,
        seed=args.seed,
    )
    hybrid = HybridContrastiveGaussianSampler.fit(
        archive=archive,
        seed=args.seed,
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
        seed=args.seed,
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


def weight_dict(weights: list[float] | tuple[float, ...]) -> dict[str, float]:
    return {name: float(weight) for name, weight in zip(COMPONENT_ORDER, weights)}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()

