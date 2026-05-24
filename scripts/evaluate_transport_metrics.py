from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import (  # noqa: E402
    ConditionalSegmentSampler,
    ContrastiveTransitionSampler,
    AdaptiveGaussianBayesSegmentSampler,
    GaussianBayesSegmentSampler,
    HybridContrastiveGaussianSampler,
    PairAwareRerankGaussianSampler,
    ShortHorizonRerankGaussianSampler,
    SegmentArchive,
    UnconditionalSegmentSampler,
    breakthrough_score,
    breakthrough_times,
    dilution_index,
    load_trajectories,
    pair_separation_summary,
    quantile_curve_mae,
    reaction_encounter_probability,
    scalar_curve_log_mae,
    summarize_breakthroughs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TTA samplers on BTC, dilution, pair separation, and reaction proxy metrics."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-input", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--planes", default="6,10,14")
    parser.add_argument("--time-indices", default="100,200,300,400")
    parser.add_argument("--segment-steps", type=int, default=36)
    parser.add_argument("--match-steps", type=int, default=20)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--gaussian-bandwidth", type=float, default=0.25)
    parser.add_argument("--gaussian-candidate-limit", type=int, default=256)
    parser.add_argument("--adaptive-bins", type=int, default=4)
    parser.add_argument("--adaptive-bandwidth", type=float, default=1.0)
    parser.add_argument("--rerank-horizon-segments", type=int, default=3)
    parser.add_argument("--rerank-horizon-weight", type=float, default=0.25)
    parser.add_argument("--pair-rerank-weight", type=float, default=0.25)
    parser.add_argument("--pair-neighbor-k", type=int, default=32)
    parser.add_argument("--contrastive-epochs", type=int, default=600)
    parser.add_argument("--contrastive-negative-ratio", type=int, default=4)
    parser.add_argument("--hybrid-gaussian-weight", type=float, default=1.0)
    parser.add_argument("--hybrid-learned-weight", type=float, default=1.0)
    parser.add_argument("--knn-k", type=int, default=96)
    parser.add_argument("--knn-temperature", type=float, default=0.8)
    parser.add_argument("--n-generated", type=int, default=90)
    parser.add_argument("--n-segments", type=int, default=32)
    parser.add_argument("--bin-size", type=float, default=3.0)
    parser.add_argument("--pair-samples", type=int, default=3000)
    parser.add_argument("--reaction-radius", type=float, default=3.0)
    parser.add_argument("--missing-penalty", type=float, default=250.0)
    parser.add_argument("--generated-origin-source", choices=["train", "reference", "zero"], default="train")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "transport_metric_evaluation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planes = parse_float_list(args.planes)
    time_indices = parse_int_list(args.time_indices)

    train, reference = load_or_split(args)
    archive = SegmentArchive.from_trajectories(
        train,
        segment_steps=args.segment_steps,
        match_steps=args.match_steps,
        stride=args.stride,
        dt=args.dt,
    )

    samplers = {
        "unconditional": UnconditionalSegmentSampler(archive=archive, seed=args.seed),
        "knn_conditional": ConditionalSegmentSampler(
            archive=archive,
            k=args.knn_k,
            temperature=args.knn_temperature,
            seed=args.seed,
        ),
        "gaussian_bayes": GaussianBayesSegmentSampler(
            archive=archive,
            diffusivity=args.diffusivity,
            candidate_limit=args.gaussian_candidate_limit,
            bandwidth_multiplier=args.gaussian_bandwidth,
            seed=args.seed,
        ),
        "adaptive_gaussian": AdaptiveGaussianBayesSegmentSampler.fit(
            archive=archive,
            n_bins=args.adaptive_bins,
            bandwidth_multiplier=args.adaptive_bandwidth,
            candidate_limit=args.gaussian_candidate_limit,
            seed=args.seed,
        ),
        "horizon_rerank": ShortHorizonRerankGaussianSampler.fit(
            archive=archive,
            seed=args.seed,
            n_bins=args.adaptive_bins,
            horizon_segments=args.rerank_horizon_segments,
            diffusivity=args.diffusivity,
            bandwidth_multiplier=args.gaussian_bandwidth,
            candidate_limit=args.gaussian_candidate_limit,
            horizon_weight=args.rerank_horizon_weight,
        ),
        "pair_rerank": PairAwareRerankGaussianSampler.fit(
            archive=archive,
            seed=args.seed,
            n_bins=args.adaptive_bins,
            horizon_segments=args.rerank_horizon_segments,
            neighbor_k=args.pair_neighbor_k,
            diffusivity=args.diffusivity,
            bandwidth_multiplier=args.gaussian_bandwidth,
            candidate_limit=args.gaussian_candidate_limit,
            pair_weight=args.pair_rerank_weight,
        ),
        "contrastive": ContrastiveTransitionSampler.fit(
            archive=archive,
            seed=args.seed,
            negative_ratio=args.contrastive_negative_ratio,
            epochs=args.contrastive_epochs,
            candidate_limit=args.gaussian_candidate_limit,
        ),
        "hybrid": HybridContrastiveGaussianSampler.fit(
            archive=archive,
            seed=args.seed,
            negative_ratio=args.contrastive_negative_ratio,
            epochs=args.contrastive_epochs,
            candidate_limit=args.gaussian_candidate_limit,
            diffusivity=args.diffusivity,
            bandwidth_multiplier=args.gaussian_bandwidth,
            gaussian_weight=args.hybrid_gaussian_weight,
            learned_weight=args.hybrid_learned_weight,
        ),
    }

    reference_metrics = evaluate_ensemble(
        reference,
        planes=planes,
        time_indices=time_indices,
        bin_size=args.bin_size,
        pair_samples=args.pair_samples,
        reaction_radius=args.reaction_radius,
        seed=args.seed,
    )

    origin_pool = choose_origin_pool(args.generated_origin_source, train, reference)
    generated_payload: dict[str, dict[str, object]] = {}
    rng = np.random.default_rng(args.seed)
    for name, sampler in samplers.items():
        generated = generate_with_origins(
            sampler,
            n_trajectories=args.n_generated,
            n_segments=args.n_segments,
            origin_pool=origin_pool,
            rng=rng,
        )
        metrics = evaluate_ensemble(
            generated,
            planes=planes,
            time_indices=time_indices,
            bin_size=args.bin_size,
            pair_samples=args.pair_samples,
            reaction_radius=args.reaction_radius,
            seed=args.seed,
        )
        generated_payload[name] = {
            "metrics": metrics,
            "errors": compare_metrics(
                reference_metrics,
                metrics,
                missing_penalty=args.missing_penalty,
            ),
        }

    payload = {
        "input": str(args.input),
        "n_train": len(train),
        "n_reference": len(reference),
        "archive": {
            "size": archive.size,
            "segment_steps": archive.segment_steps,
            "match_steps": archive.match_steps,
            "dt": archive.dt,
        },
        "settings": {
            "planes": planes,
            "time_indices": time_indices,
            "bin_size": args.bin_size,
            "pair_samples": args.pair_samples,
            "reaction_radius": args.reaction_radius,
            "generated_origin_source": args.generated_origin_source,
        },
        "reference": reference_metrics,
        "generated": generated_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Training trajectories: {len(train)}")
    print(f"Reference trajectories: {len(reference)}")
    print(f"Archive segments: {archive.size}")
    print(f"Wrote metrics: {args.output}")
    print()
    print_metric_table(generated_payload)


def evaluate_ensemble(
    trajectories: list[np.ndarray],
    *,
    planes: list[float],
    time_indices: list[int],
    bin_size: float,
    pair_samples: int,
    reaction_radius: float,
    seed: int,
) -> dict[str, object]:
    return {
        "breakthrough": summarize_breakthroughs(breakthrough_times(trajectories, planes)),
        "dilution": dilution_index(trajectories, time_indices, bin_size=bin_size),
        "pair_separation": pair_separation_summary(
            trajectories,
            time_indices,
            n_pairs=pair_samples,
            seed=seed,
        ),
        "reaction": reaction_encounter_probability(
            trajectories,
            reaction_radius=reaction_radius,
            n_pairs=pair_samples,
            max_time_index=max(time_indices),
            seed=seed,
        ),
    }


def compare_metrics(
    reference: dict[str, object],
    generated: dict[str, object],
    *,
    missing_penalty: float,
) -> dict[str, float]:
    ref_breakthrough = reference["breakthrough"]
    gen_breakthrough = generated["breakthrough"]
    btc = breakthrough_score(ref_breakthrough, gen_breakthrough, missing_penalty=missing_penalty)

    ref_dilution = reference["dilution"]
    gen_dilution = generated["dilution"]
    dilution_log_mae = scalar_curve_log_mae(ref_dilution, gen_dilution, "dilution_index")

    ref_pairs = reference["pair_separation"]
    gen_pairs = generated["pair_separation"]
    pair_quantile_mae = quantile_curve_mae(ref_pairs, gen_pairs)

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


def load_or_split(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray]]:
    trajectories = load_trajectories(args.input, key=args.key)
    if args.reference_input is not None:
        return trajectories, load_trajectories(args.reference_input, key=args.key)
    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are needed when no reference input is provided")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    split = max(1, min(len(trajectories) - 1, int(round(args.train_fraction * len(trajectories)))))
    return trajectories[:split], trajectories[split:]


def print_metric_table(generated_payload: dict[str, dict[str, object]]) -> None:
    print("Sampler metric errors")
    print("sampler             btc_score  dilution_log  pair_mae  reaction_abs")
    for name, payload in generated_payload.items():
        errors = payload["errors"]
        print(
            f"{name:<18}"
            f"{errors['btc_score']:10.2f}"
            f"{errors['dilution_log_mae']:14.3f}"
            f"{errors['pair_quantile_mae']:10.2f}"
            f"{errors['reaction_abs_error']:14.3f}"
        )


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
