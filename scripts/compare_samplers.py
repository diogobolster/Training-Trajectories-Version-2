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
    generate_channel_trajectories,
    load_trajectories,
    summarize_breakthroughs,
    velocity_autocorrelation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TTA sampler baselines on synthetic or real trajectory data."
    )
    parser.add_argument("--input", type=Path, help="Optional .npy, .npz, or .csv trajectory file.")
    parser.add_argument("--reference-input", type=Path, help="Optional separate reference trajectory file.")
    parser.add_argument("--key", help="NPZ key for trajectory array.")
    parser.add_argument("--segment-steps", type=int, default=36)
    parser.add_argument("--match-steps", type=int, default=6)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--diffusivity", type=float, default=4e-5)
    parser.add_argument("--n-generated", type=int, default=180)
    parser.add_argument("--n-segments", type=int, default=26)
    parser.add_argument("--knn-k", type=int, default=96)
    parser.add_argument("--knn-temperature", type=float, default=0.8)
    parser.add_argument("--gaussian-bandwidth", type=float, default=1.0)
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
    parser.add_argument("--missing-penalty", type=float, default=250.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--planes", default="4,8,12", help="Comma-separated x control planes.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sampler_comparison.json")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planes = [float(item) for item in args.planes.split(",") if item.strip()]

    train, reference = load_or_generate_data(args)
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

    reference_summary = summarize_breakthroughs(breakthrough_times(reference, planes))
    reference_corr = velocity_autocorrelation(reference, max_lag=40)

    generated_payload = {}
    for name, sampler in samplers.items():
        generated = sampler.generate_many(
            n_trajectories=args.n_generated,
            n_segments=args.n_segments,
        )
        summary = summarize_breakthroughs(breakthrough_times(generated, planes))
        corr = velocity_autocorrelation(generated, max_lag=40)
        score = breakthrough_score(reference_summary, summary, missing_penalty=args.missing_penalty)
        generated_payload[name] = {
            "breakthrough": summary,
            **score,
            "velocity_autocorrelation_lags_0_10": [float(x) for x in corr[:11]],
        }

    payload = {
        "archive": {
            "size": archive.size,
            "segment_steps": archive.segment_steps,
            "match_steps": archive.match_steps,
            "dt": archive.dt,
        },
        "reference": {
            "n_trajectories": len(reference),
            "breakthrough": reference_summary,
            "velocity_autocorrelation_lags_0_10": [float(x) for x in reference_corr[:11]],
        },
        "generated": generated_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Training trajectories: {len(train)}")
    print(f"Reference trajectories: {len(reference)}")
    print(f"Archive segments: {archive.size}")
    print(f"Wrote comparison: {args.output}")
    print()
    print_breakthrough_table(reference_summary, generated_payload, planes)


def load_or_generate_data(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if args.input is None:
        train = generate_channel_trajectories(n_trajectories=350, n_steps=900, seed=7)
        reference = generate_channel_trajectories(n_trajectories=180, n_steps=900, seed=99)
        return train, reference

    trajectories = load_trajectories(args.input, key=args.key)
    if args.reference_input is not None:
        reference = load_trajectories(args.reference_input, key=args.key)
        return trajectories, reference

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    split = max(1, min(len(trajectories) - 1, int(round(args.train_fraction * len(trajectories)))))
    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are needed when no reference input is provided")
    return trajectories[:split], trajectories[split:]


def print_breakthrough_table(
    reference_summary: dict[float, dict[str, float]],
    generated_payload: dict[str, dict[str, object]],
    planes: list[float],
) -> None:
    print("Breakthrough q10/q50/q90 by control plane")
    for plane in planes:
        ref = reference_summary[plane]
        print(f"x={plane:>6.2f} | reference      {format_quantiles(ref)}")
        for name, payload in generated_payload.items():
            summary = payload["breakthrough"][plane]  # type: ignore[index]
            score = payload["score"]
            coverage_deficit = payload["coverage_deficit"]
            print(
                f"         | {name:<14} {format_quantiles(summary)} "
                f"| score={score:6.2f} cov_def={coverage_deficit:4.2f}"
            )
        print()


def format_quantiles(stats: dict[str, float]) -> str:
    return f"{stats['q10']:7.1f} {stats['q50']:7.1f} {stats['q90']:7.1f}"


if __name__ == "__main__":
    main()
