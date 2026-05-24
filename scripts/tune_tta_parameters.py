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
    GaussianBayesSegmentSampler,
    SegmentArchive,
    breakthrough_score,
    breakthrough_times,
    load_trajectories,
    summarize_breakthroughs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan TTA segment lengths and transition-kernel parameters.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--planes", default="6,10,14")
    parser.add_argument("--segment-steps", default="16,20,24,30,36")
    parser.add_argument("--match-steps", default="3,4,5,6")
    parser.add_argument("--gaussian-bandwidths", default="0.5,1,2,4,8,16")
    parser.add_argument("--knn-k", type=int, default=96)
    parser.add_argument("--knn-temperature", type=float, default=0.8)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--n-generated", type=int, default=80)
    parser.add_argument("--n-segments", type=int, default=30)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--missing-penalty", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "tta_parameter_scan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planes = parse_float_list(args.planes)
    segment_steps = parse_int_list(args.segment_steps)
    match_steps = parse_int_list(args.match_steps)
    gaussian_bandwidths = parse_float_list(args.gaussian_bandwidths)

    trajectories = load_trajectories(args.input, key=args.key)
    train, reference = split_trajectories(trajectories, args.train_fraction)
    reference_summary = summarize_breakthroughs(breakthrough_times(reference, planes))

    results: list[dict[str, object]] = []
    for segment_step in segment_steps:
        for match_step in match_steps:
            if match_step >= segment_step:
                continue
            archive = SegmentArchive.from_trajectories(
                train,
                segment_steps=segment_step,
                match_steps=match_step,
                dt=args.dt,
            )

            knn = ConditionalSegmentSampler(
                archive=archive,
                k=args.knn_k,
                temperature=args.knn_temperature,
                seed=args.seed,
            )
            results.append(
                evaluate_sampler(
                    "knn_conditional",
                    knn,
                    reference_summary,
                    planes,
                    args.n_generated,
                    args.n_segments,
                    args.missing_penalty,
                    {
                        "segment_steps": segment_step,
                        "match_steps": match_step,
                        "archive_size": archive.size,
                        "k": args.knn_k,
                        "temperature": args.knn_temperature,
                    },
                )
            )

            for bandwidth in gaussian_bandwidths:
                gaussian = GaussianBayesSegmentSampler(
                    archive=archive,
                    diffusivity=args.diffusivity,
                    candidate_limit=256,
                    bandwidth_multiplier=bandwidth,
                    seed=args.seed,
                )
                results.append(
                    evaluate_sampler(
                        "gaussian_bayes",
                        gaussian,
                        reference_summary,
                        planes,
                        args.n_generated,
                        args.n_segments,
                        args.missing_penalty,
                        {
                            "segment_steps": segment_step,
                            "match_steps": match_step,
                            "archive_size": archive.size,
                            "bandwidth_multiplier": bandwidth,
                            "sigma_velocity": gaussian.sigma_velocity,
                        },
                    )
                )

            print(f"scanned segment_steps={segment_step}, match_steps={match_step}, archive={archive.size}")

    ranked = sorted(results, key=lambda item: float(item["score"]))
    payload = {
        "input": str(args.input),
        "n_train": len(train),
        "n_reference": len(reference),
        "planes": planes,
        "reference": reference_summary,
        "results": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote scan: {args.output}")
    print(f"Top {min(args.top, len(ranked))} settings")
    for idx, item in enumerate(ranked[: args.top], start=1):
        params = item["params"]
        print(
            f"{idx:>2}. {item['sampler']:<15} "
            f"score={item['score']:7.2f} mae={item['quantile_mae']:7.2f} "
            f"cov_def={item['coverage_deficit']:.3f} params={params}"
        )


def evaluate_sampler(
    name: str,
    sampler,
    reference_summary: dict[float, dict[str, float]],
    planes: list[float],
    n_generated: int,
    n_segments: int,
    missing_penalty: float,
    params: dict[str, object],
) -> dict[str, object]:
    generated = sampler.generate_many(n_trajectories=n_generated, n_segments=n_segments)
    generated_summary = summarize_breakthroughs(breakthrough_times(generated, planes))
    score = breakthrough_score(reference_summary, generated_summary, missing_penalty=missing_penalty)
    return {
        "sampler": name,
        **score,
        "params": params,
        "breakthrough": generated_summary,
    }


def split_trajectories(
    trajectories: list[np.ndarray],
    train_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are needed")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    split = max(1, min(len(trajectories) - 1, int(round(train_fraction * len(trajectories)))))
    return trajectories[:split], trajectories[split:]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
