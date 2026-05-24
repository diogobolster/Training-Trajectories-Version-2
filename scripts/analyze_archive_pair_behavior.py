from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import SegmentArchive, load_trajectories  # noqa: E402
from tta_v2.sampler import archive_pair_behavior_descriptors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze archive-level pair behavior descriptors.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--segment-steps", type=int, default=36)
    parser.add_argument("--match-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--horizon-segments", type=int, default=3)
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--speed-bins", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "archive_pair_behavior.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.input, key=args.key)
    split = max(1, min(len(trajectories) - 1, int(round(args.train_fraction * len(trajectories)))))
    train = trajectories[:split]
    archive = SegmentArchive.from_trajectories(
        train,
        segment_steps=args.segment_steps,
        match_steps=args.match_steps,
        dt=args.dt,
    )
    descriptors = archive_pair_behavior_descriptors(
        archive,
        horizon_segments=args.horizon_segments,
        neighbor_k=args.neighbor_k,
    )
    speed = np.linalg.norm(archive.start_velocities, axis=1)
    edges = np.quantile(speed, np.linspace(0.0, 1.0, args.speed_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    names = [
        "median_future_divergence",
        "q90_future_divergence",
        "median_growth",
        "convergence_fraction",
        "future_divergence_std",
    ]
    by_bin = []
    for idx in range(args.speed_bins):
        in_bin = (speed >= edges[idx]) & (speed <= edges[idx + 1])
        values = descriptors[in_bin]
        by_bin.append(
            {
                "bin": idx,
                "count": int(np.sum(in_bin)),
                "speed_min": float(np.min(speed[in_bin])) if np.any(in_bin) else None,
                "speed_max": float(np.max(speed[in_bin])) if np.any(in_bin) else None,
                "descriptor_median": {
                    name: float(value) for name, value in zip(names, np.median(values, axis=0))
                }
                if len(values)
                else {},
                "descriptor_q10": {
                    name: float(value) for name, value in zip(names, np.quantile(values, 0.1, axis=0))
                }
                if len(values)
                else {},
                "descriptor_q90": {
                    name: float(value) for name, value in zip(names, np.quantile(values, 0.9, axis=0))
                }
                if len(values)
                else {},
            }
        )

    payload = {
        "input": str(args.input),
        "archive_size": archive.size,
        "segment_steps": archive.segment_steps,
        "match_steps": archive.match_steps,
        "horizon_segments": args.horizon_segments,
        "neighbor_k": args.neighbor_k,
        "descriptor_names": names,
        "global_median": {name: float(value) for name, value in zip(names, np.median(descriptors, axis=0))},
        "by_speed_bin": by_bin,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Archive segments: {archive.size}")
    print(f"Wrote pair behavior: {args.output}")
    print("Global descriptor medians")
    for name, value in payload["global_median"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

