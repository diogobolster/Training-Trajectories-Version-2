from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import (  # noqa: E402
    adjacent_segment_velocity_correlation,
    load_trajectories,
    suggest_segment_scales,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan adjacent segment velocity correlation K(lambda)|1,2.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--steps", default="2,3,4,5,6,8,10,12,16,20,24,30,36,42,48")
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "velocity_autocorrelation_scan.json")
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [int(item) for item in args.steps.split(",") if item.strip()]
    trajectories = load_trajectories(args.input, key=args.key)
    correlations = adjacent_segment_velocity_correlation(
        trajectories,
        steps,
        component=args.component,
        dt=args.dt,
    )
    suggestion = suggest_segment_scales(correlations)

    payload = {
        "input": str(args.input),
        "component": args.component,
        "dt": args.dt,
        "correlations": {str(step): value for step, value in correlations.items()},
        "suggestion": suggestion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["segment_steps", "correlation"])
            for step, corr in sorted(correlations.items()):
                writer.writerow([step, corr])

    print(f"Wrote scan: {args.output}")
    print("Suggested discrete scales")
    print(f"match_steps:   {suggestion['match_steps']:.0f}")
    print(f"segment_steps: {suggestion['segment_steps']:.0f}")
    print()
    print("K(lambda)|1,2")
    for step, corr in sorted(correlations.items()):
        print(f"{step:>4}: {corr: .4f}")


if __name__ == "__main__":
    main()

