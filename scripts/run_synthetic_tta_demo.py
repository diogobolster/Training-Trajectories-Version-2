from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import (  # noqa: E402
    ConditionalSegmentSampler,
    SegmentArchive,
    breakthrough_times,
    generate_channel_trajectories,
    summarize_breakthroughs,
    velocity_autocorrelation,
)


def main() -> None:
    train = generate_channel_trajectories(n_trajectories=350, n_steps=900, seed=7)
    reference = generate_channel_trajectories(n_trajectories=180, n_steps=900, seed=99)

    archive = SegmentArchive.from_trajectories(
        train,
        segment_steps=36,
        match_steps=6,
    )
    sampler = ConditionalSegmentSampler(archive=archive, k=96, temperature=0.8, seed=123)
    generated = sampler.generate_many(n_trajectories=180, n_segments=26)

    planes = [4.0, 8.0, 12.0]
    reference_summary = summarize_breakthroughs(breakthrough_times(reference, planes))
    generated_summary = summarize_breakthroughs(breakthrough_times(generated, planes))

    reference_corr = velocity_autocorrelation(reference, max_lag=40)
    generated_corr = velocity_autocorrelation(generated, max_lag=40)

    payload = {
        "archive_size": archive.size,
        "segment_steps": archive.segment_steps,
        "match_steps": archive.match_steps,
        "breakthrough_reference": reference_summary,
        "breakthrough_generated": generated_summary,
        "velocity_autocorrelation_lags_0_10": {
            "reference": [float(x) for x in reference_corr[:11]],
            "generated": [float(x) for x in generated_corr[:11]],
        },
    }

    output_path = ROOT / "outputs" / "synthetic_tta_demo_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Archive segments: {archive.size}")
    print(f"Wrote summary: {output_path}")
    print("\nBreakthrough q10/q50/q90 by control plane")
    for plane in planes:
        ref = reference_summary[plane]
        gen = generated_summary[plane]
        print(
            f"x={plane:>4.1f} | "
            f"ref {ref['q10']:6.1f} {ref['q50']:6.1f} {ref['q90']:6.1f} | "
            f"gen {gen['q10']:6.1f} {gen['q50']:6.1f} {gen['q90']:6.1f}"
        )


if __name__ == "__main__":
    main()

