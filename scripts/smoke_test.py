from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import ConditionalSegmentSampler, SegmentArchive  # noqa: E402


def main() -> None:
    t = np.linspace(0.0, 1.0, 101)
    trajectories = [
        np.column_stack([t, 0.1 * np.sin(4.0 * np.pi * t)]),
        np.column_stack([t, 0.1 * np.cos(4.0 * np.pi * t)]),
    ]

    archive = SegmentArchive.from_trajectories(trajectories, segment_steps=12, match_steps=3)
    sampler = ConditionalSegmentSampler(archive=archive, k=8, seed=4)
    generated = sampler.generate(n_segments=5)

    assert archive.size > 0
    assert generated.shape[1] == 2
    assert np.all(np.isfinite(generated))
    assert np.max(np.linalg.norm(np.diff(generated, axis=0), axis=1)) < 0.25

    print(f"smoke test passed: archive_size={archive.size}, generated_shape={generated.shape}")


if __name__ == "__main__":
    main()

