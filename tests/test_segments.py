from __future__ import annotations

import numpy as np

from tta_v2 import ConditionalSegmentSampler, SegmentArchive


def test_archive_and_sampler_generate_continuous_path() -> None:
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

