# Full-Metric Baseline

## Purpose

This run evaluates the first real-geometry TTA baselines on more than breakthrough curves:

- breakthrough curves,
- dilution / occupancy entropy,
- particle-pair separation,
- reaction encounter probability.

This is closer to the spirit of the 2019 paper, where the interesting claim was not only arrival-time matching but preservation of 3D mixing and reaction-relevant structure.

## Dataset

```text
data/processed/bentheimer_6um_downsample3_trajectories.npz
```

Generated from the 6 micrometer Bentheimer CT volume downsampled to a 75^3 bootstrap simulation.

## Command

```bash
python3 scripts/evaluate_transport_metrics.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --planes 6,10,14 \
  --time-indices 100,200,300,400 \
  --segment-steps 36 \
  --match-steps 20 \
  --n-generated 90 \
  --n-segments 32 \
  --diffusivity 0.001 \
  --gaussian-bandwidth 0.25 \
  --bin-size 3 \
  --pair-samples 3000 \
  --reaction-radius 3 \
  --output outputs/bentheimer_6um_downsample3_full_metrics_with_contrastive.json
```

## Result

Original learned-scorer run:

```text
sampler             btc_score  dilution_log  pair_mae  reaction_abs
unconditional         110.92         0.062      1.99         0.004
knn_conditional        40.88         0.083      2.95         0.003
gaussian_bayes         30.87         0.083      1.56         0.003
contrastive            60.94         0.047      3.60         0.003
```

Contextual/adaptive run:

```text
sampler             btc_score  dilution_log  pair_mae  reaction_abs
unconditional         110.92         0.062      1.99         0.004
knn_conditional        40.88         0.083      2.95         0.003
gaussian_bayes         30.87         0.083      1.56         0.003
adaptive_gaussian      49.11         0.088      3.23         0.002
contrastive            51.73         0.084      4.47         0.003
hybrid                 54.84         0.119      3.71         0.000
```

Short-horizon reranking run:

```text
sampler             btc_score  dilution_log  pair_mae  reaction_abs
unconditional         110.92         0.062      1.99         0.004
knn_conditional        40.88         0.083      2.95         0.003
gaussian_bayes         30.87         0.083      1.56         0.003
adaptive_gaussian      49.11         0.088      3.23         0.002
horizon_rerank         86.95         0.048      4.08         0.004
contrastive            57.39         0.121      3.87         0.002
hybrid                 51.67         0.069      1.96         0.012
```

## Interpretation

The tuned Gaussian/Bayes sampler is still the strongest overall baseline. It wins the BTC score and particle-pair separation, which are the two most transport-specific metrics in this run.

The first learned contrastive sampler is not a win yet, but it is not useless: it gives the best dilution error. That means the learned transition score is capturing some occupancy/plume-volume structure, but it is not yet preserving pathwise pair dynamics.

The contextual contrastive, hybrid contrastive-Gaussian, adaptive Gaussian, and short-horizon reranking variants do not yet beat the fixed tuned Gaussian baseline. This is a productive failure mode. It points to the next model: the learned scorer should train not only on true segment continuations or single-particle future descriptors, but also on pair-aware multi-step consequences.

## Reference Values

Reference reaction encounter probability:

```text
radius:      3 voxels
probability: 0.0233
```

Reference dilution index:

```text
t=100: 1837.85
t=200: 1917.51
t=300: 2103.17
t=400: 2071.03
```

## Next Step

Improve the learned transition model in one of two ways:

1. Learn a state-dependent Gaussian/Bayes bandwidth from true continuation residuals.
2. Train a contrastive scorer with richer history features, not just endpoint/start velocities.

We tried first versions of both, plus a first short-horizon reranker. None beats the tuned Gaussian baseline yet. The better next scientific path is to make the learned objective pair-aware: train or tune transition scores against downstream pair separation and late-time BTC behavior.
