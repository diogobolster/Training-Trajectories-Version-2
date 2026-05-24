# Short-Horizon Reranking

## Goal

Test whether a transition model can improve on the Gaussian/Bayes seam kernel by looking ahead a few archive segments.

## Model

`ShortHorizonRerankGaussianSampler` keeps the tuned Gaussian/Bayes seam score but adds a short-horizon descriptor score:

```text
transition score = gaussian seam score + horizon_weight * future descriptor score
```

The future descriptor is estimated by following archive successors for a few segments and summarizing:

- future displacement velocity,
- total speed,
- longitudinal speed,
- transverse speed,
- speed variability.

Targets are learned in current-speed bins from true archive continuations.

## Full-Metric Result

Command:

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
  --rerank-horizon-segments 3 \
  --rerank-horizon-weight 0.25 \
  --output outputs/bentheimer_6um_downsample3_full_metrics_rerank.json
```

Result:

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

## Weight Sweep

Quick lower-cost runs with 70 generated trajectories and 1500 pair samples:

```text
horizon_weight  btc_score  dilution_log  pair_mae
0.02              33.41        0.197        3.92
0.05              61.81        0.181        4.16
0.10              69.52        0.152        4.10
0.25              86.95        0.048        4.08
```

## Interpretation

The short-horizon descriptor can improve dilution at larger weights, but it hurts BTC and pair separation. At very small weight it behaves closer to the Gaussian baseline for BTC, but still does not repair pair dynamics.

This tells us that the current horizon descriptor is not the right multi-step objective. It describes single-particle future displacement, while the hard metric is particle-pair separation. The next reranker should be trained directly on pair-aware local ensemble behavior, not only individual future movement.

