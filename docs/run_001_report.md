# Run 001: Bentheimer TTA-v2 Baselines

## Summary

This run uses the 6 micrometer Bentheimer sandstone CT volume, block-averaged to a 75^3 bootstrap simulation, then evaluates trajectory samplers against held-out particle trajectories.

- metrics file: `../outputs/bentheimer_6um_downsample3_full_metrics_rerank.json`
- train trajectories: `210`
- reference trajectories: `90`
- archive segments: `6300`
- segment steps: `36`
- match steps: `20`
- planes: `[6.0, 10.0, 14.0]`
- time indices: `[100, 200, 300, 400]`

## Main Result

- Best BTC score: `gaussian_bayes`
- Best pair-separation score: `gaussian_bayes`
- Best dilution score: `horizon_rerank`

The tuned Gaussian/Bayes kernel remains the strongest overall baseline. The learned and adaptive variants are informative, but they do not yet beat the physics-informed seam kernel on the most transport-specific metrics.

## Metric Table

| sampler | BTC score | dilution log MAE | pair MAE | reaction abs error |
|---|---:|---:|---:|---:|
| gaussian_bayes | 30.87 | 0.083 | 1.56 | 0.003 |
| knn_conditional | 40.88 | 0.083 | 2.95 | 0.003 |
| adaptive_gaussian | 49.11 | 0.088 | 3.23 | 0.002 |
| hybrid | 51.67 | 0.069 | 1.96 | 0.012 |
| contrastive | 57.39 | 0.121 | 3.87 | 0.002 |
| horizon_rerank | 86.95 | 0.048 | 4.08 | 0.004 |
| unconditional | 110.92 | 0.062 | 1.99 | 0.004 |

## Figures

![BTC median](../outputs/run_001_assets/btc_median.svg)

![Dilution index](../outputs/run_001_assets/dilution_index.svg)

![Pair separation](../outputs/run_001_assets/pair_separation_median.svg)

![Reaction probability](../outputs/run_001_assets/reaction_probability.svg)

![Metric errors](../outputs/run_001_assets/metric_errors.svg)

## Interpretation

The old TTA Gaussian/Bayes transition rule is not just a historical baseline; it is a strong inductive bias. The first learned transition models improve isolated metrics in places, but they are too local and can damage pair dynamics. The next useful model should optimize short-horizon or multi-step transport consequences rather than one-step continuation plausibility.
