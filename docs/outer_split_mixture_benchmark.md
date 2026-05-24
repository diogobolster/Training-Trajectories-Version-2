# Outer-Split Mixture Benchmark

## Motivation

Run 004 showed that repeated validation can produce a strong held-out mixture on one fixed test split. Run 005 asks the next reviewer-proof question:

```text
Does the validation-driven mixture result survive different held-out test splits?
```

The benchmark repeats the full selection workflow over multiple outer splits:

- choose a held-out test set,
- run repeated inner fit/validation selection on the remaining trajectories,
- evaluate selected mixtures and individual samplers on the outer test set,
- aggregate objective, rank, and win counts across outer splits.

## Script

```text
scripts/outer_split_mixture_benchmark.py
```

Component order:

```text
gaussian_bayes
knn_conditional
hybrid
pair_rerank
```

Evaluated samplers:

```text
bootstrap_mean_mixture
pooled_validation_mixture
gaussian_bayes
knn_conditional
hybrid
pair_rerank
```

The two mixture-selection rules are different:

- `bootstrap_mean_mixture`: average the best weights selected by each inner validation repeat.
- `pooled_validation_mixture`: choose the grid weight vector with the best mean validation score across inner repeats.

## Command

```bash
python3 scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --n-outer-splits 5 \
  --n-repeats 4 \
  --grid-step 0.25 \
  --n-validation-generated 45 \
  --n-test-generated 80 \
  --pair-samples 1200 \
  --contrastive-epochs 400 \
  --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 \
  --pair-rerank-weight 0.25 \
  --btc-weight 1 \
  --pair-weight 20 \
  --dilution-weight 120 \
  --reaction-weight 1000 \
  --output outputs/bentheimer_6um_downsample3_outer_split_mixture_benchmark.json
```

## Summary

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
pooled_validation_mixture    121.64     47.75       2.00     2        3        3
gaussian_bayes               125.48     50.22       2.00     2        0        4
hybrid                       127.71     57.25       3.20     1        1        0
bootstrap_mean_mixture       132.05     50.44       3.20     0        1        3
knn_conditional              142.47     38.49       4.80     0        0        1
pair_rerank                  167.72     60.16       5.80     0        0        0
```

Mean outer selected weights for the averaged-weight rule:

```text
gaussian_bayes:    0.4125
knn_conditional:   0.1500
hybrid:            0.4125
pair_rerank:       0.0250
```

## Per-Split Winners

```text
outer split  winner                     objective
1            gaussian_bayes                128.98
2            hybrid                         52.82
3            pooled_validation_mixture     194.69
4            pooled_validation_mixture     115.96
5            gaussian_bayes                 92.20
```

## Interpretation

The result is intentionally not over-clean:

- the pooled validation mixture has the best mean objective,
- it beats both Gaussian/Bayes and hybrid on 3 of 5 outer splits,
- Gaussian/Bayes remains extremely competitive and ties the pooled mixture on mean rank,
- the averaged-weight rule from Run 004 is not the most robust rule across outer splits.

This is a stronger paper result than a single victory table. It says validation-driven selection is valuable, but the exact aggregation rule matters. The defensible next TTA-v2 claim is:

```text
held-out multi-objective selection can identify physics/ML mixtures that are
competitive with, and often better than, the strongest hand-designed kernel,
while revealing the remaining split-to-split uncertainty.
```

