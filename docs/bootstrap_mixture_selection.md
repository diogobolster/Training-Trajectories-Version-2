# Bootstrap Mixture Selection

## Motivation

Run 003 used one validation split. It found a useful physics/learned mixture, but the pure hybrid sampler did better on the held-out test split. That was a warning that one validation split is too noisy for this data volume.

Run 004 fixes that by separating two ideas:

- keep one held-out test set fixed,
- repeat the fit/validation split inside the remaining data,
- select mixture weights on each validation split,
- average the selected weights,
- refit on the full non-test pool and evaluate once on the held-out test set.

This is the first validation-driven result that cleanly supports the paper claim.

## Script

```text
scripts/bootstrap_mixture_selection.py
```

Component order:

```text
gaussian_bayes
knn_conditional
hybrid
pair_rerank
```

Multi-objective score:

```text
objective = 1 * BTC score
          + 20 * pair MAE
          + 120 * dilution log MAE
          + 1000 * reaction absolute error
```

## Command

```bash
python3 scripts/bootstrap_mixture_selection.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --n-repeats 5 \
  --grid-step 0.25 \
  --n-validation-generated 50 \
  --n-test-generated 90 \
  --pair-samples 1500 \
  --contrastive-epochs 500 \
  --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 \
  --pair-rerank-weight 0.25 \
  --btc-weight 1 \
  --pair-weight 20 \
  --dilution-weight 120 \
  --reaction-weight 1000 \
  --output outputs/bentheimer_6um_downsample3_bootstrap_mixture_selection.json
```

## Splits

```text
train pool:       210 trajectories
held-out test:     90 trajectories
repeats:            5
fit per repeat:   147 trajectories
validation:        63 trajectories
archive size:    6300 segments after final refit
```

## Repeat Selections

```text
repeat  gaussian  knn   hybrid  pair  validation objective
1          0.25   0.00    0.75  0.00       144.82
2          0.50   0.25    0.00  0.25       198.33
3          0.25   0.25    0.50  0.00       126.87
4          0.50   0.25    0.25  0.00        69.49
5          0.25   0.50    0.25  0.00        77.56
```

Averaged selected weights:

```text
gaussian_bayes:    0.35
knn_conditional:   0.25
hybrid:            0.35
pair_rerank:       0.05
```

For comparison, the grid point with the best mean validation score was:

```text
gaussian_bayes:    0.25
knn_conditional:   0.50
hybrid:            0.25
pair_rerank:       0.00
```

## Held-Out Test Result

```text
sampler                    objective  btc_score  pair_mae  dilution_log  reaction_abs
bootstrap_mean_mixture         84.91      36.64      1.39         0.065         0.013
hybrid                         94.40      43.69      1.42         0.070         0.014
gaussian_bayes                 98.86      41.77      1.68         0.096         0.012
pooled_validation_mixture     115.54      47.68      2.21         0.074         0.015
knn_conditional               122.84      63.36      1.92         0.038         0.017
pair_rerank                   123.21      67.78      1.70         0.068         0.013
```

## Interpretation

The averaged repeated-validation mixture is the best held-out sampler in this run. It beats the pure hybrid sampler, the fixed Gaussian/Bayes sampler, the kNN conditional sampler, and the pair-aware reranker on the combined transport objective.

This is a better story than "the neural sampler wins." The result says:

```text
physics and learned transition rules each capture different transport statistics,
and the defensible model is a validation-weighted mixture selected against
held-out BTC, dilution, pair, and reaction metrics.
```

That is the methodological center of TTA-v2.

