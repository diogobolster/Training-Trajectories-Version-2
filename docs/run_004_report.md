# Run 004: Repeated Validation Mixture Selection

## Summary

Run 004 repeats the Run 003 mixture-selection idea over five fit/validation resamples while holding one test set fixed. The selected weights from the five validation splits are averaged, the components are refit on the full non-test pool, and the averaged mixture is evaluated once on the held-out test set.

## Averaged Weights

```text
gaussian_bayes:    0.35
knn_conditional:   0.25
hybrid:            0.35
pair_rerank:       0.05
```

## Test Result

```text
sampler                    objective  btc_score  pair_mae  dilution_log  reaction_abs
bootstrap_mean_mixture         84.91      36.64      1.39         0.065         0.013
hybrid                         94.40      43.69      1.42         0.070         0.014
gaussian_bayes                 98.86      41.77      1.68         0.096         0.012
pooled_validation_mixture     115.54      47.68      2.21         0.074         0.015
knn_conditional               122.84      63.36      1.92         0.038         0.017
pair_rerank                   123.21      67.78      1.70         0.068         0.013
```

## Takeaway

The repeated-validation average is now the best held-out sampler. This strengthens the main paper point: learned context should be coupled to the physics kernel through a validation-driven, multi-objective mixture rather than treated as a universal replacement.

Detailed notes:

[bootstrap_mixture_selection.md](bootstrap_mixture_selection.md)

