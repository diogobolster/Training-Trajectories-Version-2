# Run 003: Validation-Driven Mixture Selection

## Summary

Run 003 moves from hand-selecting samplers to validation-driven weighting. The selected mixture is chosen on a held-out validation split using a multi-objective score over BTC, dilution, pair separation, and reaction probability.

## Selected Weights

```text
gaussian_bayes:    0.00
knn_conditional:   0.50
hybrid:            0.50
pair_rerank:       0.00
```

## Test Result

```text
sampler             objective  btc_score  pair_mae  dilution_log  reaction_abs
hybrid                109.78      29.94      3.29         0.071         0.005
selected_mixture      122.87      47.67      3.24         0.049         0.005
gaussian_bayes        129.38      42.63      3.54         0.088         0.005
knn_conditional       139.11      60.45      3.40         0.039         0.006
pair_rerank           145.00      57.63      3.50         0.098         0.005
```

## Takeaway

The selected mixture improves over the fixed Gaussian/Bayes baseline on the held-out multi-objective score, but the pure hybrid sampler is best on this test split. The result strengthens the paper argument: learned context can help, but it creates metric tradeoffs, so model selection must be validation-driven and multi-objective.

Detailed notes:

[validation_driven_mixture_selection.md](validation_driven_mixture_selection.md)

