# Run 006: Objective-Weight Sensitivity

## Summary

Run 006 reruns validation-driven mixture selection under seven multi-objective weighting regimes. This checks whether the mixture result depends on the particular balanced score used in Run 005.

## Main Result

```text
regime           best_mean_sampler            mean_obj  mean_rank  wins  beats_g  beats_h
balanced         gaussian_bayes                 145.26       2.25     1        0        2
btc_heavy        gaussian_bayes                 188.73       2.25     0        0        1
pair_heavy       knn_conditional                204.41       2.75     0        2        2
dilution_heavy   hybrid                         170.07       2.50     2        3        0
reaction_light   gaussian_bayes                 140.53       2.25     1        0        2
reaction_heavy   gaussian_bayes                  85.75       1.75     1        0        3
no_reaction      gaussian_bayes                 140.01       2.00     1        0        2
```

## Takeaway

Gaussian/Bayes is the most stable mean performer in this faster four-split sensitivity sweep, while hybrid and validation-selected mixtures win under specific metric priorities and held-out splits. The result is not universal mixture dominance; it is evidence that the framework exposes metric tradeoffs and lets the objective choose the appropriate physics/ML blend.

Detailed notes:

[objective_weight_sensitivity.md](objective_weight_sensitivity.md)

