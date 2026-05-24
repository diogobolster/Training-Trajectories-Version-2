# Run 005: Outer-Split Robustness Benchmark

## Summary

Run 005 repeats the full repeated-validation mixture-selection workflow over five independent held-out test splits. This tests whether the Run 004 result survives test-set resampling.

## Main Result

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
pooled_validation_mixture    121.64     47.75       2.00     2        3        3
gaussian_bayes               125.48     50.22       2.00     2        0        4
hybrid                       127.71     57.25       3.20     1        1        0
bootstrap_mean_mixture       132.05     50.44       3.20     0        1        3
knn_conditional              142.47     38.49       4.80     0        0        1
pair_rerank                  167.72     60.16       5.80     0        0        0
```

## Takeaway

The pooled validation mixture has the best mean objective and beats both Gaussian/Bayes and hybrid on 3 of 5 outer splits. Gaussian/Bayes remains very strong and ties the pooled mixture on mean rank, so the honest result is not universal dominance. The robust claim is that validation-driven physics/ML mixtures are competitive with, and often better than, the strongest hand-designed kernel.

Detailed notes:

[outer_split_mixture_benchmark.md](outer_split_mixture_benchmark.md)

