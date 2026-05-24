# Pair-Aware Reranking

## Goal

Run 002 tests whether a transition rule can improve the particle-pair separation metric by adding an archive-level pair behavior proxy to the Gaussian/Bayes seam score.

## Pair Behavior Proxy

Each archive segment is assigned a local pair-divergence descriptor. For a segment, we find nearby segments in start-velocity state space, follow each segment's short-horizon archive future, and summarize how much the futures diverge.

Descriptor components:

- median future divergence,
- q90 future divergence,
- median growth relative to current state separation,
- convergence fraction,
- future-divergence standard deviation.

Archive diagnostic:

```text
median_future_divergence: 0.0146
q90_future_divergence:    0.0242
median_growth:            0.0094
convergence_fraction:     0.0312
future_divergence_std:    0.0071
```

The low convergence fraction is important: most local state-space neighbors diverge over the short horizon in this bootstrap trajectory set, so pair-preserving transitions are relatively rare.

## Sampler

`PairAwareRerankGaussianSampler` uses:

```text
score = Gaussian seam score + pair_weight * pair_behavior_score
```

where the pair behavior target is learned from true archive continuations in current-speed bins.

## Full Run 002 Result

```text
sampler             btc_score  dilution_log  pair_mae  reaction_abs
unconditional         110.92         0.062      1.99         0.004
knn_conditional        40.88         0.083      2.95         0.003
gaussian_bayes         30.87         0.083      1.56         0.003
adaptive_gaussian      49.11         0.088      3.23         0.002
horizon_rerank         55.23         0.082      4.37         0.003
pair_rerank            88.05         0.049      3.77         0.001
contrastive            48.56         0.078      2.18         0.006
hybrid                 53.73         0.067      0.92         0.003
```

## Interpretation

The explicit pair-reranker did not beat the tuned Gaussian/Bayes baseline. It improved dilution and reaction error, but it damaged BTC and pair-separation performance.

The surprise is the hybrid learned/physics sampler: it achieved the best pair MAE in this run (`0.92`), but with a worse BTC score (`53.73`) than Gaussian/Bayes (`30.87`). This is the first signal that learned context can help pair structure, but the improvement trades away arrival-time fidelity.

## Conclusion

The pair-aware idea is right, but the current proxy is still too indirect. It uses pair behavior inferred from local velocity-state neighbors, not actual generated ensemble pair dynamics. The next useful model should treat pair-separation error as an explicit validation/tuning objective, likely by selecting sampler hyperparameters or transition mixtures against a held-out pair metric.

