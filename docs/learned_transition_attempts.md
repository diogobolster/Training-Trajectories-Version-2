# Learned Transition Attempts

## Current Baseline To Beat

The best current baseline is the tuned fixed Gaussian/Bayes transition kernel:

```text
segment_steps:        36
match_steps:          20
bandwidth_multiplier: 0.25
btc_score:            30.87
pair_mae:              1.56
```

This remains the anchor model.

## Attempt 1: Endpoint Contrastive Scorer

Training objective:

```text
score(true observed continuation) > score(random candidate continuation)
```

Features:

- endpoint/start velocity mismatch,
- speed mismatch,
- directional cosine mismatch.

Result:

```text
btc_score:     60.94
dilution_log:   0.047
pair_mae:       3.60
```

Interpretation:

The model improved dilution but damaged trajectory-pair structure.

## Attempt 2: Contextual Contrastive Scorer

Additional features:

- mean segment velocity,
- displacement direction,
- speed mean and standard deviation,
- start/end deviation from segment mean,
- curvature-like endpoint velocity change.

Result:

```text
btc_score:     51.73
dilution_log:   0.084
pair_mae:       4.47
```

Interpretation:

Richer local segment features improve BTC relative to the first contrastive attempt, but still do not preserve pair statistics.

## Attempt 3: Hybrid Contrastive-Gaussian

Transition score:

```text
hybrid_score = learned_weight * contrastive_score + gaussian_weight * gaussian_score
```

Best tested setting so far:

```text
learned_weight:  0.25
gaussian_weight: 1.0
```

Result:

```text
btc_score:     54.84
dilution_log:   0.119
pair_mae:       3.71
```

Interpretation:

The learned term perturbs the Gaussian prior but does not improve the important pair metric.

## Attempt 4: Adaptive Gaussian Bandwidth

Bandwidths are estimated from local archive density in speed-state bins.

Result:

```text
btc_score:     49.11
dilution_log:   0.088
pair_mae:       3.23
```

Interpretation:

The state-dependent bandwidth is plausible but not yet tuned enough to beat the fixed Gaussian kernel.

## Lesson

The 2019 physics-informed Gaussian/Bayes transition kernel is a strong inductive bias. A learned model that only learns local transition plausibility is not enough. The next learned transition model should optimize for multi-step transport consequences:

- late-time breakthrough tails,
- particle-pair separation,
- reaction encounter probability,
- persistent slow/fast states.

That suggests a reranking or policy-learning style model rather than a one-step binary continuation classifier.

