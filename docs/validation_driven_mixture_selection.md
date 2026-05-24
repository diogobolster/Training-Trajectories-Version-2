# Validation-Driven Mixture Selection

## Motivation

Run 002 showed an important tradeoff:

```text
learned/contextual samplers can improve pair structure,
but they can trade away BTC fidelity.
```

That is a paper-worthy point. It means the right ML framing is not "replace the Gaussian/Bayes kernel." It is:

```text
choose or weight physics and learned samplers against held-out transport metrics.
```

## Implementation

`MixtureSegmentSampler` combines transition distributions from component samplers:

```text
p(next | state) = sum_i w_i p_i(next | state)
```

The selector script:

```text
scripts/select_sampler_mixture.py
```

uses three splits:

- fit: build archive and train component samplers,
- validation: select mixture weights,
- test: evaluate selected mixture against individual components.

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

The weights are intentionally explicit so we can make metric priorities visible rather than hiding them in a learned black box.

## Command

```bash
python3 scripts/select_sampler_mixture.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --grid-step 0.25 \
  --n-validation-generated 70 \
  --n-test-generated 90 \
  --pair-samples 2000 \
  --contrastive-epochs 700 \
  --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 \
  --pair-rerank-weight 0.25 \
  --btc-weight 1 \
  --pair-weight 20 \
  --dilution-weight 120 \
  --reaction-weight 1000 \
  --output outputs/bentheimer_6um_downsample3_mixture_selection.json
```

## Selected Mixture

Validation selected:

```text
gaussian_bayes:    0.00
knn_conditional:   0.50
hybrid:            0.50
pair_rerank:       0.00
```

Top validation objective:

```text
objective:      122.63
BTC score:       42.60
pair MAE:         3.01
dilution log:     0.091
reaction abs:     0.009
```

## Held-Out Test Result

```text
sampler             objective  btc_score  pair_mae  dilution_log  reaction_abs
hybrid                109.78      29.94      3.29         0.071         0.005
selected_mixture      122.87      47.67      3.24         0.049         0.005
gaussian_bayes        129.38      42.63      3.54         0.088         0.005
knn_conditional       139.11      60.45      3.40         0.039         0.006
pair_rerank           145.00      57.63      3.50         0.098         0.005
```

## Interpretation

The selected mixture does beat the fixed Gaussian/Bayes sampler on the held-out multi-objective score:

```text
selected mixture: 122.87
Gaussian/Bayes:   129.38
```

However, the pure hybrid sampler is best on this particular test split:

```text
hybrid: 109.78
```

This tells us two things:

1. Validation-driven mixture selection is doing useful work: it found a nontrivial physics/learned mixture and improved over the fixed Gaussian/Bayes baseline.
2. The validation split is still noisy at this sample size; the selected mixture did not generalize as well as the best individual component.

## Paper Point

This is not a failure. It is a strong methodological point:

```text
Modern ML introduces useful transport tradeoffs, but those tradeoffs must be selected against held-out multi-objective physics metrics.
```

That is a better contribution than claiming a learned sampler universally dominates the physics kernel.

## Follow-Up

Run 004 uses repeated validation splits:

```text
select mixture weights over multiple validation splits,
average the selected weights,
then evaluate once on the held-out test split.
```

The repeated-validation mixture beats all individual components on the held-out objective:

```text
bootstrap mean mixture:  84.91
hybrid:                  94.40
Gaussian/Bayes:          98.86
```

See [bootstrap_mixture_selection.md](bootstrap_mixture_selection.md).
