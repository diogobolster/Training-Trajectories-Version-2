# Objective-Weight Sensitivity

## Motivation

Run 005 used one multi-objective score:

```text
1 * BTC + 20 * pair + 120 * dilution + 1000 * reaction
```

That score is transparent, but a reviewer can still ask whether the validation-mixture result is an artifact of those weights. Run 006 tests that directly by rerunning validation-driven selection under several objective-weight regimes.

This is not a post-hoc rescore only. Each regime reselects mixture weights from the inner validation results, then evaluates the regime-selected mixtures on held-out outer test splits.

## Script

```text
scripts/objective_weight_sensitivity.py
```

## Regimes

```text
regime           BTC   pair  dilution  reaction
balanced         1.0   20.0    120.0    1000.0
breakthrough_only 1.0    0.0      0.0       0.0
btc_heavy        3.0   10.0     60.0     500.0
pair_heavy       0.5   60.0     80.0     500.0
dilution_heavy   0.5   10.0    360.0     500.0
reaction_light   1.0   20.0    120.0     100.0
reaction_heavy   0.5   10.0     60.0    3000.0
no_reaction      1.0   20.0    120.0       0.0
```

## Command

```bash
python3 scripts/objective_weight_sensitivity.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --n-outer-splits 4 \
  --n-repeats 3 \
  --grid-step 0.25 \
  --n-validation-generated 35 \
  --n-test-generated 60 \
  --pair-samples 1000 \
  --contrastive-epochs 300 \
  --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 \
  --pair-rerank-weight 0.25 \
  --output outputs/bentheimer_6um_downsample3_objective_weight_sensitivity.json
```

This is a faster sensitivity sweep than Run 005, so objective values should not be compared one-to-one with the Run 005 table.

## Best Mean Sampler by Regime

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

## Mixture Performance Highlights

The pooled validation mixture remains competitive, but it is not the best mean sampler in this faster sweep.

```text
regime           pooled mean_obj  pooled wins  pooled beats_g  pooled beats_h
balanced                 152.45            1               1               2
pair_heavy               208.90            2               2               2
dilution_heavy           175.49            0               2               2
reaction_heavy            87.33            2               2               3
```

The strongest mixture case is `reaction_heavy`: Gaussian/Bayes has the best mean objective, but the pooled validation mixture wins 2 of 4 outer splits and beats both Gaussian/Bayes and hybrid on multiple splits.

## Selection Weights

The selected pooled mixtures shift with the objective:

```text
regime           gaussian  knn     hybrid  pair
balanced           0.438  0.125    0.438  0.000
pair_heavy         0.250  0.125    0.625  0.000
dilution_heavy     0.375  0.063    0.563  0.000
reaction_heavy     0.250  0.125    0.625  0.000
```

This is the valuable paper point: changing the scientific priority changes the selected transport mechanism. Pair-, dilution-, and reaction-sensitive objectives generally move selection toward the learned hybrid component, while Gaussian/Bayes remains the most stable mean performer in this small sensitivity sweep.

## Interpretation

Run 006 prevents overclaiming. It says:

```text
there is no universal winner under all reasonable objective weights.
```

That is not bad news. It is a more defensible framing for TTA-v2:

```text
the method exposes and validates metric tradeoffs across physics kernels,
learned transition rules, and their mixtures.
```

For the manuscript, this should be a Pareto or sensitivity figure rather than a single ranking table.

## OpenFOAM Extension

Run 011 repeats this sensitivity analysis on the Core2 OpenFOAM-derived trajectory set. The result is similar in spirit but sharper physically: Gaussian/Bayes remains the best mean sampler in six of seven objective regimes, while the pair-heavy regime selects the hybrid sampler. Bootstrap-mean mixtures are more competitive than pooled-validation mixtures in that run, suggesting that repeated-weight averaging can regularize selection when validation ensembles are small.

See `run_011_openfoam_objective_sensitivity.md`.
