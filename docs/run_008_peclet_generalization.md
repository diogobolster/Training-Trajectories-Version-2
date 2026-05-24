# Run 008: First Peclet Generalization Test

## Summary

Run 008 tests whether the validation-driven mixture story survives a changed transport condition. We regenerated the Bentheimer 6 micrometer downsampled trajectory set with lower diffusivity:

```text
baseline diffusivity:       0.001
high-Peclet diffusivity:    0.0003
```

The geometry, segmentation, connected pore network, approximate pressure solve, and particle-tracking pipeline were otherwise kept the same. This is not as strong as a second rock geometry, but it is a meaningful first generalization test because finite-Peclet behavior was one of the original motivations for training trajectories.

## New Trajectory Set

```text
file:                  data/processed/bentheimer_6um_downsample3_D0003_trajectories.npz
raw volume:            Core1_Subvol1_6micron_225cube_16bit_LE.raw
simulation shape:      75 x 75 x 75
connected porosity:    0.22543
particles:             500
steps:                 800
dt:                    0.5
diffusivity:           0.0003
```

## Benchmark Command

```bash
python3 scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_6um_downsample3_D0003_trajectories.npz \
  --n-outer-splits 4 \
  --n-repeats 3 \
  --grid-step 0.25 \
  --n-validation-generated 45 \
  --n-test-generated 80 \
  --pair-samples 1200 \
  --contrastive-epochs 300 \
  --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 \
  --pair-rerank-weight 0.25 \
  --btc-weight 1 \
  --pair-weight 20 \
  --dilution-weight 120 \
  --reaction-weight 1000 \
  --output outputs/bentheimer_6um_downsample3_D0003_outer_split_mixture_benchmark.json
```

## Result

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
pooled_validation_mixture    276.15     15.81       2.00     2        3        3
knn_conditional              278.85     32.20       2.50     1        3        2
gaussian_bayes               294.79     21.38       3.25     1        0        2
bootstrap_mean_mixture       295.54     25.40       4.00     0        1        2
hybrid                       301.31     24.16       3.25     0        2        0
pair_rerank                  368.33     30.91       6.00     0        0        0
```

Mean selected weights:

```text
gaussian_bayes:    0.4375
knn_conditional:   0.1875
hybrid:            0.3333
pair_rerank:       0.0417
```

## Comparison to Baseline Diffusivity

The pooled validation mixture remains the best mean-objective sampler:

```text
D = 0.001:   pooled mixture mean rank 2.00, wins 2/5
D = 0.0003:  pooled mixture mean rank 2.00, wins 2/4
```

The most interesting change is the kNN conditional sampler:

```text
D = 0.001:   kNN mean rank 4.80, wins 0/5
D = 0.0003:  kNN mean rank 2.50, wins 1/4
```

This is physically interpretable. At lower diffusivity, local velocity continuity and nearest-neighbor velocity matching become more valuable. The selected mixtures also shift slightly away from the learned hybrid component and toward Gaussian/Bayes/kNN matching.

## Interpretation

Run 008 materially improves the manuscript story. The validation-driven mixture framework remains competitive under a changed transport condition, and the component ranking changes in a physically meaningful way.

The result still does not eliminate the need for second-geometry or high-fidelity DNS validation. But it turns the current paper from a single-condition prototype into a first regime-sensitivity study:

```text
selection is not fixed; it responds to the Peclet regime.
```

