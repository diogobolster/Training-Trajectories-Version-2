# Run 009: Three-Regime Peclet Sweep

## Summary

Run 009 completes the first Peclet-regime sweep by adding the high-diffusion / low-Peclet case:

```text
high Pe:    D = 0.0003
baseline:   D = 0.001
low Pe:     D = 0.003
```

The low-Peclet trajectory set uses the same Bentheimer geometry, segmentation, connected pore network, pressure solve, and particle-tracking pipeline as the previous two runs. Only diffusivity changes.

## New Low-Peclet Trajectory Set

```text
file:                  data/processed/bentheimer_6um_downsample3_D003_trajectories.npz
raw volume:            Core1_Subvol1_6micron_225cube_16bit_LE.raw
simulation shape:      75 x 75 x 75
connected porosity:    0.22543
particles:             500
steps:                 800
dt:                    0.5
diffusivity:           0.003
```

## Low-Peclet Benchmark Result

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
hybrid                       243.53     29.51       2.50     1        2        0
gaussian_bayes               246.54     22.34       2.50     1        0        2
bootstrap_mean_mixture       252.25     26.16       2.75     1        2        2
pooled_validation_mixture    254.57     33.02       3.50     1        1        1
knn_conditional              257.59     16.76       3.75     0        1        1
pair_rerank                  316.62     54.25       6.00     0        0        0
```

Mean selected weights:

```text
gaussian_bayes:    0.2708
knn_conditional:   0.0000
hybrid:            0.7083
pair_rerank:       0.0208
```

## Three-Regime Pattern

Mean selected weights shift systematically:

```text
condition          Gaussian/Bayes   kNN      hybrid   pair
D = 0.0003             0.4375      0.1875   0.3333   0.0417
D = 0.0010             0.4125      0.1500   0.4125   0.0250
D = 0.0030             0.2708      0.0000   0.7083   0.0208
```

Mean sampler ranks also shift:

```text
condition          best mean-rank samplers
D = 0.0003         pooled mixture, kNN
D = 0.0010         pooled mixture, Gaussian/Bayes
D = 0.0030         hybrid, Gaussian/Bayes
```

This is the cleanest mechanistic result so far:

```text
as diffusion increases, validation shifts weight away from local velocity
matching and toward the learned hybrid transition rule.
```

## Figures

![Peclet sampler ranks](../figures/run_009_peclet_sampler_ranks.svg)

![Peclet selected weights](../figures/run_009_peclet_selected_weights.svg)

## Interpretation

The Peclet sweep strengthens the manuscript substantially. We now have evidence that the selected sampler is not fixed; it responds to physical regime. High-Peclet transport makes local velocity matching more competitive, while low-Peclet transport favors the learned hybrid sampler. The pooled mixture remains valuable in the high-Peclet and baseline cases, but the low-Peclet case shows that a pure hybrid can be the best mean performer.

This still does not replace a true second-geometry or high-fidelity-flow validation. But it reduces the single-condition risk and gives the paper a mechanistic axis:

```text
Peclet regime controls which transport memory mechanism validation selects.
```

