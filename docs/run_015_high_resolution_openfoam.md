# Run 015: Higher-Resolution OpenFOAM Flow

## Purpose

This run tests whether the OpenFOAM conclusion from Run 010 survives a less-downsampled pore geometry. The earlier OpenFOAM case used a `75^3` voxel mesh from downsample factor 3. Here we trim the original `225^3` Core2 volume to `224^3`, downsample by factor 2, and solve on a `112^3` voxel mesh.

The goal is not yet a smoothed `snappyHexMesh` DNS-quality case. It is a direct higher-resolution version of the same voxel OpenFOAM workflow.

## Mesh And Solve

```text
case:       openfoam_cases/bentheimer_core2_subvol1_6um_downsample2_voxel_flow
shape:      112 x 112 x 112
voxel size: 12 micrometers
cells:      322,524
faces:      1,155,659
```

For comparison, the previous downsample-factor-3 OpenFOAM case had 98,270 cells and 373,175 faces.

`checkMesh` passed with `Mesh OK`. The solve converged cleanly:

```text
simpleFoam: SIMPLE solution converged in 105 iterations
OpenFOAM clock time: about 30 seconds
```

Flow summary:

```text
mean speed:              1.934e-08 m/s
median speed:            1.077e-08 m/s
95th percentile speed:   6.708e-08 m/s
maximum speed:           4.997e-07 m/s
outlet flux:             4.960e-15 m^3/s
net boundary flux:       5.760e-19 m^3/s
apparent permeability:   3.690e-12 m^2
```

The previous downsample-factor-3 apparent permeability was `4.277e-12 m^2`, so the finer voxel geometry lowers the inferred permeability by about 14%.

## Physical Scaling Choice

Changing voxel size changes the meaning of one grid cell. To make this a physical-resolution comparison rather than a purely grid-coordinate comparison, trajectories were generated with:

```text
target mean speed: 0.09 cells / step-unit
diffusivity:       0.00225 grid^2 / step-unit
planes:            9, 15, 21 cells
bin size:          4.5 cells
reaction radius:   4.5 cells
```

These choices match the approximate physical distances and physical diffusion represented by the previous downsample-factor-3 settings:

```text
target mean speed: 0.06 cells / step-unit
diffusivity:       0.001 grid^2 / step-unit
planes:            6, 10, 14 cells
bin size:          3 cells
reaction radius:   3 cells
```

## Outputs

```text
data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_trajectories.npz
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_objective_weight_sensitivity.json
```

## Reference Trajectory Physics

The finer OpenFOAM trajectories have stronger axial velocity autocorrelation than the coarser OpenFOAM trajectories:

```text
lag    d3 OpenFOAM    d2 OpenFOAM
1        0.3577         0.4290
2        0.3489         0.4184
5        0.3284         0.3979
10       0.3029         0.3683
20       0.2662         0.3215
40       0.2144         0.2564
80       0.1521         0.1909
```

This strengthens the physical premise that higher-resolution flow retains stronger velocity memory. The sampler result, however, becomes more subtle.

## Balanced Outer-Split Benchmark

Previous downsample-factor-3 OpenFOAM result:

```text
sampler                    mean_obj   mean_rank  wins
gaussian_bayes               261.39       1.25     3
hybrid                       273.63       2.50     1
bootstrap_mean_mixture       285.49       3.75     0
pooled_validation_mixture    299.46       3.50     0
knn_conditional              311.05       4.50     0
pair_rerank                  317.19       5.50     0
```

New downsample-factor-2 OpenFOAM result:

```text
sampler                    mean_obj   mean_rank  wins
gaussian_bayes               286.84       2.75     2
hybrid                       291.50       2.75     0
pooled_validation_mixture    296.86       2.75     1
bootstrap_mean_mixture       303.38       3.00     1
knn_conditional              314.03       3.75     0
pair_rerank                  358.02       6.00     0
```

The main change is not that Gaussian/Bayes disappears. It remains narrowly best by mean objective. The change is that its dominance weakens: Gaussian/Bayes, hybrid, and the pooled validation mixture have the same mean rank, and both mixture summaries beat Gaussian/Bayes on two of four held-out splits.

Mean selected weights in the finer balanced run:

```text
gaussian_bayes:    0.396
knn_conditional:   0.146
hybrid:            0.417
pair_rerank:       0.042
```

## Objective-Weight Sensitivity

Best mean mechanism by regime:

```text
regime           d3 OpenFOAM best              d2 OpenFOAM best
balanced         gaussian_bayes                pooled_validation_mixture
btc_heavy        gaussian_bayes                hybrid
pair_heavy       hybrid                        bootstrap_mean_mixture
dilution_heavy   gaussian_bayes                bootstrap_mean_mixture
reaction_light   gaussian_bayes                bootstrap_mean_mixture
reaction_heavy   gaussian_bayes                gaussian_bayes
no_reaction      gaussian_bayes                bootstrap_mean_mixture
```

In the coarser OpenFOAM run, Gaussian/Bayes was the best mean mechanism in six of seven regimes. In the finer physically scaled run, Gaussian/Bayes is best in only the reaction-heavy regime. Validation-selected mixtures and learned context become much more competitive.

## Interpretation

This is a useful complication. Higher-resolution OpenFOAM does strengthen measured velocity autocorrelation, but it does not simply make the velocity-continuity kernel dominate more strongly. Instead, the finer pore field appears to expose additional structure that a single velocity-memory rule does not fully protect.

The manuscript claim should therefore be sharpened:

```text
Higher-fidelity flow strengthens velocity memory, but higher resolution can also expose
additional plume-organization and context memories that make validation-selected mixtures
more valuable.
```

That is even better for the memory-adequacy thesis. It says the OpenFOAM result is not merely "better physics makes the old kernel win." The more general result is that changing the reference physics changes which memories are adequate. At downsample factor 3, velocity memory is enough for many objectives. At downsample factor 2, velocity memory remains important but no longer sufficient across objective regimes.

