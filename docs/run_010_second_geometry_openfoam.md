# Run 010: Second Geometry and OpenFOAM Flow

## Summary

Run 010 addresses the two largest validation risks at once:

- a second geometry, using `Core2_Subvol1_6micron_225cube_16bit_LE.raw` from the Zenodo multi-resolution Bentheimer dataset,
- a higher-fidelity finite-volume pore-flow field, using an OpenFOAM voxel mesh on the connected pore space.

Core2 is still Bentheimer, so this is not yet cross-lithology generalization. It is nevertheless a meaningful stress test because it is a different core/subvolume and the OpenFOAM solve replaces the in-house graph-Laplace pressure approximation.

Source dataset:

```text
https://zenodo.org/records/5542624
```

## New Files

```text
data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw
data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_trajectories.npz
data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_trajectories.npz
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_outer_split_mixture_benchmark.json
openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow/
figures/run_010_generalization_sampler_ranks.svg
figures/run_010_generalization_selected_weights.svg
```

## Commands

```bash
python3 scripts/download_bentheimer_zenodo.py --file core2_subvol1_6um

python3 scripts/build_bentheimer_trajectories.py \
  --raw data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw \
  --shape 225,225,225 \
  --voxel-size 6e-6 \
  --downsample-factor 3 \
  --particles 500 \
  --steps 800 \
  --diffusivity 0.001 \
  --output data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_trajectories.npz

python3 scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_trajectories.npz \
  --n-outer-splits 4 --n-repeats 3 --grid-step 0.25 \
  --n-validation-generated 45 --n-test-generated 80 --pair-samples 1200 \
  --contrastive-epochs 300 --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 --pair-rerank-weight 0.25 \
  --btc-weight 1 --pair-weight 20 --dilution-weight 120 --reaction-weight 1000 \
  --output outputs/bentheimer_core2_subvol1_6um_downsample3_D001_outer_split_mixture_benchmark.json

python3 scripts/export_openfoam_voxel_case.py \
  --raw data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw \
  --shape 225,225,225 \
  --voxel-size 6e-6 \
  --downsample-factor 3 \
  --output-dir openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow \
  --end-time 300 --write-interval 100

docker run --rm --entrypoint /bin/bash \
  -v '/Users/dbolster/Documents/Codex/Training Trajectories Neural Net:/work' \
  -w /work/openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow \
  openeuler/openfoam:2506-oe2403sp2 \
  -lc 'source /opt/OpenFOAM-v2506/etc/bashrc && checkMesh'

docker run --rm --entrypoint /bin/bash \
  -v '/Users/dbolster/Documents/Codex/Training Trajectories Neural Net:/work' \
  -w /work/openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow \
  openeuler/openfoam:2506-oe2403sp2 \
  -lc 'source /opt/OpenFOAM-v2506/etc/bashrc && simpleFoam'

python3 scripts/summarize_openfoam_flow.py \
  --case-dir openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow

python3 scripts/build_openfoam_trajectories.py \
  --raw data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw \
  --case-dir openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow \
  --shape 225,225,225 \
  --voxel-size 6e-6 \
  --downsample-factor 3 \
  --particles 500 \
  --steps 800 \
  --diffusivity 0.001 \
  --output data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_trajectories.npz
```

## Geometry And Trajectory Summary

```text
raw volume:            Core2_Subvol1_6micron_225cube_16bit_LE.raw
input shape:           225 x 225 x 225
simulation shape:       75 x 75 x 75
effective voxel size:   18 micrometers
raw porosity:            0.23553
connected porosity:      0.23294
particles:               500
steps:                   800
dt:                      0.5
diffusivity:             0.001
```

## Second-Geometry Benchmark With Graph Flow

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
gaussian_bayes               255.40     39.43       1.75     2        0        3
hybrid                       268.50     38.71       3.25     1        1        0
pooled_validation_mixture    269.00     40.17       3.25     0        1        2
bootstrap_mean_mixture       272.22     33.19       3.50     1        1        2
knn_conditional              273.90     45.61       3.50     0        0        2
pair_rerank                  320.67     43.86       5.75     0        0        0
```

Mean selected weights:

```text
gaussian_bayes:    0.4792
knn_conditional:   0.1875
hybrid:            0.2708
pair_rerank:       0.0625
```

Interpretation: on Core2 with the same approximate graph-flow pipeline, Gaussian/Bayes becomes the best mean sampler. The validation mixture remains close to hybrid and beats it on two of four splits, but it no longer wins outright. This supports the paper's key claim that the physics kernel remains valuable and that sampler choice is condition-dependent.

## OpenFOAM Flow

The new exporter writes each connected pore voxel as one OpenFOAM hexahedral finite-volume cell. Pore-solid faces are no-slip walls; inlet and outlet faces are fixed kinematic pressure boundaries.

OpenFOAM case summary:

```text
cells:                 98,270
points:               184,515
faces:                373,175
internal faces:       216,445
inlet faces:            1,415
outlet faces:           1,226
wall faces:           154,089
```

`checkMesh` result:

```text
Mesh OK.
```

`simpleFoam` result:

```text
SIMPLE solution converged in 103 iterations
```

Flow summary at time 103:

```text
mean speed:                    2.284e-08 m/s
median speed:                  1.346e-08 m/s
95th percentile speed:         7.716e-08 m/s
maximum speed:                 4.767e-07 m/s
outlet flux:                   5.774e-15 m^3/s
net boundary flux:             7.720e-19 m^3/s
bulk Darcy velocity:           3.168e-09 m/s
apparent permeability:         4.277e-12 m^2
```

The OpenFOAM field was converted back to a voxel velocity array and normalized to the same mean advective speed used in the graph-flow trajectory runs:

```text
physical mean speed:      0.001269 voxels/s
target simulation speed:  0.060000 voxels/step-unit
trajectories generated:   500
mean path length:         801 samples
```

## OpenFOAM-Trajectory Benchmark

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
gaussian_bayes               261.39     28.17       1.25     3        0        3
hybrid                       273.63     28.38       2.50     1        1        0
bootstrap_mean_mixture       285.49     17.76       3.75     0        0        1
pooled_validation_mixture    299.46     33.33       3.50     0        0        1
knn_conditional              311.05     19.82       4.50     0        0        1
pair_rerank                  317.19     31.80       5.50     0        0        0
```

Mean selected weights:

```text
gaussian_bayes:    0.3333
knn_conditional:   0.1667
hybrid:            0.4167
pair_rerank:       0.0833
```

## Figures

![Generalization sampler ranks](../figures/run_010_generalization_sampler_ranks.svg)

![Generalization selected weights](../figures/run_010_generalization_selected_weights.svg)

## Interpretation

This is a useful, honest result. The validation-selected mixture is not the winner under the OpenFOAM trajectory set; Gaussian/Bayes is. That actually strengthens the scientific framing. The original physics-informed transition kernel remains hard to beat when the velocity field is more physically resolved, and the role of validation is to reveal this rather than force a learned sampler to win.

Across the current evidence:

```text
Peclet changes mixture weights.
Geometry changes sampler ranking.
Flow fidelity shifts weight back toward the original physics kernel.
```

That is a better paper than a simple ML victory lap. The story is now: TTA-v2 is a validation-driven scaffold for physics/ML transport mechanisms, and the original 2019 idea remains a very strong component inside that scaffold.

## Remaining Caveats

The OpenFOAM case is a stair-step voxel finite-volume mesh on a 75^3 downsampled pore geometry. It is a real finite-volume Stokes/simpleFoam solve, but it is not yet a polished DNS/LBM study on a high-resolution smoothed pore surface. The next physical-validation step would be one of:

- run the same OpenFOAM workflow at a less aggressive downsample factor,
- generate a smoothed surface mesh with `snappyHexMesh`,
- compare against an LBM velocity field from a public pore-scale benchmark,
- run objective-weight sensitivity on the OpenFOAM trajectories.

The final item has now been completed in Run 011. The main result is that Gaussian/Bayes remains the best mean sampler in six of seven objective regimes, while the pair-heavy regime selects the hybrid sampler. See `run_011_openfoam_objective_sensitivity.md`.
