# First Real-Rock Run

## Dataset

Bentheimer sandstone from the Zenodo multi-resolution dataset:

```text
Core1_Subvol1_18micron_75cube_16bit_LE.raw
```

Source:

https://zenodo.org/records/5542624

## What We Ran

```bash
python3 scripts/download_bentheimer_zenodo.py

python3 scripts/build_bentheimer_trajectories.py \
  --raw data/raw/Core1_Subvol1_18micron_75cube_16bit_LE.raw \
  --shape 75,75,75 \
  --voxel-size 18e-6 \
  --particles 300 \
  --steps 500 \
  --pressure-iters 800

python3 scripts/compare_samplers.py \
  --input data/processed/bentheimer_trajectories.npz \
  --planes 6,10,14 \
  --segment-steps 24 \
  --match-steps 4 \
  --n-generated 90 \
  --n-segments 30 \
  --diffusivity 0.001 \
  --output outputs/bentheimer_sampler_comparison.json
```

## Geometry And Trajectory Summary

```text
raw porosity:        0.26965
connected porosity:  0.26644
trajectories:        300
mean path length:    501 samples
```

The pressure solve was deliberately lightweight:

```text
solver: graph-Laplace Jacobi relaxation on connected pore voxels
iterations: 800
last max pressure update: 2.25e-4
```

## TTA Baseline Result

Initial breakthrough quantile mean absolute error over the tested control planes:

```text
unconditional      195.02
knn_conditional     98.68
gaussian_bayes     197.48
```

The kNN conditional sampler is already substantially better than unconditional segment shuffling on this first real-geometry run. The Gaussian/Bayes kernel is not yet tuned for this approximate voxel-flow setting; the next step is to expose and scan its bandwidth.

After adding the parameter scan, a better 18 micrometer setting was:

```text
best scanned setting: knn_conditional
segment_steps:        20
match_steps:          5
score:                116.63
quantile MAE:          59.22
coverage deficit:       0.230
```

The 6 micrometer source run in `second_real_rock_run.md` gives cleaner results.

## Caveats

- The CT volume is coarse: 18 micrometer, 75^3 voxels.
- Segmentation uses automatic Otsu thresholding.
- Flow is an approximate graph-Laplace pore-network relaxation, not Stokes, OpenFOAM, or LBM.
- Particle tracking uses nearest-voxel velocities and simple rejection at solid cells.
- This run is meant to validate the data-generation loop, not make physics claims.

## Why It Matters

We now have the full loop running without the original DNS data:

```text
real CT -> pore geometry -> approximate flow -> particle trajectories -> TTA archive -> generated trajectories -> benchmark table
```

This is enough to start method development while we improve the physics backend.
