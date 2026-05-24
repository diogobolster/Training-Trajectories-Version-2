# Simulation Pipeline

## Goal

Generate particle trajectories from a real 3D porous medium so the TTA-v2 samplers can be trained and benchmarked without relying on the original 2019 DNS data.

## First-Pass Pipeline

```text
raw CT volume
  -> threshold pore space
  -> keep inlet-outlet connected pore network
  -> solve approximate single-phase pressure field
  -> compute voxel velocity field
  -> advective-diffusive particle tracking
  -> trajectories.npz
  -> compare_samplers.py
```

## Why Start Approximate

The first goal is not a publishable pore-scale CFD solver. The first goal is to create a reproducible, inspectable trajectory dataset from a real complex geometry so we can develop:

- TTA segment archives,
- transition kernels,
- autocorrelation-based memory scales,
- dilution and pair statistics,
- learned transition models.

Once that loop is healthy, we can replace the approximate solver with OpenFOAM, GeoChemFoam, LBM, or precomputed velocity fields.

## Commands

Download the tiny Bentheimer smoke-test volume:

```bash
python3 scripts/download_bentheimer_zenodo.py
```

Build trajectories:

```bash
python3 scripts/build_bentheimer_trajectories.py \
  --raw data/raw/Core1_Subvol1_18micron_75cube_16bit_LE.raw \
  --shape 75,75,75 \
  --voxel-size 18e-6 \
  --particles 500 \
  --steps 800
```

Use the 6 micrometer volume as source but downsample to a 75^3 bootstrap simulation:

```bash
python3 scripts/download_bentheimer_zenodo.py --file core1_subvol1_6um

python3 scripts/build_bentheimer_trajectories.py \
  --raw data/raw/Core1_Subvol1_6micron_225cube_16bit_LE.raw \
  --shape 225,225,225 \
  --voxel-size 6e-6 \
  --downsample-factor 3 \
  --particles 500 \
  --steps 800 \
  --output data/processed/bentheimer_6um_downsample3_trajectories.npz
```

Run TTA baselines:

```bash
python3 scripts/compare_samplers.py \
  --input data/processed/bentheimer_trajectories.npz \
  --planes 20,40,60 \
  --segment-steps 30 \
  --match-steps 5
```

Coordinates are initially in voxel units for convenience. The physical voxel size is recorded in the summary JSON.

Scan discrete memory lengths and transition bandwidths:

```bash
python3 scripts/tune_tta_parameters.py \
  --input data/processed/bentheimer_trajectories.npz \
  --planes 6,10,14 \
  --segment-steps 16,20,24,30,36 \
  --match-steps 3,4,5,6 \
  --gaussian-bandwidths 0.5,1,2,4,8,16
```

The scan ranks settings by a coverage-aware score:

```text
score = breakthrough_quantile_mae + missing_penalty * coverage_deficit
```

This prevents a sampler from looking artificially good when it simply fails to reach the downstream control planes.

Estimate discrete memory scales from adjacent-segment velocity autocorrelation:

```bash
python3 scripts/scan_velocity_autocorrelation.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --steps 2,3,4,5,6,8,10,12,16,20,24,30,36,42,48
```

## Upgrade Path

1. Replace Otsu threshold with curated segmentation.
2. Replace graph-Laplace pressure solve with OpenFOAM/LBM.
3. Replace nearest-cell particle tracking with trilinear interpolation and boundary-aware Brownian reflection.
4. Add validation against measured permeability or concentration images.
5. Add learned transition scorer once trajectory generation is stable.
