# Run 019: Tight Particle Tracking Across the OpenFOAM Resolution Ladder

## Purpose

Run 019 applies the Run 018 tight particle-tracking protocol to the two lower-resolution OpenFOAM cases in the resolution ladder. Together, Runs 018 and 019 produce a consistent three-resolution trajectory archive:

```text
18 um OpenFOAM:  downsample factor 3
12 um OpenFOAM:  downsample factor 2
6 um OpenFOAM:   full resolution, strict Run 017 flow
```

Each archive uses `5000` particles, `4000` saved steps, and `dt = 0.1`, preserving the same total duration as the earlier `500`-particle, `dt = 0.5`, `800`-step archives.

## Physical Scaling

The transport parameters preserve the original physical scaling across voxel sizes:

```text
case    target mean speed   diffusivity
18 um   0.06 cells/time     0.001 grid^2/time
12 um   0.09 cells/time     0.00225 grid^2/time
6 um    0.18 cells/time     0.009 grid^2/time
```

The internal substep tolerances were scaled by physical distance rather than by voxel count. The full-resolution tolerance was `0.10` voxel advective displacement and `0.075` voxel RMS diffusive displacement. Equivalent physical tolerances give:

```text
case    max advective step   max diffusive RMS step
18 um   0.033333 voxel       0.0250 voxel
12 um   0.050000 voxel       0.0375 voxel
6 um    0.100000 voxel       0.0750 voxel
```

## Commands

### 18 um

```bash
python3 scripts/build_openfoam_trajectories.py \
  --raw data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw \
  --case-dir openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow \
  --time 103 \
  --downsample-factor 3 \
  --target-mean-speed 0.06 \
  --particles 5000 \
  --steps 4000 \
  --dt 0.1 \
  --diffusivity 0.001 \
  --max-advective-step 0.03333333333333333 \
  --max-diffusive-step 0.025 \
  --max-substeps 128 \
  --seed 20260524 \
  --output data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz
```

### 12 um

```bash
python3 scripts/build_openfoam_trajectories.py \
  --raw data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw \
  --case-dir openfoam_cases/bentheimer_core2_subvol1_6um_downsample2_voxel_flow \
  --time 105 \
  --trim-to-factor \
  --downsample-factor 2 \
  --target-mean-speed 0.09 \
  --particles 5000 \
  --steps 4000 \
  --dt 0.1 \
  --diffusivity 0.00225 \
  --max-advective-step 0.05 \
  --max-diffusive-step 0.0375 \
  --max-substeps 128 \
  --seed 20260524 \
  --output data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz
```

## Outputs

```text
data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz
data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.summary.json
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_autocorrelation_scan.json
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_autocorrelation_scan.csv

data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz
data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.summary.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_autocorrelation_scan.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_autocorrelation_scan.csv
```

Each trajectory archive is about `416M`.

## Tracking Diagnostics

```text
case    trajectories   mean length   max substeps   cap hits   mean substeps/output step
18 um   5000           4001.000      4              0          1.0086
12 um   5000           4000.842      5              0          1.0108
6 um    5000           4000.689      9              0          1.0101
```

The key audit result is that no case hit the internal substep cap. The requested particle-integration tolerances were therefore enforced across the full resolution ladder.

## Basic Archive QA

```text
case    mean axial displacement   median axial displacement   outlet fraction   sample p95 step norm
18 um   11.577 voxels             7.344 voxels                0.0000            0.0420 voxels
12 um   18.366 voxels             11.297 voxels               0.0008            0.0638 voxels
6 um    36.542 voxels             19.435 voxels               0.0016            0.1252 voxels
```

The axial displacements scale with grid resolution because the transport parameters preserve physical distance rather than voxel count.

## Axial Memory Scans

The scans use the same saved-step lags for all three `dt = 0.1` archives.

```text
lag   18 um    12 um    6 um
10    0.5458   0.5827   0.5872
20    0.6802   0.7140   0.7254
30    0.7349   0.7665   0.7788
40    0.7611   0.7897   0.8030
60    0.7809   0.8027   0.8144
80    0.7841   0.7991   0.8088
100   0.7802   0.7900   0.7994
120   0.7734   0.7786   0.7866
160   0.7563   0.7553   0.7648
200   0.7398   0.7355   0.7482
240   0.7228   0.7159   0.7332
300   0.7062   0.6974   0.7152
360   0.6911   0.6826   0.7029
420   0.6831   0.6757   0.6890
480   0.6716   0.6702   0.6804
```

Suggested discrete scales:

```text
case    match_steps   segment_steps
18 um   80            160
12 um   60            160
6 um    60            160
```

## Interpretation

The tight-tracking resolution ladder strengthens the OpenFOAM evidence in two ways. First, the trajectory archives are no longer sampling-limited at `500` particles. Second, the particle integration has a documented tolerance audit, with no substep-cap hits in any case.

The axial velocity-memory scan remains physically coherent. Velocity memory strengthens from `18 um` to `12 um` and remains high at `6 um`, matching the earlier interpretation from the lower-count archives. The next step is to rerun the memory-selection benchmarks on these tight archives using `segment_steps = 160` and `match_steps = 60-80`, with planes and reaction radii scaled consistently with each resolution.
