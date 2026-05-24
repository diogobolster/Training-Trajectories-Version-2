# Run 018: Strict Full-Resolution OpenFOAM Particle Tracking

## Purpose

Run 018 rebuilds the full-resolution Core2 OpenFOAM trajectory archive using the strictly converged Run 017 velocity field rather than the earlier loose Run 016 field. It also tightens the particle integration relative to the first full-resolution archive and increases the ensemble size by a factor of 10.

The first full-resolution particle archive used:

```text
OpenFOAM time:       215
particles:           500
saved steps:         800
dt:                  0.5
diffusivity:         0.009 grid^2 / step-unit
```

The new archive uses:

```text
OpenFOAM time:       604
particles:           5000
saved steps:         4000
dt:                  0.1
diffusivity:         0.009 grid^2 / step-unit
max advective step:  0.10 voxel per internal substep
max diffusive step:  0.075 voxel RMS per internal substep
```

The total simulated duration is unchanged: `800 * 0.5 = 4000 * 0.1 = 400` step-units. The new run therefore tightens the time discretization without shortening the transport experiment.

## Code Changes

The particle tracker now supports optional internal substepping controls:

```text
--max-advective-step
--max-diffusive-step
--max-substeps
```

These controls limit the proposed advective displacement and the three-dimensional RMS diffusive displacement within an internal substep. The saved trajectory interval can therefore be chosen independently of the smaller internal stability/tolerance scale.

The tracker also reports diagnostics:

```text
output steps attempted/completed
total internal substeps
maximum substeps in any saved output step
substep cap hits
diffusive wall rejections
advective fallback accepts
immobile rejections
```

## Command

```bash
python3 scripts/build_openfoam_trajectories.py \
  --raw data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw \
  --case-dir openfoam_cases/bentheimer_core2_subvol1_6um_fullres_voxel_flow_strict \
  --time 604 \
  --downsample-factor 1 \
  --target-mean-speed 0.18 \
  --particles 5000 \
  --steps 4000 \
  --dt 0.1 \
  --diffusivity 0.009 \
  --max-advective-step 0.10 \
  --max-diffusive-step 0.075 \
  --max-substeps 128 \
  --seed 20260524 \
  --output data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz
```

## Outputs

```text
data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz
data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_autocorrelation_scan.json
outputs/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_autocorrelation_scan.csv
```

The trajectory archive is about `417M`.

## Tracking Diagnostics

```text
particles requested:              5000
trajectories written:             5000
mean trajectory length:           4000.689
output steps attempted:           19,998,445
output steps completed:           19,998,445
internal substeps:                20,201,143
mean substeps / output step:      1.0101
maximum substeps / output step:   9
substep cap hits:                 0
diffusive wall rejections:        202,548
advective fallback accepts:       183,415
immobile rejections:              19,133
```

The critical audit result is that the substep cap was never hit. The requested advective and diffusive step tolerances were therefore enforced throughout the run.

## Basic Archive QA

Compared with the earlier 500-particle full-resolution archive:

```text
archive                                  n       mean length   mean axial displacement   sample p95 step norm
loose-flow dt0.5 n500                    500     800.918       34.88 voxels              0.359 voxels
strict-flow dt0.1 n5000                 5000    4000.689       36.54 voxels              0.125 voxels
```

Only a small fraction of particles reach the outlet within the fixed duration in either archive. The larger ensemble makes the breakthrough tail and pair statistics less sampling-starved, while the finer saved time step substantially reduces typical per-step displacement.

## Axial Memory Scan

The saved time step is five times smaller than in the earlier archive, so the axial velocity-memory scan used lag values in multiples of 10 saved steps:

```text
lag steps   K(lambda)|1,2
10          0.5872
20          0.7254
30          0.7788
40          0.8030
60          0.8144
80          0.8088
100         0.7994
120         0.7866
160         0.7648
200         0.7482
240         0.7332
300         0.7152
360         0.7029
420         0.6890
480         0.6804
```

The scan suggests:

```text
match_steps:   60
segment_steps: 160
```

These are saved-step counts for the `dt = 0.1` archive. In physical time they are comparable to the coarser archive's segment scales.

## Interpretation

Run 018 removes two important vulnerabilities in the full-resolution OpenFOAM evidence. First, the trajectories now come from the strictly converged Run 017 velocity field. Second, the particle integrator no longer relies on a single large proposal per saved output step. The new archive is both larger and better resolved in time.

This does not by itself update the memory-selection benchmark; that requires rerunning the outer-split and objective-sensitivity analyses with segment and match steps appropriate for `dt = 0.1`. The expected next benchmark settings are approximately:

```text
match_steps:     60
segment_steps:   160
planes:          18,30,42
time_indices:    500,1000,1500,2000
bin_size:        9
reaction_radius: 9
diffusivity:     0.009
```

The scientific role of this run is to make the high-fidelity OpenFOAM trajectory archive more defensible before rerunning memory adequacy selection.
